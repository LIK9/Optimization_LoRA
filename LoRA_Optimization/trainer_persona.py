from typing import Any, Callable, Optional, TypeVar, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer, Trainer
from datasets import Dataset, IterableDataset
from argparse import Namespace
import dataclasses
from dataclasses import dataclass
from packaging import version
import peft
from utils.train_utils import peft_module_casting_to_bf16, pad, truncate_dataset
import warnings
from collections import defaultdict
from collections.abc import Mapping
import contextlib
from pathlib import Path
import os
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import json
from datetime import datetime


TListOrMapping = TypeVar("TListOrMapping", list, Mapping)
def remove_none_values(example: TListOrMapping) -> TListOrMapping:

    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")

class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    pad_token_id: int
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        if self.return_position_ids:
            if "position_ids" in examples[0]:
                position_ids = [torch.tensor(example["position_ids"]) for example in examples]
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
            # print("label있음")
            # print(labels)
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
            # print("label없음")
            # print(labels)

        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.return_position_ids:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if "assistant_masks" in examples[0]:
            assistant_masks = pad(
                assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][assistant_masks == 0] = -100
        return output



class PersonaTrainer():
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Namespace,
        train_dataset: Optional[Union[Dataset, IterableDataset, str]] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        current_time: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.writer = SummaryWriter(log_dir=f'./logs/{current_time}')
        self.output_dir=f"./outputs/{current_time}"

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100).to(model.device)

        first_example = next(iter(train_dataset)) # 확인용

        # data_collator
        pad_token = tokenizer.pad_token or tokenizer.eos_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id=pad_token_id,
            return_position_ids=model.config._attn_implementation == "flash_attention_2",
            pad_to_multiple_of=None,
        )


        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch,
            shuffle=False,
            collate_fn=data_collator,
        )

        steps_per_epoch = math.ceil(len(self.train_dataloader))  
        num_training_steps = (steps_per_epoch * args.train_epochs) // args.grad_acc_step

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,)
        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_training_steps*args.warmup_step,
            num_training_steps=num_training_steps,
        )




    def train(self):
        self.model.train()

        global_step = 0

        self.optimizer.zero_grad()

        self._metrics = { "epoch":[], "step":[], 'loss':[], 'gpu':[], 'end_time':'' }

        device_index = (
            self.model.device.index 
            if hasattr(self.model.device, "index") and self.model.device.index is not None
            else 0
        )


        for epoch in range(self.args.train_epochs):
            step_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))

            for step, batch in progress_bar:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}


                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss / self.args.grad_acc_step
                loss.backward()
                step_loss += loss.item()
                
                if (step + 1) % self.args.grad_acc_step == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self.optimizer.zero_grad()

                    global_step += 1

                    if global_step % self.args.log_step == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.writer.add_scalar("training/loss", step_loss, global_step=global_step)
                        self.writer.add_scalar("training/lr", lr, global_step=global_step)
                        progress_bar.set_description(f"epoch: {epoch} | loss: {step_loss:.4f}")

                        mem_reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)

                        self._metrics["epoch"].append(epoch+1)
                        self._metrics["step"].append(step+1)
                        self._metrics["loss"].append(step_loss)
                        self._metrics["gpu"].append(mem_reserved)

                    step_loss = 0.0

                    # break

            self.model.save_pretrained(f"{self.output_dir}/epoch{epoch}")
            self.tokenizer.save_pretrained(f"{self.output_dir}/epoch{epoch}")

        self._metrics["end_time"] = datetime.now().strftime('%b%d_%H-%M-%S')

        with open(f"{self.output_dir}/train_process", "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, ensure_ascii=False, indent=4)


    
    
