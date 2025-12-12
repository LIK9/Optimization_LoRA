import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import os
from tqdm import tqdm
import json
import copy
from trl import SFTConfig, SFTTrainer, setup_chat_format
from evaluate import load
import numpy as np
import config
from transformers import TrainingArguments as HFTrainingArguments
from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader

from typing import Any, Callable, Optional, TypeVar, Union
from trainer_persona import PersonaTrainer
import prompt


def make_dataset(file_name):
    path = f'./original_dataset/{file_name}.txt'

    with open(path, 'r', encoding='utf-8') as file:
        data = []
        cnt = 0
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx]) # 1

            if int(dialog_idx) == 1:
                # if cnt == 1:
                #     break
                data.append({'persona_info': [], 'dialog': []})
                cnt += 1

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line] # ['your persona: i like to remodel homes.']

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '') # 'i like to remodel homes.'
                if persona_info[-1] == '.' and persona_info[-2] != ' ':
                    persona_info = persona_info[:-1] + ' .' # 'i like to remodel homes .'
                data[-1]['persona_info'].append(persona_info) # {'persona_info': ['i like to remodel homes .'], 'dialog': [], 'candidates': []}
            if dialog_line[0].startswith('partner\'s person'):
                if not data[-1].__contains__('partner_persona_info'):
                    data[-1]['partner_persona_info'] = []
                persona_info = dialog_line[0].replace('partner\'s persona: ', '')
                if persona_info[-1] == '.' and persona_info[-2] != ' ':
                    persona_info = persona_info[:-1] + ' .'
                data[-1]['partner_persona_info'].append(persona_info)

            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])

    dataset = []
    for idx, chat in enumerate(tqdm(data)):
        context = { 'persona_info':[], 'dialogue':[]}

        persona_info = [s for s in chat['persona_info']]

        context['persona_info'] = persona_info

        for i, replica in enumerate(chat['dialog'], 1):
            context['dialogue'].append(replica)
            if not i % 2: # response까지 넣었으면
                
                dataset.append(copy.deepcopy(context))

    with open(f'./original_dataset/{file_name}.json', "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


def get_model(args):
    if args.use_lora:
        # bnb_config = BitsAndBytesConfig( # model parameter가 4bit로 quantization
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=args.torch_dtype, # bfloat16
        #     bnb_4bit_use_double_quant=True,
        # )

        target_modules_list = []
        if args.lora_q:
            target_modules_list.append('q_proj')
        if args.lora_k:
            target_modules_list.append('k_proj')
        if args.lora_v:
            target_modules_list.append('v_proj')

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha, # Lora scaling.
            lora_dropout=args.lora_dropout,
            bias="none", #  If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
            task_type="CAUSAL_LM",
            # target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
            target_modules=target_modules_list,
            )
        print(target_modules_list)


        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # quantization_config=bnb_config, # 수정 quantization를 안함
            device_map={"": torch.cuda.current_device()},
            # device_map="auto",
        )
        model.config.use_cache=False # K, V cache 사용X
        model.config.pretraining_tp=1 # Tensor Parallelism weight 분할을 몇 조각으로 나눌지
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map={"": torch.cuda.current_device()},
        )

    return model

def main(args):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    set_seed(args.seed)


    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # tokenizer.pad_token = tokenizer.eos_token

    # dataset = load_dataset("json", data_files={"train": args.train_data})
    # dataset = dataset.shuffle(seed=args.seed)
    # dataset = dataset.map(partial(prompt.format_prompt_train_single, tokenizer=tokenizer), load_from_cache_file=False)

    # prompt_test = dataset['train']['text'][0]
    # print(prompt_test) # prompt 테스트용
    # args.prompt_test = prompt_test


    model = get_model(args)




    # trainer = PersonaTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=args,
    #     train_dataset=dataset["train"],
    #     current_time = current_time,
    # )

    # args_dict = vars(args)
    # with open(f'./args/{current_time}.json', "w", encoding="utf-8") as f:
    #     json.dump(args_dict, f, ensure_ascii=False, indent=4)
    # trainer.train()




if __name__ == '__main__':
    args = config.get_args()
    # make_dataset(file_name = 'train_self_original')
    main(args)
