import torch
import numpy as np
from typing import Any, Callable, Optional, TypeVar, Union
from datasets import Dataset, DatasetDict
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types

DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

def truncate_dataset(
    dataset: DatasetType, max_length: int, map_kwargs: Optional[dict[str, Any]] = None
) -> DatasetType:
    if map_kwargs is None:
        map_kwargs = {}
    if isinstance(dataset, Dataset):
        # Fast truncation with pyarrow
        def truncate(examples):
            truncated_columns = []
            for column in examples.columns:
                if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                    column = pc.list_slice(column, 0, max_length)
                truncated_columns.append(column)
            return pa.Table.from_arrays(truncated_columns, names=examples.column_names)

        dataset = dataset.with_format("arrow")
        dataset = dataset.map(truncate, batched=True, **map_kwargs)
        dataset = dataset.with_format(None)
    else:

        def truncate(examples):
            truncated_examples = {}
            for key, column in examples.items():
                if column and isinstance(column[0], list):
                    column = [val[:max_length] for val in column]
                truncated_examples[key] = column
            return truncated_examples

        dataset = dataset.map(
            truncate,
            batched=True,
            **map_kwargs,
        )
    return dataset



def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:

    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

