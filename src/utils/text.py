from typing import Callable

import torch
from transformers import PreTrainedTokenizer


def create_transform_fn_from_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizer, max_length: int, padding: bool = True
) -> Callable[[list[str | list[str]]], torch.Tensor]:
    def transform(texts: list[str | list[str]]) -> torch.Tensor:
        return tokenizer(texts, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)[
            "input_ids"
        ]

    return transform
