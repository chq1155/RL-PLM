from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


class AMPQueryDataset(Dataset):
    """
    Minimal dataset that yields prompt token IDs for autoregressive decoding.
    """

    def __init__(self, input_ids: Sequence[Sequence[int]]):
        self._items: List[torch.Tensor] = [
            torch.tensor(ids, dtype=torch.long) for ids in input_ids
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return {"input_ids": self._items[idx]}


def _load_prompts_from_file(prompt_file: Path) -> List[str]:
    with prompt_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _repeat_prompts(prompts: Sequence[str], target_size: int) -> List[str]:
    if not prompts:
        raise ValueError("At least one prompt is required.")
    repeated: List[str] = []
    while len(repeated) < target_size:
        repeated.extend(prompts)
    return repeated[:target_size]


def loading_dataset(
    steps: int,
    batch_size: int,
    *,
    tokenizer,
    prompt: str | None = "<|bos|>",
    prompt_file: str | Path | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
    return_dataset: bool = False,
) -> DataLoader | AMPQueryDataset:
    """
    Build a dataloader that repeatedly feeds prompt tokens to the model.
    """
    total_items = steps * batch_size

    prompts: List[str]
    if prompt_file:
        prompts = _load_prompts_from_file(Path(prompt_file))
    elif prompt is not None:
        prompts = [prompt]
    else:
        raise ValueError("Either `prompt` or `prompt_file` must be provided.")

    prompts = _repeat_prompts(prompts, total_items)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_token = getattr(tokenizer, "bos_token", None)
        if bos_token:
            bos_id = tokenizer.convert_tokens_to_ids(bos_token)
        else:
            bos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is None:
        raise ValueError("Tokenizer must provide a BOS or EOS token id.")

    encoded = []
    for text in prompts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        encoded.append(tokens or [bos_id])

    dataset = AMPQueryDataset(encoded)
    if return_dataset:
        return dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
