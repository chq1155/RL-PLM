from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import torch

def generate_mask(sequences: Sequence[str]) -> torch.Tensor:
    """
    Create a boolean mask where tokens corresponding to actual residues are 1.
    """
    lengths = [len(seq) for seq in sequences]
    batch_size = len(sequences)
    max_length = max(lengths) if lengths else 0
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for idx, length in enumerate(lengths):
        if length:
            mask[idx, :length] = 1
    return mask


def average_token_embeddings(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Average token representations across valid residues for each sequence.
    """
    if token_embeddings.size(0) != mask.size(0):
        raise ValueError("Batch size mismatch between embeddings and mask.")
    if token_embeddings.size(1) != mask.size(1):
        raise ValueError("Sequence length mismatch between embeddings and mask.")

    mask = mask.unsqueeze(-1)
    masked_embedding = token_embeddings * mask
    seq_lengths = mask.sum(dim=1).clamp_min(1)
    return masked_embedding.sum(dim=1) / seq_lengths


@torch.no_grad()
def encode(
    sequences: Iterable[str],
    esm_model,
    batch_converter,
    alphabet,
    *,
    device: str | torch.device = "cuda",
    representation_layer: int = 6,
) -> torch.Tensor:
    """
    Encode sequences with an ESM model and return per-sequence embeddings.
    """
    device = torch.device(device)
    esm_model = esm_model.to(device).eval()

    sequence_list = list(sequences)
    data = [(f"protein_{idx}", seq) for idx, seq in enumerate(sequence_list)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device, non_blocking=True)

    masks = generate_mask(sequence_list).to(device)

    output = esm_model(batch_tokens, repr_layers=[representation_layer], return_contacts=False)
    token_reps = output["representations"][representation_layer]
    token_reps = token_reps[:, 1:-1, :]

    return average_token_embeddings(token_reps, masks)


def reward_amp_cls(
    sequences: Iterable[str],
    esm_model,
    batch_converter,
    alphabet,
    classifier,
    *,
    device: str | torch.device = "cuda",
    threshold: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Score sequences with a classifier on top of ESM embeddings.

    Returns a tuple of (reward, pass_mask).
    """
    embeddings = encode(sequences, esm_model, batch_converter, alphabet, device=device)
    classifier = classifier.to(device).eval()

    scores = classifier(embeddings.to(device))
    scores = scores.reshape(scores.shape[0], -1)
    if scores.size(1) != 1:
        raise ValueError("Classifier must output a single score per sequence.")
    scores = scores.squeeze(-1)

    rewards = (scores - threshold) * 2.0
    mask = scores >= threshold
    return rewards, mask
