from __future__ import annotations

from typing import List

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    ret_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)

    for step in reversed(range(rewards.size(1))):
        delta = rewards[:, step] + gamma * next_values[:, step] - values[:, step]
        gae = delta + gamma * gae_lambda * gae
        advantages[:, step] = gae

    returns = advantages + values
    if ret_clip is not None:
        returns = torch.clamp(returns, min=-ret_clip, max=ret_clip)
    return advantages, returns


def build_kl_mask(
    positions: List[torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for batch_idx, pos_list in enumerate(positions):
        if len(pos_list) > 0:
            mask[batch_idx, pos_list.to(device).long()] = True
    return mask
