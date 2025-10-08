from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim import Adam
from tqdm import tqdm
import wandb

from eval import EvalDataset, load_cdr_info, load_eval_dataset
from lit_model import LitModel
from model import freeze_for_mutation_finetune_v2
from ppo import build_kl_mask


@dataclass
class TrainConfig:
    data_path: str = "./data/sigmul_data/AB1101_multiple_cdr_balance_train.csv"
    cdr_info_path: str = "./data/sigmul_data/cdr_info.csv"
    checkpoint_path: str = "./checkpoints_identity_sigmul/AB1101/esm2_t33_650M_UR50D_AB1101-val_pearson_corr_lr-3e-05_loss-mse_tok33.ckpt"
    output_dir: str = "grpo_ckpt_H3"
    epochs: int = 30
    lr: float = 4e-5
    weight_decay: float = 1e-4
    rollout_steps: int = 4
    max_mutations: int = 4
    reward_scale: float = 100.0
    temperature_start: float = 1.0
    temperature_min: float = 0.5
    position_temperature_start: float = 1.0
    position_temperature_min: float = 0.5
    anneal_steps: int = 1000
    group_tau: float = 0.5
    policy_loss_coef: float = 20.0
    position_loss_weight: float = 0.5
    entropy_weight: float = 0.1
    entropy_min: float = 0.01
    target_kl: float = 2.0
    kl_coef: float = 0.05
    kl_alpha: float = 0.95
    kl_max: float = 10.0
    max_grad_norm: float = 0.5
    freeze_layers: int = 2


class MutationPolicy(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int = 1280, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        wt_ab_inputs_ids,
        wt_ab_inputs_mask,
        wt_ag_inputs_ids,
        wt_ag_inputs_mask,
        mut_ab_inputs_ids=None,
        mt_ab_inputs_mask=None,
        mut_ag_inputs_ids=None,
        mt_ag_inputs_mask=None,
        *,
        with_position: bool = False,
    ):
        outputs = self.base_model(
            wt_ab_inputs_ids=wt_ab_inputs_ids,
            wt_ab_inputs_mask=wt_ab_inputs_mask,
            wt_ag_inputs_ids=wt_ag_inputs_ids,
            wt_ag_inputs_mask=wt_ag_inputs_mask,
            mut_ab_inputs_ids=mut_ab_inputs_ids,
            mt_ab_inputs_mask=mt_ab_inputs_mask,
            mut_ag_inputs_ids=mut_ag_inputs_ids,
            mt_ag_inputs_mask=mt_ag_inputs_mask,
            with_value=False,
            output_embeddings=True,
        )
        logits = outputs[3]
        embeddings = outputs[5]
        position_probs = self.position_head(embeddings).squeeze(-1) if with_position else None
        return logits, embeddings, position_probs


def sample_mutations(
    logits: torch.Tensor,
    position_probs: torch.Tensor,
    wt_seq: torch.Tensor,
    cdr_masks: torch.Tensor,
    *,
    max_mutations: int,
    temperature: float,
    position_temperature: float,
    pad_id: int = 1,
    stochastic: bool = True,
    position_threshold: float = 0.5,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    batch_size, seq_len, vocab_size = logits.shape
    valid_mask = (wt_seq != pad_id) & cdr_masks
    masked_position_probs = position_probs * valid_mask.float()
    mutated = wt_seq.clone()
    positions: List[torch.Tensor] = []
    position_actions = torch.zeros(batch_size, seq_len, device=logits.device)

    for i in range(batch_size):
        valid_positions = torch.nonzero(valid_mask[i]).squeeze(-1)
        if len(valid_positions) == 0:
            positions.append(torch.tensor([], device=logits.device, dtype=torch.long))
            continue
        pos_probs = masked_position_probs[i, valid_positions]
        if stochastic:
            if position_temperature != 1.0:
                pos_probs = F.softmax(torch.log(pos_probs + 1e-8) / position_temperature, dim=-1)
            mask = pos_probs > position_threshold
            if mask.any():
                candidates = valid_positions[mask]
                probs = pos_probs[mask]
                if len(candidates) > max_mutations:
                    idx = torch.multinomial(probs, max_mutations, replacement=False)
                    selected = candidates[idx]
                else:
                    selected = candidates
            else:
                top_k = min(max_mutations, len(valid_positions))
                _, idx = torch.topk(pos_probs, top_k)
                selected = valid_positions[idx]
        else:
            top_k = min(max_mutations, len(valid_positions))
            _, idx = torch.topk(pos_probs, top_k)
            selected = valid_positions[idx]

        positions.append(selected)
        position_actions[i, selected] = position_probs[i, selected]

        for pos in selected.tolist():
            aa_probs = F.softmax(logits[i, pos] / temperature, dim=-1)
            wt_idx = int(wt_seq[i, pos])
            aa_probs = aa_probs.clone()
            aa_probs[wt_idx] = 0.0
            if aa_probs.sum() == 0:
                aa_probs = torch.ones_like(aa_probs) / (vocab_size - 1)
                aa_probs[wt_idx] = 0.0
                aa_probs = aa_probs / aa_probs.sum()
            else:
                aa_probs = aa_probs / aa_probs.sum()
            new_idx = torch.multinomial(aa_probs, 1).item() if stochastic else torch.argmax(aa_probs).item()
            mutated[i, pos] = new_idx

    return mutated, positions, position_actions


def compute_action_logp(
    logits: torch.Tensor,
    position_probs: torch.Tensor,
    position_actions: torch.Tensor,
    mutated: torch.Tensor,
    positions: List[torch.Tensor],
    *,
    temperature: float,
    position_weight: float,
) -> torch.Tensor:
    aa_log = []
    pos_log = []
    batch_size = logits.size(0)
    for i in range(batch_size):
        pos_list = positions[i]
        if len(pos_list) == 0:
            aa_log.append(torch.tensor(0.0, device=logits.device))
            pos_log.append(torch.tensor(0.0, device=logits.device))
            continue
        aa_logits = logits[i, pos_list] / temperature
        aa_probs = F.softmax(aa_logits, dim=-1)
        selected = mutated[i, pos_list].long()
        aa_log.append(torch.log(aa_probs.gather(1, selected.unsqueeze(-1)).squeeze(-1) + 1e-8).mean())
        mask = position_actions[i] > 0
        pos_log_val = torch.log(position_probs[i, mask] + 1e-8).mean() if mask.any() else torch.tensor(0.0, device=logits.device)
        pos_log.append(pos_log_val)
    return torch.stack(aa_log) + position_weight * torch.stack(pos_log)


class MutationGRPOTrainer:
    def __init__(self, args: argparse.Namespace, cfg: TrainConfig):
        self.args = args
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = load_eval_dataset(cfg.data_path)
        cdr_df = load_cdr_info(cfg.cdr_info_path)
        dataset = EvalDataset(args, data, cdr_df)
        self.dataloader = dataset.get_dataloader()

        self.policy, self.reference, self.reward = self._load_models(args.hidden_size)
        self.optimizer = Adam(self.policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.global_step = 0
        self.running_kl = 0.0
        self.kl_coef = cfg.kl_coef

        torch.save(self.policy.state_dict(), self.output_dir / "base.pt")

    def _load_models(self, hidden_size: int) -> Tuple[MutationPolicy, nn.Module, nn.Module]:
        ckpt = self.cfg.checkpoint_path
        base = LitModel.load_from_checkpoint(ckpt).model
        policy = MutationPolicy(base, hidden_size=hidden_size).to(self.device)

        reference = LitModel.load_from_checkpoint(ckpt).model.to(self.device).eval()
        for p in reference.parameters():
            p.requires_grad = False

        reward = LitModel.load_from_checkpoint(ckpt).model.to(self.device).eval()
        for p in reward.parameters():
            p.requires_grad = False

        freeze_for_mutation_finetune_v2(
            policy.base_model,
            {
                "strategy": "light",
                "num_unfreeze_layers": self.cfg.freeze_layers,
                "unfreeze_embeddings": False,
            },
        )
        for p in policy.position_head.parameters():
            p.requires_grad = True

        return policy, reference, reward

    def fit(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config={**{f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}, **vars(self.args)},
            )

        for epoch in range(self.cfg.epochs):
            self.policy.train()
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self._train_batch(batch)
                pbar.set_postfix(
                    loss=f"{metrics['total_loss']:.3f}",
                    pol=f"{metrics['policy_loss']:.3f}",
                    pos=f"{metrics['position_loss']:.3f}",
                    kl=f"{metrics['kl_divergence']:.3f}",
                    rew=f"{metrics['reward_mean']:.3f}",
                    grad=f"{metrics['grad_norm']:.3f}",
                )
                if self.args.use_wandb:
                    wandb.log(metrics)

            ckpt = self.output_dir / f"grpo_rel_policy_epoch_{epoch + 1}.pt"
            torch.save(self.policy.state_dict(), ckpt)
            print(f"Saved checkpoint to {ckpt}")

        final_path = self.output_dir / "grpo_rel_policy_final.pt"
        torch.save(self.policy.state_dict(), final_path)
        print(f"Training complete. Final model stored at {final_path}")
        if self.args.use_wandb:
            wandb.finish()

    def _train_batch(self, batch) -> dict:
        wt_ab = batch["wt_ab_inputs_ids"].to(self.device)
        wt_ab_mask = batch["wt_ab_inputs_mask"].to(self.device)
        wt_ag = batch["wt_ag_inputs_ids"].to(self.device)
        wt_ag_mask = batch["wt_ag_inputs_mask"].to(self.device)
        cdr_masks = batch["cdr_masks"].to(self.device)

        with torch.inference_mode():
            rew_wt, *_ = self.reward(
                wt_ab_inputs_ids=wt_ab,
                wt_ab_inputs_mask=wt_ab_mask,
                mut_ab_inputs_ids=wt_ab,
                mt_ab_inputs_mask=wt_ab_mask,
                wt_ag_inputs_ids=wt_ag,
                wt_ag_inputs_mask=wt_ag_mask,
                mut_ag_inputs_ids=wt_ag,
                mt_ag_inputs_mask=wt_ag_mask,
            )
            rew_wt = rew_wt.squeeze()

        rollouts = {
            "mutated": [],
            "positions": [],
            "position_actions": [],
            "rewards": [],
            "raw_rewards": [],
        }
        current_ab = wt_ab.clone()
        temp = self._anneal(self.cfg.temperature_start, self.cfg.temperature_min)
        pos_temp = self._anneal(self.cfg.position_temperature_start, self.cfg.position_temperature_min)

        for _ in range(self.cfg.rollout_steps):
            logits, _, pos_probs = self.policy(
                wt_ab_inputs_ids=wt_ab,
                wt_ab_inputs_mask=wt_ab_mask,
                wt_ag_inputs_ids=wt_ag,
                wt_ag_inputs_mask=wt_ag_mask,
                mut_ab_inputs_ids=current_ab,
                mt_ab_inputs_mask=wt_ab_mask,
                mut_ag_inputs_ids=wt_ag,
                mt_ag_inputs_mask=wt_ag_mask,
                with_position=True,
            )
            mutated, positions, position_actions = sample_mutations(
                logits,
                pos_probs,
                current_ab,
                cdr_masks,
                max_mutations=self.cfg.max_mutations,
                temperature=temp,
                position_temperature=pos_temp,
                stochastic=True,
            )
            with torch.inference_mode():
                rew_mut, *_ = self.reward(
                    wt_ab_inputs_ids=wt_ab,
                    wt_ab_inputs_mask=wt_ab_mask,
                    mut_ab_inputs_ids=mutated,
                    mt_ab_inputs_mask=wt_ab_mask,
                    wt_ag_inputs_ids=wt_ag,
                    wt_ag_inputs_mask=wt_ag_mask,
                    mut_ag_inputs_ids=wt_ag,
                    mt_ag_inputs_mask=wt_ag_mask,
                )
                rew_mut = rew_mut.squeeze()

            reward = (rew_wt - rew_mut) * self.cfg.reward_scale

            rollouts["mutated"].append(mutated)
            rollouts["positions"].append(positions)
            rollouts["position_actions"].append(position_actions)
            rollouts["rewards"].append(reward)
            rollouts["raw_rewards"].append(rew_mut)
            current_ab = mutated

        rewards_tensor = torch.stack(rollouts["rewards"], dim=1)
        rewards_ori_tensor = torch.stack(rollouts["raw_rewards"], dim=1)

        total_kl = 0.0
        total_entropy = 0.0
        logp_steps: List[torch.Tensor] = []

        for step in range(self.cfg.rollout_steps):
            logits, _, pos_probs = self.policy(
                wt_ab_inputs_ids=wt_ab,
                wt_ab_inputs_mask=wt_ab_mask,
                wt_ag_inputs_ids=wt_ag,
                wt_ag_inputs_mask=wt_ag_mask,
                mut_ab_inputs_ids=rollouts["mutated"][step],
                mt_ab_inputs_mask=wt_ab_mask,
                mut_ag_inputs_ids=wt_ag,
                mt_ag_inputs_mask=wt_ag_mask,
                with_position=True,
            )
            logp_steps.append(
                compute_action_logp(
                    logits,
                    pos_probs,
                    rollouts["position_actions"][step],
                    rollouts["mutated"][step],
                    rollouts["positions"][step],
                    temperature=self._anneal(self.cfg.temperature_start, self.cfg.temperature_min),
                    position_weight=self.cfg.position_loss_weight,
                )
            )

            with torch.inference_mode():
                ref_logits = self.reference(
                    wt_ab_inputs_ids=wt_ab,
                    wt_ab_inputs_mask=wt_ab_mask,
                    wt_ag_inputs_ids=wt_ag,
                    wt_ag_inputs_mask=wt_ag_mask,
                    mut_ab_inputs_ids=rollouts["mutated"][step],
                    mt_ab_inputs_mask=wt_ab_mask,
                    mut_ag_inputs_ids=wt_ag,
                    mt_ag_inputs_mask=wt_ag_mask,
                )[3]

            mask = build_kl_mask(
                rollouts["positions"][step],
                logits.size(0),
                logits.size(1),
                logits.device,
            )
            if mask.any():
                p_log = F.log_softmax(logits[mask], dim=-1)
                q = F.softmax(ref_logits[mask], dim=-1)
                kl_val = F.kl_div(p_log, q, reduction="batchmean")
            else:
                kl_val = torch.tensor(0.0, device=logits.device)

            entropy = -(pos_probs * torch.log(pos_probs + 1e-8) + (1 - pos_probs) * torch.log(1 - pos_probs + 1e-8))
            entropy = entropy[cdr_masks].mean()

            total_kl += kl_val
            total_entropy += entropy

        total_kl /= self.cfg.rollout_steps
        mean_entropy = total_entropy / self.cfg.rollout_steps
        total_position_loss = mean_entropy * 0.01

        baseline = rewards_tensor.mean(dim=1, keepdim=True)
        std = rewards_tensor.std(dim=1, keepdim=True, unbiased=False)
        centered = (rewards_tensor - baseline) / (std + 1e-8)
        group_weights = F.softmax(centered / self.cfg.group_tau, dim=1).detach()

        logp_mat = torch.stack(logp_steps, dim=1)
        policy_loss = -(group_weights * logp_mat).sum(dim=1).mean()

        self._update_kl(total_kl.item())
        entropy_coef = self._anneal(self.cfg.entropy_weight, self.cfg.entropy_min)

        loss = (
            policy_loss * self.cfg.policy_loss_coef
            + torch.clamp(self.kl_coef * total_kl, max=self.cfg.kl_max)
            + total_position_loss
            - mean_entropy * entropy_coef * 0.1
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1

        metrics = {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "position_loss": total_position_loss.item(),
            "kl_divergence": total_kl.item(),
            "running_kl": self.running_kl,
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std(unbiased=False).item(),
            "reward_ori_mean": rewards_ori_tensor.mean().item(),
            "reward_ori_std": rewards_ori_tensor.std(unbiased=False).item(),
            "kl_coef": self.kl_coef,
            "lr": self.optimizer.param_groups[0]["lr"],
            "position_entropy": mean_entropy.item(),
            "grad_norm": grad_norm.item(),
        }
        return metrics

    def _anneal(self, start: float, end: float) -> float:
        progress = min(self.global_step / max(self.cfg.anneal_steps, 1), 1.0)
        return start + (end - start) * progress

    def _update_kl(self, current_kl: float) -> None:
        self.running_kl = self.cfg.kl_alpha * self.running_kl + (1 - self.cfg.kl_alpha) * current_kl
        if current_kl > self.cfg.target_kl * 2:
            self.kl_coef = min(self.kl_coef * 1.2, 10.0)
        if self.running_kl > 1.5 * self.cfg.target_kl:
            self.kl_coef = min(self.kl_coef * 1.5, 5.0)
        elif self.running_kl < 0.5 * self.cfg.target_kl:
            self.kl_coef = max(self.kl_coef * 0.5, 0.01)


def parse_args() -> argparse.Namespace:
    cfg_defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for antibody mutation policy.")
    parser.add_argument("--model_locate", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--wandb_project", type=str, default="ProtAttBA")
    parser.add_argument("--wandb_name", type=str, default="mutation_grpo")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--data_path", type=str, default=cfg_defaults.data_path)
    parser.add_argument("--cdr_info_path", type=str, default=cfg_defaults.cdr_info_path)
    parser.add_argument("--checkpoint_path", type=str, default=cfg_defaults.checkpoint_path)
    parser.add_argument("--output_dir", type=str, default=cfg_defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg_defaults.epochs)
    parser.add_argument("--lr", type=float, default=cfg_defaults.lr)
    parser.add_argument("--rollout_steps", type=int, default=cfg_defaults.rollout_steps)
    parser.add_argument("--max_mutations", type=int, default=cfg_defaults.max_mutations)
    parser.add_argument("--group_tau", type=float, default=cfg_defaults.group_tau)
    args = parser.parse_args()
    seed_everything(args.seed)
    return args


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.data_path = args.data_path
    cfg.cdr_info_path = args.cdr_info_path
    cfg.checkpoint_path = args.checkpoint_path
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.rollout_steps = args.rollout_steps
    cfg.max_mutations = args.max_mutations
    cfg.group_tau = args.group_tau
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    trainer = MutationGRPOTrainer(args, cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
