from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW

from dataset import loading_dataset
from mlp import MLP
from reward import reward_amp_cls
from utils import clean_sequences, load_esm, load_pretrained_progen_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    tracker_project_name: str = "ampgen_dpo"
    exp_name: str = "dpo_run"
    steps: int = 100
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: int = 2
    beta: float = 0.2
    num_candidates: int = 3
    candidate_batch_size: int = 4
    lr: float = 5e-5
    warmup_steps: int = 15
    save_every: int = 10
    max_new_tokens: int = 48
    max_sequence_length: int = 50
    top_p: float = 0.95
    top_k: int = 20
    temperature: float = 0.9
    ref_cache_size: int = 2
    reward_margin_threshold: float = 0.01
    device: str = "cuda"
    seed: int = 913
    prompt: str = "<|bos|>"
    prompt_file: str | None = None
    tokenizer_path: Path | None = None
    base_model_path: Path | None = None
    lora_checkpoint: Path | None = None
    classifier_checkpoint: Path | None = None
    output_dir: Path = Path("dpo_runs")
    esm_mode: str = "8M"
    wandb_entity: str | None = None
    use_wandb: bool = True


class UltraLowMemoryDPOTrainer:
    def __init__(self, policy, tokenizer, device: torch.device, config: TrainingConfig):
        self.policy = policy.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = config
        self.original_state = {k: v.clone().detach().cpu() for k, v in self.policy.state_dict().items()}

        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        total_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameter count: {total_trainable_params:,}")

        self.optimizer = AdamW(
            trainable_params,
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        self.step = 0
        self.ref_cache: dict[str, torch.Tensor] = {}

    def _get_cache_key(self, query: torch.Tensor, response: torch.Tensor) -> str:
        query_info = f"{len(query)}_{query[0].item() if len(query) > 0 else 0}"
        response_info = f"{len(response)}_{response[0].item() if len(response) > 0 else 0}"
        return f"{query_info}_{response_info}"

    @staticmethod
    def _logits(output):
        return output[0] if isinstance(output, tuple) else output.logits

    def truncate_sequences(self, sequences: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [
            seq[-self.cfg.max_sequence_length :] if len(seq) > self.cfg.max_sequence_length else seq
            for seq in sequences
        ]

    def get_ref_logprobs(self, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        cache_key = self._get_cache_key(query, response)
        if cache_key in self.ref_cache:
            return self.ref_cache[cache_key]

        current_state = {k: v.clone() for k, v in self.policy.state_dict().items()}
        try:
            self.policy.load_state_dict(self.original_state, strict=False)
            self.policy.eval()

            with torch.no_grad():
                input_ids = torch.cat([query, response]).unsqueeze(0).to(self.device)
                ref_output = self.policy(input_ids, use_cache=False)
                ref_logits = self._logits(ref_output)[0][len(query) :]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)

                response_tokens = response[: ref_logprobs.size(0)].unsqueeze(1).to(self.device)
                token_logprobs = ref_logprobs[: response_tokens.size(0)].gather(1, response_tokens).squeeze(1)

            result = token_logprobs.detach().cpu()
            if len(self.ref_cache) < self.cfg.ref_cache_size:
                self.ref_cache[cache_key] = result

            return result.to(self.device)
        finally:
            self.policy.load_state_dict(current_state)
            self.policy.train()

    def policy_logprobs(self, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        input_ids = torch.cat([query, response]).unsqueeze(0).to(self.device)
        output = self.policy(input_ids, use_cache=False)
        logits = self._logits(output)[0][len(query) :]
        logprobs = F.log_softmax(logits, dim=-1)
        response_tokens = response[: logprobs.size(0)].unsqueeze(1).to(self.device)
        token_logprobs = logprobs[: response_tokens.size(0)].gather(1, response_tokens).squeeze(1)
        return token_logprobs.sum()

    def dpo_loss(self, pair: dict) -> torch.Tensor:
        query = pair["query"].to(self.device, non_blocking=True)
        preferred = pair["preferred"].to(self.device, non_blocking=True)
        rejected = pair["rejected"].to(self.device, non_blocking=True)

        ref_pref = self.get_ref_logprobs(query, preferred).sum()
        ref_rej = self.get_ref_logprobs(query, rejected).sum()
        policy_pref = self.policy_logprobs(query, preferred)
        policy_rej = self.policy_logprobs(query, rejected)

        diff = self.cfg.beta * (policy_pref - policy_rej - (ref_pref - ref_rej))
        loss = -F.logsigmoid(diff)
        return loss

    def step_batch(self, pairs: Sequence[dict]) -> float:
        if not pairs:
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        successful = 0

        for pair_idx, pair in enumerate(pairs):
            try:
                loss = self.dpo_loss(pair)
                if not torch.isfinite(loss) or loss.item() < 0:
                    continue

                (loss / len(pairs)).backward()
                total_loss += loss.item()
                successful += 1
                print(f"Pair {pair_idx} loss={loss.item():.6f}")
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on pair {pair_idx}, skipping")
                torch.cuda.empty_cache()
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error on pair {pair_idx}: {exc}")

        if successful == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0

        scale = len(pairs) / successful
        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad *= scale

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        if torch.isfinite(grad_norm) and grad_norm > 1e-8:
            self.optimizer.step()
            self.step += 1
            print(
                f"Batch update - avg_loss={total_loss / successful:.6f}, "
                f"grad_norm={grad_norm:.4f}, pairs={successful}"
            )
            if self.cfg.use_wandb:
                wandb.log(
                    {
                        "batch_loss": total_loss / successful,
                        "grad_norm": grad_norm.item(),
                        "successful_pairs": successful,
                        "step": self.step,
                    }
                )
        else:
            print(f"Gradient warning: {grad_norm}")
        return total_loss / successful

    def create_preference_pairs(
        self,
        queries: Sequence[torch.Tensor],
        candidates_list: Sequence[Sequence[torch.Tensor]],
        reward_fn,
    ) -> List[dict]:
        pairs = []
        for query_idx, (query, candidates) in enumerate(zip(queries, candidates_list)):
            valid_candidates: List[torch.Tensor] = []
            sequences: List[str] = []

            query_cpu = query.cpu()
            for cand in candidates:
                if cand.numel() == 0:
                    continue
                try:
                    cand_cpu = cand.cpu()
                    sequence = self.tokenizer.decode(
                        torch.cat([query_cpu, cand_cpu]),
                        skip_special_tokens=True,
                    )
                    if len(sequence.strip()) > 5:
                        sequences.append(sequence)
                        valid_candidates.append(cand_cpu)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Decode failure: {exc}")

            if len(valid_candidates) < 2:
                print(f"Only {len(valid_candidates)} valid candidates for query {query_idx}, skipping.")
                continue

            try:
                rewards, _ = reward_fn(clean_sequences(sequences))
                rewards = rewards.detach().cpu().numpy().astype(np.float32).flatten()
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Reward calculation failure: {exc}")
                continue

            if rewards.size != len(valid_candidates):
                print(f"Rewards do not match candidates: {rewards.size} vs {len(valid_candidates)}")
                continue

            sorted_idx = np.argsort(rewards)[::-1]
            best_idx = int(sorted_idx[0])
            worst_idx = int(sorted_idx[-1])

            margin = float(rewards[best_idx] - rewards[worst_idx])
            if margin < self.cfg.reward_margin_threshold:
                print(f"Skipping pair. Margin {margin:.6f} below threshold {self.cfg.reward_margin_threshold}.")
                continue

            pairs.append(
                {
                    "query": query_cpu,
                    "preferred": valid_candidates[best_idx],
                    "rejected": valid_candidates[worst_idx],
                    "preferred_reward": float(rewards[best_idx]),
                    "rejected_reward": float(rewards[worst_idx]),
                    "reward_margin": margin,
                }
            )
        print(f"{len(pairs)} preference pairs in total.")
        return pairs

    def generate_candidates(self, prompts: Sequence[torch.Tensor]) -> List[List[torch.Tensor]]:
        bad_ids = [self.tokenizer.encode(token, add_special_tokens=False) for token in ["B", "O", "U", "X", "Z"]]
        gen_cfg = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            temperature=self.cfg.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            bad_words_ids=bad_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict_in_generate=False,
        )

        all_candidates: List[List[torch.Tensor]] = []
        self.policy.eval()
        torch.set_grad_enabled(False)

        try:
            for prompt in prompts:
                candidates: List[torch.Tensor] = []
                for _ in range(self.cfg.num_candidates):
                    try:
                        prompt_input = prompt.unsqueeze(0).to(self.device, non_blocking=True)
                        output = self.policy.generate(prompt_input, **gen_cfg)
                        candidate = output[0][len(prompt) :].detach().cpu()
                        candidates.append(candidate)
                    except torch.cuda.OutOfMemoryError:
                        print("Skip. OOM during candidate generation.")
                        torch.cuda.empty_cache()
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"Skip. Error during candidate generation: {exc}")
                all_candidates.append(candidates)
        finally:
            torch.set_grad_enabled(True)
            self.policy.train()
        return all_candidates


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Run Direct Preference Optimization for AMP design.")
    parser.add_argument("--base-model-path", type=Path, required=True, help="Path to the base ProGen2 checkpoint.")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--lora-checkpoint", type=Path, help="Optional LoRA checkpoint to load.")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Path to the classifier weights.")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="Directory to store checkpoints.")
    parser.add_argument("--steps", type=int, default=defaults.steps, help="Number of optimisation steps.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=defaults.beta, help="DPO beta parameter.")
    parser.add_argument("--num-candidates", type=int, default=defaults.num_candidates, help="Samples per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens, help="Max generation length.")
    parser.add_argument("--max-sequence-length", type=int, default=defaults.max_sequence_length, help="Prompt trim length.")
    parser.add_argument("--top-p", type=float, default=defaults.top_p, help="Top-p sampling parameter.")
    parser.add_argument("--top-k", type=int, default=defaults.top_k, help="Top-k sampling parameter.")
    parser.add_argument("--temperature", type=float, default=defaults.temperature, help="Sampling temperature.")
    parser.add_argument("--device", type=str, default=defaults.device, help="Torch device to use.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    parser.add_argument("--prompt", type=str, default=defaults.prompt, help="Default prompt to seed decoding.")
    parser.add_argument("--prompt-file", type=Path, help="Optional prompt file, one prompt per line.")
    parser.add_argument("--esm-mode", type=str, default=defaults.esm_mode, choices=["8M", "650M"], help="ESM model size.")
    parser.add_argument("--tracker-project-name", type=str, default=defaults.tracker_project_name, help="wandb project name.")
    parser.add_argument("--exp-name", type=str, default=defaults.exp_name, help="wandb run name.")
    parser.add_argument("--reward-margin-threshold", type=float, default=defaults.reward_margin_threshold, help="Minimum reward margin to keep a pair.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb-entity", type=str, help="Optional wandb entity.")
    args = parser.parse_args()

    return TrainingConfig(
        tracker_project_name=args.tracker_project_name,
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        num_candidates=args.num_candidates,
        candidate_batch_size=defaults.candidate_batch_size,
        lr=args.lr,
        warmup_steps=defaults.warmup_steps,
        save_every=defaults.save_every,
        max_new_tokens=args.max_new_tokens,
        max_sequence_length=args.max_sequence_length,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        ref_cache_size=defaults.ref_cache_size,
        reward_margin_threshold=args.reward_margin_threshold,
        device=args.device,
        seed=args.seed,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        tokenizer_path=args.tokenizer_path,
        base_model_path=args.base_model_path,
        lora_checkpoint=args.lora_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        output_dir=args.output_dir,
        esm_mode=args.esm_mode,
        wandb_entity=args.wandb_entity,
        use_wandb=not args.no_wandb,
    )


def load_policy_and_tokenizer(cfg: TrainingConfig):
    tokenizer, model = load_pretrained_progen_model(
        base_model_path=str(cfg.base_model_path),
        tokenizer_path=str(cfg.tokenizer_path),
        lora_checkpoint=str(cfg.lora_checkpoint) if cfg.lora_checkpoint else None,
    )
    return tokenizer, model


def load_reward_model(cfg: TrainingConfig, device: torch.device):
    batch_converter, esm_model, alphabet = load_esm(cfg.esm_mode, device=device)
    classifier = MLP(input_dim=320, hidden_dim=128)
    if cfg.classifier_checkpoint is None:
        raise ValueError("Classifier checkpoint must be provided.")
    state = torch.load(cfg.classifier_checkpoint, map_location="cpu")
    classifier.load_state_dict(state)
    classifier = classifier.to(device).eval()
    return batch_converter, esm_model, alphabet, classifier


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = parse_args()
    if cfg.base_model_path is None or cfg.tokenizer_path is None:
        raise ValueError("Both base model path and tokenizer path are required.")

    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if cfg.use_wandb:
        wandb.init(
            project=cfg.tracker_project_name,
            name=cfg.exp_name,
            config=asdict(cfg),
            entity=cfg.wandb_entity,
        )

    tokenizer, model = load_policy_and_tokenizer(cfg)
    trainer = UltraLowMemoryDPOTrainer(model, tokenizer, device, cfg)

    batch_converter, esm_model, alphabet, classifier = load_reward_model(cfg, device)

    def reward_fn(seqs: Iterable[str]):
        return reward_amp_cls(
            seqs,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            classifier=classifier,
            device=device,
        )

    dataloader = loading_dataset(
        cfg.steps,
        cfg.batch_size,
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        prompt_file=cfg.prompt_file,
        shuffle=False,
    )

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1}")
            try:
                prompts = batch["input_ids"].to(device)
                prompts = [prompt for prompt in prompts]
                prompts = trainer.truncate_sequences(prompts)

                candidates = trainer.generate_candidates(prompts)
                pairs = trainer.create_preference_pairs(prompts, candidates, reward_fn)
                if not pairs:
                    print("Skip. No valid preference pairs.")
                    continue

                trainer.step_batch(pairs)

                if trainer.step % cfg.save_every == 0 and trainer.step > 0:
                    checkpoint_dir = cfg.output_dir / f"checkpoint_step_{trainer.step}"
                    ensure_output_dir(checkpoint_dir)
                    trainer.policy.save_pretrained(checkpoint_dir, safe_serialization=True)
                    print(f"Saving checkpoint: {checkpoint_dir}")
            except torch.cuda.OutOfMemoryError:
                print("Skip. OOM during batch processing.")
                torch.cuda.empty_cache()
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error on batch processing: {exc}")

    final_dir = cfg.output_dir / "final_model"
    ensure_output_dir(final_dir)
    trainer.policy.save_pretrained(final_dir, safe_serialization=True)
    print(f"Saved final checkpoint: {final_dir}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
