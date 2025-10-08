from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from dataset import loading_dataset
from mlp import MLP
from reward import reward_amp_cls
from utils import clean_sequences, load_esm


def set_seed(seed: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_generate_config(args, tokenizer) -> dict:
    bad_words = ["B", "O", "U", "X", "Z"]
    bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
    return {
        "max_length": args.max_length,
        "num_return_sequences": 1,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "bad_words_ids": bad_words_ids,
    }


def log_stats(sequences: List[str], rewards: torch.Tensor) -> None:
    if rewards.numel() == 0:
        return
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.float32)
    wandb.log(
        {
            "reward/max": rewards.max().item(),
            "reward/mean": rewards.mean().item(),
            "reward/min": rewards.min().item(),
            "response/avg_length": lengths.mean().item(),
        }
    )


def prepare_reward_model(classifier_path: Path, esm_mode: str, device: torch.device) -> Tuple[callable, torch.nn.Module]:
    batch_converter, esm_model, alphabet = load_esm(esm_mode, device=device)
    classifier = MLP(input_dim=320, hidden_dim=128).to(device)
    state = torch.load(classifier_path, map_location="cpu")
    classifier.load_state_dict(state)
    classifier.eval()

    def reward_fn(seqs: Iterable[str]):
        return reward_amp_cls(
            seqs,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            classifier=classifier,
            device=device,
        )

    return reward_fn, classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO fine-tuning for AMP design.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a causal LM checkpoint.")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Tokenizer directory.")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Classifier weights for rewards.")
    parser.add_argument("--output-dir", type=Path, default=Path("ppo_runs"), help="Directory to store checkpoints.")
    parser.add_argument("--steps", type=int, default=500, help="Number of PPO steps.")
    parser.add_argument("--batch-size", type=int, default=128, help="Global batch size.")
    parser.add_argument("--mini-batch-size", type=int, default=32, help="Mini batch size.")
    parser.add_argument("--ppo-epochs", type=int, default=2, help="Number of PPO epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Optimizer learning rate.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p value for sampling.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k value for sampling.")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam search width during generation.")
    parser.add_argument("--max-length", type=int, default=51, help="Maximum total sequence length during generation.")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty during generation.")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty during generation.")
    parser.add_argument("--seed", type=int, default=822, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use.")
    parser.add_argument("--prompt", type=str, default="<|bos|>", help="Prompt used to seed generation.")
    parser.add_argument("--prompt-file", type=Path, help="Optional file with prompts, one per line.")
    parser.add_argument("--esm-mode", type=str, default="8M", choices=["8M", "650M"], help="ESM model size.")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N PPO steps.")
    parser.add_argument("--tracker-project-name", type=str, default="ampgen_ppo", help="wandb project name.")
    parser.add_argument("--exp-name", type=str, default="ppo_run", help="wandb run name.")
    parser.add_argument("--wandb-entity", type=str, help="Optional wandb entity.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    return parser.parse_args()


def create_ppo_config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        tracker_project_name=args.tracker_project_name,
        exp_name=args.exp_name,
        log_with=None if args.no_wandb else "wandb",
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        early_stopping=True,
        is_peft_model=False,
        seed=args.seed,
        optimize_cuda_cache=True,
        optimize_device_cache=True,
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
    )


def train(
    args: argparse.Namespace,
    tokenizer,
    model: AutoModelForCausalLMWithValueHead,
    reward_fn,
    generate_config: dict,
) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataloader = loading_dataset(
        args.steps,
        args.batch_size,
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        shuffle=False,
    )

    trainer = PPOTrainer(
        model=model,
        config=create_ppo_config(args),
        tokenizer=tokenizer,
    )

    if not args.no_wandb:
        wandb.init(
            project=args.tracker_project_name,
            name=args.exp_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    step = 0
    ensure_dir(args.output_dir)

    for epoch in range(args.ppo_epochs):
        for batch in dataloader:
            queries = batch["input_ids"].to(device)
            query_tensors = [queries[i] for i in range(queries.size(0))]

            responses = trainer.generate(query_tensors, **generate_config)
            decoded = clean_sequences(
                [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
            )

            rewards, _ = reward_fn(decoded)
            rewards = rewards.to(device)
            reward_tensors = [torch.tensor([val], device=device, dtype=torch.float32) for val in rewards]

            if not args.no_wandb:
                log_stats(decoded, rewards.detach().cpu())

            trainer.step(query_tensors, responses, reward_tensors)
            step += 1

            if step % args.save_every == 0:
                checkpoint_dir = args.output_dir / f"checkpoint_step_{step}"
                ensure_dir(checkpoint_dir)
                trainer.save_pretrained(checkpoint_dir)

    final_dir = args.output_dir / "final_model"
    ensure_dir(final_dir)
    trainer.save_pretrained(final_dir)

    if not args.no_wandb:
        wandb.finish()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = model.to(torch.device(args.device if torch.cuda.is_available() else "cpu"))

    reward_fn, _ = prepare_reward_model(args.classifier_checkpoint, args.esm_mode, torch.device(args.device if torch.cuda.is_available() else "cpu"))
    generate_config = create_generate_config(args, tokenizer)

    train(args, tokenizer, model, reward_fn, generate_config)


if __name__ == "__main__":
    main()
