from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mlp import MLP
from progen2hf.models import ProGenConfig, ProGenForCausalLM
from reward import reward_amp_cls
from utils import load_esm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample sequences from a TRL checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the TRL checkpoint.")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Tokenizer directory.")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Classifier weights for rewards.")
    parser.add_argument("--output-dir", type=Path, default=Path("samples"), help="Directory to store CSV outputs.")
    parser.add_argument("--num-samples", type=int, default=1024, help="Total number of samples to generate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during sampling.")
    parser.add_argument("--prompt", type=str, default="<|bos|>", help="Prompt used to seed generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use.")
    parser.add_argument("--esm-mode", type=str, default="8M", choices=["8M", "650M"], help="ESM model size.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling value.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling value.")
    parser.add_argument("--max-length", type=int, default=51, help="Total sequence length during generation.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--csv-prefix", type=str, default="samples", help="Prefix for CSV output files.")
    return parser.parse_args()


def load_trl_model(model_path: Path, tokenizer_path: Path):
    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def build_reward_fn(classifier_path: Path, esm_mode: str, device: torch.device) -> Callable[[Iterable[str]], Tuple[torch.Tensor, torch.Tensor]]:
    batch_converter, esm_model, alphabet = load_esm(esm_mode, device=device)
    classifier = MLP(input_dim=320, hidden_dim=128).to(device)
    state = torch.load(classifier_path, map_location="cpu")
    classifier.load_state_dict(state)
    classifier.eval()

    def scorer(seqs: Iterable[str]):
        return reward_amp_cls(
            seqs,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            classifier=classifier,
            device=device,
        )

    return scorer


def generate_sequences(
    model,
    tokenizer,
    prompt: str,
    generate_config: Dict,
    device: torch.device,
) -> List[str]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, **generate_config)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def sampling_loop(
    model,
    tokenizer,
    reward_fn,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[str], List[float]]:
    generate_config = {
        "max_length": args.max_length,
        "num_return_sequences": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "bad_words_ids": [tokenizer.encode(word, add_special_tokens=False) for word in ["B", "O", "U", "X", "Z"]],
    }

    num_batches = max(1, math.ceil(args.num_samples / args.batch_size))
    sequences: List[str] = []
    rewards: List[float] = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating sequences"):
            batch_sequences = generate_sequences(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                generate_config=generate_config,
                device=device,
            )
            sequences.extend(batch_sequences)
            reward_tensor, _ = reward_fn(batch_sequences)
            rewards.extend(reward_tensor.cpu().numpy().tolist())

    return sequences[: args.num_samples], rewards[: args.num_samples]


def write_csv(path: Path, sequences: Iterable[str], rewards: Iterable[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "reward"])
        for seq, reward in zip(sequences, rewards):
            writer.writerow([seq, reward])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ensure_dir(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_trl_model(args.model_path, args.tokenizer_path)
    reward_fn = build_reward_fn(args.classifier_checkpoint, args.esm_mode, device)

    sequences, rewards = sampling_loop(model, tokenizer, reward_fn, args, device)

    csv_path = args.output_dir / f"{args.csv_prefix}.csv"
    write_csv(csv_path, sequences, rewards)
    print(f"Saved {len(sequences)} samples to {csv_path}")


if __name__ == "__main__":
    main()
