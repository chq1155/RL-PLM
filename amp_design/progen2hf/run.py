import argparse

import torch
from models import ProGenConfig, ProGenForCausalLM, ProGenTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Load a ProGen checkpoint for inspection.")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face checkpoint directory.")
    parser.add_argument("--tokenizer-path", help="Tokenizer directory. Defaults to --model-path.")
    parser.add_argument("--manual-tokenizer", action="store_true", help="Use the local ProGenTokenizer implementation.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_path

    if args.manual_tokenizer:
        tokenizer = ProGenTokenizer(f"{tokenizer_path}/tokenizer.json")
        progen_model = ProGenForCausalLM.from_pretrained(
            args.model_path,
            revision=args.revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        AutoConfig.register("progen", ProGenConfig)
        AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        progen_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            revision=args.revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    for _, param in progen_model.named_parameters():
        param.requires_grad = False
    print(f"Loaded model with vocab size {len(tokenizer)}")


if __name__ == "__main__":
    main()
