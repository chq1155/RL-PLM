# AMP Design Reinforcement Learning Toolkit

This directory provides training and sampling scripts for preference-based fine-tuning
of antimicrobial peptide models built on top of ProGen2. The updated CLI interfaces
avoid hard-coded paths, remove legacy dependencies, and expose concise arguments so
you can integrate the flows into your own infrastructure.

All scripts expect Python 3.10+, PyTorch with CUDA support, PEFT, TRL, and the
`progen2hf` package that ships with this repository available on `PYTHONPATH`.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt  # populate with your environment specifics
   ```
2. Make sure the ProGen2 base checkpoint and tokenizer are accessible on disk.
3. Collect a classifier checkpoint compatible with the reward function (see `mlp.py`).

> **Tip**: Set `CUDA_VISIBLE_DEVICES` to control which GPUs the scripts use and
> run `nvidia-smi` in another terminal to track progress.

## Direct Preference Optimisation (DPO)

`dpo.py` trains a LoRA-augmented policy against a binary classifier reward.

```bash
python dpo.py \
  --base-model-path /path/to/progen2/checkpoint \
  --tokenizer-path /path/to/progen2/tokenizer \
  --classifier-checkpoint /path/to/reward.pt \
  --output-dir runs/dpo \
  --steps 200 \
  --batch-size 8 \
  --epochs 3 \
  --device cuda:0
```

Important flags:

- `--prompt` or `--prompt-file` define starting contexts for generation.
- `--lora-checkpoint` loads a PEFT state dict and automatically remaps legacy keys.
- `--no-wandb` disables Weights & Biases logging if you prefer offline runs.

## Grouped Reinforcement Preference Optimisation (GRPO)

`grpo.py` launches a distributed GRPO trainer. Provide the same resource paths as
for DPO and supply the number of GPUs to use (defaults to all visible devices).

```bash
python grpo.py \
  --base-model-path /path/to/progen2/checkpoint \
  --tokenizer-path /path/to/progen2/tokenizer \
  --classifier-checkpoint /path/to/reward.pt \
  --world-size 4 \
  --batch-size 128 \
  --steps 400 \
  --output-dir runs/grpo
```

The script uses `torch.multiprocessing.spawn`; make sure `WORLD_SIZE` does not exceed
the number of GPUs. To queue the job until sufficient memory is available you can
wrap the invocation with `grpo.sh`:

```bash
GPU_IDS=0,1,2,3 MEMORY_THRESHOLD_MB=1200 REQUIRED_FREE_GPUS=4 \
  ./grpo.sh "python grpo.py --base-model-path ..."
```

## Proximal Policy Optimisation (PPO)

`ppo.py` fine-tunes a TRL `AutoModelForCausalLMWithValueHead` checkpoint.

```bash
python ppo.py \
  --model-path /path/to/trl/checkpoint \
  --tokenizer-path /path/to/progen2/tokenizer \
  --classifier-checkpoint /path/to/reward.pt \
  --output-dir runs/ppo \
  --steps 500 \
  --batch-size 64 \
  --ppo-epochs 2 \
  --device cuda
```

The script saves a checkpoint every `--save-every` steps and always writes a final
model to `<output-dir>/final_model`.

## Sequence Sampling

`generation.py` samples sequences from a fine-tuned policy and scores them with the
classifier reward to simplify candidate triage.

```bash
python generation.py \
  --model-path runs/dpo/final_model \
  --tokenizer-path /path/to/progen2/tokenizer \
  --classifier-checkpoint /path/to/reward.pt \
  --num-samples 2048 \
  --batch-size 64 \
  --output-dir exports \
  --csv-prefix dpo_samples
```

The generated CSV contains one row per sequence with its raw reward score so you
can post-process or filter downstream.

## Utilities

- `utils.py` exposes helpers for loading ESM models, preparing ProGen2 LoRA
  checkpoints, and cleaning noisy sequences.
- `reward.py` implements reward functions that expect ESM embeddings and outputs
  both raw scores and pass/fail masks.
- `dataset.py` constructs dataloaders backed by configurable prompts; pass your
  tokenizer and either a single prompt (`--prompt`) or a text file of prompts
  (`--prompt-file`).

Feel free to tailor the configuration defaults to your workloads—every CLI flag has
a sensible default so the scripts can be slotted into automated pipelines quickly.
