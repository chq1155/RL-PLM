# Kinase Mutation

Main-stream code for the PhoQ kinase mutation experiments. The entrypoint is
`PhoQ_env.py`, which trains an ESM-backed mutation policy with PPO, DPO, or GRPO.

## Requirements

```bash
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build separately if your cluster requires a
specific CUDA version.

## Data And Model Files

Expected default layout:

```text
kinase_mutation/
  data/
    PhoQ.csv
    train_init_sequences.csv
  esm_8m/
    config.json
    pytorch_model.bin
    tokenizer_config.json
    vocab.txt
```

You can also pass all paths explicitly with `--train_init_path`,
`--fitness_path`, and `--model_dir`.

## Training

PPO:

```bash
python PhoQ_env.py \
  --algorithm PPO \
  --model_name ESM_8M \
  --model_dir ./esm_8m \
  --train_init_path ./data/train_init_sequences.csv \
  --fitness_path ./data/PhoQ.csv \
  --path ./checkpoints \
  --device cuda:0 \
  --seed 42 \
  --steps 10000 \
  --num_envs 10 \
  --max_step 3 \
  --score_stop_criteria 60
```

DPO:

```bash
python PhoQ_env.py \
  --algorithm DPO \
  --model_name ESM_8M \
  --model_dir ./esm_8m \
  --path ./checkpoints \
  --device cuda:0 \
  --seed 42 \
  --steps 10000
```

GRPO:

```bash
python PhoQ_env.py \
  --algorithm GRPO \
  --model_name ESM_8M \
  --model_dir ./esm_8m \
  --path ./checkpoints \
  --device cuda:0 \
  --seed 42 \
  --steps 10000
```

## Key Arguments

- `--algorithm`: one of `PPO`, `DPO`, or `GRPO`.
- `--model_name`: one of `ESM_8M`, `ESM_35M`, or `ESM_650M`.
- `--model_dir`: local Hugging Face model directory for the selected ESM model.
- `--train_init_path` / `--fitness_path`: CSV inputs for initial states and rewards.
- `--device` / `--seed`: hardware selection and deterministic initialization.
- `--path`: checkpoint output directory.
- `--tensorboard_log`: optional TensorBoard log directory.

Generated checkpoints and logs are ignored by git.

## Monitoring

```bash
tensorboard --logdir ./tensorboard_logs
```

## Acknowledgments

This folder builds on Stable Baselines3 and KnowRLM-style reinforcement learning
components.
