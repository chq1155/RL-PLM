# Antibody Mutation

This folder contains the main antibody affinity and mutation-policy workflows used by RL-PLM.

## Setup

```bash
conda create -n rl-plm-ab python=3.10
conda activate rl-plm-ab
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel is not suitable.

## Data And Checkpoints

Download antibody data and checkpoints from the project Google Drive linked in the root README. The commands below assume this layout after download:

```text
antibody_mutation/
  data/
    identity_data/
      csv_AB1101/
      csv_AB645/
      csv_S1131/
    sigmul_data/
      AB1101_multiple_cdr_balance_train.csv
      AB1101_multiple_cdr_balance_test.csv
      cdr_info.csv
  model/
    esm2_650m/
  checkpoints_identity_sigmul/
    AB1101/
      <ProtAttBA checkpoint>.ckpt
```

The `data/`, checkpoint, result, and run-output folders are ignored by git.

## Supervised Reward Models

Sequence-identity split:

```bash
python trainer_identity.py \
  --model_locate ./model/esm2_650m \
  --data_folder ./data/identity_data/csv_AB1101 \
  --data_name AB1101 \
  --devices 1 \
  --accelerator gpu \
  --strategy auto \
  --seed 42 \
  --rm_abnormal false
```

Single-mutation training with multi-mutation evaluation:

```bash
python trainer_sigmul.py \
  --model_locate ./model/esm2_650m \
  --data_folder ./data/sigmul_data \
  --data_name AB1101 \
  --devices 1 \
  --accelerator gpu \
  --strategy auto \
  --seed 42 \
  --rm_abnormal true
```

Equivalent shell wrappers are in `scripts/bash_seq_identity.sh` and `scripts/bash_seq_sigmul.sh`; edit paths in the shell wrapper or pass the Python arguments directly as above.

## Evaluation

```bash
python eval.py \
  --model_locate ./model/esm2_650m \
  --ckpt_locate ./checkpoints_identity_sigmul/AB1101/<ProtAttBA checkpoint>.ckpt \
  --filt_path ./data/sigmul_data/AB1101_multiple_cdr_balance_test.csv \
  --preds_path ./test_preds/AB1101 \
  --device cuda:0 \
  --seed 42
```

## Mutation Policy Fine-Tuning

PPO:

```bash
python mutation_policy.py \
  --data_path ./data/sigmul_data/AB1101_multiple_cdr_balance_train.csv \
  --cdr_info_path ./data/sigmul_data/cdr_info.csv \
  --checkpoint_path ./checkpoints_identity_sigmul/AB1101/<ProtAttBA checkpoint>.ckpt \
  --output_dir ppo_runs/AB1101 \
  --device cuda:0 \
  --seed 42 \
  --batch_size 32 \
  --rollout_steps 4 \
  --max_mutations 4
```

GRPO:

```bash
python mutation_policy_grpo.py \
  --data_path ./data/sigmul_data/AB1101_multiple_cdr_balance_train.csv \
  --cdr_info_path ./data/sigmul_data/cdr_info.csv \
  --checkpoint_path ./checkpoints_identity_sigmul/AB1101/<ProtAttBA checkpoint>.ckpt \
  --output_dir grpo_runs/AB1101 \
  --device cuda:0 \
  --seed 42 \
  --batch_size 32 \
  --rollout_steps 4 \
  --max_mutations 4 \
  --group_tau 0.5
```

Key flags:

- `--data_path` / `--cdr_info_path`: training CSV and CDR annotations.
- `--checkpoint_path`: pretrained ProtAttBA checkpoint used for initialization, reference, and reward models.
- `--output_dir`: destination for policy checkpoints.
- `--device` / `--seed`: hardware selection and deterministic initialization.
- `--rollout_steps` / `--max_mutations`: rollout length and number of simultaneous point mutations.
- `--use_wandb`: optional Weights & Biases logging.
