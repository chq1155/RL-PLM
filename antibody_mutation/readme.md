# Sequence-Only Prediction of Binding Affinity Changes: A Robust and Interpretable Model for Antibody Engineering

## Introduction

ProtAttBA is a protein language model that predicts binding affinity changes based solely on the sequence information of antibody-antigen complexes.

## Usage

### Install

1. Create conda environment 

```bash
conda create -n protab python==3.10
```

2. Install environment dependency

```bash
# activate environment
source activate protab
# install pytorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
(or use pip: pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118)

# install dependencies
pip install -r ./requirments.txt
```

### dataset

Cross validation dataset is located in the  ```cross_validation/data/csv``` folder  (Source: [Jin et al., 2024](https://github.com/ruofanjin/AttABseq)  ）

Sequence identity dataset is located in the ```seq-identity_sig-mul/data/identity_data``` folder (Use MMseqs with ```--min-seq-id 0.3```)

Single mutation training and multi-mutation testing dataset is located in the ```seq-identity_sig-mul/data/sigmul_data``` folder

### Training

```bash
# For cross validation you can use cross_validation/scripts/bash_cross-validation.sh with different args
cp bash_cross-validation.sh ../
bash bash_cross-validation.sh 

# For Sequence identity you can use seq-identity_sig-mul/scripts/bash_seq_identity.sh with different args
cp bash_seq_identity.sh ../ 
bash bash_seq_identity.sh

# For Single mutation training and multi-mutation testing you can use seq-identity_sig-mul/scripts/bash_seq_sigmul.sh with different args
cp bash_seq_sigmul.sh ../ 
bash bash_seq_sigmul.sh
```

### Evaluation

```bash
# For evaluation you can use the seq-identity_sig-mul/eval.py to predict the result by change the args
python eval.py
```

### Mutation Policy Fine-Tuning

Two reinforcement-learning scripts are provided to adapt the base ProtAttBA policy for antibody mutation design:

| Script | Algorithm | Notes |
| ------ | --------- | ----- |
| `mutation_policy.py` | PPO with value and position heads | Deterministic mutation sampling, GAE advantages |
| `mutation_policy_grpo.py` | Grouped GRPO | Stochastic sampling with softmax-based credit assignment |

Both scripts share the same CLI. Typical usage:

```bash
# PPO-style fine-tuning
python mutation_policy.py \
  --data_path ./data/sigmul_data/AB1101_multiple_cdr_balance_train.csv \
  --checkpoint_path ./checkpoints_identity_sigmul/AB1101/esm2_t33_650M_UR50D_AB1101-val_pearson_corr_lr-3e-05_loss-mse_tok33.ckpt \
  --output_dir ppo_runs/AB1101 \
  --batch_size 32 \
  --rollout_steps 4 \
  --max_mutations 4 \
  --use_wandb

# GRPO-style fine-tuning
python mutation_policy_grpo.py \
  --data_path ./data/sigmul_data/AB1101_multiple_cdr_balance_train.csv \
  --checkpoint_path ./checkpoints_identity_sigmul/AB1101/esm2_t33_650M_UR50D_AB1101-val_pearson_corr_lr-3e-05_loss-mse_tok33.ckpt \
  --output_dir grpo_runs/AB1101 \
  --batch_size 32 \
  --rollout_steps 4 \
  --max_mutations 4 \
  --group_tau 0.5
```

Key flags:

- `--data_path` / `--cdr_info_path`: CSV with training sequences and optional CDR annotations.
- `--checkpoint_path`: Pre-trained ProtAttBA checkpoint used for initialization, reference, and reward models.
- `--output_dir`: Destination directory for intermediate and final policy checkpoints.
- `--rollout_steps` / `--max_mutations`: Control the length of each mutation rollout and the number of simultaneous point mutations.
- `--use_wandb`: Enable Weights & Biases logging (project/name configured via `--wandb_project` and `--wandb_name`).

Both trainers automatically thaw the top transformer blocks, add a position head, and save `base.pt` before optimization as well as per-epoch checkpoints and a final policy snapshot.
