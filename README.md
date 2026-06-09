# RL-PLM

Code release for **The Forgetting-Learning Trade-off: Making Reinforcement
Learning Work for Protein Language Models**.

The paper studies when reinforcement learning expands or shrinks a protein
language model's capabilities. This repository keeps the source needed to run
the available experiment pipelines from this code snapshot and to reproduce the
paper's main capability diagnostics from generated samples.

## What Is Included

- `amp_design/`: antimicrobial peptide generation with DPO, PPO, and GRPO on
  ProGen2-style autoregressive policies.
- `kinase_mutation/`: PhoQ kinase mutation optimization with PPO, DPO, and GRPO
  using ESM-backed policies.
- `antibody_mutation/`: antibody CDR mutation policy optimization with PPO and
  GRPO on ProtAttBA-style reward models.
- `analysis/`: pass@k, ESR, Dual-Reward ESR, and delta-ESR utilities for
  reproducing the paper's main measurement tables from scored sample CSVs.

Large datasets, model checkpoints, generated samples, TensorBoard logs, W&B
runs, and result exports are intentionally not committed.

## Main Results To Reproduce

The paper's main empirical claim is not only that pass@k improves after RL, but
that capability-level diagnostics reveal different regimes:

- ESR > 1 with delta-ESR near 0: genuine expansion.
- ESR < 1 with delta-ESR near 0: coverage bottleneck.
- training-reward ESR much larger than independent-evaluator ESR: reward hacking.

For the tasks available in this repository, the relevant reproduction targets are:

| Task | Folder | Main diagnostic |
| --- | --- | --- |
| AMP design | `amp_design/` | predicted full-coverage reward; Dual-Reward ESR exposes reward hacking |
| Kinase mutation | `kinase_mutation/` | verifiable sparse reward; ESR exposes a coverage bottleneck |
| Antibody mutation | `antibody_mutation/` | predicted sparse reward; Dual-Reward ESR exposes reward hacking |

The inverse-folding experiments discussed in the paper were not present as code
in this source snapshot, so this OSS tree does not include an empty submodule or
placeholder implementation for them.

## Data And Checkpoints

Datasets and pretrained checkpoints are distributed separately:

[RL_PLM_data on Google Drive](https://drive.google.com/drive/folders/1_B0OEdwxUbMbncftXQypsoLvuMgIxrxu?usp=sharing)

Download only the files needed for the task you want to run and place them under
the layout described in each task README. You can also pass explicit paths
through the command-line arguments.

## Setup

Use one environment per task when possible because the AMP, antibody, and kinase
workflows depend on different model stacks.

```bash
git clone https://github.com/chq1155/RL-PLM.git
cd RL-PLM

python -m venv .venv
source .venv/bin/activate
```

Install the task dependencies you need:

```bash
pip install -r amp_design/requirements.txt
# or
pip install -r antibody_mutation/requirements.txt
# or
pip install -r kinase_mutation/requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel is
not appropriate for your machine.

For the lightweight analysis utilities:

```bash
pip install pandas pytest
```

## Reproduction Workflow

1. Train or load the base and RL-tuned policies for one task.
2. Generate samples for the base PLM and the RL-tuned PLM using the same prompts
   or initial states.
3. Score each sample with the task reward and, for predicted-reward tasks, an
   independent evaluator.
4. Save one row per generated sequence in a CSV.
5. Run `analysis/metrics.py` to compute pass@k, ESR, Dual-Reward ESR, and
   delta-ESR.

Example single-reward ESR for kinase:

```bash
python analysis/metrics.py \
  --samples exports/kinase_scored_samples.csv \
  --context-col context_id \
  --model-col model \
  --base-model base \
  --rl-model rl \
  --rank-col rank \
  --k 32 \
  --score-col fitness \
  --threshold 0.0 \
  --direction higher
```

Example Dual-Reward ESR for AMP:

```bash
python analysis/metrics.py \
  --samples exports/amp_scored_samples.csv \
  --context-col context_id \
  --model-col model \
  --base-model base \
  --rl-model rl \
  --rank-col rank \
  --k 32 \
  --train-score-col apexmic_score \
  --independent-score-col heldout_mic_score \
  --train-threshold 0.5 \
  --independent-threshold 0.5 \
  --train-direction higher \
  --independent-direction higher
```

Task-specific training and generation commands are documented in:

- [AMP design](amp_design/README.md)
- [Kinase mutation](kinase_mutation/README.md)
- [Antibody mutation](antibody_mutation/readme.md)
- [Analysis utilities](analysis/README.md)

## Repository Hygiene

The `.gitignore` excludes generated artifacts:

- Python bytecode and local caches
- datasets and downloaded model checkpoints
- training checkpoints and model weights
- TensorBoard and Weights & Biases logs
- generated result CSVs and sampled exports

Keep experiment-specific paths in shell scripts, job files, or command-line
arguments instead of editing Python source files.

## Citation

```bibtex
@inproceedings{cao2026forgetting,
  title={The Forgetting-Learning Trade-off: Making Reinforcement Learning Work for Protein Language Models},
  author={Cao, Hanqun and Zhang, Hongrui and Xu, Junde and Zhang, Zhou and Shen, Lingdong and Sun, Minghao and Liu, Ge and Xu, Jinbo and Li, Wu-Jun and Ni, Jinren and de la Fuente-Nunez, Cesar and Fu, Tianfan and Jin, Shuting and Heng, Pheng-Ann and Wu, Fang},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026},
  doi={10.1145/3770855.3818895}
}
```
