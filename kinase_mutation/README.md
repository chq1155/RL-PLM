# Kinase Mutation 

Implementation of Kinase Mutation

## Requirements

- Python 3.10+
- PyTorch 1.13.1+ (with CUDA support)
- Transformers 4.29.0+
- Stable Baselines3
- Gym 0.26.2+

## Installation

```bash
pip install -r requirements.txt
```

Download ESM 8M, ESM 35M, or ESM 650M Model to this folder.
## Quick Start

### 1. Training Models

Train with PPO algorithm:

```bash
python PhoQ_env.py \
  --algorithm PPO \
  --path ./checkpoints \
  --steps 10000 \
  --num_envs 10 \
  --max_step 3 \
  --score_stop_criteria 60
```

Train with DPO algorithm:

```bash
python PhoQ_env.py \
  --algorithm DPO \
  --path ./checkpoints \
  --steps 10000 \
  --num_envs 10 \
  --max_step 3 \
  --score_stop_criteria 60
```

### 2. Main Parameters

- `--algorithm`: Reinforcement learning algorithm 
- `--path`: Model save path
- `--steps`: Total training steps
- `--num_envs`: Number of parallel environments
- `--max_step`: Maximum steps per episode
- `--score_stop_criteria`: Fitness threshold for stopping training
- `--gamma`: Discount factor (default: 0)
- `--ent_coef`: Entropy coefficient for encouraging exploration (default: 0)
- `--clip`: PPO clipping parameter (default: 0.2)

## Model Configuration

### ESM Model Selection

Modify the `model_name` variable in `ESM_PhoQ.py`:

```python
model_name = "ESM_8M"    # 8M parameter version
model_name = "ESM_35M"   # 35M parameter version  
model_name = "ESM_650M"  # 650M parameter version
```

### Device Configuration

Default CUDA device usage:

```python
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```

## Training Monitoring

Training process generates TensorBoard logs, which can be viewed with the following command:

```bash
tensorboard --logdir ./tensorboard_logs
```

## Eval

```bash
python test_passk.py
```
And
```bash
python calculate_passk.py
```

## Acknowledgments

Thanks to [StableBaseline3](https://github.com/DLR-RM/stable-baselines3) and [KnowRLM](https://github.com/HICAI-ZJU/KnowRLM). We build this library based on their codebase.


