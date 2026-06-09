# progen2hf

Utilities for converting, evaluating, and PEFT-tuning ProGen-style Hugging Face checkpoints used by the AMP workflows.

## Convert Megatron Checkpoints

After merging Megatron tensor/pipeline parallel shards into a single rank, convert the checkpoint with:

```bash
python tools/convert_from_megatron.py /path/to/model_optim_rng.pt
```

The converter writes `config.json` and `pytorch_model.bin` next to the input checkpoint.

## Inspect A Hugging Face Checkpoint

```bash
python run.py \
  --model-path /path/to/progen2/checkpoint \
  --tokenizer-path /path/to/progen2/tokenizer
```

## PEFT Fine-Tuning

```bash
MODEL_PATH=/path/to/progen2-base \
DATA_PATH=/path/to/training_data \
OUTPUT_DIR=output/progen2-base-lora \
./run_peft.sh
```

`peft_progen.py` exposes the full training configuration through CLI arguments. Keep local checkpoints, training data, and generated outputs outside git.
