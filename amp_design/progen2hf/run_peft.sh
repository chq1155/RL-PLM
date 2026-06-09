#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/progen2-base}"
DATA_PATH="${DATA_PATH:-/path/to/training_data}"
OUTPUT_DIR="${OUTPUT_DIR:-output/progen2-base-lora}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate.yaml}"

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    peft_progen.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --lora_rank 128 \
    --lora_alpha 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-4 \
    --max_train_steps 4000 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 1
