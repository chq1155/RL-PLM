#!/bin/bash
export PYTHONPATH="$PYTHONPATH:./"

DEVICES=1
NUM_NODES=1
SEED=3407
MAX_EPOCHS=150
ACC_BATCH=2
LR=3e-5
PATIENCE=15
STRATEGY="auto"
BATCH_SIZE=16
HIDDEN_SIZE=1280
NUM_HEADS=4
DROPOUT=0.1
MODEL_LOCATE="facebook/esm2_t33_650M_UR50D"
loss="mse"
data_folder="./data/identity_data/csv_AB645"
test_ratio=0.2
data_name="AB645"
monitor="val_mse"

python ./trainer_identity.py \
  --batch_size $BATCH_SIZE \
  --seed $SEED \
  --max_epochs $MAX_EPOCHS \
  --patience $PATIENCE \
  --lr $LR \
  --devices $DEVICES \
  --num_nodes $NUM_NODES \
  --strategy $STRATEGY \
  --accumulate_grad_batches $ACC_BATCH \
  --hidden_size $HIDDEN_SIZE \
  --num_heads $NUM_HEADS \
  --dropout $DROPOUT \
  --model_locate $MODEL_LOCATE \
  --loss $loss \
  --data_folder $data_folder \
  --data_name $data_name \
  --monitor $monitor \
  --test_ratio $test_ratio \
  --freeze_backbone