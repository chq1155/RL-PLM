accelerate launch \
    --config_file configs/accelerate.yaml \
    peft_progen.py \
    --model_path /HOME/progen2/progen2-base \
    --data_path /root/cath \
    --lora_rank 128 \
    --lora_alpha 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-4 \
    --max_train_steps 4000 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --output_dir output/progen2-base-lora-cath \
    --seed 1 \
    --with_tracking 

# accelerate launch \
#     --config_file configs/accelerate.yaml \
#     peft_progen.py \
#     --model_path /HOME/progen2/progen2-base \
#     --data_path /root/cath \
#     --lora_rank 32 \
#     --lora_alpha 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 4e-4 \
#     --max_train_steps 4000 \
#     --gradient_accumulation_steps 1 \
#     --num_warmup_steps 100 \
#     --output_dir output/progen2-base-lora-cath \
#     --seed 1 \
#     --shuffle_seed 99 \
#     --with_tracking 

# accelerate launch \
#     --config_file configs/accelerate.yaml \
#     peft_progen.py \
#     --model_path /HOME/progen2/progen2-base \
#     --data_path /root/cath \
#     --lora_rank 128 \
#     --lora_alpha 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 4e-4 \
#     --max_train_steps 4000 \
#     --gradient_accumulation_steps 1 \
#     --num_warmup_steps 100 \
#     --output_dir output/progen2-base-lora-cath \
#     --seed 1 \
#     --shuffle_seed 123 \
#     --with_tracking 

# /usr/local/dros/python/bin/accelerate launch \
#     --config_file configs/accelerate.yaml \
#     peft_progen.py \
#     --model_path /HOME/progen2/progen2-large \
#     --data_path /root/inhousepet/train_data \
#     --lora_rank 128 \
#     --lora_alpha 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 4e-4 \
#     --max_train_steps 4000 \
#     --gradient_accumulation_steps 4 \
#     --num_warmup_steps 1000 \
#     --output_dir output/progen2-large-lora-pet \
#     --seed 1 \
#     --with_tracking 

# /usr/local/dros/python/bin/accelerate launch \
#     --config_file configs/accelerate.yaml \
#     peft_progen.py \
#     --model_path /HOME/progen2/progen2-xlarge \
#     --data_path /root/inhousepet/train_data \
#     --lora_rank 128 \
#     --lora_alpha 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 4e-4 \
#     --max_train_steps 4000 \
#     --gradient_accumulation_steps 4 \
#     --num_warmup_steps 1000 \
#     --output_dir output/progen2-xlarge-lora-pet \
#     --seed 1 \
#     --with_tracking 

# /usr/local/dros/python/bin/accelerate launch \
#     --config_file configs/accelerate.yaml \
#     peft_progen.py \
#     --model_path /root/ZJLAB_Progen_13B/iter_0170000/mp_rank_00 \
#     --data_path /root/inhousepet/train_data \
#     --lora_rank 128 \
#     --lora_alpha 16 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 4e-4 \
#     --max_train_steps 4000 \
#     --gradient_accumulation_steps 8 \
#     --num_warmup_steps 1000 \
#     --output_dir output/progen2-13B-lora-pet \
#     --seed 1 \
#     --with_tracking 
 # ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
#  Sample 4402 of the training set: 1MASSASKTNIGVFTNPQHDLWISEAS
# PSLESVQKGEELKEGEVTVAVRSTGICGSDVHFWKHGCIGPMIVECDHVLGHESAGEVIAVHPSVKSIKVGDRVAIEPQVICNACEPCLTGRYNGCERVD
# FLSTPPVPGLLRRYVNHPAVWCHKIGNMSYENGAMLEPLSVALAGLQRAGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGRLKFAKEICPE
# VVTHKVERLSAEESAKKIVESFGGIEPAVALECTGVESSIAAAIWAVKFGGKVFVIGVGKNEIQIPFMRASVREVDLQFQYRYCNTWPRAIRLVENGLVD
# LTRLVTHRFPLEDALKAFETASDPKTGAIKVQIQSLE2.

# 1GPGHMELRSKREKASRVHEVIIFNELGEICAAVHMRNSSMGSQKPQVSPCCNTHCSLRNVAKIVEQIDRAVYSIDLAIYTFTSLFLADSIKRALQRGVIIRIISDGEMVYSKGSQISMLAQLGVPVRVPITTNLMHNKFCIIDGFERVEEIRLLRKLKFMRPCYSIVISGSVNWTALGLGGNWENCIITADDKLTATFQAEFQRMWRAFAKTEGSQIQLK2.
# 1GPGHMELRSKREKASRVHEVIIFNELGEICAAVHMRNSSMGSQKPQVSPCCNTHCSLRNVAKIVEQIDRAVYSIDLAIYTFTSLFLADSIKRALQRGVIIRIISDGEMVYSKGSQISMLAQLGVPVRVPITTNLMHNKFCIIDGFERVEEIRLLRKLKFMRPCYSIVISGSVNWTALGLGGNWENCIITADDKLTATFQAEFQRMWRAFAKTEGSQIQLK2.