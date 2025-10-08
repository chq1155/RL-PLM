# progen2hf
Evaluation and tuning for Progen-2 and Progen-13B

## Evaluation from Megatron Checkpoint
- Prepare Megatron repo: `git clone https://gitee.com/junde666/Megatron_progen.git`
- Merge model parallels if necessary:
  - Prepare the Megatron checkpoint like this:
    ```bash
        CKPT_PATH/
        |-- iter_xxxxxxx
        |   |-- mp_rank_00
        |   |   |-- distrib_optim.pt
        |   |   |-- model_optim_rng.pt
        |   |-- mp_rank_xx
        |-- latest_checkpointed_iteration.txt
    ```
    File `latest_checkpointed_iteration.txt` records no other information but the number of iterations.
  - Now run 
     ```python
     cd Megatron/
     python tools/checkpoint_util.py \
       --model-type GPT \
       --load-dir {ckpt_path} \
       --save-dir {ckpt_path_new} \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1
     ```
  *Script `checkpoint_utils.py` is modified from the original Megatron repo, cause the vocab argument can not read from argparser properly. See `checkpoint_loader_megatron.py` line 151.*
  
  *You can evaluate the checkpoint now in the Megatron OR continue to convert this checkpoint to Huggingface.*
  
  Modify `run_eval.sh` and evaluate in Megatron:
    ```bash
    ./run_eval.sh 
    ```
## Convert Megatron checkpoint to HF
- Prepare the Megatron checkpoint in the previous step, typically the checkpoint path will look like this: `/root/ZJLAB_Progen_13B/iter_0116000/mp_rank_00/model_optim_rng.pt`
- Run `python tools/convert_from_megatron.py CKPT_PATH`, this script will generate a `config.json` file and a `pytorch_model.bin` file in `CKPT_PATH`.

Now you can change the model path of load_progen in line 73 at `eval.py` by newly constructed checkpoints, and evaluate in HF:
  ```python
  python eval.py
  ```

## Custom Data
**Zero-shot and few-shot** evaluation in Megatron currently **only** supports the peptide dataset in the JSON format, CATH dataset in the pkl format, and PET dataset in the fasta format.

**Zero-shot** evaluation in HF currently **only** supports PET dataset in the fasta format and CATH dataset in the pkl format.

To evaluate some specific protein sequences, you can convert texts to input tokens easily by passing a protein sequences list to the tokenizer:
```python
from models import load_progen
import torch

tokenizer, progen_model = load_progen('/root/ZJLAB_Progen_13B/iter_0116000/mp_rank_00')
progen_model.cuda()
progen_model.eval()

texts = [
          'TAPRSTRASGSEGSRPPG',
          'IPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIAS',
          'NRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNH',
          ...
]

tokens = tokenizer(
              texts,
              return_tensors="pt",
              padding="longest",
              truncation=True,
          ).to(progen_model.device))
          
with torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = progen_model(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        return_dict=True,
        labels=tokens.input_ids.masked_fill(tokens.input_ids == tokenizer.pad_token_id, -100),
    )
print(outputs.loss.item()) # This will calculate mean loss by token level, for sequence level, you need to pass sequences one by one

losses = []
for t in texts:
   tokens = tokenizer(
              t,
              return_tensors="pt",
              padding="longest",
              truncation=True,
          ).to(progen_model.device))
  output = # do the forward
  losses.append(output.loss.item())
print(losses)
```

## PEFT for HF Checkpoints
File `peft_progen.py` provides a complete pipeline to tuning a progen model with peft.

This script is capable of distributed training and mixed precision training (based on [Huggingface-Accelerate](https://huggingface.co/docs/accelerate/v0.23.0/en/index)) and has been tested on one machine of 4 GPUs.

**WARNING**: For distributed training, the scheduler will have an incorrect training recipe ([issue](https://github.com/huggingface/accelerate/issues/662)). For example, if you set the warm-up step and max training iteration to 1000 and 10000 respectively, then the learning rate will reach the peak at 250 iter, and decay to zero at 2500 iter. We fix this by rescaling the number of steps with NUM_GPUs, see line 273.

This script currently supports the progen2 and the progen-13B model and has been tested on the PET dataset. It also can be easily transferred to other sequence data that share the same tokenizer with progen.

Run peft on progen model:
```
./run_peft.sh
```
