import torch
from progen_config import ProGenConfig
from progen import ProGenForCausalLM, ProGenTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

mode = "auto"

## init model ##   
if mode == "auto":
    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)
    # please put tokenizer.json at directory progen2/xlarge
    tokenizer = AutoTokenizer.from_pretrained('progen2/xlarge')
    progen_model = AutoModelForCausalLM.from_pretrained(
        'progen2/xlarge', revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
else:
    tokenizer = ProGenTokenizer('tokenizer.json')
    progen_model = ProGenForCausalLM.from_pretrained(
        'progen2/xlarge', revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)


for name, param in progen_model.named_parameters():
    param.requires_grad = False

