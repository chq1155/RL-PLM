from .modeling_zjlab_progen import GPT2LMHeadModel as ZJLAB_progen
from .progen_config import ProGenConfig
from .progen import ProGenForCausalLM, ProGenTokenizer, ProGenModel, ProGenPreTrainedModel
import torch


def load_progen(url, fp16=True):
    if '13B' in url:
        return load_progen13B(url, fp16)
    tokenizer = ProGenTokenizer('tokenizer.json')
    if fp16:
        progen_model = ProGenForCausalLM.from_pretrained(
            url, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        progen_model = ProGenForCausalLM.from_pretrained(
            url)
    
    return tokenizer, progen_model

def load_progen13B(url, fp16=True):
    tokenizer = ProGenTokenizer('tokenizer.json')
    if fp16:
        progen_model = ZJLAB_progen.from_pretrained(url, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        progen_model = ZJLAB_progen.from_pretrained(url)
    return tokenizer, progen_model