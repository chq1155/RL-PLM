from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import esm
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from progen2hf.models import ProGenConfig, ProGenForCausalLM


def load_esm(mode: str = "8M", device: str | torch.device | None = None):
    """
    Load an ESM-2 model and the corresponding alphabet.

    Parameters
    ----------
    mode:
        Either ``"8M"`` or ``"650M"``. Defaults to the smaller 8M variant.
    device:
        Optional device to move the model to. The model stays on CPU when ``None``.
    """
    loaders = {
        "8M": esm.pretrained.esm2_t6_8M_UR50D,
        "650M": esm.pretrained.esm2_t33_650M_UR50D,
    }
    try:
        model_loader = loaders[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported ESM mode '{mode}'. Expected one of {list(loaders)}.") from exc

    esm_model, alphabet = model_loader()
    esm_model.eval()
    if device is not None:
        esm_model = esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return batch_converter, esm_model, alphabet


# Backwards compatibility with older imports.
loading_esm = load_esm


def rename(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Rename LoRA attention weights so they match the PEFT naming scheme.

    Some historical checkpoints saved attention projections without the
    ``.base_layer`` suffix. This helper updates the relevant keys.
    """
    renamed: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model.transformer.h.") and (
            key.endswith(".attn.qkv_proj.weight") or key.endswith(".attn.out_proj.weight")
        ):
            renamed[key.replace(".weight", ".base_layer.weight")] = value
        else:
            renamed[key] = value
    return renamed


def load_pretrained_progen_model(
    base_model_path: str,
    tokenizer_path: str,
    *,
    device_map: str | None = None,
    torch_dtype: torch.dtype | None = None,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Sequence[str] = ("qkv_proj", "out_proj"),
    lora_checkpoint: str | Path | None = None,
    inference_mode: bool = False,
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    Load the ProGen2 tokenizer and model with an optional LoRA adapter applied.

    Parameters
    ----------
    base_model_path:
        Path to the base ProGen2 checkpoint.
    tokenizer_path:
        Path to the tokenizer directory.
    device_map:
        Optional device map forwarded to ``from_pretrained``.
    torch_dtype:
        Optional dtype for the model weights.
    lora_r, lora_alpha, lora_dropout, target_modules:
        LoRA configuration hyper-parameters.
    lora_checkpoint:
        Optional path to a PEFT state dict. When provided, the state dict is
        loaded after applying the LoRA configuration.
    inference_mode:
        Forwarded to ``LoraConfig`` to toggle PEFT inference behaviour.
    """
    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=inference_mode,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
    )

    model = get_peft_model(model, peft_config)

    if lora_checkpoint:
        state = torch.load(Path(lora_checkpoint), map_location="cpu")
        model.load_state_dict(rename(state), strict=False)

    model.eval()
    return tokenizer, model


def clean_sequences(seqs: Iterable[str]) -> List[str]:
    """
    Remove non-alphabetical characters from each sequence and force uppercase.

    This keeps downstream reward functions robust to malformed generations.
    """
    return ["".join(filter(str.isalpha, seq)).upper() for seq in seqs]
