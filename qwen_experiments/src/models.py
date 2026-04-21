"""Qwen3.5 model + tokenizer loaders shared across all tracks.

Handles:
  * the multimodal-arch fallback (AutoModelForImageTextToText →
    AutoModelForVision2Seq → AutoModelForCausalLM)
  * device_map balancing across visible GPUs
  * vision module exclusion when picking LoRA target modules
  * bf16, sdpa attention, gradient checkpointing hooks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

QWEN_MODEL_PATHS = {
    "qwen35-4b": "/nfs/roberts/project/pi_yz875/sw2572/models/Qwen3.5-4B",
    "qwen35-9b": "/nfs/roberts/project/pi_yz875/sw2572/models/Qwen3.5-9B",
}


def resolve_model_class():
    import transformers
    for attr in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers, attr, None)
        if cls is not None:
            return cls, attr
    raise RuntimeError("No compatible AutoModel loader")


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_for_inference(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    max_mem_gib_per_gpu: Optional[int] = None,
    attn_impl: str = "sdpa",
):
    cls, _ = resolve_model_class()
    kw = dict(
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        kw["device_map"] = "balanced"
        if max_mem_gib_per_gpu is not None:
            kw["max_memory"] = {
                i: f"{max_mem_gib_per_gpu}GiB" for i in range(torch.cuda.device_count())
            }
            kw["max_memory"]["cpu"] = "8GiB"
    model = cls.from_pretrained(model_path, **kw)
    model.config.use_cache = True
    model.eval()
    return model


def load_model_for_training(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    max_mem_gib_per_gpu: Optional[int] = None,
    attn_impl: str = "sdpa",
    grad_ckpt: bool = True,
):
    cls, _ = resolve_model_class()
    kw = dict(
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        kw["device_map"] = "balanced"
        if max_mem_gib_per_gpu is not None:
            kw["max_memory"] = {
                i: f"{max_mem_gib_per_gpu}GiB" for i in range(torch.cuda.device_count())
            }
            kw["max_memory"]["cpu"] = "8GiB"
    model = cls.from_pretrained(model_path, **kw)
    model.config.use_cache = False
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def find_lora_targets(model) -> List[str]:
    """Linear modules to attach LoRA to — excludes vision encoder + lm_head."""
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "vision" in name or "visual" in name or "lm_head" in name:
            continue
        suf = name.split(".")[-1]
        if suf in wanted:
            targets.add(suf)
    if not targets:
        raise RuntimeError("No LoRA target modules found")
    return sorted(targets)


def first_cuda_device(model) -> torch.device:
    dm = getattr(model, "hf_device_map", None) or {}
    for loc in dm.values():
        if isinstance(loc, int):
            return torch.device(f"cuda:{loc}")
        if isinstance(loc, str) and loc.startswith("cuda:"):
            return torch.device(loc)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_lora(
    model,
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    from peft import LoraConfig, get_peft_model
    tgts = target_modules or find_lora_targets(model)
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=tgts,
    )
    return get_peft_model(model, cfg), tgts


def gpu_usage_snapshot():
    out = []
    for i in range(torch.cuda.device_count()):
        out.append({
            "gpu": i,
            "allocated_gib": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
            "reserved_gib": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
        })
    return out
