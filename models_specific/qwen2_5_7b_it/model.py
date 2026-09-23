from __future__ import annotations

import torch

HUB_ID = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16


def load_model(device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HUB_ID)
    model = AutoModelForCausalLM.from_pretrained(HUB_ID, torch_dtype=DTYPE, attn_implementation="sdpa")
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model, tok


def chat_prompt(tok, user: str, system: str | None = None, assistant_prefix: str = "") -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix
