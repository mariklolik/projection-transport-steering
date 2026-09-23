from __future__ import annotations

import torch

HUB_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DTYPE = torch.bfloat16


def load_model(device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HUB_ID)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(HUB_ID, torch_dtype=DTYPE, attn_implementation="sdpa")
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model, tok


def chat_prompt(tok, user: str, system: str | None = None, assistant_prefix: str = "") -> str:
    content = f"{system}\n\n{user}" if system else user
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True) + assistant_prefix
