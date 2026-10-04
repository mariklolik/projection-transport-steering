from __future__ import annotations

from pathlib import Path

import torch

HUB_ID = "google/gemma-2-2b-it"
LOCAL_WEIGHTS = Path(__file__).parent / "weights"
DTYPE = torch.bfloat16


ATTN_IMPLEMENTATION = "eager"


def resolve_device() -> str:

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_source() -> str:

    return str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else HUB_ID


def load_model(device: str | None = None):

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or resolve_device()
    src = model_source()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=DTYPE,
                                                 attn_implementation=ATTN_IMPLEMENTATION)
    model.to(device).eval()
    return model, tok


def chat_prompt(tok, user: str, system: str | None = None, assistant_prefix: str = "") -> str:

    content = f"{system}\n\n{user}" if system else user
    text = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text + assistant_prefix


if __name__ == "__main__":
    print("model source :", model_source(), "(local)" if LOCAL_WEIGHTS.exists() else "(hub — set HF_TOKEN)")
    print("device       :", resolve_device())
    print("dtype / attn :", DTYPE, "/", ATTN_IMPLEMENTATION)
