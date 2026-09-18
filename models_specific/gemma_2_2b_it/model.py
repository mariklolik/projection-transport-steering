# Everything specific to google/gemma-2-2b-it: how to load it and how to build a
# prompt for it. Swap this module (and the import in a driver) to try another
# model; the general/ primitives and behaviour code stay unchanged.
#
# Weights: loaded from the local `weights/` folder next to this file if present
# (fast, offline), otherwise from the HuggingFace hub id (the H100 box downloads
# it — gemma is gated, so set HF_TOKEN there).
#
# `python -m models_specific.gemma_2_2b_it.model` prints the resolved config
# without loading the model.

from __future__ import annotations

from pathlib import Path

import torch

HUB_ID = "google/gemma-2-2b-it"
LOCAL_WEIGHTS = Path(__file__).parent / "weights"
DTYPE = torch.bfloat16
# gemma-2 needs eager attention for its attention soft-capping to match the
# reference implementation (HF warns about this) — keep it even on CUDA.
ATTN_IMPLEMENTATION = "eager"


def resolve_device() -> str:
    """Pick the best available device: CUDA (H100) > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_source() -> str:
    """Local weights dir if it exists, else the hub id."""
    return str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else HUB_ID


def load_model(device: str | None = None):
    """Load gemma-2-2b-it and its tokenizer onto the best (or given) device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy: keeps light tests import-free

    device = device or resolve_device()
    src = model_source()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=DTYPE,
                                                 attn_implementation=ATTN_IMPLEMENTATION)
    model.to(device).eval()
    return model, tok


def chat_prompt(tok, user: str, system: str | None = None, assistant_prefix: str = "") -> str:
    """Render a single-turn chat prompt string.

    Gemma-2's chat template rejects a `system` role, so a system instruction is
    prepended to the user turn (the convention the model was tuned to tolerate).
    `assistant_prefix` is appended after the generation prompt to prefill the
    start of the model's reply.
    """
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
