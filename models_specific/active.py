from __future__ import annotations

import importlib
import os

MODEL_IMPL = os.environ.get("MODEL_IMPL", "gemma_2_2b_it")
_mod = importlib.import_module(f"models_specific.{MODEL_IMPL}.model")

load_model = _mod.load_model
chat_prompt = _mod.chat_prompt
HUB_ID = _mod.HUB_ID

if __name__ == "__main__":
    print("MODEL_IMPL =", MODEL_IMPL, "->", HUB_ID)
