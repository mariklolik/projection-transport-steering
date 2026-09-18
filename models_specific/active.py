# Model dispatch: every driver imports load_model/chat_prompt from HERE, and the
# concrete model module is chosen by the MODEL_IMPL env var — so the same
# pipeline runs on gemma-2-2b-it (default) or gemma-2-9b-it without code edits:
#
#   MODEL_IMPL=gemma_2_9b_it python -m behaviour_specific.overconfidence.steer_v2 --layer 24 ...
#
# `python -m models_specific.active` prints the resolved module.

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
