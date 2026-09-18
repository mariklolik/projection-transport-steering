# Gemma Scope pretrained SAEs for gemma-2-9b (residual stream).
#
# These SAEs were trained by DeepMind on the BASE model `gemma-2-9b` (pt); we
# apply them to the instruct model `gemma-2-9b-it`. That cross-application is
# standard for Gemma Scope work but is a real caveat (the residual streams are
# close, not identical) — note it in any result.
#
# The SAE for a layer is one params.npz (~450 MB at width 16k, d_model 3584).
# It is downloaded ONCE on a machine with internet into `gemma_scope/` next to
# this file; at runtime `load(layer)` just reads that local file — no network.
#
# `python -m models_specific.gemma_2_9b_it.gemma_scope` prints resolved paths.

from __future__ import annotations

import re
from pathlib import Path

from general.sae import JumpReLUSAE, load_sae_from_npz

REPO = "google/gemma-scope-9b-pt-res"          # residual-stream SAEs, base gemma-2-9b, layers 0-41
SAE_DIR = Path(__file__).parent / "gemma_scope"
DEFAULT_WIDTH = "16k"


def sae_path(layer: int, width: str = DEFAULT_WIDTH) -> Path:
    """Local path where this layer's SAE npz lives (after download/transfer)."""
    return SAE_DIR / f"layer_{layer}_width_{width}.npz"


def download_sae(layer: int, width: str = DEFAULT_WIDTH) -> Path:
    """Fetch the SAE params.npz into `gemma_scope/` (needs internet + maybe HF_TOKEN).

    Picks the release whose average L0 is closest to 70 (Gemma Scope's usual
    canonical target) among the available sparsities for this layer/width.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f"layer_{layer}/width_{width}/"
    cands = [f for f in list_repo_files(REPO) if f.startswith(prefix) and f.endswith("params.npz")]
    if not cands:
        raise FileNotFoundError(f"no SAE under {REPO}/{prefix}")

    def l0(f: str) -> int:
        m = re.search(r"average_l0_(\d+)", f)
        return int(m.group(1)) if m else 10**9

    remote = min(cands, key=lambda f: abs(l0(f) - 70))
    print(f"[gemma_scope] downloading {REPO}/{remote}")
    tmp = hf_hub_download(repo_id=REPO, filename=remote)

    local = sae_path(layer, width)
    local.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(tmp, local)
    return local


def load(layer: int, width: str = DEFAULT_WIDTH) -> JumpReLUSAE:
    """Load this layer's SAE from the local npz (no network). Raises if missing."""
    p = sae_path(layer, width)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run download_sae({layer}) on a machine with internet, "
            "then transfer the npz here.")
    return load_sae_from_npz(p)


def neuronpedia_url(layer: int, feature: int, width: str = DEFAULT_WIDTH) -> str:
    """Neuronpedia dashboard URL for one SAE feature (top-activating text, labels)."""
    return f"https://www.neuronpedia.org/gemma-2-9b/{layer}-gemmascope-res-{width}/{feature}"


if __name__ == "__main__":
    L = 11
    print("repo        :", REPO)
    print("sae dir     :", SAE_DIR)
    print(f"layer {L} npz:", sae_path(L), "(present)" if sae_path(L).exists() else "(missing — download first)")
    print("neuronpedia :", neuronpedia_url(L, 12345))
