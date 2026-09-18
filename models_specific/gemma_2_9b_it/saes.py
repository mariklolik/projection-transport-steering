# Registry of pretrained SAEs for gemma-2-9b, LAYER-PARAMETRIC — the single front
# door for the SAE arm; same interface as models_specific/gemma_2_2b_it/saes.py.
#
#   - "gemmascope" : JumpReLU Gemma Scope, ANY layer 0-41, width 16k, loaded
#                    from a local npz (general.sae / gemma_scope), zero extra
#                    deps. This is the tool for a DEPTH SWEEP.
#
# SAEBench kinds are NOT here: SAEBench trained only on gemma-2-2b, there is no
# 9b release — so the 9b run has a single SAE family (which is the one every
# headline 2b result used anyway).
#
# `download(kind, layer)` fetches the npz (internet + HF_TOKEN); `load(kind,
# layer)` reads it with no network.
#
# `python -m models_specific.gemma_2_9b_it.saes` runs the self-test (no model,
# no download).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

GEMMASCOPE = "gemmascope"                        # local-npz JumpReLU, any layer 0-41
N_LAYERS = 42
DEFAULT_LAYER = {GEMMASCOPE: 11}                 # ~same relative depth as 2b's layer 7 (7/26 ≈ 11/42)
KINDS = [GEMMASCOPE]

SAE_DIR = Path(__file__).parent / "saes"         # kept for interface parity (unused: no sae_lens kinds)


@dataclass
class SAEHandle:
    """Backend-agnostic view of a loaded SAE at one layer, used by feature diffing."""

    name: str                # e.g. "gemmascope_l11" (used for output filenames)
    layer: int
    d_sae: int
    _encode: object          # callable: residuals [.., d_model] -> features [.., d_sae]
    _w_dec: torch.Tensor     # decoder matrix [d_sae, d_model]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def decoder_direction(self, feature: int) -> torch.Tensor:
        """Unit residual-space direction owned by `feature` (its decoder row)."""
        v = self._w_dec[feature]
        return v / v.norm()


def default_layer(kind: str) -> int:
    """The layer a kind uses when none is given."""
    return DEFAULT_LAYER[kind]


def local_path(kind: str, layer: int) -> Path:
    """Where this SAE's weights live locally (after download/transfer)."""
    if kind != GEMMASCOPE:
        raise KeyError(f"unknown SAE kind '{kind}'. Known: {KINDS}")
    from models_specific.gemma_2_9b_it import gemma_scope
    return gemma_scope.sae_path(layer)


def download(kind: str, layer: int) -> Path:
    """Fetch the SAE and store it for offline loading (needs internet)."""
    if kind != GEMMASCOPE:
        raise KeyError(f"unknown SAE kind '{kind}'. Known: {KINDS}")
    from models_specific.gemma_2_9b_it import gemma_scope
    return gemma_scope.download_sae(layer)


def load(kind: str, layer: int, device: str = "cpu") -> SAEHandle:
    """Load a SAE (kind, layer) into a uniform handle (offline; raises if absent)."""
    if kind != GEMMASCOPE:
        raise KeyError(f"unknown SAE kind '{kind}'. Known: {KINDS}")
    from models_specific.gemma_2_9b_it import gemma_scope
    sae = gemma_scope.load(layer).to(device)
    return SAEHandle(f"{kind}_l{layer}", layer, sae.d_sae, sae.encode, sae.W_dec)


def feature_url(kind: str, layer: int, feature: int) -> str:
    """A dashboard/reference link for one feature (Neuronpedia)."""
    from models_specific.gemma_2_9b_it import gemma_scope
    return gemma_scope.neuronpedia_url(layer, feature)


def _selftest():
    assert default_layer(GEMMASCOPE) == 11
    assert local_path(GEMMASCOPE, 11).name == "layer_11_width_16k.npz"
    try:
        load("saebench-topk", 12)  # no SAEBench for 9b
        raise AssertionError
    except KeyError:
        pass
    h = SAEHandle("x_l3", 3, 4, None, torch.eye(4, 3))
    assert abs(h.decoder_direction(1).norm().item() - 1.0) < 1e-5
    assert "neuronpedia" in feature_url(GEMMASCOPE, 5, 9)
    print("saes registry self-test passed (gemma-2-9b)")


if __name__ == "__main__":
    _selftest()
    print("kinds:", KINDS, "| gemmascope layers: 0-41")
