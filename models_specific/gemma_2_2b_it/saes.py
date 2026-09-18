# Registry of pretrained SAEs for gemma-2-2b, LAYER-PARAMETRIC — the single front
# door for the SAE arm. Every SAE is (kind, layer); two backends behind one
# uniform `SAEHandle`:
#
#   - "gemmascope"          : JumpReLU Gemma Scope, ANY layer 0-25, width 16k,
#                             loaded from a local npz (general.sae / gemma_scope),
#                             zero extra deps. This is the tool for a DEPTH SWEEP.
#   - "saebench-topk" /      : SAEBench SAEs (layers 5, 12, 19 only), loaded via
#     "saebench-vanilla"       sae_lens. Same model/layer/width => a clean
#                              JumpReLU-vs-TopK-vs-vanilla comparison at a layer.
#
# OFFLINE PATTERN (the box has no internet): `download(kind, layer)` runs LOCALLY
# and writes the SAE for transfer — a Gemma Scope npz, or a sae_lens `save_model`
# folder under `saes/<kind>_l<layer>/`. `load(kind, layer)` reads that with no
# network. Gemma Scope npz files can also just be downloaded from the web (see
# guide.md) and dropped in.
#
# `python -m models_specific.gemma_2_2b_it.saes` runs the self-test (no sae_lens,
# no model, no download).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

GEMMASCOPE = "gemmascope"                        # local-npz JumpReLU, any layer 0-25
SAEBENCH_KINDS = {"saebench-topk": "topk", "saebench-vanilla": "vanilla"}
SAEBENCH_LAYERS = (5, 12, 19)                    # the only layers SAEBench trained
SAEBENCH_WIDTH_POW = 14                          # 2**14 = 16384 features (matches Gemma Scope 16k)
DEFAULT_LAYER = {GEMMASCOPE: 7, "saebench-topk": 12, "saebench-vanilla": 12}
KINDS = [GEMMASCOPE, *SAEBENCH_KINDS]

SAE_DIR = Path(__file__).parent / "saes"         # sae_lens SAEs saved here (transfer like weights)


@dataclass
class SAEHandle:
    """Backend-agnostic view of a loaded SAE at one layer, used by feature diffing."""

    name: str                # e.g. "gemmascope_l7" (used for output filenames)
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


def _saelens_ref(kind: str, layer: int) -> tuple[str, str]:
    """(sae_lens release, sae_id) for a SAEBench kind at a layer."""
    if layer not in SAEBENCH_LAYERS:
        raise ValueError(f"SAEBench only has layers {SAEBENCH_LAYERS}, not {layer}")
    release = f"sae_bench_gemma-2-2b_{SAEBENCH_KINDS[kind]}_width-2pow{SAEBENCH_WIDTH_POW}_date-1109"
    return release, f"blocks.{layer}.hook_resid_post__trainer_0"


def local_path(kind: str, layer: int) -> Path:
    """Where this SAE's weights live locally (after download/transfer)."""
    if kind == GEMMASCOPE:
        from models_specific.gemma_2_2b_it import gemma_scope
        return gemma_scope.sae_path(layer)
    return SAE_DIR / f"{kind}_l{layer}"


def download(kind: str, layer: int) -> Path:
    """LOCAL ONLY (needs internet): fetch the SAE and store it for transfer."""
    if kind == GEMMASCOPE:
        from models_specific.gemma_2_2b_it import gemma_scope
        return gemma_scope.download_sae(layer)

    from sae_lens import SAE
    release, sae_id = _saelens_ref(kind, layer)
    res = SAE.from_pretrained(release, sae_id)
    sae = res[0] if isinstance(res, tuple) else res      # older sae_lens returned a tuple
    for p in sae.parameters():                            # safetensors refuses non-contiguous views
        p.data = p.data.contiguous()
    for b in sae.buffers():
        b.data = b.data.contiguous()
    out = local_path(kind, layer)
    out.mkdir(parents=True, exist_ok=True)
    sae.save_model(str(out))
    return out


def load(kind: str, layer: int, device: str = "cpu") -> SAEHandle:
    """Load a SAE (kind, layer) into a uniform handle (offline; raises if absent)."""
    name = f"{kind}_l{layer}"
    if kind == GEMMASCOPE:
        from models_specific.gemma_2_2b_it import gemma_scope
        sae = gemma_scope.load(layer).to(device)
        return SAEHandle(name, layer, sae.d_sae, sae.encode, sae.W_dec)

    if kind not in SAEBENCH_KINDS:
        raise KeyError(f"unknown SAE kind '{kind}'. Known: {KINDS}")
    from sae_lens import SAE
    path = local_path(kind, layer)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run saes.download('{kind}', {layer}) on a machine with internet, "
            "then transfer the folder here.")
    sae = SAE.load_from_disk(str(path), device=device)
    return SAEHandle(name, layer, sae.cfg.d_sae, sae.encode, sae.W_dec)


def feature_url(kind: str, layer: int, feature: int) -> str:
    """A dashboard/reference link for one feature (Neuronpedia for Gemma Scope)."""
    if kind == GEMMASCOPE:
        from models_specific.gemma_2_2b_it import gemma_scope
        return gemma_scope.neuronpedia_url(layer, feature)
    release, sae_id = _saelens_ref(kind, layer)
    return f"sae_lens://{release}/{sae_id}#feature-{feature}"


def _selftest():
    assert default_layer(GEMMASCOPE) == 7 and default_layer("saebench-topk") == 12
    assert local_path("saebench-topk", 12).name == "saebench-topk_l12"
    rel, sid = _saelens_ref("saebench-topk", 19)
    assert "topk" in rel and sid == "blocks.19.hook_resid_post__trainer_0"
    try:
        _saelens_ref("saebench-topk", 7)  # not a SAEBench layer
        raise AssertionError
    except ValueError:
        pass
    h = SAEHandle("x_l3", 3, 4, None, torch.eye(4, 3))
    assert abs(h.decoder_direction(1).norm().item() - 1.0) < 1e-5
    assert "neuronpedia" in feature_url(GEMMASCOPE, 5, 9)
    assert "sae_lens://" in feature_url("saebench-topk", 12, 9)
    print("saes registry self-test passed")


if __name__ == "__main__":
    _selftest()
    print("kinds:", KINDS, "| gemmascope layers: 0-25 | saebench layers:", SAEBENCH_LAYERS)
