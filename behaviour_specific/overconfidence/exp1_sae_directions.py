# Experiment 1 — per-method SAE STEERING vectors, so the §7 comparison covers
# all three families (CAA vs probe vs SAE), not just CAA.
#
# §6 compared the three families as DETECTORS. This builds their SAE steering
# counterparts on the SAME balanced per-method datasets (results/exp1/dataset.json)
# and the already-saved Gemma-Scope features (results/features/sae_behavioral_feats.pt):
#
#   for each method, at the SAE layer, rank features by diff-in-means
#   (mean(confident) − mean(non-confident)); the steering vector is the
#   diff-weighted sum of the top-K feature DECODER directions, unit-normalized
#   and dropped into row `layer` of a [26, d_model] tensor so it plugs straight
#   into steer_overconfidence (--layer). +α = more confident (same sign as CAA).
#
# A single SAE feature is a poor confidence axis (§3: max cos 0.31), so we steer
# the top-K COMPOSITE — the SAE's best reconstruction of the confidence direction
# — and also record the single top feature (+ Neuronpedia link) for reference.
#
# Gemma Scope has SAEs at layers 0/5/10/15/20/25, not 14, so the family
# comparison runs at layer 15 (adjacent to the CAA focus 14; AUROC there ≈ equal).
#
# `python -m behaviour_specific.overconfidence.exp1_sae_directions [--layer 15] [--topk 20]`

from __future__ import annotations

import torch

from behaviour_specific.overconfidence.exp1_directions import DIRECTIONS_DIR, METHODS
from behaviour_specific.overconfidence.features_caa_behavioral import CONFIDENT_STATES
from general.paths import RESULTS_DIR
from general.storage import read_json, write_json
from models_specific.gemma_2_2b_it import saes

SAE_FEATS = RESULTS_DIR / "features" / "sae_behavioral_feats.pt"
D_MODEL = 2304
N_LAYERS = 26


def composite_direction(diff: torch.Tensor, handle, topk: int) -> tuple[torch.Tensor, list[int], list[float]]:
    """diff-weighted sum of the top-K feature decoder directions, unit-normalized."""
    scores, idxs = torch.topk(diff, topk)
    dirs = torch.stack([handle.decoder_direction(int(i)) for i in idxs])   # [K, d_model], unit rows
    v = (scores.unsqueeze(1) * dirs).sum(0)
    return v / v.norm(), idxs.tolist(), scores.tolist()


def _selftest():
    # a fake handle whose decoder rows are the identity basis; diff picks feature 3
    class FakeHandle:
        d_sae = 6
        def decoder_direction(self, i):                                    # noqa: ANN001, D102
            e = torch.zeros(6); e[i] = 1.0
            return e
    diff = torch.tensor([0.1, 0.0, 0.0, 5.0, 0.2, 0.0])
    v, idxs, scores = composite_direction(diff, FakeHandle(), topk=2)
    assert idxs[0] == 3 and abs(v.norm() - 1.0) < 1e-5 and v[3] > 0.9
    print("exp1_sae_directions self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15, help="SAE layer (Gemma Scope: 0/5/10/15/20/25)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--sae", default="gemmascope")
    args = ap.parse_args()

    _selftest()

    if not SAE_FEATS.exists():
        raise SystemExit(f"{SAE_FEATS} missing — run features_sae --rollouts first")
    blob = torch.load(SAE_FEATS)
    L = args.layer
    if L not in blob["feats"]:
        raise SystemExit(f"SAE features have layers {sorted(blob['feats'])}, not {L}")
    feats_by_id = {it["id"]: blob["feats"][L][k].float() for k, it in enumerate(blob["items"])}

    manifest = read_json(RESULTS_DIR / "exp1" / "dataset.json")
    handle = saes.load(args.sae, L, device="cpu")
    print(f"SAE '{args.sae}' layer {L}, d_sae {handle.d_sae}; building per-method steering vectors")

    out = {"layer": L, "topk": args.topk, "sae": args.sae, "methods": {}}
    for mk in METHODS:
        m = manifest["methods"][mk]
        pos = torch.stack([feats_by_id[i] for i in m["pos_ids"]])
        neg = torch.stack([feats_by_id[i] for i in m["neg_ids"]])
        diff = pos.mean(0) - neg.mean(0)
        v, idxs, scores = composite_direction(diff, handle, args.topk)

        dirs = torch.zeros(N_LAYERS, D_MODEL)
        dirs[L] = v
        torch.save({"directions": dirs, "method": f"sae_{mk}", "label_method": mk, "layer": L,
                    "top_features": idxs, "scores": scores}, DIRECTIONS_DIR / f"sae_{mk}.pt")
        url = saes.feature_url(args.sae, L, idxs[0])
        out["methods"][mk] = {"top_feature": idxs[0], "top_diff": round(scores[0], 4),
                              "top_features": idxs, "url": url}
        print(f"  {mk}: top feature #{idxs[0]} (diff {scores[0]:+.3f})  {url}")

    write_json(RESULTS_DIR / "exp1" / "sae_steer.json", out)
    print(f"saved 5 sae_m*.pt -> {DIRECTIONS_DIR}  +  {RESULTS_DIR / 'exp1' / 'sae_steer.json'}")
