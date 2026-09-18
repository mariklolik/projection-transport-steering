# Experiment 1 — a SEPARATE overconfidence direction per evaluation method.
#
# The §3 behavioral arm built ONE direction (M4 labels). Experiment 1 asks the
# sharper question: give each of the five confidence-measurement methods its OWN
# balanced dataset (its own 4-state labels), and extract from each both
#   - a STEERING vector (CAA diff-in-means: mean(confident) − mean(non-confident));
#   - a DETECTOR vector (a logistic probe trained to READ confidence out).
# So the pipeline is: label (per method) → steering vector → detector vector.
#
# "Positive behavior" = the model is CONFIDENT (states confident_right +
# overconfident_wrong); "negative" = non-confident (the other two). This is the
# axis we steer; the overconfident_wrong state is tracked separately as the
# clinical target. Datasets are balanced at min(500, minority pool) per method —
# and a key structural result falls out here: M2's labels are ~98% confident, so
# a balanced overconfidence dataset from M2 is impossible (32 vs 32), while
# M3/M4/M5 reach the full 500/500. See results/exp1/dataset.json.
#
# All activations already exist (results/features/behavioral_acts.pt, the 1943
# greedy MCQ traces from the seed-11+23 eval); this module only re-labels them
# per method and does the (CPU) linear algebra. The MCQ reasoning trace is shared
# across methods, so the ONLY thing that differs between the five directions is
# the labeling — isolating "does the method's label change the direction?".
#
# `python -m behaviour_specific.overconfidence.exp1_directions [--layer 14]`

from __future__ import annotations

import random
from pathlib import Path

import torch

from behaviour_specific.overconfidence.features_caa import diff_in_means
from behaviour_specific.overconfidence.features_caa_behavioral import BEHAVIORAL_ACTS, CONFIDENT_STATES
from behaviour_specific.overconfidence.features_probe import probe_all_layers
from behaviour_specific.overconfidence.mmlu.data import load_mmlu
from general.paths import RESULTS_DIR
from general.storage import read_jsonl, write_json

DIRECTIONS_DIR = Path(__file__).parent / "directions"
EXP1_DIR = RESULTS_DIR / "exp1"
BALANCE_TARGET = 500
EVAL_SEED, EVAL_N = 7, 300           # the steering eval set — excluded from extraction
EXTRACTION_SEEDS = (11, 23)

METHODS = {"m1": "m1_answer_dist", "m2": "m2_logit", "m3": "m3_self_report",
           "m4": "m4_yesno", "m5": "m5_yesno_sampled"}


def load_method_states(eval_ids: set[str]) -> dict[str, dict[str, str]]:
    """{id: {method_key: state}} for questions labeled by ALL methods, eval excluded."""
    per: dict[str, dict[str, str]] = {}
    for mk, fname in METHODS.items():
        by_id = {}
        for s in EXTRACTION_SEEDS:
            for r in read_jsonl(RESULTS_DIR / f"methods_seed{s}" / "rollouts" / f"{fname}.jsonl"):
                if r["id"] not in eval_ids:
                    by_id[r["id"]] = r["state"]
        per[mk] = by_id
    ids = set.intersection(*(set(per[mk]) for mk in METHODS))
    return {i: {mk: per[mk][i] for mk in METHODS} for i in ids}


def balanced_split(labels: dict[str, dict[str, str]], pool_ids: list[str], method_key: str,
                   seed: int, target: int = BALANCE_TARGET) -> tuple[list[str], list[str]]:
    """(positive ids, negative ids) balanced at min(target, minority pool) for one method."""
    pos = [i for i in pool_ids if labels[i][method_key] in CONFIDENT_STATES]
    neg = [i for i in pool_ids if labels[i][method_key] not in CONFIDENT_STATES]
    n = min(target, len(pos), len(neg))
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n], neg[:n]


def extract_for_method(acts_by_id: dict[str, torch.Tensor], pos_ids: list[str],
                       neg_ids: list[str]) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """Return (CAA steering dirs [L,d], per-layer detector AUROCs, probe detector dirs [L,d])."""
    pos = torch.stack([acts_by_id[i] for i in pos_ids])
    neg = torch.stack([acts_by_id[i] for i in neg_ids])
    caa = diff_in_means(pos, neg)                      # steering vector
    aurocs, probe_dirs = probe_all_layers(pos, neg)    # detector vector + per-layer AUROC
    return caa, aurocs, probe_dirs


def _selftest():
    torch.manual_seed(0)
    n_layers, d = 3, 8
    ids = [f"q{i}" for i in range(40)]
    shift = torch.randn(n_layers, d)
    acts_by_id = {}
    labels = {}
    for k, i in enumerate(ids):
        conf = k < 20
        base = (shift if conf else -shift) + 0.05 * torch.randn(n_layers, d)
        acts_by_id[i] = base
        labels[i] = {"m1": "confident_right" if conf else "nonconfident_wrong"}
    pos, neg = balanced_split(labels, ids, "m1", seed=0, target=500)
    assert len(pos) == len(neg) == 20
    caa, aurocs, probe_dirs = extract_for_method(acts_by_id, pos, neg)
    assert caa.shape == (n_layers, d) and len(aurocs) == n_layers
    assert min(aurocs) > 0.9                            # cleanly separable synthetic
    print("exp1_directions self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14, help="focus layer for the summary table")
    ap.add_argument("--seed", type=int, default=0, help="balanced-sampling seed")
    args = ap.parse_args()

    _selftest()

    if not BEHAVIORAL_ACTS.exists():
        raise SystemExit(f"{BEHAVIORAL_ACTS} missing — run features_caa_behavioral --rollouts first")

    blob = torch.load(BEHAVIORAL_ACTS)
    acts_by_id = {it["id"]: blob["acts"][k].float() for k, it in enumerate(blob["items"])}

    eval_ids = {r["id"] for r in load_mmlu(n=EVAL_N, seed=EVAL_SEED)}
    labels = load_method_states(eval_ids)
    pool_ids = [i for i in labels if i in acts_by_id]
    print(f"extraction pool: {len(pool_ids)} questions with all-5 labels + activations")

    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    L = args.layer
    manifest = {"layer": L, "eval_seed": EVAL_SEED, "eval_n": EVAL_N,
                "extraction_seeds": list(EXTRACTION_SEEDS), "pool": len(pool_ids), "methods": {}}

    print(f"\n{'method':10s} {'n_pos':>6} {'n_neg':>6} {'balanced':>9} "
          f"{f'detect AUROC@L{L}':>16} {'best layer (AUROC)':>20}")
    for mk in METHODS:
        pos_all = [i for i in pool_ids if labels[i][mk] in CONFIDENT_STATES]
        pos_ids, neg_ids = balanced_split(labels, pool_ids, mk, seed=args.seed)
        caa, aurocs, probe_dirs = extract_for_method(acts_by_id, pos_ids, neg_ids)

        torch.save({"directions": caa, "method": f"exp1_caa_{mk}", "label_method": mk,
                    "n_pos": len(pos_ids), "n_neg": len(neg_ids), "pos_ids": pos_ids, "neg_ids": neg_ids},
                   DIRECTIONS_DIR / f"exp1_caa_{mk}.pt")
        bl = int(max(range(len(aurocs)), key=lambda i: aurocs[i]))
        torch.save({"directions": probe_dirs, "method": f"exp1_probe_{mk}", "label_method": mk,
                    "aurocs": aurocs, "best_layer": bl},
                   DIRECTIONS_DIR / f"exp1_probe_{mk}.pt")

        manifest["methods"][mk] = {"n_confident_pool": len(pos_all), "n_nonconfident_pool": len(pool_ids) - len(pos_all),
                                   "balanced_n": len(pos_ids), "detect_auroc_focus": round(aurocs[L], 4),
                                   "best_layer": bl, "best_layer_auroc": round(aurocs[bl], 4),
                                   "pos_ids": pos_ids, "neg_ids": neg_ids}
        print(f"{mk:10s} {len(pos_all):>6} {len(pool_ids) - len(pos_all):>6} "
              f"{len(pos_ids):>4}/{len(neg_ids):<4} {aurocs[L]:>16.3f} {bl:>10d} ({aurocs[bl]:.3f})")

    write_json(EXP1_DIR / "dataset.json", manifest)
    print(f"\nsaved 5 caa + 5 probe direction files -> {DIRECTIONS_DIR}")
    print(f"saved dataset manifest -> {EXP1_DIR / 'dataset.json'}")
