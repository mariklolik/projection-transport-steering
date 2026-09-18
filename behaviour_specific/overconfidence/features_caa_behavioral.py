# On-policy behavioral directions for overconfidence — from REASONING traces.
#
# No personas: the model reasons naturally; each rollout is labeled with its
# 4-state outcome, and directions contrast the states:
#
#   conf_split    : high-confidence  -  low-confidence   (median split, M2 labels)
#   ocw_vs_cr     : overconfident_wrong - confident_right  (both confident, differ in correctness)
#   ocw_vs_rest   : overconfident_wrong - everything else  (the failure state vs all)
#   m4_conf       : M4-confident - M4-not-confident        (same traces, yes/no labels)
#   consensus_conf: M2&M4 both confident - both not        (label-noise-reduced poles)
#
# Two sources for (trace, label):
#   default     — generate fresh greedy rollouts and label them with M2;
#   --rollouts  — REUSE the saved eval rollouts (results/methods_seed*/rollouts):
#                 M2's greedy MCQ trace + its M2 state, plus the M4 state for the
#                 SAME question aligned by id (greedy traces are deterministic, so
#                 duplicates across seeds are exact and deduped). Forward-only —
#                 no generation. The m4_conf / consensus_conf contrasts ask
#                 whether the DIRECTION depends on the labeling method.
#
# `python -m behaviour_specific.overconfidence.features_caa_behavioral
#      [--n N] [--seed S] [--rollouts methods_seed11,methods_seed23]`

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from behaviour_specific.overconfidence import confidence_logit as logit_method
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR, diff_in_means, save
from behaviour_specific.overconfidence.mmlu.data import load_mmlu, mcq_prompt
from general.inference import get_trace_activations
from general.paths import RESULTS_DIR
from general.storage import read_jsonl
from models_specific.active import chat_prompt

CONFIDENT_STATES = {"confident_right", "overconfident_wrong"}
BEHAVIORAL_ACTS = RESULTS_DIR / "features" / "behavioral_acts.pt"


def collect(model, tok, records: list[dict]) -> tuple[torch.Tensor, list[dict]]:
    """Per-record: the M2 reasoning measurement + trace-pooled all-layer activations.

    M2.measure generates the greedy trace and labels it; we reuse that trace for
    the activations (mean over generated tokens), so labels and activations
    describe the SAME rollout.
    """
    lids = logit_method.letter_token_ids(tok)
    acts, labels = [], []
    for r in records:
        res = logit_method.measure(model, tok, r, letter_ids=lids)
        prompt = chat_prompt(tok, mcq_prompt(r))
        trace = res["generations"][0]["text"]
        acts.append(get_trace_activations(model, tok, prompt, trace))
        labels.append(res)
    return torch.stack(acts), labels


def load_rollout_records(seed_dirs: list[Path]) -> list[dict]:
    """(trace, labels) units from saved eval rollouts, deduped by question id.

    Each item: the rendered prompt + M2's greedy trace, the M2 state/confidence,
    and the M4 state for the same question (None if M4 never saw it).
    """
    m4_by_id = {}
    for d in seed_dirs:
        for r in read_jsonl(d / "rollouts" / "m4_yesno.jsonl"):
            m4_by_id[r["id"]] = r["state"]
    items, seen = [], set()
    for d in seed_dirs:
        for r in read_jsonl(d / "rollouts" / "m2_logit.jsonl"):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            g = r["generations"][0]
            items.append({"id": r["id"], "prompt": g["prompt"], "trace": g["text"],
                          "m2_state": r["state"], "m2_confidence": r["confidence"],
                          "m4_state": m4_by_id.get(r["id"])})
    return items


def collect_from_rollouts(model, tok, items: list[dict]) -> torch.Tensor:
    """Trace-pooled all-layer activations for saved traces -> [n, n_layers, d]. Forward-only."""
    acts = []
    for i, it in enumerate(items):
        acts.append(get_trace_activations(model, tok, it["prompt"], it["trace"]))
        if (i + 1) % 100 == 0:
            print(f"  activations {i + 1}/{len(items)}", flush=True)
    return torch.stack(acts)


def contrast(acts: torch.Tensor, labels: list[dict], pos_sel, neg_sel) -> tuple[torch.Tensor, int, int]:
    """diff-in-means between two label-selected groups -> (dirs, n_pos, n_neg)."""
    pos_idx = [i for i, l in enumerate(labels) if pos_sel(l)]
    neg_idx = [i for i, l in enumerate(labels) if neg_sel(l)]
    dirs = diff_in_means(acts[pos_idx], acts[neg_idx])
    return dirs, len(pos_idx), len(neg_idx)


def build_all(acts: torch.Tensor, labels: list[dict]) -> dict[str, tuple[torch.Tensor, int, int]]:
    """Build the three candidate directions from collected activations + labels."""
    conf = [l["confidence"] for l in labels]
    med = sorted(conf)[len(conf) // 2]
    is_ocw = lambda l: l["state"] == "overconfident_wrong"   # noqa: E731
    is_cr = lambda l: l["state"] == "confident_right"        # noqa: E731
    return {
        "conf_split": contrast(acts, labels, lambda l: l["confidence"] > med, lambda l: l["confidence"] <= med),
        "ocw_vs_cr": contrast(acts, labels, is_ocw, is_cr),
        "ocw_vs_rest": contrast(acts, labels, is_ocw, lambda l: not is_ocw(l)),
    }


def build_all_rollout(acts: torch.Tensor, items: list[dict]) -> dict[str, tuple[torch.Tensor, int, int]]:
    """The M2 contrasts plus M4- and consensus-labeled ones over the SAME activations."""
    m2 = lambda it: it["m2_state"] in CONFIDENT_STATES                                # noqa: E731
    m4 = lambda it: it["m4_state"] is not None and it["m4_state"] in CONFIDENT_STATES  # noqa: E731
    has4 = lambda it: it["m4_state"] is not None                                       # noqa: E731
    labels = [{"confidence": it["m2_confidence"], "state": it["m2_state"]} for it in items]
    built = build_all(acts, labels)
    built["m4_conf"] = contrast(acts, items, m4, lambda it: has4(it) and not m4(it))
    built["consensus_conf"] = contrast(acts, items, lambda it: m2(it) and m4(it),
                                       lambda it: has4(it) and not m2(it) and not m4(it))
    return built


def _selftest():
    torch.manual_seed(0)
    n_layers, d = 2, 8
    acts = torch.randn(6, n_layers, d)
    labels = [{"confidence": c, "state": s} for c, s in [
        (0.9, "overconfident_wrong"), (0.9, "confident_right"), (0.8, "overconfident_wrong"),
        (0.85, "confident_right"), (0.3, "nonconfident_wrong"), (0.2, "nonconfident_right")]]
    dirs, npos, nneg = contrast(acts, labels, lambda l: l["state"] == "overconfident_wrong",
                                lambda l: l["state"] == "confident_right")
    assert npos == 2 and nneg == 2
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(n_layers), atol=1e-5)
    assert set(build_all(acts, labels)) == {"conf_split", "ocw_vs_cr", "ocw_vs_rest"}

    items = [{"m2_confidence": l["confidence"], "m2_state": l["state"],
              "m4_state": m4} for l, m4 in zip(labels, [
        "nonconfident_wrong", "confident_right", "overconfident_wrong",
        "confident_right", None, "nonconfident_right"])]
    built = build_all_rollout(acts, items)
    assert set(built) == {"conf_split", "ocw_vs_cr", "ocw_vs_rest", "m4_conf", "consensus_conf"}
    assert built["m4_conf"][1:] == (3, 2)         # 3 M4-confident, 2 not (None excluded)
    assert built["consensus_conf"][1:] == (3, 1)  # both-confident x3; both-not only the last
    print("features_caa_behavioral self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--rollouts", default="",
                    help="comma-separated results/methods_seed* dirs to reuse instead of regenerating")
    args = ap.parse_args()

    _selftest()

    import torch.nn.functional as F
    from models_specific.active import load_model

    model, tok = load_model()
    if args.rollouts:
        items = load_rollout_records([RESULTS_DIR / s for s in args.rollouts.split(",")])
        print(f"{len(items)} unique traces from {args.rollouts}")
        print("m2 states:", dict(Counter(it["m2_state"] for it in items)))
        print("m4 states:", dict(Counter(str(it["m4_state"]) for it in items)))
        acts = collect_from_rollouts(model, tok, items)
        BEHAVIORAL_ACTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"acts": acts.to(torch.float16),
                    "items": [{k: it[k] for k in ("id", "m2_state", "m2_confidence", "m4_state")}
                              for it in items]}, BEHAVIORAL_ACTS)
        print(f"saved activations -> {BEHAVIORAL_ACTS}")
        built = build_all_rollout(acts, items)
    else:
        records = load_mmlu(n=args.n, seed=args.seed)
        acts, labels = collect(model, tok, records)
        print("state histogram (extraction split):", dict(Counter(l["state"] for l in labels)))
        built = build_all(acts, labels)
    persona = torch.load(DIRECTIONS_DIR / "caa.pt")["directions"] if (DIRECTIONS_DIR / "caa.pt").exists() else None
    L = args.layer
    print(f"\ncandidate directions @ layer {L}:")
    for name, (dirs, npos, nneg) in built.items():
        save(dirs, name=f"caa_{name}")
        extra = "" if persona is None else f"  cos_to_persona={F.cosine_similarity(dirs[L], persona[L], dim=0):+.3f}"
        print(f"  {name:14s} (n_pos={npos:>3}, n_neg={nneg:>3}){extra}")

    names = list(built)
    print("\npairwise cosine @ layer", L)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = F.cosine_similarity(built[names[i]][0][L], built[names[j]][0][L], dim=0).item()
            print(f"  cos({names[i]}, {names[j]}) = {c:+.3f}")
