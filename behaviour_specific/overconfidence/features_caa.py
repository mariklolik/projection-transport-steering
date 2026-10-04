from __future__ import annotations

import os
import random
from pathlib import Path

import torch

from behaviour_specific.overconfidence.mmlu.data import load_mmlu, mcq_prompt
from behaviour_specific.overconfidence.personas import CALIBRATED_PERSONA, OVERCONFIDENT_PERSONA
from general.inference import generate, get_trace_activations
from general.paths import RESULTS_DIR
from models_specific.active import chat_prompt

DIRECTIONS_DIR = Path(os.environ.get("PTS_DIRECTIONS", Path(__file__).parent / "directions"))
PERSONA_ACTS = RESULTS_DIR / "features" / "persona_acts.pt"


def extraction_records(n: int, seed: int, eval_seeds=(7, 11, 23), eval_n: int = 1000) -> list[dict]:
    eval_ids = {r["id"] for s in eval_seeds for r in load_mmlu(n=eval_n, seed=s)}
    pool = [r for r in load_mmlu() if r["id"] not in eval_ids]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def diff_in_means(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    d = pos.mean(0) - neg.mean(0)
    return d / d.norm(dim=-1, keepdim=True)


def collect_pole_activations(model, tok, records: list[dict], persona: str) -> torch.Tensor:
    acts = []
    for i, r in enumerate(records):
        prompt = chat_prompt(tok, mcq_prompt(r), system=persona)
        trace = generate(model, tok, prompt)
        acts.append(get_trace_activations(model, tok, prompt, trace))
        if (i + 1) % 50 == 0:
            print(f"  [{persona[:20]}...] {i + 1}/{len(records)}", flush=True)
    return torch.stack(acts)


def save(directions: torch.Tensor, name: str = "caa") -> Path:
    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIRECTIONS_DIR / f"{name}.pt"
    torch.save({"directions": directions, "method": name,
                "positive": OVERCONFIDENT_PERSONA, "negative": CALIBRATED_PERSONA}, path)
    return path


def _selftest():
    torch.manual_seed(0)
    n_layers, d = 3, 8
    shift = torch.randn(n_layers, d)
    pos = torch.randn(n_layers, d) + shift + 0.01 * torch.randn(5, n_layers, d)
    neg = pos - shift
    dirs = diff_in_means(pos, neg)
    assert dirs.shape == (n_layers, d)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(n_layers), atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(dirs, shift / shift.norm(dim=-1, keepdim=True), dim=-1)
    assert (cos > 0.95).all()
    print("features_caa self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="extraction questions (default 20)")
    ap.add_argument("--seed", type=int, default=2, help="extraction split seed (disjoint from eval)")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    model, tok = load_model()
    records = extraction_records(args.n, args.seed)
    pos = collect_pole_activations(model, tok, records, OVERCONFIDENT_PERSONA)
    neg = collect_pole_activations(model, tok, records, CALIBRATED_PERSONA)
    directions = diff_in_means(pos, neg)
    path = save(directions)
    PERSONA_ACTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pos": pos, "neg": neg, "ids": [r["id"] for r in records], "seed": args.seed}, PERSONA_ACTS)
    print(f"extracted {tuple(directions.shape)} directions from {len(records)} questions -> {path}")
    print(f"saved pole activations -> {PERSONA_ACTS}")
    import torch.nn.functional as F
    mid = directions.shape[0] // 2
    print(f"cos(layer {mid}, layer {mid + 1}) = {F.cosine_similarity(directions[mid], directions[mid + 1], dim=0):.3f}")
