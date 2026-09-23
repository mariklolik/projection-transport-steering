from __future__ import annotations

import argparse

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.mmlu.data import load_mmlu
from behaviour_specific.overconfidence.steer_overconfidence import score_m2_batch, score_m4_batch, score_m5_batch
from behaviour_specific.overconfidence.sweep_parity import EVAL_SEEDS, tuning_records
from general.inference import get_activations_all_layers, get_trace_activations, get_trace_token_activations_multi
from general.paths import RESULTS_DIR
from general.storage import read_jsonl, write_jsonl

LAYERS = (8, 10, 12, 14, 16, 18, 20, 22)
PREFIX_LAYERS, PREFIX_T = (14, 16, 18), (16, 32, 64, 128)
DETECTOR_N, CONFIRM_N, POOL_SEED = 4000, 3000, 303


def extraction_records(n: int = 400, seed: int = 2) -> list[dict]:
    held = {r["id"] for r in load_mmlu(n=300, seed=7)}
    return [r for r in load_mmlu(seed=seed) if r["id"] not in held][:n]


def used_ids() -> set[str]:
    ids = {r["id"] for r in extraction_records()} | {r["id"] for r in load_mmlu(n=400, seed=2)}
    ids |= {r["id"] for s in EVAL_SEEDS for r in load_mmlu(n=300, seed=s)}
    return ids | {r["id"] for r in tuning_records(1200, 101)}


def split_records(name: str) -> list[dict]:
    if name == "extraction":
        return extraction_records()
    if name == "tuning":
        return tuning_records(1200, 101)
    if name in ("arc", "gsm8k"):
        from behaviour_specific.overconfidence.benchmarks import load_records
        return load_records(name, n=300, seed=7)
    if name.startswith("eval"):
        return load_mmlu(n=300, seed=int(name[4:]))
    used = used_ids()
    rest = [r for r in load_mmlu(seed=POOL_SEED) if r["id"] not in used]
    return {"detector": rest[:DETECTOR_N], "confirm": rest[DETECTOR_N:DETECTOR_N + CONFIRM_N]}[name]


def trace_features(model, tok, m2_rows: list[dict]) -> torch.Tensor:
    return torch.stack([get_trace_activations(model, tok, r["generations"][0]["prompt"],
                                              r["generations"][0]["text"])[list(LAYERS)].half()
                        for r in m2_rows])


def prompt_features(model, tok, m2_rows: list[dict]) -> torch.Tensor:
    return torch.stack([torch.stack([get_activations_all_layers(model, tok, r["generations"][0]["prompt"], pos=pos)
                                     [list(LAYERS)] for pos in ("last", "mean")]).half() for r in m2_rows])


def prefix_features(model, tok, m2_rows: list[dict]) -> torch.Tensor:
    out = []
    for r in m2_rows:
        acts = get_trace_token_activations_multi(model, tok, r["generations"][0]["prompt"],
                                                 r["generations"][0]["text"], list(PREFIX_LAYERS))
        out.append(torch.stack([torch.stack([acts[L][:t].mean(0) for t in PREFIX_T]) for L in PREFIX_LAYERS]).half())
    return torch.stack(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir")
    ap.add_argument("--baseline-from")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from models_specific.active import load_model

    records = split_records(args.split)[args.shard::args.nshards]
    out = RESULTS_DIR / (args.outdir or f"v4_{args.split}")
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    lids, ids4, roll = letter_token_ids(tok), yes_no_token_ids(tok), out / "rollouts"
    own = roll / f"baseline_m2__shard{args.shard}.jsonl"
    if args.baseline_from:
        keep = {r["id"] for r in records}
        m2 = [r for r in read_jsonl(RESULTS_DIR / args.baseline_from / "rollouts" / "baseline_m2__shard0.jsonl")
              if r["id"] in keep]
    elif own.exists():
        m2 = read_jsonl(own)
    else:
        m2 = score_m2_batch(model, tok, records, lids, args.batch_size)
        write_jsonl(own, m2)
        write_jsonl(roll / f"baseline_m4__shard{args.shard}.jsonl",
                    score_m4_batch(model, tok, records, ids4, args.batch_size))
    if not (roll / f"baseline_m5__shard{args.shard}.jsonl").exists():
        write_jsonl(roll / f"baseline_m5__shard{args.shard}.jsonl",
                    score_m5_batch(model, tok, records, lids, ids4, args.batch_size, m2_rows=m2))
    if not (out / f"feats__shard{args.shard}.pt").exists():
        torch.save({"ids": [r["id"] for r in m2], "layers": LAYERS, "X": trace_features(model, tok, m2)},
                   out / f"feats__shard{args.shard}.pt")
    if not (out / f"prompt__shard{args.shard}.pt").exists():
        torch.save({"ids": [r["id"] for r in m2], "layers": LAYERS, "pos": ("last", "mean"),
                    "X": prompt_features(model, tok, m2)}, out / f"prompt__shard{args.shard}.pt")
    if not (out / f"prefix__shard{args.shard}.pt").exists():
        torch.save({"ids": [r["id"] for r in m2], "layers": PREFIX_LAYERS, "t": PREFIX_T,
                    "X": prefix_features(model, tok, m2)}, out / f"prefix__shard{args.shard}.pt")
    print(f"{args.split} shard {args.shard}: {len(m2)} records -> {out}", flush=True)
