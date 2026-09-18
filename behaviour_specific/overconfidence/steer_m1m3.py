# M1 (answer-distribution) and M3 (self-report) readouts for the winning
# condition; cross-instrument robustness check.
# Run: python -m behaviour_specific.overconfidence.steer_m1m3

from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence import confidence_answer_distribution as m1
from behaviour_specific.overconfidence import confidence_self_reported as m3
from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import q_at
from behaviour_specific.overconfidence.steer_v2_gated import gate_quality
from general.paths import RESULTS_DIR
from general.steering import steer_ablate, steering_hook
from general.storage import read_jsonl, write_json, write_jsonl


def _selftest():
    assert hasattr(m1, "measure") and hasattr(m3, "measure")
    print("steer_m1m3 self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--outdir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    v_ = pts["dirs"]["m4_conf"]
    u1 = pts["dirs"]["ocw_vs_cr"].float()
    tau = q_at(pstats["trace_stats"]["ocw_vs_cr"]["confident_right"], 0.50)

    model, tok = load_model()
    v = v_.to(model.device, torch.float32)
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    base_m2 = {r["id"]: r for r in read_jsonl(out_dir / "baseline_m2__shard0.jsonl")}

    print("gate pass ...", flush=True)
    flags = {}
    for i, (rid, r) in enumerate(base_m2.items()):
        g = r["generations"][0]
        _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
        flags[rid] = float((ht @ u1).mean()) > tau
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(base_m2)}", flush=True)
    print("gate:", gate_quality(flags, list(base_m2.values())), flush=True)

    fn = lambda h: steer_ablate(h, v)  # noqa: E731

    for mname, mod in (("m3", m3), ("m1", m1)):
        t0 = time.time()
        rows = [mod.measure(model, tok, r) for r in records]
        write_jsonl(out_dir / f"baseline_{mname}__shard0.jsonl", rows)
        print(f"baseline_{mname}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)

        t0 = time.time()
        steered = []
        for r in records:
            if flags.get(r["id"], False):
                handle = steering_hook(model.model.layers[args.layer], fn)
                try:
                    steered.append(mod.measure(model, tok, r))
                finally:
                    handle.remove()
        sb = {r["id"]: r for r in steered}
        merged = [sb.get(r["id"], b) for r, b in zip(records, rows)]
        write_jsonl(out_dir / f"sweep_ocwcr-crq50_ablate_{mname}__shard0.jsonl", merged)
        print(f"gated-ablate_{mname}: steered {len(steered)}/{len(merged)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    write_json(RESULTS_DIR / args.outdir / "meta_m1m3.json",
               {"config": vars(args), "tau": tau,
                "n_flagged": int(sum(flags.values()))})
    print("done ->", out_dir)
