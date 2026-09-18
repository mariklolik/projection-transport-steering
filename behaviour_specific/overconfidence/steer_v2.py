# PTS grid driver: baseline / additive / ablate / clamp / OT conditions on one
# behavioral direction, M2+M4 readouts, per-question rollouts for analysis.
# Run: python -m behaviour_specific.overconfidence.steer_v2 --n 300 --layer 14 \
#      [--benchmark mmlu|arc|gsm8k|gpqa] [--conditions all|list]

from __future__ import annotations

import argparse
import time

import torch

from behaviour_specific.overconfidence.confidence_logit import (
    cut_at_box, letter_token_ids, probs_over_letters,
)
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.labeling import label_from_score
from behaviour_specific.overconfidence import confidence_logit as logit_method
from behaviour_specific.overconfidence.mmlu.data import LETTERS, correct_letter, load_mmlu, mcq_prompt, parse_boxed_letter
from behaviour_specific.overconfidence.steer_overconfidence import (
    _reasoning_len, mean_activation_norm, score_m2_batch, score_m4_batch,
)
from general.inference import generate, next_token_logits
from general.paths import RESULTS_DIR
from general.reasoning import THINK_CLOSE
from general.steering import (
    steer_ablate, steer_add, steer_clamp, steer_ot_gauss, steer_ot_quantile, steered, steering_hook,
)
from general.storage import write_json, write_jsonl
from models_specific.active import chat_prompt

QGRID = torch.linspace(0.01, 0.99, 41)


def q_at(group: dict, q: float) -> float:
    """The value of quantile q from a saved group's 41-point grid."""
    idx = int(torch.argmin((QGRID - q).abs()))
    return group["q"][idx]


def build_conditions(stats: dict, norm: float) -> dict[str, callable]:
    """{name: v -> (h -> h')} — every transfer function of the grid.

    `stats` is one direction's group dict from projection_stats. Ops close over
    plain floats/tensors so the returned fn is device-agnostic (tensors move in
    the driver).
    """
    src, cal, non = stats["src_all"], stats["tgt_calibrated"], stats["tgt_non_ocw"]
    src_q = torch.tensor(src["q"])
    conds = {
        "add_a-0.75": lambda v: (lambda h: steer_add(h, v, -0.75 * norm)),
        "ablate": lambda v: (lambda h: steer_ablate(h, v)),
        "clamp_q70": lambda v: (lambda h, t=q_at(src, 0.70): steer_clamp(h, v, t)),
        "clamp_q50": lambda v: (lambda h, t=q_at(src, 0.50): steer_clamp(h, v, t)),
        "clamp_q30": lambda v: (lambda h, t=q_at(src, 0.30): steer_clamp(h, v, t)),
    }
    if "q" in cal:
        conds["otg_cal"] = lambda v: (lambda h: steer_ot_gauss(h, v, src["mu"], src["sig"], cal["mu"], cal["sig"]))
        conds["otq_cal"] = lambda v: (lambda h, tq=torch.tensor(cal["q"]): steer_ot_quantile(h, v, src_q, tq))
    if "q" in non:
        conds["otq_nonocw"] = lambda v: (lambda h, tq=torch.tensor(non["q"]): steer_ot_quantile(h, v, src_q, tq))
    return conds


def measure_think_steered(model, tok, record, fn, layer, letter_ids):
    """M2 measurement with an arbitrary op steering ONLY the <think> phase."""
    user = mcq_prompt(record)
    prompt = chat_prompt(tok, user)
    with steered(model, layer, fn):
        think_part = generate(model, tok, prompt, stop_strings=[THINK_CLOSE])
    rest = generate(model, tok, prompt + think_part)
    trace = think_part + rest
    prefix, forced = cut_at_box(trace)
    logits = next_token_logits(model, tok, prompt + prefix)
    probs = probs_over_letters(logits, letter_ids)
    idx = int(probs.argmax())
    pred, conf = LETTERS[idx], float(probs[idx])
    is_correct = pred == correct_letter(record)
    return {
        "id": record["id"], "subject": record["subject"], "method": "logit_steer_think",
        "system_prompt": None, "user_prompt": user,
        "final_answer": pred, "gold": correct_letter(record),
        "confidence": conf, "is_correct": is_correct,
        "state": label_from_score(conf, is_correct, threshold=logit_method.LOGIT_CONF_THRESHOLD),
        "probs": [round(p, 4) for p in probs.tolist()],
        "trace_answer": parse_boxed_letter(trace), "forced_box": forced,
        **_reasoning_len(tok, trace),
        "generations": [{"role": "steer_think", "prompt": prompt, "text": trace}],
    }


def _selftest():
    stats = {"src_all": {"q": torch.linspace(-2, 6, 41).tolist(), "mu": 2.0, "sig": 2.0},
             "tgt_calibrated": {"q": torch.linspace(-2, 2, 41).tolist(), "mu": 0.0, "sig": 1.0},
             "tgt_non_ocw": {"n": 5}}  # too small -> no otq_nonocw
    conds = build_conditions(stats, norm=100.0)
    assert "otq_cal" in conds and "otq_nonocw" not in conds
    v = torch.zeros(8); v[0] = 1.0
    h = torch.zeros(3, 8); h[:, 0] = torch.tensor([1.0, 3.0, 5.0])
    hc = conds["clamp_q50"](v)(h)  # src q50 = 2.0
    assert torch.allclose(hc[:, 0], torch.tensor([1.0, 2.0, 2.0]), atol=1e-5)
    hq = conds["otq_cal"](v)(h)    # map [-2,6] -> [-2,2] linearly: p/2 - 1... (1->-0.5, 3->0.5, 5->1.5)
    assert torch.allclose(hq[:, 0], torch.tensor([-0.5, 0.5, 1.5]), atol=1e-4)
    assert abs(q_at({"q": torch.linspace(0, 1, 41).tolist()}, 0.50) - 0.5) < 1e-6
    print("steer_v2 self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu", help="mmlu | arc | gsm8k (see benchmarks.py)")
    ap.add_argument("--direction", default="m4_conf")
    ap.add_argument("--conditions", default="all", help="'all' or comma-list of condition names")
    ap.add_argument("--methods", default="m2,m4")
    ap.add_argument("--think-only", default="clamp_q50,otq_cal",
                    help="conditions to also run think-only (M2), '' to skip")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outdir", default="steering_v2")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    assert pts["layer"] == args.layer
    v_cpu = pts["dirs"][args.direction]
    stats = pstats["stats"][args.direction]

    from behaviour_specific.overconfidence.benchmarks import load_records

    model, tok = load_model()
    v = v_cpu.to(model.device, torch.float32)
    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    lids = letter_token_ids(tok)
    ids4 = yes_no_token_ids(tok)
    bs = args.batch_size

    norm = mean_activation_norm(model, tok, records, args.layer, lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)
    wanted = list(conds) if args.conditions == "all" else args.conditions.split(",")
    scorers = {}
    for m in args.methods.split(","):
        scorers[m] = (lambda recs: score_m2_batch(model, tok, recs, lids, bs)) if m == "m2" \
            else (lambda recs: score_m4_batch(model, tok, recs, ids4, bs))

    out_dir = RESULTS_DIR / args.outdir / "rollouts"
    print(f"n={len(records)} L={args.layer} dir={args.direction} ||h||={norm:.1f} "
          f"conditions={wanted} methods={list(scorers)}", flush=True)

    def dump(tag, rows, t0):
        write_jsonl(out_dir / f"{tag}__shard0.jsonl", rows)
        print(f"  {tag}: {len(rows)} records ({time.time() - t0:.0f}s)", flush=True)

    for mname, scorer in scorers.items():
        t0 = time.time()
        dump(f"baseline_{mname}", scorer(records), t0)

    for cname in wanted:
        make = conds[cname]
        for mname, scorer in scorers.items():
            t0 = time.time()
            fn = make(v)
            handle = steering_hook(model.model.layers[args.layer], fn)
            try:
                rows = scorer(records)
            finally:
                handle.remove()
            dump(f"{args.direction}_{cname}_{mname}", rows, t0)

    for cname in [c for c in args.think_only.split(",") if c and c in conds]:
        t0 = time.time()
        fn = conds[cname](v)
        rows = [measure_think_steered(model, tok, r, fn, args.layer, lids) for r in records]
        dump(f"{args.direction}_{cname}-think_m2", rows, t0)

    write_json(RESULTS_DIR / args.outdir / "meta_shard0.json",
               {"config": vars(args), "norm": norm, "n_records": len(records),
                "clamp_thresholds": {q: q_at(stats["src_all"], q) for q in (0.70, 0.50, 0.30)},
                "src_moments": {k: stats["src_all"].get(k) for k in ("mu", "sig")},
                "tgt_cal_moments": {k: stats["tgt_calibrated"].get(k) for k in ("mu", "sig")}})
    print("done ->", out_dir)
