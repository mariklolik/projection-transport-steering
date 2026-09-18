# Fit data for PTS maps: label natural traces with M2+M4, build diff-in-means
# directions (m4_conf, ocw_vs_cr, ...), collect per-token projection quantiles
# per state, dump per-trace projections (gate calibration).
# Run: python -m behaviour_specific.overconfidence.extract_projection_stats \
#      --n 400 --layer 14   (outputs to directions/)

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import torch

from behaviour_specific.overconfidence.confidence_logit import letter_token_ids
from behaviour_specific.overconfidence.confidence_yesno import yes_no_token_ids
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.mmlu.data import load_mmlu, mcq_prompt
from behaviour_specific.overconfidence.steer_overconfidence import score_m2_batch, score_m4_batch
from general.storage import write_json
from models_specific.active import chat_prompt

QUANTILES = torch.linspace(0.01, 0.99, 41)
CALIBRATED = {"confident_right", "nonconfident_wrong"}  # confidence matched correctness


@torch.no_grad()
def token_projections(model, tok, prompt: str, trace: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward over prompt+trace -> (prompt-span, trace-span) residuals at `layer`, fp32 CPU."""
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    store = {}

    def hook(_m, _inp, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        ids = tok(prompt + trace, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        handle.remove()
    h = store["h"][0].float().cpu()
    return h[:n_prompt], h[n_prompt:]


def diff_in_means_1l(acts: torch.Tensor, pos: list[int], neg: list[int]) -> torch.Tensor:
    """Unit-norm mean difference between two index groups of [n, d] vectors."""
    d = acts[pos].mean(0) - acts[neg].mean(0)
    return d / d.norm()


def group_stats(proj: torch.Tensor) -> dict:
    """Quantile grid + moments of a 1-D projection sample (empty-safe)."""
    if proj.numel() < 20:
        return {"n": int(proj.numel())}
    return {"n": int(proj.numel()), "q": proj.quantile(QUANTILES).tolist(),
            "mu": proj.mean().item(), "sig": proj.std().item(),
            "min": proj.min().item(), "max": proj.max().item()}


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank AUROC (probability a positive outranks a negative; ties not corrected)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r + 1
    pos = [i for i, l in enumerate(labels) if l == 1]
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    if not n_pos or not n_neg:
        return float("nan")
    return (sum(ranks[i] for i in pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _selftest():
    torch.manual_seed(0)
    acts = torch.randn(10, 8)
    d = diff_in_means_1l(acts, [0, 1, 2], [3, 4, 5])
    assert abs(d.norm().item() - 1.0) < 1e-5
    s = group_stats(torch.randn(1000))
    assert s["n"] == 1000 and len(s["q"]) == len(QUANTILES)
    assert group_stats(torch.randn(3)) == {"n": 3}
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert abs(auroc([0.1, 0.9, 0.2, 0.8], [1, 0, 1, 0]) - 0.0) < 1e-9
    print("extract_projection_stats self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--eval-n", type=int, default=300, help="steering eval subset to EXCLUDE")
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    eval_ids = {r["id"] for r in load_mmlu(n=args.eval_n, seed=args.eval_seed)}
    pool = [r for r in load_mmlu(n=None, seed=args.seed) if r["id"] not in eval_ids]
    records = pool[: args.n]
    print(f"extraction split: {len(records)} records (seed {args.seed}, "
          f"{args.eval_n} eval ids excluded), layer {args.layer}", flush=True)

    model, tok = load_model()
    lids = letter_token_ids(tok)
    ids4 = yes_no_token_ids(tok)

    print("scoring M2 (greedy traces)...", flush=True)
    m2 = score_m2_batch(model, tok, records, lids, args.batch_size)
    print("scoring M4 (yes/no)...", flush=True)
    m4 = score_m4_batch(model, tok, records, ids4, args.batch_size)
    m4_by_id = {r["id"]: r["state"] for r in m4}
    print("m2 states:", dict(Counter(r["state"] for r in m2)))
    print("m4 states:", dict(Counter(m4_by_id[r["id"]] for r in m2)), flush=True)

    # per-token activations at the target layer, one forward per record
    prompt_toks, trace_toks, pooled = [], [], []
    for i, r in enumerate(m2):
        g = r["generations"][0]
        hp, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
        if ht.numel() == 0:
            ht = hp[-1:]
        prompt_toks.append(hp.half())
        trace_toks.append(ht.half())
        pooled.append(ht.float().mean(0))
        if (i + 1) % 100 == 0:
            print(f"  activations {i + 1}/{len(m2)}", flush=True)
    pooled = torch.stack(pooled)

    # rebuild on-policy directions from THIS split (repo only ships M2-label dirs)
    m2_conf = [r["confidence"] for r in m2]
    med = sorted(m2_conf)[len(m2_conf) // 2]
    m4_states = [m4_by_id[r["id"]] for r in m2]
    m2_states = [r["state"] for r in m2]
    CONF = {"confident_right", "overconfident_wrong"}
    sel = {
        "m4_conf": ([i for i, s in enumerate(m4_states) if s in CONF],
                    [i for i, s in enumerate(m4_states) if s not in CONF]),
        "conf_split": ([i for i, c in enumerate(m2_conf) if c > med],
                       [i for i, c in enumerate(m2_conf) if c <= med]),
        "ocw_vs_rest": ([i for i, s in enumerate(m4_states) if s == "overconfident_wrong"],
                        [i for i, s in enumerate(m4_states) if s != "overconfident_wrong"]),
        "ocw_vs_cr": ([i for i, s in enumerate(m4_states) if s == "overconfident_wrong"],
                      [i for i, s in enumerate(m4_states) if s == "confident_right"]),
        "consensus_conf": ([i for i in range(len(m2)) if m2_states[i] in CONF and m4_states[i] in CONF],
                           [i for i in range(len(m2)) if m2_states[i] not in CONF and m4_states[i] not in CONF]),
    }
    dirs = {}
    for name, (pos, neg) in sel.items():
        if len(pos) >= 5 and len(neg) >= 5:
            dirs[name] = diff_in_means_1l(pooled, pos, neg)
            print(f"direction {name}: n_pos={len(pos)} n_neg={len(neg)}")
        else:
            print(f"direction {name}: SKIPPED (n_pos={len(pos)} n_neg={len(neg)})")

    import torch.nn.functional as F
    for a in dirs:
        for b in dirs:
            if a < b:
                print(f"  cos({a},{b}) = {F.cosine_similarity(dirs[a], dirs[b], dim=0):+.3f}")

    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"layer": args.layer, "dirs": dirs,
                "counts": {k: (len(p), len(n)) for k, (p, n) in sel.items()}},
               DIRECTIONS_DIR / f"pts_L{args.layer}.pt")

    # per-trace records: states + trace-mean projection on every direction
    # (feeds the gate calibration of conditional/2-D steering + the trace-level
    # separability table). Saved BEFORE the token-level stats so a crash there
    # still leaves the trace dump.
    trace_rows = []
    for i, r in enumerate(m2):
        row = {"id": r["id"], "m2_state": m2_states[i], "m2_confidence": m2_conf[i],
               "m4_state": m4_states[i], "n_trace_tokens": int(trace_toks[i].shape[0]),
               "prompt": m2[i]["generations"][0]["prompt"], "trace": m2[i]["generations"][0]["text"]}
        for name, v in dirs.items():
            row[f"proj_{name}"] = round((trace_toks[i].float() @ v).mean().item(), 4)
        trace_rows.append(row)
    from general.storage import write_jsonl
    write_jsonl(DIRECTIONS_DIR / f"trace_projections_L{args.layer}.jsonl", trace_rows)

    # trace-level per-state stats per direction (for gate thresholds)
    trace_stats = {}
    for name in dirs:
        by_state = defaultdict(list)
        for row in trace_rows:
            by_state[row["m4_state"]].append(row[f"proj_{name}"])
        trace_stats[name] = {s: group_stats(torch.tensor(v_)) for s, v_ in by_state.items()}

    # projection stats per direction × group
    stats, summary = {}, {}
    for name, v in dirs.items():
        per_state = defaultdict(list)
        src_all, prompt_all, by_id_mean = [], [], {}
        for i, r in enumerate(m2):
            pt = trace_toks[i].float() @ v
            per_state[m4_states[i]].append(pt)
            src_all.append(pt)
            prompt_all.append(prompt_toks[i].float() @ v)
            by_id_mean[r["id"]] = pt.mean().item()
        cat = lambda xs: torch.cat(xs) if xs else torch.empty(0)  # noqa: E731
        g = {
            "src_all": group_stats(cat(src_all)),
            "prompt": group_stats(cat(prompt_all)),
            "tgt_calibrated": group_stats(cat(sum([per_state[s] for s in CALIBRATED if s in per_state], []))),
            "tgt_non_ocw": group_stats(cat(sum([v_ for s, v_ in per_state.items()
                                                if s != "overconfident_wrong"], []))),
            "tgt_cr": group_stats(cat(per_state.get("confident_right", []))),
            "per_state": {s: group_stats(cat(v_)) for s, v_ in per_state.items()},
        }
        stats[name] = g
        # sanity: does the trace-mean projection predict the M4 confidence label?
        lab = [1 if s in CONF else 0 for s in m4_states]
        sc = [by_id_mean[r["id"]] for r in m2]
        summary[name] = {"auroc_traceproj_vs_m4conf": round(auroc(sc, lab), 4),
                         "groups": {k: v_["n"] for k, v_ in g.items() if k != "per_state"}}
        print(f"{name}: AUROC(trace-mean proj -> M4 confident) = {summary[name]['auroc_traceproj_vs_m4conf']}")

    torch.save({"layer": args.layer, "quantiles": QUANTILES, "stats": stats,
                "trace_stats": trace_stats},
               DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    write_json(DIRECTIONS_DIR / f"projection_stats_L{args.layer}_summary.json",
               {"n_records": len(records), "layer": args.layer,
                "m2_states": dict(Counter(m2_states)), "m4_states": dict(Counter(m4_states)),
                "directions": summary})
    print(f"saved -> {DIRECTIONS_DIR}/pts_L{args.layer}.pt, projection_stats_L{args.layer}.pt")
