# Where in depth does the behavior become linearly readable, and is layer 14 the
# right place to act? One forward per saved extraction trace, hooking every
# decoder layer, gives for each layer: the confidence axis, the detection axis,
# the token-level separation of the two confidence subpopulations, and the
# trace-level LDA AUROC of the gate. The same pass dumps the per-neuron and
# full-space moments the layer-tuned baselines need.
# Run: python -m behaviour_specific.overconfidence.diag_layers --layers 4,7,10,14,18,22,25

from __future__ import annotations

import argparse

import torch

from behaviour_specific.overconfidence.extract_joint_stats import lda_fit
from behaviour_specific.overconfidence.extract_projection_stats import auroc
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from general.paths import RESULTS_DIR
from general.steering import orthonormalize
from general.storage import read_jsonl, write_json

CALIBRATED = {"confident_right", "nonconfident_wrong"}
OCW = "overconfident_wrong"
CONFIDENT = {"confident_right", OCW}


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    s = (((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2)).clamp(min=1e-12).sqrt()
    return float((a.mean() - b.mean()) / s)


def unit_diff(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    d = pos.mean(0) - neg.mean(0)
    return d / d.norm().clamp(min=1e-12)


def _selftest():
    a, b = torch.zeros(10), torch.ones(10) * 2
    assert cohens_d(b, a) > 10
    u = unit_diff(torch.ones(4, 3), torch.zeros(4, 3))
    assert abs(float(u.norm()) - 1.0) < 1e-5
    print("diag_layers self-test passed")


@torch.no_grad()
def all_layer_trace(model, tok, prompt: str, trace: str, layers: list[int]) -> dict[int, torch.Tensor]:
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    store: dict[int, torch.Tensor] = {}
    handles = []
    for L in layers:
        def hook(_m, _i, out, L=L):
            h = out[0] if isinstance(out, tuple) else out
            store[L] = h[0].float().cpu()
        handles.append(model.model.layers[L].register_forward_hook(hook))
    try:
        ids = tok(prompt + trace, return_tensors="pt", add_special_tokens=False).to(model.device)
        model(**ids)
    finally:
        for h in handles:
            h.remove()
    return {L: v[n_prompt:] for L, v in store.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="4,7,10,14,18,22,25")
    ap.add_argument("--ref-layer", type=int, default=14, help="layer whose saved traces we reuse")
    ap.add_argument("--max-tokens-per-trace", type=int, default=120)
    ap.add_argument("--dump-moments", action="store_true", help="also save per-layer AcT/MiMiC moments")
    ap.add_argument("--outdir", default="v3_layers")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    layers = [int(x) for x in args.layers.split(",")]
    rows = read_jsonl(DIRECTIONS_DIR / f"trace_projections_L{args.ref_layer}.jsonl")
    model, tok = load_model()
    print(f"{len(rows)} traces, layers {layers}", flush=True)

    tok_acts: dict[int, list] = {L: [] for L in layers}
    trace_means: dict[int, list] = {L: [] for L in layers}
    states = []
    torch.manual_seed(0)
    for i, r in enumerate(rows):
        per = all_layer_trace(model, tok, r["prompt"], r["trace"], layers)
        idx = torch.randperm(next(iter(per.values())).shape[0])[: args.max_tokens_per_trace]
        for L, ht in per.items():
            tok_acts[L].append(ht[idx].half())
            trace_means[L].append(ht.mean(0))
        states.append(r["m4_state"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    st = torch.tensor([0 if s == OCW else 1 for s in states])
    is_ocw = st == 0
    is_cr = torch.tensor([s == "confident_right" for s in states])
    is_conf_tr = torch.tensor([s in CONFIDENT for s in states])
    out = {"layers": {}, "n_traces": len(rows)}
    for L in layers:
        X = torch.stack(trace_means[L])                     # [n, d] trace means
        A = torch.cat(tok_acts[L]).float()                  # [~48k, d] tokens
        per_trace = torch.stack([t.float().mean(0) for t in tok_acts[L]])
        v = unit_diff(per_trace[is_conf_tr], per_trace[~is_conf_tr])       # confidence axis
        u = unit_diff(per_trace[is_ocw], per_trace[is_cr])                 # detection axis
        V = orthonormalize(torch.stack([v, u]))
        q = X @ V.T
        w, b = lda_fit(q[is_ocw], q[~is_ocw])
        s = q @ w + b
        both = is_ocw | is_cr
        a_cr = auroc(s[both].tolist(), is_ocw[both].int().tolist())
        a_rest = auroc(s.tolist(), is_ocw.int().tolist())
        pv_tok = torch.cat([t.float() @ v for t in tok_acts[L]])
        rep = torch.cat([torch.full((t.shape[0],), int(k)) for k, t in
                         zip([s_ == OCW for s_ in states], tok_acts[L])])
        cr_rep = torch.cat([torch.full((t.shape[0],), int(k)) for k, t in
                            zip([s_ == "confident_right" for s_ in states], tok_acts[L])])
        d_ocw_cr = cohens_d(pv_tok[rep == 1], pv_tok[cr_rep == 1])
        conf_rep = torch.cat([torch.full((t.shape[0],), int(s_ in CONFIDENT)) for s_, t in
                              zip(states, tok_acts[L])])
        d_conf = cohens_d(pv_tok[conf_rep == 1], pv_tok[conf_rep == 0])
        out["layers"][L] = {"gate_auroc_ocw_vs_cr": round(a_cr, 4),
                            "gate_auroc_ocw_vs_rest": round(a_rest, 4),
                            "cohen_d_ocw_vs_cr_tokens": round(d_ocw_cr, 4),
                            "cohen_d_confident_vs_not_tokens": round(d_conf, 4),
                            "ceiling_selectivity": round(2 * a_cr - 1, 4)}
        print(f"  L{L:2d} AUROC(OCW|CR)={a_cr:.3f} d(OCW,CR)={d_ocw_cr:+.3f} "
              f"d(conf)={d_conf:+.3f}", flush=True)
        if args.dump_moments:
            tgt = torch.cat([t.float() for t, s_ in zip(tok_acts[L], states) if s_ in CALIBRATED])
            torch.save({"mu_s": A.mean(0), "sig_s": A.std(0).clamp(min=1e-4),
                        "mu_t": tgt.mean(0), "sig_t": tgt.std(0).clamp(min=1e-4)},
                       DIRECTIONS_DIR / f"perneuron_stats_L{L}.pt")
            torch.save({"m_s": A.mean(0), "S_s": torch.cov(A.T),
                        "m_t": tgt.mean(0), "S_t": torch.cov(tgt.T)},
                       DIRECTIONS_DIR / f"fullspace_moments_L{L}.pt")
            q_tok = A @ V.T
            torch.save({"layer": L, "V": V, "v": v, "u": u, "S_tok": torch.cov(q_tok.T),
                        "gate_lda": {"w": w, "b": b, "auroc_ocw_vs_cr": a_cr,
                                     "score_quantiles": {
                                         "all": s.quantile(torch.linspace(0.01, 0.99, 41)),
                                         "cr": s[is_cr].quantile(torch.linspace(0.01, 0.99, 41))}}},
                       DIRECTIONS_DIR / f"layerfit_L{L}.pt")

    best = max(out["layers"], key=lambda L: out["layers"][L]["gate_auroc_ocw_vs_cr"])
    out["best_layer_by_gate_auroc"] = best
    write_json(RESULTS_DIR / args.outdir / "layer_sweep.json", out)
    print("best layer:", best, "->", RESULTS_DIR / args.outdir / "layer_sweep.json")
