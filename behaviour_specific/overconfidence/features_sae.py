# SAE behavioral diffing, across one or MANY layers — which SAE features fire
# more in OVER-CONFIDENT reasoning than in calibrated reasoning, and at what depth?
#
# HYPOTHESIS (plan Phase 2b): overconfidence is carried by a few interpretable
# SAE features. Under each persona the model writes a <think> trace; at each
# chosen layer we encode every trace token through that layer's SAE, mean the
# feature activations over the trace, and rank features by
#
#   diff[f] = mean(feature f | over-confident traces) - mean(feature f | calibrated)
#
# Sweeping layers is the point: we don't know where overconfidence lives, so we
# scan (e.g. every 5th layer) and read off (a) the best layer, (b) the DEPTH
# PROFILE of how strongly it is encoded vs layer, and (c) a feature card per
# layer. Traces are generated ONCE and read at every layer in a single forward.
#
# SAE chosen with --sae (see saes.py): gemmascope (JumpReLU, any layer 0-25),
# saebench-topk / saebench-vanilla (layers 5/12/19, via sae_lens). Weights must
# be present locally (download + transfer, or Gemma Scope npz from the web).
#
# Two sources of (trace, label), mirroring features_caa_behavioral:
#   default     — persona contrast (generate traces under the two personas);
#   --rollouts  — BEHAVIORAL: reuse the saved eval rollouts (forward-only), rank
#                 features by mean(M4-confident traces) − mean(M4-not-confident).
#                 §3 showed persona directions are prompt features — this asks
#                 whether the same split appears in the SAE dictionary view.
#
# `python -m ...features_sae [--sae KIND] [--layers 0,5,10,15,20,25] [--n N] [--topk K]
#      [--rollouts methods_seed11,methods_seed23]`

from __future__ import annotations

import torch

from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.mmlu.data import load_mmlu, mcq_prompt
from behaviour_specific.overconfidence.personas import CALIBRATED_PERSONA, OVERCONFIDENT_PERSONA
from general.inference import generate, get_trace_token_activations_multi
from general.paths import RESULTS_DIR
from general.storage import write_json
from models_specific.gemma_2_2b_it import saes
from models_specific.active import chat_prompt


def collect_features_multi(model, tok, saes_by_layer: dict, records: list[dict], persona: str) -> dict:
    """Mean SAE feature activations per record per layer -> {layer: [n, d_sae]}.

    Each record's reasoning trace is generated ONCE; a single forward reads every
    layer; each layer's tokens are encoded through that layer's SAE and meaned.
    """
    layers = sorted(saes_by_layer)
    acc = {L: [] for L in layers}
    for r in records:
        prompt = chat_prompt(tok, mcq_prompt(r), system=persona)
        trace = generate(model, tok, prompt)
        toks = get_trace_token_activations_multi(model, tok, prompt, trace, layers)
        for L in layers:
            acc[L].append(saes_by_layer[L].encode(toks[L].to(model.device)).float().mean(0).cpu())
    return {L: torch.stack(v) for L, v in acc.items()}


def collect_features_from_rollouts(model, tok, saes_by_layer: dict, items: list[dict]) -> dict:
    """Mean SAE feature activations for SAVED traces -> {layer: [n, d_sae]}. Forward-only."""
    layers = sorted(saes_by_layer)
    acc = {L: [] for L in layers}
    for i, it in enumerate(items):
        toks = get_trace_token_activations_multi(model, tok, it["prompt"], it["trace"], layers)
        for L in layers:
            acc[L].append(saes_by_layer[L].encode(toks[L].to(model.device)).float().mean(0).cpu())
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(items)}", flush=True)
    return {L: torch.stack(v) for L, v in acc.items()}


def feature_diff(pos_feats: torch.Tensor, neg_feats: torch.Tensor) -> torch.Tensor:
    """Per-feature mean-activation difference (over-confident minus calibrated) -> [d_sae]."""
    return pos_feats.mean(0) - neg_feats.mean(0)


def top_features(diff: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    """The k features that fire most in over-confident traces. Returns (indices, scores)."""
    scores, idxs = torch.topk(diff, k)
    return idxs.tolist(), scores.tolist()


def _selftest():
    torch.manual_seed(0)
    # two fake layers, each a fake SAE (identity encode): exercise the sweep logic
    class FakeSAE:
        def __init__(self, bump): self.bump = bump
        def encode(self, x): return x
        d_sae = 20

    def diff_at(bump):
        pos = torch.rand(6, 20); neg = torch.rand(5, 20); pos[:, bump] += 5.0
        return feature_diff(pos, neg)

    profile = {L: top_features(diff_at(b), 3) for L, b in [(5, 3), (10, 7)]}
    assert profile[5][0][0] == 3 and profile[10][0][0] == 7   # each layer finds its own feature
    print("features_sae self-test passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default=saes.GEMMASCOPE, help=f"SAE kind {saes.KINDS}")
    ap.add_argument("--layers", default="", help="comma list, e.g. 0,5,10,15,20,25 (default: the kind's layer)")
    ap.add_argument("--n", type=int, default=20, help="persona questions per pole")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--rollouts", default="",
                    help="comma-separated results/methods_seed* dirs: behavioral labels instead of personas")
    args = ap.parse_args()

    _selftest()

    from models_specific.active import load_model

    layers = [int(x) for x in args.layers.split(",")] if args.layers else [saes.default_layer(args.sae)]

    model, tok = load_model()
    dev = str(model.device)
    saes_by_layer = {L: saes.load(args.sae, L, device=dev) for L in layers}
    d_sae = next(iter(saes_by_layer.values())).d_sae
    print(f"SAE '{args.sae}' over layers {layers} (d_sae {d_sae})")

    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rollouts:
        from behaviour_specific.overconfidence.features_caa_behavioral import CONFIDENT_STATES, load_rollout_records

        items = load_rollout_records([RESULTS_DIR / s for s in args.rollouts.split(",")])
        eval_ids = {r["id"] for r in load_mmlu(n=300, seed=7)}  # steering eval set — keep dirs leak-free
        items = [it for it in items if it["id"] not in eval_ids]
        print(f"{len(items)} unique traces from {args.rollouts} (steering-eval overlap excluded)")

        feats = collect_features_from_rollouts(model, tok, saes_by_layer, items)
        feats_path = RESULTS_DIR / "features" / "sae_behavioral_feats.pt"
        feats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": {L: f.to(torch.float16) for L, f in feats.items()},
                    "items": [{k: it[k] for k in ("id", "m2_state", "m4_state")} for it in items]}, feats_path)
        print(f"saved feature matrices -> {feats_path}")

        has4 = torch.tensor([it["m4_state"] is not None for it in items])
        m4y = torch.tensor([it["m4_state"] in CONFIDENT_STATES if it["m4_state"] else False for it in items])
        m2y = torch.tensor([it["m2_state"] in CONFIDENT_STATES for it in items])

        profile = []
        for L in layers:
            f = feats[L]
            diff_m4 = feature_diff(f[has4 & m4y], f[has4 & ~m4y])
            idxs, scores = top_features(diff_m4, args.topk)
            handle = saes_by_layer[L]
            decoder_dirs = torch.stack([handle.decoder_direction(i).cpu() for i in idxs])
            torch.save({"method": "sae_behavioral", "sae": args.sae, "layer": L, "labels": "m4_conf",
                        "feature_idxs": idxs, "scores": scores, "decoder_dirs": decoder_dirs},
                       DIRECTIONS_DIR / f"sae_beh_{handle.name}.pt")
            i2, s2 = top_features(feature_diff(f[m2y], f[~m2y]), args.topk)
            profile.append({"layer": L, "top_feature_m4": idxs[0], "top_diff_m4": scores[0],
                            "topk_mean_diff_m4": sum(scores) / len(scores),
                            "top_feature_m2": i2[0], "top_diff_m2": s2[0]})
            print(f"\n[layer {L}] top {min(args.topk, 8)} M4-confident features (natural traces):")
            for i, s in list(zip(idxs, scores))[:8]:
                print(f"  feature {i:6d}  diff={s:+.4f}  {saes.feature_url(args.sae, L, i)}")

        print(f"\n=== depth profile ('{args.sae}' behavioral, confident-feature strength vs layer) ===")
        print(f"{'layer':>6} {'top_diff_m4':>12} {'topk_mean':>10} {'top_feat_m4':>12} {'top_feat_m2':>12}")
        for e in profile:
            print(f"{e['layer']:>6} {e['top_diff_m4']:>12.4f} {e['topk_mean_diff_m4']:>10.4f} "
                  f"{e['top_feature_m4']:>12} {e['top_feature_m2']:>12}")
        out = RESULTS_DIR / f"sae_{args.sae}_behavioral"
        write_json(out / "depth_profile.json", {"sae": args.sae, "layers": layers, "topk": args.topk,
                                                "n_traces": len(items), "profile": profile})
        print(f"saved per-layer directions (sae_beh_{args.sae}_l*.pt) + {out / 'depth_profile.json'}")
    else:
        records = load_mmlu(n=args.n, seed=args.seed)
        pos = collect_features_multi(model, tok, saes_by_layer, records, OVERCONFIDENT_PERSONA)
        neg = collect_features_multi(model, tok, saes_by_layer, records, CALIBRATED_PERSONA)

        profile = []
        for L in layers:
            diff = feature_diff(pos[L], neg[L])
            idxs, scores = top_features(diff, args.topk)
            handle = saes_by_layer[L]
            decoder_dirs = torch.stack([handle.decoder_direction(i).cpu() for i in idxs])
            torch.save({"method": "sae", "sae": args.sae, "layer": L, "feature_idxs": idxs,
                        "scores": scores, "decoder_dirs": decoder_dirs},
                       DIRECTIONS_DIR / f"sae_{handle.name}.pt")
            profile.append({"layer": L, "top_feature": idxs[0], "top_diff": scores[0],
                            "topk_mean_diff": sum(scores) / len(scores)})
            print(f"\n[layer {L}] top {min(args.topk, 8)} over-confident features:")
            for i, s in list(zip(idxs, scores))[:8]:
                print(f"  feature {i:6d}  diff={s:+.4f}  {saes.feature_url(args.sae, L, i)}")

        print(f"\n=== depth profile ('{args.sae}', over-confident feature strength vs layer) ===")
        print(f"{'layer':>6} {'top_diff':>10} {'topk_mean':>10} {'top_feature':>12}")
        for e in profile:
            print(f"{e['layer']:>6} {e['top_diff']:>10.4f} {e['topk_mean_diff']:>10.4f} {e['top_feature']:>12}")
        best = max(profile, key=lambda e: e["top_diff"])
        print(f"strongest layer: {best['layer']} (top_diff {best['top_diff']:+.4f})")

        out = RESULTS_DIR / f"sae_{args.sae}"
        write_json(out / "depth_profile.json", {"sae": args.sae, "layers": layers, "n": args.n,
                                                "seed": args.seed, "topk": args.topk, "profile": profile})
        print(f"saved per-layer directions (sae_{args.sae}_l*.pt) + {out / 'depth_profile.json'}")
