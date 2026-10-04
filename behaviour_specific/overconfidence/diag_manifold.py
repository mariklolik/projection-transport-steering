from __future__ import annotations

import argparse

import torch

from behaviour_specific.overconfidence.extract_projection_stats import token_projections
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.steer_v2 import build_conditions
from behaviour_specific.overconfidence.steer_overconfidence import mean_activation_norm
from behaviour_specific.overconfidence.mmlu.data import mcq_prompt
from general.paths import RESULTS_DIR
from general.steering import (
    bw_map, steer_ablate, steer_add, steer_fullspace_affine, steer_perneuron_affine,
)
from general.storage import read_jsonl, write_json
from models_specific.active import chat_prompt


def mahalanobis(x: torch.Tensor, mean: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
    z = x - mean
    return ((z @ prec) * z).sum(-1).clamp(min=0).sqrt()


def conditional_gaussian(V: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor,
                         m: int = 64, reg: float = 1e-3):
    k, d = V.shape
    P = torch.eye(d) - V.T @ V
    C = P @ cov @ P.T
    _, vecs = torch.linalg.eigh(C)
    B = vecs[:, -m:].T
    m_q, m_p = V @ mean, B @ mean
    S_qq, S_qp, S_pp = V @ cov @ V.T, V @ cov @ B.T, B @ cov @ B.T
    K = S_qp @ torch.linalg.inv(S_pp + reg * torch.trace(S_pp) / m * torch.eye(m))
    return B, m_q, m_p, K, torch.linalg.inv(S_qq - K @ S_qp.T + 1e-6 * torch.eye(k))


def conditional_distance(h: torch.Tensor, V: torch.Tensor, B, m_q, m_p, K, prec_c):
    q, pth = h @ V.T, h @ B.T
    z = q - (m_q + (pth - m_p) @ K.T)
    return ((z @ prec_c) * z).sum(-1).clamp(min=0)


@torch.no_grad()
def kl_next_token(model, tok, texts: list[str], fn, layer: int, batch: int = 16) -> float:
    from general.steering import steering_hook
    tot, n = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        base = model(**ids).logits[:, -1].float().log_softmax(-1)
        handle = steering_hook(model.model.layers[layer], fn)
        try:
            steered = model(**ids).logits[:, -1].float().log_softmax(-1)
        finally:
            handle.remove()
        tot += float((base.exp() * (base - steered)).sum(-1).sum())
        n += len(chunk)
    return tot / max(n, 1)


def nn_distance(x: torch.Tensor, bank: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, x.shape[0], chunk):
        out.append(torch.cdist(x[i:i + chunk], bank).min(-1).values)
    return torch.cat(out)


def _selftest():
    m = torch.zeros(3)
    p = torch.eye(3)
    assert abs(mahalanobis(torch.tensor([[3.0, 4.0, 0.0]]), m, p).item() - 5.0) < 1e-5
    bank = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    assert abs(nn_distance(torch.tensor([[1.0, 0.0]]), bank).item() - 1.0) < 1e-5
    Vt = torch.eye(6)[:2]
    parts = conditional_gaussian(Vt, torch.zeros(6), torch.eye(6), m=4)
    dd = conditional_distance(torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]), Vt, *parts)
    assert abs(float(dd) - 25.0) < 0.2, float(dd)
    print("diag_manifold self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--benchmark", default="mmlu")
    ap.add_argument("--rollouts", default="v3_mmlu_s7", help="run dir holding baseline_m2 rollouts")
    ap.add_argument("--bank", type=int, default=4096, help="natural-activation bank size")
    ap.add_argument("--outdir", default="v3_manifold")
    args = ap.parse_args()

    _selftest()

    from behaviour_specific.overconfidence.benchmarks import load_records
    from models_specific.active import load_model

    pts = torch.load(DIRECTIONS_DIR / f"pts_L{args.layer}.pt")
    pstats = torch.load(DIRECTIONS_DIR / f"projection_stats_L{args.layer}.pt")
    fm = torch.load(DIRECTIONS_DIR / f"fullspace_moments_L{args.layer}.pt")
    pn = torch.load(DIRECTIONS_DIR / f"perneuron_stats_L{args.layer}.pt")
    stats = pstats["stats"]["m4_conf"]

    model, tok = load_model()
    dev = model.device
    v = pts["dirs"]["m4_conf"].to(dev, torch.float32)
    joint = torch.load(DIRECTIONS_DIR / f"joint_stats_L{args.layer}.pt")
    pts_V = joint["V"].float().to(dev)
    d = fm["m_s"].shape[0]
    m_s = fm["m_s"].float()
    prec = torch.linalg.inv(fm["S_s"].double() + 1e-2 * torch.eye(d, dtype=torch.float64)).float()

    records = load_records(args.benchmark, n=args.n, seed=args.seed)
    base = {r["id"]: r for r in read_jsonl(RESULTS_DIR / args.rollouts / "rollouts" / "baseline_m2__shard0.jsonl")}
    norm = mean_activation_norm(model, tok, records, args.layer,
                                lambda r: chat_prompt(tok, mcq_prompt(r)))
    conds = build_conditions(stats, norm)

    A = bw_map(fm["S_s"].double() + 1e-4 * torch.eye(d, dtype=torch.float64),
               fm["S_t"].double() + 1e-4 * torch.eye(d, dtype=torch.float64)).float()
    from behaviour_specific.overconfidence.eval_openended import tuned_winner
    tuned_alpha = tuned_winner("additive").get("alpha", -0.25)
    torch.manual_seed(0)
    rnd = torch.randn_like(v)
    rnd = rnd / rnd.norm()
    ops = {
        "random norm-matched": lambda h: steer_add(h, rnd, -0.75 * norm),
        "additive": lambda h: steer_add(h, v, -0.75 * norm),
        "additive (tuned)": lambda h: steer_add(h, v, tuned_alpha * norm),
        "ablation": lambda h: steer_ablate(h, v),
        "clamp_q50": conds["clamp_q50"](v),
        "quantile-OT": conds["otq_cal"](v),
        "MiMiC": lambda h: steer_fullspace_affine(h, fm["m_s"].to(dev), A.to(dev), fm["m_t"].to(dev)),
        "Linear-AcT": lambda h: steer_perneuron_affine(h, pn["mu_s"].to(dev), pn["sig_s"].to(dev),
                                                       pn["mu_t"].to(dev), pn["sig_t"].to(dev), 1.0),
    }

    bank, acts = [], []
    for i, r in enumerate(records):
        b = base.get(r["id"])
        if b is None:
            continue
        g = b["generations"][0]
        _, ht = token_projections(model, tok, g["prompt"], g["text"], args.layer)
        acts.append(ht)
        if sum(x.shape[0] for x in bank) < args.bank:
            bank.append(ht)
    H = torch.cat(acts)
    B = torch.cat(bank)[: args.bank]
    print(f"{len(acts)} traces, {H.shape[0]} tokens, bank {B.shape[0]}", flush=True)

    base_maha = mahalanobis(H, m_s, prec)
    base_nn = nn_distance(H, B)
    Vc = pts_V.cpu()
    cond = conditional_gaussian(Vc, m_s, fm["S_s"].float())
    from scipy.stats import chi2 as _chi2
    chi2_99 = float(_chi2.ppf(0.99, df=Vc.shape[0]))
    base_exceed = float((conditional_distance(H, Vc, *cond) > chi2_99).float().mean())
    prompts = [chat_prompt(tok, mcq_prompt(r)) for r in records]
    out = {"benchmark": args.benchmark, "n_traces": len(acts), "n_tokens": int(H.shape[0]),
           "chi2_99": round(chi2_99, 3),
           "baseline": {"norm": round(float(H.norm(dim=-1).mean()), 2),
                        "mahalanobis": round(float(base_maha.mean()), 3),
                        "nn_distance": round(float(base_nn.mean()), 3),
                        "cond_exceed_99": round(base_exceed, 4)}, "ops": {}}
    for name, fn in ops.items():
        Hp = fn(H.to(dev)).float().cpu()
        rn = (Hp.norm(dim=-1) / H.norm(dim=-1).clamp(min=1e-6))
        disp = ((Hp - H).norm(dim=-1) / H.norm(dim=-1).clamp(min=1e-6))
        mh = mahalanobis(Hp, m_s, prec)
        nnd = nn_distance(Hp, B)
        out["ops"][name] = {
            "norm_ratio_mean": round(float(rn.mean()), 4), "norm_ratio_p99": round(float(rn.quantile(0.99)), 4),
            "rel_displacement_mean": round(float(disp.mean()), 4),
            "rel_displacement_p99": round(float(disp.quantile(0.99)), 4),
            "mahalanobis_mean": round(float(mh.mean()), 3),
            "mahalanobis_ratio": round(float(mh.mean() / base_maha.mean()), 3),
            "nn_distance_mean": round(float(nnd.mean()), 3),
            "nn_ratio": round(float(nnd.mean() / base_nn.mean()), 3),
            "frac_outside_p999": round(float((mh > base_maha.quantile(0.999)).float().mean()), 4),
            "cond_exceed_99": round(float((conditional_distance(Hp, Vc, *cond) > chi2_99).float().mean()), 4),
            "kl_next_token": round(kl_next_token(model, tok, prompts[:64], fn, args.layer), 4)}
        print(f"  {name:12s} {out['ops'][name]}", flush=True)

    write_json(RESULTS_DIR / args.outdir / f"manifold_{args.benchmark}.json", out)
    print("done ->", RESULTS_DIR / args.outdir)
