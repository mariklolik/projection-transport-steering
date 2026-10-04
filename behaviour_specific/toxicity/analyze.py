from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from behaviour_specific.overconfidence.analyze_pooled import ci
from behaviour_specific.overconfidence.gate_law import auroc
from behaviour_specific.toxicity.run import CLEAN, DIRS, OUT, TOXIC, decision_scores, load, load_feats, threshold
from general.storage import write_json

SIM = 0.5
ALPHAS = ("-0.125", "-0.25", "-0.5", "-0.75", "-1.0")
UNGATED = tuple(f"add{a}" for a in ALPHAS) + ("abl",)
DECISIONS = ("probe", "post", "cast")
QS = (30, 50, 70)


class Embedder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").cuda().eval()
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 128):
            b = self.tok(todo[i:i + 128], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            a = b["attention_mask"].unsqueeze(-1)
            e = torch.nn.functional.normalize((self.model(**b).last_hidden_state * a).sum(1) / a.sum(1), dim=-1).cpu().numpy()
            self.cache.update(zip(todo[i:i + 128], e))
        return np.stack([self.cache[t] for t in texts])


EMBED: Embedder | None = None


def kept(base: list[dict], rows: list[dict]) -> np.ndarray:
    global EMBED
    EMBED = EMBED or Embedder()
    same = np.array([b["text"] == r["text"] for b, r in zip(base, rows)])
    sim = (EMBED([b["text"] for b in base]) * EMBED([r["text"] for r in rows])).sum(1)
    return (np.array([r["state"] == CLEAN for r in rows]) & (same | (sim >= SIM))).astype(float)


def arrays(base: list[dict], rows: list[dict]) -> tuple[np.ndarray, ...]:
    t0 = np.array([b["state"] == TOXIC for b in base], float)
    c0 = np.array([b["state"] == CLEAN for b in base], float)
    t1 = np.array([r["state"] == TOXIC for r in rows], float)
    return t0, c0, t1, kept(base, rows)


def sel(w: np.ndarray, base: list[dict], rows: list[dict]) -> np.ndarray:
    t0, c0, t1, c1 = arrays(base, rows)
    return (w * t0 * (1 - t1)).sum(-1) / (w * t0).sum(-1) - (w * c0 * (1 - c1)).sum(-1) / (w * c0).sum(-1)


def override_sel(w: np.ndarray, base: list[dict], f: np.ndarray) -> np.ndarray:
    t0, c0, _, _ = arrays(base, base)
    return (w * t0 * f).sum(-1) / (w * t0).sum(-1) - (w * c0 * f).sum(-1) / (w * c0).sum(-1)


def compose(base: list[dict], ung: list[dict], f: np.ndarray) -> list[dict]:
    return [u if g else b for b, u, g in zip(base, ung, f)]


def rates(base: list[dict], rows: list[dict], f: np.ndarray) -> dict:
    t0, c0, t1, c1 = arrays(base, rows)
    return {"tpr": float(f[t0 == 1].mean()), "fpr": float(f[c0 == 1].mean()),
            "rho_t": float((1 - t1)[t0 == 1].mean()), "rho_c": float((1 - c1)[c0 == 1].mean())}


def flags(split: str, kind: str, q: int, base: list[dict], feats: dict) -> np.ndarray:
    return (decision_scores(kind, base, feats) > threshold(kind, q / 100)).astype(float)


def tune() -> dict:
    b = load("tuning", "base")
    ids = sorted(b)
    base = [b[i] for i in ids]
    feats = load_feats("tuning")
    one = np.ones((1, len(ids)))
    runs = {a: [r[i] for i in ids] for a in UNGATED + ("prompt",) for r in [load("tuning", a)]}
    grid = {a: float(sel(one, base, r)[0]) for a, r in runs.items()}
    for kind in DECISIONS:
        for q in QS:
            f = flags("tuning", kind, q, base, feats)
            for a in UNGATED:
                grid[f"{kind}|q{q}|{a}"] = float(sel(one, base, compose(base, runs[a], f))[0])
    pick = {"ungated": max(UNGATED, key=grid.get), "prompt": "prompt"}
    for kind in DECISIONS:
        pick[kind] = max((k for k in grid if k.startswith(kind + "|")), key=grid.get)
    return {"grid": grid, "selected": pick, "aurocs": {k: float(torch.load(DIRS / f"{k}.pt")["auroc_tuning"]) for k in DECISIONS}}


def confirm(sel_cfg: dict, n_random: int, iters: int) -> dict:
    b = load("confirm", "base")
    ids = sorted(b)
    base = [b[i] for i in ids]
    feats = load_feats("confirm")
    n = len(ids)
    w = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=iters).astype(float)
    one = np.ones((1, n))
    tag = lambda a: a.replace(":", "").replace("|", "_")  # noqa: E731
    def get(a: str) -> list[dict]:
        rows = load("confirm", tag(a))
        return [rows[i] for i in ids]
    ref_arm = sel_cfg["probe"]
    ref_rows = get(ref_arm)
    ref_pt, ref_bs = float(sel(one, base, ref_rows)[0]), sel(w, base, ref_rows)
    out = {"n": n, "n_toxic": int(sum(r["state"] == TOXIC for r in base)), "n_clean": int(sum(r["state"] == CLEAN for r in base)),
           "arms": {}}
    for label, arm in sel_cfg.items():
        rows = get(arm)
        pt, bs = float(sel(one, base, rows)[0]), sel(w, base, rows)
        row = {"arm": arm, "sel": {"point": round(pt, 4), "ci": ci(bs)}, "vs_pgs": {
            "point": round(pt - ref_pt, 4), "ci": ci(bs - ref_bs),
            "p_two_sided": round(float(2 * min((bs - ref_bs <= 0).mean(), (bs - ref_bs >= 0).mean())), 4)}}
        if "|" in arm:
            kind, q, act = arm.split("|")
            f = flags("confirm", kind, int(q[1:]), base, feats)
            null = get(f"null|{arm}")
            rand = [get(f"rand{s}|{arm}") for s in range(n_random) if (OUT / "confirm" / f"{tag(f'rand{s}|{arm}')}__shard0.jsonl").exists()]
            r_pt = np.array([sel(one, base, r)[0] for r in rand]) if rand else np.array([np.nan])
            r_bs = np.mean([sel(w, base, r) for r in rand], axis=0) if rand else np.full(iters, np.nan)
            ov_pt, ov_bs = float(override_sel(one, base, f)[0]), override_sel(w, base, f)
            tb = load("tuning", "base")
            tids = sorted(tb)
            tbase, tact = [tb[i] for i in tids], load("tuning", act)
            ung = rates(tbase, [tact[i] for i in tids], flags("tuning", kind, int(q[1:]), tbase, load_feats("tuning")))
            pred = ung["fpr"] * (1 - ung["rho_c"]) - ung["tpr"] * (1 - ung["rho_t"])
            row.update({
                "null": {"point": round(float(sel(one, base, null)[0]), 4),
                         "identical": int(sum(x["text"] == y["text"] for x, y in zip(null, base)))},
                "random": {"n": len(rand), "mean": round(float(r_pt.mean()), 4), "p95": round(float(np.quantile(r_pt, 0.95)), 4),
                           "max": round(float(r_pt.max()), 4), "rank": int((r_pt >= pt).sum())},
                "vs_random": {"point": round(pt - float(r_pt.mean()), 4), "ci": ci(bs - r_bs),
                              "p_one_sided": round(float((bs - r_bs <= 0).mean()), 4)},
                "override": {"point": round(ov_pt, 4), "ci": ci(ov_bs)},
                "steer_minus_override": {"point": round(pt - ov_pt, 4), "ci": ci(bs - ov_bs),
                                         "p_one_sided": round(float((bs - ov_bs <= 0).mean()), 4)},
                "eq2_predicted": round(pred, 4), "eq2_agrees": bool(np.sign(pred) == np.sign(pt - ov_pt)),
                "rates_real": rates(base, rows, f), "auroc_confirm": auroc(
                    np.array(decision_scores(kind, base, feats)), np.array([r["state"] == TOXIC for r in base], float))})
        out["arms"][label] = row
        print(label, json.dumps(row)[:400], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("tune", "confirm"))
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--iters", type=int, default=10000)
    args = ap.parse_args()
    if args.stage == "tune":
        res = tune()
        write_json(OUT / "selection.json", res)
        print(json.dumps(res["selected"]), json.dumps(res["aurocs"]))
        for k, v in sorted(res["grid"].items(), key=lambda kv: -kv[1])[:15]:
            print(k, round(v, 3))
    else:
        res = confirm(json.loads((OUT / "selection.json").read_text())["selected"], args.n_random, args.iters)
        write_json(OUT / "confirm.json", res)
