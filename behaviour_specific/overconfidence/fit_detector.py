from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from behaviour_specific.overconfidence.analyze_pooled import OCW, CR
from behaviour_specific.overconfidence.analyze_steering import collect
from behaviour_specific.overconfidence.features_caa import DIRECTIONS_DIR
from behaviour_specific.overconfidence.gate_law import auroc
from behaviour_specific.overconfidence.label_pool import LAYERS, PREFIX_LAYERS, PREFIX_T
from general.paths import RESULTS_DIR
from general.storage import write_json

CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def load_blob(dirs: list[str], kind: str) -> dict[str, torch.Tensor]:
    out = {}
    for d in dirs:
        for f in sorted((RESULTS_DIR / d).glob(f"{kind}__shard*.pt")):
            blob = torch.load(f)
            out.update({i: x for i, x in zip(blob["ids"], blob["X"].float())})
    return out


def labelled(dirs: list[str], rollout_dirs: list[str], readout: str, kind: str):
    feats = load_blob(dirs, kind)
    states = {}
    for d in rollout_dirs:
        states.update({r["id"]: r["state"] for r in collect(RESULTS_DIR / d / "rollouts").get(f"baseline_{readout}", [])})
    ids = [i for i in feats if i in states]
    s = np.array([states[i] for i in ids])
    return torch.stack([feats[i] for i in ids]).numpy(), (s == OCW).astype(float), np.isin(s, [OCW, CR])


def fit(Z: np.ndarray, y: np.ndarray, m: np.ndarray, C: float) -> tuple[np.ndarray, float]:
    mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000).fit((Z[m] - mu) / sd, y[m])
    return clf.coef_[0] / sd, float(clf.intercept_[0] - (mu / sd) @ clf.coef_[0])


def select(views: dict, train, sel, held) -> dict:
    grid = []
    for key, view in views.items():
        for C in CS:
            w, b = fit(view(train[0]), train[1], train[2], C)
            grid.append((auroc((view(sel[0]) @ w + b)[sel[2]], sel[1][sel[2]]), key, C))
            print(f"{key} C {C:g}: select AUROC {grid[-1][0]:.3f}", flush=True)
    a_sel, key, C = max(grid, key=lambda g: g[0])
    w, b = fit(views[key](train[0]), train[1], train[2], C)
    s_train = views[key](train[0]) @ w + b
    return {"key": key, "C": C, "w": w, "b": b, "s_train": s_train, "auroc_select": a_sel,
            "auroc_train": auroc(s_train[train[2]], train[1][train[2]]),
            "auroc_heldout": auroc((views[key](held[0]) @ w + b)[held[2]], held[1][held[2]])}


def save(name: str, res: dict, layer, layers, extra: dict) -> None:
    torch.save({"layer": layer, "layers": layers, "w": torch.tensor(res["w"]), "b": res["b"],
                "score_quantiles": torch.tensor(np.quantile(res["s_train"], np.linspace(0.01, 0.99, 41))), **extra},
               DIRECTIONS_DIR / f"{name}.pt")
    report = {k: v for k, v in res.items() if k not in ("w", "b", "s_train")}
    write_json(RESULTS_DIR / "v4_pooled" / f"{name}.json", report)
    print(name, report, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="v4_detector,v4_extraction")
    ap.add_argument("--select", default="v4_tuning")
    ap.add_argument("--heldout", default="v4_eval7,v4_eval11,v4_eval23,v4_eval31,v4_eval47")
    ap.add_argument("--readout", default="m5")
    ap.add_argument("--name", default="detector_m5")
    ap.add_argument("--kind", default="feats", choices=("feats", "prefix", "prompt", "cast"))
    ap.add_argument("--ood")
    args = ap.parse_args()

    kind = args.kind
    sets = [labelled(d.split(","), d.split(","), args.readout, "prompt" if kind == "cast" else kind)
            for d in (args.train, args.select, args.heldout)]
    if args.ood:
        Xo = torch.stack(list(load_blob([args.ood], "prompt" if kind == "cast" else kind).values())).numpy()
        X0, y0, m0 = sets[0]
        sets[0] = (np.concatenate([X0, Xo]), np.concatenate([y0, np.zeros(len(Xo))]),
                   np.concatenate([m0, np.ones(len(Xo), bool)]))
    print(f"train {sets[0][2].sum()} (ocw {int(sets[0][1][sets[0][2]].sum())}) select {sets[1][2].sum()} "
          f"heldout {sets[2][2].sum()}", flush=True)
    if kind == "feats":
        views = {L: (lambda X, k=LAYERS.index(L): X[:, k]) for L in LAYERS}
        res = select(views, *sets)
        save(args.name, res, res["key"], LAYERS, {})
    elif kind == "cast":
        for L in (14, 16):
            Z = [x[:, 1, LAYERS.index(L)] for x, _, _ in sets]
            (Xt, yt, mt) = sets[0]
            c = Z[0][mt & (yt == 1)].mean(0) - Z[0][mt & (yt == 0)].mean(0)
            cos = [z @ c / (np.linalg.norm(z, axis=1) * np.linalg.norm(c)) for z in Z]
            rep = {f"auroc_{n}": auroc(cs[st[2]], st[1][st[2]]) for n, cs, st in zip(("train", "select", "heldout"), cos, sets)}
            torch.save({"layer": L, "c": torch.tensor(c), "score_quantiles": torch.tensor(np.quantile(cos[0], np.linspace(0.01, 0.99, 41)))},
                       DIRECTIONS_DIR / f"cast_condition_L{L}.pt")
            write_json(RESULTS_DIR / "v4_pooled" / f"cast_condition_L{L}.json", rep)
            print(f"cast_condition_L{L}", rep, flush=True)
    elif kind == "prompt":
        views = {(pos, L): (lambda X, i=i, k=k: X[:, i, k]) for i, pos in enumerate(("last", "mean"))
                 for k, L in enumerate(LAYERS)}
        res = select(views, *sets)
        save(f"{args.name}_prompt", res, res["key"][1], LAYERS, {"pos": res["key"][0]})
    else:
        for ti, t in enumerate(PREFIX_T):
            views = {L: (lambda X, k=k, ti=ti: X[:, k, ti]) for k, L in enumerate(PREFIX_LAYERS)}
            res = select(views, *sets)
            save(f"{args.name}_prefix_t{t}", res, res["key"], PREFIX_LAYERS, {"t": t})
