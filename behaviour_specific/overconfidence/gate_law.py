from __future__ import annotations


import numpy as np

from behaviour_specific.overconfidence.analyze_pooled import OCW, CR


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    r = s.argsort().argsort() + 1.0
    npos, nneg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def law(base, after, scores, taus) -> dict:
    ocw = np.array([r["state"] == OCW for r in base])
    cr = np.array([r["state"] == CR for r in base])
    left = np.array([r["state"] != OCW for r in after])
    lost = np.array([r["state"] != CR for r in after])
    rho_o, rho_c = left[ocw].mean(), lost[cr].mean()
    rows = []
    for tau in taus:
        f = scores > tau
        tpr, fpr = f[ocw].mean(), f[cr].mean()
        sim = (f & left)[ocw].mean() - (f & lost)[cr].mean()
        rows.append({"tau": float(tau), "fire": float(f.mean()), "tpr": float(tpr), "fpr": float(fpr),
                     "sel_sim": float(sim), "sel_law": float(tpr * rho_o - fpr * rho_c),
                     "rho_o_flag": float(left[ocw & f].mean()) if (ocw & f).any() else None,
                     "rho_c_flag": float(lost[cr & f].mean()) if (cr & f).any() else None})
    return {"rho_o": float(rho_o), "rho_c": float(rho_c), "curve": rows}
