from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() > 0:
            total += mask.sum() / n * abs(corr[mask].mean() - conf[mask].mean())
    return total


def overconfidence_gap(confidences: list[float], correct: list[bool]) -> float:

    return float(np.mean(confidences) - np.mean(np.asarray(correct, dtype=float)))


def bootstrap_ci(results: list, metric_fn: Callable[[list], float],
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:

    rng = np.random.default_rng(seed)
    n = len(results)
    point = metric_fn(results)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric_fn([results[i] for i in idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


if __name__ == "__main__":
    conf = [0.7] * 100
    correct = [True] * 70 + [False] * 30
    assert ece(conf, correct) < 1e-9, ece(conf, correct)

    conf = [1.0] * 100
    correct = [True] * 50 + [False] * 50
    assert abs(ece(conf, correct) - 0.5) < 1e-9
    assert abs(overconfidence_gap(conf, correct) - 0.5) < 1e-9

    recs = [{"c": 1.0, "ok": i < 50} for i in range(100)]
    point, lo, hi = bootstrap_ci(recs, lambda rs: ece([r["c"] for r in rs], [r["ok"] for r in rs]))
    assert lo <= point <= hi and hi > lo
    print(f"general.metrics self-tests passed  (bootstrap ECE={point:.2f} [{lo:.2f}, {hi:.2f}])")
