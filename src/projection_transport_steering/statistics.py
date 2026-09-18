import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from collections.abc import Callable


def _bca_limits(
    estimate: float,
    bootstrap: np.ndarray,
    jackknife: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    probability = np.clip(
        np.mean(bootstrap < estimate),
        1.0 / len(bootstrap),
        1.0 - 1.0 / len(bootstrap),
    )
    bias = norm.ppf(probability)
    centered = np.mean(jackknife) - jackknife
    denominator = 6.0 * np.sum(centered**2) ** 1.5
    acceleration = (
        0.0
        if denominator == 0.0
        else float(np.sum(centered**3) / denominator)
    )
    normal_quantiles = norm.ppf([alpha / 2.0, 1.0 - alpha / 2.0])
    adjusted = norm.cdf(
        bias
        + (bias + normal_quantiles)
        / (1.0 - acceleration * (bias + normal_quantiles))
    )
    lower, upper = np.quantile(bootstrap, adjusted)
    return float(lower), float(upper)


def resampled_bca_interval(
    size: int,
    statistic: Callable[[np.ndarray], float],
    resamples: int = 10_000,
    seed: int = 20260905,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if size < 2 or resamples < 2 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid BCa interval configuration")
    indices = np.arange(size)
    estimate = statistic(indices)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [statistic(rng.integers(0, size, size=size)) for _ in range(resamples)]
    )
    jackknife = np.asarray(
        [statistic(np.delete(indices, index)) for index in range(size)]
    )
    if not np.all(np.isfinite([estimate, *bootstrap, *jackknife])):
        raise ValueError("statistic must return finite values")
    if np.all(bootstrap == estimate):
        return estimate, estimate, estimate
    lower, upper = _bca_limits(estimate, bootstrap, jackknife, alpha)
    return estimate, lower, upper


def paired_bca_interval(
    differences: ArrayLike,
    resamples: int = 10_000,
    seed: int = 20260905,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("differences must be a finite vector with at least two values")
    estimate = float(np.mean(values))
    if np.all(values == values[0]):
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples)
    batch_size = min(1_000, resamples)
    for offset in range(0, resamples, batch_size):
        size = min(batch_size, resamples - offset)
        indices = rng.integers(0, values.size, size=(size, values.size))
        bootstrap_means[offset : offset + size] = np.mean(values[indices], axis=1)
    jackknife = (np.sum(values) - values) / (values.size - 1)
    lower, upper = _bca_limits(estimate, bootstrap_means, jackknife, alpha)
    return estimate, lower, upper
