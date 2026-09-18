from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with at least two values")
    return array


@dataclass(frozen=True)
class EmpiricalQuantileTransport:
    source_knots: FloatArray
    target_knots: FloatArray
    left: float
    right: float

    @classmethod
    def fit(
        cls,
        source: ArrayLike,
        target: ArrayLike,
        clip: float,
    ) -> "EmpiricalQuantileTransport":
        source_array = _finite_vector(source, "source")
        target_array = _finite_vector(target, "target")
        if not 0.0 < clip < 0.5:
            raise ValueError("clip must be between zero and one half")
        source_knots, counts = np.unique(source_array, return_counts=True)
        midranks = (np.cumsum(counts) - 0.5 * counts) / source_array.size
        probabilities = np.clip(midranks, clip, 1.0 - clip)
        target_knots = np.quantile(target_array, probabilities, method="linear")
        boundaries = np.quantile(target_array, [clip, 1.0 - clip], method="linear")
        return cls(source_knots, target_knots, float(boundaries[0]), float(boundaries[1]))

    def transform(self, values: ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("values must be finite")
        return np.interp(array, self.source_knots, self.target_knots, self.left, self.right)


def minimum_metric_update(
    states: ArrayLike,
    basis: ArrayLike,
    targets: ArrayLike,
    metric: ArrayLike,
) -> FloatArray:
    state_array = np.asarray(states, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    metric_array = np.asarray(metric, dtype=np.float64)
    if state_array.ndim != 2 or basis_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("states, basis, and targets must be matrices")
    rows, dimension = state_array.shape
    if basis_array.shape[0] != dimension or target_array.shape != (rows, basis_array.shape[1]):
        raise ValueError("state, basis, and target shapes are incompatible")
    if metric_array.shape != (dimension, dimension):
        raise ValueError("metric shape is incompatible with states")
    if not all(np.all(np.isfinite(x)) for x in (state_array, basis_array, target_array, metric_array)):
        raise ValueError("inputs must be finite")
    if not np.allclose(metric_array, metric_array.T):
        raise ValueError("metric must be symmetric positive definite")
    try:
        np.linalg.cholesky(metric_array)
    except np.linalg.LinAlgError as error:
        raise ValueError("metric must be symmetric positive definite") from error
    if np.linalg.matrix_rank(basis_array) != basis_array.shape[1]:
        raise ValueError("basis must have full column rank")
    inverse_metric_basis = np.linalg.solve(metric_array, basis_array)
    gram = basis_array.T @ inverse_metric_basis
    residual = target_array - state_array @ basis_array
    coefficients = np.linalg.solve(gram, residual.T).T
    return coefficients @ inverse_metric_basis.T
