import numpy as np

from projection_transport_steering.statistics import paired_bca_interval, resampled_bca_interval


def test_paired_bca_interval_is_deterministic():
    differences = np.linspace(-0.2, 0.4, 50)

    first = paired_bca_interval(differences, resamples=1000, seed=7)
    second = paired_bca_interval(differences, resamples=1000, seed=7)

    assert first == second
    assert first[1] < first[0] < first[2]


def test_paired_bca_interval_handles_constant_effect():
    estimate, lower, upper = paired_bca_interval(np.full(20, 0.25), resamples=1000, seed=7)

    assert estimate == 0.25
    assert lower == 0.25
    assert upper == 0.25


def test_paired_bca_interval_expands_for_smaller_alpha():
    differences = np.linspace(-0.2, 0.4, 50)

    standard = paired_bca_interval(differences, resamples=2000, seed=7)
    adjusted = paired_bca_interval(differences, resamples=2000, seed=7, alpha=0.01)

    assert adjusted[1] < standard[1]
    assert adjusted[2] > standard[2]


def test_resampled_bca_interval_recomputes_nonlinear_statistic():
    values = np.arange(1.0, 7.0)

    result = resampled_bca_interval(
        len(values),
        lambda indices: float(np.median(values[indices])),
        resamples=1000,
        seed=7,
    )

    assert result == resampled_bca_interval(
        len(values),
        lambda indices: float(np.median(values[indices])),
        resamples=1000,
        seed=7,
    )
    assert result[0] == 3.5
    assert result[1] < result[0] < result[2]
