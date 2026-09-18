import importlib.util
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "analyze_cacheback_sentinel.py"
    spec = importlib.util.spec_from_file_location("analyze_cacheback_sentinel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_method_reports_effect_retention_concentration_and_calibration():
    module = load_script()
    rows = [
        {
            "concept": "a",
            "methods": {
                "cacheback_h0": {
                    "cumulative_forward_kl_mean": 10.0,
                    "immediate_semantic_effect_mean": 0.05,
                },
                "cacheback_h8": {
                    "cumulative_forward_kl_mean": 5.0,
                    "immediate_semantic_effect_mean": 0.049,
                    "offset_forward_kl_mean": [4.0, 1.0],
                    "predicted_cumulative_quadratic_cost": 4.8,
                },
            },
        },
        {
            "concept": "b",
            "methods": {
                "cacheback_h0": {
                    "cumulative_forward_kl_mean": 4.0,
                    "immediate_semantic_effect_mean": 0.05,
                },
                "cacheback_h8": {
                    "cumulative_forward_kl_mean": 2.0,
                    "immediate_semantic_effect_mean": 0.05,
                    "offset_forward_kl_mean": [1.0, 1.0],
                    "predicted_cumulative_quadratic_cost": 2.0,
                },
            },
        },
    ]

    result = module.summarize_method(rows, "cacheback_h8", bootstrap_samples=100)

    assert result["cumulative_kl_reduction"]["median"] == pytest.approx(0.5)
    assert result["immediate_effect_retention"]["minimum"] == pytest.approx(0.98)
    assert result["offset_concentration"]["maximum"] == pytest.approx(0.8)
    assert result["predicted_to_full_trajectory_relative_error"]["maximum"] == pytest.approx(
        0.04
    )


def test_shard_directories_exclude_log_files(tmp_path):
    module = load_script()
    (tmp_path / "shard_00").mkdir()
    (tmp_path / "shard_00.log").write_text("log")

    result = module.shard_directories(tmp_path)

    assert result == [tmp_path / "shard_00"]
