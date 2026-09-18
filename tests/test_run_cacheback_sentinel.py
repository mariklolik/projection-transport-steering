import importlib.util
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_cacheback_sentinel.py"
    spec = importlib.util.spec_from_file_location("run_cacheback_sentinel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_exposes_frozen_sentinel_defaults():
    module = load_script()

    args = module.parse_args(
        [
            "--model",
            "/models/gpt2",
            "--data",
            "/data/cacheback.json",
            "--output",
            "/results/shard",
            "--shard-index",
            "2",
            "--shard-count",
            "4",
        ]
    )

    assert args.layer == 6
    assert args.horizons == (0, 2, 4, 8)
    assert args.trajectories == 8
    assert args.target == 0.05
