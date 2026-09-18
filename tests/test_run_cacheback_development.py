import importlib.util
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_cacheback_development.py"
    spec = importlib.util.spec_from_file_location("run_cacheback_development", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_exposes_the_frozen_development_grid():
    module = load_script()

    args = module.parse_args(
        [
            "--model",
            "/models/gpt2",
            "--data",
            "/data/cacheback.json",
            "--output",
            "/results/cacheback",
            "--shard-index",
            "3",
        ]
    )

    assert module.LAYERS == (3, 6, 9)
    assert module.REGULARIZATION_MULTIPLIERS == (0.01, 0.1, 1.0)
    assert module.DOSES == (0.125, 0.25, 0.5)
    assert module.CPU_THREADS == 2
    assert module.INTEROP_THREADS == 1
    assert args.shard_count == 8
    assert args.shard_index == 3
