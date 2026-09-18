import importlib.util
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_cache_path_smoke.py"
    spec = importlib.util.spec_from_file_location("run_cache_path_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_exposes_the_frozen_cache_path_smoke():
    module = load_script()

    args = module.parse_args(
        [
            "--model",
            "/models/gpt2",
            "--data",
            "/data/cacheback.json",
            "--output",
            "/results/cache-path",
        ]
    )

    assert args.layer == 6
    assert args.tokens == 8
    assert args.trajectories == 8
    assert args.target == 0.05
    assert not hasattr(args, "action_norm")
