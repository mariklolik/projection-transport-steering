# Canonical filesystem locations, computed from the repo root so everything
# works regardless of the current working directory.

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]     # repo root (general/ lives directly under it)
DATA_DIR = ROOT / "data_cache"                 # one-time dataset downloads (jsonl)
RESULTS_DIR = ROOT / "results"                 # run outputs: rollouts + summaries
