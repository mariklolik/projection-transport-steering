from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"
DIRECTIONS_DIR = Path(os.environ.get("PTS_DIRECTIONS", ROOT / "behaviour_specific" / "overconfidence" / "directions"))
