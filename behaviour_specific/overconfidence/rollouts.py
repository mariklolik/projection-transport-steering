from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from general.paths import RESULTS_DIR
from general.storage import read_jsonl

STEER_DIR = RESULTS_DIR / "steering"


def collect(rollouts_dir: Path) -> dict[str, list[dict]]:

    pooled: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(rollouts_dir.glob("*__shard*.jsonl")):
        tag = f.stem.rsplit("__shard", 1)[0]
        pooled[tag].extend(read_jsonl(f))
    return dict(pooled)
