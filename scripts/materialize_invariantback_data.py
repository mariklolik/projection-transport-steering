import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from projection_transport_steering.invariantback_data import (
    allocate_groups,
    mnli_groups,
    normbank_groups,
    sc101_groups,
)

SEED = 20260906
SIZES = {"construction": 80, "sentinel": 8, "development": 24, "pilot": 48}
SOURCES = {
    "normbank": {
        "dataset": "SALT-NLP/NormBank",
        "revision": "dacdc9a905d509d9d1aca1c9a031e4927d8ab815",
        "file": "NormBank.csv",
    },
    "mnli": {
        "dataset": "nyu-mll/multi_nli",
        "revision": "da70db2af9d09693783c3320c4249840212ee221",
        "file": "data/train-00000-of-00001.parquet",
    },
    "sc101": {
        "dataset": "wassname/social_chemistry_101",
        "revision": "a7869978d5d441d89327067d8ff543d6dcc3fc32",
        "file": "social_chem_101.parquet",
        "source_split": "train",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normbank", type=Path, required=True)
    parser.add_argument("--mnli", type=Path, required=True)
    parser.add_argument("--sc101", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        temporary = Path(handle.name)
    os.replace(temporary, path)


def source_record(path: Path, name: str, row_count: int, eligible_count: int) -> dict[str, Any]:
    return {
        **SOURCES[name],
        "eligible_group_count": eligible_count,
        "local_file_sha256": file_sha256(path),
        "row_count_after_source_filter": row_count,
    }


def main() -> None:
    args = parse_args()
    normbank_rows = pd.read_csv(args.normbank).to_dict("records")
    mnli_rows = pd.read_parquet(args.mnli).to_dict("records")
    sc101_frame = pd.read_parquet(args.sc101)
    sc101_rows = sc101_frame.loc[sc101_frame["split"] == "train"].to_dict("records")
    eligible = {
        "normbank": normbank_groups(normbank_rows),
        "mnli": mnli_groups(mnli_rows),
        "sc101": sc101_groups(sc101_rows, SEED),
    }
    allocations = {
        dataset: allocate_groups(groups, SIZES, SEED)
        for dataset, groups in eligible.items()
    }
    pilot = {
        "allocation_seed": SEED,
        "datasets": {dataset: split["pilot"] for dataset, split in allocations.items()},
        "status": "sealed_no_model_output",
    }
    write_new(args.pilot_output, pilot)
    pilot_file_sha256 = file_sha256(args.pilot_output)
    sources = {
        "normbank": source_record(
            args.normbank, "normbank", len(normbank_rows), len(eligible["normbank"])
        ),
        "mnli": source_record(args.mnli, "mnli", len(mnli_rows), len(eligible["mnli"])),
        "sc101": source_record(
            args.sc101, "sc101", len(sc101_rows), len(eligible["sc101"])
        ),
    }
    public = {
        "allocation_seed": SEED,
        "allocation_sizes_per_dataset": SIZES,
        "datasets": {
            dataset: {name: groups for name, groups in split.items() if name != "pilot"}
            for dataset, split in allocations.items()
        },
        "freeze": "research/benchmark-freeze-invariantback-v2.md",
        "pilot_file_sha256": pilot_file_sha256,
        "sources": sources,
        "status": "frozen_before_model_output",
    }
    public["content_sha256"] = payload_sha256(public)
    write_new(args.public_output, public)


if __name__ == "__main__":
    main()
