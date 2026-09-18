import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

SEED = 20260907
SENTINEL_IDS = (1, 5, 16, 24, 29, 33, 34, 48, 58, 59, 60, 73)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized(value: object) -> str:
    return " ".join(str(value).split())


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(frame: pd.DataFrame, count: int, allocation: str) -> list[dict[str, object]]:
    records = []
    for row in frame.to_dict("records"):
        record = {
            "input": normalized(row["input"]),
            "output": normalized(row["output"]),
            "source_category": normalized(row["category"]),
            "source_concept_id": int(row["concept_id"]),
        }
        record["source_id"] = payload_sha256(record)
        records.append(record)
    ordered = sorted(records, key=lambda row: payload_sha256([SEED, allocation, row["source_id"]]))
    if len(ordered) < count or len({row["source_id"] for row in ordered}) != len(ordered):
        raise ValueError("source allocation is incomplete or duplicated")
    return ordered[:count]


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    train = pd.read_parquet(args.train)
    latent = pd.read_parquet(args.latent)
    metadata_rows = [json.loads(line) for line in args.metadata.read_text().splitlines()]
    metadata = {int(row["concept_id"]): row for row in metadata_rows}
    concepts = {}
    for concept_id in SENTINEL_IDS:
        construction = train[
            (train["concept_id"] == concept_id) & (train["category"] == "positive")
        ]
        evaluation = latent[
            (latent["concept_id"] == concept_id) & (latent["category"] == "positive")
        ]
        if concept_id not in metadata:
            raise ValueError("sentinel concept metadata is missing")
        concepts[str(concept_id)] = {
            "concept": normalized(metadata[concept_id]["concept"]),
            "construction": selected_rows(construction, 8, f"construction-{concept_id}"),
            "evaluation": selected_rows(evaluation, 8, f"evaluation-{concept_id}"),
        }
    neutral = selected_rows(train[train["category"] == "negative"], 16, "neutral")
    payload = {
        "allocation_seed": SEED,
        "benchmark_freeze": "research/benchmark-freeze-semantic-outcome-v1.md",
        "concepts": concepts,
        "neutral": neutral,
        "sources": {
            "latent_sha256": file_sha256(args.latent),
            "metadata_sha256": file_sha256(args.metadata),
            "train_sha256": file_sha256(args.train),
        },
        "status": "frozen_before_model_output",
    }
    payload["content_sha256"] = payload_sha256(payload)
    write_new(args.output, payload)


if __name__ == "__main__":
    main()
