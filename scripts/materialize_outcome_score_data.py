import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from projection_transport_steering.outcome_score import partition_aime_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_sha = file_sha256(args.historical)
    rows = pq.read_table(args.historical).to_pylist()
    packets = partition_aime_rows(rows, source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent))
    try:
        for stage, stage_rows in packets.items():
            packet = {
                "allocation": stage,
                "benchmark_freeze": "research/benchmark-freeze-outcome-score-transport-v1.md",
                "rows": stage_rows,
                "source_sha256": source_sha,
                "status": "frozen_before_model_output",
            }
            packet["content_sha256"] = payload_sha256(packet)
            (temporary / f"{stage}.json").write_text(
                json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False)
            )
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
