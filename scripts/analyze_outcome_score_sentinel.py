import argparse
import hashlib
import json
from pathlib import Path

import torch
from run_outcome_score_sentinel_shard import file_sha256

from projection_transport_steering.outcome_score_sentinel import (
    summarize_sentinel,
    summarize_termination,
)
from projection_transport_steering.outcome_score_source import summarize_source_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("basis_completion", "class_mix", "fit", "termination"),
        default="class_mix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    receipts = []
    listing = []
    trace_rows = 0
    directories = sorted(path for path in args.input.glob("shard_*") if path.is_dir())
    if len(directories) != 8:
        raise RuntimeError("sentinel requires exactly eight shard directories")
    for directory in directories:
        rows_path = directory / "rows.json"
        traces_path = directory / "traces.pt"
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["rows_sha256"] != file_sha256(rows_path):
            raise RuntimeError("row hash mismatch")
        if receipt["traces_sha256"] != file_sha256(traces_path):
            raise RuntimeError("trace hash mismatch")
        shard_rows = json.loads(rows_path.read_text())
        traces = torch.load(traces_path, weights_only=True)
        if set(traces) != {row["generation_id"] for row in shard_rows}:
            raise RuntimeError("trace keys differ from row keys")
        for row in shard_rows:
            trace = traces[row["generation_id"]]
            if (
                trace.ndim != 2
                or len(trace) != row["trace_length"]
                or not torch.all(torch.isfinite(trace))
            ):
                raise RuntimeError("invalid activation trace")
        trace_rows += len(traces)
        rows.extend(shard_rows)
        receipts.append(receipt)
        for path in (receipt_path, rows_path, traces_path):
            listing.append({"path": str(path.relative_to(args.input)), "sha256": file_sha256(path)})
    if args.stage == "termination":
        summary = summarize_termination(rows, receipts)
    elif args.stage == "class_mix":
        summary = summarize_sentinel(rows, receipts)
    else:
        summary = summarize_source_stage(rows, receipts, args.stage)
    encoded_listing = json.dumps(listing, sort_keys=True, separators=(",", ":")).encode()
    summary.update(
        {
            "packet_listing": listing,
            "packet_listing_sha256": hashlib.sha256(encoded_listing).hexdigest(),
            "trace_rows": trace_rows,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
