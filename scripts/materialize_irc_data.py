import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from math_verify.errors import TimeoutException

from audit_irc_sources import load_math_author
from materialize_outcome_score_data import file_sha256, payload_sha256
from projection_transport_steering.outcome_score import (
    score_math_author_completion,
    score_math_completion,
)
from projection_transport_steering.splits import (
    allocate_groups,
    normalize_problem,
    problem_clusters,
    stable_group_id,
)

QUOTAS = {"fit": 1500, "response": 512, "selection": 256, "development": 256}
SALT = "irc-20260907"


def read_exposure(spec: dict, tokenizer) -> list[str]:
    path = Path(spec["path"])
    fields = set(spec["fields"])
    if spec["format"] == "parquet":
        values = pq.read_table(path, columns=sorted(fields)).to_pylist()
    elif spec["format"] == "csv":
        with path.open() as source:
            values = list(csv.DictReader(source))
    elif spec["format"] == "jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        values = json.loads(path.read_text())
    texts = []

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in fields and isinstance(child, str) and child.strip():
                    texts.append(child)
                elif key == "token_ids" and spec["format"] == "token_json":
                    text = tokenizer.decode(child, skip_special_tokens=True)
                    if text.strip():
                        texts.append(text)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(values)
    return texts


def partition_math(
    rows: list[dict], clusters: list[str], exposed: set[str], quotas: dict
) -> list[dict]:
    training = {group for row, group in zip(rows, clusters, strict=True) if row["split"] == "train"}
    result = []
    representatives = {}
    for row, group in zip(rows, clusters, strict=True):
        stage = "reserve"
        if not row["reference_eligible"]:
            stage = "excluded_reference"
        elif group in exposed:
            stage = "excluded_history"
        elif row["split"] == "test" and group in training:
            stage = "excluded_training_overlap"
        else:
            key = (row["split"], group)
            representatives[key] = min(representatives.get(key, row["row_id"]), row["row_id"])
        result.append({**row, "cluster_id": group, "allocation": stage})
    allocation = allocate_groups(
        (key[1] for key in representatives if key[0] == "train"), quotas, SALT
    )
    for row in result:
        if row["allocation"] != "reserve":
            continue
        key = (row["split"], row["cluster_id"])
        if row["row_id"] != representatives[key]:
            row["allocation"] = "duplicate"
        else:
            row["allocation"] = (
                allocation.get(row["cluster_id"], "reserve")
                if row["split"] == "train"
                else "sealed_test"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    extract, equivalent = load_math_author(root)
    boxed = runpy.run_path(str(root / ".external/math-author-source/modeling/dataset/util.py"))[
        "last_boxed_only_string"
    ]
    source_audit = json.loads((root / "research/irc-source-audit-v1.json").read_text())
    source_hashes = {}
    rows = []
    for name, expected in source_audit["source_files_sha256"].items():
        if not name.startswith(".external/math-eleuther/") or not name.endswith(".parquet"):
            continue
        path = root / name
        if file_sha256(path) != expected:
            raise ValueError(f"MATH source hash mismatch: {name}")
        source_hashes[name] = expected
        split = path.name.split("-")[0]
        for index, raw in enumerate(pq.read_table(path).to_pylist()):
            try:
                eligible = score_math_author_completion(
                    raw["solution"],
                    raw["solution"],
                    terminated=True,
                    extract_answer=extract,
                    equivalent=equivalent,
                )["correct"]
            except ValueError:
                eligible = False
            diagnostic_failure = None
            try:
                supported = score_math_completion(
                    raw["solution"], raw["solution"], terminated=True, extract_answer=boxed
                )["correct"]
            except (ValueError, RuntimeError, TimeoutError, TimeoutException) as error:
                supported = False
                diagnostic_failure = type(error).__name__
            rows.append(
                {
                    **raw,
                    "row_id": stable_group_id(expected, path.parent.name, split, raw),
                    "source_id": f"{split}:{path.parent.name}:{index}",
                    "split": split,
                    "source_path": name,
                    "source_index": index,
                    "reference_eligible": bool(eligible),
                    "sensitivity_supported": bool(supported),
                    "sensitivity_failure": diagnostic_failure,
                    "problem_sha256": hashlib.sha256(
                        normalize_problem(raw["problem"]).encode()
                    ).hexdigest(),
                    "solution_sha256": payload_sha256(raw["solution"]),
                }
            )
    reference_counts = {
        split: dict(Counter(row["reference_eligible"] for row in rows if row["split"] == split))
        for split in ("train", "test")
    }
    print(json.dumps({"stage": "references", "counts": reference_counts}), flush=True)
    if any(
        counts.get(False, 0) / sum(counts.values()) >= 0.01 for counts in reference_counts.values()
    ):
        raise ValueError("reference exclusion ceiling exceeded")
    spec = json.loads(args.exposure_spec.read_text())
    for name, expected in spec.get("tokenizer_files_sha256", {}).items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"tokenizer hash mismatch: {name}")
    tokenizer = None
    if any(source["format"] == "token_json" for source in spec["sources"]):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"], local_files_only=True)
    exposure = []
    exposure_sources = []
    for source in spec["sources"]:
        path = Path(source["path"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError(f"exposure source hash mismatch: {path}")
        texts = read_exposure(source, tokenizer)
        exposure_sources.append({**source, "texts": len(texts)})
        exposure.extend(
            {"source_path": str(path), "index": i, "problem": text} for i, text in enumerate(texts)
        )
    print(
        json.dumps({"stage": "clustering", "math": len(rows), "history": len(exposure)}), flush=True
    )
    clusters = problem_clusters([row["problem"] for row in rows + exposure], query_count=len(rows))
    result = partition_math(rows, clusters[: len(rows)], set(clusters[len(rows) :]), QUOTAS)
    counts = dict(Counter(row["allocation"] for row in result))
    print(json.dumps({"stage": "allocation", "counts": counts}), flush=True)
    if counts.get("sealed_test", 0) < 3000:
        raise ValueError("fewer than 3000 eligible test clusters")
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent))
    try:
        files = {}
        for stage in (*QUOTAS, "reserve", "sealed_test"):
            selected = sorted(
                (row for row in result if row["allocation"] == stage),
                key=lambda row: row["cluster_id"],
            )
            if stage == "sealed_test":
                selected = [
                    {key: value for key, value in row.items() if key not in {"problem", "solution"}}
                    for row in selected
                ]
            packet = {"schema": "irc-allocation-v1", "allocation": stage, "rows": selected}
            packet["content_sha256"] = payload_sha256(packet)
            path = temporary / f"{stage}.json"
            path.write_text(json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            files[path.name] = file_sha256(path)
        response = [row for row in result if row["allocation"] == "response"]
        pilot = list(allocate_groups((row["cluster_id"] for row in response), {"pilot": 128}, SALT))
        manifest = {
            "schema": "irc-data-manifest-v1",
            "model_outcomes_inspected": 0,
            "measurement_amendment_sha256": file_sha256(
                root / "research/irc-measurement-and-allocation-v1.md"
            ),
            "exposure_spec_sha256": file_sha256(args.exposure_spec),
            "exposure_scope": spec["scope"],
            "sources_sha256": source_hashes,
            "exposure_sources": exposure_sources,
            "source_code_sha256": {
                name: file_sha256(root / name)
                for name in (
                    "scripts/materialize_irc_data.py",
                    "scripts/audit_irc_sources.py",
                    "src/projection_transport_steering/outcome_score.py",
                    "src/projection_transport_steering/splits.py",
                )
            },
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "math-verify",
                    "latex2sympy2-extended",
                    "scikit-learn",
                    "scipy",
                    "numpy",
                    "pyarrow",
                )
            },
            "reference_counts": reference_counts,
            "allocation_counts": counts,
            "pilot_cluster_ids": pilot,
            "packet_files_sha256": files,
            "math_rows": [
                {key: value for key, value in row.items() if key not in {"problem", "solution"}}
                for row in result
            ],
            "history_rows": [
                {
                    "source_path": row["source_path"],
                    "index": row["index"],
                    "cluster_id": group,
                    "problem_sha256": hashlib.sha256(
                        normalize_problem(row["problem"]).encode()
                    ).hexdigest(),
                }
                for row, group in zip(exposure, clusters[len(rows) :], strict=True)
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "ready_for_fitting": True,
            "scientific_engine_or_behavioral_gate_passed": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": file_sha256(args.output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
