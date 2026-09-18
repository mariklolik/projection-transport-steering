import argparse
import ast
import importlib.metadata
import json
import runpy
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from materialize_outcome_score_data import file_sha256, payload_sha256
from projection_transport_steering.outcome_score import score_math_completion


def load_math_author(root: Path):
    source = root / ".external/math-author-source/modeling"
    for name, expected in {
        "dataset/util.py": "9183a9ea7bce3c126ac33f993d21bf85cffbdb7bab10088216d286699c9e084a",
        "evaluate_gpt3.py": "03d3388de66ac897be653b709df67711e400e56154a879f91206572ca6c83389",
        "math_equivalence.py": "c4101b1f51a2bb65665194aecd9f761d668c0c248a0350f196c2950e87488527",
    }.items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"author source hash mismatch: {name}")
    extract = runpy.run_path(str(source / "dataset/util.py"))["last_boxed_only_string"]
    equivalent = runpy.run_path(str(source / "math_equivalence.py"))["is_equiv"]
    path = source / "evaluate_gpt3.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "remove_boxed"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return lambda text: namespace["remove_boxed"](extract(text)), equivalent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    snapshots = {
        "math-author-linked": (
            "qwedsacf/competition_math",
            "e839825f9ec5c6cfa585c654a59610969ec13993",
        ),
        "math-eleuther": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"),
        "math-eleuther-original-loader": (
            "EleutherAI/hendrycks_math",
            "3730e0d9543219af79a2ddf93da274bb55d7fc27",
        ),
        "irc-qwen3-8b-metadata": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
        "irc-deepseek-llama8b-metadata": (
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "6a6f4aa4197940add57724a7707d069478df56b1",
        ),
    }
    files = {}
    for directory, (_, revision) in snapshots.items():
        source = root / ".external" / directory
        for path in sorted(source.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            relative = path.relative_to(source)
            metadata = source / ".cache/huggingface/download" / f"{relative}.metadata"
            if metadata.read_text().splitlines()[0] != revision:
                raise ValueError(f"snapshot revision mismatch: {path}")
            files[str(path.relative_to(root))] = file_sha256(path)
    repositories = {}
    for directory, expected in {
        "math-author-source": "985bdc1696e88e8643f081a0ff4719da39f2ae2a",
        "pyreft": "dafd0995a366d7b47160a337dcc388eda7431821",
    }.items():
        path = root / ".external" / directory
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != expected:
            raise ValueError(f"source revision mismatch: {path}")
        subprocess.run(
            ["git", "-C", str(path), "diff", "--exit-code", "HEAD"], check=True, capture_output=True
        )
        repositories[directory] = revision
    extractor_source = root / ".external/math-author-source/modeling/dataset/util.py"
    extract_answer = runpy.run_path(str(extractor_source))["last_boxed_only_string"]
    for relative in [
        ".external/math-author-source/modeling/dataset/util.py",
        ".external/math-author-source/modeling/math_equivalence.py",
        ".external/pyreft/pyreft/interventions.py",
        "scripts/audit_irc_sources.py",
        "src/projection_transport_steering/outcome_score.py",
        "src/projection_transport_steering/torch_runtime.py",
        "tests/test_math_outcomes.py",
        "tests/test_reft_conformance.py",
        "pyproject.toml",
        "uv.lock",
    ]:
        files[relative] = file_sha256(root / relative)
    split_rows = {"train": [], "test": []}
    training = []
    per_subject = {}
    for path in sorted((root / ".external/math-eleuther").glob("*/*.parquet")):
        split = path.name.split("-")[0]
        rows = pq.read_table(path).to_pylist()
        split_rows[split].extend(rows)
        per_subject.setdefault(path.parent.name, {})[split] = len(rows)
        if split == "train":
            training.extend((f"{path.parent.name}:{index}", row) for index, row in enumerate(rows))
    merged_path = next((root / ".external/math-author-linked/data").glob("*.parquet"))
    merged = pq.read_table(merged_path).to_pylist()
    keys = ("problem", "level", "type", "solution")
    counters = {
        split: Counter(payload_sha256({key: row[key] for key in keys}) for row in rows)
        for split, rows in split_rows.items()
    }
    author_counter = Counter(payload_sha256({key: row[key] for key in keys}) for row in merged)
    union = counters["train"] + counters["test"]
    failures = []
    self_correct = 0
    for row_id, row in training:
        try:
            score = score_math_completion(
                row["solution"], row["solution"], terminated=True, extract_answer=extract_answer
            )
            if score["correct"]:
                self_correct += 1
                continue
            reason = "gold_does_not_verify_against_itself"
        except (ValueError, RuntimeError, TimeoutError) as error:
            reason = f"{type(error).__name__}: {error}"
        failures.append(
            {"row_id": row_id, "solution_sha256": payload_sha256(row["solution"]), "reason": reason}
        )
    models = {}
    for directory in ("irc-qwen3-8b-metadata", "irc-deepseek-llama8b-metadata"):
        path = root / ".external" / directory
        config = json.loads((path / "config.json").read_text())
        models[snapshots[directory][0]] = {
            "revision": snapshots[directory][1],
            "config": {
                key: config[key]
                for key in (
                    "model_type",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "max_position_embeddings",
                )
            },
            "generation_config": json.loads((path / "generation_config.json").read_text()),
            "weights_downloaded_or_verified": False,
        }
    result = {
        "schema": "irc-source-audit-v1",
        "benchmark_ready": False,
        "model_outcomes_inspected": 0,
        "snapshots": {
            key: {"repository": value[0], "revision": value[1]} for key, value in snapshots.items()
        },
        "source_repositories": repositories,
        "source_files_sha256": files,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "math-verify",
                "latex2sympy2-extended",
                "pyarrow",
                "torch",
                "transformers",
                "pyvene",
            )
        },
        "math": {
            "split_rows": {key: len(value) for key, value in split_rows.items()},
            "per_subject": per_subject,
            "author_linked_rows": len(merged),
            "canonical_fields": keys,
            "union_missing_rows": sum((author_counter - union).values()),
            "union_extra_rows": sum((union - author_counter).values()),
            "full_record_train_test_overlap": sum((counters["train"] & counters["test"]).values()),
            "split_provenance_limit": "Pinned EleutherAI conversion and predecessor loader mapping the original train/test directories; union equals current author-linked mirror. The original archive itself was not recovered.",
            "problem_only_or_near_duplicate_audit_completed": False,
        },
        "training_gold_self_check": {
            "self_correct": self_correct,
            "failures": failures,
            "test_gold_evaluated": False,
            "configuration": "author last_boxed_only_string; Math-Verify 0.9.0; units=False; no_fallback; strict=True; first_match; 5-second timeouts",
            "scope": "Reference measurement audit only, not model accuracy; no rows excluded or allocated by this audit.",
        },
        "models": models,
        "remaining_prerequisites": [
            "Final heterogeneous-answer and invalid-reference evaluator contract",
            "Historical exposure inventory and problem-cluster overlap manifest",
            "Frozen allocation and complete decoding configuration",
            "Real-weight CUDA and batched branch conformance",
        ],
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha256(args.output),
                "self_correct": self_correct,
                "gold_failures": len(failures),
            }
        )
    )


if __name__ == "__main__":
    main()
