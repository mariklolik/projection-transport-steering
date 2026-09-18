import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.invariantback_runner import (
    LABELS,
    candidate_mean_log_likelihood,
    encode_candidate_batch,
    render_mapping_check,
    semantic_mappings,
)

VOCABULARIES = (
    ("A", "B", "C"),
    ("X", "Y", "Z"),
    ("1", "2", "3"),
    ("I", "II", "III"),
    ("one", "two", "three"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cells() -> list[dict[str, object]]:
    return [
        {
            "case_id": f"{dataset}-mapping{mapping_index}-vocab{vocabulary_index}",
            "dataset": dataset,
            "mapping": mapping,
            "mapping_index": mapping_index,
            "vocabulary": vocabulary,
            "vocabulary_index": vocabulary_index,
        }
        for dataset in LABELS
        for mapping_index, mapping in enumerate(semantic_mappings(dataset))
        for vocabulary_index, vocabulary in enumerate(VOCABULARIES)
    ]


def run_cell(model, tokenizer, cell: dict[str, object]) -> dict[str, object]:
    labels = LABELS[cell["dataset"]]
    prompts = []
    candidates = []
    expected = []
    metadata = []
    for label in labels:
        for phrasing in range(3):
            prompt, rendered = render_mapping_check(
                cell["dataset"],
                cell["mapping"],
                cell["vocabulary"],
                label,
                phrasing,
            )
            for candidate_label in labels:
                prompts.append(prompt)
                candidates.append(rendered[candidate_label])
            expected.append(labels.index(label))
            metadata.append({"label": label, "phrasing": phrasing})
    batch = encode_candidate_batch(tokenizer, prompts, candidates, device="cuda")
    with torch.inference_mode():
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    scores = []
    for index in range(len(prompts)):
        score = candidate_mean_log_likelihood(
            logits[index : index + 1],
            batch["candidate_ids"][index : index + 1],
            batch["candidate_mask"][index : index + 1],
            batch["prefix_lengths"][index : index + 1],
        )
        scores.append(float(score))
    matrix = torch.tensor(scores).reshape(9, 3)
    predictions = matrix.argmax(dim=1)
    expected_tensor = torch.tensor(expected)
    query_rows = []
    for index, meta in enumerate(metadata):
        alternatives = matrix[index].clone()
        alternatives[expected[index]] = -torch.inf
        margin = float(matrix[index, expected[index]] - alternatives.max())
        query_rows.append(
            {
                **meta,
                "correct": bool(predictions[index] == expected_tensor[index]),
                "margin": margin,
                "predicted_label": labels[int(predictions[index])],
                "scores": {
                    label: float(matrix[index, label_index])
                    for label_index, label in enumerate(labels)
                },
            }
        )
    accuracy = sum(row["correct"] for row in query_rows) / len(query_rows)
    minimum_margin = min(row["margin"] for row in query_rows)
    return {
        **cell,
        "accuracy": accuracy,
        "forward_count": 1,
        "minimum_correct_margin": minimum_margin,
        "queries": query_rows,
        "status": "pass" if accuracy >= 8 / 9 and minimum_margin > 0.0 else "fail",
    }


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard parameters are invalid")
    selected = cells()[args.shard_index :: args.shard_count]
    if not selected:
        raise ValueError("shard has no cells")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = [run_cell(model, tokenizer, cell) for cell in selected]
    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    receipt = {
        "cell_count": len(results),
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "elapsed_seconds": time.perf_counter() - started,
        "forward_count": len(results),
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "model": str(args.model),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "platform": platform.platform(),
        "results_sha256": file_sha256(results_path),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "status": "pass" if all(row["status"] == "pass" for row in results) else "fail",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "pass":
        raise RuntimeError("InvariantBack mapping sentinel shard failed")


if __name__ == "__main__":
    main()
