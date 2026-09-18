import argparse
import hashlib
import json
import platform
import socket
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import __version__ as datasets_version
from datasets import load_dataset
from materialize_amortized_pullback_data import (
    CONCEPTS,
    DATASET_REVISION,
    DUAL_STEERING_REVISION,
    MODEL_REVISION,
    token_digest,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.splits import allocate_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=4096)
    parser.add_argument("--max-documents", type=int, default=100_000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_quotas(concept: str) -> dict[str, int]:
    if concept == "third":
        return {"smoke": 1, "sentinel": 3, "development": 8, "pilot": 16}
    if concept == "ing":
        return {"sentinel": 3, "development": 8, "pilot": 16}
    if concept == "past":
        return {"sentinel": 2, "development": 8, "pilot": 16}
    raise ValueError(f"unknown concept: {concept}")


def allocate_candidates(
    concept: str,
    candidates: list[dict[str, Any]],
    excluded_hashes: set[str],
) -> list[dict[str, Any]]:
    retained = {
        row["context_sha256"]: row
        for row in candidates
        if row["context_sha256"] not in excluded_hashes
    }
    allocation = allocate_groups(
        retained,
        stage_quotas(concept),
        f"projection-transport-steering/cacheback-v1/{concept}",
    )
    return sorted(
        [dict(retained[group], stage=stage) for group, stage in allocation.items()],
        key=lambda row: (row["stage"], row["context_sha256"]),
    )


def prior_packet(path: Path) -> tuple[dict[str, Any], set[str]]:
    data = json.loads(path.read_text())
    excluded = {
        row["context_sha256"]
        for rows in data["contexts"].values()
        for row in rows
    }
    return data, excluded


def quotas_reached(candidates: dict[str, list[dict[str, Any]]]) -> bool:
    return all(len(candidates[concept]) >= sum(stage_quotas(concept).values()) for concept in CONCEPTS)


def validate_global_disjoint(contexts: dict[str, list[dict[str, Any]]]) -> None:
    rows = [row for concept_rows in contexts.values() for row in concept_rows]
    hashes = [row["context_sha256"] for row in rows]
    document_indices = [row["dataset_index"] for row in rows]
    if len(set(hashes)) != len(rows) or len(set(document_indices)) != len(rows):
        raise ValueError("cacheback source states must be globally disjoint")


def screened_data(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
    ).cuda().eval()
    prior, excluded = prior_packet(args.prior_data)
    mappings = prior["mappings"]
    mapping_hashes = prior["mapping_sha256"]
    lookups: dict[str, torch.Tensor] = {}
    for concept, mapping in mappings.items():
        lookup = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device="cuda")
        lookup[mapping["base_ids"]] = True
        lookups[concept] = lookup
    candidates: dict[str, list[dict[str, Any]]] = {concept: [] for concept in CONCEPTS}
    global_candidate_hashes: set[str] = set()
    dataset = islice(
        load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
            revision=DATASET_REVISION,
        ),
        args.start_index,
        None,
    )
    scanned = 0
    started = time.perf_counter()
    while scanned < args.max_documents and not quotas_reached(candidates):
        rows = list(islice(dataset, min(args.batch_size, args.max_documents - scanned)))
        if not rows:
            break
        encoded = tokenizer(
            [row["text"] for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=False,
        ).to("cuda")
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits.float()
        probabilities = torch.softmax(logits, dim=-1)
        top_probabilities, top_ids = probabilities.topk(3, dim=-1)
        for batch_index in range(len(rows)):
            dataset_index = args.start_index + scanned + batch_index
            length = int(encoded.attention_mask[batch_index].sum())
            tokens = encoded.input_ids[batch_index, :length].cpu().tolist()
            accepted_row = False
            for concept in CONCEPTS:
                if len(candidates[concept]) >= sum(stage_quotas(concept).values()):
                    continue
                matches = lookups[concept][top_ids[batch_index, :length]].all(-1)
                matches &= top_probabilities[batch_index, :length].sum(-1) > 0.7
                for position in matches.nonzero(as_tuple=False).flatten().cpu().tolist():
                    prefix = tokens[: position + 1]
                    digest = token_digest(prefix)
                    if digest in excluded or digest in global_candidate_hashes:
                        continue
                    candidates[concept].append(
                        {
                            "context_sha256": digest,
                            "dataset_index": dataset_index,
                            "token_ids": prefix,
                            "token_position": position,
                        }
                    )
                    global_candidate_hashes.add(digest)
                    accepted_row = True
                    break
                if accepted_row:
                    break
        scanned += len(rows)
        del encoded, logits, probabilities, top_probabilities, top_ids
    if not quotas_reached(candidates):
        counts = {concept: len(rows) for concept, rows in candidates.items()}
        raise RuntimeError(f"screening quotas were not reached: {counts}")
    contexts = {
        concept: allocate_candidates(concept, rows, excluded)
        for concept, rows in candidates.items()
    }
    validate_global_disjoint(contexts)
    return {
        "contexts": contexts,
        "dataset_revision": DATASET_REVISION,
        "dual_steering_revision": DUAL_STEERING_REVISION,
        "excluded_prior_data_sha256": file_sha256(args.prior_data),
        "mapping_sha256": mapping_hashes,
        "mappings": mappings,
        "model_revision": MODEL_REVISION,
        "receipt": {
            "cuda": torch.version.cuda,
            "datasets": datasets_version,
            "gpu": torch.cuda.get_device_name(0),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "scanned_documents": scanned,
            "source_start_index": args.start_index,
            "screening_seconds": time.perf_counter() - started,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    data = screened_data(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "data.json").write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
