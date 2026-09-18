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
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.splits import allocate_groups


MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DUAL_STEERING_REVISION = "ec4d6657faeba9e13e6b6e07c5cfd53e19ebe7fa"
CONCEPTS = ("third", "ing", "past")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-documents", type=int, default=200_000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def retained_token_pairs(mapping: dict[str, str], tokenizer: Any) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for base, target in mapping.items():
        base_surface = f" {base[1:]}" if base.startswith("▁") else base
        target_surface = f" {target[1:]}" if target.startswith("▁") else target
        base_ids = tokenizer.encode(base_surface, add_special_tokens=False)
        target_ids = tokenizer.encode(target_surface, add_special_tokens=False)
        if len(base_ids) == len(target_ids) == 1:
            retained.append(
                {
                    "base": base_surface,
                    "base_id": base_ids[0],
                    "target": target_surface,
                    "target_id": target_ids[0],
                }
            )
    return retained


def matching_positions(logits: torch.Tensor, concept_ids: set[int], threshold: float) -> list[int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_probabilities, top_ids = probabilities.topk(3, dim=-1)
    lookup = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    lookup[list(concept_ids)] = True
    matches = lookup[top_ids].all(-1) & (top_probabilities.sum(-1) > threshold)
    return matches.nonzero(as_tuple=False).flatten().cpu().tolist()


def screening_pending(scanned: int, accepted: dict[str, list[Any]]) -> bool:
    return scanned < 4096 or any(len(rows) < 100 for rows in accepted.values())


def load_mappings(root: Path, tokenizer: Any) -> tuple[dict[str, Any], dict[str, str]]:
    mappings: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for concept in CONCEPTS:
        path = root / f"mapping_verb_{concept}.json"
        raw = json.loads(path.read_text())
        retained = retained_token_pairs(raw, tokenizer)
        if len(retained) < 3:
            raise ValueError(f"insufficient retained pairs for {concept}")
        mappings[concept] = {
            "pairs": retained,
            "base_ids": sorted({row["base_id"] for row in retained}),
            "target_ids": sorted({row["target_id"] for row in retained}),
        }
        hashes[concept] = file_sha256(path)
    return mappings, hashes


def screened_data(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).cuda().eval()
    mappings, mapping_hashes = load_mappings(args.mapping_root, tokenizer)
    lookups: dict[str, torch.Tensor] = {}
    for concept, mapping in mappings.items():
        lookup = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device="cuda")
        lookup[mapping["base_ids"]] = True
        lookups[concept] = lookup
    accepted: dict[str, list[dict[str, Any]]] = {concept: [] for concept in CONCEPTS}
    accepted_hashes: dict[str, set[str]] = {concept: set() for concept in CONCEPTS}
    basis_candidates: dict[str, list[int]] = {}
    dataset = iter(
        load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
            revision=DATASET_REVISION,
        )
    )
    scanned = 0
    started = time.perf_counter()
    while scanned < args.max_documents and screening_pending(scanned, accepted):
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
            dataset_index = scanned + batch_index
            length = int(encoded.attention_mask[batch_index].sum())
            tokens = encoded.input_ids[batch_index, :length].cpu().tolist()
            if dataset_index < 4096 and length >= 32:
                basis_candidates[token_digest(tokens)] = tokens
            for concept in CONCEPTS:
                if len(accepted[concept]) >= 100:
                    continue
                matches = lookups[concept][top_ids[batch_index, :length]].all(-1)
                matches &= top_probabilities[batch_index, :length].sum(-1) > 0.7
                for position in matches.nonzero(as_tuple=False).flatten().cpu().tolist():
                    prefix = tokens[: position + 1]
                    digest = token_digest(prefix)
                    if digest in accepted_hashes[concept]:
                        continue
                    accepted[concept].append(
                        {
                            "context_sha256": digest,
                            "dataset_index": dataset_index,
                            "token_ids": prefix,
                            "token_position": position,
                        }
                    )
                    accepted_hashes[concept].add(digest)
                    if len(accepted[concept]) == 100:
                        break
        scanned += len(rows)
        del encoded, logits, probabilities, top_probabilities, top_ids
    if any(len(rows) != 100 for rows in accepted.values()):
        raise RuntimeError(f"screening quotas were not reached: {[len(accepted[c]) for c in CONCEPTS]}")
    evaluation_prefixes = {
        tuple(row["token_ids"])
        for concept_rows in accepted.values()
        for row in concept_rows
    }
    basis_rows = [
        {"context_sha256": digest, "token_ids": tokens}
        for digest, tokens in basis_candidates.items()
        if tuple(tokens) not in evaluation_prefixes
    ]
    basis_rows.sort(
        key=lambda row: hashlib.sha256(
            f"projection-transport-steering/v25/basis/{row['context_sha256']}".encode()
        ).digest()
    )
    if len(basis_rows) < 256:
        raise RuntimeError("insufficient target-blind basis contexts")
    allocations: dict[str, list[dict[str, Any]]] = {}
    for concept, concept_rows in accepted.items():
        stages = allocate_groups(
            [row["context_sha256"] for row in concept_rows],
            {"development": 12, "validation": 88},
            f"projection-transport-steering/v25/context/{concept}",
        )
        allocations[concept] = [dict(row, stage=stages[row["context_sha256"]]) for row in concept_rows]
    return {
        "basis_contexts": basis_rows[:256],
        "contexts": allocations,
        "dataset_revision": DATASET_REVISION,
        "dual_steering_revision": DUAL_STEERING_REVISION,
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
            "screening_seconds": time.perf_counter() - started,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = screened_data(args)
    (args.output / "data.json").write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
