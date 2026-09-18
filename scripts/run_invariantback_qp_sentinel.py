import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.claim_relative_geometry import (
    reduced_identity_robust_action,
)
from projection_transport_steering.invariantback_runner import (
    CONSTRUCTION_ROW_ORDERS,
    LABELS,
    IndexedAdditiveAction,
    candidate_mean_log_likelihood,
    encode_candidate_batch,
    render_prompt,
    semantic_mappings,
    sentinel_cases,
)
from projection_transport_steering.torch_runtime import TorchLayerAction

VOCABULARIES = (("A", "B", "C"), ("X", "Y", "Z"), ("1", "2", "3"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--microbatch-groups", type=int, default=2)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_slice(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    rows: slice,
) -> torch.Tensor:
    return candidate_mean_log_likelihood(
        logits[rows],
        batch["candidate_ids"][rows],
        batch["candidate_mask"][rows],
        batch["prefix_lengths"][rows],
    )


def view_covector(
    model,
    tokenizer,
    case: dict[str, object],
    groups: list[dict[str, object]],
    mapping: tuple[str, str, str],
    vocabulary: tuple[str, str, str],
    row_order: tuple[int, int, int],
    microbatch_groups: int,
) -> tuple[torch.Tensor, float, int]:
    covector = torch.zeros(model.config.hidden_size, dtype=torch.float64)
    margin_sum = 0.0
    forward_count = 0
    for offset in range(0, len(groups), microbatch_groups):
        subset = groups[offset : offset + microbatch_groups]
        prompts = []
        target_candidates = []
        source_candidates = []
        for group in subset:
            prompt, candidates = render_prompt(
                case["dataset"],
                group["endpoints"][case["source"]]["text"],
                mapping,
                vocabulary,
                row_order,
                0,
            )
            prompts.append(prompt)
            target_candidates.append(candidates[case["target"]])
            source_candidates.append(candidates[case["source"]])
        count = len(prompts)
        batch = encode_candidate_batch(
            tokenizer,
            prompts + prompts,
            target_candidates + source_candidates,
            device="cuda",
        )
        delta = torch.zeros(model.config.hidden_size, device="cuda", requires_grad=True)
        action = IndexedAdditiveAction(delta, batch["prefix_lengths"] - 1)
        with TorchLayerAction(model, case["layer"], action).installed():
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        margin = score_slice(logits, batch, slice(0, count)) - score_slice(
            logits,
            batch,
            slice(count, 2 * count),
        )
        gradient = torch.autograd.grad(margin, delta)[0].detach().double().cpu()
        weight = count / len(groups)
        covector += weight * gradient
        margin_sum += weight * float(margin.detach())
        forward_count += 1
        del logits, delta, action, batch, gradient
    return covector, margin_sum, forward_count


def replay_error(model, tokenizer, case: dict[str, object], group: dict[str, object]) -> float:
    labels = LABELS[case["dataset"]]
    prompt, candidates = render_prompt(
        case["dataset"],
        group["endpoints"][case["source"]]["text"],
        labels,
        VOCABULARIES[0],
        CONSTRUCTION_ROW_ORDERS[0],
        0,
    )
    batch = encode_candidate_batch(
        tokenizer,
        [prompt, prompt],
        [candidates[case["target"]], candidates[case["source"]]],
        device="cuda",
    )
    with torch.inference_mode():
        base = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        action = IndexedAdditiveAction(
            torch.zeros(model.config.hidden_size, device="cuda"),
            batch["prefix_lengths"] - 1,
        )
        with TorchLayerAction(model, case["layer"], action).installed():
            replay = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
    return float((base - replay).abs().max())


def run_case(
    model,
    tokenizer,
    data: dict[str, object],
    case: dict[str, object],
    output: Path,
    microbatch_groups: int,
) -> dict[str, object]:
    started = time.perf_counter()
    groups = data["datasets"][case["dataset"]]["construction"][:8]
    vectors = []
    views = []
    base_margins = []
    forward_count = 0
    for mapping in semantic_mappings(case["dataset"]):
        for vocabulary in VOCABULARIES:
            for row_order in CONSTRUCTION_ROW_ORDERS:
                vector, margin, forwards = view_covector(
                    model,
                    tokenizer,
                    case,
                    groups,
                    mapping,
                    vocabulary,
                    row_order,
                    microbatch_groups,
                )
                vectors.append(vector.numpy())
                base_margins.append(margin)
                forward_count += forwards
                views.append(
                    {
                        "mapping": mapping,
                        "row_order": row_order,
                        "vocabulary": vocabulary,
                    }
                )
    matrix = np.stack(vectors, axis=1)
    norms = np.linalg.norm(matrix, axis=0)
    if not np.all(np.isfinite(matrix)) or np.any(norms == 0.0):
        raise RuntimeError("semantic view covectors are nonfinite or zero")
    normalized = matrix / norms
    solution = reduced_identity_robust_action(
        normalized,
        target=1.0,
        tolerance=1e-7,
    )
    replay = replay_error(model, tokenizer, case, groups[0])
    tensor_path = output / f"{case['case_id']}.npz"
    np.savez_compressed(
        tensor_path,
        base_margins=np.asarray(base_margins),
        covectors=matrix,
        normalized_covectors=normalized,
        solution_action=np.asarray(solution["action"]),
    )
    margins = np.asarray(solution["margins"])
    passed = (
        replay == 0.0
        and bool(solution["feasible"])
        and float(solution["stationarity_residual"]) <= 1e-5
        and float(solution["complementarity_residual"]) <= 1e-5
        and float(np.min(margins)) >= 1.0 - 1e-5
    )
    return {
        **case,
        "base_margin_max": float(np.max(base_margins)),
        "base_margin_min": float(np.min(base_margins)),
        "complementarity_residual": float(solution["complementarity_residual"]),
        "covector_norm_max": float(np.max(norms)),
        "covector_norm_min": float(np.min(norms)),
        "elapsed_seconds": time.perf_counter() - started,
        "feasible": bool(solution["feasible"]),
        "forward_count": forward_count + 2,
        "minimum_normalized_margin": float(np.min(margins)),
        "reduced_rank": int(solution["reduced_rank"]),
        "replay_max_abs_logit_error": replay,
        "stationarity_residual": float(solution["stationarity_residual"]),
        "status": "pass" if passed else "fail",
        "tensor_file": tensor_path.name,
        "tensor_sha256": file_sha256(tensor_path),
        "view_count": len(views),
        "views": views,
    }


def main() -> None:
    args = parse_args()
    if (
        args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or args.microbatch_groups <= 0
    ):
        raise ValueError("shard or microbatch parameters are invalid")
    cases = sentinel_cases()[args.shard_index :: args.shard_count]
    if not cases:
        raise ValueError("shard has no cases")
    data = json.loads(args.data.read_text())
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
    results = [
        run_case(model, tokenizer, data, case, args.output, args.microbatch_groups)
        for case in cases
    ]
    results_path = args.output / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    receipt = {
        "case_count": len(results),
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "microbatch_groups": args.microbatch_groups,
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
        raise RuntimeError("InvariantBack QP sentinel shard failed")


if __name__ == "__main__":
    main()
