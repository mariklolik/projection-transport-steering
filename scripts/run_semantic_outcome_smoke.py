import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.semantic_outcome import fit_semantic_outcome_actions
from projection_transport_steering.semantic_outcome_runtime import (
    encode_candidates,
    generate,
    generate_many,
    score_gradient,
    sequence_scores,
)
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction

SEED = 20260907


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--alpaca", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept-id", type=int, default=1)
    parser.add_argument("--layer", type=int, default=20)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def model_manifest(path: Path) -> list[dict[str, object]]:
    entries = []
    for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        resolved = file.resolve()
        entries.append(
            {
                "blob": resolved.name,
                "path": str(file.relative_to(path)),
                "size": resolved.stat().st_size,
            }
        )
    if not entries:
        raise ValueError("model snapshot is empty")
    return entries


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    data = json.loads(args.data.read_text())
    concept = data["concepts"][str(args.concept_id)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    first = concept["construction"][0]
    batch, _ = encode_candidates(tokenizer, first["input"], [first["output"]], device)
    with torch.inference_mode():
        base_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        zero = torch.zeros(model.config.hidden_size, device=device)
        with TorchLayerAction(model, args.layer, AdditiveAction(zero)).installed():
            replay_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
    replay_error = float((base_logits - replay_logits).abs().max())
    matched = {"construction": [], "evaluation": []}
    rows = concept["construction"] + concept["evaluation"]
    negatives = generate_many(
        model,
        tokenizer,
        args.layer,
        [row["input"] for row in rows],
        device,
    )
    offset = 0
    for allocation in matched:
        for row in concept[allocation]:
            negative = negatives[offset]
            offset += 1
            if not negative:
                raise RuntimeError("matched negative generation is empty")
            matched[allocation].append({**row, "negative": negative})
    neutral_gradients = []
    for row in data["neutral"]:
        gradient, _, _ = score_gradient(
            model, tokenizer, args.layer, row["input"], [row["output"]], device
        )
        neutral_gradients.append(gradient.cpu())
    covectors = []
    construction_scores = []
    construction_lengths = []
    for row in matched["construction"]:
        gradient, scores, lengths = score_gradient(
            model,
            tokenizer,
            args.layer,
            row["input"],
            [row["output"], row["negative"]],
            device,
        )
        covectors.append(gradient.cpu())
        construction_scores.append(scores)
        construction_lengths.append(lengths)
    covector_matrix = torch.stack(covectors).transpose(0, 1).numpy().astype(np.float64)
    neutral_matrix = torch.stack(neutral_gradients).numpy().astype(np.float64)
    fitted = fit_semantic_outcome_actions(covector_matrix, neutral_matrix)
    candidate = fitted["semantic_outcome"]
    candidate_vector = torch.tensor(candidate["action"], device=device, dtype=torch.float32)
    factor = 0.25
    held_out_changes = []
    evaluation_lengths = []
    for row in matched["evaluation"]:
        base_scores, lengths = sequence_scores(
            model,
            tokenizer,
            args.layer,
            row["input"],
            [row["output"], row["negative"]],
            device,
        )
        steered_scores, _ = sequence_scores(
            model,
            tokenizer,
            args.layer,
            row["input"],
            [row["output"], row["negative"]],
            device,
            vector=factor * candidate_vector,
        )
        held_out_changes.append(
            float(
                (steered_scores[0] - steered_scores[1] - base_scores[0] + base_scores[1]).detach()
            )
        )
        evaluation_lengths.append(lengths)
    alpaca = pd.DataFrame(json.loads(args.alpaca.read_text()))
    tuning_prompt = alpaca.sample(10, random_state=args.concept_id)["instruction"].tolist()[0]
    generation_no_op = generate(model, tokenizer, args.layer, tuning_prompt, device)
    generation_candidate = generate(
        model,
        tokenizer,
        args.layer,
        tuning_prompt,
        device,
        vector=factor * candidate_vector,
    )
    actions = {
        "metric_diagonal": torch.tensor(fitted["metric_diagonal"]),
        "pooled_euclidean": torch.tensor(fitted["pooled_euclidean"]),
        "pooled_sequence_metric": torch.tensor(fitted["pooled_sequence_metric"]),
        "robust_euclidean": torch.tensor(fitted["robust_euclidean"]["action"]),
        "score_covectors": torch.tensor(covector_matrix),
        "semantic_outcome": candidate_vector.cpu(),
        "unit_covectors": torch.tensor(fitted["unit_covectors"]),
    }
    torch.save(actions, args.output / "actions.pt")
    (args.output / "matched_continuations.json").write_text(
        json.dumps(matched, indent=2, sort_keys=True, ensure_ascii=False)
    )
    torch.cuda.synchronize()
    finite = all(torch.isfinite(value).all() for value in actions.values())
    status = (
        "pass"
        if replay_error == 0.0
        and candidate["feasible"]
        and candidate["stationarity_residual"] < 1e-5
        and candidate["complementarity_residual"] < 1e-5
        and finite
        and all(np.isfinite(held_out_changes))
        and generation_no_op
        and generation_candidate
        else "fail"
    )
    return {
        "actions_sha256": file_sha256(args.output / "actions.pt"),
        "alpaca_sha256": file_sha256(args.alpaca),
        "candidate_complementarity_residual": candidate["complementarity_residual"],
        "candidate_feasible": candidate["feasible"],
        "candidate_metric_cost": candidate["metric_cost"],
        "candidate_stationarity_residual": candidate["stationarity_residual"],
        "construction_candidate_lengths": construction_lengths,
        "construction_scores": construction_scores,
        "covector_norms": [float(value.norm()) for value in covectors],
        "covectors_sha256": tensor_sha256(torch.stack(covectors)),
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_candidate_lengths": evaluation_lengths,
        "generation_candidate": generation_candidate,
        "generation_no_op": generation_no_op,
        "gpu": torch.cuda.get_device_name(0),
        "held_out_score_changes_factor_0_25": held_out_changes,
        "host": socket.gethostname(),
        "layer": args.layer,
        "matched_continuations_sha256": file_sha256(args.output / "matched_continuations.json"),
        "model_manifest": model_manifest(args.model),
        "model_snapshot": str(args.model.resolve()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "platform": platform.platform(),
        "reduced_rank": candidate["reduced_rank"],
        "replay_max_abs_logit_error": replay_error,
        "source_concept": concept["concept"],
        "source_concept_id": args.concept_id,
        "status": status,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tuning_input_id": 0,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = run(args)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "pass":
        raise RuntimeError("semantic-outcome smoke failed")


if __name__ == "__main__":
    main()
