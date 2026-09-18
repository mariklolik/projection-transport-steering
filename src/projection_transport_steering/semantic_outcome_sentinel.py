from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor

from projection_transport_steering.semantic_outcome import (
    fit_semantic_outcome_actions_from_metric,
)
from projection_transport_steering.semantic_outcome_evaluation import (
    match_metric_cost,
    row_forward_kl,
    row_mean_log_likelihood,
)
from projection_transport_steering.semantic_outcome_runtime import (
    encode_prompt_candidates,
    generate_many,
    score_gradient,
)
from projection_transport_steering.torch_runtime import AdditiveAction, TorchLayerAction

FACTORS = (0.25, 0.5, 1.0, 1.5)


def model_logits(
    model: object,
    batch: dict[str, Tensor],
    layer: int,
    vector: Tensor | None = None,
) -> Tensor:
    context = (
        TorchLayerAction(model, layer, AdditiveAction(vector)).installed()
        if vector is not None
        else nullcontext()
    )
    with torch.inference_mode(), context:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits


def matched_rows(model, tokenizer, layer, concept, device):
    source_rows = concept["construction"] + concept["evaluation"]
    negatives = generate_many(
        model,
        tokenizer,
        layer,
        [row["input"] for row in source_rows],
        device,
    )
    if any(not value for value in negatives):
        raise RuntimeError("matched negative generation is empty")
    construction_count = len(concept["construction"])
    enriched = [{**row, "negative": value} for row, value in zip(source_rows, negatives)]
    return {
        "construction": enriched[:construction_count],
        "evaluation": enriched[construction_count:],
    }


def diffmean_action(model, tokenizer, layer, rows, device):
    prompts = [row["input"] for row in rows] * 2
    candidates = [row["output"] for row in rows] + [row["negative"] for row in rows]
    batch, _ = encode_prompt_candidates(tokenizer, prompts, candidates, device)
    with torch.inference_mode():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    hidden = output.hidden_states[layer + 1].float()
    width = batch["candidate_mask"].shape[1]
    offsets = torch.arange(width, device=device)
    positions = batch["prefix_lengths"].unsqueeze(1) + offsets
    positions = positions.masked_fill(~batch["candidate_mask"], 0)
    indices = torch.arange(len(prompts), device=device).unsqueeze(1)
    selected = hidden[indices, positions]
    mask = batch["candidate_mask"]
    count = len(rows)
    positive = selected[:count][mask[:count]].mean(dim=0)
    negative = selected[count:][mask[count:]].mean(dim=0)
    return (positive - negative).cpu()


def solver_fields(result: dict[str, object]) -> dict[str, object]:
    dual = np.asarray(result["dual"])
    return {
        "active_constraints": int(np.count_nonzero(dual > 1e-9)),
        "complementarity_residual": result["complementarity_residual"],
        "dual": dual.tolist(),
        "dual_concentration": float(np.max(dual) / np.sum(dual)),
        "feasible": result["feasible"],
        "margins": np.asarray(result["margins"]).tolist(),
        "reduced_rank": result["reduced_rank"],
        "stationarity_residual": result["stationarity_residual"],
        "status": result["status"],
    }


def construct_actions(model, tokenizer, layer, rows, metric, device):
    mean_covectors = []
    last_covectors = []
    mean_scores = []
    token_lengths = []
    for row in rows:
        mean, scores, lengths = score_gradient(
            model,
            tokenizer,
            layer,
            row["input"],
            [row["output"], row["negative"]],
            device,
        )
        last, _, _ = score_gradient(
            model,
            tokenizer,
            layer,
            row["input"],
            [row["output"], row["negative"]],
            device,
            last_token=True,
        )
        mean_covectors.append(mean.cpu())
        last_covectors.append(last.cpu())
        mean_scores.append(scores)
        token_lengths.append(lengths)
    mean_matrix = torch.stack(mean_covectors).transpose(0, 1).numpy().astype(np.float64)
    last_matrix = torch.stack(last_covectors).transpose(0, 1).numpy().astype(np.float64)
    fitted = fit_semantic_outcome_actions_from_metric(mean_matrix, metric)
    last_fitted = fit_semantic_outcome_actions_from_metric(last_matrix, metric)
    if not fitted["semantic_outcome"]["feasible"] or not fitted["robust_euclidean"]["feasible"]:
        raise RuntimeError("mean-score hard action is infeasible")
    if not last_fitted["semantic_outcome"]["feasible"]:
        raise RuntimeError("last-token hard action is infeasible")
    raw = {
        "pooled_euclidean": fitted["pooled_euclidean"],
        "pooled_sequence_metric": fitted["pooled_sequence_metric"],
        "robust_euclidean": fitted["robust_euclidean"]["action"],
        "semantic_outcome": fitted["semantic_outcome"]["action"],
        "last_token_semantic_outcome": last_fitted["semantic_outcome"]["action"],
        "diffmean": diffmean_action(model, tokenizer, layer, rows, device).numpy(),
    }
    reference = raw["semantic_outcome"]
    actions = {
        name: match_metric_cost(reference, action, metric) for name, action in raw.items()
    }
    unit = np.asarray(fitted["unit_covectors"])
    cosine = unit.T @ unit
    return actions, {
        "construction_scores": mean_scores,
        "construction_token_lengths": token_lengths,
        "last_token_solver": solver_fields(last_fitted["semantic_outcome"]),
        "mean_covector_norms": np.linalg.norm(mean_matrix, axis=0).tolist(),
        "mean_solver": solver_fields(fitted["semantic_outcome"]),
        "metric_condition_ratio": float(np.max(metric) / np.min(metric)),
        "robust_euclidean_solver": solver_fields(fitted["robust_euclidean"]),
        "witness_cosine_max": float(np.max(cosine - np.eye(len(rows)))),
        "witness_cosine_min": float(np.min(cosine - np.eye(len(rows)))),
    }


def evaluation_batch(tokenizer, rows, device):
    prompts = []
    candidates = []
    for row in rows:
        prompts.extend([row["input"], row["input"]])
        candidates.extend([row["output"], row["negative"]])
    return encode_prompt_candidates(tokenizer, prompts, candidates, device)[0]


def evaluate_actions(model, tokenizer, layer, actions, evaluation, neutral, device):
    heldout_batch = evaluation_batch(tokenizer, evaluation, device)
    neutral_batch = encode_prompt_candidates(
        tokenizer,
        [row["input"] for row in neutral],
        [row["output"] for row in neutral],
        device,
    )[0]
    base_heldout_logits = model_logits(model, heldout_batch, layer)
    base_neutral_logits = model_logits(model, neutral_batch, layer)
    base_heldout = row_mean_log_likelihood(base_heldout_logits, heldout_batch)
    base_neutral = row_mean_log_likelihood(base_neutral_logits, neutral_batch)
    results = {}
    for name, action in actions.items():
        for factor in FACTORS:
            vector = factor * torch.tensor(action, device=device, dtype=torch.float32)
            heldout_logits = model_logits(model, heldout_batch, layer, vector)
            neutral_logits = model_logits(model, neutral_batch, layer, vector)
            heldout = row_mean_log_likelihood(heldout_logits, heldout_batch)
            neutral_scores = row_mean_log_likelihood(neutral_logits, neutral_batch)
            contrasts = heldout[0::2] - heldout[1::2]
            base_contrasts = base_heldout[0::2] - base_heldout[1::2]
            key = f"{name}@{factor:g}"
            results[key] = {
                "factor": factor,
                "heldout_score_changes": (contrasts - base_contrasts).cpu().tolist(),
                "method": name,
                "neutral_forward_kl": row_forward_kl(
                    base_neutral_logits, neutral_logits, neutral_batch
                ).cpu().tolist(),
                "neutral_score_changes": (neutral_scores - base_neutral).cpu().tolist(),
            }
    return results
