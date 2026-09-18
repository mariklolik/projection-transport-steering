import numpy as np
import torch
from torch import Tensor

from projection_transport_steering.residual_flow import (
    fit_residual_actions,
    flas_model_logits,
)
from projection_transport_steering.semantic_outcome import sequence_score_metric
from projection_transport_steering.semantic_outcome_evaluation import (
    match_metric_cost,
    row_forward_kl,
    row_mean_log_likelihood,
)
from projection_transport_steering.semantic_outcome_runtime import (
    encode_candidates,
    encode_prompt_candidates,
)
from projection_transport_steering.semantic_outcome_sentinel import (
    evaluation_batch,
    model_logits,
    solver_fields,
)

QUANTILES = (0.5, 0.75, 1.0)


def _contrasts(logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
    scores = row_mean_log_likelihood(logits, batch)
    return scores[0::2] - scores[1::2]


def _flow_gradient(
    generator: object,
    batch: dict[str, Tensor],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    contrast: bool,
) -> tuple[np.ndarray, list[float]]:
    delta = torch.zeros(
        generator.llm.config.hidden_size,
        device=batch["input_ids"].device,
        requires_grad=True,
    )
    logits = flas_model_logits(
        generator,
        batch,
        concept_hidden,
        concept_mask,
        vector=delta,
    )
    scores = row_mean_log_likelihood(logits, batch)
    objective = scores[0] - scores[1] if contrast else scores[0]
    gradient = torch.autograd.grad(objective, delta)[0].float().cpu().numpy()
    return gradient, scores.detach().float().cpu().tolist()


def construct_geometry(
    generator: object,
    tokenizer: object,
    construction: list[dict[str, object]],
    neutral: list[dict[str, object]],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    construction_batch = evaluation_batch(tokenizer, construction, device)
    with torch.inference_mode():
        base_logits = model_logits(generator.llm, construction_batch, generator.layer)
        flow_logits = flas_model_logits(
            generator,
            construction_batch,
            concept_hidden,
            concept_mask,
        )
        base_contrasts = _contrasts(base_logits, construction_batch)
        flow_contrasts = _contrasts(flow_logits, construction_batch)
    covectors = []
    construction_scores = []
    construction_lengths = []
    for row in construction:
        batch, lengths = encode_candidates(
            tokenizer,
            row["input"],
            [row["output"], row["negative"]],
            device,
        )
        gradient, scores = _flow_gradient(
            generator,
            batch,
            concept_hidden,
            concept_mask,
            contrast=True,
        )
        covectors.append(gradient)
        construction_scores.append(scores)
        construction_lengths.append(lengths)
    neutral_gradients = []
    for row in neutral:
        batch, _ = encode_candidates(
            tokenizer,
            row["input"],
            [row["output"]],
            device,
        )
        gradient, _ = _flow_gradient(
            generator,
            batch,
            concept_hidden,
            concept_mask,
            contrast=False,
        )
        neutral_gradients.append(gradient)
    covector_matrix = np.stack(covectors, axis=1).astype(np.float64)
    neutral_matrix = np.stack(neutral_gradients).astype(np.float64)
    metric = sequence_score_metric(neutral_matrix)
    changes = (flow_contrasts - base_contrasts).float().cpu().numpy()
    cosine = covector_matrix.T @ covector_matrix
    norms = np.linalg.norm(covector_matrix, axis=0)
    cosine /= norms[:, None] * norms[None, :]
    return (
        {
            "base_contrasts": base_contrasts.float().cpu().tolist(),
            "changes": changes.tolist(),
            "construction_lengths": construction_lengths,
            "construction_scores_at_flow": construction_scores,
            "covector_norms": norms.tolist(),
            "flas_contrasts": flow_contrasts.float().cpu().tolist(),
            "metric_condition_ratio": float(np.max(metric) / np.min(metric)),
            "metric_max": float(np.max(metric)),
            "metric_min": float(np.min(metric)),
            "neutral_gradient_norms": np.linalg.norm(neutral_matrix, axis=1).tolist(),
            "witness_cosine_max": float(np.max(cosine - np.eye(len(covectors)))),
            "witness_cosine_min": float(np.min(cosine - np.eye(len(covectors)))),
        },
        covector_matrix,
        metric,
    )


def _condition(
    generator: object,
    heldout_batch: dict[str, Tensor],
    neutral_batch: dict[str, Tensor],
    flow_heldout_logits: Tensor,
    flow_neutral_logits: Tensor,
    flow_heldout_contrasts: Tensor,
    flow_neutral_scores: Tensor,
    concept_hidden: Tensor,
    concept_mask: Tensor,
    action: np.ndarray,
    candidate: np.ndarray,
    covectors: np.ndarray,
    metric: np.ndarray,
) -> dict[str, object]:
    vector = torch.tensor(action, device=heldout_batch["input_ids"].device, dtype=torch.float32)
    with torch.inference_mode():
        heldout_logits = flas_model_logits(
            generator,
            heldout_batch,
            concept_hidden,
            concept_mask,
            vector=vector,
        )
        neutral_logits = flas_model_logits(
            generator,
            neutral_batch,
            concept_hidden,
            concept_mask,
            vector=vector,
        )
        heldout_changes = _contrasts(heldout_logits, heldout_batch) - flow_heldout_contrasts
        neutral_changes = row_mean_log_likelihood(neutral_logits, neutral_batch) - flow_neutral_scores
        divergence = row_forward_kl(flow_neutral_logits, neutral_logits, neutral_batch)
    denominator = np.linalg.norm(action) * np.linalg.norm(candidate)
    return {
        "action_cosine_to_candidate": float(action @ candidate / denominator),
        "action_norm": float(np.linalg.norm(action)),
        "heldout_score_changes": heldout_changes.float().cpu().tolist(),
        "metric_cost": float(action @ (metric * action)),
        "neutral_forward_kl": divergence.float().cpu().tolist(),
        "neutral_score_changes": neutral_changes.float().cpu().tolist(),
        "realized_margins": (covectors.T @ action).tolist(),
    }


def evaluate_quantiles(
    generator: object,
    tokenizer: object,
    evaluation: list[dict[str, object]],
    neutral: list[dict[str, object]],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    covectors: np.ndarray,
    metric: np.ndarray,
    changes: np.ndarray,
    retained_action: np.ndarray,
    concept_id: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, dict[str, Tensor]], dict[str, object]]:
    heldout_batch = evaluation_batch(tokenizer, evaluation, device)
    neutral_batch = encode_prompt_candidates(
        tokenizer,
        [row["input"] for row in neutral],
        [row["output"] for row in neutral],
        device,
    )[0]
    with torch.inference_mode():
        base_neutral_logits = model_logits(generator.llm, neutral_batch, generator.layer)
        flow_heldout_logits = flas_model_logits(
            generator,
            heldout_batch,
            concept_hidden,
            concept_mask,
        )
        flow_neutral_logits = flas_model_logits(
            generator,
            neutral_batch,
            concept_hidden,
            concept_mask,
        )
        flow_heldout_contrasts = _contrasts(flow_heldout_logits, heldout_batch)
        flow_neutral_scores = row_mean_log_likelihood(flow_neutral_logits, neutral_batch)
        flow_neutral_kl = row_forward_kl(
            base_neutral_logits,
            flow_neutral_logits,
            neutral_batch,
        )
    quantiles = {}
    saved_actions = {}
    for quantile in QUANTILES:
        fitted = fit_residual_actions(covectors, metric, changes, quantile, concept_id)
        candidate = np.asarray(fitted["actions"]["residual_metric"])
        actions = dict(fitted["actions"])
        actions["retained_pooled_sequence_metric"] = match_metric_cost(
            candidate,
            retained_action,
            metric,
        )
        conditions = {
            name: _condition(
                generator,
                heldout_batch,
                neutral_batch,
                flow_heldout_logits,
                flow_neutral_logits,
                flow_heldout_contrasts,
                flow_neutral_scores,
                concept_hidden,
                concept_mask,
                np.asarray(action),
                candidate,
                covectors,
                metric,
            )
            for name, action in actions.items()
        }
        key = f"{quantile:g}"
        saved_actions[key] = {
            name: torch.tensor(action) for name, action in actions.items()
        }
        quantiles[key] = {
            "candidate_solver": solver_fields(fitted["candidate_solver"]),
            "conditions": conditions,
            "deficits": np.asarray(fitted["deficits"]).tolist(),
            "euclidean_solver": solver_fields(fitted["euclidean_solver"]),
            "permutation": np.asarray(fitted["permutation"]).tolist(),
            "pooled_solver": solver_fields(fitted["pooled_solver"]),
            "shuffled_solver": solver_fields(fitted["shuffled_solver"]),
            "target": fitted["target"],
        }
    return quantiles, saved_actions, {
        "flas_neutral_forward_kl_from_base": flow_neutral_kl.float().cpu().tolist(),
        "flas_neutral_scores": flow_neutral_scores.float().cpu().tolist(),
    }
