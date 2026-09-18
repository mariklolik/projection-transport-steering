import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from projection_transport_steering.semantic_outcome_runtime import candidate_score_mask

FloatArray = NDArray[np.float64]


def _candidate_positions(logits: Tensor, batch: dict[str, Tensor], mask: Tensor) -> Tensor:
    offsets = torch.arange(mask.shape[1], device=logits.device)
    positions = batch["prefix_lengths"].to(logits.device).unsqueeze(1) - 1 + offsets
    if torch.any(positions[mask] < 0) or torch.any(positions[mask] >= logits.shape[1]):
        raise ValueError("candidate positions are outside logits")
    return positions.masked_fill(~mask.to(logits.device), 0)


def row_mean_log_likelihood(
    logits: Tensor,
    batch: dict[str, Tensor],
    last_token: bool = False,
) -> Tensor:
    mask = candidate_score_mask(batch["candidate_mask"], last_token).to(logits.device)
    positions = _candidate_positions(logits, batch, mask)
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    token_logits = logits[rows, positions]
    token_scores = torch.log_softmax(token_logits.float(), dim=-1)
    selected = token_scores.gather(
        -1, batch["candidate_ids"].to(logits.device).unsqueeze(-1)
    ).squeeze(-1)
    weights = mask.to(logits.device)
    return (selected * weights).sum(dim=1) / weights.sum(dim=1)


def row_forward_kl(
    base_logits: Tensor,
    steered_logits: Tensor,
    batch: dict[str, Tensor],
) -> Tensor:
    mask = batch["candidate_mask"].to(base_logits.device)
    positions = _candidate_positions(base_logits, batch, mask)
    rows = torch.arange(base_logits.shape[0], device=base_logits.device).unsqueeze(1)
    base_log = torch.log_softmax(base_logits[rows, positions].float(), dim=-1)
    steered_log = torch.log_softmax(steered_logits[rows, positions].float(), dim=-1)
    token_kl = torch.sum(torch.exp(base_log) * (base_log - steered_log), dim=-1)
    weights = mask.to(base_logits.device)
    return (token_kl * weights).sum(dim=1) / weights.sum(dim=1)


def match_metric_cost(
    reference: ArrayLike,
    action: ArrayLike,
    metric_diagonal: ArrayLike,
) -> FloatArray:
    reference_vector = np.asarray(reference, dtype=np.float64)
    action_vector = np.asarray(action, dtype=np.float64)
    metric = np.asarray(metric_diagonal, dtype=np.float64)
    if (
        reference_vector.shape != action_vector.shape
        or metric.shape != reference_vector.shape
        or not np.all(np.isfinite([*reference_vector, *action_vector, *metric]))
        or np.any(metric <= 0.0)
    ):
        raise ValueError("actions or metric are invalid")
    reference_cost = float(reference_vector @ (metric * reference_vector))
    action_cost = float(action_vector @ (metric * action_vector))
    if reference_cost <= 0.0 or action_cost <= 0.0:
        raise ValueError("actions or metric are invalid")
    return action_vector * np.sqrt(reference_cost / action_cost)
