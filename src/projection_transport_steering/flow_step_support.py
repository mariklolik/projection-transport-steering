from itertools import combinations

import torch
from torch import Tensor

from projection_transport_steering.residual_flow import flas_model_logits
from projection_transport_steering.semantic_outcome_evaluation import (
    row_forward_kl,
    row_mean_log_likelihood,
)
from projection_transport_steering.semantic_outcome_runtime import (
    encode_prompt_candidates,
)
from projection_transport_steering.semantic_outcome_sentinel import (
    evaluation_batch,
    model_logits,
)


def condition_specifications() -> list[dict[str, object]]:
    subsets = [
        subset
        for size in (1, 2, 3)
        for subset in combinations(range(3), size)
    ]
    raw = [
        {
            "family": "raw",
            "intervention": (
                None
                if len(subset) == 3
                else {"steps": list(subset)}
            ),
            "n_steps": 3,
            "name": "full" if len(subset) == 3 else f"raw_{''.join(map(str, subset))}",
            "subset": list(subset),
        }
        for subset in subsets
    ]
    equal = [
        {
            "family": "equal_time",
            "intervention": {
                "scale": 3.0 / len(subset),
                "steps": list(subset),
            },
            "n_steps": 3,
            "name": f"equal_{''.join(map(str, subset))}",
            "subset": list(subset),
        }
        for size in (1, 2)
        for subset in combinations(range(3), size)
    ]
    return raw + equal + [
        {
            "family": "official_step_count",
            "intervention": None,
            "n_steps": 2,
            "name": "official_n2",
            "subset": None,
        }
    ]


def _contrasts(scores: Tensor) -> Tensor:
    return scores[0::2] - scores[1::2]


def _condition(
    generator: object,
    batches: tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]],
    base_logits: tuple[Tensor, Tensor, Tensor],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    specification: dict[str, object],
) -> dict[str, object]:
    logits = tuple(
        flas_model_logits(
            generator,
            batch,
            concept_hidden,
            concept_mask,
            n_steps=int(specification["n_steps"]),
            intervention=specification["intervention"],
        )
        for batch in batches
    )
    if not all(bool(torch.isfinite(value).all()) for value in logits):
        raise RuntimeError(f"nonfinite logits in {specification['name']}")
    base_construction = row_mean_log_likelihood(base_logits[0], batches[0])
    base_heldout = row_mean_log_likelihood(base_logits[1], batches[1])
    base_neutral = row_mean_log_likelihood(base_logits[2], batches[2])
    construction = row_mean_log_likelihood(logits[0], batches[0])
    heldout = row_mean_log_likelihood(logits[1], batches[1])
    neutral = row_mean_log_likelihood(logits[2], batches[2])
    return {
        "construction_score_changes": (
            _contrasts(construction) - _contrasts(base_construction)
        ).float().cpu().tolist(),
        "heldout_score_changes": (
            _contrasts(heldout) - _contrasts(base_heldout)
        ).float().cpu().tolist(),
        "neutral_forward_kl": row_forward_kl(
            base_logits[2], logits[2], batches[2]
        ).float().cpu().tolist(),
        "neutral_score_changes": (neutral - base_neutral).float().cpu().tolist(),
    }


def evaluate_flow_step_support(
    generator: object,
    tokenizer: object,
    construction: list[dict[str, object]],
    evaluation: list[dict[str, object]],
    neutral: list[dict[str, object]],
    concept_hidden: Tensor,
    concept_mask: Tensor,
    device: torch.device,
) -> dict[str, object]:
    batches = (
        evaluation_batch(tokenizer, construction, device),
        evaluation_batch(tokenizer, evaluation, device),
        encode_prompt_candidates(
            tokenizer,
            [row["input"] for row in neutral],
            [row["output"] for row in neutral],
            device,
        )[0],
    )
    with torch.inference_mode():
        base_logits = tuple(
            model_logits(generator.llm, batch, generator.layer) for batch in batches
        )
        specifications = condition_specifications()
        conditions = {
            str(specification["name"]): _condition(
                generator,
                batches,
                base_logits,
                concept_hidden,
                concept_mask,
                specification,
            )
            for specification in specifications
        }
    return {
        "base": {
            "construction_contrasts": _contrasts(
                row_mean_log_likelihood(base_logits[0], batches[0])
            ).float().cpu().tolist(),
            "heldout_contrasts": _contrasts(
                row_mean_log_likelihood(base_logits[1], batches[1])
            ).float().cpu().tolist(),
            "neutral_scores": row_mean_log_likelihood(
                base_logits[2], batches[2]
            ).float().cpu().tolist(),
        },
        "condition_order": [specification["name"] for specification in specifications],
        "conditions": conditions,
    }
