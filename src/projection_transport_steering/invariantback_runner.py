from itertools import permutations

import torch
from torch import Tensor, nn

CONSTRUCTION_ROW_ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
HELDOUT_ROW_ORDERS = ((0, 2, 1), (2, 1, 0), (1, 0, 2))
LABELS = {
    "normbank": ("taboo", "normal", "expected"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "sc101": ("bad", "ok", "good"),
}
DESCRIPTIONS = {
    "normbank": {
        "taboo": "socially forbidden or unacceptable",
        "normal": "socially ordinary or acceptable",
        "expected": "socially expected or encouraged",
    },
    "mnli": {
        "entailment": "the hypothesis follows from the premise",
        "neutral": "neither entailment nor contradiction",
        "contradiction": "the hypothesis conflicts with the premise",
    },
    "sc101": {
        "bad": "socially inappropriate or harmful",
        "ok": "socially neutral or acceptable",
        "good": "socially appropriate or beneficial",
    },
}
QUESTIONS = {
    "normbank": (
        "In this context, how should the behavior be classified?",
        "What is this behavior's social-norm status in the stated context?",
        "Choose the norm label that best fits the behavior in context.",
    ),
    "mnli": (
        "What is the logical relationship between the premise and hypothesis?",
        "Classify whether the premise entails, contradicts, or is neutral toward the hypothesis.",
        "Choose the inference label that best describes this pair.",
    ),
    "sc101": (
        "Is this action bad, ok, or good?",
        "How should this action be judged socially?",
        "Choose the social judgment that best fits this action.",
    ),
}
INSTRUCTIONS = (
    "Return exactly one answer.",
    "Provide one classification only.",
    "Give exactly one label.",
)


def semantic_mappings(dataset: str) -> tuple[tuple[str, str, str], ...]:
    if dataset not in LABELS:
        raise ValueError("unknown dataset")
    return tuple(permutations(LABELS[dataset]))


def sentinel_cases() -> list[dict[str, object]]:
    cases = []
    for dataset, labels in LABELS.items():
        for source_index, target_index in ((0, 1), (0, 2), (1, 2)):
            for layer in (12, 18, 24):
                source = labels[source_index]
                target = labels[target_index]
                cases.append(
                    {
                        "case_id": f"{dataset}-{source}-to-{target}-layer{layer}",
                        "dataset": dataset,
                        "layer": layer,
                        "source": source,
                        "target": target,
                    }
                )
    return cases


def render_prompt(
    dataset: str,
    text: str,
    mapping: tuple[str, str, str],
    vocabulary: tuple[str, str, str],
    row_order: tuple[int, int, int],
    paraphrase: int,
) -> tuple[str, dict[str, str]]:
    if (
        dataset not in LABELS
        or set(mapping) != set(LABELS[dataset])
        or len(set(vocabulary)) != 3
        or set(row_order) != {0, 1, 2}
        or paraphrase not in {0, 1, 2}
    ):
        raise ValueError("prompt encoding is invalid")
    title = {
        "normbank": "You are classifying a behavior under a social-norm context.",
        "mnli": "You are classifying a natural-language inference relation.",
        "sc101": "You are judging a social norm.",
    }[dataset]
    rows = [
        f"{vocabulary[index]}. {mapping[index]} - {DESCRIPTIONS[dataset][mapping[index]]}"
        for index in row_order
    ]
    prompt = "\n".join(
        (
            title,
            text,
            f"Question: {QUESTIONS[dataset][paraphrase]}",
            *rows,
            INSTRUCTIONS[paraphrase],
            "Answer:",
        )
    )
    candidates = {mapping[index]: f" {vocabulary[index]}" for index in range(3)}
    return prompt, candidates


def render_mapping_check(
    dataset: str,
    mapping: tuple[str, str, str],
    vocabulary: tuple[str, str, str],
    label: str,
    phrasing: int,
) -> tuple[str, dict[str, str]]:
    if (
        dataset not in LABELS
        or set(mapping) != set(LABELS[dataset])
        or len(set(vocabulary)) != 3
        or label not in LABELS[dataset]
        or phrasing not in {0, 1, 2}
    ):
        raise ValueError("mapping check is invalid")
    questions = (
        f"Which identifier means {label}?",
        f"Return the identifier assigned to {label}.",
        f"If the correct category is {label}, what should be returned?",
    )
    key = [f"{vocabulary[index]} means {mapping[index]}" for index in range(3)]
    prompt = "\n".join(
        (
            "Use this temporary answer key:",
            *key,
            questions[phrasing],
            f"Return exactly one identifier from: {', '.join(vocabulary)}.",
            "Answer:",
        )
    )
    candidates = {mapping[index]: f" {vocabulary[index]}" for index in range(3)}
    return prompt, candidates


def candidate_mean_log_likelihood(
    logits: Tensor,
    candidate_ids: Tensor,
    candidate_mask: Tensor,
    prefix_lengths: Tensor,
) -> Tensor:
    if (
        logits.ndim != 3
        or candidate_ids.ndim != 2
        or candidate_mask.shape != candidate_ids.shape
        or candidate_mask.dtype != torch.bool
        or prefix_lengths.shape != (logits.shape[0],)
        or candidate_ids.shape[0] != logits.shape[0]
    ):
        raise ValueError("candidate scoring tensors are invalid")
    offsets = torch.arange(candidate_ids.shape[1], device=logits.device)
    positions = prefix_lengths.to(logits.device).unsqueeze(1) - 1 + offsets
    if torch.any(positions[candidate_mask] < 0) or torch.any(
        positions[candidate_mask] >= logits.shape[1]
    ):
        raise ValueError("candidate positions are outside logits")
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    token_logits = logits[rows, positions]
    token_scores = torch.log_softmax(token_logits.float(), dim=-1)
    selected = token_scores.gather(-1, candidate_ids.to(logits.device).unsqueeze(-1)).squeeze(-1)
    mask = candidate_mask.to(logits.device)
    return (selected * mask).sum() / mask.sum()


def encode_candidate_batch(
    tokenizer: object,
    prompts: list[str],
    candidates: list[str],
    device: torch.device | str,
) -> dict[str, Tensor]:
    if not prompts or len(prompts) != len(candidates):
        raise ValueError("prompts and candidates must be nonempty and aligned")
    prefixes = []
    full_rows = []
    candidate_rows = []
    for prompt, candidate in zip(prompts, candidates, strict=True):
        prefix = tokenizer.encode(prompt, add_special_tokens=True)
        full = tokenizer.encode(prompt + candidate, add_special_tokens=True)
        if full[: len(prefix)] != prefix or len(full) == len(prefix):
            raise RuntimeError("candidate tokenization does not preserve the prompt prefix")
        prefixes.append(len(prefix))
        full_rows.append(full)
        candidate_rows.append(full[len(prefix) :])
    full_width = max(map(len, full_rows))
    candidate_width = max(map(len, candidate_rows))
    input_ids = torch.full(
        (len(full_rows), full_width),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    candidate_ids = torch.zeros((len(full_rows), candidate_width), dtype=torch.long)
    candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
    for index, (full, candidate) in enumerate(zip(full_rows, candidate_rows, strict=True)):
        input_ids[index, : len(full)] = torch.tensor(full)
        attention_mask[index, : len(full)] = 1
        candidate_ids[index, : len(candidate)] = torch.tensor(candidate)
        candidate_mask[index, : len(candidate)] = True
    return {
        "attention_mask": attention_mask.to(device),
        "candidate_ids": candidate_ids.to(device),
        "candidate_mask": candidate_mask.to(device),
        "input_ids": input_ids.to(device),
        "prefix_lengths": torch.tensor(prefixes, device=device),
    }


class IndexedAdditiveAction(nn.Module):
    def __init__(self, vector: Tensor, positions: Tensor) -> None:
        super().__init__()
        if vector.ndim != 1 or positions.ndim != 1:
            raise ValueError("indexed action inputs are invalid")
        self.vector = vector
        self.positions = positions

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[0] != self.positions.shape[0]:
            raise ValueError("indexed action does not match hidden states")
        positions = self.positions.to(hidden.device)
        if torch.any(positions < 0) or torch.any(positions >= hidden.shape[1]):
            raise ValueError("indexed action position is outside hidden states")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        result = hidden.clone()
        result[rows, positions] += self.vector.to(device=hidden.device, dtype=hidden.dtype)
        return result
