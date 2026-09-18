# MMLU loading and the reasoning MCQ prompt (benchmark-specific).
#
# The model thinks step-by-step inside <think>...</think> and then answers
# \boxed{letter}. The full split is downloaded ONCE to data_cache/ and read from
# disk afterwards (no network on later calls — metered-connection friendly).
#
# A record is a plain dict:
#   {"id", "subject", "question", "options": [4 strings], "answer_idx": 0..3}
#
# `python -m behaviour_specific.overconfidence.mmlu.data` runs self-tests and
# prints one formatted prompt (needs the cache or one download).

from __future__ import annotations

import json
import random
import string

from general.paths import DATA_DIR
from general.reasoning import parse_boxed_letter, split_think  # re-exported for callers

LETTERS = ["A", "B", "C", "D"]
CONFIG, SPLIT = "all", "test"


def _cache_path():
    return DATA_DIR / f"mmlu_{CONFIG}_{SPLIT}.jsonl"


def _download_to_cache():
    """One-time: pull the whole split from HuggingFace and store it as jsonl."""
    from datasets import load_dataset  # lazy so light parts stay import-free

    ds = load_dataset("cais/mmlu", CONFIG, split=SPLIT)
    rows = [{"subject": r["subject"], "question": r["question"],
             "choices": list(r["choices"]), "answer": int(r["answer"])} for r in ds]
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def load_mmlu(n: int | None = None, seed: int = 0, subjects: list[str] | None = None) -> list[dict]:
    """Load MMLU MCQ records, shuffled by `seed`. `n=None` -> the WHOLE split.

    Downloaded once to data_cache/ then read from disk with no network. `id` is
    derived from the stable pre-shuffle position, so a record keeps the same id
    across calls regardless of `n`.
    """
    path = _cache_path()
    if not path.exists():
        print(f"[mmlu] downloading {CONFIG}/{SPLIT} once -> {path}")
        _download_to_cache()

    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [{"id": f"mmlu-{i}", "subject": r["subject"], "question": r["question"],
                "options": r["choices"], "answer_idx": r["answer"]} for i, r in enumerate(raw)]
    if subjects:
        records = [r for r in records if r["subject"] in subjects]

    random.Random(seed).shuffle(records)
    return records if n is None else records[:n]


MCQ_INSTRUCTION = (
    "Given the following multiple choice question, choose the single best answer "
    "(A, B, C, or D).\n"
    "Please think step-by-step and put your thinking inside <think>...</think> tokens.\n"
    "After </think>, give ONLY your final answer in the format: \\boxed{[letter]}"
)


def mcq_prompt(record: dict) -> str:
    """Format a record as the reasoning MCQ user prompt."""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(record["options"]))
    return f"{MCQ_INSTRUCTION}\n\nQuestion: {record['question']}\nOptions:\n{opts}"


def correct_letter(record: dict) -> str:
    """The gold answer letter for a record."""
    return LETTERS[record["answer_idx"]]


if __name__ == "__main__":
    assert set(string.ascii_uppercase[:4]) == set(LETTERS)
    # re-exports work
    assert split_think("<think>x</think>y")[1] == "y"
    assert parse_boxed_letter("<think>a</think>\\boxed{C}") == "C"

    demo = {"id": "demo-0", "subject": "arithmetic", "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"], "answer_idx": 1}
    print(mcq_prompt(demo))
    print("gold:", correct_letter(demo))
    print("full split size:", len(load_mmlu()))
    print("behaviour.overconfidence.mmlu.data self-tests passed")
