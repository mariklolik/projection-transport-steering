from __future__ import annotations

from general.reasoning import parse_boxed_letter as parse_boxed_letter

import json
import random

from general.paths import DATA_DIR

LETTERS = ["A", "B", "C", "D"]
CONFIG, SPLIT = "all", "test"


def _cache_path():
    return DATA_DIR / f"mmlu_{CONFIG}_{SPLIT}.jsonl"


def _download_to_cache():

    from datasets import load_dataset

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

    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(record["options"]))
    return f"{MCQ_INSTRUCTION}\n\nQuestion: {record['question']}\nOptions:\n{opts}"


def correct_letter(record: dict) -> str:

    return LETTERS[record["answer_idx"]]
