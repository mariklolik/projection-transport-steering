from __future__ import annotations

import json
import random
import re

from behaviour_specific.overconfidence.mmlu.data import load_mmlu
from general.paths import DATA_DIR


def _cache(name: str):
    return DATA_DIR / f"{name}.jsonl"


def _load_cached(name: str, build_rows) -> list[dict]:
    path = _cache(name)
    if not path.exists():
        rows = build_rows()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_arc(n: int | None = None, seed: int = 0, split: str = "test") -> list[dict]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        rows = []
        for i, r in enumerate(ds):
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            if len(texts) != 4 or r["answerKey"] not in labels:
                continue
            rows.append({"id": f"arc-{i}" if split == "test" else f"arc{split}-{i}", "subject": "arc_challenge",
                         "question": r["question"], "options": texts, "answer_idx": labels.index(r["answerKey"])})
        return rows

    records = _load_cached("arc_challenge_test" if split == "test" else f"arc_challenge_{split}", build)
    random.Random(seed).shuffle(records)
    return records if n is None else records[:n]


GSM_ANSWER_RE = re.compile(r"####\s*([\-0-9,\.]+)")


def gsm_gold(answer_text: str) -> int | None:
    m = GSM_ANSWER_RE.search(answer_text)
    if not m:
        return None
    raw = m.group(1).replace(",", "").rstrip(".")
    try:
        return int(raw)
    except ValueError:
        return None


def gsm_distractors(gold: int, idx: int) -> list[int]:
    rng = random.Random(1000 + idx)
    cands = [gold + 1, gold - 1, gold + 2, gold - 2, gold + 10, gold - 10,
             gold * 2, max(0, gold // 2), round(gold * 1.5), round(gold * 0.8),
             gold + rng.randint(3, 9), gold - rng.randint(3, 9)]
    seen, out = {gold}, []
    for c in cands if rng.random() < 0.5 else cands[::-1]:
        if c not in seen:
            seen.add(c)
            out.append(c)
    rng.shuffle(out)
    return out[:3]


def load_gsm8k_mcq(n: int | None = None, seed: int = 0) -> list[dict]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = []
        for i, r in enumerate(ds):
            gold = gsm_gold(r["answer"])
            if gold is None:
                continue
            opts = [gold] + gsm_distractors(gold, i)
            if len(opts) < 4:
                continue
            rng = random.Random(2000 + i)
            rng.shuffle(opts)
            rows.append({"id": f"gsm8k-{i}", "subject": "gsm8k", "question": r["question"],
                         "options": [str(o) for o in opts], "answer_idx": opts.index(gold)})
        return rows

    records = _load_cached("gsm8k_test_mcq", build)
    random.Random(seed).shuffle(records)
    return records if n is None else records[:n]


def load_gpqa(n: int | None = None, seed: int = 0, subset: str = "gpqa_main") -> list[dict]:
    def build():
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", subset, split="train")
        rows = []
        for i, r in enumerate(ds):
            opts = [r["Correct Answer"].strip(), r["Incorrect Answer 1"].strip(),
                    r["Incorrect Answer 2"].strip(), r["Incorrect Answer 3"].strip()]
            rng = random.Random(3000 + i)
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [opts[j] for j in order]
            rows.append({"id": f"gpqa-{i}", "subject": r.get("High-level domain", "gpqa"),
                         "question": r["Question"].strip(), "options": shuffled,
                         "answer_idx": order.index(0)})
        return rows

    records = _load_cached(f"{subset}_mcq", build)
    random.Random(seed).shuffle(records)
    return records if n is None else records[:n]


LOADERS = {"mmlu": load_mmlu, "arc": load_arc, "gsm8k": load_gsm8k_mcq, "gpqa": load_gpqa}


def load_records(benchmark: str, n: int | None = None, seed: int = 0) -> list[dict]:
    if benchmark not in LOADERS:
        raise ValueError(f"unknown benchmark {benchmark!r}; have {sorted(LOADERS)}")
    return LOADERS[benchmark](n=n, seed=seed)


def _selftest():
    assert gsm_gold("blah\n#### 1,234") == 1234
    assert gsm_gold("#### 7.") == 7
    assert gsm_gold("no answer") is None
    for idx in range(50):
        ds = gsm_distractors(42, idx)
        assert len(ds) == 3 and 42 not in ds and len(set(ds)) == 3
    ds1, ds2 = gsm_distractors(42, 5), gsm_distractors(42, 5)
    assert ds1 == ds2
    print("benchmarks self-test passed")


if __name__ == "__main__":
    _selftest()
