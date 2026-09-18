from math_verify.errors import TimeoutException
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from math_verify import LatexExtractionConfig, parse, verify
from numpy.typing import ArrayLike, NDArray

from projection_transport_steering.splits import allocate_groups, stable_group_id
from projection_transport_steering.transport import minimum_metric_update

FloatArray = NDArray[np.float64]
_AIME_ANSWER = re.compile(r"^\s*Final Answer:\s*(\d{1,3})\s*$", re.IGNORECASE | re.MULTILINE)
_AIME_BOXED_ANSWER = re.compile(
    r"\\boxed\s*\{\s*(\d{1,3})\s*\}\s*(?:\\[\])]|\$\$|\$)?\s*$",
    re.IGNORECASE,
)
_AIME_REWARD_LINE = re.compile(
    r"^\s*Final Answer:\s*(-?\d{1,12})\s*$", re.IGNORECASE | re.MULTILINE
)
_AIME_REWARD_BOX = re.compile(
    r"\\boxed\s*\{\s*(-?\d{1,12})\s*\}\s*(?:\\[\])]|\$\$|\$)?\s*$",
    re.IGNORECASE,
)
_AIME_MATH_VERIFY_CONFIG = (LatexExtractionConfig(try_extract_without_anchor=False),)
_MATH_VERIFY_CONFIG = (
    replace(
        _AIME_MATH_VERIFY_CONFIG[0],
        normalization_config=replace(_AIME_MATH_VERIFY_CONFIG[0].normalization_config, units=False),
    ),
)
_AIME_QUOTAS = {
    "basis": 32,
    "fit": 256,
    "calibration": 128,
    "validation": 128,
    "reserve": 431,
}


def extract_aime_answer(completion: str) -> int | None:
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    stripped = completion.rstrip()
    matches = list(_AIME_ANSWER.finditer(stripped))
    if len(matches) != 1 or matches[0].end() != len(stripped):
        return None
    answer = int(matches[0].group(1))
    return answer if 0 <= answer <= 999 else None


def extract_aime_benchmark_answer(completion: str) -> int | None:
    answer = extract_aime_answer(completion)
    if answer is not None:
        return answer
    match = _AIME_BOXED_ANSWER.search(completion)
    if match is None:
        return None
    answer = int(match.group(1))
    return answer if 0 <= answer <= 999 else None


def extract_aime_reward_answer(completion: str) -> int | None:
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    stripped = completion.rstrip()
    matches = list(_AIME_REWARD_LINE.finditer(stripped))
    if len(matches) == 1 and matches[0].end() == len(stripped):
        return int(matches[0].group(1))
    match = _AIME_REWARD_BOX.search(stripped)
    return int(match.group(1)) if match is not None else None


def score_aime_completion(completion: str, answer: int) -> dict[str, object]:
    if not isinstance(answer, int) or not 0 <= answer <= 999:
        raise ValueError("answer must be an integer between zero and 999")
    extracted = extract_aime_answer(completion)
    return {
        "correct": extracted == answer if extracted is not None else False,
        "extracted_answer": extracted,
        "parse_status": "pass" if extracted is not None else "fail",
    }


def score_aime_benchmark_completion(completion: str, answer: int) -> dict[str, object]:
    if not isinstance(answer, int) or not 0 <= answer <= 999:
        raise ValueError("answer must be an integer between zero and 999")
    extracted = extract_aime_benchmark_answer(completion)
    return {
        "correct": extracted == answer if extracted is not None else False,
        "extracted_answer": extracted,
        "parse_status": "pass" if extracted is not None else "fail",
    }


def score_aime_reward_completion(completion: str, answer: int) -> dict[str, object]:
    if not isinstance(answer, int) or not 0 <= answer <= 999:
        raise ValueError("answer must be an integer between zero and 999")
    extracted = extract_aime_reward_answer(completion)
    return {
        "correct": extracted == answer if extracted is not None else False,
        "extracted_answer": extracted,
        "parse_status": "pass" if extracted is not None else "fail",
    }


def score_aime_math_verify_completion(completion: str, answer: int) -> dict[str, object]:
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    if not isinstance(answer, int) or not 0 <= answer <= 999:
        raise ValueError("answer must be an integer between zero and 999")
    parsed = parse(
        completion,
        extraction_config=_AIME_MATH_VERIFY_CONFIG,
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=5,
        raise_on_error=True,
    )
    if not parsed:
        return {"correct": False, "extracted_answer": None, "parse_status": "fail"}
    gold = parse(str(answer), parsing_timeout=5, raise_on_error=True)
    candidate = parsed[0]
    extracted = int(candidate) if getattr(candidate, "is_Integer", False) else str(candidate)
    return {
        "correct": verify(
            gold,
            parsed,
            strict=True,
            timeout_seconds=5,
            raise_on_error=True,
        ),
        "extracted_answer": extracted,
        "parse_status": "pass",
    }


def score_math_completion(
    completion: str,
    gold_solution: str,
    *,
    terminated: bool,
    extract_answer: Callable[[str], str | None],
) -> dict[str, object]:
    if not isinstance(completion, str) or not isinstance(gold_solution, str):
        raise TypeError("completion and gold solution must be strings")
    if not isinstance(terminated, bool):
        raise TypeError("terminated must be a boolean")
    options = {
        "extraction_config": _MATH_VERIFY_CONFIG,
        "fallback_mode": "no_fallback",
        "extraction_mode": "first_match",
        "parsing_timeout": 5,
        "raise_on_error": True,
    }
    gold_text = (extract_answer(gold_solution) or "").replace("\\fbox", "\\boxed", 1)
    gold = parse(gold_text, **options)
    if not gold:
        raise ValueError("gold solution has no parseable mathematical answer")
    prediction = (extract_answer(completion) or "").replace("\\fbox", "\\boxed", 1)
    parsed = parse(prediction, **options)
    return {
        "correct": bool(
            terminated
            and parsed
            and verify(gold, parsed, strict=True, timeout_seconds=5, raise_on_error=True)
        ),
        "extracted_answer": str(parsed[0]) if parsed else None,
        "parse_status": "pass" if parsed else "fail",
    }


def score_math_author_completion(
    completion: str,
    gold_solution: str,
    *,
    terminated: bool,
    extract_answer: Callable[[str], str | None],
    equivalent: Callable[[str, str], bool],
) -> dict[str, object]:
    if not isinstance(completion, str) or not isinstance(gold_solution, str):
        raise TypeError("completion and gold solution must be strings")
    if not isinstance(terminated, bool):
        raise TypeError("terminated must be a boolean")
    gold = extract_answer(gold_solution)
    if not gold or not gold.strip():
        raise ValueError("gold solution has no nonempty author-extracted answer")
    prediction = extract_answer(completion)
    valid = bool(prediction and prediction.strip())
    return {
        "correct": bool(terminated and valid and equivalent(prediction, gold)),
        "extracted_answer": prediction if valid else None,
        "parse_status": "pass" if valid else "fail",
    }


def _aime_identity(row: Mapping[str, Any]) -> dict[str, object]:
    try:
        year = int(row["year"])
        index = int(row["index"])
        problem = str(row["problem"])
        answer = int(row["answer"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid AIME row") from error
    if not problem or not 0 <= answer <= 999:
        raise ValueError("invalid AIME row")
    return {"answer": answer, "index": index, "problem": problem, "year": year}


def allocate_aime_rows(
    rows: Iterable[Mapping[str, Any]],
    dataset_sha: str,
) -> list[dict[str, object]]:
    identities = [_aime_identity(row) for row in rows]
    historical = [row for row in identities if int(row["year"]) <= 2023]
    if len(historical) != 975:
        raise ValueError("historical AIME allocation requires exactly 975 rows")
    if sum(int(row["year"]) == 2024 for row in identities) != 30:
        raise ValueError("AIME 2024 allocation requires exactly 30 rows")
    if sum(int(row["year"]) == 2025 for row in identities) != 30:
        raise ValueError("AIME 2025 allocation requires exactly 30 rows")
    group_ids = {
        stable_group_id(dataset_sha, "aime", "historical-through-2023", row): row
        for row in historical
    }
    if len(group_ids) != len(historical):
        raise ValueError("AIME group identities must be unique")
    allocation = allocate_groups(group_ids, _AIME_QUOTAS, "ost-v1")
    result = [
        {**row, "allocation": allocation[group_id], "group_id": group_id}
        for group_id, row in group_ids.items()
    ]
    for year, stage in ((2024, "development"), (2025, "pilot")):
        for row in identities:
            if int(row["year"]) == year:
                group_id = stable_group_id(dataset_sha, "aime", f"aime-{year}", row)
                result.append({**row, "allocation": stage, "group_id": group_id})
    return sorted(result, key=lambda row: str(row["group_id"]))


def partition_aime_rows(
    rows: Iterable[Mapping[str, Any]],
    dataset_sha: str,
) -> dict[str, list[dict[str, object]]]:
    allocated = allocate_aime_rows(rows, dataset_sha)
    stages = (*_AIME_QUOTAS, "development", "pilot")
    return {stage: [row for row in allocated if row["allocation"] == stage] for stage in stages}


def minimum_score_update(
    gradient: ArrayLike,
    metric: ArrayLike,
    increment: float,
) -> FloatArray:
    gradient_array = np.asarray(gradient, dtype=np.float64)
    if gradient_array.ndim != 1 or not np.all(np.isfinite(gradient_array)):
        raise ValueError("gradient must be a finite vector")
    if not np.isfinite(increment) or increment < 0.0:
        raise ValueError("increment must be finite and nonnegative")
    update = minimum_metric_update(
        np.zeros((1, gradient_array.size)),
        gradient_array[:, None],
        np.array([[increment]], dtype=np.float64),
        metric,
    )
    return update[0]


def score_math_outputs(completion, gold, terminated, supported, extract, equivalent, boxed) -> dict:
    result = {"primary": None, "sensitivity": None, "evaluator_errors": {}}
    try:
        result["primary"] = score_math_author_completion(
            completion, gold, terminated=terminated, extract_answer=extract, equivalent=equivalent
        )
    except (Exception, TimeoutException) as error:
        result["evaluator_errors"]["primary"] = type(error).__name__
    if supported:
        try:
            result["sensitivity"] = score_math_completion(
                completion, gold, terminated=terminated, extract_answer=boxed
            )
        except (Exception, TimeoutException) as error:
            result["evaluator_errors"]["sensitivity"] = type(error).__name__
    return result
