import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _complete_groups(
    dataset: str,
    rows: Iterable[tuple[str, str, str, Mapping[str, object]]],
    labels: tuple[str, str, str],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for group_key, label, text, source in rows:
        if not group_key or label not in labels or not text:
            continue
        endpoint = {
            "source_id": _hash(source),
            "text": text,
        }
        existing = grouped[group_key].get(label)
        if existing is None or endpoint["source_id"] < existing["source_id"]:
            grouped[group_key][label] = endpoint
    results = []
    for group_key, endpoints in grouped.items():
        if set(endpoints) != set(labels):
            continue
        results.append(
            {
                "dataset": dataset,
                "group_id": _hash([dataset, group_key]),
                "group_key": group_key,
                "endpoints": {label: endpoints[label] for label in labels},
            }
        )
    return sorted(results, key=lambda group: group["group_id"])


def normbank_groups(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    prepared = []
    for row in rows:
        setting = _clean(row.get("setting", ""))
        behavior = _clean(row.get("behavior", ""))
        constraints = _clean(row.get("constraints", ""))
        label = _clean(row.get("norm", "")).lower()
        group_key = f"{setting}\n{behavior}"
        text = f"Setting: {setting}. Behavior: {behavior}. Constraints: {constraints}."
        prepared.append((group_key, label, text, dict(row)))
    return _complete_groups("normbank", prepared, ("taboo", "normal", "expected"))


def mnli_groups(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    prepared = []
    for row in rows:
        premise = _clean(row.get("premise", ""))
        hypothesis = _clean(row.get("hypothesis", ""))
        try:
            label = labels[int(row.get("label", -1))]
        except (KeyError, TypeError, ValueError):
            continue
        text = f"Premise: {premise}\nHypothesis: {hypothesis}"
        prepared.append((premise.lower(), label, text, dict(row)))
    return _complete_groups(
        "mnli", prepared, ("entailment", "neutral", "contradiction")
    )


def sc101_groups(
    rows: Iterable[Mapping[str, object]], seed: int
) -> list[dict[str, object]]:
    pools: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        action = _clean(row.get("action", ""))
        try:
            judgment = int(row.get("action-moral-judgment", ""))
        except (TypeError, ValueError):
            continue
        if not action or judgment not in {-2, -1, 0, 1, 2}:
            continue
        label = "bad" if judgment < 0 else "good" if judgment > 0 else "ok"
        endpoint = {"source_id": _hash(dict(row)), "text": f"Action: {action}"}
        current = pools[label].get(action.lower())
        if current is None or endpoint["source_id"] < current["source_id"]:
            pools[label][action.lower()] = endpoint
    ordered = {
        label: sorted(
            endpoints.values(),
            key=lambda endpoint: _hash([seed, label, endpoint["source_id"]]),
        )
        for label, endpoints in pools.items()
    }
    count = min((len(ordered.get(label, ())) for label in ("bad", "ok", "good")), default=0)
    groups = []
    for index in range(count):
        endpoints = {label: ordered[label][index] for label in ("bad", "ok", "good")}
        source_ids = [endpoints[label]["source_id"] for label in ("bad", "ok", "good")]
        groups.append(
            {
                "dataset": "sc101",
                "group_id": _hash(["sc101", source_ids]),
                "group_key": "\n".join(source_ids),
                "endpoints": endpoints,
            }
        )
    return sorted(groups, key=lambda group: group["group_id"])


def allocate_groups(
    groups: Iterable[Mapping[str, object]],
    sizes: Mapping[str, int],
    seed: int,
) -> dict[str, list[Mapping[str, object]]]:
    requested = sum(sizes.values())
    ordered = sorted(
        groups,
        key=lambda group: _hash([seed, group["dataset"], group["group_id"]]),
    )
    if any(size <= 0 for size in sizes.values()) or len(ordered) < requested:
        raise ValueError("allocation sizes are invalid or exceed eligible groups")
    allocation = {}
    offset = 0
    for split, size in sizes.items():
        allocation[split] = ordered[offset : offset + size]
        offset += size
    return allocation
