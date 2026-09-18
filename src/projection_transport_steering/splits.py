import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


def normalize_problem(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("problem must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise ValueError("problem must be nonempty")
    return normalized


def problem_clusters(problems: Iterable[str], *, query_count: int | None = None) -> list[str]:
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn import config_context
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    normalized = [normalize_problem(text) for text in problems]
    if query_count is not None and not 0 <= query_count <= len(normalized):
        raise ValueError("query count is outside the problem frame")
    queried = set(normalized[:query_count])
    unique = sorted(set(normalized))
    identities = {text: hashlib.sha256(text.encode()).hexdigest() for text in unique}
    eligible = [text for text in unique if len(text) >= 5]
    if any(text in queried for text in eligible):
        vectors = TfidfVectorizer(
            analyzer="char", ngram_range=(5, 5), lowercase=False, norm="l2", smooth_idf=True
        ).fit_transform(eligible)
        indices = np.array([index for index, text in enumerate(eligible) if text in queried])
        with config_context(working_memory=128):
            graph = (
                NearestNeighbors(radius=0.05, metric="cosine", algorithm="brute", n_jobs=1)
                .fit(vectors)
                .radius_neighbors_graph(vectors[indices], mode="connectivity")
                .tocoo()
            )
        graph = coo_matrix(
            (graph.data, (indices[graph.row], graph.col)), shape=(len(eligible), len(eligible))
        )
        _, labels = connected_components(graph, directed=False)
        representatives = {}
        for text, label in zip(eligible, labels, strict=True):
            representatives[label] = min(
                representatives.get(label, identities[text]), identities[text]
            )
        identities.update(
            (text, representatives[label]) for text, label in zip(eligible, labels, strict=True)
        )
    return [identities[text] for text in normalized]


def stable_group_id(
    dataset_sha: str,
    config: str,
    split: str,
    row: Mapping[str, Any],
) -> str:
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = "\0".join((dataset_sha, config, split, canonical)).encode()
    return hashlib.sha256(payload).hexdigest()


def allocate_groups(
    group_ids: Iterable[str],
    quotas: Mapping[str, int],
    salt: str,
) -> dict[str, str]:
    if any(not isinstance(size, int) or size < 0 for size in quotas.values()):
        raise ValueError("quotas must be nonnegative integers")
    groups = set(group_ids)
    required = sum(quotas.values())
    if len(groups) < required:
        raise ValueError("insufficient unique groups for requested quotas")
    ordered = sorted(
        groups,
        key=lambda group: (hashlib.sha256(f"{salt}\0{group}".encode()).digest(), group),
    )
    allocation: dict[str, str] = {}
    offset = 0
    for stage, size in quotas.items():
        for group in ordered[offset : offset + size]:
            allocation[group] = stage
        offset += size
    return allocation
