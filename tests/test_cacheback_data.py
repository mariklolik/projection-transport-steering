import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def data_module():
    return importlib.import_module("materialize_cacheback_data")


def test_stage_quotas_balance_development_and_pilot_without_reusing_v18():
    assert data_module().stage_quotas("third") == {
        "smoke": 1,
        "sentinel": 3,
        "development": 8,
        "pilot": 16,
    }
    assert data_module().stage_quotas("ing") == {
        "sentinel": 3,
        "development": 8,
        "pilot": 16,
    }
    assert data_module().stage_quotas("past") == {
        "sentinel": 2,
        "development": 8,
        "pilot": 16,
    }


def test_allocate_candidates_excludes_prior_context_hashes_and_is_deterministic():
    candidates = [
        {"context_sha256": f"hash-{index}", "token_ids": [index]}
        for index in range(40)
    ]
    excluded = {"hash-0", "hash-1"}

    first = data_module().allocate_candidates("third", candidates, excluded)
    second = data_module().allocate_candidates("third", list(reversed(candidates)), excluded)

    assert first == second
    assert len(first) == 28
    assert not excluded & {row["context_sha256"] for row in first}
    assert {row["stage"] for row in first} == {"smoke", "sentinel", "development", "pilot"}


def test_prior_packet_reuses_pinned_mappings_and_excludes_every_prior_stage(tmp_path):
    path = tmp_path / "data.json"
    packet = {
        "contexts": {
            "third": [{"context_sha256": "a"}],
            "ing": [{"context_sha256": "b"}],
        },
        "mapping_sha256": {"third": "mapping-hash"},
        "mappings": {"third": {"base_ids": [1]}},
    }
    path.write_text(json.dumps(packet))

    loaded, excluded = data_module().prior_packet(path)

    assert loaded == packet
    assert excluded == {"a", "b"}


def test_validate_global_disjoint_rejects_cross_concept_state_reuse():
    contexts = {
        "third": [{"context_sha256": "a", "dataset_index": 10}],
        "ing": [{"context_sha256": "b", "dataset_index": 10}],
        "past": [{"context_sha256": "a", "dataset_index": 11}],
    }

    with pytest.raises(ValueError, match="globally disjoint"):
        data_module().validate_global_disjoint(contexts)
