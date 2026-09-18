import copy
import importlib
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def probe():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_action_source_probe")


@pytest.fixture
def requests():
    return [
        {"request_id": "q0:0", "input_ids": [10, 11], "seed": 17},
        {"request_id": "q1:0", "input_ids": [12], "seed": 18},
    ]


@pytest.fixture
def responses():
    return [
        {
            "output_ids": [41, 99],
            "meta_info": {
                "id": "q0:0",
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "finish_reason": {"type": "stop", "matched": 99},
                "num_retractions": 0,
                "weight_version": "default",
            },
        },
        {
            "output_ids": [42, 43, 44],
            "meta_info": {
                "id": "q1:0",
                "prompt_tokens": 1,
                "completion_tokens": 3,
                "finish_reason": {"type": "length", "length": 3},
                "num_retractions": 0,
                "weight_version": "default",
            },
        },
    ]


def test_output_join_uses_ids_and_preserves_native_tokens(probe, requests, responses):
    original = copy.deepcopy(responses)
    result = probe.validate_outputs(requests, responses[::-1], [98, 99], 3, vocab_size=100)
    assert [r["request_id"] for r in result] == ["q0:0", "q1:0"]
    assert [r["terminated"] for r in result] == [True, False]
    assert result[0]["output_ids"] == [41, 99]
    assert result[1]["seed"] == 18
    assert result[0]["meta_info"] == responses[0]["meta_info"]
    assert responses == original


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "input_duplicate"])
def test_output_join_rejects_invalid_coverage(probe, requests, responses, change):
    if change == "missing":
        responses.pop()
    elif change == "duplicate":
        responses[1] = responses[0]
    elif change == "unknown":
        responses[1]["meta_info"]["id"] = "q2:0"
    else:
        requests[1] = requests[0]
    with pytest.raises(ValueError, match="coverage"):
        probe.validate_outputs(requests, responses, [98, 99], 3, vocab_size=100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_tokens", 8),
        ("completion_tokens", 1),
        ("finish_reason", {"type": "abort", "message": "OOM"}),
        ("finish_reason", {"type": "stop", "matched": 98}),
        ("finish_reason", {"type": "stop", "matched": "end"}),
        ("num_retractions", 1),
    ],
)
def test_output_join_rejects_inconsistent_metadata(probe, requests, responses, field, value):
    responses[0]["meta_info"][field] = value
    with pytest.raises(ValueError):
        probe.validate_outputs(requests, responses, [98, 99], 3, vocab_size=100)


@pytest.mark.parametrize("tokens", [[], [41], [99, 41], [99, 99], [41, True], [-1, 99]])
def test_output_join_rejects_trimmed_or_invalid_tokens(probe, requests, responses, tokens):
    responses[0]["output_ids"] = tokens
    responses[0]["meta_info"]["completion_tokens"] = len(tokens)
    with pytest.raises(ValueError):
        probe.validate_outputs(requests, responses, [98, 99], 3, vocab_size=100)


def test_output_join_rejects_short_length_stop(probe, requests, responses):
    responses[1]["output_ids"] = [42, 43]
    responses[1]["meta_info"]["completion_tokens"] = 2
    with pytest.raises(ValueError):
        probe.validate_outputs(requests, responses, [98, 99], 3, vocab_size=100)


def test_output_join_rejects_token_at_model_vocab_size(probe, requests, responses):
    responses[0]["output_ids"] = [100, 99]
    with pytest.raises(ValueError):
        probe.validate_outputs(requests, responses, [98, 99], 3, vocab_size=100)


def test_cleanup_error_cannot_suppress_receipt(probe, tmp_path):
    import json

    class FailingEngine:
        def shutdown(self):
            raise RuntimeError("shutdown failed")

    receipt = {"status": "failed", "error": "original"}
    path = tmp_path / "receipt.json"
    telemetry = (tmp_path / "gpu.csv").open("w")
    probe.finalize_receipt(receipt, path, time.perf_counter(), FailingEngine(), None, telemetry)
    saved = json.loads(path.read_text())
    assert telemetry.closed
    assert saved["status"] == "failed" and saved["error"] == "original"
    assert saved["cleanup_errors"] == ["engine: RuntimeError: shutdown failed"]
    assert saved["elapsed_seconds"] >= 0 and saved["finished_at"]


def test_frozen_preparation_reuses_only_fit_rows_and_rollout_seeds(probe):
    prepare = importlib.import_module("prepare_action_source_probe")
    from projection_transport_steering.outcome_score_runtime import rollout_seed

    root = Path(__file__).parents[1]
    packet = prepare.prepare_packet(root, root / "tmp/irc-qwen3-tokenizer-v1")
    rows = packet["requests"]
    assert len(rows) == len({r["request_id"] for r in rows}) == 32
    assert len({r["cluster_id"] for r in rows}) == 8
    assert {r["allocation"] for r in rows} == {"fit"}
    assert all(r["seed"] == rollout_seed(r["cluster_id"], r["rollout_index"]) for r in rows)
    assert all(r["input_ids"] and r["rollout_index"] in range(4) for r in rows)
    assert packet["sampling"]["max_new_tokens"] == 8192
    assert packet["stages"][2]["indices"] == [3, 2, 1, 0]
    assert packet["stages"][-1]["indices"] == [1]
