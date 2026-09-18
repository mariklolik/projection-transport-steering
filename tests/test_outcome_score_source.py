import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from projection_transport_steering.outcome_score_sentinel import (
    select_sentinel_rows,
    sentinel_protocol,
)
from projection_transport_steering.outcome_score_source import summarize_source_stage


MODEL_SHA256 = "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"
EVALUATOR = "math-verify==0.9.0:latex-anchored-no-fallback-first-match"


def test_version_five_source_protocols_bind_allocations_and_rollouts():
    shared = {
        "answer_format": "math_verify_strict",
        "enable_thinking": False,
        "evaluator": EVALUATOR,
        "layer": 14,
        "max_new_tokens": 2048,
        "model_config_sha256": MODEL_SHA256,
    }

    assert sentinel_protocol(5, "basis_completion") == {
        **shared,
        "allocation": "basis",
        "group_count": 4,
        "rollout_indices": (2, 3),
        "source_sha256": "8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7",
    }
    assert sentinel_protocol(5, "fit") == {
        **shared,
        "allocation": "fit",
        "group_count": 32,
        "rollout_indices": (0, 1, 2, 3),
        "source_sha256": "abee6c7c85376e085756d2f60a427969105ce908abbe00a83027a55931d0fd1b",
    }


def test_select_sentinel_rows_accepts_only_the_declared_allocation():
    packet = {
        "allocation": "fit",
        "rows": [{"allocation": "fit", "group_id": "a", "problem": "p"}],
        "status": "frozen_before_model_output",
    }

    assert select_sentinel_rows(packet, ["a"], allocation="fit")[0]["group_id"] == "a"
    with pytest.raises(ValueError):
        select_sentinel_rows(packet, ["a"], allocation="basis")


def source_rows(group_count, rollout_indices, generated_tokens=1800):
    return [
        {
            "correct": group < 64,
            "generated_tokens": generated_tokens,
            "generation_id": f"q{group}:{rollout}",
            "group_id": f"q{group}",
            "parse_status": "pass",
            "rollout_index": rollout,
            "trace_length": generated_tokens,
        }
        for group in range(group_count)
        for rollout in rollout_indices
    ]


def source_receipts(stage):
    protocol = sentinel_protocol(5, stage)
    groups_per_shard = protocol["group_count"]
    rows_per_group = len(protocol["rollout_indices"])
    return [
        {
            "allocation": protocol["allocation"],
            "answer_format": protocol["answer_format"],
            "elapsed_seconds": 600.0,
            "enable_thinking": protocol["enable_thinking"],
            "evaluator": protocol["evaluator"],
            "group_ids": [
                f"q{shard * groups_per_shard + offset}" for offset in range(groups_per_shard)
            ],
            "layer": protocol["layer"],
            "max_new_tokens": protocol["max_new_tokens"],
            "model_config_sha256": protocol["model_config_sha256"],
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 5,
            "replay_exact": True,
            "rollout_indices": protocol["rollout_indices"],
            "rows": groups_per_shard * rows_per_group,
            "source_sha256": protocol["source_sha256"],
            "stage": stage,
            "status": "pass",
        }
        for shard in range(8)
    ]


def test_source_summary_passes_complete_basis_completion():
    summary = summarize_source_stage(
        source_rows(32, (2, 3)),
        source_receipts("basis_completion"),
        "basis_completion",
    )

    assert summary["decision"] == "pass_open_basis"
    assert all(summary["gates"].values())


def test_source_summary_passes_fit_class_and_progress_support():
    summary = summarize_source_stage(
        source_rows(256, (0, 1, 2, 3)),
        source_receipts("fit"),
        "fit",
    )

    assert summary["decision"] == "pass_open_observer_fit"
    assert summary["correct_rows"] == 256
    assert all(summary["gates"].values())


def test_source_summary_closes_on_late_progress_or_interface_failure():
    receipts = source_receipts("fit")
    rows = source_rows(256, (0, 1, 2, 3), generated_tokens=1000)
    receipts[0]["evaluator"] = "unfrozen"

    summary = summarize_source_stage(rows, receipts, "fit")

    assert summary["decision"] == "close_before_observer_fit"
    assert not summary["gates"]["packet_complete"]
    assert not summary["gates"]["progress_support"]


def test_fit_runner_rejects_basis_packet_before_model_or_cuda(tmp_path):
    run = runpy.run_path("scripts/run_outcome_score_sentinel_shard.py")["run"]
    args = SimpleNamespace(
        data=Path("artifacts/data/outcome_score_v1_allocations/basis.json"),
        group_ids=[f"q{index}" for index in range(32)],
        layer=14,
        model=tmp_path,
        output=tmp_path / "output",
        protocol_version=5,
        stage="fit",
    )

    with pytest.raises(RuntimeError, match="source packet"):
        run(args)


def test_basis_completion_replays_first_frozen_rollout():
    replay_rollout_index = runpy.run_path("scripts/run_outcome_score_sentinel_shard.py")[
        "replay_rollout_index"
    ]

    assert replay_rollout_index(sentinel_protocol(5, "basis_completion")) == 2
    assert replay_rollout_index(sentinel_protocol(5, "fit")) == 0
