import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from projection_transport_steering.outcome_score_sentinel import (
    select_sentinel_rows,
    sentinel_protocol,
    summarize_sentinel,
    summarize_termination,
)


def test_select_sentinel_rows_requires_exact_unique_basis_ids():
    packet = {
        "allocation": "basis",
        "rows": [
            {"allocation": "basis", "group_id": "a", "problem": "p1"},
            {"allocation": "basis", "group_id": "b", "problem": "p2"},
        ],
        "status": "frozen_before_model_output",
    }

    selected = select_sentinel_rows(packet, ["b", "a"])

    assert [row["group_id"] for row in selected] == ["b", "a"]


@pytest.mark.parametrize(
    ("packet", "group_ids"),
    [
        ({"allocation": "fit", "rows": [], "status": "frozen_before_model_output"}, []),
        ({"allocation": "basis", "rows": [], "status": "open"}, []),
        (
            {
                "allocation": "basis",
                "rows": [{"allocation": "basis", "group_id": "a"}],
                "status": "frozen_before_model_output",
            },
            ["a", "a"],
        ),
        (
            {
                "allocation": "basis",
                "rows": [{"allocation": "basis", "group_id": "a"}],
                "status": "frozen_before_model_output",
            },
            ["missing"],
        ),
    ],
)
def test_select_sentinel_rows_rejects_nonfrozen_or_incomplete_requests(packet, group_ids):
    with pytest.raises(ValueError):
        select_sentinel_rows(packet, group_ids)


def test_runner_import_does_not_require_flas():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path('scripts/run_outcome_score_sentinel_shard.py')",
        ],
        check=True,
    )


def test_runner_rejects_version_four_model_mismatch_before_cuda(tmp_path):
    run = runpy.run_path("scripts/run_outcome_score_sentinel_shard.py")["run"]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    args = SimpleNamespace(
        data=Path("artifacts/data/outcome_score_v1_allocations/basis.json"),
        group_ids=["q"],
        layer=14,
        model=model,
        protocol_version=4,
        stage="termination",
    )

    with pytest.raises(ValueError, match="model config"):
        run(args)


def test_summarize_sentinel_applies_all_frozen_gates():
    rows = [
        {
            "correct": index < 32,
            "generation_id": f"q{index // 2}:{index % 2}",
            "group_id": f"q{index // 2}",
            "parse_status": "pass",
            "rollout_index": index % 2,
            "trace_length": 10,
        }
        for index in range(64)
    ]
    receipts = [
        {
            "elapsed_seconds": 60.0,
            "group_ids": [f"q{shard * 4 + offset}" for offset in range(4)],
            "peak_memory_bytes": 20 * 1024**3,
            "replay_exact": True,
            "rows": 8,
            "status": "pass",
        }
        for shard in range(8)
    ]

    summary = summarize_sentinel(rows, receipts)

    assert summary["decision"] == "pass_open_fit"
    assert summary["correct_rows"] == 32
    assert all(summary["gates"].values())


def test_summarize_sentinel_closes_on_parse_or_class_mix_failure():
    rows = [
        {
            "correct": False,
            "generation_id": f"q{index // 2}:{index % 2}",
            "group_id": f"q{index // 2}",
            "parse_status": "fail" if index == 0 else "pass",
            "rollout_index": index % 2,
            "trace_length": 10,
        }
        for index in range(64)
    ]
    receipts = [
        {
            "elapsed_seconds": 60.0,
            "group_ids": [f"q{shard * 4 + offset}" for offset in range(4)],
            "peak_memory_bytes": 20 * 1024**3,
            "replay_exact": True,
            "rows": 8,
            "status": "pass",
        }
        for shard in range(8)
    ]

    summary = summarize_sentinel(rows, receipts)

    assert summary["decision"] == "close_before_fit"
    assert not summary["gates"]["parse_complete"]
    assert not summary["gates"]["class_mix"]


def test_version_two_protocol_separates_termination_and_class_mix():
    assert sentinel_protocol(2, "termination") == {
        "answer_format": "line",
        "enable_thinking": True,
        "group_count": 1,
        "layer": 18,
        "max_new_tokens": 4096,
        "rollout_indices": (0,),
    }
    assert sentinel_protocol(2, "class_mix") == {
        "answer_format": "line",
        "enable_thinking": True,
        "group_count": 4,
        "layer": 18,
        "max_new_tokens": 4096,
        "rollout_indices": (0, 1),
    }


def test_version_three_protocol_is_nonthinking_and_bounded():
    assert sentinel_protocol(3, "termination") == {
        "answer_format": "terminal_integer",
        "enable_thinking": False,
        "group_count": 1,
        "layer": 18,
        "max_new_tokens": 2048,
        "rollout_indices": (0,),
    }
    assert sentinel_protocol(3, "class_mix") == {
        "answer_format": "terminal_integer",
        "enable_thinking": False,
        "group_count": 4,
        "layer": 18,
        "max_new_tokens": 2048,
        "rollout_indices": (0, 1),
    }


def test_version_four_protocol_binds_model_layer_and_total_reward():
    expected = {
        "answer_format": "reward_integer",
        "enable_thinking": False,
        "group_count": 1,
        "layer": 14,
        "max_new_tokens": 2048,
        "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        "rollout_indices": (0,),
    }

    assert sentinel_protocol(4, "termination") == expected
    assert sentinel_protocol(4, "class_mix") == {
        **expected,
        "group_count": 4,
        "rollout_indices": (0, 1),
    }


def test_version_five_protocol_binds_strict_standard_evaluator():
    expected = {
        "answer_format": "math_verify_strict",
        "enable_thinking": False,
        "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
        "group_count": 1,
        "layer": 14,
        "max_new_tokens": 2048,
        "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        "rollout_indices": (0,),
    }

    assert sentinel_protocol(5, "termination") == expected
    assert sentinel_protocol(5, "class_mix") == {
        **expected,
        "group_count": 4,
        "rollout_indices": (0, 1),
    }


def test_sentinel_protocol_rejects_unfrozen_version_or_stage():
    with pytest.raises(ValueError, match="protocol"):
        sentinel_protocol(6, "termination")
    with pytest.raises(ValueError, match="protocol"):
        sentinel_protocol(2, "other")


def test_summarize_termination_requires_all_rows_to_parse_and_stop_before_cap():
    rows = [
        {
            "correct": index % 2 == 0,
            "generated_tokens": 2000,
            "generation_id": f"q{index}:0",
            "group_id": f"q{index}",
            "parse_status": "pass",
            "trace_length": 2000,
        }
        for index in range(8)
    ]
    receipts = [
        {
            "answer_format": "line",
            "enable_thinking": True,
            "group_ids": [f"q{index}"],
            "max_new_tokens": 4096,
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 2,
            "replay_exact": True,
            "rows": 1,
            "stage": "termination",
            "status": "pass",
        }
        for index in range(8)
    ]

    summary = summarize_termination(rows, receipts)

    assert summary["decision"] == "pass_open_class_mix"
    assert all(summary["gates"].values())


def test_summarize_termination_closes_on_one_capped_row():
    rows = [
        {
            "correct": False,
            "generated_tokens": 4096 if index == 0 else 2000,
            "generation_id": f"q{index}:0",
            "group_id": f"q{index}",
            "parse_status": "fail" if index == 0 else "pass",
            "trace_length": 4096 if index == 0 else 2000,
        }
        for index in range(8)
    ]
    receipts = [
        {
            "answer_format": "line",
            "enable_thinking": True,
            "group_ids": [f"q{index}"],
            "max_new_tokens": 4096,
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 2,
            "replay_exact": True,
            "rows": 1,
            "stage": "termination",
            "status": "pass",
        }
        for index in range(8)
    ]

    summary = summarize_termination(rows, receipts)

    assert summary["decision"] == "close_before_class_mix"
    assert not summary["gates"]["terminated_before_cap"]
    assert not summary["gates"]["parse_complete"]


def test_summarize_termination_uses_version_three_cap_and_interface():
    rows = [
        {
            "correct": index % 2 == 0,
            "generated_tokens": 1000,
            "generation_id": f"q{index}:0",
            "group_id": f"q{index}",
            "parse_status": "pass",
            "trace_length": 1000,
        }
        for index in range(8)
    ]
    receipts = [
        {
            "answer_format": "terminal_integer",
            "enable_thinking": False,
            "group_ids": [f"q{index}"],
            "max_new_tokens": 2048,
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 3,
            "replay_exact": True,
            "rollout_indices": [0],
            "rows": 1,
            "stage": "termination",
            "status": "pass",
        }
        for index in range(8)
    ]

    summary = summarize_termination(rows, receipts)

    assert summary["decision"] == "pass_open_class_mix"
    assert all(summary["gates"].values())


def test_summarize_class_mix_rejects_version_three_cap_hit():
    rows = [
        {
            "correct": index < 32,
            "generated_tokens": 2048 if index == 0 else 1000,
            "generation_id": f"q{index // 2}:{index % 2}",
            "group_id": f"q{index // 2}",
            "parse_status": "pass",
            "rollout_index": index % 2,
            "trace_length": 2048 if index == 0 else 1000,
        }
        for index in range(64)
    ]
    receipts = [
        {
            "answer_format": "terminal_integer",
            "elapsed_seconds": 60.0,
            "enable_thinking": False,
            "group_ids": [f"q{shard * 4 + offset}" for offset in range(4)],
            "max_new_tokens": 2048,
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 3,
            "replay_exact": True,
            "rollout_indices": [0, 1],
            "rows": 8,
            "stage": "class_mix",
            "status": "pass",
        }
        for shard in range(8)
    ]

    summary = summarize_sentinel(rows, receipts)

    assert summary["decision"] == "close_before_fit"
    assert not summary["gates"]["terminated_before_cap"]


def test_summarize_termination_requires_version_four_model_and_layer():
    rows = [
        {
            "correct": index % 2 == 0,
            "generated_tokens": 1000,
            "generation_id": f"q{index}:0",
            "group_id": f"q{index}",
            "parse_status": "pass",
            "trace_length": 1000,
        }
        for index in range(8)
    ]
    receipts = [
        {
            "answer_format": "reward_integer",
            "enable_thinking": False,
            "group_ids": [f"q{index}"],
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 4,
            "replay_exact": True,
            "rows": 1,
            "stage": "termination",
            "status": "pass",
        }
        for index in range(8)
    ]

    assert summarize_termination(rows, receipts)["decision"] == "pass_open_class_mix"
    receipts[0]["layer"] = 18
    assert not summarize_termination(rows, receipts)["gates"]["packet_complete"]


def test_summarize_termination_requires_version_five_evaluator_identity():
    rows = [
        {
            "correct": index % 2 == 0,
            "generated_tokens": 1000,
            "generation_id": f"q{index}:0",
            "group_id": f"q{index}",
            "parse_status": "pass",
            "trace_length": 1000,
        }
        for index in range(8)
    ]
    receipts = [
        {
            "answer_format": "math_verify_strict",
            "enable_thinking": False,
            "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
            "group_ids": [f"q{index}"],
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 5,
            "replay_exact": True,
            "rows": 1,
            "stage": "termination",
            "status": "pass",
        }
        for index in range(8)
    ]

    assert summarize_termination(rows, receipts)["decision"] == "pass_open_class_mix"
    receipts[0]["evaluator"] = "unfrozen"
    assert not summarize_termination(rows, receipts)["gates"]["packet_complete"]


def test_summarize_class_mix_requires_version_five_evaluator_identity():
    rows = [
        {
            "correct": index < 32,
            "generated_tokens": 1000,
            "generation_id": f"q{index // 2}:{index % 2}",
            "group_id": f"q{index // 2}",
            "parse_status": "pass",
            "rollout_index": index % 2,
            "trace_length": 1000,
        }
        for index in range(64)
    ]
    receipts = [
        {
            "answer_format": "math_verify_strict",
            "elapsed_seconds": 60.0,
            "enable_thinking": False,
            "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
            "group_ids": [f"q{shard * 4 + offset}" for offset in range(4)],
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 5,
            "replay_exact": True,
            "rows": 8,
            "stage": "class_mix",
            "status": "pass",
        }
        for shard in range(8)
    ]

    assert summarize_sentinel(rows, receipts)["decision"] == "pass_open_fit"
    receipts[0]["evaluator"] = "unfrozen"
    assert not summarize_sentinel(rows, receipts)["gates"]["packet_complete"]
