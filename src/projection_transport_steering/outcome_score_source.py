from collections.abc import Mapping, Sequence
from typing import Any

from projection_transport_steering.outcome_score_sentinel import sentinel_protocol


def summarize_source_stage(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, object]:
    protocol = sentinel_protocol(5, stage)
    rollout_indices = tuple(protocol["rollout_indices"])
    groups_per_shard = int(protocol["group_count"])
    expected_groups = groups_per_shard * 8
    expected_rows = expected_groups * len(rollout_indices)
    identifiers = [str(row.get("generation_id")) for row in rows]
    group_ids = [str(row.get("group_id")) for row in rows]
    unique_groups = set(group_ids)
    receipt_groups = [
        str(group_id) for receipt in receipts for group_id in receipt.get("group_ids", [])
    ]
    interface_complete = len(receipts) == 8 and all(
        receipt.get("allocation") == protocol.get("allocation")
        and receipt.get("answer_format") == protocol.get("answer_format")
        and receipt.get("enable_thinking") == protocol.get("enable_thinking")
        and receipt.get("evaluator") == protocol.get("evaluator")
        and receipt.get("layer") == protocol.get("layer")
        and receipt.get("max_new_tokens") == protocol.get("max_new_tokens")
        and receipt.get("model_config_sha256") == protocol.get("model_config_sha256")
        and receipt.get("protocol_version") == 5
        and tuple(receipt.get("rollout_indices", ())) == rollout_indices
        and receipt.get("source_sha256") == protocol.get("source_sha256")
        and receipt.get("stage") == stage
        and receipt.get("status") == "pass"
        and receipt.get("rows") == groups_per_shard * len(rollout_indices)
        and len(receipt.get("group_ids", [])) == groups_per_shard
        for receipt in receipts
    )
    structural_complete = (
        len(rows) == expected_rows
        and len(set(identifiers)) == expected_rows
        and len(unique_groups) == expected_groups
        and len(receipt_groups) == len(set(receipt_groups)) == expected_groups
        and set(receipt_groups) == unique_groups
        and all(group_ids.count(group_id) == len(rollout_indices) for group_id in unique_groups)
        and all(
            {
                row.get("rollout_index")
                for row in rows
                if str(row.get("group_id")) == group_id
            }
            == set(rollout_indices)
            for group_id in unique_groups
        )
        and interface_complete
    )
    correct = sum(row.get("correct") is True for row in rows)
    correct_groups = sum(
        any(row.get("correct") is True for row in rows if str(row.get("group_id")) == group_id)
        for group_id in unique_groups
    )
    incorrect_groups = sum(
        any(row.get("correct") is False for row in rows if str(row.get("group_id")) == group_id)
        for group_id in unique_groups
    )
    progress = {}
    for start in (0, 512, 1024, 1536):
        eligible = [row for row in rows if int(row.get("generated_tokens", 0)) > start]
        progress[str(start)] = {
            "correct_rows": sum(row.get("correct") is True for row in eligible),
            "groups": len({str(row.get("group_id")) for row in eligible}),
            "incorrect_rows": sum(row.get("correct") is False for row in eligible),
            "rows": len(eligible),
        }
    fit = stage == "fit"
    progress_complete = not fit or all(
        counts["groups"] >= 32
        and counts["correct_rows"] >= 16
        and counts["incorrect_rows"] >= 16
        for counts in progress.values()
    )
    class_support = not fit or (
        correct >= 128
        and len(rows) - correct >= 128
        and correct_groups >= 64
        and incorrect_groups >= 64
    )
    peak_memory = max((int(receipt.get("peak_memory_bytes", 0)) for receipt in receipts), default=0)
    maximum_elapsed = max(
        (float(receipt.get("elapsed_seconds", 0.0)) for receipt in receipts), default=0.0
    )
    runtime_limit = 6 * 60 * 60 if fit else 20 * 60
    max_new_tokens = int(protocol["max_new_tokens"])
    gates = {
        "class_support": class_support,
        "finite_traces": all(
            0 < int(row.get("trace_length", 0)) == int(row.get("generated_tokens", 0))
            for row in rows
        ),
        "memory": 0 < peak_memory < 70 * 1024**3,
        "packet_complete": structural_complete,
        "parse_complete": all(row.get("parse_status") == "pass" for row in rows),
        "progress_support": progress_complete,
        "replay": len(receipts) == 8
        and all(receipt.get("replay_exact") is True for receipt in receipts),
        "runtime": 0.0 < maximum_elapsed < runtime_limit,
        "terminated_before_cap": all(
            0 < int(row.get("generated_tokens", 0)) < max_new_tokens for row in rows
        ),
    }
    passed = all(gates.values())
    decisions = {
        "basis_completion": "pass_open_basis",
        "fit": "pass_open_observer_fit",
    }
    closed = {
        "basis_completion": "close_before_basis",
        "fit": "close_before_observer_fit",
    }
    return {
        "correct_groups": correct_groups,
        "correct_rows": correct,
        "decision": decisions[stage] if passed else closed[stage],
        "gates": gates,
        "groups": len(unique_groups),
        "incorrect_groups": incorrect_groups,
        "incorrect_rows": len(rows) - correct,
        "maximum_elapsed_seconds": maximum_elapsed,
        "maximum_peak_memory_bytes": peak_memory,
        "progress_support": progress,
        "rows": len(rows),
    }
