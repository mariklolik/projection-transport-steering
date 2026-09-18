from collections.abc import Mapping, Sequence
from typing import Any


def sentinel_protocol(version: int, stage: str) -> dict[str, object]:
    protocols = {
        (2, "termination"): {
            "answer_format": "line",
            "enable_thinking": True,
            "group_count": 1,
            "layer": 18,
            "max_new_tokens": 4096,
            "rollout_indices": (0,),
        },
        (2, "class_mix"): {
            "answer_format": "line",
            "enable_thinking": True,
            "group_count": 4,
            "layer": 18,
            "max_new_tokens": 4096,
            "rollout_indices": (0, 1),
        },
        (3, "termination"): {
            "answer_format": "terminal_integer",
            "enable_thinking": False,
            "group_count": 1,
            "layer": 18,
            "max_new_tokens": 2048,
            "rollout_indices": (0,),
        },
        (3, "class_mix"): {
            "answer_format": "terminal_integer",
            "enable_thinking": False,
            "group_count": 4,
            "layer": 18,
            "max_new_tokens": 2048,
            "rollout_indices": (0, 1),
        },
        (4, "termination"): {
            "answer_format": "reward_integer",
            "enable_thinking": False,
            "group_count": 1,
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "rollout_indices": (0,),
        },
        (4, "class_mix"): {
            "answer_format": "reward_integer",
            "enable_thinking": False,
            "group_count": 4,
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "rollout_indices": (0, 1),
        },
        (5, "termination"): {
            "answer_format": "math_verify_strict",
            "enable_thinking": False,
            "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
            "group_count": 1,
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "rollout_indices": (0,),
        },
        (5, "class_mix"): {
            "answer_format": "math_verify_strict",
            "enable_thinking": False,
            "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
            "group_count": 4,
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "rollout_indices": (0, 1),
        },
        (5, "basis_completion"): {
            "allocation": "basis",
            "answer_format": "math_verify_strict",
            "enable_thinking": False,
            "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
            "group_count": 4,
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "rollout_indices": (2, 3),
            "source_sha256": "8094ab483c43bb9c2e12024c6f2f4cae59077d47d385e2a9757b77586d4a1db7",
        },
        (5, "fit"): {
            "allocation": "fit",
            "answer_format": "math_verify_strict",
            "enable_thinking": False,
            "evaluator": "math-verify==0.9.0:latex-anchored-no-fallback-first-match",
            "group_count": 32,
            "layer": 14,
            "max_new_tokens": 2048,
            "model_config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
            "rollout_indices": (0, 1, 2, 3),
            "source_sha256": "abee6c7c85376e085756d2f60a427969105ce908abbe00a83027a55931d0fd1b",
        },
    }
    try:
        return protocols[(version, stage)]
    except KeyError as error:
        raise ValueError("unknown sentinel protocol") from error


def select_sentinel_rows(
    packet: Mapping[str, Any],
    group_ids: Sequence[str],
    allocation: str = "basis",
) -> list[dict[str, object]]:
    if packet.get("allocation") != allocation or packet.get("status") != "frozen_before_model_output":
        raise ValueError("sentinel requires the frozen basis packet")
    if not group_ids or len(set(group_ids)) != len(group_ids):
        raise ValueError("sentinel group IDs must be nonempty and unique")
    rows = packet.get("rows")
    if not isinstance(rows, list):
        raise ValueError("basis rows are missing")
    indexed = {
        str(row.get("group_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("allocation") == allocation
    }
    if len(indexed) != len(rows) or any(group_id not in indexed for group_id in group_ids):
        raise ValueError("sentinel request is outside the basis allocation")
    return [dict(indexed[group_id]) for group_id in group_ids]


def summarize_sentinel(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    reported_versions = [receipt.get("protocol_version") for receipt in receipts]
    requires_versioned_interface = any(version is not None for version in reported_versions)
    versions = set(reported_versions)
    version = next(iter(versions)) if len(versions) == 1 else None
    try:
        protocol = sentinel_protocol(int(version), "class_mix")
    except (TypeError, ValueError):
        protocol = {}
    interface_complete = not requires_versioned_interface or (
        bool(protocol)
        and all(
            receipt.get("answer_format") == protocol.get("answer_format")
            and receipt.get("enable_thinking") == protocol.get("enable_thinking")
            and receipt.get("max_new_tokens") == protocol.get("max_new_tokens")
            and receipt.get("stage") == "class_mix"
            and (
                version not in {4, 5}
                or (
                    receipt.get("layer") == protocol.get("layer")
                    and receipt.get("model_config_sha256") == protocol.get("model_config_sha256")
                )
            )
            and (version != 5 or receipt.get("evaluator") == protocol.get("evaluator"))
            for receipt in receipts
        )
    )
    identifiers = [str(row.get("generation_id")) for row in rows]
    group_ids = [str(row.get("group_id")) for row in rows]
    receipt_groups = [
        str(group_id) for receipt in receipts for group_id in receipt.get("group_ids", [])
    ]
    group_counts = {group_id: group_ids.count(group_id) for group_id in set(group_ids)}
    structural_complete = (
        len(rows) == 64
        and len(set(identifiers)) == 64
        and len(group_counts) == 32
        and set(group_counts.values()) == {2}
        and all(
            {row.get("rollout_index") for row in rows if row.get("group_id") == group_id} == {0, 1}
            for group_id in group_counts
        )
        and len(receipts) == 8
        and all(
            receipt.get("status") == "pass" and receipt.get("rows") == 8 for receipt in receipts
        )
        and len(receipt_groups) == len(set(receipt_groups)) == 32
        and set(receipt_groups) == set(group_ids)
        and interface_complete
    )
    correct = sum(row.get("correct") is True for row in rows)
    parse_failures = sum(row.get("parse_status") != "pass" for row in rows)
    peak_memory = max((int(receipt.get("peak_memory_bytes", 0)) for receipt in receipts), default=0)
    maximum_elapsed = max(
        (float(receipt.get("elapsed_seconds", 0.0)) for receipt in receipts), default=0.0
    )
    estimated_fit_wall = maximum_elapsed * 128.0 / 9.0
    gates = {
        "class_mix": correct >= 8 and len(rows) - correct >= 8,
        "finite_traces": all(int(row.get("trace_length", 0)) > 0 for row in rows),
        "memory": 0 < peak_memory < 70 * 1024**3,
        "packet_complete": structural_complete,
        "parse_complete": parse_failures == 0,
        "replay": len(receipts) == 8
        and all(receipt.get("replay_exact") is True for receipt in receipts),
        "runtime": 0.0 < estimated_fit_wall < 6 * 60 * 60,
    }
    if requires_versioned_interface:
        max_new_tokens = int(protocol.get("max_new_tokens", 0))
        gates["finite_traces"] = all(
            0 < int(row.get("trace_length", 0)) == int(row.get("generated_tokens", 0))
            for row in rows
        )
        gates["runtime"] = 0.0 < maximum_elapsed < 20 * 60
        gates["terminated_before_cap"] = all(
            0 < int(row.get("generated_tokens", 0)) < max_new_tokens for row in rows
        )
    return {
        "correct_rows": correct,
        "decision": "pass_open_fit" if all(gates.values()) else "close_before_fit",
        "estimated_fit_wall_seconds": estimated_fit_wall,
        "gates": gates,
        "incorrect_rows": len(rows) - correct,
        "maximum_elapsed_seconds": maximum_elapsed,
        "maximum_peak_memory_bytes": peak_memory,
        "parse_failures": parse_failures,
        "rows": len(rows),
    }


def summarize_termination(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    versions = {receipt.get("protocol_version") for receipt in receipts}
    version = next(iter(versions)) if len(versions) == 1 else None
    try:
        protocol = sentinel_protocol(int(version), "termination")
    except (TypeError, ValueError):
        protocol = {}
    max_new_tokens = int(protocol.get("max_new_tokens", 0))
    identifiers = [str(row.get("generation_id")) for row in rows]
    row_groups = [str(row.get("group_id")) for row in rows]
    receipt_groups = [
        str(receipt["group_ids"][0])
        for receipt in receipts
        if isinstance(receipt.get("group_ids"), list) and len(receipt["group_ids"]) == 1
    ]
    complete = (
        len(rows) == len(receipts) == 8
        and len(set(identifiers)) == len(set(row_groups)) == 8
        and set(row_groups) == set(receipt_groups)
        and all(
            receipt.get("status") == "pass"
            and receipt.get("stage") == "termination"
            and receipt.get("rows") == 1
            and receipt.get("max_new_tokens") == max_new_tokens
            and (
                version == 2
                or (
                    receipt.get("answer_format") == protocol.get("answer_format")
                    and receipt.get("enable_thinking") == protocol.get("enable_thinking")
                    and (
                        version not in {4, 5}
                        or (
                            receipt.get("layer") == protocol.get("layer")
                            and receipt.get("model_config_sha256")
                            == protocol.get("model_config_sha256")
                        )
                    )
                    and (version != 5 or receipt.get("evaluator") == protocol.get("evaluator"))
                )
            )
            for receipt in receipts
        )
    )
    parse_failures = sum(row.get("parse_status") != "pass" for row in rows)
    capped = sum(int(row.get("generated_tokens", 0)) >= max_new_tokens for row in rows)
    peak_memory = max((int(receipt.get("peak_memory_bytes", 0)) for receipt in receipts), default=0)
    gates = {
        "finite_traces": all(
            0 < int(row.get("trace_length", 0)) == int(row.get("generated_tokens", 0))
            for row in rows
        ),
        "memory": 0 < peak_memory < 70 * 1024**3,
        "packet_complete": complete,
        "parse_complete": parse_failures == 0,
        "replay": len(receipts) == 8
        and all(receipt.get("replay_exact") is True for receipt in receipts),
        "terminated_before_cap": capped == 0,
    }
    return {
        "capped_rows": capped,
        "correct_rows": sum(row.get("correct") is True for row in rows),
        "decision": "pass_open_class_mix" if all(gates.values()) else "close_before_class_mix",
        "gates": gates,
        "maximum_peak_memory_bytes": peak_memory,
        "parse_failures": parse_failures,
        "rows": len(rows),
    }
