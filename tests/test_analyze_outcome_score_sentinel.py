import hashlib
import json
import subprocess
import sys

import torch


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyzer_validates_and_summarizes_complete_packet(tmp_path):
    packet = tmp_path / "packet"
    for shard in range(8):
        directory = packet / f"shard_{shard}"
        directory.mkdir(parents=True)
        group_ids = [f"q{shard * 4 + offset}" for offset in range(4)]
        rows = [
            {
                "correct": int(group_id[1:]) < 16,
                "generation_id": f"{group_id}:{rollout}",
                "group_id": group_id,
                "parse_status": "pass",
                "rollout_index": rollout,
                "trace_length": 10,
            }
            for group_id in group_ids
            for rollout in range(2)
        ]
        traces = {row["generation_id"]: torch.ones((10, 4)) for row in rows}
        rows_path = directory / "rows.json"
        traces_path = directory / "traces.pt"
        rows_path.write_text(json.dumps(rows, sort_keys=True))
        torch.save(traces, traces_path)
        receipt = {
            "elapsed_seconds": 60.0,
            "group_ids": group_ids,
            "peak_memory_bytes": 20 * 1024**3,
            "replay_exact": True,
            "rows": 8,
            "rows_sha256": file_sha256(rows_path),
            "status": "pass",
            "traces_sha256": file_sha256(traces_path),
        }
        (directory / "receipt.json").write_text(json.dumps(receipt))
    output = tmp_path / "analysis.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_outcome_score_sentinel.py",
            "--input",
            str(packet),
            "--output",
            str(output),
        ],
        check=True,
    )

    analysis = json.loads(output.read_text())
    assert analysis["decision"] == "pass_open_fit"
    assert analysis["trace_rows"] == 64


def test_analyzer_supports_version_two_termination_packet(tmp_path):
    packet = tmp_path / "packet"
    for shard in range(8):
        directory = packet / f"shard_{shard}"
        directory.mkdir(parents=True)
        group_id = f"q{shard}"
        rows = [
            {
                "correct": shard % 2 == 0,
                "generated_tokens": 2000,
                "generation_id": f"{group_id}:0",
                "group_id": group_id,
                "parse_status": "pass",
                "rollout_index": 0,
                "trace_length": 2000,
            }
        ]
        traces = {rows[0]["generation_id"]: torch.ones((2000, 4))}
        rows_path = directory / "rows.json"
        traces_path = directory / "traces.pt"
        rows_path.write_text(json.dumps(rows, sort_keys=True))
        torch.save(traces, traces_path)
        receipt = {
            "answer_format": "line",
            "enable_thinking": True,
            "group_ids": [group_id],
            "max_new_tokens": 4096,
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 2,
            "replay_exact": True,
            "rows": 1,
            "rows_sha256": file_sha256(rows_path),
            "stage": "termination",
            "status": "pass",
            "traces_sha256": file_sha256(traces_path),
        }
        (directory / "receipt.json").write_text(json.dumps(receipt))
    output = tmp_path / "analysis.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_outcome_score_sentinel.py",
            "--input",
            str(packet),
            "--output",
            str(output),
            "--stage",
            "termination",
        ],
        check=True,
    )

    assert json.loads(output.read_text())["decision"] == "pass_open_class_mix"


def test_analyzer_supports_version_three_termination_packet(tmp_path):
    packet = tmp_path / "packet"
    for shard in range(8):
        directory = packet / f"shard_{shard}"
        directory.mkdir(parents=True)
        group_id = f"q{shard}"
        rows = [
            {
                "correct": shard % 2 == 0,
                "generated_tokens": 1000,
                "generation_id": f"{group_id}:0",
                "group_id": group_id,
                "parse_status": "pass",
                "rollout_index": 0,
                "trace_length": 1000,
            }
        ]
        traces = {rows[0]["generation_id"]: torch.ones((1000, 4))}
        rows_path = directory / "rows.json"
        traces_path = directory / "traces.pt"
        rows_path.write_text(json.dumps(rows, sort_keys=True))
        torch.save(traces, traces_path)
        receipt = {
            "answer_format": "terminal_integer",
            "enable_thinking": False,
            "group_ids": [group_id],
            "max_new_tokens": 2048,
            "peak_memory_bytes": 20 * 1024**3,
            "protocol_version": 3,
            "replay_exact": True,
            "rows": 1,
            "rows_sha256": file_sha256(rows_path),
            "stage": "termination",
            "status": "pass",
            "traces_sha256": file_sha256(traces_path),
        }
        (directory / "receipt.json").write_text(json.dumps(receipt))
    output = tmp_path / "analysis.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_outcome_score_sentinel.py",
            "--input",
            str(packet),
            "--output",
            str(output),
            "--stage",
            "termination",
        ],
        check=True,
    )

    assert json.loads(output.read_text())["decision"] == "pass_open_class_mix"
