import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def packet():
    containers = [
        {
            "id": str(i),
            "name": f"c{i}",
            "image": "pinned",
            "gpu": [str(i)],
            "mounts": [],
            "state": {
                "StartedAt": "2026-09-07T22:40:00Z",
                "FinishedAt": "0001-01-01T00:00:00Z",
                "Status": "running",
                "Running": True,
                "Paused": False,
                "Dead": False,
                "Restarting": False,
                "OOMKilled": False,
                "Error": "",
                "ExitCode": 0,
            },
        }
        for i in range(8)
    ]
    dispatch = {
        "snapshot": {"containers": copy.deepcopy(containers)},
        "dispatch": [{"index": i, "container_id": str(i)} for i in range(8)],
    }
    for row in containers:
        row["state"].update(Status="exited", Running=False, FinishedAt="2026-09-07T23:40:00Z")
    return containers, dispatch


def module():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("audit_irc_baseline")


def test_complete_cost_includes_every_worker_lifetime():
    result = module().audit_cost(*packet())
    assert result["h100_hours"] == 8
    assert result["all_terminal_success"] and result["within_budget"]


def test_sequential_component_costs_fit_inside_runner():
    part = {"generation_seconds": 2, "grading_seconds": 0.2, "check_seconds": 0.01}
    result = module().audit_component_cost({"fit_seconds": 3}, part, part, 10)
    assert result["accounted_seconds"] == pytest.approx(7.42)
    assert result["unattributed_runner_seconds"] == pytest.approx(2.58)


def test_inflated_sequential_component_cost_blocks_audit():
    part = {"generation_seconds": 2, "grading_seconds": 0.2, "check_seconds": 0.01}
    with pytest.raises(ValueError):
        module().audit_component_cost({"fit_seconds": 3}, part, part, 5)


def test_failed_worker_cost_is_retained_without_success():
    containers, dispatch = packet()
    containers[3]["state"]["ExitCode"] = 124
    result = module().audit_cost(containers, dispatch)
    assert result["h100_hours"] == 8
    assert not result["all_terminal_success"]


@pytest.mark.parametrize("fault", ["running", "missing", "duplicate", "image", "start", "negative"])
def test_incomplete_or_changed_container_identity_blocks_cost_completion(fault):
    containers, dispatch = packet()
    if fault == "running":
        containers[0]["state"]["Running"] = True
    elif fault == "missing":
        containers.pop()
    elif fault == "duplicate":
        containers[1]["id"] = "0"
    elif fault == "image":
        containers[1]["image"] = "changed"
    elif fault == "start":
        containers[1]["state"]["StartedAt"] = "2026-09-07T22:50:00Z"
    else:
        containers[0]["state"]["FinishedAt"] = "2026-09-07T21:40:00Z"
    with pytest.raises(ValueError):
        module().audit_cost(containers, dispatch)


def test_over_budget_cost_is_visible_and_not_a_pass():
    containers, dispatch = packet()
    for row in containers:
        row["state"]["FinishedAt"] = "2026-09-08T05:40:00Z"
    result = module().audit_cost(containers, dispatch)
    assert result["h100_hours"] == 56
    assert not result["within_budget"]


def test_cli_retains_failed_cost_without_reading_incomplete_scores(tmp_path):
    containers, dispatch = packet()
    containers[3]["state"]["ExitCode"] = 124
    for name, value in (("terminal", {"containers": containers}), ("dispatch", dispatch)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        *(["-S"] if sys.flags.no_site else []),
        str(Path(__file__).parents[1] / "scripts/audit_irc_baseline.py"),
        "--root",
        str(tmp_path),
        "--config",
        str(tmp_path / "absent-config"),
        "--input",
        str(tmp_path / "absent-scores"),
        "--model",
        str(tmp_path / "absent-model"),
        "--dispatch",
        str(tmp_path / "dispatch.json"),
        "--terminal",
        str(tmp_path / "terminal.json"),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert output.exists(), result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "blocked"
    assert report["cost"]["h100_hours"] == 8
    assert "analysis" not in report
    assert len(report["analysis_source_sha256"]) == 3
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
