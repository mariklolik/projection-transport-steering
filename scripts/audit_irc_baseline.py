import argparse
import json
import math
import runpy
import time
from datetime import datetime
from pathlib import Path

from analyze_irc_baseline import summarize
from audit_irc_sources import load_math_author
from irc_baseline_audit import audit_component_cost, audit_condition, audit_fit
from irc_baseline_support import prepare_inputs
from materialize_outcome_score_data import file_sha256


def audit_cost(containers: list[dict], dispatch: dict) -> dict:
    expected = {row["id"]: row for row in dispatch["snapshot"]["containers"]}
    ids = [row["id"] for row in containers]
    if (
        len(ids) != 8
        or len(set(ids)) != 8
        or set(ids) != set(expected)
        or {row["container_id"] for row in dispatch["dispatch"]} != set(ids)
    ):
        raise ValueError("incomplete or duplicate container identities")
    costs = []
    for row in containers:
        initial = expected[row["id"]]
        state = row["state"]
        if (
            any(row[key] != initial[key] for key in ("id", "name", "image", "gpu"))
            or sorted(row["mounts"], key=lambda m: m["Destination"])
            != sorted(initial["mounts"], key=lambda m: m["Destination"])
            or state["StartedAt"] != initial["state"]["StartedAt"]
            or state["Status"] != "exited"
            or any(state[key] for key in ("Running", "Paused", "Restarting"))
        ):
            raise ValueError("container incomplete, restarted or lineage changed")
        seconds = (
            datetime.fromisoformat(state["FinishedAt"]) - datetime.fromisoformat(state["StartedAt"])
        ).total_seconds()
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid container lifetime")
        costs.append(
            {
                "id": row["id"],
                "seconds": seconds,
                "exit_code": state["ExitCode"],
                "success": state["ExitCode"] == 0
                and not any(state[key] for key in ("OOMKilled", "Dead", "Error")),
            }
        )
    total = sum(row["seconds"] for row in costs) / 3600
    return {
        "workers": costs,
        "h100_hours": total,
        "timing_resolution_seconds": 1e-6,
        "all_terminal_success": all(row["success"] for row in costs),
        "within_budget": total <= 48 and all(row["seconds"] <= 21600 for row in costs),
        "includes_compilation_loading_fitting_generation_grading_shutdown": True,
        "historical_engineering_excluded_from_this_stage_total": True,
    }


def complete_packet(args, dispatch: dict, cost: dict, report: dict) -> dict:
    config = json.loads(args.config.read_text())
    digest = file_sha256(args.config)
    if dispatch["config_sha256"] != digest:
        raise ValueError("dispatch config hash mismatch")
    tokenizer, examples, sources = prepare_inputs(config, args.root, args.model)
    generation_config = {
        **config,
        "vocab_size": json.loads((args.model / "config.json").read_text())["vocab_size"],
    }
    if len(config["candidates"]) != 8 or {path.name for path in args.input.iterdir()} != {
        f"candidate_{i:02d}" for i in range(8)
    }:
        raise ValueError("unexpected or missing candidate output directory")
    extract, equivalent = load_math_author(args.root)
    boxed = runpy.run_path(
        str(args.root / ".external/math-author-source/modeling/dataset/util.py")
    )["last_boxed_only_string"]
    zero, outcomes, evidence = {}, [], []
    report["workers"] = evidence
    costs = {row["id"]: row["seconds"] for row in cost["workers"]}
    ids = {row["index"]: row["container_id"] for row in dispatch["dispatch"]}
    for index, candidate in enumerate(config["candidates"]):
        directory = args.input / f"candidate_{index:02d}" / "run"
        path = directory / "receipt.json"
        item = {
            "index": index,
            "input_file_sha256": {
                str(p.relative_to(directory)): file_sha256(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
        }
        evidence.append(item)
        receipt = json.loads(path.read_text())
        if (
            receipt["status"] != "pass"
            or receipt["candidate"] != candidate
            or receipt["config_sha256"] != digest
            or receipt["source_sha256"] != config["source_sha256"]
            or receipt["scientific_role"] != "exploratory_baseline_selection"
            or receipt["positive_action_admitted"] is not False
            or receipt["training_resume_supported"] is not False
            or receipt["base_parameter_versions_unchanged"] is not True
            or not 0 < receipt["total_elapsed_seconds"] <= costs[ids[index]] + 1
        ):
            raise ValueError(f"invalid candidate receipt: {index}")
        fit = audit_fit(directory, receipt["fit"], config["recipe"], examples, candidate)
        shares = [sources[i] for i in config["zero_shares"][index]]
        zero_rows, zero_audit = audit_condition(
            directory / "zero",
            shares,
            generation_config,
            receipt["zero"],
            tokenizer,
            (extract, equivalent, boxed),
        )
        rows, selection_audit = audit_condition(
            directory / "selection",
            sources,
            generation_config,
            receipt["selection"],
            tokenizer,
            (extract, equivalent, boxed),
        )
        for row in zero_rows:
            if row["cluster_id"] in zero:
                raise ValueError("duplicated common-zero identity")
            zero[row["cluster_id"]] = row
        outcomes.append(rows)
        item.update(
            {
                "index": index,
                "receipt_sha256": file_sha256(path),
                "fit": fit,
                "zero": zero_audit,
                "selection": selection_audit,
                "runner_seconds": receipt["total_elapsed_seconds"],
                "component_cost": audit_component_cost(
                    fit, zero_audit, selection_audit, receipt["total_elapsed_seconds"]
                ),
            }
        )
    shared_zero = [zero[row["cluster_id"]] for row in sources]
    return {
        "analysis": summarize(sources, shared_zero, outcomes, config["candidates"]),
        "workers": evidence,
        "config_sha256": digest,
        "source_sha256": config["source_sha256"],
        "model_files_sha256": config["model_files_sha256"],
        "fit_stdout_join": "not_available_in_worker_packet",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("root", "config", "input", "model", "dispatch", "terminal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    report = {
        "schema": "irc-baseline-result-audit-v2",
        "status": "blocked",
        "evidence_label": "exploratory_selection",
        "sota_achieved": False,
        "config_sha256": file_sha256(args.config) if args.config.is_file() else None,
        "analysis_source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "audit_irc_baseline.py",
                "irc_baseline_audit.py",
                "analyze_irc_baseline.py",
            )
        },
    }
    try:
        dispatch = json.loads(args.dispatch.read_text())
        terminal = json.loads(args.terminal.read_text())
        report["cost"] = audit_cost(terminal["containers"], dispatch)
        report["dispatch_sha256"] = file_sha256(args.dispatch)
        report["terminal_sha256"] = file_sha256(args.terminal)
        if not report["cost"]["all_terminal_success"] or not report["cost"]["within_budget"]:
            raise ValueError("failed or over-budget stage; no incomplete-pair selection")
        report.update(complete_packet(args, dispatch, report["cost"], report))
        report["status"] = "pass"
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    report["cpu_audit_seconds"] = time.perf_counter() - started
    with args.output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
