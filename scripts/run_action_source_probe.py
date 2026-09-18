import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def validate_outputs(
    requests: list[dict], outputs: list[dict], eos: list[int], cap: int, *, vocab_size: int
) -> list[dict]:
    expected = {r["request_id"]: r for r in requests}
    actual = {r["meta_info"]["id"]: r for r in outputs}
    if len(expected) != len(requests) or len(actual) != len(outputs) or expected.keys() != actual.keys():
        raise ValueError("request/output coverage mismatch")
    result = []
    for key, request in expected.items():
        output = actual[key]
        tokens, meta = output["output_ids"], output["meta_info"]
        reason = meta["finish_reason"]
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= cap
            or not all(type(token) is int and 0 <= token < vocab_size for token in tokens)
            or meta["prompt_tokens"] != len(request["input_ids"])
            or meta["completion_tokens"] != len(tokens)
            or meta["num_retractions"] != 0
        ):
            raise ValueError("invalid raw tokens, lengths or retractions")
        ended = reason.get("type") == "stop"
        positions = [i for i, token in enumerate(tokens) if token in eos]
        if ended:
            if positions != [len(tokens) - 1] or reason.get("matched") != tokens[-1]:
                raise ValueError("EOS missing, trimmed or inconsistent")
        elif reason.get("type") != "length" or len(tokens) != cap or positions:
            raise ValueError("invalid capped or aborted completion")
        result.append({**request, **output, "terminated": ended})
    return result


def finalize_receipt(receipt, path, started, engine, monitor, telemetry) -> None:
    actions = [("telemetry", telemetry.close)]
    if engine is not None:
        actions.insert(0, ("engine", engine.shutdown))
    if monitor is not None:
        actions.extend([
            ("monitor_terminate", monitor.terminate),
            ("monitor_wait", lambda: monitor.wait(timeout=10)),
        ])
    errors = []
    for name, action in actions:
        try:
            action()
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if monitor is not None and monitor.poll() is None:
        try:
            monitor.kill()
            monitor.wait(timeout=10)
        except Exception as error:
            errors.append(f"monitor_kill: {type(error).__name__}: {error}")
    if errors:
        receipt.update(status="failed", cleanup_errors=errors)
    receipt["elapsed_seconds"] = time.perf_counter() - started
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    raw = args.packet.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.packet_sha256:
        raise ValueError("frozen packet hash mismatch")
    packet = json.loads(raw)
    root = Path(__file__).resolve().parents[1]
    for directory, hashes in (
        (root, packet["source_sha256"]),
        (Path(packet["engine"]["model_path"]), packet["model_files_sha256"]),
    ):
        for name, expected in hashes.items():
            with (directory / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError(f"source/model hash mismatch: {name}")
    versions = {name: importlib.metadata.version(name) for name in packet["expected_versions"]}
    if versions != packet["expected_versions"]:
        raise ValueError("source runtime version mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "packet_sha256": args.packet_sha256,
        "started_at": datetime.now(UTC).isoformat(),
        "versions": versions,
        "stages": [],
    }
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    monitor = None
    engine = None
    telemetry = (args.output / "gpu.csv").open("x")
    try:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-l", "5",
            ],
            stdout=telemetry,
            stderr=subprocess.STDOUT,
        )
        import sglang

        loading = time.perf_counter()
        engine = sglang.Engine(**packet["engine"])
        receipt["engine_startup_seconds"] = time.perf_counter() - loading
        (args.output / "server-info.json").write_text(json.dumps(engine.get_server_info(), indent=2))
        reference = {}
        for stage in packet["stages"]:
            requests = [packet["requests"][i] for i in stage["indices"]]
            sampling = {**packet["sampling"], "max_new_tokens": stage["cap"]}
            before = time.perf_counter()
            outputs = engine.generate(
                input_ids=[r["input_ids"] for r in requests],
                sampling_params=[{**sampling, "sampling_seed": r["seed"]} for r in requests],
                rid=[r["request_id"] for r in requests],
            )
            elapsed = time.perf_counter() - before
            stage_path = args.output / f"{stage['name']}.json"
            with stage_path.open("x") as handle:
                json.dump({"stage": stage, "seconds": elapsed, "outputs": outputs}, handle)
            rows = validate_outputs(
                requests, outputs, sampling["stop_token_ids"], stage["cap"],
                vocab_size=packet["vocab_size"],
            )
            summary = {
                "name": stage["name"],
                "seconds": elapsed,
                "requests": len(rows),
                "tokens": sum(len(r["output_ids"]) for r in rows),
                "eos": sum(r["terminated"] for r in rows),
                "prompt_lengths": [len(r["input_ids"]) for r in rows],
                "output_lengths": [len(r["output_ids"]) for r in rows],
            }
            if stage["name"] == "main":
                reference = {r["request_id"]: r for r in rows}
            elif stage["name"] != "warmup":
                summary["repeated_tokens_equal"] = all(
                    r["output_ids"] == reference[r["request_id"]]["output_ids"]
                    and r["meta_info"]["finish_reason"]
                    == reference[r["request_id"]]["meta_info"]["finish_reason"]
                    for r in rows
                )
            receipt["stages"].append(summary)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
        receipt["status"] = (
            "pass" if all(s.get("repeated_tokens_equal", True) for s in receipt["stages"])
            else "failed_repeat_invariance"
        )
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        finalize_receipt(receipt, receipt_path, started, engine, monitor, telemetry)
    if receipt["status"] != "pass":
        raise RuntimeError("source probe did not qualify; see final receipt")


if __name__ == "__main__":
    main()
