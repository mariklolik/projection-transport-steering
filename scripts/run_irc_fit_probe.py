import argparse
import importlib.metadata
import json
import runpy
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from audit_irc_sources import load_math_author
from materialize_outcome_score_data import file_sha256, payload_sha256
from projection_transport_steering.attention_backend import register_fp32_prefill_flex
from projection_transport_steering.reft_training import fit_step, supervised_example


def load_examples(config: dict, root: Path, tokenizer) -> dict:
    path = root / config["packet"]
    if file_sha256(path) != config["packet_sha256"]:
        raise ValueError("fit packet hash mismatch")
    packet = json.loads(path.read_text())
    expected = packet.pop("content_sha256")
    if payload_sha256(packet) != expected or packet["allocation"] != "fit":
        raise ValueError("invalid fit packet content or allocation")
    rows = {row["cluster_id"]: row for row in packet["rows"]}
    if len(rows) != len(packet["rows"]) or not all(
        row["allocation"] == "fit" and row["split"] == "train" and row["reference_eligible"]
        for row in rows.values()
    ):
        raise ValueError("fit packet violates source eligibility")
    extract, _ = load_math_author(root)
    groups = {}
    for name, ids in config["groups"].items():
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("invalid probe group identities")
        groups[name] = [
            supervised_example(
                tokenizer,
                [{"role": "user", "content": config["instruction"] + rows[key]["problem"]}],
                rows[key]["solution"],
                extract(rows[key]["solution"]),
                config["maximum_length"],
                eos_token_ids=tuple(config.get("eos_token_ids", ())),
            )
            for key in ids
        ]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    versions = {name: importlib.metadata.version(name) for name in config["expected_versions"]}
    if versions != config["expected_versions"]:
        raise ValueError("fit probe package versions differ")
    for name, digest in config.get("runtime_files_sha256", {}).items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"fit probe installed source hash differs: {name}")
    for name, digest in config["source_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"fit probe source hash differs: {name}")
    for name, digest in config["model_files_sha256"].items():
        if file_sha256(args.model / name) != digest:
            raise ValueError(f"fit probe model file hash differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, padding_side="right"
    )
    groups = load_examples(config, root, tokenizer)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if config["attention"] == "fp32_prefill_flex":
        register_fp32_prefill_flex()
    action_type = runpy.run_path(str(root / ".external/pyreft/pyreft/interventions.py"))[
        "LoreftIntervention"
    ]
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "baseline_eligible": False,
        "step_sha256": {},
        "config_sha256": file_sha256(args.config),
        "source_sha256": config["source_sha256"],
        "versions": versions,
        "groups": config["groups"],
        "device": args.device,
        "target_example_sha256": {
            name: payload_sha256(examples) for name, examples in groups.items()
        },
    }
    failure = None
    cuda = args.device == "cuda"
    try:
        dtype = torch.bfloat16 if cuda else torch.float32
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                attn_implementation=config["attention"],
                local_files_only=True,
            )
            .to(args.device)
            .eval()
            .requires_grad_(False)
        )
        base_versions = {name: parameter._version for name, parameter in model.named_parameters()}
        receipt.update(
            attention=model.config._attn_implementation,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            base_dtype=str(model.dtype),
            gpu=torch.cuda.get_device_name() if cuda else None,
        )
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=config["pad_multiple"])
        for application in config["applications"]:
            torch.manual_seed(config["seed"])
            action = action_type(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config["rank"],
                dtype=dtype,
                dropout=0.0,
            ).to(args.device)
            optimizer = torch.optim.AdamW(
                action.parameters(),
                lr=config["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            receipt.setdefault("adapter_parameters", {})[application] = {
                name: {"count": p.numel(), "dtype": str(p.dtype)}
                for name, p in action.named_parameters()
            }
            for group, examples in groups.items():
                batch = {key: value.to(args.device) for key, value in collator(examples).items()}
                for repeat in range(config["repeats"]):
                    if cuda:
                        torch.cuda.synchronize()
                    step_started = time.perf_counter()
                    step = fit_step(model, action, batch, config["site"], application, optimizer)
                    if cuda:
                        torch.cuda.synchronize()
                    step.update(
                        application=application,
                        group=group,
                        repeat=repeat,
                        seconds=time.perf_counter() - step_started,
                        padded_shape=list(batch["input_ids"].shape),
                        peak_memory_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                    )
                    path = args.output / f"{application}-{group}-{repeat}.json"
                    with path.open("x") as handle:
                        json.dump(step, handle, sort_keys=True, indent=2)
                        handle.write("\n")
                    receipt["step_sha256"][path.name] = file_sha256(path)
                    print(json.dumps(step), flush=True)
            del optimizer, action
        unchanged = base_versions == {name: p._version for name, p in model.named_parameters()}
        receipt["base_parameter_versions_unchanged"] = unchanged
        if not unchanged or len(receipt["step_sha256"]) != (
            len(config["applications"]) * len(config["groups"]) * config["repeats"]
        ):
            raise RuntimeError("incomplete fit probe or base parameter mutation")
        receipt["status"] = "pass"
    except Exception as error:
        failure = error
        receipt.update(status="fail", failure_type=type(error).__name__, failure_message=str(error))
    receipt["total_elapsed_seconds"] = time.perf_counter() - started
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "event": "fit_probe_complete",
                "status": receipt["status"],
                "steps": len(receipt["step_sha256"]),
            }
        ),
        flush=True,
    )
    if failure is not None:
        raise RuntimeError("fit probe failed; diagnostic artifacts retained") from failure


if __name__ == "__main__":
    main()
