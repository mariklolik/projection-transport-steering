import argparse
import json
import runpy
import time
from pathlib import Path

import torch
from math_verify.errors import TimeoutException
from transformers import AutoModelForCausalLM

from audit_irc_sources import load_math_author
from irc_baseline_support import evaluate, prepare_inputs, train_adapter
from materialize_outcome_score_data import file_sha256, payload_sha256
from run_irc_fit_probe import load_examples
from projection_transport_steering.attention_backend import register_fp32_prefill_flex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    if not 0 <= args.candidate < len(config["candidates"]):
        raise ValueError("candidate index outside the frozen family")
    candidate = config["candidates"][args.candidate]
    tokenizer, examples, rows = prepare_inputs(config, root, args.model)
    extract, equivalent = load_math_author(root)
    boxed = runpy.run_path(str(root / ".external/math-author-source/modeling/dataset/util.py"))[
        "last_boxed_only_string"
    ]
    action_type = runpy.run_path(str(root / ".external/pyreft/pyreft/interventions.py"))[
        "LoreftIntervention"
    ]
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "candidate": candidate,
        "config_sha256": file_sha256(args.config),
        "source_sha256": config["source_sha256"],
        "scientific_role": "exploratory_baseline_selection",
        "positive_action_admitted": False,
        "training_resume_supported": False,
    }
    failure = None
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if config["attention"] in {"fp32_prefill_flex", "fp32_prefill_flex_eager"}:
            register_fp32_prefill_flex()
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
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
        versions = {name: p._version for name, p in model.named_parameters()}
        shares = config["zero_shares"][args.candidate]
        receipt["zero"] = evaluate(
            model,
            tokenizer,
            [rows[i] for i in shares],
            config,
            candidate,
            None,
            (extract, equivalent, boxed),
            args.output / "zero",
        )
        tokenizer.padding_side = "right"
        torch.manual_seed(config["recipe"]["seed"])
        action = action_type(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=candidate["rank"],
            dtype=dtype,
            dropout=0.0,
        ).to(args.device)
        wrapper = None
        if args.device == "cuda":
            from transformers.integrations.flex_attention import (
                WrappedFlexAttention,
                flex_attention,
            )

            wrapper = WrappedFlexAttention(False)
            original = wrapper._compiled_flex_attention
            torch.compiler.reset()
            compiled = torch.compile(flex_attention, dynamic=False)
            wrapper._compiled_flex_attention = compiled
        receipt["fit"] = train_adapter(
            model, action, examples, tokenizer, config["recipe"], candidate, args.output
        )
        if wrapper is not None:
            if wrapper._compiled_flex_attention is not compiled or wrapper.training:
                raise RuntimeError("fit compiler callable changed")
            torch.compiler.reset()
            wrapper._compiled_flex_attention = original
        restored = action_type(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=candidate["rank"],
            dtype=dtype,
            dropout=0.0,
        ).to(args.device)
        raw = torch.load(args.output / "adapter.pt", map_location=args.device, weights_only=True)
        torch.nn.Module.load_state_dict(restored, raw, strict=True)
        if not all(
            torch.equal(value, raw[key])
            for key, value in torch.nn.Module.state_dict(action).items()
        ):
            raise RuntimeError("saved adapter state differs")
        del action, raw
        receipt["selection"] = evaluate(
            model,
            tokenizer,
            rows,
            config,
            candidate,
            restored.eval(),
            (extract, equivalent, boxed),
            args.output / "selection",
        )
        receipt["base_parameter_versions_unchanged"] = versions == {
            name: p._version for name, p in model.named_parameters()
        }
        if not receipt["base_parameter_versions_unchanged"]:
            raise RuntimeError("frozen base was modified")
        receipt["status"] = "pass"
    except (Exception, TimeoutException) as error:
        failure = error
        receipt.update(status="fail", failure_type=type(error).__name__, failure_message=str(error))
    receipt["total_elapsed_seconds"] = time.perf_counter() - started
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"event": "baseline_terminal", "status": receipt["status"]}), flush=True)
    if failure is not None:
        raise RuntimeError("baseline incomplete; artifacts retained") from failure


if __name__ == "__main__":
    main()
