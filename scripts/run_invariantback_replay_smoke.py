import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.invariantback_runner import (
    CONSTRUCTION_ROW_ORDERS,
    LABELS,
    IndexedAdditiveAction,
    candidate_mean_log_likelihood,
    encode_candidate_batch,
    render_prompt,
)
from projection_transport_steering.torch_runtime import TorchLayerAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(LABELS), default="normbank")
    parser.add_argument("--layer", type=int, default=12)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    data = json.loads(args.data.read_text())
    group = data["datasets"][args.dataset]["sentinel"][0]
    labels = LABELS[args.dataset]
    source, target = labels[0], labels[-1]
    prompt, candidates = render_prompt(
        args.dataset,
        group["endpoints"][source]["text"],
        labels,
        ("A", "B", "C"),
        CONSTRUCTION_ROW_ORDERS[0],
        0,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    batch = encode_candidate_batch(
        tokenizer,
        [prompt, prompt],
        [candidates[target], candidates[source]],
        device="cuda",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        base_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        zero_action = IndexedAdditiveAction(
            torch.zeros(model.config.hidden_size, device="cuda"),
            batch["prefix_lengths"] - 1,
        )
        with TorchLayerAction(model, args.layer, zero_action).installed():
            replay_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
    replay_error = float((base_logits - replay_logits).abs().max())
    delta = torch.zeros(model.config.hidden_size, device="cuda", requires_grad=True)
    action = IndexedAdditiveAction(delta, batch["prefix_lengths"] - 1)
    with TorchLayerAction(model, args.layer, action).installed():
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    scores = [
        candidate_mean_log_likelihood(
            logits[index : index + 1],
            batch["candidate_ids"][index : index + 1],
            batch["candidate_mask"][index : index + 1],
            batch["prefix_lengths"][index : index + 1],
        )
        for index in range(2)
    ]
    margin = scores[0] - scores[1]
    covector = torch.autograd.grad(margin, delta)[0].float()
    receipt = {
        "covector_norm": float(covector.norm()),
        "covector_sha256": hashlib.sha256(covector.cpu().numpy().tobytes()).hexdigest(),
        "cuda": torch.version.cuda,
        "data_sha256": file_sha256(args.data),
        "dataset": args.dataset,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "group_id": group["group_id"],
        "layer": args.layer,
        "margin": float(margin.detach()),
        "model": str(args.model),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "platform": platform.platform(),
        "replay_max_abs_logit_error": replay_error,
        "status": "pass" if replay_error == 0.0 and torch.isfinite(covector).all() else "fail",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "host": socket.gethostname(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "pass" or receipt["covector_norm"] == 0.0:
        raise RuntimeError("InvariantBack replay smoke failed")


if __name__ == "__main__":
    main()
