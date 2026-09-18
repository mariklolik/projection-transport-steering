import argparse
import ast
import hashlib
import json
import platform
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from projection_transport_steering.flow_step_support import condition_specifications

MODEL_CONFIG_SHA256 = "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"
MODEL_MANIFEST_SHA256 = "78a536738d6391c113dc4a5c239644b4336c9db223bba7a576d07c7cb6a1055b"
UPSTREAM_JUDGE_SHA256 = "2a8997810b1930d817262e9c3c249299cdda8804ddab67d72057d1ebe7b62755"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--upstream-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top_level_manifest(path: Path) -> tuple[list[dict[str, object]], str]:
    files = sorted(file for file in path.iterdir() if file.is_file())
    lines = "".join(f"{file.name} {file.stat().st_size}\n" for file in files)
    manifest = [{"name": file.name, "size": file.stat().st_size} for file in files]
    return manifest, hashlib.sha256(lines.encode()).hexdigest()


def load_upstream(path: Path) -> SimpleNamespace:
    names = {
        "CONCEPT_PROMPT",
        "FLUENCY_PROMPT",
        "INSTRUCTION_PROMPT",
        "extract_rating",
        "harmonic_mean",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        assigned = {
            target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)
        }
        if isinstance(node, ast.FunctionDef) and node.name in names or assigned & names:
            body.append(node)
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    if not names <= namespace.keys():
        raise RuntimeError("upstream judge definitions are incomplete")
    return SimpleNamespace(**{name: namespace[name] for name in names})


def validate_generations(rows: list[dict[str, object]]) -> tuple[list[int], int]:
    condition_names = [str(specification["name"]) for specification in condition_specifications()]
    concept_ids = list(dict.fromkeys(int(row["concept_id"]) for row in rows))
    if not rows or not concept_ids or len(set(row["generation_id"] for row in rows)) != len(rows):
        raise RuntimeError("invalid generation identities")
    prompt_sets = {
        concept_id: {
            int(row["prompt_index"]) for row in rows if int(row["concept_id"]) == concept_id
        }
        for concept_id in concept_ids
    }
    prompt_limit = len(next(iter(prompt_sets.values())))
    expected_prompts = set(range(prompt_limit))
    expected = {
        (concept_id, condition, prompt_index)
        for concept_id in concept_ids
        for condition in condition_names
        for prompt_index in expected_prompts
    }
    observed = {
        (int(row["concept_id"]), str(row["condition"]), int(row["prompt_index"])) for row in rows
    }
    if (
        prompt_limit < 1
        or prompt_limit > 8
        or any(prompts != expected_prompts for prompts in prompt_sets.values())
        or observed != expected
        or len(rows) != len(expected)
        or any(not str(row["generation"]).strip() for row in rows)
    ):
        raise RuntimeError("generation grid is incomplete")
    return concept_ids, prompt_limit


def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.batch_size != 8 or args.max_new_tokens != 160:
        raise ValueError("judge settings differ from the freeze")
    generation_receipt_path = args.generations.parent / "receipt.json"
    generation_receipt = json.loads(generation_receipt_path.read_text())
    rows = json.loads(args.generations.read_text())
    concept_ids, prompt_limit = validate_generations(rows)
    generation_sha = file_sha256(args.generations)
    if (
        generation_receipt["status"] != "pass"
        or generation_receipt["generations_sha256"] != generation_sha
        or not generation_receipt["generation_serialization_exact"]
        or generation_receipt["planned_concept_ids"] != concept_ids
        or generation_receipt["prompt_limit"] != prompt_limit
        or generation_receipt["generations"] != len(rows)
    ):
        raise RuntimeError("generation receipt is invalid")
    manifest, manifest_sha = top_level_manifest(args.model)
    if file_sha256(args.model / "config.json") != MODEL_CONFIG_SHA256:
        raise RuntimeError("judge model configuration mismatch")
    if manifest_sha != MODEL_MANIFEST_SHA256:
        raise RuntimeError("judge model manifest mismatch")
    if file_sha256(args.upstream_judge) != UPSTREAM_JUDGE_SHA256:
        raise RuntimeError("upstream judge mismatch")
    upstream = load_upstream(args.upstream_judge)
    prompts = []
    for row in rows:
        prompts.extend(
            (
                upstream.CONCEPT_PROMPT.format(concept=row["concept"], sentence=row["generation"]),
                upstream.INSTRUCTION_PROMPT.format(
                    instruction=row["prompt"], sentence=row["generation"]
                ),
                upstream.FLUENCY_PROMPT.format(sentence=row["generation"]),
            )
        )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    forward_calls = 0

    def count_forward(module, inputs):
        nonlocal forward_calls
        forward_calls += 1

    counter = model.register_forward_pre_hook(count_forward)
    completions = []
    for offset in range(0, len(prompts), args.batch_size):
        batch = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts[offset : offset + args.batch_size]
        ]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded.input_ids.shape[1] :]
        completions.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    counter.remove()
    judged = []
    for index, row in enumerate(rows):
        outputs = completions[3 * index : 3 * index + 3]
        ratings = [upstream.extract_rating(output) for output in outputs]
        scores = None
        if all(rating is not None for rating in ratings):
            concept, instruction, fluency = ratings
            scores = {
                "concept": concept,
                "fluency": fluency,
                "harmonic_mean": upstream.harmonic_mean(ratings),
                "instruction": instruction,
            }
        judged.append({**row, "judge_completions": outputs, "scores": scores})
    torch.cuda.synchronize()
    parse_counts = {
        name: sum(row["condition"] == name and row["scores"] is not None for row in judged)
        for name in [str(specification["name"]) for specification in condition_specifications()]
    }
    return judged, {
        "batch_size": args.batch_size,
        "completions": len(completions),
        "concept_ids": concept_ids,
        "cuda": torch.version.cuda,
        "elapsed_seconds": time.perf_counter() - started,
        "generation_receipt_sha256": file_sha256(generation_receipt_path),
        "generations_sha256": generation_sha,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "judge_forward_calls": forward_calls,
        "max_new_tokens": args.max_new_tokens,
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "model_manifest": manifest,
        "model_manifest_sha256": manifest_sha,
        "model_snapshot": str(args.model.resolve()),
        "parse_counts_by_condition": parse_counts,
        "parsed": sum(row["scores"] is not None for row in judged),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "platform": platform.platform(),
        "prompt_limit": prompt_limit,
        "rows": len(judged),
        "status": "pass" if all(row["scores"] is not None for row in judged) else "parse_failure",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "upstream_judge_sha256": file_sha256(args.upstream_judge),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        rows, receipt = run(args)
        judgments = args.output / "judgments.json"
        judgments.write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))
        receipt["judgment_serialization_exact"] = json.loads(judgments.read_text()) == rows
        receipt["judgments_sha256"] = file_sha256(judgments)
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
        )
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": str(error), "status": "fail"}, indent=2, sort_keys=True)
        )
        raise


if __name__ == "__main__":
    main()
