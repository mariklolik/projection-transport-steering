import copy
import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from projection_transport_steering.branch_runtime import advance_branches, start_branches
from projection_transport_steering.outcome_score import (
    score_math_outputs,
    score_aime_benchmark_completion,
    score_aime_completion,
    score_aime_math_verify_completion,
    score_aime_reward_completion,
)
from projection_transport_steering.torch_runtime import (
    AdditiveAction,
    TorchLayerAction,
    _hidden,
    _resolve_layers,
)


def rollout_seed(group_id: str, rollout_index: int) -> int:
    if not group_id or rollout_index < 0:
        raise ValueError("invalid rollout identity")
    digest = hashlib.sha256(f"ost-v1|{group_id}|{rollout_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def format_aime_prompt(problem: str, answer_format: str = "line") -> str:
    normalized = " ".join(problem.split())
    if not normalized:
        raise ValueError("problem must be nonempty")
    if answer_format == "line":
        return (
            "Solve the following problem. Show your reasoning, then end with exactly one line "
            f"in the required form.\n\nProblem: {normalized}\n\nFinal Answer: <integer>"
        )
    if answer_format in {"math_verify_strict", "reward_integer", "terminal_integer"}:
        return (
            "Solve the following problem, showing your reasoning concisely. End with exactly "
            f"one boxed integer.\n\nProblem: {normalized}\n\n\\boxed{{integer}}"
        )
    raise ValueError("unknown answer format")


class TorchLayerTrace:
    def __init__(self, model: nn.Module, site: int) -> None:
        layers = _resolve_layers(model)
        if site < 0 or site >= len(layers):
            raise ValueError("trace site is outside the decoder")
        self.layer = layers[site]
        self._states: list[Tensor] = []

    def start(self) -> None:
        self._states = []

    def _capture(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> None:
        hidden = _hidden(output)
        self._states.append(hidden[:, -1, :].detach())

    @contextmanager
    def installed(self) -> Iterator["TorchLayerTrace"]:
        handle = self.layer.register_forward_hook(self._capture)
        try:
            yield self
        finally:
            handle.remove()

    def finish(self) -> Tensor:
        if not self._states:
            raise RuntimeError("activation trace is empty")
        states = torch.cat(self._states, dim=0).float().cpu()
        if not torch.all(torch.isfinite(states)):
            raise RuntimeError("activation trace is nonfinite")
        return states


def generate_rollout(
    model: nn.Module,
    tokenizer: Any,
    trace: TorchLayerTrace,
    source: Mapping[str, object],
    rollout_index: int,
    max_new_tokens: int,
    device: torch.device,
    enable_thinking: bool = True,
    answer_format: str = "line",
) -> tuple[dict[str, object], Tensor]:
    group_id = str(source["group_id"])
    seed = rollout_seed(group_id, rollout_index)
    prompt = format_aime_prompt(str(source["problem"]), answer_format)
    tokenized = tokenizer.apply_chat_template(
        [{"content": prompt, "role": "user"}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True,
        return_tensors="pt",
        tokenize=True,
    ).to(device)
    if isinstance(tokenized, Tensor):
        model_inputs = {"input_ids": tokenized}
    elif isinstance(tokenized, Mapping) and isinstance(tokenized.get("input_ids"), Tensor):
        model_inputs = dict(tokenized)
    else:
        raise TypeError("chat template returned unsupported inputs")
    input_ids = model_inputs["input_ids"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    trace.start()
    with torch.inference_mode(), trace.installed():
        output_ids = model.generate(
            **model_inputs,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            top_p=0.95,
            use_cache=True,
        )
    generated_ids = output_ids[0, input_ids.shape[1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    states = trace.finish()
    if states.shape[0] != generated_ids.numel():
        raise RuntimeError("activation trace and generated-token counts differ")
    scorers = {
        "line": score_aime_completion,
        "math_verify_strict": score_aime_math_verify_completion,
        "reward_integer": score_aime_reward_completion,
        "terminal_integer": score_aime_benchmark_completion,
    }
    score = scorers[answer_format](completion, int(source["answer"]))
    return {
        **score,
        "completion": completion,
        "generated_tokens": int(generated_ids.numel()),
        "generation_id": f"{group_id}:{rollout_index}",
        "group_id": group_id,
        "rollout_index": rollout_index,
        "seed": seed,
        "trace_length": len(states),
    }, states


def run_math_batch(
    model, tokenizer, rows, config, condition, graders, *, reference_path, hash_payload
) -> dict:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": config["instruction"] + row["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(
        model.device
    )
    seeds = [rollout_seed(row["cluster_id"], 0) for row in rows]
    state = start_branches(inputs.input_ids, inputs.attention_mask, seeds)
    eos = model.generation_config.eos_token_id
    if eos is None or model.generation_config.pad_token_id is None:
        raise ValueError("model must declare EOS and padding")
    settings = {
        **config["sampling"],
        "eos_token_ids": tuple(eos) if isinstance(eos, list) else (int(eos),),
        "pad_token_id": model.generation_config.pad_token_id,
    }
    cuda = model.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    reference = advance_branches(model, state, steps=condition["steps"], **settings)
    if cuda:
        torch.cuda.synchronize()
    reference_seconds = time.perf_counter() - started
    width = inputs.input_ids.shape[1]
    scored = []
    for index, row in enumerate(rows):
        length = int(reference.generated_lengths[index])
        tokens = reference.input_ids[index, width : width + length].tolist()
        completion = tokenizer.decode(tokens, skip_special_tokens=True)
        terminated = bool(reference.terminated[index])
        scored.append(
            {
                "row_id": row["row_id"],
                "cluster_id": row["cluster_id"],
                "seed": seeds[index],
                "prompt_sha256": hash_payload(prompts[index]),
                "prompt_tokens": int(inputs.attention_mask[index].sum()),
                "prompt_token_sha256": hash_payload(
                    inputs.input_ids[index][inputs.attention_mask[index].bool()].tolist()
                ),
                "token_ids": tokens,
                "completion": completion,
                "generated_tokens": length,
                "terminated": terminated,
                "sensitivity_supported": row["sensitivity_supported"],
            }
        )
    with reference_path.open("x") as handle:
        json.dump(
            {"rows": scored, "reference_seconds": reference_seconds},
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
    checks = {}
    coverage = {"intervention_path": "not_requested"}
    check_tokens = {}
    check_started = time.perf_counter()
    if condition["checks"]:
        site = model.config.num_hidden_layers // 2 - 1
        with TorchLayerAction(model, site, nn.Identity()).installed():
            zero = advance_branches(model, state, steps=condition["steps"], **settings)
        prefix = advance_branches(model, state, steps=condition["steps"] // 2, **settings)
        split = advance_branches(model, prefix, steps=condition["steps"] // 2, **settings)
        saved_cache = copy.deepcopy(prefix.cache)
        vector = torch.linspace(
            -0.01, 0.01, model.config.hidden_size, device=model.device, dtype=model.dtype
        )
        with TorchLayerAction(model, site, AdditiveAction(vector)).installed():
            action = advance_branches(
                model, prefix, steps=min(64, condition["steps"] // 4), **settings
            )
        checks = {
            "zero_replay_exact": torch.equal(reference.input_ids, zero.input_ids),
            "split_replay_exact": torch.equal(reference.input_ids, split.input_ids),
            "zero_rng_exact": all(
                torch.equal(a, b)
                for a, b in zip(reference.rng_states, zero.rng_states, strict=True)
            ),
            "split_rng_exact": all(
                torch.equal(a, b)
                for a, b in zip(reference.rng_states, split.rng_states, strict=True)
            ),
            "parent_cache_unchanged": all(
                torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
                for a, b in zip(saved_cache.layers, prefix.cache.layers, strict=True)
            ),
        }
        at_risk = int((~prefix.terminated).sum())
        coverage = {
            "at_risk_at_cut": at_risk,
            "action_forward_calls": action.input_ids.shape[1] - prefix.input_ids.shape[1],
            "action_generated_lengths": (
                action.generated_lengths - prefix.generated_lengths
            ).tolist(),
            "intervention_path": "exercised" if at_risk else "not_exercised_absorbing",
        }
        check_tokens = {
            name: value.input_ids[:, width:].tolist()
            for name, value in {"zero": zero, "split": split, "action": action}.items()
        }
    if cuda:
        torch.cuda.synchronize()
    check_seconds = time.perf_counter() - check_started
    grade_started = time.perf_counter()
    for output, source in zip(scored, rows, strict=True):
        output.update(
            score_math_outputs(
                output["completion"],
                source["solution"],
                output["terminated"],
                source["sensitivity_supported"],
                *graders,
            )
        )
    return {
        "rows": scored,
        "checks": checks,
        "check_coverage": coverage,
        "check_tokens": check_tokens,
        "reference_seconds": reference_seconds,
        "check_seconds": check_seconds,
        "grading_seconds": time.perf_counter() - grade_started,
    }
