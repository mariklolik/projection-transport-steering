import json

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from projection_transport_steering.cacheback_runner import (
    _forward_kl,
    development_cases,
    match_actions_to_effect,
    rollout_seed,
    run_case,
    run_sentinel_shard,
    sentinel_cases,
)


def test_match_actions_to_effect_uses_the_same_bounded_scale_bracket():
    actions = {
        "short": np.array([1.0, 0.0]),
        "long": np.array([2.0, 0.0]),
    }

    matched, evaluations, metadata = match_actions_to_effect(
        actions,
        lambda action: {
            "immediate_semantic_effect_mean": float(action[0] + 0.5 * action[0] ** 2)
        },
        target=0.5,
    )

    assert set(matched) == set(actions)
    assert all(abs(row["immediate_semantic_effect_mean"] - 0.5) <= 1e-4 for row in evaluations.values())
    assert all(row["scale_bracket"] == [0.0, 1.0] for row in metadata.values())
    assert all(row["evaluation_count"] <= 24 for row in metadata.values())
    assert all(0.0 <= row["scale"] <= 1.0 for row in metadata.values())


def test_match_actions_to_effect_rejects_extrapolation():
    with pytest.raises(RuntimeError, match="unreachable"):
        match_actions_to_effect(
            {"weak": np.array([0.1])},
            lambda action: {"immediate_semantic_effect_mean": float(action[0])},
            target=0.5,
        )


def test_match_actions_to_effect_can_record_unreachable_endpoints():
    matched, evaluations, metadata = match_actions_to_effect(
        {"weak": np.array([0.1])},
        lambda action: {"immediate_semantic_effect_mean": float(action[0])},
        target=0.5,
        require_all=False,
    )

    assert matched["weak"][0] == pytest.approx(0.1)
    assert evaluations["weak"]["immediate_semantic_effect_mean"] == pytest.approx(0.1)
    assert metadata["weak"]["reached"] is False
    assert metadata["weak"]["scale"] == 1.0


def test_forward_kl_is_stable_for_nearly_identical_logits():
    torch.manual_seed(0)
    base = torch.randn(9, 31) * 20
    steered = base + torch.randn_like(base) * 1e-4

    result = _forward_kl(base, steered)

    assert result.dtype == torch.float64
    assert torch.all(result >= 0)


def test_rollout_seed_is_state_specific_and_deterministic():
    assert rollout_seed("state-a", 0) == rollout_seed("state-a", 0)
    assert rollout_seed("state-a", 0) != rollout_seed("state-a", 1)
    assert rollout_seed("state-a", 0) != rollout_seed("state-b", 0)


def test_sentinel_cases_are_exact_and_never_open_later_stages():
    data = {
        "contexts": {
            "third": [
                {"context_sha256": f"t-{index}", "stage": "sentinel"}
                for index in range(3)
            ]
            + [{"context_sha256": "t-pilot", "stage": "pilot"}],
            "ing": [
                {"context_sha256": f"i-{index}", "stage": "sentinel"}
                for index in range(3)
            ],
            "past": [
                {"context_sha256": f"p-{index}", "stage": "sentinel"}
                for index in range(2)
            ]
            + [{"context_sha256": "p-development", "stage": "development"}],
        }
    }

    cases = sentinel_cases(data)

    assert len(cases) == 8
    assert all(case["stage"] == "sentinel" for case in cases)
    assert {case["concept"] for case in cases} == {"third", "ing", "past"}


def test_development_cases_are_balanced_and_never_open_pilot():
    data = {
        "contexts": {
            concept: [
                {"context_sha256": f"{concept}-{index}", "stage": "development"}
                for index in range(8)
            ]
            + [{"context_sha256": f"{concept}-pilot", "stage": "pilot"}]
            for concept in ("third", "ing", "past")
        }
    }

    cases = development_cases(data)

    assert len(cases) == 24
    assert all(case["stage"] == "development" for case in cases)
    assert {concept: sum(case["concept"] == concept for case in cases) for concept in data["contexts"]} == {
        "third": 8,
        "ing": 8,
        "past": 8,
    }


def test_run_case_produces_nested_exact_horizon_actions_on_tiny_gpt2():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=2,
            n_positions=16,
            vocab_size=31,
            eos_token_id=30,
            pad_token_id=30,
        )
    ).eval()
    case = {
        "concept": "third",
        "context_sha256": "tiny-state",
        "stage": "sentinel",
        "token_ids": [3, 5, 7],
    }
    mapping = {
        "base_ids": [1],
        "target_ids": [2],
        "pairs": [{"base_id": 1, "target_id": 2}],
    }

    result = run_case(
        model,
        case,
        mapping,
        layer=0,
        horizons=(0, 1, 2),
        trajectories=2,
        target=0.01,
        top_k=10,
        directions={
            "fixed_direction": model.lm_head.weight[2].detach()
            - model.lm_head.weight[1].detach()
        },
        realized_effect_target=0.005,
        include_actions=True,
    )

    assert result["status"] == "pass"
    assert result["trajectory_count"] == 2
    assert result["fisher_top_k"] == 10
    assert result["effect_matching"]["target"] == 0.005
    assert result["automatic_differentiation"] in {"forward_mode", "reverse_mode_fallback"}
    assert result["jacobian_calls"] == 2
    if result["automatic_differentiation"] == "forward_mode":
        assert result["jvp_count"] == 24
        assert result["vjp_count"] == 0
    else:
        assert result["jvp_count"] == 0
        assert result["vjp_count"] == 72
    assert result["full_case_seconds"] > result["geometry_seconds"]
    assert set(result["methods"]) == {
        "cacheback_h0",
        "cacheback_h1",
        "cacheback_h2",
        "euclidean",
        "fixed_direction",
    }
    assert set(result["effect_matching"]["methods"]) == set(result["methods"])
    assert set(result["actions"]) == set(result["methods"])
    assert all(len(action) == 12 for action in result["actions"].values())
    assert result["minimum_increment_eigenvalue"] >= -1e-8
    assert all(len(method["offset_forward_kl_mean"]) == 3 for method in result["methods"].values())
    assert all("action_norm" in method for method in result["methods"].values())
    assert all(
        "predicted_cumulative_quadratic_cost" in method for method in result["methods"].values()
    )
    assert all("predicted_horizon_quadratic_cost" in method for method in result["methods"].values())
    assert all(
        abs(method["immediate_semantic_effect_mean"] - 0.005) <= 1e-4
        for method in result["methods"].values()
    )
    json.dumps(result, sort_keys=True)


def test_run_case_supports_an_h0_only_timing_path():
    torch.manual_seed(13)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=2,
            n_positions=16,
            vocab_size=31,
            eos_token_id=30,
            pad_token_id=30,
        )
    ).eval()
    case = {
        "concept": "past",
        "context_sha256": "h0-only",
        "stage": "sentinel",
        "token_ids": [3, 5, 7],
    }
    mapping = {
        "base_ids": [1],
        "target_ids": [2],
        "pairs": [{"base_id": 1, "target_id": 2}],
    }

    result = run_case(model, case, mapping, layer=0, horizons=(0,), trajectories=2, target=0.01)

    assert result["horizons"] == [0]
    assert result["derivative_rows"] == 12
    assert result["requested_trajectory_count"] == 2
    assert result["trajectory_count"] == 1
    assert set(result["methods"]) == {"cacheback_h0", "euclidean"}


def test_run_case_reuses_one_geometry_for_a_configuration_grid():
    torch.manual_seed(23)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=2,
            n_positions=16,
            vocab_size=31,
            eos_token_id=30,
            pad_token_id=30,
        )
    ).eval()
    case = {
        "concept": "third",
        "context_sha256": "grid-state",
        "stage": "development",
        "token_ids": [3, 5, 7],
    }
    mapping = {
        "base_ids": [1],
        "target_ids": [2],
        "pairs": [{"base_id": 1, "target_id": 2}],
    }

    result = run_case(
        model,
        case,
        mapping,
        layer=0,
        horizons=(0, 1),
        trajectories=1,
        configurations=((0.01, 0.1, 0.005), (0.01, 1.0, 0.005)),
    )

    assert result["jacobian_calls"] == 1
    assert len(result["configurations"]) == 2
    assert [row["regularization_multiplier"] for row in result["configurations"]] == [0.1, 1.0]
    assert all(set(row["effect_matching"]["methods"]) == set(row["methods"]) for row in result["configurations"])


def test_run_sentinel_shard_writes_complete_receipt_without_pilot_access(tmp_path):
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=2,
            n_positions=16,
            vocab_size=31,
            bos_token_id=30,
            eos_token_id=30,
            pad_token_id=30,
        )
    ).eval()
    contexts = {
        "third": [
            {"context_sha256": f"t-{index}", "stage": "sentinel", "token_ids": [3, 5, 7]}
            for index in range(3)
        ],
        "ing": [
            {"context_sha256": f"i-{index}", "stage": "sentinel", "token_ids": [3, 5, 7]}
            for index in range(3)
        ],
        "past": [
            {"context_sha256": f"p-{index}", "stage": "sentinel", "token_ids": [3, 5, 7]}
            for index in range(2)
        ]
        + [{"context_sha256": "p-pilot", "stage": "pilot", "token_ids": [3, 5, 7]}],
    }
    mapping = {
        "base_ids": [1],
        "target_ids": [2],
        "pairs": [{"base_id": 1, "target_id": 2}],
    }
    data = {"contexts": contexts, "mappings": {name: mapping for name in contexts}}
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    output = tmp_path / "shard"

    run_sentinel_shard(
        model,
        data,
        data_path,
        output,
        shard_index=0,
        shard_count=8,
        layer=0,
        horizons=(0, 1),
        trajectories=1,
        target=0.01,
    )

    receipt = json.loads((output / "receipt.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert receipt["case_count"] == 1
    assert receipt["pilot_states_observed"] == 0
    assert len(rows) == 1
    assert rows[0]["status"] == "pass"
