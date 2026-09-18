import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from projection_transport_steering.amortized_pullback import (
    GPT2Downstream,
    build_subspace,
    full_pullback_geometry,
    restricted_pullback_direction,
)
from projection_transport_steering.amortized_pullback_experiment import (
    concept_probability,
    fixed_efficiency_direction,
    localize_targets,
    off_target_kl,
    pooled_metric,
    random_orthonormal_basis,
    run_trajectory,
    select_step_size,
)
from projection_transport_steering.amortized_pullback_runner import development_cases, prepare_case


def test_build_subspace_contains_covector_and_removes_dependence():
    covector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    shared = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )

    basis = build_subspace(covector, shared)

    assert basis.shape == (3, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2, dtype=torch.float64))
    assert torch.linalg.vector_norm(basis @ basis.T @ covector - covector) < 1e-10


def test_fixed_efficiency_direction_has_unit_norm_and_declared_cosine():
    covector = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    raw = torch.tensor([1.0, 3.0, 0.0], dtype=torch.float64)

    direction = fixed_efficiency_direction(raw, covector, 0.3)

    assert direction.norm() == pytest.approx(1.0)
    assert torch.dot(direction, covector / covector.norm()) == pytest.approx(0.3)


def test_concept_probability_and_off_target_kl_exclude_concept_tokens():
    base = torch.log(torch.tensor([0.4, 0.1, 0.2, 0.2, 0.1], dtype=torch.float64))
    steered = torch.log(torch.tensor([0.2, 0.3, 0.1, 0.3, 0.1], dtype=torch.float64))

    probability = concept_probability(steered, [0, 1], [2, 3])
    divergence = off_target_kl(base, steered, [0, 1, 2, 3])

    assert probability == pytest.approx(4.0 / 9.0)
    assert divergence == pytest.approx(0.0)


def test_pooled_regularization_is_median_positive_fisher_eigenvalue():
    gradients = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    metric, regularization = pooled_metric(gradients)

    expected = torch.diag(torch.tensor([3.0, 4.0 / 3.0, 1.0 / 3.0], dtype=gradients.dtype))
    assert torch.allclose(metric, expected)
    assert regularization == pytest.approx(4.0 / 3.0)


def test_gpt2_downstream_replays_unmodified_suffix():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=8,
            n_positions=16,
            vocab_size=23,
            bos_token_id=0,
            eos_token_id=0,
        )
    ).eval()
    input_ids = torch.tensor([[1, 4, 9, 3]])
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False, output_hidden_states=True)
    downstream = GPT2Downstream(model, 0, outputs.hidden_states[1])

    replayed = downstream(outputs.hidden_states[1][0, -1])

    assert torch.allclose(replayed, outputs.hidden_states[-1][0, -1], atol=1e-6, rtol=1e-5)


def test_prepare_case_replays_final_gpt2_block_without_double_normalization():
    torch.manual_seed(8)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=8,
            n_positions=16,
            vocab_size=23,
            bos_token_id=0,
            eos_token_id=0,
        )
    ).eval()
    case = {"layer": 1, "token_ids": [1, 4, 9, 3]}

    _, _, replayed_logits = prepare_case(model, case)
    with torch.no_grad():
        expected_logits = model(torch.tensor([case["token_ids"]]), use_cache=False).logits[0, -1]

    assert torch.allclose(replayed_logits, expected_logits, atol=1e-6, rtol=1e-5)


def test_gpt2_downstream_supports_restricted_forward_and_reverse_autodiff():
    torch.manual_seed(9)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=8,
            n_positions=16,
            vocab_size=23,
            bos_token_id=0,
            eos_token_id=0,
            _attn_implementation="eager",
        )
    ).eval()
    model.requires_grad_(False)
    outputs = model(
        torch.tensor([[1, 4, 9, 3]]),
        use_cache=False,
        output_hidden_states=True,
    )
    downstream = GPT2Downstream(model, 0, outputs.hidden_states[1])

    result = restricted_pullback_direction(
        downstream,
        outputs.hidden_states[1][0, -1].detach(),
        torch.randn(8),
        model.lm_head.weight.detach(),
        torch.linalg.qr(torch.randn(8, 2)).Q,
        0.2,
        10,
    )

    assert torch.isfinite(result["direction"]).all()
    assert torch.dot(result["covector"], result["direction"]) == pytest.approx(1.0)


def test_full_rank_restricted_pullback_matches_exact_direction():
    torch.manual_seed(11)
    matrix = torch.randn(4, 3, dtype=torch.float64)
    unembedding = torch.randn(7, 4, dtype=torch.float64)
    state = torch.randn(3, dtype=torch.float64)
    beta = torch.randn(4, dtype=torch.float64)

    def downstream(value):
        return matrix @ value

    geometry = full_pullback_geometry(downstream, state, beta, unembedding, 7)

    restricted = restricted_pullback_direction(
        downstream,
        state,
        beta,
        unembedding,
        torch.eye(3, dtype=torch.float64),
        0.2,
        7,
    )
    exact = torch.linalg.solve(geometry["metric"] + 0.2 * torch.eye(3), geometry["covector"])
    exact = exact / torch.dot(geometry["covector"], exact)

    assert torch.allclose(restricted["direction"], exact, atol=1e-8, rtol=1e-7)
    assert torch.allclose(restricted["covector"], geometry["covector"], atol=1e-8, rtol=1e-7)
    reconstructed = restricted["basis"] @ restricted["metric"] @ restricted["basis"].T
    assert torch.allclose(reconstructed, geometry["metric"] + 0.2 * torch.eye(3), atol=1e-8)


def test_random_basis_is_deterministic_orthonormal_and_nested():
    first = random_orthonormal_basis(12, 6, 20260909, torch.float64)
    second = random_orthonormal_basis(12, 6, 20260909, torch.float64)

    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(6, dtype=torch.float64), atol=1e-12)
    assert torch.equal(first[:, :3], second[:, :3])


def test_step_selection_uses_smallest_reaching_scale_and_fallback_tie_rule():
    reaching = {
        0.5: [{"concept_probability": 0.2}, {"concept_probability": 0.91}],
        0.25: [{"concept_probability": 0.2}, {"concept_probability": 0.9}],
        0.125: [{"concept_probability": 0.2}, {"concept_probability": 0.8}],
    }
    fallback = {
        0.5: [{"concept_probability": 0.2}, {"concept_probability": 0.8}],
        0.25: [{"concept_probability": 0.2}, {"concept_probability": 0.8}],
        0.125: [{"concept_probability": 0.2}, {"concept_probability": 0.7}],
    }

    assert select_step_size(reaching) == 0.25
    assert select_step_size(fallback) == 0.25


def test_target_localization_bisects_first_upward_crossing():
    states = [torch.tensor(value, dtype=torch.float64) for value in (0.1, 0.4, 0.8, 1.0)]
    trajectory = [{"concept_probability": float(state)} for state in states]

    rows = localize_targets(
        lambda state: {
            "concept_probability": float(state),
            "off_target_kl": float(state.square()),
        },
        trajectory,
        states,
        (0.3, 0.5, 0.9),
        24,
    )

    assert [row["target"] for row in rows] == [0.3, 0.5, 0.9]
    assert all(abs(row["concept_probability"] - row["target"]) < 1e-7 for row in rows)


def test_trajectory_stops_after_first_requested_probability():
    trajectory, states = run_trajectory(
        torch.tensor(0.0),
        0.25,
        lambda _state: {
            "covector": torch.tensor([1.0]),
            "direction": torch.tensor([1.0]),
        },
        lambda state: {"concept_probability": float(state)},
        10,
        1.0,
        0.6,
    )

    assert len(trajectory) == 4
    assert len(states) == 4
    assert trajectory[-1]["concept_probability"] >= 0.6


def test_case_registry_uses_only_frozen_development_rows():
    data = {
        "contexts": {
            concept: [
                {
                    "context_sha256": f"{concept}-{index:02d}",
                    "stage": "development" if index < 12 else "validation",
                    "token_ids": [index],
                }
                for index in range(100)
            ]
            for concept in ("third", "ing", "past")
        }
    }

    cases = development_cases(data)

    assert len(cases) == 144
    assert {case["stage"] for case in cases} == {"development"}
