import torch
from transformers import GPT2Config, GPT2LMHeadModel

from projection_transport_steering.cacheback_baselines import (
    caa_direction,
    matched_linear_action,
)


def test_matched_linear_action_satisfies_the_same_local_target():
    direction = torch.tensor([1.0, 2.0])
    covector = torch.tensor([0.5, -0.1])

    action = matched_linear_action(direction, covector, 0.1)

    assert torch.allclose(covector @ action, torch.tensor(0.1))


def test_caa_direction_uses_the_declared_token_pairs_at_the_intervention_layer():
    torch.manual_seed(23)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=12,
            n_head=3,
            n_layer=2,
            n_positions=8,
            vocab_size=31,
            bos_token_id=30,
            eos_token_id=30,
        )
    ).eval()
    mapping = {
        "pairs": [
            {"base_id": 1, "target_id": 2},
            {"base_id": 3, "target_id": 4},
            {"base_id": 5, "target_id": 6},
        ]
    }

    direction = caa_direction(model, mapping, layer=0, batch_size=2)

    assert direction.shape == (12,)
    assert torch.all(torch.isfinite(direction))
    assert direction.norm() > 0
