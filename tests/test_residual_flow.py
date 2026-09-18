from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from projection_transport_steering.residual_flow import (
    fit_residual_actions,
    flas_model_logits,
    residual_deficits,
)


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeDecoder()

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        assert not use_cache
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(logits=self.model.layers[0](hidden))


class FakeFlasGenerator:
    def __init__(self):
        self.llm = FakeModel()
        self.layer = 0
        self._hook_handle = None
        self._active = False

    def _install_hook(self):
        def multiply(module, inputs, output):
            assert self._active
            assert self._n_steps == 3
            assert self._flowtimes.tolist() == [2.0, 2.0]
            if self._iv is not None:
                assert self._iv == {"scale": 1.5, "steps": [1, 2]}
            return 2.0 * output

        self._hook_handle = self.llm.model.layers[0].register_forward_hook(multiply)

    def _remove_hook(self):
        self._hook_handle.remove()
        self._hook_handle = None


def test_residual_deficits_use_frozen_quantile():
    rho, deficits = residual_deficits(np.array([0.0, 1.0, 2.0, 3.0]), 0.5)

    assert rho == 1.5
    assert np.array_equal(deficits, np.array([1.5, 0.5, 0.0, 0.0]))


def test_fit_residual_actions_preserves_vector_targets_and_cost():
    result = fit_residual_actions(
        np.eye(2),
        np.array([2.0, 1.0]),
        np.array([0.0, 1.0]),
        quantile=1.0,
        concept_id=7,
    )

    candidate = result["actions"]["residual_metric"]
    costs = [
        action @ (np.array([2.0, 1.0]) * action)
        for action in result["actions"].values()
    ]
    assert np.array_equal(result["deficits"], np.array([1.0, 0.0]))
    assert np.allclose(candidate, np.array([1.0, 0.0]))
    assert np.allclose(costs, costs[0])
    assert result["candidate_solver"]["feasible"]


def test_flas_model_logits_installs_residual_after_flow_and_cleans_state():
    generator = FakeFlasGenerator()
    batch = {
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
    }
    concept_hidden = torch.ones(1, 2, 2)
    concept_mask = torch.ones(1, 2)

    logits = flas_model_logits(
        generator,
        batch,
        concept_hidden,
        concept_mask,
        vector=torch.ones(2),
        flowtime=2.0,
        n_steps=3,
    )

    expected = 2.0 * batch["input_ids"].unsqueeze(-1).repeat(1, 1, 2) + 1.0
    assert torch.equal(logits, expected)
    assert not generator._active
    assert generator._hook_handle is None
    assert generator._sa_caches is None
    assert torch.equal(
        generator.llm(**batch, position_ids=torch.zeros_like(batch["input_ids"]), use_cache=False).logits,
        batch["input_ids"].float().unsqueeze(-1).repeat(1, 1, 2),
    )


def test_flas_model_logits_restores_step_intervention_state():
    generator = FakeFlasGenerator()
    generator._iv = None
    batch = {
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "input_ids": torch.ones(2, 2, dtype=torch.long),
    }

    flas_model_logits(
        generator,
        batch,
        torch.ones(1, 2, 2),
        torch.ones(1, 2),
        intervention={"scale": 1.5, "steps": [1, 2]},
    )

    assert generator._iv is None
