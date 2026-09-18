import torch
from torch import nn

from projection_transport_steering.torch_runtime import (
    AdditiveAction,
    NormMatchedDirectionAction,
    ProjectedActionValueObserver,
    ThresholdedActionValueObserver,
    TorchLayerAction,
    TorchMaskedActionController,
    TorchSameForwardController,
)


class ToyBlock(nn.Module):
    def forward(self, hidden):
        return hidden


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(), ToyBlock(), ToyBlock()])


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class FixedObserver(nn.Module):
    def forward(self, hidden):
        return torch.tensor([[1.0, 0.0]], device=hidden.device).expand(hidden.shape[0], -1)


def test_torch_controller_selects_before_action_and_reuses_decode():
    model = ToyModel()
    actions = torch.tensor([[0.0, 0.0], [2.0, -1.0]])
    controller = TorchSameForwardController(model, 0, 2, FixedObserver(), actions)
    controller.start_trajectory()

    with controller.installed():
        prefill = model(torch.zeros((1, 3, 2)))
        decode = model(torch.zeros((1, 1, 2)))

    assert torch.equal(prefill, actions[1].expand(1, 3, 2))
    assert torch.equal(decode, actions[1].expand(1, 1, 2))
    assert controller.selected_actions.tolist() == [1]
    assert controller.trajectory_count == 1
    assert controller.forward_count == 2


def test_torch_controller_removes_hooks_after_context():
    model = ToyModel()
    actions = torch.tensor([[0.0, 0.0], [2.0, -1.0]])
    controller = TorchSameForwardController(model, 0, 2, FixedObserver(), actions)
    controller.start_trajectory()

    with controller.installed():
        model(torch.zeros((1, 1, 2)))
    output = model(torch.zeros((1, 1, 2)))

    assert torch.equal(output, torch.zeros((1, 1, 2)))


def test_torch_controller_requires_ordered_sites():
    model = ToyModel()
    actions = torch.zeros((2, 2))

    try:
        TorchSameForwardController(model, 2, 1, FixedObserver(), actions)
    except ValueError as error:
        assert "precede" in str(error)
    else:
        raise AssertionError("unordered sites were accepted")


def test_torch_layer_action_applies_module_and_removes_hook():
    model = ToyModel()
    vector = torch.tensor([2.0, -1.0])
    action = TorchLayerAction(model, 1, AdditiveAction(vector))

    with action.installed():
        steered = model(torch.zeros((1, 2, 2)))
    unsteered = model(torch.zeros((1, 2, 2)))

    assert torch.equal(steered, vector.expand(1, 2, 2))
    assert torch.equal(unsteered, torch.zeros((1, 2, 2)))


def test_norm_matched_direction_action_preserves_reference_update_norm():
    reference = nn.Linear(2, 2, bias=False)
    reference.weight.data = 2.0 * torch.eye(2)
    action = NormMatchedDirectionAction(reference, torch.tensor([0.0, 2.0]))
    hidden = torch.tensor([[[3.0, 4.0]]])

    reference_delta = reference(hidden) - hidden
    null_delta = action(hidden) - hidden

    assert torch.allclose(null_delta.norm(dim=-1), reference_delta.norm(dim=-1))
    assert null_delta[0, 0, 0] == 0.0


class PromptObserver(nn.Module):
    def forward(self, hidden):
        return torch.stack([torch.zeros_like(hidden[:, 0]), -hidden[:, 0]], dim=1)


def test_masked_controller_selects_from_prompt_and_changes_only_action_span():
    model = ToyModel()
    controller = TorchMaskedActionController(
        model,
        0,
        2,
        PromptObserver(),
        [nn.Identity(), AdditiveAction(torch.tensor([10.0, 0.0]))],
    )
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [-1.0, 0.0], [6.0, 0.0]],
        ]
    )
    action_mask = torch.tensor([[False, True, True], [True, False, True]])
    controller.start_trajectory(torch.tensor([0, 1]), action_mask)

    with controller.installed():
        output = model(hidden)

    expected = hidden.clone()
    expected[0, 1:, 0] += 10.0
    assert torch.equal(output, expected)
    assert controller.selected_actions.tolist() == [1, 0]


def test_masked_controller_removes_hooks():
    model = ToyModel()
    controller = TorchMaskedActionController(
        model,
        0,
        2,
        PromptObserver(),
        [nn.Identity(), AdditiveAction(torch.tensor([10.0, 0.0]))],
    )
    hidden = torch.ones((1, 2, 2))
    controller.start_trajectory(torch.tensor([0]), torch.ones((1, 2), dtype=torch.bool))

    with controller.installed():
        model(hidden)
    output = model(hidden)

    assert torch.equal(output, hidden)


def test_masked_controller_requires_observer_before_action():
    model = ToyModel()

    with torch.no_grad():
        try:
            TorchMaskedActionController(
                model,
                2,
                1,
                PromptObserver(),
                [nn.Identity(), nn.Identity()],
            )
        except ValueError as error:
            assert "precede" in str(error)
        else:
            raise AssertionError("unordered masked controller sites were accepted")


def test_projected_action_value_observer_reproduces_exported_ridge():
    observer = ProjectedActionValueObserver(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([[1.0], [-1.0]]),
        torch.tensor([0.0, 0.0]),
    )

    losses = observer(torch.tensor([[2.0, 5.0], [-2.0, 5.0]]))

    assert torch.equal(
        losses,
        torch.tensor([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]]),
    )


def test_projected_action_value_observer_restores_nonzero_reference_index():
    observer = ProjectedActionValueObserver(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([[1.0], [-1.0]]),
        torch.tensor([0.0, 0.0]),
        reference_action=1,
    )

    losses = observer(torch.tensor([[2.0, 5.0]]))

    assert torch.equal(losses, torch.tensor([[2.0, 0.0, -2.0]]))


class MatrixObserver(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses

    def forward(self, hidden):
        return self.losses.to(hidden.device)


def test_thresholded_observer_uses_reference_below_gain_threshold():
    observer = ThresholdedActionValueObserver(
        MatrixObserver(torch.tensor([[0.0, 0.1, 0.05], [0.0, 0.3, 0.4]])),
        reference_action=1,
        threshold=0.2,
    )

    losses = observer(torch.zeros((2, 4)))

    assert torch.argmin(losses, dim=1).tolist() == [1, 0]


def test_thresholded_observer_fallback_applies_reference_action_on_mask():
    model = ToyModel()
    observer = ThresholdedActionValueObserver(
        MatrixObserver(torch.tensor([[0.0, 1.0]])),
        reference_action=1,
        threshold=float("inf"),
    )
    controller = TorchMaskedActionController(
        model,
        0,
        2,
        observer,
        [AdditiveAction(torch.tensor([1.0, 0.0])), AdditiveAction(torch.tensor([2.0, 0.0]))],
    )
    hidden = torch.zeros((1, 3, 2))
    action_mask = torch.tensor([[False, True, True]])
    controller.start_trajectory(torch.tensor([0]), action_mask)

    with controller.installed():
        output = model(hidden)

    assert controller.selected_actions.tolist() == [1]
    assert torch.equal(output[0, 0], hidden[0, 0])
    assert torch.equal(output[0, 1:, 0], torch.tensor([2.0, 2.0]))
