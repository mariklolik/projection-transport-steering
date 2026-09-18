import torch
from torch import nn

from projection_transport_steering.outcome_score_runtime import (
    TorchLayerTrace,
    format_aime_prompt,
    generate_rollout,
    rollout_seed,
)


class ToyLayer(nn.Module):
    def forward(self, hidden):
        return hidden + 1.0


class ToyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer()])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBody()

    def forward(self, hidden):
        return self.model(hidden)


class ToyGenerator(ToyModel):
    def generate(self, input_ids, **kwargs):
        self(torch.zeros((1, input_ids.shape[1], 4)))
        return torch.cat((input_ids, torch.tensor([[42]])), dim=1)


class ToyTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "user"
        self.template_kwargs = kwargs
        return torch.tensor([[10, 11]])

    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [42]
        return "reasoning\nFinal Answer: 42"


class ToyBatchEncoding(dict):
    def to(self, device):
        return ToyBatchEncoding({key: value.to(device) for key, value in self.items()})


class ToyMappingTokenizer(ToyTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return ToyBatchEncoding(
            {"attention_mask": torch.ones((1, 2)), "input_ids": torch.tensor([[10, 11]])}
        )


class ToyBoxedTokenizer(ToyTokenizer):
    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [42]
        return "reasoning\n\\boxed{42}"


def test_layer_trace_captures_last_position_without_changing_output():
    model = ToyModel()
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    trace = TorchLayerTrace(model, 0)

    trace.start()
    with trace.installed():
        actual = model(hidden)

    assert torch.equal(actual, hidden + 2.0)
    assert torch.equal(trace.finish(), (hidden + 1.0)[:, -1, :])


def test_layer_trace_requires_a_started_nonempty_trajectory():
    trace = TorchLayerTrace(ToyModel(), 0)

    try:
        trace.finish()
    except RuntimeError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty trace was accepted")


def test_rollout_seed_is_stable_and_rollout_specific():
    assert rollout_seed("question", 0) == rollout_seed("question", 0)
    assert rollout_seed("question", 0) != rollout_seed("question", 1)


def test_aime_prompt_declares_the_exact_terminal_contract():
    prompt = format_aime_prompt("What is 40 + 2?")

    assert "What is 40 + 2?" in prompt
    assert prompt.endswith("Final Answer: <integer>")


def test_benchmark_prompt_declares_the_boxed_terminal_contract():
    prompt = format_aime_prompt("What is 40 + 2?", answer_format="terminal_integer")

    assert "reasoning concisely" in prompt
    assert prompt.endswith("\\boxed{integer}")
    assert format_aime_prompt("What is 40 + 2?", answer_format="reward_integer") == prompt


def test_generate_rollout_returns_scored_row_and_finite_trace():
    model = ToyGenerator()
    trace = TorchLayerTrace(model, 0)

    row, states = generate_rollout(
        model,
        ToyTokenizer(),
        trace,
        {"answer": 42, "group_id": "q", "problem": "40 + 2"},
        rollout_index=0,
        max_new_tokens=8,
        device=torch.device("cpu"),
    )

    assert row["correct"]
    assert row["extracted_answer"] == 42
    assert row["generated_tokens"] == 1
    assert row["trace_length"] == 1
    assert states.shape == (1, 4)


def test_generate_rollout_accepts_transformers_five_batch_encoding():
    model = ToyGenerator()

    row, states = generate_rollout(
        model,
        ToyMappingTokenizer(),
        TorchLayerTrace(model, 0),
        {"answer": 42, "group_id": "q", "problem": "40 + 2"},
        rollout_index=0,
        max_new_tokens=8,
        device=torch.device("cpu"),
    )

    assert row["correct"]
    assert states.shape == (1, 4)


def test_generate_rollout_routes_nonthinking_benchmark_scoring():
    model = ToyGenerator()
    tokenizer = ToyBoxedTokenizer()

    row, states = generate_rollout(
        model,
        tokenizer,
        TorchLayerTrace(model, 0),
        {"answer": 42, "group_id": "q", "problem": "40 + 2"},
        rollout_index=0,
        max_new_tokens=8,
        device=torch.device("cpu"),
        enable_thinking=False,
        answer_format="terminal_integer",
    )

    assert row["correct"]
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert states.shape == (1, 4)


class ToyOutOfRangeTokenizer(ToyBoxedTokenizer):
    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [42]
        return "reasoning\n\\boxed{1786}"


class ToyMathVerifyTokenizer(ToyBoxedTokenizer):
    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [42]
        return "reasoning\n\\[\\boxed{42}.\\]"


def test_generate_rollout_routes_total_reward_scoring():
    model = ToyGenerator()

    row, _ = generate_rollout(
        model,
        ToyOutOfRangeTokenizer(),
        TorchLayerTrace(model, 0),
        {"answer": 504, "group_id": "q", "problem": "problem"},
        rollout_index=0,
        max_new_tokens=8,
        device=torch.device("cpu"),
        enable_thinking=False,
        answer_format="reward_integer",
    )

    assert row["parse_status"] == "pass"
    assert row["extracted_answer"] == 1786
    assert row["correct"] is False


def test_generate_rollout_routes_strict_math_verify_scoring():
    model = ToyGenerator()

    row, _ = generate_rollout(
        model,
        ToyMathVerifyTokenizer(),
        TorchLayerTrace(model, 0),
        {"answer": 42, "group_id": "q", "problem": "problem"},
        rollout_index=0,
        max_new_tokens=8,
        device=torch.device("cpu"),
        enable_thinking=False,
        answer_format="math_verify_strict",
    )

    assert row["parse_status"] == "pass"
    assert row["correct"] is True
