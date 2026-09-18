import torch

from projection_transport_steering.invariantback_runner import (
    CONSTRUCTION_ROW_ORDERS,
    HELDOUT_ROW_ORDERS,
    IndexedAdditiveAction,
    candidate_mean_log_likelihood,
    encode_candidate_batch,
    render_prompt,
    render_mapping_check,
    sentinel_cases,
)


def test_row_order_families_are_disjoint_and_balanced():
    assert set(CONSTRUCTION_ROW_ORDERS).isdisjoint(HELDOUT_ROW_ORDERS)
    assert set(CONSTRUCTION_ROW_ORDERS) | set(HELDOUT_ROW_ORDERS) == {
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    }
    for position in range(3):
        assert {order[position] for order in CONSTRUCTION_ROW_ORDERS} == {0, 1, 2}
        assert {order[position] for order in HELDOUT_ROW_ORDERS} == {0, 1, 2}


def test_render_prompt_separates_identifier_mapping_and_row_order():
    prompt, candidates = render_prompt(
        "mnli",
        "Premise: P\nHypothesis: H",
        ("contradiction", "entailment", "neutral"),
        ("X", "Y", "Z"),
        (2, 0, 1),
        paraphrase=1,
    )

    assert prompt.index("Z. neutral") < prompt.index("X. contradiction")
    assert prompt.index("X. contradiction") < prompt.index("Y. entailment")
    assert candidates == {
        "contradiction": " X",
        "entailment": " Y",
        "neutral": " Z",
    }
    assert prompt.endswith("Answer:")


def test_candidate_mean_log_likelihood_uses_teacher_forced_positions():
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [4.0, 0.0, 0.0],
            ]
        ]
    )

    score = candidate_mean_log_likelihood(
        logits,
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([2]),
    )
    expected = torch.stack(
        (
            torch.log_softmax(logits[0, 1], dim=-1)[1],
            torch.log_softmax(logits[0, 2], dim=-1)[2],
        )
    ).mean()

    assert torch.allclose(score, expected)


def test_indexed_action_changes_only_declared_positions_and_is_differentiable():
    vector = torch.tensor([1.0, -2.0], requires_grad=True)
    action = IndexedAdditiveAction(vector, torch.tensor([1, 2]))
    hidden = torch.zeros(2, 4, 2)

    result = action(hidden)
    result.sum().backward()

    assert torch.equal(result[0, 1], vector)
    assert torch.equal(result[1, 2], vector)
    assert torch.count_nonzero(result) == 4
    assert torch.equal(vector.grad, torch.tensor([2.0, 2.0]))


def test_encode_candidate_batch_preserves_prefixes_and_pads_candidates():
    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens):
            return ([99] if add_special_tokens else []) + [ord(character) for character in text]

    batch = encode_candidate_batch(
        Tokenizer(),
        ["p:", "long:"],
        [" a", " bc"],
        device="cpu",
    )

    assert batch["prefix_lengths"].tolist() == [3, 6]
    assert batch["candidate_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["candidate_ids"][0, :2].tolist() == [ord(" "), ord("a")]


def test_sentinel_cases_cover_each_dataset_contrast_and_layer_once():
    cases = sentinel_cases()

    assert len(cases) == 27
    assert len({case["case_id"] for case in cases}) == 27
    assert {case["layer"] for case in cases} == {12, 18, 24}
    assert {case["dataset"] for case in cases} == {"normbank", "mnli", "sc101"}


def test_mapping_check_uses_current_semantic_assignment():
    prompt, candidates = render_mapping_check(
        "normbank",
        ("expected", "taboo", "normal"),
        ("I", "II", "III"),
        "taboo",
        1,
    )

    assert "I means expected" in prompt
    assert "II means taboo" in prompt
    assert "Return the identifier assigned to taboo." in prompt
    assert candidates["taboo"] == " II"
