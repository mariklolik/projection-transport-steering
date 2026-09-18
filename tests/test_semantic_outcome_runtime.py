import torch

from projection_transport_steering.semantic_outcome_runtime import (
    candidate_score_mask,
    encode_candidates,
    encode_prompt_candidates,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize
        assert add_generation_prompt
        return {"prompt": [1, 2, 3], "other": [9]}[messages[0]["content"]]

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"short": [4, 5], "long": [6, 7, 8]}[text]


class BatchEncodingFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {
            "attention_mask": [1, 1, 1],
            "input_ids": super().apply_chat_template(
                messages, tokenize, add_generation_prompt
            ),
        }


def test_encode_candidates_preserves_prefix_and_candidate_positions():
    batch, lengths = encode_candidates(
        FakeTokenizer(),
        "prompt",
        ["short", "long"],
        torch.device("cpu"),
    )

    assert lengths == [2, 3]
    assert batch["prefix_lengths"].tolist() == [3, 3]
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5, 0], [1, 2, 3, 6, 7, 8]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
    assert batch["candidate_ids"].tolist() == [[4, 5, 0], [6, 7, 8]]
    assert batch["candidate_mask"].tolist() == [[True, True, False], [True, True, True]]


def test_encode_candidates_accepts_transformers_batch_encoding():
    batch, _ = encode_candidates(
        BatchEncodingFakeTokenizer(),
        "prompt",
        ["short"],
        torch.device("cpu"),
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]


def test_encode_prompt_candidates_aligns_distinct_chat_prefixes():
    batch, lengths = encode_prompt_candidates(
        FakeTokenizer(),
        ["prompt", "other"],
        ["short", "long"],
        torch.device("cpu"),
    )

    assert lengths == [2, 3]
    assert batch["prefix_lengths"].tolist() == [3, 1]
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5], [9, 6, 7, 8, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]


def test_candidate_score_mask_selects_only_each_last_token():
    mask = torch.tensor([[True, True, False], [True, True, True]])

    assert torch.equal(candidate_score_mask(mask, False), mask)
    assert torch.equal(
        candidate_score_mask(mask, True),
        torch.tensor([[False, True, False], [False, False, True]]),
    )
