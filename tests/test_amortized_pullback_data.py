import importlib
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def data_module():
    return importlib.import_module("materialize_amortized_pullback_data")


class Tokenizer:
    def __init__(self):
        self.vocab_size = 8
        self.tokens = {
            " run": [1],
            " runs": [2],
            " split": [3, 4],
            " splits": [5],
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.tokens[text]


def test_retained_token_pairs_convert_sentencepiece_boundary_and_filter_multitokens():
    retained = data_module().retained_token_pairs(
        {"▁run": "▁runs", "▁split": "▁splits"},
        Tokenizer(),
    )

    assert retained == [{"base": " run", "base_id": 1, "target": " runs", "target_id": 2}]


def test_matching_positions_require_three_base_tokens_and_probability_threshold():
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.0, 0.0],
            [2.0, 1.0, 0.5, 2.5, 2.4],
        ]
    )

    positions = data_module().matching_positions(logits, {0, 1, 2}, threshold=0.7)

    assert positions == [0]


def test_screening_continues_through_basis_document_boundary_after_evaluation_quota():
    complete = {"third": [None] * 100, "ing": [None] * 100, "past": [None] * 100}

    assert data_module().screening_pending(3_999, complete) is True
    assert data_module().screening_pending(4_096, complete) is False
