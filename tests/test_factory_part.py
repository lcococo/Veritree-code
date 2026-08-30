import unittest

import torch
import torch.nn.functional as functional

from factory_part.data import SpotlightExample, attach_token_weights
from factory_part.objective import spotlight_loss
from factory_part.projection import build_token_weights


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class FactoryPartTests(unittest.TestCase):
    def test_example_contract(self):
        example = SpotlightExample.from_mapping(
            {"instruction": "model", "input": "", "output": "seq\n  A", "hotlines": [1, 1, 0]}
        )
        self.assertEqual(example.hotlines, (0, 1))

    def test_hotline_projection(self):
        response = "seq\n  A\n  B"
        labels = [-100, 900] + [ord(char) for char in response] + [901]
        weights = build_token_weights(CharacterTokenizer(), response, labels, [1], 3.0)
        weighted = [labels[index] for index, weight in enumerate(weights) if weight == 3.0]
        self.assertEqual("".join(chr(token) for token in weighted), "  A\n")

    def test_projection_falls_back_to_uniform_weights(self):
        weights = build_token_weights(CharacterTokenizer(), "A", [-100, ord("B")], [0], 3.0)
        self.assertEqual(weights, [1.0, 1.0])

    def test_batch_projection(self):
        labels = torch.tensor([[-100, ord("A"), ord("\n"), ord("B")]])
        weights = attach_token_weights(CharacterTokenizer(), ["A\nB"], labels, [[1]], 5.0)
        self.assertTrue(torch.equal(weights, torch.tensor([[1.0, 1.0, 1.0, 5.0]])))

    def test_uniform_spotlight_matches_cross_entropy(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 5, 11)
        labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, -100, 5, 6, 7]])
        weights = torch.ones_like(labels, dtype=torch.float32)
        actual = spotlight_loss(logits, labels, weights, chunk_size=2)
        expected = functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_weighted_spotlight_matches_manual_objective(self):
        logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [1.0, 2.0], [2.0, 1.0]]])
        labels = torch.tensor([[-100, 0, 1, 0]])
        weights = torch.tensor([[1.0, 1.0, 4.0, 1.0]])
        losses = functional.cross_entropy(logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1), reduction="none")
        expected = (losses * torch.tensor([1.0, 4.0, 1.0])).sum() / 6.0
        actual = spotlight_loss(logits, labels, weights, chunk_size=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_ignored_positions_do_not_change_loss(self):
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        labels = torch.tensor([[-100, -100, 0]])
        first = spotlight_loss(logits, labels, torch.tensor([[1.0, 1000.0, 2.0]]))
        second = spotlight_loss(logits, labels, torch.tensor([[1.0, 1.0, 2.0]]))
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
