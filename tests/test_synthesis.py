import unittest

from self_iter.synthesis import Reference, SynthesisConfig, build_spotlight_example, coverage_missing, synthesize


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, prompt, temperature, max_tokens):
        self.calls.append((prompt, temperature, max_tokens))
        return next(self.outputs)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.reference = Reference.from_spec(
            "r1",
            "A happens before either B or C.",
            ["A", "B", "C"],
            {"op": "seq", "children": ["A", {"op": "xor", "children": ["B", "C"]}]},
        )

    def test_coverage_uses_exact_activity_substrings(self):
        self.assertEqual(coverage_missing("A then B", ["A", "B", "C"]), ["C"])

    def test_reference_rejects_activity_mismatch(self):
        with self.assertRaises(ValueError):
            Reference.from_spec("x", "template", ["A"], {"op": "seq", "children": ["A", "B"]})

    def test_coverage_failure_uses_next_rewrite(self):
        generator = SequenceGenerator(
            [
                "A then B",
                "A is followed by a choice between B and C",
                '```json\n{"op":"seq","children":["A",{"op":"xor","children":["B","C"]}]}\n```',
            ]
        )
        result = synthesize(self.reference, generator)
        self.assertIsNotNone(result)
        self.assertEqual(result.rewrite_attempt, 2)
        self.assertEqual(result.candidate_attempt, 1)

    def test_parse_and_rel_failures_consume_candidate_budget(self):
        generator = SequenceGenerator(
            [
                "A is followed by a choice between B and C",
                "not json",
                '{"op":"par","children":["A","B","C"]}',
                '{"op":"seq","children":["A",{"op":"xor","children":["B","C"]}]}',
            ]
        )
        config = SynthesisConfig(candidate_budget=3, gate_threshold=0.8)
        result = synthesize(self.reference, generator, config)
        self.assertIsNotNone(result)
        self.assertEqual(result.candidate_attempt, 3)
        self.assertEqual(result.rel, 1.0)

    def test_spotlight_example_uses_reference_target(self):
        generator = SequenceGenerator(
            [
                "A is followed by a choice between B and C",
                '{"op":"seq","children":["A",{"op":"par","children":["B","C"]}]}',
            ]
        )
        result = synthesize(self.reference, generator, SynthesisConfig(gate_threshold=0.6))
        self.assertIsNotNone(result)
        example = build_spotlight_example(self.reference, result)
        self.assertIn('"op": "xor"', example["output"])
        self.assertNotIn('"op": "par"', example["output"])
        self.assertEqual(example["hotlines"], [2])

    def test_exhausted_budget_discards_instance(self):
        generator = SequenceGenerator(
            [
                "A then B",
                "A then B",
            ]
        )
        result = synthesize(self.reference, generator, SynthesisConfig(rewrite_budget=2, candidate_budget=1))
        self.assertIsNone(result)
        self.assertEqual(len(generator.calls), 2)


if __name__ == "__main__":
    unittest.main()
