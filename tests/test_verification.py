import unittest

from core.petri_compile import activities_of, net_labels, spec_to_net
from core.powl_json import parse_spec, parse_spec_tolerant, spec_to_canon, spec_to_text
from core.verification import assert_no_dup_labels, hot_lines_of, matrix, rel_score, verify


class VerificationTests(unittest.TestCase):
    def test_strict_and_tolerant_parsing(self):
        spec = {"op": "seq", "children": ["A", "B"]}
        text = spec_to_text(spec)
        self.assertEqual(parse_spec(text), spec)
        parsed, method = parse_spec_tolerant(text + ",")
        self.assertEqual(parsed, spec)
        self.assertEqual(method, "strip_junk")

    def test_order_matrix_distinguishes_sequence_direction(self):
        forward = spec_to_canon({"op": "seq", "children": ["A", "B"]})
        reverse = spec_to_canon({"op": "seq", "children": ["B", "A"]})
        self.assertEqual(matrix(forward)[("A", "B")], ("order", 1))
        self.assertEqual(matrix(reverse)[("A", "B")], ("order", 2))
        self.assertEqual(rel_score(reverse, forward), 0.0)

    def test_unordered_operators_ignore_child_order(self):
        left = spec_to_canon({"op": "par", "children": ["A", "B", "C"]})
        right = spec_to_canon({"op": "par", "children": ["C", "A", "B"]})
        self.assertEqual(matrix(left), matrix(right))
        self.assertEqual(rel_score(right, left), 1.0)

    def test_missing_activity_reduces_reference_centered_score(self):
        reference = spec_to_canon({"op": "seq", "children": ["A", "B", "C"]})
        student = spec_to_canon({"op": "seq", "children": ["A", "B"]})
        self.assertEqual(rel_score(student, reference), 1 / 3)

    def test_duplicate_labels_are_rejected(self):
        tree = spec_to_canon({"op": "seq", "children": ["A", "A"]})
        with self.assertRaises(ValueError):
            assert_no_dup_labels(tree)

    def test_relation_and_missing_content_hotlines(self):
        reference = spec_to_canon({"op": "seq", "children": ["A", {"op": "xor", "children": ["B", "C"]}]})
        student = spec_to_canon({"op": "par", "children": ["A", "B"]})
        hotlines = hot_lines_of(student, reference)
        self.assertIn(0, hotlines)
        self.assertIn(2, hotlines)

    def test_gate_and_exact_result(self):
        reference = spec_to_canon({"op": "seq", "children": ["A", "B"]})
        result = verify(reference, reference, 0.8)
        self.assertEqual(result["rel"], 1.0)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["exact"])
        self.assertEqual(result["hotlines"], [])

    def test_petri_compilation_preserves_visible_labels(self):
        spec = {"op": "seq", "children": ["A", {"op": "xor", "children": ["B", "tau"]}]}
        net, _, _ = spec_to_net(spec)
        self.assertEqual(net_labels(net), activities_of(spec))


if __name__ == "__main__":
    unittest.main()
