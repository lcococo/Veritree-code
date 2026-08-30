import difflib
import itertools
import json
import re
from typing import Any, Iterable, Sequence

from .powl_json import canon_to_spec, spec_to_text

TOKEN_RE = re.compile(r'"op"|"children"|"edges"|"seq"|"par"|"xor"|"loop"|"po"|"tau"|[{}:,\[\]]|"[^"]*"')


def leaf_instances(tree: dict[str, Any], path: tuple[int, ...] = ()) -> list[tuple[str, tuple[int, ...]]]:
    if tree["t"] == "act":
        return [(tree["label"], path)]
    if tree["t"] == "silent":
        return []
    result = []
    for index, child in enumerate(tree["children"]):
        result.extend(leaf_instances(child, path + (index,)))
    return result


def node_at(tree: dict[str, Any], path: Sequence[int]) -> dict[str, Any]:
    node = tree
    for index in path:
        node = node["children"][index]
    return node


def assert_no_dup_labels(tree: dict[str, Any]) -> None:
    labels = [label for label, _ in leaf_instances(tree)]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise ValueError(f"duplicate activity labels: {repeated}")


def _interleave(sequences: Sequence[Sequence[str]]) -> list[list[str]]:
    if not sequences:
        return [[]]
    if len(sequences) == 1:
        return [list(sequences[0])]
    head = sequences[0]
    result = set()
    for remainder in _interleave(sequences[1:]):
        width = len(remainder) + len(head)
        for positions in itertools.combinations(range(width), len(head)):
            selected = set(positions)
            merged = []
            head_index = 0
            remainder_index = 0
            for index in range(width):
                if index in selected:
                    merged.append(head[head_index])
                    head_index += 1
                else:
                    merged.append(remainder[remainder_index])
                    remainder_index += 1
            result.add(tuple(merged))
    return [list(value) for value in result]


def traces(node: dict[str, Any], cap: int = 5000) -> list[list[str]]:
    if node["t"] == "act":
        return [[node["label"]]]
    if node["t"] == "silent":
        return [[]]
    children = [traces(child, cap) for child in node["children"]]
    result = set()
    if node["t"] == "seq":
        for combination in itertools.product(*children):
            result.add(tuple(item for sequence in combination for item in sequence))
            if len(result) >= cap:
                break
    elif node["t"] == "xor":
        for child in children:
            for value in child:
                result.add(tuple(value))
    elif node["t"] == "par":
        for combination in itertools.product(*children):
            for value in _interleave(combination):
                result.add(tuple(value))
                if len(result) >= cap:
                    break
            if len(result) >= cap:
                break
    elif node["t"] == "loop":
        body, redo = children
        for body_trace in body:
            result.add(tuple(body_trace))
            for redo_trace in redo:
                for repeated_body in body:
                    if len(body_trace) + len(redo_trace) + len(repeated_body) <= 15:
                        result.add(tuple(body_trace + redo_trace + repeated_body))
    elif node["t"] == "po":
        result.update(_po_traces(node, children, cap))
    return [list(value) for value in result]


def _po_traces(node: dict[str, Any], child_traces: Sequence[Sequence[Sequence[str]]], cap: int) -> set[tuple[str, ...]]:
    result = set()
    edges = node["edges"]
    if edges and isinstance(edges[0][0], dict):
        positions = {id(child): index for index, child in enumerate(node["children"])}
        edges = [(positions[id(left)], positions[id(right)]) for left, right in edges]
    for combination in itertools.product(*child_traces):
        for order in _topological_orders(len(node["children"]), edges):
            result.add(tuple(item for index in order for item in combination[index]))
            if len(result) >= cap:
                return result
    return result


def _topological_orders(size: int, edges: Iterable[tuple[int, int]]) -> list[tuple[int, ...]]:
    edge_set = set(edges)
    return [
        order
        for order in itertools.permutations(range(size))
        if all(order.index(left) < order.index(right) for left, right in edge_set)
    ]


def _trace_relation(left: str, right: str, trace_set: Sequence[Sequence[str]]) -> tuple[Any, ...]:
    both = [trace for trace in trace_set if left in trace and right in trace]
    if not both:
        return ("xor",)
    left_before = any(trace.index(left) < len(trace) - 1 - list(reversed(trace)).index(right) for trace in both)
    right_before = any(trace.index(right) < len(trace) - 1 - list(reversed(trace)).index(left) for trace in both)
    if left_before and right_before:
        return ("par",)
    return ("order", 1 if left_before else 2)


def relation(left_path: Sequence[int], right_path: Sequence[int], tree: dict[str, Any]) -> tuple[Any, ...]:
    common = 0
    while common < min(len(left_path), len(right_path)) and left_path[common] == right_path[common]:
        common += 1
    lca = node_at(tree, left_path[:common])
    loop = None
    for depth in range(common, -1, -1):
        ancestor = node_at(tree, left_path[:depth])
        if ancestor["t"] == "loop":
            loop = ancestor
            break
    if loop is not None:
        left = node_at(tree, left_path)["label"]
        right = node_at(tree, right_path)["label"]
        return _trace_relation(left, right, traces(loop))
    if lca["t"] == "seq":
        return ("order", 1 if left_path[common] < right_path[common] else 2)
    if lca["t"] == "par":
        return ("par",)
    if lca["t"] == "xor":
        return ("xor",)
    if lca["t"] == "po":
        left = node_at(tree, left_path)["label"]
        right = node_at(tree, right_path)["label"]
        return _trace_relation(left, right, traces(lca))
    raise ValueError("distinct activities must have a control-flow ancestor")


def matrix(tree: dict[str, Any]) -> dict[tuple[str, str], tuple[Any, ...]]:
    assert_no_dup_labels(tree)
    paths = {label: path for label, path in leaf_instances(tree)}
    labels = sorted(paths)
    return {
        (labels[left], labels[right]): relation(paths[labels[left]], paths[labels[right]], tree)
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    }


def discrepancies(student: dict[str, Any], reference: dict[str, Any]) -> set[tuple[str, str]]:
    reference_matrix = matrix(reference)
    student_matrix = matrix(student)
    return {pair for pair, value in reference_matrix.items() if student_matrix.get(pair) != value}


def rel_score(student: dict[str, Any], reference: dict[str, Any]) -> float:
    reference_matrix = matrix(reference)
    if not reference_matrix:
        return 1.0
    student_matrix = matrix(student)
    matches = sum(student_matrix.get(pair) == value for pair, value in reference_matrix.items())
    return matches / len(reference_matrix)


def equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["t"] != right["t"]:
        return False
    if left["t"] == "act":
        return left["label"] == right["label"]
    if left["t"] == "silent":
        return True
    if left["t"] == "po":
        left_spec = canon_to_spec(left)
        right_spec = canon_to_spec(right)
        if {tuple(edge) for edge in left_spec["edges"]} != {tuple(edge) for edge in right_spec["edges"]}:
            return False
    left_children = left["children"]
    right_children = right["children"]
    if len(left_children) != len(right_children):
        return False
    if left["t"] in ("seq", "loop", "po"):
        return all(equal(a, b) for a, b in zip(left_children, right_children))
    key = lambda node: json.dumps(canon_to_spec(node), sort_keys=True)
    return all(equal(a, b) for a, b in zip(sorted(left_children, key=key), sorted(right_children, key=key)))


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _node_lines(spec: Any) -> dict[tuple[int, ...], int]:
    lines = {}

    def visit(node: Any, path: tuple[int, ...], line: int) -> int:
        if not isinstance(node, dict):
            return line
        lines[path] = line
        next_line = line + 1
        for index, child in enumerate(node["children"]):
            next_line = visit(child, path + (index,), next_line) + 1
        return next_line - 1

    visit(spec, (), 0)
    return lines


def _token_lines(text: str) -> dict[int, int]:
    result = {}
    token_index = 0
    for line_index, line in enumerate(text.split("\n")):
        for _ in _tokens(line):
            result[token_index] = line_index
            token_index += 1
    return result


def hot_lines_of(student: dict[str, Any], reference: dict[str, Any]) -> list[int]:
    reference_spec = canon_to_spec(reference)
    reference_text = spec_to_text(reference_spec)
    paths = {label: path for label, path in leaf_instances(reference)}
    node_lines = _node_lines(reference_spec)
    hot = set()
    for left, right in discrepancies(student, reference):
        left_path = paths[left]
        right_path = paths[right]
        common = 0
        while common < min(len(left_path), len(right_path)) and left_path[common] == right_path[common]:
            common += 1
        hot.add(node_lines[left_path[:common]])
    student_text = spec_to_text(canon_to_spec(student))
    matcher = difflib.SequenceMatcher(None, _tokens(student_text), _tokens(reference_text), autojunk=False)
    token_lines = _token_lines(reference_text)
    for tag, _, _, reference_start, reference_end in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            for token_index in range(reference_start, reference_end):
                if token_index in token_lines:
                    hot.add(token_lines[token_index])
    return sorted(hot)


def verify(student: dict[str, Any], reference: dict[str, Any], threshold: float) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    score = rel_score(student, reference)
    return {
        "rel": score,
        "accepted": score >= threshold,
        "exact": equal(student, reference),
        "discrepancies": sorted(discrepancies(student, reference)),
        "hotlines": hot_lines_of(student, reference),
    }
