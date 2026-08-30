import json
import re
from collections import Counter
from typing import Any, Optional

OPS = ("seq", "par", "xor", "loop")
OPS_ALL = OPS + ("po",)
TAU = "tau"
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def validate_spec(node: Any) -> None:
    if isinstance(node, str):
        if not node:
            raise ValueError("activity labels must not be empty")
        return
    if not isinstance(node, dict):
        raise ValueError("nodes must be strings or objects")
    extra = set(node) - {"op", "children", "edges"}
    if extra:
        raise ValueError(f"unknown keys: {sorted(extra)}")
    operator = node.get("op")
    if operator not in OPS_ALL:
        raise ValueError(f"unknown operator: {operator}")
    children = node.get("children")
    if not isinstance(children, list) or not children:
        raise ValueError(f"{operator} children must be a nonempty list")
    if operator == "loop" and len(children) != 2:
        raise ValueError("loop must have exactly two children")
    if operator == "po":
        edges = node.get("edges")
        if not isinstance(edges, list):
            raise ValueError("po edges must be a list")
        for edge in edges:
            valid = (
                isinstance(edge, list)
                and len(edge) == 2
                and all(isinstance(index, int) and not isinstance(index, bool) for index in edge)
                and 0 <= edge[0] < len(children)
                and 0 <= edge[1] < len(children)
            )
            if not valid:
                raise ValueError(f"invalid po edge: {edge}")
    for child in children:
        validate_spec(child)


def parse_spec(text: str) -> Any:
    match = FENCE.search(text)
    raw = match.group(1) if match else text.strip()
    try:
        value = json.loads(raw)
    except Exception as error:
        raise ValueError(f"invalid JSON: {error}") from error
    validate_spec(value)
    return value


def parse_spec_tolerant(text: str) -> tuple[Any, str]:
    try:
        return parse_spec(text), "strict"
    except ValueError:
        pass
    body = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(body)
        if not body[end:].strip(" \t\r\n,"):
            validate_spec(value)
            return value, "strip_junk"
    except Exception:
        pass
    values = _decode_multiple(body, decoder)
    if values:
        value = values[0] if len(values) == 1 else {"op": "seq", "children": values}
        validate_spec(value)
        return value, "multiroot_seq" if len(values) > 1 else "strip_junk"
    fixed = _close_json(body)
    if fixed is not None:
        try:
            value = json.loads(fixed)
            validate_spec(value)
            return value, "autoclose"
        except Exception:
            values = _decode_multiple(fixed, decoder)
            if values:
                value = values[0] if len(values) == 1 else {"op": "seq", "children": values}
                validate_spec(value)
                return value, "autoclose_multiroot"
    raise ValueError("unable to parse process tree")


def _decode_multiple(text: str, decoder: json.JSONDecoder) -> list[Any]:
    values = []
    index = 0
    try:
        while index < len(text):
            while index < len(text) and text[index] in " \t\r\n,":
                index += 1
            if index >= len(text):
                break
            value, index = decoder.raw_decode(text, index)
            values.append(value)
        return values
    except Exception:
        return []


def _close_json(text: str) -> Optional[str]:
    openings = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            openings.append(char)
        elif char in "}]" and openings:
            openings.pop()
    if not openings or in_string:
        return None
    closing = "".join("}" if char == "{" else "]" for char in reversed(openings))
    return text.rstrip().rstrip(",") + closing


def spec_to_text(spec: Any) -> str:
    validate_spec(spec)
    return "\n".join(_render(spec, 0))


def _render(spec: Any, indentation: int) -> list[str]:
    prefix = "  " * indentation
    if isinstance(spec, str):
        return [prefix + json.dumps(spec, ensure_ascii=False)]
    if spec["op"] == "po":
        first = prefix + '{"op": "po", "edges": ' + json.dumps(spec["edges"]) + ', "children": ['
    else:
        first = prefix + '{"op": ' + json.dumps(spec["op"]) + ', "children": ['
    lines = [first]
    for index, child in enumerate(spec["children"]):
        child_lines = _render(child, indentation + 1)
        if index < len(spec["children"]) - 1:
            child_lines[-1] += ","
        lines.extend(child_lines)
    lines[-1] += "]}"
    return lines


def spec_to_canon(spec: Any) -> dict[str, Any]:
    validate_spec(spec)
    if isinstance(spec, str):
        return {"t": "silent"} if spec == TAU else {"t": "act", "label": spec}
    node = {"t": spec["op"], "children": [spec_to_canon(child) for child in spec["children"]]}
    if spec["op"] == "po":
        node["edges"] = [tuple(edge) for edge in spec["edges"]]
    return node


def canon_to_spec(node: dict[str, Any]) -> Any:
    if node["t"] == "act":
        return node["label"]
    if node["t"] == "silent":
        return TAU
    value = {"op": node["t"], "children": [canon_to_spec(child) for child in node["children"]]}
    if node["t"] == "po":
        if node["edges"] and isinstance(node["edges"][0][0], dict):
            index = {id(child): position for position, child in enumerate(node["children"])}
            value["edges"] = [[index[id(left)], index[id(right)]] for left, right in node["edges"]]
        else:
            value["edges"] = [list(edge) for edge in node["edges"]]
    return value


def spec_to_powl(spec: Any):
    from pm4py.objects.powl.obj import OperatorPOWL, SilentTransition, StrictPartialOrder, Transition
    from pm4py.objects.process_tree.obj import Operator

    validate_spec(spec)
    if isinstance(spec, str):
        return SilentTransition() if spec == TAU else Transition(label=spec)
    children = [spec_to_powl(child) for child in spec["children"]]
    if spec["op"] == "xor":
        return OperatorPOWL(Operator.XOR, children)
    if spec["op"] == "loop":
        return OperatorPOWL(Operator.LOOP, children)
    order = StrictPartialOrder(children)
    if spec["op"] == "seq":
        for index in range(len(children) - 1):
            order.order.add_edge(children[index], children[index + 1])
    elif spec["op"] == "po":
        for left, right in spec["edges"]:
            order.order.add_edge(children[left], children[right])
    return order


def _transitive_reduction(size: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    adjacency = [[] for _ in range(size)]
    for left, right in edges:
        adjacency[left].append(right)

    def reachable(source: int, target: int, skipped: tuple[int, int]) -> bool:
        stack = [source]
        seen = {source}
        while stack:
            current = stack.pop()
            for successor in adjacency[current]:
                if (current, successor) == skipped:
                    continue
                if successor == target:
                    return True
                if successor not in seen:
                    seen.add(successor)
                    stack.append(successor)
        return False

    return [edge for edge in edges if not reachable(edge[0], edge[1], edge)]


def canon(value: Any) -> dict[str, Any]:
    from pm4py.objects.powl.obj import OperatorPOWL, SilentTransition, StrictPartialOrder, Transition
    from pm4py.objects.process_tree.obj import Operator

    if isinstance(value, SilentTransition):
        return {"t": "silent"}
    if isinstance(value, Transition):
        return {"t": "act", "label": value.label}
    if isinstance(value, OperatorPOWL):
        operator = "xor" if value.operator == Operator.XOR else "loop"
        return {"t": operator, "children": [canon(child) for child in value.children]}
    if isinstance(value, StrictPartialOrder):
        nodes = list(value.children)
        positions = {id(node): index for index, node in enumerate(nodes)}
        raw = value.order.edges
        if raw and isinstance(raw[0][0], bool):
            pairs = [
                (value.order._map_id_to_node[left], value.order._map_id_to_node[right])
                for left, row in enumerate(raw)
                for right, present in enumerate(row)
                if present
            ]
        else:
            pairs = raw
        edges = _transitive_reduction(len(nodes), [(positions[id(left)], positions[id(right)]) for left, right in pairs])
        children = [canon(node) for node in nodes]
        if not edges:
            return {"t": "par", "children": children}
        indegree = Counter(right for _, right in edges)
        outdegree = Counter(left for left, _ in edges)
        chain = len(edges) == len(nodes) - 1 and all(indegree[i] <= 1 for i in range(len(nodes))) and all(
            outdegree[i] <= 1 for i in range(len(nodes))
        )
        if chain:
            current = next(index for index in range(len(nodes)) if indegree[index] == 0)
            ordering = [current]
            while len(ordering) < len(nodes):
                successors = [right for left, right in edges if left == current]
                if not successors:
                    break
                current = successors[0]
                ordering.append(current)
            if len(ordering) == len(nodes):
                return {"t": "seq", "children": [children[index] for index in ordering]}
        return {
            "t": "po",
            "children": children,
            "edges": [(children[left], children[right]) for left, right in edges],
        }
    raise TypeError(f"unsupported POWL node: {type(value)}")
