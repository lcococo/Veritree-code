from bisect import bisect_right
from typing import Any, Sequence


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        return -1
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            return start
    return -1


def build_token_weights(
    tokenizer: Any,
    response: str,
    labels: Sequence[int],
    hotlines: Sequence[int],
    hot_weight: float,
    ignore_index: int = -100,
) -> list[float]:
    if hot_weight < 1:
        raise ValueError("hot_weight must be at least 1")
    if any(not isinstance(line, int) or isinstance(line, bool) or line < 0 for line in hotlines):
        raise ValueError("hotlines must contain nonnegative integers")
    weights = [1.0] * len(labels)
    if not hotlines:
        return weights
    target_positions = [index for index, token in enumerate(labels) if token != ignore_index]
    target_ids = [labels[index] for index in target_positions]
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    content_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(content_ids) != len(offsets):
        raise ValueError("token ids and offset mappings must have the same length")
    base = find_subsequence(target_ids, content_ids)
    if base < 0:
        return weights
    line_starts = [0]
    for index, char in enumerate(response):
        if char == "\n" and index + 1 < len(response):
            line_starts.append(index + 1)
    selected = set(hotlines)
    for relative_index, offset in enumerate(offsets):
        start, end = offset
        if start == end:
            continue
        line = bisect_right(line_starts, start) - 1
        if line in selected:
            weights[target_positions[base + relative_index]] = float(hot_weight)
    return weights
