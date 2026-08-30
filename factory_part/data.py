from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .projection import build_token_weights


@dataclass(frozen=True)
class SpotlightExample:
    instruction: str
    input: str
    output: str
    hotlines: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpotlightExample":
        instruction = value.get("instruction", "")
        input_text = value.get("input", "")
        output = value.get("output")
        hotlines = value.get("hotlines", ())
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        if not isinstance(input_text, str):
            raise TypeError("input must be a string")
        if not isinstance(output, str):
            raise TypeError("output must be a string")
        if not isinstance(hotlines, Sequence) or isinstance(hotlines, (str, bytes)):
            raise TypeError("hotlines must be a sequence of integers")
        normalized = tuple(sorted(set(hotlines)))
        if any(not isinstance(line, int) or isinstance(line, bool) or line < 0 for line in normalized):
            raise ValueError("hotlines must contain nonnegative integers")
        return cls(instruction, input_text, output, normalized)


@dataclass(frozen=True)
class SpotlightBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    token_weights: torch.Tensor


def attach_token_weights(
    tokenizer: Any,
    responses: Sequence[str],
    labels: torch.Tensor,
    hotlines: Sequence[Sequence[int]],
    hot_weight: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if len(responses) != labels.shape[0] or len(hotlines) != labels.shape[0]:
        raise ValueError("responses, labels, and hotlines must have the same batch size")
    rows = [
        build_token_weights(
            tokenizer,
            response,
            labels[index].tolist(),
            hotlines[index],
            hot_weight,
            ignore_index,
        )
        for index, response in enumerate(responses)
    ]
    return torch.tensor(rows, dtype=torch.float32, device=labels.device)
