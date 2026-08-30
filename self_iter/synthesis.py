from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

from core.powl_json import canon_to_spec, parse_spec_tolerant, spec_to_canon, spec_to_text
from core.verification import assert_no_dup_labels, verify

from .prompts import GENERATION_PROMPT, NATURALIZE_PROMPT

Generator = Callable[[str, float, int], str]


@dataclass(frozen=True)
class Reference:
    identifier: str
    template: str
    activities: tuple[str, ...]
    tree: dict[str, Any]

    @classmethod
    def from_spec(
        cls,
        identifier: str,
        template: str,
        activities: Sequence[str],
        spec: Any,
    ) -> "Reference":
        tree = spec_to_canon(spec)
        assert_no_dup_labels(tree)
        expected = {label for label, _ in _leaf_instances(tree)}
        supplied = set(activities)
        if supplied != expected:
            raise ValueError("activities must match the visible labels in the reference tree")
        return cls(identifier, template, tuple(activities), tree)


@dataclass(frozen=True)
class SynthesisConfig:
    rewrite_temperature: float = 0.3
    candidate_temperature: float = 0.3
    rewrite_budget: int = 2
    candidate_budget: int = 4
    gate_threshold: float = 0.8
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        if self.rewrite_budget < 1 or self.candidate_budget < 1:
            raise ValueError("generation budgets must be positive")
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("gate_threshold must be between 0 and 1")


@dataclass(frozen=True)
class SynthesisResult:
    identifier: str
    description: str
    answer: str
    candidate: dict[str, Any]
    rel: float
    hotlines: tuple[int, ...]
    rewrite_attempt: int
    candidate_attempt: int


def _leaf_instances(tree: dict[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    if tree["t"] == "act":
        return [(tree["label"], ())]
    if tree["t"] == "silent":
        return []
    result = []
    for index, child in enumerate(tree["children"]):
        for label, path in _leaf_instances(child):
            result.append((label, (index,) + path))
    return result


def coverage_missing(text: str, activities: Iterable[str]) -> list[str]:
    return [activity for activity in activities if activity not in text]


def _activity_text(activities: Sequence[str]) -> str:
    return ", ".join(f"'{activity}'" for activity in activities)


def rewrite_prompt(reference: Reference) -> str:
    return NATURALIZE_PROMPT.format(
        template=reference.template,
        activities=_activity_text(reference.activities),
    )


def candidate_prompt(description: str, activities: Sequence[str]) -> str:
    return GENERATION_PROMPT.replace("{description}", description).replace(
        "{activities}", _activity_text(activities)
    )


def synthesize(
    reference: Reference,
    generator: Generator,
    config: Optional[SynthesisConfig] = None,
) -> Optional[SynthesisResult]:
    settings = config or SynthesisConfig()
    rewrite = rewrite_prompt(reference)
    for rewrite_attempt in range(1, settings.rewrite_budget + 1):
        description = generator(rewrite, settings.rewrite_temperature, settings.max_tokens).strip()
        if coverage_missing(description, reference.activities):
            continue
        prompt = candidate_prompt(description, reference.activities)
        for candidate_attempt in range(1, settings.candidate_budget + 1):
            answer = generator(prompt, settings.candidate_temperature, settings.max_tokens).strip()
            try:
                spec, _ = parse_spec_tolerant(answer)
                candidate = spec_to_canon(spec)
                assert_no_dup_labels(candidate)
            except (TypeError, ValueError):
                continue
            result = verify(candidate, reference.tree, settings.gate_threshold)
            if result["accepted"]:
                return SynthesisResult(
                    identifier=reference.identifier,
                    description=description,
                    answer=answer,
                    candidate=candidate,
                    rel=result["rel"],
                    hotlines=tuple(result["hotlines"]),
                    rewrite_attempt=rewrite_attempt,
                    candidate_attempt=candidate_attempt,
                )
    return None


def build_spotlight_example(reference: Reference, result: SynthesisResult) -> dict[str, Any]:
    if reference.identifier != result.identifier:
        raise ValueError("reference and synthesis result identifiers must match")
    return {
        "instruction": candidate_prompt(result.description, reference.activities),
        "input": "",
        "output": spec_to_text(canon_to_spec(reference.tree)),
        "hotlines": list(result.hotlines),
    }


def run_round(
    references: Iterable[Reference],
    generator: Generator,
    config: Optional[SynthesisConfig] = None,
) -> tuple[list[SynthesisResult], list[dict[str, Any]]]:
    accepted = []
    examples = []
    for reference in references:
        result = synthesize(reference, generator, config)
        if result is not None:
            accepted.append(result)
            examples.append(build_spotlight_example(reference, result))
    return accepted, examples
