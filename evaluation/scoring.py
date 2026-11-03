"""F1 scoring for LongBench-Pro T9.

Vendored (trimmed to the four public names used here) from
caskcsg/longcontext/LongBench-Pro/modules/utils.py, preserving its scoring
semantics.
"""
from __future__ import annotations

import re
from typing import List, Set


_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove one leading Qwen reasoning block.

    Some chat templates include ``<think>`` in the assistant prefill, so the
    returned generation starts with the reasoning body and only contains the
    closing ``</think>`` tag. Accept both that form and the self-contained
    ``<think>...</think>`` form.
    """
    value = str(text or "")
    stripped = _THINK_RE.sub("", value, count=1)
    if stripped == value and "</think>" in value:
        stripped = value.split("</think>", 1)[1].lstrip()
    return stripped


def get_answer_area(text: str) -> str:
    text = strip_thinking(text)
    if "[Answer]" in text or "[答案]" in text:
        if "[Answer]" in text:
            last = text.rfind("[Answer]")
            if last != -1:
                text = text[last + len("[Answer]"):]
        else:
            last = text.rfind("[答案]")
            if last != -1:
                text = text[last + len("[答案]"):]
    return text.strip()


def _fix_space(text: str) -> str:
    return " ".join(text.split())


def normalize_answers(answers: List[str]) -> List[str]:
    return [_fix_space(a.lower().strip()) for a in answers]


def normalize_prediction(prediction: str) -> List[str]:
    area = get_answer_area(prediction).lower()
    return [_fix_space(line.strip()) for line in area.split("\n")]


def F1_Score(answers: List[str], prediction: str) -> float:
    a = normalize_answers(answers)
    p = normalize_prediction(prediction)
    a_set: Set[str] = set(a)
    p_set: Set[str] = set(p)
    common = a_set & p_set
    if not common or not p_set or not a_set:
        return 0.0
    precision = len(common) / len(p_set)
    recall = len(common) / len(a_set)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
