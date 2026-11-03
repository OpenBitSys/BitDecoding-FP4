"""OpenAI MRCR benchmark (SequenceMatcher ratio).

Uses the pre-tokenized/filtered cache selected in ``evaluation/config.py``: each row
carries its multi-turn ``messages``, gold ``answer`` and the
``random_string_to_prepend`` hash prefix.  Grading is the official metric — the
response must start with the hash prefix (a single leading Qwen ``<think>``
block is stripped first), then it's the difflib ratio vs the gold answer.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import List

from . import config
from .scoring import strip_thinking


def _strip_think(response: str) -> str:
    return strip_thinking(response)


def grade_one(response: str, answer: str, prefix: str) -> float:
    if response is None:
        return 0.0
    response = _strip_think(response)
    if not response.startswith(prefix):
        return 0.0
    r = response[len(prefix):]
    a = answer[len(prefix):] if answer.startswith(prefix) else answer
    return float(SequenceMatcher(None, r, a).ratio())


def load_examples(cache_path: str = None) -> List[dict]:
    rows = []
    with open(cache_path or config.MRCR_FILTERED_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(examples):
    """Return (conversations, prepared) for the vLLM chat runner."""
    conversations, prepared = [], []
    for ex in examples:
        conversations.append(ex["messages"])
        prepared.append({"example": ex, "prompt_tokens": ex.get("prompt_tokens")})
    return conversations, prepared


def grade(prepared, completions) -> List[dict]:
    records = []
    for item, comp in zip(prepared, completions, strict=True):
        ex = item["example"]
        pred = str(comp.get("text") or "")
        prefix = ex["random_string_to_prepend"]
        stripped = _strip_think(pred)
        records.append({
            "id": ex.get("id"),
            "n_needles": ex.get("n_needles"),
            "ratio": grade_one(pred, ex["answer"], prefix),
            "ok_prefix": bool(stripped) and stripped.startswith(prefix),
            "prompt_tokens": item["prompt_tokens"],
            "prediction": pred,
            "gen_tokens": comp.get("gen_tokens"),
            "finish_reason": comp.get("finish_reason"),
        })
    return records


METRIC_KEY = "ratio"
