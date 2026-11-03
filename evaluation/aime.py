"""AIME 2024 accuracy at Avg@32, scored on the last boxed integer.

This is the official contest metric (integer 0--999, exact match) plus the
Qwen3 eval protocol: ``Please reason step by step, and put your final answer
within \\boxed{}``, thinking-mode sampling, Avg@32. The grader is the same as
:mod:`evaluation.aime25`. Repeats are made at load time from the 30-problem file;
they are a sampling decision, not data.

DAPO's Minerva ``Answer:`` scorer is a training reward and is not used here.
"""
from __future__ import annotations

from . import aime25, config

METRIC_KEY = aime25.METRIC_KEY
REPEATS = aime25.REPEATS
last_boxed = aime25.last_boxed
as_integer = aime25.as_integer
build = aime25.build
seeds = aime25.seeds
grade = aime25.grade
score_jsonl = aime25.score_jsonl


def load_examples(path: str = None):
    return aime25.load_examples(path or config.AIME24_PARQUET)


def main() -> None:
    aime25.main()


if __name__ == "__main__":
    main()
