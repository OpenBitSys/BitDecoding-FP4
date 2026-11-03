"""AIME 2025 accuracy at Avg@32, scored on the boxed integer.

The staged file holds the 30 problems once; the 32 repeats are made here.
Repeats are a sampling decision, not data, so they are not written to disk.
:mod:`evaluation.aime` uses the same grader on the 2024 set.

Temperature is 0.6, so the 32 repeats of a problem must not share a seed or they
would decode identically and Avg@32 would collapse to Avg@1. :func:`seeds`
derives one per row from the problem and repeat number rather than from the
row's position, so it survives sharding and is the same in every arm -- two arms
then differ by their KV cache and nothing else.

Grading is exact match on the last ``\\boxed{}``, read as an integer. AIME
answers are integers 0--999, so this is not an approximation of symbolic
verification but an equivalent of it on this answer space, and it needs no
dependency that can silently change its mind between runs. ``math_verify``
still runs when it is installed, and its verdict is recorded alongside rather
than instead: the two disagreeing is a fact about the extraction worth seeing,
and the report states the rate.
"""
from __future__ import annotations

import json
import re
import sys
from typing import List

from . import config


METRIC_KEY = "acc"

# Avg@32. One request per row; the repeats live in the example list.
REPEATS = 32

# Qwen's own instruction for math, from the Qwen3-8B model card. The answer
# extractor below depends on it, so the two move together.
ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def load_examples(path: str = None) -> List[dict]:
    import pandas as pd

    frame = pd.read_parquet(path or config.AIME25_PARQUET)
    examples = []
    for problem, record in enumerate(frame.to_dict(orient="records")):
        content = f"{record['problem']}\n\n{ANSWER_INSTRUCTION}"
        for repeat in range(REPEATS):
            examples.append({
                "problem": problem,
                "repeat": repeat,
                "messages": [{"role": "user", "content": content}],
                "ground_truth": str(record["answer"]),
            })
    return examples


def build(examples):
    return [ex["messages"] for ex in examples], [{"example": ex} for ex in examples]


def seeds(prepared, base_seed: int) -> List[int]:
    """One sampling seed per row, keyed by what the row *is*.

    Not by where it sits: shards are strided, so a position-derived seed would
    give a row a different one at a different shard count, and the arms would
    stop being comparable to anything run earlier.
    """
    return [
        base_seed + 1_000_003 * item["example"]["problem"] + item["example"]["repeat"]
        for item in prepared
    ]


def last_boxed(text: str) -> str | None:
    """The content of the final ``\\boxed{...}``, brace-balanced.

    The final one rather than the first because a thinking model boxes its
    working as it goes; the answer is the last thing it boxes. Balanced rather
    than a regex because the content can itself contain braces.
    """
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:index]
    return None


def as_integer(boxed: str | None) -> int | None:
    """An AIME answer, or None if what was boxed is not one."""
    if boxed is None:
        return None
    cleaned = boxed.strip()
    for junk in ("\\!", "\\,", "\\;", "\\ ", "$", ",", " ", "\\left", "\\right"):
        cleaned = cleaned.replace(junk, "")
    cleaned = re.sub(r"^\\text\{(.*)\}$", r"\1", cleaned).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", cleaned):
        return None
    try:
        return int(cleaned.split(".", 1)[0])
    except ValueError:
        # Invalid model output can exceed Python's integer-string limit.
        return None


def _math_verify(prediction: str, ground_truth: str):
    """``math_verify``'s verdict, or None if it is unavailable or fails.

    Never allowed to decide the metric, and never allowed to end a run: an
    optional cross-check that raises is a cross-check that stops the
    evaluation it was added to observe.
    """
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        gold = parse(f"${ground_truth}$")
        answer = parse(prediction)
        return float(bool(verify(gold, answer)))
    except Exception:
        return None


def _score_record(record: dict) -> None:
    boxed = last_boxed(record["solution_str"])
    predicted = as_integer(boxed)
    try:
        gold = int(str(record["ground_truth"]).strip())
    except ValueError:
        gold = None
    record["boxed"] = boxed
    record["predicted"] = predicted
    record["acc"] = float(predicted is not None and predicted == gold)
    record["no_answer"] = float(predicted is None)
    record["acc_math_verify"] = _math_verify(
        record["solution_str"], str(record["ground_truth"])
    )


def grade(prepared, completions) -> List[dict]:
    records = []
    for item, completion in zip(prepared, completions, strict=True):
        example = item["example"]
        record = {
            "problem": example["problem"],
            "repeat": example["repeat"],
            "ground_truth": example["ground_truth"],
            "solution_str": str(completion.get("text") or ""),
            "gen_tokens": completion.get("gen_tokens"),
            "finish_reason": completion.get("finish_reason"),
        }
        _score_record(record)
        records.append(record)
    return records


def score_jsonl(path: str) -> dict:
    """Re-score an ``aime25__<mode>.jsonl`` in place."""
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        _score_record(record)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    n = len(records)
    checked = [r for r in records if r.get("acc_math_verify") is not None]
    return {
        "n": n,
        "acc_mean": sum(r["acc"] for r in records) / n if n else 0.0,
        "n_no_answer": sum(int(r["no_answer"]) for r in records),
        "n_truncated": sum(1 for r in records if r.get("finish_reason") == "length"),
        "n_math_verify_disagree": sum(
            1 for r in checked if r["acc_math_verify"] != r["acc"]
        ),
        "n_math_verify_checked": len(checked),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-score aime25__<mode>.jsonl on the boxed integer"
    )
    parser.add_argument("jsonl", nargs="+")
    args = parser.parse_args()
    for path in args.jsonl:
        summary = score_jsonl(path)
        print(
            f"{path}: n={summary['n']} acc={summary['acc_mean']:.4f} "
            f"no_answer={summary['n_no_answer']} "
            f"truncated={summary['n_truncated']} "
            f"mv_disagree={summary['n_math_verify_disagree']}"
            f"/{summary['n_math_verify_checked']}",
            file=sys.stdout,
        )


if __name__ == "__main__":
    main()
