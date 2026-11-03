"""GPQA Diamond multiple choice at Avg@8."""
from __future__ import annotations

import json
import random
import re
from typing import List

from . import config

METRIC_KEY = "correct"
REPEATS = 8
SHUFFLE_SEED = 0
_ANSWER_RE = re.compile(r"Answer[ \t]*:[ \t]*\$?([A-D])\$?", re.IGNORECASE)


def _rows(path):
    if path.endswith(".jsonl"):
        with open(path) as handle:
            return [json.loads(line) for line in handle if line.strip()]
    import pandas as pd
    frame = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _normalise(row):
    question = row.get("Question", row.get("question"))
    choices = row.get("Choices", row.get("choices"))
    answer = row.get("Answer", row.get("answer"))
    if choices is None and row.get("Correct Answer") is not None:
        choices = [
            row["Correct Answer"],
            row.get("Incorrect Answer 1"),
            row.get("Incorrect Answer 2"),
            row.get("Incorrect Answer 3"),
        ]
        answer = 0
    if question is None or choices is None or answer is None:
        raise ValueError("GPQA row must contain question, choices, and answer")
    if isinstance(choices, str):
        choices = json.loads(choices)
    choices = list(choices)
    if len(choices) != 4:
        raise ValueError(f"GPQA expects four choices, got {len(choices)}")
    if isinstance(answer, str) and answer.strip().upper() in "ABCD":
        answer_index = "ABCD".index(answer.strip().upper())
    else:
        answer_index = int(answer)
    return str(question), choices, answer_index


def load_examples(path: str = None) -> List[dict]:
    examples = []
    rng = random.Random(SHUFFLE_SEED)
    for problem, row in enumerate(_rows(path or config.GPQA_PATH)):
        question, choices, answer_index = _normalise(row)
        for repeat in range(REPEATS):
            order = rng.sample(range(4), 4)
            content = (
                "Answer the following multiple choice question. The last line "
                "of your response should be of the following format: 'Answer: "
                "$LETTER' (without quotes) where LETTER is one of ABCD. Think "
                "step by step before answering.\n\n"
                f"{question}\n\n" +
                "\n".join(
                    f"{letter}) {choices[index]}"
                    for letter, index in zip("ABCD", order)
                )
            )
            examples.append({
                "problem": problem,
                "repeat": repeat,
                "messages": [{"role": "user", "content": content}],
                "correct_letter": "ABCD"[order.index(answer_index)],
                "choice_order": order,
            })
    return examples


def build(examples):
    return [ex["messages"] for ex in examples], [{"example": ex} for ex in examples]


def seeds(prepared, base_seed: int):
    return [
        base_seed + 1_000_003 * item["example"]["problem"] + item["example"]["repeat"]
        for item in prepared
    ]


def extract_letter(text: str):
    text = str(text or "")
    if "</think>" not in text:
        return None
    matches = _ANSWER_RE.findall(text.split("</think>", 1)[1])
    return matches[-1].upper() if matches else None


def grade(prepared, completions):
    records = []
    for item, completion in zip(prepared, completions, strict=True):
        example = item["example"]
        predicted = extract_letter(completion.get("text") or "")
        records.append({
            "problem": example["problem"],
            "repeat": example["repeat"],
            "correct_letter": example["correct_letter"],
            "predicted_letter": predicted,
            "correct": int(predicted == example["correct_letter"]),
            "answered": int(predicted is not None),
            "raw_output": str(completion.get("text") or ""),
            "gen_tokens": completion.get("gen_tokens"),
            "finish_reason": completion.get("finish_reason"),
        })
    return records
