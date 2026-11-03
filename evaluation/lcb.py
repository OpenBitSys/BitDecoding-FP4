"""LiveCodeBench v6 code generation (Avg@4 pass@1).

Three things about the official harness shape this module.

The official loader is a dataset script, which ``datasets`` 4.0 removed, so
this module never imports ``lcb_runner``. The benchmark is read once into the
flat JSONL this module consumes -- prompts only, no test cases. LiveCodeBench
stays on ``PYTHONPATH`` for ``tools/score_lcb.py``; it is not pip-installed.

It does no extraction. ``custom_evaluator`` passes the string it is given as
both the raw output and the extracted code, so what we submit must already be
runnable Python. :func:`extract_code` is LCB's own rule, reproduced exactly:
the content between the last two lines containing a fence, and the empty string
when there are fewer than two.

It zips our list against the benchmark in load order, sorting only when the
dict form is used. Rows therefore carry ``bench_index``, their position in that
load order, and ``tools/score_lcb.py`` submits the positional form. Nothing then
depends on ``question_id`` sorting the way the benchmark happens to be ordered.

Scoring is deferred for a fourth reason: it executes model-written code, which
must not happen in a process holding a CUDA context. Generation writes
``pass1=None``; ``tools/score_lcb.py`` fills it in.
Each of the 175 problems is generated four times with stable per-row seeds;
the sandbox grader evaluates every repeat and averages by problem.
"""
from __future__ import annotations

import json
from typing import List

from . import config
from .scoring import strip_thinking


METRIC_KEY = "pass1"
REPEATS = 4

# Reproduced from lcb_runner/prompts/code_generation.py::PromptConstants so the
# prompt is byte-identical to the one the published numbers were produced with.
SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)

def load_examples(path: str = None) -> List[dict]:
    rows = []
    with open(path or config.LCB_PROMPTS) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Load order is the contract with the evaluator, so it is restored here
    # rather than trusted to have survived the file.
    rows.sort(key=lambda row: row["bench_index"])
    repeated = []
    for row in rows:
        problem_index = row["bench_index"]
        for repeat in range(REPEATS):
            item = dict(row)
            item["problem_index"] = problem_index
            item["repeat"] = repeat
            item["bench_index"] = problem_index * REPEATS + repeat
            repeated.append(item)
    return repeated


def seeds(prepared, base_seed: int):
    return [
        base_seed + 1_000_003 * item["example"]["problem_index"] + item["example"]["repeat"]
        for item in prepared
    ]


def build_prompt(example: dict) -> str:
    prompt = f"### Question:\n{example['question_content']}\n\n"
    if example.get("starter_code"):
        prompt += f"### Format: {FORMATTING_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{example['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def build(examples):
    conversations, prepared = [], []
    for example in examples:
        conversations.append([
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_prompt(example)},
        ])
        prepared.append({"example": example})
    return conversations, prepared


def extract_code(model_output: str) -> str:
    """LCB's own extraction: between the last two fence lines, else empty.

    The thinking block is stripped first. A reasoning model fences code inside
    its reasoning, and the last two fences in the whole string would otherwise
    be able to land there rather than on the answer.
    """
    answer = strip_thinking(model_output)
    lines = answer.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1:fences[-1]])


def grade(prepared, completions) -> List[dict]:
    records = []
    for item, completion in zip(prepared, completions, strict=True):
        example = item["example"]
        text = str(completion.get("text") or "")
        records.append({
            "bench_index": example["bench_index"],
            "problem_index": example["problem_index"],
            "repeat": example["repeat"],
            "question_id": example["question_id"],
            "platform": example.get("platform"),
            "difficulty": example.get("difficulty"),
            "contest_date": example.get("contest_date"),
            "code": extract_code(text),
            "raw_output": text,
            "pass1": None,
            "gen_tokens": completion.get("gen_tokens"),
            "finish_reason": completion.get("finish_reason"),
        })
    return records
