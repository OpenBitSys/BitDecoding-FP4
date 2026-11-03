#!/usr/bin/env python3
"""Load the LiveCodeBench v6 slice without paying for the other five.

``load_code_generation_dataset`` cannot fetch one version. Its loader script
calls ``dl_manager.download_and_extract(_URLS)`` on the whole mapping and only
then filters by ``version_tag``, so asking for the 175-problem v6 slice
downloads 4,486 MB to read 134 of it. The slice is exactly ``test6.jsonl``.

Order is the file's own line order, which is what the official loader returns:
its ``_generate_examples`` walks files then lines, and ``load_dataset`` does
not reorder. That order is the contract with the grader, so ``bench_index`` is
the line number and nothing here sorts.

This module imports ``lcb_runner`` from a LiveCodeBench checkout on
``PYTHONPATH`` (``_local/LiveCodeBench``). Do not pip-install LiveCodeBench:
its ``pyproject.toml`` would pull a different vLLM. Grading runs in a **new
process** of the same inference environment so it does not share a CUDA
context with the engine.
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LCB_CHECKOUT = os.path.join(REPO_ROOT, "_local", "LiveCodeBench")
LCB_REPO = "livecodebench/code_generation_lite"
V6_SHARD = "test6.jsonl"
PROMPTS_PATH = os.path.join(REPO_ROOT, "datasets", "lcb_v6", "prompts.jsonl")
LCB_TESTS_PATH_ENV = "BITDECODING_LCB_TESTS_PATH"

if LCB_CHECKOUT not in sys.path:
    sys.path.insert(0, LCB_CHECKOUT)


def _resolve_v6_path() -> str:
    """Use an explicit offline shard, or fall back to the Hub cache."""
    local_path = os.environ.get(LCB_TESTS_PATH_ENV)
    if local_path:
        path = os.path.abspath(os.path.expanduser(local_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{LCB_TESTS_PATH_ENV} points to a missing file: {path}"
            )
        return path

    from huggingface_hub import hf_hub_download

    return hf_hub_download(LCB_REPO, V6_SHARD, repo_type="dataset")


def load_v6_problems():
    """The v6 contest window as ``CodeGenerationProblem``, in benchmark order."""
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    path = _resolve_v6_path()
    problems = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                problems.append(CodeGenerationProblem(**json.loads(line)))
    return problems


def export_prompts(out_path: str = PROMPTS_PATH) -> int:
    """Write the fields generation needs, and only those.

    Test cases stay out of the file on purpose: the generating process must
    not be able to read the answers, and the private cases are most of the
    134 MB.
    """
    problems = load_v6_problems()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        for index, problem in enumerate(problems):
            handle.write(json.dumps({
                "bench_index": index,
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "question_content": problem.question_content,
                "starter_code": problem.starter_code,
                "platform": problem.platform.value,
                "difficulty": problem.difficulty.value,
                "contest_date": problem.contest_date.isoformat(),
            }, ensure_ascii=False) + "\n")
    return len(problems)


def main() -> None:
    count = export_prompts()
    problems = load_v6_problems()
    dates = sorted(problem.contest_date for problem in problems)
    starters = sum(1 for problem in problems if problem.starter_code)
    print(f"[lcb] {count} problems -> {PROMPTS_PATH}")
    print(f"[lcb] contest dates {dates[0].date()} .. {dates[-1].date()}")
    print(f"[lcb] {starters} with starter code, {count - starters} stdin")


if __name__ == "__main__":
    main()
