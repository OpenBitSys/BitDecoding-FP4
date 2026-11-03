#!/usr/bin/env python3
"""Grade LiveCodeBench generations with the official checker, in a sandbox.

Runs in a **new process** of the inference environment
(``_local/envs/bd_fp4``), never in a process that already holds a CUDA
context. LiveCodeBench is imported from a checkout on ``PYTHONPATH``; it is
not pip-installed, because that would replace this repository's vLLM.

Grading itself is ``lcb_runner.evaluation.codegen_metrics`` -- the same
function ``custom_evaluator`` calls. What is skipped is that CLI's benchmark
load, because it goes through a loader that downloads 4,486 MB to read the 134
that hold v6 (see tools/lcb_data.py), and its alignment step, which sorts
generations by ``str(question_id)`` and zips them against the benchmark.
Records carry ``bench_index`` instead, so alignment is positional and cannot
silently pair a solution with another problem's tests.

Isolation is a user namespace with no network and its own PID namespace, plus
an address-space limit. LCB's ``reliability_guard`` says in its own docstring
that it is "not a security sandbox" -- it nulls out some Python attributes --
and it is called with ``maximum_memory_bytes=None``, so nothing else here caps
memory.

    _local/envs/bd_fp4/bin/python tools/score_lcb.py \\
        --run-dir _local/eval_smoke --mode bitdecoding-nvfp4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from tools.lcb_data import load_v6_problems  # noqa: E402
from evaluation.lcb import REPEATS  # noqa: E402

SANDBOX_MARKER = "BITDECODING_LCB_SANDBOXED"


def _reexec_sandboxed(memory_gib: int) -> None:
    """Re-run this program inside a network-less user + PID namespace."""
    os.environ[SANDBOX_MARKER] = "1"
    os.environ["BITDECODING_LCB_MEMORY_GIB"] = str(memory_gib)
    command = [
        "unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
        sys.executable, os.path.abspath(__file__), *sys.argv[1:],
    ]
    os.execvp(command[0], command)


def load_records(run_dir: str, mode: str) -> dict:
    """Every graded row for one arm, keyed by its position in the benchmark."""
    pattern = os.path.join(run_dir, "**", f"lcb__{mode}.jsonl")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"no lcb__{mode}.jsonl under {run_dir}")
    by_index, source = {}, {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                index = (
                    record.get("problem_index", record["bench_index"]),
                    record.get("repeat", 0),
                )
                if index in by_index:
                    raise SystemExit(
                        f"row {index} appears in both "
                        f"{source[index]} and {path}"
                    )
                by_index[index] = record
                source[index] = path
    return by_index, source


def write_back(source: dict, by_index: dict) -> None:
    """Rewrite each shard file with ``pass1`` filled in, order preserved."""
    for path in sorted(set(source.values())):
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows:
            key = (
                row.get("problem_index", row["bench_index"]),
                row.get("repeat", 0),
            )
            row["pass1"] = by_index[key]["pass1"]
        with open(path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--num-process", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--memory-gib", type=int, default=16,
                        help="address-space cap for this process and its children")
    parser.add_argument("--no-sandbox", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="grade a subset (smoke runs), instead of requiring "
                             "every problem in the slice")
    args = parser.parse_args()

    if not args.no_sandbox and not os.environ.get(SANDBOX_MARKER):
        _reexec_sandboxed(args.memory_gib)

    limit = int(os.environ.get("BITDECODING_LCB_MEMORY_GIB", args.memory_gib)) * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    from lcb_runner.evaluation import codegen_metrics, extract_instance_results

    problems = load_v6_problems()
    by_index, source = load_records(args.run_dir, args.mode)
    keys = sorted(by_index)
    problem_indices = sorted({key[0] for key in keys})
    if problem_indices and (problem_indices[-1] >= len(problems)):
        raise SystemExit(
            f"problem_index {problem_indices[-1]} is outside the {len(problems)}-problem "
            "slice; the prompts and the grader disagree about the benchmark"
        )
    expected_keys = {
        (problem_index, repeat)
        for problem_index in problem_indices
        for repeat in range(REPEATS)
    }
    missing = expected_keys - set(keys)
    if missing and not args.allow_partial:
        raise SystemExit(f"missing repeat rows, for example {sorted(missing)[:3]}")
    if len(problem_indices) != len(problems) and not args.allow_partial:
        raise SystemExit(
            f"{len(problem_indices)} of {len(problems)} problems present; pass "
            "--allow-partial to grade a subset"
        )

    samples = [problems[key[0]].get_evaluation_sample() for key in keys]
    generations = [[by_index[key]["code"] or ""] for key in keys]
    empty = sum(1 for g in generations if not g[0].strip())
    print(
        f"[lcb] grading {len(keys)} rows for {args.mode} "
        f"({empty} with no extracted code)",
        flush=True,
    )

    metrics, results, _metadata = codegen_metrics(
        samples, generations,
        num_process_evaluate=args.num_process,
        timeout=args.timeout,
    )
    graded = extract_instance_results(results)

    for position, key in enumerate(keys):
        by_index[key]["pass1"] = float(bool(graded[position][0]))
    write_back(source, by_index)

    per_problem = {}
    for key in keys:
        per_problem.setdefault(key[0], []).append(by_index[key]["pass1"])
    if not args.allow_partial:
        bad = [index for index, values in per_problem.items() if len(values) != REPEATS]
        if bad:
            raise SystemExit(f"problem repeats incomplete, for example {bad[:3]}")
    problem_means = [sum(values) / len(values) for values in per_problem.values()]
    passed = sum(problem_means)
    print(
        f"[lcb] Avg@{REPEATS} = {passed / len(problem_means):.4f} "
        f"({passed:.2f}/{len(problem_means)} problems)"
    )
    print(f"[lcb] official metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
