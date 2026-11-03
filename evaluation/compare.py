"""Merge sharded eval output and compare arms on the examples they share.

``run_eval`` scores one configuration at a time and writes one record per
example per shard. Deciding whether one configuration beats another is a
separate question, and the honest form of it is paired: every arm answers the
same benchmark, and examples differ from each other by far more than arms
differ from each other, so an unpaired interval is dominated by variation that
cancels.

Shards are strided (``examples[shard_index::num_shards]``), so an example's
position in the benchmark is ``shard_index + row * num_shards``. That is the
identity two arms are joined on; nothing else in the record identifies an
example, since ``id`` is null in the filtered MRCR file.

Usage:

    python -m evaluation.compare --root _local/accuracy \\
        --baseline bf16 --benchmark mrcr --mode bitdecoding-nvfp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np


METRIC_KEYS = {
    "mrcr": "ratio",
    "longbench_pro": "score",
    "aime": "acc",
    "aime25": "acc",
    "lcb": "pass1",
    "ruler_niah": "recall",
    "gpqa": "correct",
}
# Benchmarks whose rows are not independent: AIME is 30 problems sampled 32
# times each, so a row-level resample treats 960 draws as 960 problems and
# reports an interval about sqrt(32) too narrow. The unit of replication is
# the problem, and both the interval and the standard error use it.
CLUSTER_KEYS = {
    "aime": "problem", "aime25": "problem", "gpqa": "problem",
    "lcb": "problem_index",
}
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 42


def _shard_file(shard_dir: str, benchmark: str, mode: str | None) -> str:
    """The one result file for this benchmark in this shard.

    A named ``mode`` is required to be there. Without one, the file is found by
    benchmark alone. Each arm directory must contain exactly one mode when
    ``--mode`` is omitted.
    """
    if mode is not None:
        raw = os.path.join(shard_dir, f"{benchmark}__{mode}.jsonl")
        scored = raw.removesuffix(".jsonl") + ".scored.jsonl"
        return scored if os.path.exists(scored) else raw
    found = sorted(glob.glob(os.path.join(shard_dir, f"{benchmark}__*.jsonl")))
    scored = [path for path in found if path.endswith(".scored.jsonl")]
    if scored:
        found = scored
    if not found:
        raise ValueError(f"{shard_dir} has no {benchmark}__*.jsonl")
    if len(found) > 1:
        raise ValueError(
            f"{shard_dir} holds {len(found)} modes for {benchmark} "
            f"({', '.join(os.path.basename(p) for p in found)}); pass --mode"
        )
    return found[0]


def load_arm(
    arm_dir: str, benchmark: str, mode: str | None
) -> dict[Any, dict[str, Any]]:
    """One arm's per-example records, keyed by position in the benchmark."""
    records: dict[Any, dict[str, Any]] = {}
    shards = sorted(
        name
        for name in os.listdir(arm_dir)
        if name.startswith("shard_") and os.path.isdir(os.path.join(arm_dir, name))
    )
    if not shards:
        raise ValueError(f"{arm_dir} has no shard_* directories")
    for name in shards:
        shard_index = int(name.removeprefix("shard_"))
        path = _shard_file(os.path.join(arm_dir, name), benchmark, mode)
        if not os.path.exists(path):
            raise ValueError(f"arm {arm_dir!r} is missing {path}")
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row_index, row in enumerate(rows):
            position = shard_index + row_index * len(shards)
            selection_identity = row.get("_selection_identity")
            identity = (
                json.dumps(selection_identity, sort_keys=True)
                if isinstance(selection_identity, dict)
                else (
                    str(row["id"])
                    if benchmark == "longbench_pro" and row.get("id") is not None
                    else position
                )
            )
            if identity in records:
                raise ValueError(
                    f"arm {arm_dir!r} repeats {benchmark} row identity {identity!r}"
                )
            records[identity] = row
    return records


def _metric_value(row: dict[str, Any], benchmark: str) -> float:
    key = METRIC_KEYS[benchmark]
    value = row.get(key)
    # Accept the older LongBench-Pro metric name when reading saved results.
    if value is None and benchmark == "longbench_pro":
        value = row.get("f1")
    return float(value or 0.0)


def _cluster_means(delta: np.ndarray, clusters: np.ndarray | None) -> np.ndarray:
    """One number per unit of replication: a row, or a cluster of them."""
    if clusters is None:
        return delta
    labels = np.unique(clusters)
    return np.asarray([delta[clusters == label].mean() for label in labels])


def paired_bootstrap(
    delta: np.ndarray,
    *,
    clusters: np.ndarray | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile interval for the mean of example-wise differences.

    With ``clusters``, whole clusters are resampled rather than rows. The
    design here is balanced, so the mean over rows equals the mean over
    cluster means and only the interval changes.
    """
    units = _cluster_means(delta, clusters)
    generator = np.random.default_rng(seed)
    means = []
    remaining = resamples
    while remaining:
        take = min(256, remaining)
        indices = generator.integers(0, units.size, size=(take, units.size))
        means.append(units[indices].mean(axis=1))
        remaining -= take
    means = np.concatenate(means)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    if denominator == 0.0:
        raise ValueError("length-slope input needs at least two prompt lengths")
    return float(np.dot(centered, y - y.mean()) / denominator)


def paired_slope_bootstrap(
    prompt_tokens: np.ndarray,
    delta: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 256,
) -> tuple[float, float, float]:
    """Slope and percentile CI for paired delta versus log2 prompt length.

    Rows are resampled as paired ``(length, delta)`` observations. Resamples
    are generated in bounded batches so 20,000 resamples of the full long-
    context pool do not allocate a multi-gigabyte index matrix.
    """
    tokens = np.asarray(prompt_tokens, dtype=float)
    values = np.asarray(delta, dtype=float)
    if tokens.shape != values.shape or tokens.ndim != 1:
        raise ValueError("prompt_tokens and delta must be same-length vectors")
    if tokens.size < 2 or np.any(tokens <= 0):
        raise ValueError("length-slope input needs at least two positive lengths")
    x = np.log2(tokens)
    observed = _linear_slope(x, values)
    generator = np.random.default_rng(seed)
    slopes: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        take = min(batch_size, remaining)
        indices = generator.integers(0, x.size, size=(take, x.size))
        bx = x[indices]
        by = values[indices]
        bx = bx - bx.mean(axis=1, keepdims=True)
        by = by - by.mean(axis=1, keepdims=True)
        denominator = np.square(bx).sum(axis=1)
        valid = denominator > 0
        slopes.append((bx[valid] * by[valid]).sum(axis=1) / denominator[valid])
        remaining -= take
    sampled = np.concatenate(slopes)
    if sampled.size < max(100, resamples // 2):
        raise ValueError("too many degenerate length-slope bootstrap resamples")
    low, high = np.quantile(sampled, [0.025, 0.975])
    return observed, float(low), float(high)


def compare_length_slope(
    root: str,
    *,
    baseline: str,
    arm: str,
    mode: str | None = None,
    benchmarks: tuple[str, ...] = ("longbench_pro", "mrcr"),
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Registered long-context evaluation length-slope test pooled over long benchmarks."""
    tokens: list[float] = []
    deltas: list[float] = []
    counts: dict[str, int] = {}
    for benchmark in benchmarks:
        base_rows = load_arm(os.path.join(root, baseline), benchmark, mode)
        arm_rows = load_arm(os.path.join(root, arm), benchmark, mode)
        shared = sorted(set(base_rows) & set(arm_rows))
        counts[benchmark] = len(shared)
        for index in shared:
            prompt_tokens = base_rows[index].get("prompt_tokens")
            if prompt_tokens is None:
                raise ValueError(
                    f"{benchmark} row {index} has no prompt_tokens for slope"
                )
            tokens.append(float(prompt_tokens))
            deltas.append(
                _metric_value(arm_rows[index], benchmark)
                - _metric_value(base_rows[index], benchmark)
            )
    observed, low, high = paired_slope_bootstrap(
        np.asarray(tokens), np.asarray(deltas),
        resamples=resamples, seed=seed,
    )
    summary: dict[str, Any] = {
        "baseline": baseline,
        "arm": arm,
        "benchmarks": list(benchmarks),
        "n_examples": len(tokens),
        "n_by_benchmark": counts,
        "x": "log2(prompt_tokens)",
        "slope_per_doubling": observed,
        f"bootstrap_95_ci_seed{seed}_{resamples}": [low, high],
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def compare(
    root: str,
    benchmark: str,
    mode: str,
    *,
    baseline: str,
    arms: list[str] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    metric = METRIC_KEYS[benchmark]
    names = arms or sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )
    # Naming a subset of arms is how you compare two of many, and the baseline
    # is what they are compared against rather than one of them, so asking for
    # it twice would be the odd spelling.
    if baseline not in names:
        names = [baseline, *names]
    loaded = {
        name: load_arm(os.path.join(root, name), benchmark, mode)
        for name in names
    }
    shared = sorted(set.intersection(*(set(arm) for arm in loaded.values())))
    if not shared:
        raise ValueError("the arms share no examples")

    scores = {
        name: np.asarray(
            [_metric_value(arm[index], benchmark) for index in shared]
        )
        for name, arm in loaded.items()
    }
    cluster_key = CLUSTER_KEYS.get(benchmark)
    clusters = (
        np.asarray([loaded[baseline][index][cluster_key] for index in shared])
        if cluster_key
        else None
    )
    summary: dict[str, Any] = {
        "benchmark": benchmark,
        "mode": mode,
        "baseline": baseline,
        "n_examples": len(shared),
        "n_clusters": (int(np.unique(clusters).size) if clusters is not None
                       else None),
        "cluster_key": cluster_key,
        "arms": {},
        "paired_vs_baseline": {},
    }
    interval_key = f"bootstrap_95_ci_seed{BOOTSTRAP_SEED}_{BOOTSTRAP_RESAMPLES}"
    for name, value in scores.items():
        low, high = paired_bootstrap(value, clusters=clusters)
        summary["arms"][name] = {
            "mean": float(value.mean()),
            "n": int(value.size),
            "complete": len(loaded[name]) == len(shared),
            interval_key: [low, high],
        }
    for name, value in scores.items():
        if name == baseline:
            continue
        delta = value - scores[baseline]
        low, high = paired_bootstrap(delta, clusters=clusters)
        units = _cluster_means(delta, clusters)
        se = float(units.std(ddof=1) / np.sqrt(units.size))
        summary["paired_vs_baseline"][name] = {
            "mean_delta": float(delta.mean()),
            "se_delta": se,
            "normal_95_ci": [
                float(delta.mean() - 1.96 * se),
                float(delta.mean() + 1.96 * se),
            ],
            f"bootstrap_95_ci_seed{BOOTSTRAP_SEED}_{BOOTSTRAP_RESAMPLES}": [
                low,
                high,
            ],
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
        }

    if benchmark == "longbench_pro":
        summary["by_token_length"] = {}
        buckets = sorted({
            str(loaded[baseline][identity].get("token_length"))
            for identity in shared
        }, key=lambda value: int(value[:-1]) if value.endswith("k") else value)
        for bucket in buckets:
            positions = np.asarray([
                index for index, identity in enumerate(shared)
                if str(loaded[baseline][identity].get("token_length")) == bucket
            ])
            bucket_scores = {
                name: values[positions] for name, values in scores.items()
            }
            bucket_summary = {
                "n": int(positions.size),
                "arms": {
                    name: {"mean": float(values.mean()), "n": int(values.size)}
                    for name, values in bucket_scores.items()
                },
                "paired_vs_baseline": {},
            }
            for name, values in bucket_scores.items():
                if name == baseline:
                    continue
                delta = values - bucket_scores[baseline]
                low, high = paired_bootstrap(delta)
                bucket_summary["paired_vs_baseline"][name] = {
                    "mean_delta": float(delta.mean()),
                    f"bootstrap_95_ci_seed{BOOTSTRAP_SEED}_{BOOTSTRAP_RESAMPLES}": [
                        low,
                        high,
                    ],
                    "wins": int((delta > 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum()),
                }
            summary["by_token_length"][bucket] = bucket_summary

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def _print(summary: dict[str, Any]) -> None:
    clustered = (
        f" over {summary['n_clusters']} {summary['cluster_key']}s"
        if summary.get("n_clusters")
        else ""
    )
    print(f"{summary['benchmark']} n={summary['n_examples']}{clustered}")
    for name, arm in summary["arms"].items():
        print(f"  {name:12s} {arm['mean']:.4f}")
    key = f"bootstrap_95_ci_seed{BOOTSTRAP_SEED}_{BOOTSTRAP_RESAMPLES}"
    for name, delta in summary["paired_vs_baseline"].items():
        low, high = delta[key]
        print(
            f"  {name:12s} vs {summary['baseline']}: "
            f"{delta['mean_delta']:+.5f} "
            f"boot95=[{low:+.5f}, {high:+.5f}] "
            f"W/L/T={delta['wins']}/{delta['losses']}/{delta['ties']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--benchmark", default="mrcr", choices=sorted(METRIC_KEYS))
    parser.add_argument("--mode", default=None,
                        help="the KV mode whose file to read in each arm "
                             "directory; omit when the arm is the mode and "
                             "each directory holds exactly one")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    _print(
        compare(
            args.root,
            args.benchmark,
            args.mode,
            baseline=args.baseline,
            arms=args.arms,
            output_path=args.output_path,
        )
    )


if __name__ == "__main__":
    main()
