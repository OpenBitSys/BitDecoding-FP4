"""Evaluate NVFP4 KV-cache quality on long-context, reasoning and code work.

For each requested KV mode a single in-process vLLM engine is built and every
benchmark is scored against it, then the engine is torn down and the next mode
loads.  Results stream to per-cell JSONL and a final ``summary.md`` /
``summary.json`` table (mean per cell + delta vs the BF16 baseline).

Modes: ``bitdecoding-nvfp4`` and ``bitdecoding-mxfp4`` (CUSTOM), ``bf16``, ``fp8``,
``turboquant-4bit-nc``, ``turboquant-4bit-nc-all-layers``, and ``nvfp4``
(official vLLM NVFP4).
BitDecoding-NVFP4 automatically calibrates key smoothing and probability scaling.

    MODEL_PATH=/path/to/Qwen3-4B-Thinking-2507 \\
    python -m evaluation.run_eval --modes bitdecoding-nvfp4 --limit 20 --out-dir _out/smoke
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import time
from typing import List

from . import (
    aime,
    aime25,
    config,
    gpqa,
    lcb,
    longbench_pro,
    mrcr,
    ruler,
)

_BENCHMARKS = {
    "longbench_pro": (longbench_pro, longbench_pro.METRIC_KEY, "LongBench-Pro F1"),
    "mrcr": (mrcr, mrcr.METRIC_KEY, "MRCR ratio"),
    "aime": (aime, aime.METRIC_KEY, "AIME 2024 Avg@32"),
    "aime25": (aime25, aime25.METRIC_KEY, "AIME 2025 Avg@32"),
    "lcb": (lcb, lcb.METRIC_KEY, "LiveCodeBench v6 pass@1"),
    "ruler_niah": (ruler, ruler.METRIC_KEY, "RULER NIAH recall"),
    "gpqa": (gpqa, gpqa.METRIC_KEY, "GPQA Diamond Avg@8"),
}


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _append_jsonl(path, records):
    """Durably append a completed scheduler group to a partial result."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path):
    records = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSONL checkpoint {path}:{line_number}: {error}"
                ) from error
    return records


def _partial_jsonl_path(path):
    stem, extension = os.path.splitext(path)
    return f"{stem}.partial{extension}"


_IDENTITY_KEYS = (
    "_source_index",
    "id",
    "bench_index",
    "task",
    "problem",
    "problem_index",
    "repeat",
    "token_length",
    "secondary_task",
    "bucket",
    "needle_name",
)


def _example_identity(example):
    return {key: example.get(key) for key in _IDENTITY_KEYS}


def _record_identity(record):
    identity = record.get("_selection_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("resumable result row lacks _selection_identity")
    return identity


def _load_result_prefix(path, prepared):
    """Load only an exact deterministic prefix of the selected shard."""
    records = _read_jsonl(path)
    if len(records) > len(prepared):
        raise RuntimeError(
            f"checkpoint {path} has {len(records)} rows for a "
            f"{len(prepared)}-row shard"
        )
    for index, record in enumerate(records):
        expected = _example_identity(prepared[index]["example"])
        observed = _record_identity(record)
        if observed != expected:
            raise RuntimeError(
                f"checkpoint {path} row {index} identity {observed!r} "
                f"does not match selected shard {expected!r}"
            )
    return records


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_or_validate_json(path, value):
    """Create an immutable manifest, or prove an existing one is identical."""
    if not os.path.exists(path):
        _write_json(path, value)
        return
    with open(path) as handle:
        existing = json.load(handle)
    if existing != value:
        raise RuntimeError(
            f"existing run manifest does not match the requested contract: {path}"
        )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path):
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    return {
        "path": os.path.realpath(path),
        "size": os.path.getsize(path),
        "sha256": _sha256(path),
    }


def _selection_sha256(examples_by_benchmark):
    """Digest the rows a run selected, not merely how many it selected.

    ``bench_index`` is what identifies a row in the benchmarks that carry no
    ``id``. Without it LiveCodeBench selections hash to the same
    value for any two selections of equal size, so a shard or ``--limit``
    mismatch would pass the resume check that exists to catch exactly that.
    """
    rows = [
        {
            "benchmark": benchmark,
            "_source_index": row.get("_source_index"),
            "id": row.get("id"),
            "bench_index": row.get("bench_index"),
            "task": row.get("task"),
            "problem": row.get("problem"),
            "problem_index": row.get("problem_index"),
            "repeat": row.get("repeat"),
            "bucket": row.get("bucket"),
            "token_length": row.get("token_length"),
            "secondary_task": row.get("secondary_task"),
        }
        for benchmark in sorted(examples_by_benchmark)
        for row in examples_by_benchmark[benchmark]
    ]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_seeds(prepared, base_seed: int):
    """One stable draw per selected row when a benchmark has no seed scheme."""
    seeds = []
    for item in prepared:
        payload = json.dumps(
            _example_identity(item["example"]),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
        seeds.append((base_seed + offset) % (2**31 - 1))
    return seeds


def _select_explicit_rows(examples, indices):
    """Select audited source rows without renumbering their identities or seeds."""
    if not indices or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("explicit selection must contain unique integer source indices")
    wanted = set(indices)
    selected = [row for row in examples if row["_source_index"] in wanted]
    if len(selected) != len(indices):
        raise ValueError("explicit selection references rows outside the selected protocol")
    return selected


def _frozen_prompt_audit(name, tokenizer, conversations, prepared, model_path, thinking):
    """Verify a schema-2 prompt freeze using the exact serving tokenizer."""
    path = os.environ.get("BITDECODING_LONG_CONTEXT_TOKEN_AUDIT")
    if not path:
        return None
    with open(path) as handle:
        audit = json.load(handle)
    if audit.get("schema") != 2:
        return None
    rows = {json.dumps(row["identity"], sort_keys=True): row
            for row in audit["rows"] if row["benchmark"] == name}
    selected = []
    for messages, item in zip(conversations, prepared, strict=True):
        expected = rows.get(json.dumps(_example_identity(item["example"]), sort_keys=True))
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, **config.chat_template_kwargs(model_path, thinking))
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        token_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        if expected is None or expected["token_sha256"] != token_hash:
            raise RuntimeError("prompt token IDs differ from the frozen evaluation input")
        if expected["rendered_sha256"] != hashlib.sha256(rendered.encode()).hexdigest():
            raise RuntimeError("rendered prompt differs from the frozen evaluation input")
        item["prompt_tokens"] = len(tokens)
        selected.append(expected)
    return selected

def _run_manifest(
    args, mode, mode_config, runner, examples_cache, selection_metadata
):
    audit_path = os.environ.get("BITDECODING_LONG_CONTEXT_TOKEN_AUDIT")
    engine = runner.engine_contract()
    if (mode == "bitdecoding-nvfp4" and getattr(args, "calibration_report", None)
            and config.RECIPE["underflow_aware_probability_scaling"]["enabled"]):
        tau = config.RECIPE["underflow_aware_probability_scaling"]["tau"]
        if engine.get("nvfp4_p_scale_tau") != tau:
            raise RuntimeError("the scoring engine did not load the calibrated tau")
    requested_rope = engine.get("requested_rope_scaling")
    resolved_rope = engine.get("resolved_rope_scaling")
    if requested_rope is not None:
        if not isinstance(resolved_rope, dict):
            raise RuntimeError("engine did not expose the requested YaRN config")
        requested_semantic = {
            "rope_type": requested_rope.get("rope_type", requested_rope.get("type")),
            "factor": float(requested_rope.get("factor")),
            "original_max_position_embeddings": int(
                requested_rope.get("original_max_position_embeddings")
            ),
        }
        resolved_semantic = {
            "rope_type": resolved_rope.get("rope_type", resolved_rope.get("type")),
            "factor": float(resolved_rope.get("factor")),
            "original_max_position_embeddings": int(
                resolved_rope.get("original_max_position_embeddings")
            ),
        }
        if resolved_semantic != requested_semantic:
            raise RuntimeError(
                f"resolved rope config {resolved_semantic} does not match "
                f"requested {requested_semantic}"
            )
    resolved_max_model_len = engine.get("resolved_max_model_len")
    if resolved_max_model_len is not None and resolved_max_model_len != args.max_model_len:
        raise RuntimeError(
            f"engine resolved max_model_len={resolved_max_model_len}, "
            f"requested {args.max_model_len}"
        )
    expected_cache_dtype = str(mode_config["kv_cache_dtype"])
    if engine.get("resolved_cache_dtype") != expected_cache_dtype:
        raise RuntimeError(
            f"engine resolved cache dtype {engine.get('resolved_cache_dtype')!r}, "
            f"but arm {mode!r} requires {expected_cache_dtype!r}"
        )
    layer_dtypes = engine.get("layer_cache_dtype_counts") or {}
    layer_backends = engine.get("layer_attention_backend_counts") or {}
    if not layer_dtypes or not layer_backends:
        raise RuntimeError(
            "engine did not expose per-layer KV dtype/backend attestation"
        )
    if mode in {"turboquant-4bit-nc", "turboquant-4bit-nc-all-layers"}:
        all_layers = bool(mode_config.get("turboquant_all_layers"))
        layer_count = _model_layer_count(args.model_path)
        expected_skip = [] if all_layers else [
            str(index) for index in (0, 1, layer_count - 2, layer_count - 1)
        ]
        if engine.get("kv_cache_dtype_skip_layers") != expected_skip:
            raise RuntimeError(
                "TurboQuant skip layers differ from the registered arm: "
                f"{engine.get('kv_cache_dtype_skip_layers')} != {expected_skip}"
            )
        expected_layer_dtypes = (
            {expected_cache_dtype: layer_count}
            if all_layers
            else {"auto": 4, expected_cache_dtype: layer_count - 4}
        )
        if layer_dtypes != expected_layer_dtypes:
            raise RuntimeError(
                f"TurboQuant layer KV dtypes are {layer_dtypes}, expected "
                f"{expected_layer_dtypes}"
            )
        expected_turboquant_layers = layer_count if all_layers else layer_count - 4
        expected_native_layers = 0 if all_layers else 4
        if (
            layer_backends.get("TURBOQUANT") != expected_turboquant_layers
            or sum(
                count for backend, count in layer_backends.items()
                if backend != "TURBOQUANT"
            ) != expected_native_layers
        ):
            raise RuntimeError(
                "TurboQuant resolved the wrong compressed/native layer split: "
                f"expected {expected_turboquant_layers}/"
                f"{expected_native_layers}, got {layer_backends}"
            )
    else:
        expected_layers = {expected_cache_dtype: sum(layer_dtypes.values())}
        if layer_dtypes != expected_layers:
            raise RuntimeError(
                f"arm {mode!r} resolved mixed layer KV dtypes {layer_dtypes}"
            )
        expected_backend = (
            "CUSTOM"
            if mode_config.get("custom_backend")
            else (mode_config.get("engine_kwargs") or {}).get(
                "attention_backend"
            )
        )
        if expected_backend is not None and layer_backends != {
            expected_backend: sum(layer_backends.values())
        }:
            raise RuntimeError(
                f"arm {mode!r} requires backend {expected_backend}, got "
                f"{layer_backends}"
            )
    return {
        "schema": 2,
        "mode": mode,
        "mode_label": mode_config["label"],
        "model": os.path.realpath(args.model_path),
        "model_layer_count": (
            _model_layer_count(args.model_path)
            if os.path.isfile(os.path.join(args.model_path, "config.json"))
            else None
        ),
        "model_revision": os.environ.get("BITDECODING_MODEL_REVISION"),
        "data_dir": os.path.realpath(args.data_dir),
        "dataset_manifest": _file_identity(
            os.path.join(args.data_dir, "MANIFEST.json")
        ),
        "ruler_generator": _file_identity(
            os.path.join(args.data_dir, "ruler", "generator.json")
        ),
        "benchmarks": list(args.benchmarks),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "scale_mode_override": getattr(args, "scale_mode", None),
        "thinking": args.thinking,
        "seed": args.seed,
        "sampling": {
            name: _sampling_for(
                name, args.thinking, mode, max_model_len=args.max_model_len,
                model_path=args.model_path,
            )
            for name in args.benchmarks
        },
        "selection_sha256": _selection_sha256(examples_cache),
        "selected_rows": {
            name: len(rows) for name, rows in examples_cache.items()
        },
        "selection_metadata": selection_metadata,
        "prompt_budget": int(os.environ.get(
            "BITDECODING_LONGBENCH_PROMPT_BUDGET", config.LONGBENCH_PROMPT_BUDGET
        )),
        "engine": engine,
        "recipe": config.RECIPE if mode_config.get("custom_backend") else None,
        "calibration": _file_identity(getattr(args, "calibration_report", None)),
        "mode_config": mode_config,
        "sidecars": {
            key: _file_identity(mode_config.get(key))
            for key in ("key_shift",)
            if mode_config.get(key) is not None
        },
        "token_audit": _file_identity(audit_path),
        "resumable_output": {
            "schema": 1,
            "unit": "completed_scheduler_group",
            "source_files": {
                name: _file_identity(os.path.join(os.path.dirname(__file__), name))
                for name in (
                    "config.py", "gpqa.py", "longbench_pro.py", "ruler.py",
                    "run_eval.py", "vllm_runner.py", "aime.py", "aime25.py",
                    "lcb.py", "mrcr.py", "scoring.py", "longbench_pro_metrics.py",
                )
            },
        },
    }


def _mean(records, key):
    vals = [float(r[key]) for r in records if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _validate_shard(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )


def _select_shard(examples, *, shard_index: int, num_shards: int):
    _validate_shard(shard_index, num_shards)
    return examples[shard_index::num_shards]


def _evenly_spaced(examples, limit: int):
    """Select deterministic coverage across an ordered benchmark."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(examples) <= limit:
        return list(examples)
    if limit == 1:
        return [examples[0]]
    return [
        examples[index * (len(examples) - 1) // (limit - 1)]
        for index in range(limit)
    ]


def _stratified_slice(name: str, examples, limit: int):
    """Select smoke rows spanning long-context length/problem coverage."""
    if name == "longbench_pro":
        buckets = {}
        for example in examples:
            buckets.setdefault(str(example.get("token_length")), []).append(example)
        ordered = sorted(
            buckets,
            key=lambda value: int(value[:-1]) if value.endswith("k") else value,
        )
        base, remainder = divmod(min(limit, len(examples)), len(ordered))
        selected = []
        for position, bucket in enumerate(ordered):
            take = base + (position < remainder)
            selected.extend(_evenly_spaced(buckets[bucket], take))
        # The official T4 scorer gate needs one English and one Chinese row,
        # while keeping the exact per-bucket smoke allocation unchanged.
        for language in ("English", "Chinese"):
            if any(
                str(row.get("secondary_task", "")).startswith("T4.")
                and row.get("language") == language
                for row in selected
            ):
                continue
            candidate = next(
                (
                    row for row in examples
                    if str(row.get("secondary_task", "")).startswith("T4.")
                    and row.get("language") == language
                ),
                None,
            )
            if candidate is None:
                continue
            bucket = str(candidate.get("token_length"))
            replace_at = next(
                (
                    index for index, row in enumerate(selected)
                    if str(row.get("token_length")) == bucket
                    and not str(row.get("secondary_task", "")).startswith("T4.")
                ),
                None,
            )
            if replace_at is not None:
                selected[replace_at] = candidate
        return selected
    if name == "mrcr":
        ordered = sorted(examples, key=lambda row: int(row["prompt_tokens"]))
        return _evenly_spaced(ordered, limit)
    if name == "ruler_niah":
        tasks = {}
        for example in examples:
            tasks.setdefault(example["task"], []).append(example)
        ordered = [task for task in ruler.TASKS if task in tasks]
        base, remainder = divmod(min(limit, len(examples)), len(ordered))
        selected = []
        for position, task in enumerate(ordered):
            selected.extend(
                _evenly_spaced(tasks[task], base + (position < remainder))
            )
        return selected
    return _evenly_spaced(examples, limit)


def _long_context_filter(name: str, examples, modes):
    if name != "longbench_pro":
        return examples
    examples = [
        row for row in examples
        if not str(row.get("secondary_task", "")).startswith("T4.")
    ]
    included_prefixes = tuple(
        value.strip() for value in
        os.environ.get("BITDECODING_LONG_CONTEXT_LBP_TASK_PREFIXES", "").split(",")
        if value.strip()
    )
    if included_prefixes:
        examples = [
            row for row in examples
            if str(row.get("secondary_task", "")).startswith(included_prefixes)
        ]
    excluded = {
        value.strip() for value in
        os.environ.get(
            "BITDECODING_LONG_CONTEXT_EXCLUDED_LBP_BUCKETS",
            ",".join(config.EVALUATION["longbench_excluded_buckets"]),
        ).split(",")
        if value.strip()
    }
    if not excluded:
        return examples
    return [row for row in examples if str(row.get("token_length")) not in excluded]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=config.MODEL_PATH)
    ap.add_argument("--data-dir", default=config.DATA_DIR)
    ap.add_argument("--selection-file", default=None,
                    help="JSON mapping benchmark names to frozen source-row indices")
    ap.add_argument("--out-dir", default="_out")
    ap.add_argument("--modes", nargs="+", default=["bitdecoding-nvfp4"],
                    choices=sorted(config.KV_MODES),
                    help="bitdecoding-nvfp4 and bitdecoding-mxfp4 use CUSTOM; bf16, fp8, "
                         "both turboquant-4bit-nc arms and official nvfp4 are "
                         "baselines")
    ap.add_argument("--benchmarks", nargs="+", default=list(_BENCHMARKS),
                    choices=list(_BENCHMARKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per benchmark (smoke runs)")
    ap.add_argument("--stratified-limit", type=int, default=None,
                    help="cap examples while spanning long-context length/problem coverage")
    ap.add_argument(
        "--resume", action="store_true",
        help="resume an exact partial shard at scheduler-group boundaries",
    )
    ap.add_argument("--shard-index", type=int, default=0,
                    help="zero-based deterministic example shard")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of deterministic example shards")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--distributed-executor-backend", choices=("mp", "ray"))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=config.MAX_MODEL_LEN)
    ap.add_argument("--enforce-eager", dest="enforce_eager",
                    action="store_true", default=None)
    ap.add_argument("--no-enforce-eager", dest="enforce_eager",
                    action="store_false")
    ap.add_argument("--attention-backend", default=None,
                    help="e.g. FLASHINFER (nvfp4/SM100) or TRITON_ATTN")
    ap.add_argument("--scale-mode", default=None,
                    help="CUSTOM NVFP4 block-scale candidates: a mode name "
                         "such as four_six for all three tensors, or "
                         "key=...,value=...,query=... to set them apart")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thinking", action="store_true",
                    help="run every benchmark in Qwen3 thinking mode, which "
                         "also switches LongBench-Pro and MRCR off greedy "
                         "(see config.THINKING_BENCH_OVERRIDES)")
    ap.add_argument("--score-longbench-t4", action="store_true",
                    help="post-score saved LongBench-Pro T4 generations in --out-dir")
    ap.add_argument("--embedding-model-path", default=None)
    ap.add_argument("--official-longbench-utils", default=None)
    ap.add_argument(
        "--official-evaluator-commit",
        default="a1d3fd6eab275981a75f03ebda2169b72b7b876e",
    )
    ap.add_argument(
        "--embedding-model-revision",
        default="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    )
    return ap


def _model_family(model_path=None):
    return config.model_family(model_path)


def _model_layer_count(model_path):
    try:
        with open(os.path.join(model_path, "config.json")) as handle:
            model_config = json.load(handle)
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read model config for layer attestation: {model_path}"
        ) from error
    for candidate in (model_config, model_config.get("text_config", {})):
        for key in ("num_hidden_layers", "num_layers", "n_layer"):
            if candidate.get(key) is not None:
                return int(candidate[key])
    raise RuntimeError(f"model config has no layer count: {model_path}")


def _sampling_for(name, thinking: bool, mode: str, max_model_len=None, model_path=None) -> dict:
    defaults = dict(config.BENCH_DEFAULTS[name])
    if thinking:
        defaults.update(config.THINKING_BENCH_OVERRIDES.get(name, {}))
    window = config.SHORT_CONTEXT_MAX_MODEL_LEN if max_model_len is None else max_model_len
    if window > config.SHORT_CONTEXT_MAX_MODEL_LEN:
        defaults.update(config.LONG_CONTEXT_BENCH_OVERRIDES.get(name, {}))
    if thinking:
        family = _model_family(model_path)
        defaults.update(config.MODEL_THINKING_SAMPLING[family])
        if family == "glm":
            defaults.pop("top_k", None)
    return defaults


def _budgets(defaults, prepared, max_model_len):
    """The generation budget: one number, or one per row filling the window."""
    want = defaults["max_new_tokens"]
    if want != config.FILL_WINDOW:
        return want
    budgets = []
    for item in prepared:
        prompt_tokens = item.get("prompt_tokens")
        if prompt_tokens is None:
            raise ValueError(
                "a benchmark asking to fill the window has to report "
                "prompt_tokens for every row"
            )
        budgets.append(
            max_model_len - prompt_tokens - config.GENERATION_MARGIN
        )
    if min(budgets) <= 0:
        raise ValueError(
            f"a prompt leaves no room to answer in a {max_model_len} "
            f"window (smallest budget {min(budgets)})"
        )
    return budgets


def _templated_prompt_lengths(tokenizer, conversations, thinking, model_path=None):
    """Count the exact chat-template tokens used by a long-window engine."""
    lengths = []
    for messages in conversations:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            **config.chat_template_kwargs(model_path, thinking),
        }
        try:
            value = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            value = tokenizer.apply_chat_template(messages, **kwargs)
        # ``apply_chat_template(tokenize=True)`` may return a Transformers
        # BatchEncoding.  ``len(value)`` then counts fields rather than token
        # IDs, typically producing 2 and giving fill-window generation an
        # invalid budget.  Unwrap the actual input-id sequence first.
        if hasattr(value, "input_ids"):
            value = value.input_ids
        elif isinstance(value, dict) and "input_ids" in value:
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        lengths.append(len(value))
    return lengths


def _verify_frozen_completions(audited_rows, completions):
    if audited_rows is None:
        return
    for expected, completion in zip(audited_rows, completions, strict=True):
        if (completion.get("prompt_tokens") != expected["prompt_tokens"]
                or completion.get("prompt_token_sha256") != expected["token_sha256"]):
            raise RuntimeError("engine consumed prompt token IDs that differ from the input freeze")


def _run_benchmark(name, runner, examples_cache, args, mode, *, output_path=None):
    module, metric_key, _label = _BENCHMARKS[name]
    thinking = args.thinking
    defaults = _sampling_for(
        name, args.thinking, mode, max_model_len=args.max_model_len,
        model_path=getattr(args, "model_path", None),
    )

    if name == "longbench_pro":
        conversations, prepared = module.build(
            examples_cache[name], tokenize=runner.tokenize,
            detokenize=runner.detokenize, thinking=thinking,
            reject_truncation=False,
        )
    else:
        conversations, prepared = module.build(examples_cache[name])

    # GLM's tokenizer is not Qwen's tokenizer.  The staged RULER/MRCR
    # ``prompt_tokens`` are Qwen-side measurements, so using them to fill a
    # GLM window would silently give GLM the wrong answer budget.  Count the
    # exact template that the serving tokenizer will consume after the engine
    # is built and retain those lengths in the manifest/results.
    if (
        name in {"ruler_niah", "mrcr"}
        and _model_family(getattr(args, "model_path", None)) == "glm"
    ):
        for item, prompt_tokens in zip(
            prepared,
            _templated_prompt_lengths(runner.tokenizer, conversations, thinking,
                                      getattr(args, "model_path", None)),
            strict=True,
        ):
            item["prompt_tokens"] = prompt_tokens
    frozen = _frozen_prompt_audit(name, getattr(runner, "tokenizer", None), conversations, prepared,
                                  getattr(args, "model_path", None), thinking)

    # A benchmark that samples the same prompt more than once has to say so;
    # one seed for such a batch makes the repeats identical.
    seed = (
        module.seeds(prepared, args.seed)
        if hasattr(module, "seeds")
        else _default_seeds(prepared, args.seed)
    )
    budget = _budgets(defaults, prepared, args.max_model_len)
    if frozen is not None:
        for i, row in enumerate(frozen):
            row_budget = budget[i] if isinstance(budget, list) else budget
            if row["sampling_seed"] != seed[i] or row["generation_budget"] != row_budget:
                raise RuntimeError("sampling seed or generation budget differs from the freeze")
    def attach_identity(records, selected_prepared, selected_seeds, selected_completions):
        row_seeds = (
            list(selected_seeds)
            if isinstance(selected_seeds, (list, tuple))
            else [selected_seeds] * len(selected_prepared)
        )
        for record, item, row_seed, completion in zip(
            records, selected_prepared, row_seeds, selected_completions, strict=True
        ):
            record["_selection_identity"] = _example_identity(item["example"])
            record["_sampling_seed"] = int(row_seed)
            if "output_token_ids" in completion:
                record["output_token_ids"] = completion["output_token_ids"]
            if "prompt_token_sha256" in completion:
                record["prompt_token_sha256"] = completion["prompt_token_sha256"]
        return records

    resumable = args.resume and output_path is not None
    if resumable:
        partial_path = _partial_jsonl_path(output_path)
        if os.path.exists(output_path):
            if os.path.exists(partial_path):
                raise RuntimeError(
                    f"both final and partial results exist: {output_path}, "
                    f"{partial_path}"
                )
            records = _load_result_prefix(output_path, prepared)
            if len(records) != len(prepared):
                raise RuntimeError(
                    f"final result {output_path} has {len(records)}/"
                    f"{len(prepared)} rows"
                )
            print(f"  [resume] {name}: final result already complete", flush=True)
            return records, metric_key

        records = records if os.path.exists(output_path) else (
            _load_result_prefix(partial_path, prepared)
            if os.path.exists(partial_path) else []
        )
        completed = len(records)
        if completed:
            print(
                f"  [resume] {name}: validated {completed}/{len(prepared)} "
                f"checkpoint rows",
                flush=True,
            )
        batch_size = int(
            getattr(runner, "_request_batch_size", None)
            or len(conversations) or 1
        )

        def selected(value, start, stop):
            return value[start:stop] if isinstance(value, (list, tuple)) else value

        for start in range(completed, len(prepared), batch_size):
            stop = min(start + batch_size, len(prepared))
            completions = runner.chat(
                conversations[start:stop],
                temperature=defaults["temperature"], top_p=defaults["top_p"],
                top_k=defaults.get("top_k"), min_p=defaults.get("min_p"),
                max_new_tokens=selected(budget, start, stop),
                seed=selected(seed, start, stop),
                enable_thinking=True if args.thinking else None,
            )
            _verify_frozen_completions(frozen[start:stop] if frozen is not None else None, completions)
            batch_records = attach_identity(
                module.grade(prepared[start:stop], completions),
                prepared[start:stop],
                selected(seed, start, stop),
                completions,
            )
            if len(batch_records) != stop - start:
                raise RuntimeError(
                    f"{name} graded {len(batch_records)} rows for a "
                    f"{stop - start}-row scheduler group"
                )
            _append_jsonl(partial_path, batch_records)
            records.extend(batch_records)
            print(
                f"  [checkpoint] {name}: {len(records)}/{len(prepared)} rows",
                flush=True,
            )
        if os.path.exists(partial_path):
            os.replace(partial_path, output_path)
        elif not os.path.exists(output_path):
            _write_jsonl(output_path, [])
    else:
        completions = runner.chat(
            conversations,
            temperature=defaults["temperature"], top_p=defaults["top_p"],
            top_k=defaults.get("top_k"), min_p=defaults.get("min_p"),
            max_new_tokens=budget, seed=seed,
            enable_thinking=True if args.thinking else None,
        )
        _verify_frozen_completions(frozen, completions)
        records = attach_identity(module.grade(prepared, completions), prepared, seed, completions)
    truncated = sum(1 for r in records if r.get("finish_reason") == "length")
    if truncated:
        # Loud because it is the failure mode that looks like a result: every
        # arm truncating at once reads as "the cache does not matter here".
        described = (f"{min(budget)}..{max(budget)}"
                     if isinstance(budget, list) else budget)
        print(
            f"  [truncation] {name}: {truncated}/{len(records)} hit "
            f"max_new_tokens={described}",
            flush=True,
        )
    return records, metric_key


def _prepare_calibration(args, scale_mode=None):
    """Fit the key mean and tau before the scoring process imports the kernel."""
    modes = set(args.modes) & {"bitdecoding-nvfp4", "bitdecoding-mxfp4"}
    if not modes:
        return
    fit_tau = "bitdecoding-nvfp4" in modes and config.needs_tau_calibration(config.RECIPE)
    smooth = config.RECIPE["outlier_aware_key_quantization"]["key_smoothing"]
    if not fit_tau and (not smooth or all(
        os.path.isfile(config.KV_MODES[mode].get("key_shift") or "") for mode in modes
    )):
        return

    directory = Path(args.out_dir).resolve() / "calibration"
    sidecar = directory / "key_smoothing.safetensors"
    report_path = sidecar.with_suffix(".json")
    mode = config.KV_MODES["bitdecoding-nvfp4" if fit_tau else sorted(modes)[0]]
    scales = scale_mode if scale_mode is not None else mode["scale_mode"]
    additional = {
        "nvfp4_scale_mode": scales, "nvfp4_query_terms": mode["query_terms"],
        "nvfp4_rotation": mode["rotation"], "nvfp4_key_smoothing": None,
    }
    expected = {
        "model_path": os.path.realpath(args.model_path),
        "max_model_len": args.max_model_len, "context_tokens": args.max_model_len - 1,
        "tensor_parallel_size": args.tp_size, "additional_config": additional,
    }
    if not report_path.exists():
        if sidecar.exists():
            raise ValueError(f"incomplete calibration at {sidecar}; choose a new output directory")
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(Path(config.REPO_ROOT) / "tools/calibrate_key_smooth_sidecar.py"),
            "--model-path", args.model_path, "--output-path", str(sidecar),
            "--max-model-len", str(args.max_model_len),
            "--context-tokens", str(args.max_model_len - 1),
            "--tensor-parallel-size", str(args.tp_size), "--max-num-seqs", "1",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--scale-mode", json.dumps(scales),
            "--query-terms", str(mode["query_terms"]), "--rotation", mode["rotation"],
        ]
        environment = os.environ.copy()
        if fit_tau:
            command += ["--probability-multiplier", str(config.RECIPE["underflow_aware_probability_scaling"]["c"])]
            probability = config.probability_environment(config.RECIPE, calibration=True)
        else:
            probability = {"NVFP4_P_SCALE_MODE": None, "NVFP4_P_SCALE_TAU": None}
        for key, value in {**probability, "NVFP4_P_SCALE_DIVISOR": None}.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        environment["PYTHONPATH"] = os.pathsep.join([
            str(Path(config.REPO_ROOT) / "src"), config.REPO_ROOT,
            environment.get("PYTHONPATH", ""),
        ])
        print("Calibrating key smoothing and probability scaling before evaluation...", flush=True)
        subprocess.run(command, env=environment, check=True)

    report = json.loads(report_path.read_text())
    if not sidecar.is_file() or any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"calibration does not match this model and recipe: {report_path}")
    if fit_tau:
        probability = report.get("probability_scaling", {})
        resolved = {**config.RECIPE, "underflow_aware_probability_scaling": {
            **config.RECIPE["underflow_aware_probability_scaling"], "tau": probability.get("tau"),
        }}
        config.probability_environment(resolved)
        if probability.get("c") != config.RECIPE["underflow_aware_probability_scaling"]["c"]:
            raise ValueError(f"calibration probability multiplier differs from the recipe: {report_path}")
        config.RECIPE["underflow_aware_probability_scaling"]["tau"] = probability["tau"]
        print(f"Calibrated tau={probability['tau']:.9g}", flush=True)
    if smooth:
        for name in modes:
            config.KV_MODES[name]["key_shift"] = str(sidecar)
    args.calibration_report = str(report_path)


def _configure_probability_scaling(mode):
    if mode == "bitdecoding-nvfp4":
        environment = config.probability_environment(config.RECIPE)
    elif mode == "bitdecoding-mxfp4":
        environment = {"NVFP4_P_SCALE_MODE": None, "NVFP4_P_SCALE_TAU": None}
    else:
        return
    environment["NVFP4_P_SCALE_DIVISOR"] = None
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def main():
    args = build_parser().parse_args()
    if args.score_longbench_t4:
        if not args.embedding_model_path or not args.official_longbench_utils:
            raise SystemExit(
                "--score-longbench-t4 requires --embedding-model-path and "
                "--official-longbench-utils"
            )
        manifest = longbench_pro.score_t4_run(
            args.out_dir,
            embedding_model_path=args.embedding_model_path,
            official_utils_path=args.official_longbench_utils,
            evaluator_commit=args.official_evaluator_commit,
            embedding_revision=args.embedding_model_revision,
        )
        print(
            f"[T4] scored {manifest['t4_rows']}/{manifest['rows']} rows "
            f"under {args.out_dir}",
            flush=True,
        )
        return
    try:
        _validate_shard(args.shard_index, args.num_shards)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.limit is not None and args.stratified_limit is not None:
        raise SystemExit("--limit and --stratified-limit are mutually exclusive")
    if args.stratified_limit is not None and args.stratified_limit <= 0:
        raise SystemExit("--stratified-limit must be positive")
    scale_mode = None
    if args.scale_mode is not None:
        from nvfp4_decode_kernel.scale_mode import parse_scale_mode_spec

        try:
            scale_mode = parse_scale_mode_spec(args.scale_mode)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    config.MODEL_PATH = args.model_path
    config.configure_data_dir(args.data_dir)

    problems = config.verify_prestaging(args.benchmarks)
    problems += config.verify_dependencies(args.benchmarks)
    if problems:
        raise SystemExit("Pre-staging incomplete:\n- " + "\n- ".join(problems))
    _prepare_calibration(args, scale_mode)
    problems = config.verify_sidecars(args.modes)
    if problems:
        raise SystemExit("Calibration incomplete:\n- " + "\n- ".join(problems))
    if "bitdecoding-nvfp4" in args.modes:
        _configure_probability_scaling("bitdecoding-nvfp4")

    # Load each dataset once (shared across modes).
    examples_cache = {}
    selection_metadata = {}
    for name in args.benchmarks:
        module = _BENCHMARKS[name][0]
        pre_sharded = False
        if (
            name == "ruler_niah"
            and args.limit is None
            and not args.selection_file
        ):
            ex = module.load_examples(
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                stratified_limit=args.stratified_limit,
            )
            pre_sharded = True
        else:
            ex = module.load_examples()
        for source_index, row in enumerate(ex):
            row.setdefault("_source_index", source_index)
        ex = _long_context_filter(name, ex, args.modes)
        if args.selection_file:
            if args.limit or args.stratified_limit or pre_sharded:
                raise ValueError("explicit selection cannot be combined with a limit or pre-sharded data")
            with open(args.selection_file) as handle:
                indices = json.load(handle)[name]
            ex = _select_explicit_rows(ex, indices)
            selection_metadata.setdefault(name, {})["selection_file"] = _file_identity(args.selection_file)
        if args.limit:
            ex = ex[: args.limit]
        elif args.stratified_limit and not pre_sharded:
            ex = _stratified_slice(name, ex, args.stratified_limit)
        if not pre_sharded:
            ex = _select_shard(
                ex,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        examples_cache[name] = ex
        print(
            f"[data] {name}: {len(ex)} examples "
            f"(shard {args.shard_index}/{args.num_shards})",
            flush=True,
        )

    results = {}  # (benchmark, mode) -> (mean, n)
    for mode in args.modes:
        _configure_probability_scaling(mode)
        mode_config = config.resolve_mode(mode, args.model_path)
        kv_dtype = mode_config["kv_cache_dtype"]
        print(f"\n=== KV mode: {mode} ({mode_config['label']}) "
              f"kv_cache_dtype={kv_dtype} ===", flush=True)
        from .vllm_runner import VllmRunner

        engine_started = time.monotonic()
        runner = VllmRunner(
            model_path=args.model_path, kv_cache_dtype=mode_config,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend=args.distributed_executor_backend,
            enforce_eager=args.enforce_eager,
            attention_backend=args.attention_backend,
            scale_mode=scale_mode,
        )
        _append_jsonl(os.path.join(args.out_dir, f"engine_startup__{mode}.jsonl"), [{
            "mode": mode, "benchmarks": args.benchmarks,
            "host": os.uname().nodename, "pid": os.getpid(),
            "started_unix": time.time() - (time.monotonic() - engine_started),
            "engine_startup_seconds": time.monotonic() - engine_started,
            "includes_tokenizer_load_and_graph_initialization": True,
            "model_path": os.path.realpath(args.model_path),
            "tensor_parallel_size": args.tp_size,
        }])
        try:
            manifest_name = (
                "manifest__" + "_".join(args.benchmarks) + f"__{mode}.json"
            )
            _write_or_validate_json(
                os.path.join(args.out_dir, manifest_name),
                _run_manifest(
                    args,
                    mode,
                    mode_config,
                    runner,
                    examples_cache,
                    selection_metadata,
                ),
            )
            for name in args.benchmarks:
                output_path = os.path.join(
                    args.out_dir, f"{name}__{mode}.jsonl"
                )
                health_path = output_path.removesuffix(".jsonl") + ".health.json"
                preserve_health = os.path.isfile(output_path) and os.path.isfile(
                    health_path
                )
                try:
                    import torch

                    torch.cuda.reset_peak_memory_stats()
                except (ImportError, RuntimeError):
                    torch = None
                started = time.monotonic()
                records, metric_key = _run_benchmark(
                    name, runner, examples_cache, args, mode,
                    output_path=output_path,
                )
                if not args.resume:
                    _write_jsonl(output_path, records)
                if not preserve_health:
                    max_gpu_memory = (
                        torch.cuda.max_memory_reserved() / 2**20
                        if torch is not None and torch.cuda.is_available()
                        else None
                    )
                    elapsed = time.monotonic() - started
                    _write_json(health_path, {
                        "schema": 1,
                        "benchmark": name,
                        "mode": mode,
                        "n": len(records),
                        "wall_clock_s": elapsed,
                        "max_gpu_mem_mib": max_gpu_memory,
                        "generated_tokens": sum(
                            int(record.get("gen_tokens") or 0)
                            for record in records
                        ),
                        "finished_on_length": sum(
                            record.get("finish_reason") == "length"
                            for record in records
                        ),
                    })
                mean = _mean(records, metric_key)
                results[(name, mode)] = (mean, len(records))
                mean_str = "n/a (deferred)" if mean is None else f"{mean:.4f}"
                print(f"  {name}: mean {metric_key} = {mean_str} (n={len(records)})",
                      flush=True)
        finally:
            runner.shutdown()
            del runner
            gc.collect()

    _write_summary(args, results)


def _write_summary(args, results):
    benchmarks = args.benchmarks
    header = ["KV mode"]
    for b in benchmarks:
        header += [b, f"{b} Δ vs bf16"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    base = {b: (results.get((b, "bf16")) or (None, 0))[0] for b in benchmarks}
    summary_json = {"results": {}}
    for mode in args.modes:
        row = [config.KV_MODES[mode]["label"]]
        for b in benchmarks:
            mean, n = results.get((b, mode), (None, 0))
            summary_json["results"][f"{b}__{mode}"] = {"mean": mean, "n": n}
            if mean is None:
                row += ["—", "—"]
                continue
            row.append(f"{mean:.4f}")
            bm = base.get(b)
            row.append(f"{mean - bm:+.4f}" if bm is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# BitDecoding-FP4 — NVFP4 KV-cache evaluation\n\n")
        f.write(f"Model: `{args.model_path}`\n\n")
        f.write(table + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
    print("\n" + table)
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
