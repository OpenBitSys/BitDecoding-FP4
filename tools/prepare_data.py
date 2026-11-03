#!/usr/bin/env python3
"""Stage every dataset BitDecoding-FP4 reads, from public sources only.

Nothing under ``datasets/`` is committed except ``MANIFEST.json``; this script
rebuilds the rest.

  python tools/prepare_data.py --steps eval
  python tools/prepare_data.py --steps long-context
  python tools/prepare_data.py --steps verify
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets")
MANIFEST_PATH = os.path.join(DATASETS_ROOT, "MANIFEST.json")
LONGBENCH_PRO_DIR = os.path.join(DATASETS_ROOT, "longbench_pro")
MRCR_DIR = os.path.join(DATASETS_ROOT, "mrcr")

LONGBENCH_PRO_REPO = "caskcsg/LongBench-Pro"
MRCR_REPO = "openai/mrcr"
AIME24_REPO = "HuggingFaceH4/aime_2024"
AIME25_REPO = "math-ai/aime25"
LONG_CONTEXT_MODEL_REPO = "Qwen/Qwen3-4B-Thinking-2507"
LONG_CONTEXT_REFERENCE_TOKENIZER_REPO = "Qwen/Qwen3-8B"
LONG_CONTEXT_MODEL_DIR = os.path.join(
    REPO_ROOT, "_local", "models", "Qwen3-4B-Thinking-2507"
)
LONG_CONTEXT_REFERENCE_TOKENIZER_DIR = os.path.join(
    REPO_ROOT, "_local", "models", "Qwen3-8B-tokenizer"
)
LONG_CONTEXT_MRCR_BUDGET = 229_120

RULER_SOURCE = os.environ.get(
    "BITDECODING_RULER_SOURCE", os.path.join(REPO_ROOT, "_local", "ruler_official")
)
RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
)
RULER_MAX_SEQ_LENGTH = 122_880
RULER_NUM_SAMPLES = 500
RULER_SEED = 42
RULER_NUM_SHARDS = 8
RULER_SMOKE_LIMIT = 20

NEEDLE_GROUPS = (2, 4, 8)
STEPS = (
    "long-context-model", "longbench-pro", "mrcr", "long-context-mrcr",
    "aime24", "aime25", "gpqa", "lcb", "ruler", "verify",
    "long-context-verify",
)
EVAL_STEPS = (
    "longbench-pro", "mrcr", "aime24", "aime25", "lcb", "ruler", "gpqa",
    "verify",
)
LONG_CONTEXT_STEPS = (
    "long-context-model", "longbench-pro", "long-context-mrcr",
    "aime24", "aime25", "lcb", "gpqa", "ruler",
    "long-context-verify",
)


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def _norm(obj):
    """Canonicalise a parquet cell so the digest ignores container types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [_norm(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_norm(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if obj is not None and not isinstance(obj, (str, int, float, bool)):
        return str(obj)
    return obj


def content_hash_rows(rows) -> str:
    """Order-sensitive digest over row dicts (writer/version independent)."""
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(_norm(row), ensure_ascii=False, sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def content_hash_parquet(path: str) -> str:
    import pandas as pd

    df = pd.read_parquet(path)
    return content_hash_rows(df.to_dict(orient="records"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _expected(manifest: dict, rel_path: str) -> dict:
    return manifest["derived"].get(rel_path, {})


# --------------------------------------------------------------------------- #
# public downloads
# --------------------------------------------------------------------------- #
def step_longbench_pro(force: bool = False) -> None:
    from datasets import load_dataset

    target = LONGBENCH_PRO_DIR
    if os.path.isfile(os.path.join(target, "dataset_info.json")) and not force:
        print(f"[longbench-pro] present, skipping ({target})")
        return
    print(f"[longbench-pro] downloading {LONGBENCH_PRO_REPO} -> {target}")
    # Upstream ships one split and has renamed it: it was ``train`` when this
    # was written and is ``test`` now. Both are the same single split, so the
    # name is taken from the download rather than asserted.
    splits = load_dataset(
        LONGBENCH_PRO_REPO,
        revision=load_manifest()["upstream"][LONGBENCH_PRO_REPO]["revision"],
    )
    if len(splits) != 1:
        raise SystemExit(
            f"[longbench-pro] expected one split, got {sorted(splits)}"
        )
    (name,) = list(splits)
    print(f"[longbench-pro] split {name!r}, {len(splits[name])} rows")
    splits[name].save_to_disk(target)


def _tokenizer_vocab_sha256(tokenizer) -> str:
    payload = json.dumps(
        sorted(tokenizer.get_vocab().items()), ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_long_context_tokenizer(model_path: str = LONG_CONTEXT_MODEL_DIR) -> str:
    """Assert that the new checkpoint keeps the Qwen3-8B MRCR vocabulary."""
    from transformers import AutoTokenizer

    old = AutoTokenizer.from_pretrained(
        LONG_CONTEXT_REFERENCE_TOKENIZER_DIR, trust_remote_code=True,
        local_files_only=True,
    )
    new = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    old_hash = _tokenizer_vocab_sha256(old)
    new_hash = _tokenizer_vocab_sha256(new)
    if old_hash != new_hash:
        raise SystemExit(
            "[long-context-tokenizer] Qwen3-4B-Thinking-2507 vocabulary differs "
            f"from Qwen3-8B: {new_hash} != {old_hash}"
        )
    print(f"[long-context-tokenizer] vocabulary match: {new_hash}")
    return new_hash


def step_long_context_model(force: bool = False) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    config_path = os.path.join(LONG_CONTEXT_MODEL_DIR, "config.json")
    reference_config = os.path.join(
        LONG_CONTEXT_REFERENCE_TOKENIZER_DIR, "tokenizer_config.json"
    )
    if not os.path.isfile(reference_config):
        try:
            reference = AutoTokenizer.from_pretrained(
                LONG_CONTEXT_REFERENCE_TOKENIZER_REPO, local_files_only=True,
                trust_remote_code=True,
            )
        except OSError:
            reference = AutoTokenizer.from_pretrained(
                LONG_CONTEXT_REFERENCE_TOKENIZER_REPO, trust_remote_code=True,
            )
        os.makedirs(LONG_CONTEXT_REFERENCE_TOKENIZER_DIR, exist_ok=True)
        reference.save_pretrained(LONG_CONTEXT_REFERENCE_TOKENIZER_DIR)
    if os.path.isfile(config_path) and not force:
        print(f"[long-context-model] present, skipping ({LONG_CONTEXT_MODEL_DIR})")
    else:
        print(f"[long-context-model] downloading {LONG_CONTEXT_MODEL_REPO} -> {LONG_CONTEXT_MODEL_DIR}")
        snapshot_download(
            LONG_CONTEXT_MODEL_REPO, local_dir=LONG_CONTEXT_MODEL_DIR,
            revision=load_manifest()["upstream"][LONG_CONTEXT_MODEL_REPO]["revision"],
        )
    verify_long_context_tokenizer()


def step_long_context_mrcr(force: bool = False, budget: int = LONG_CONTEXT_MRCR_BUDGET) -> None:
    from huggingface_hub import snapshot_download

    target = MRCR_DIR
    snapshot_download(
        MRCR_REPO,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=[f"{n}needle/*" for n in NEEDLE_GROUPS],
    )
    filename = f"_filtered_le{budget}.jsonl"
    rel = f"datasets/mrcr/{filename}"
    build_mrcr_cache(
        LONG_CONTEXT_MODEL_DIR,
        force=force,
        budget=budget,
        out_path=os.path.join(target, filename),
        manifest_key=rel,
    )


def step_mrcr(tokenizer: str, force: bool = False) -> None:
    from huggingface_hub import snapshot_download

    target = MRCR_DIR
    print(f"[mrcr] downloading {MRCR_REPO} shards -> {target}  (~1.4 GB)")
    snapshot_download(
        MRCR_REPO,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=[f"{n}needle/*" for n in NEEDLE_GROUPS],
    )
    build_mrcr_cache(tokenizer, force=force)
    build_mrcr_cache(
        tokenizer, force=force, budget=98048,
        out_path=os.path.join(MRCR_DIR, "_filtered_le98048.jsonl"),
        manifest_key="datasets/mrcr/_filtered_le98048.jsonl",
    )


def build_mrcr_cache(
    tokenizer: str,
    force: bool = False,
    *,
    budget: int | None = None,
    out_path: str | None = None,
    manifest_key: str | None = None,
) -> None:
    """Keep MRCR rows whose prompt is <= max_prompt_tokens Qwen3 tokens."""
    import pandas as pd
    from transformers import AutoTokenizer

    manifest = load_manifest()
    manifest_key = (
        manifest_key or "datasets/mrcr/_filtered_le34000.jsonl"
    )
    spec = _expected(manifest, manifest_key)
    budget = budget if budget is not None else spec.get("max_prompt_tokens", 34000)
    out_path = out_path or os.path.join(MRCR_DIR, "_filtered_le34000.jsonl")
    if os.path.isfile(out_path) and not force:
        print(f"[mrcr-cache] present, skipping ({out_path})")
        return

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = []
    for n in NEEDLE_GROUPS:
        shards = sorted(glob.glob(os.path.join(MRCR_DIR, f"{n}needle", "*.parquet")))
        if not shards:
            raise SystemExit(f"[mrcr-cache] no shards for {n}needle — run --steps mrcr first")
        for shard in shards:
            df = pd.read_parquet(shard)
            for _, r in df.iterrows():
                messages = json.loads(r["prompt"])
                encoded = tok([m["content"] for m in messages], add_special_tokens=False)
                n_tokens = sum(len(ids) for ids in encoded["input_ids"])
                if n_tokens > budget:
                    continue
                rows.append({
                    "id": None,
                    "n_needles": n,
                    "messages": messages,
                    "answer": r["answer"],
                    "random_string_to_prepend": r["random_string_to_prepend"],
                    "prompt_tokens": n_tokens,
                })
        print(f"[mrcr-cache] {n}needle: kept {sum(1 for x in rows if x['n_needles'] == n)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[mrcr-cache] wrote {len(rows)} rows -> {out_path}")
    _report("mrcr cache", sha256_file(out_path), spec.get("sha256"))
    expected_rows = spec.get("n_rows")
    if expected_rows is not None and len(rows) != expected_rows:
        raise SystemExit(
            f"[mrcr-cache] row-count mismatch: {len(rows)} != {expected_rows}"
        )


def step_aime25(force: bool = False) -> None:
    """AIME 2025, 30 problems, one row each.

    The 32 repeats Avg@32 needs are made at load time by ``evaluation.aime25`` rather
    than written out, so this file stays the upstream set and cannot drift from
    the constant that decides how many there are.
    """
    from datasets import load_dataset
    import pandas as pd

    out_path = os.path.join(DATASETS_ROOT, "aime25", "aime25.parquet")
    if os.path.isfile(out_path) and not force:
        print(f"[aime25] present, skipping ({out_path})")
        return
    splits = load_dataset(
        AIME25_REPO, revision=load_manifest()["upstream"][AIME25_REPO]["revision"],
    )
    if len(splits) != 1:
        raise SystemExit(f"[aime25] expected one split, got {sorted(splits)}")
    (name,) = list(splits)
    frame = splits[name].to_pandas()[["id", "problem", "answer"]]
    if len(frame) != 30:
        raise SystemExit(f"[aime25] expected 30 problems, got {len(frame)}")
    non_integer = [a for a in frame["answer"] if not str(a).strip().isdigit()]
    if non_integer:
        raise SystemExit(f"[aime25] non-integer answers: {non_integer[:5]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(f"[aime25] wrote {len(frame)} problems -> {out_path}")



def step_aime24(force: bool = False) -> None:
    """AIME 2024, 30 problems. Avg@32 repeats are made at load time."""
    from datasets import load_dataset

    out_path = os.path.join(DATASETS_ROOT, "aime24", "aime24.parquet")
    if os.path.isfile(out_path) and not force:
        print(f"[aime24] present, skipping ({out_path})")
        return
    splits = load_dataset(
        AIME24_REPO, revision=load_manifest()["upstream"][AIME24_REPO]["revision"],
    )
    if len(splits) != 1:
        raise SystemExit(f"[aime24] expected one split, got {sorted(splits)}")
    (name,) = list(splits)
    frame = splits[name].to_pandas()
    if not {"problem", "answer"}.issubset(frame.columns):
        raise SystemExit(f"[aime24] missing problem/answer columns: {list(frame.columns)}")
    frame = frame[["problem", "answer"]]
    if len(frame) != 30:
        raise SystemExit(f"[aime24] expected 30 problems, got {len(frame)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(f"[aime24] wrote {len(frame)} problems -> {out_path}")


def step_gpqa(force: bool = False) -> None:
    """Stage the 198-question GPQA Diamond CSV used by simple-evals."""
    import urllib.request
    import pandas as pd

    out_path = os.path.join(DATASETS_ROOT, "gpqa", "gpqa_diamond.parquet")
    if os.path.isfile(out_path) and not force:
        print(f"[gpqa] present, skipping ({out_path})")
        return
    url = "https://raw.githubusercontent.com/openai/simple-evals/main/data/gpqa_diamond.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    csv_path = out_path.removesuffix(".parquet") + ".csv"
    urllib.request.urlretrieve(url, csv_path)
    frame = pd.read_csv(csv_path)
    if len(frame) != 198:
        raise SystemExit(f"[gpqa] expected 198 Diamond rows, got {len(frame)}")
    frame.to_parquet(out_path, index=False)
    os.remove(csv_path)
    print(f"[gpqa] wrote {len(frame)} questions -> {out_path}")


def step_lcb(force: bool = False) -> None:
    """Stage the v6 prompts and the official grader checkout."""
    from tools.lcb_data import LCB_CHECKOUT, PROMPTS_PATH, export_prompts

    if not os.path.isdir(os.path.join(LCB_CHECKOUT, "lcb_runner")):
        os.makedirs(os.path.dirname(LCB_CHECKOUT), exist_ok=True)
        subprocess.run([
            "git", "clone", "https://github.com/LiveCodeBench/LiveCodeBench.git",
            LCB_CHECKOUT,
        ], check=True)
    if os.path.isfile(PROMPTS_PATH) and not force:
        print(f"[lcb] present, skipping ({PROMPTS_PATH})")
        return
    count = export_prompts()
    if count != 175:
        raise SystemExit(f"[lcb] expected 175 v6 problems, got {count}")
    print(f"[lcb] wrote {count} prompts -> {PROMPTS_PATH}")


def step_ruler(force: bool = False, tokenizer_path: str = LONG_CONTEXT_MODEL_REPO) -> None:
    """Generate the eight official RULER NIAH tasks at 122,880 tokens."""
    prepare = os.path.join(RULER_SOURCE, "scripts", "data", "prepare.py")
    revision_path = os.path.join(RULER_SOURCE, "REVISION")
    if not os.path.exists(RULER_SOURCE):
        os.makedirs(os.path.dirname(RULER_SOURCE), exist_ok=True)
        subprocess.run([
            "git", "clone", "https://github.com/NVIDIA/RULER.git", RULER_SOURCE,
        ], check=True)
        subprocess.run([
            "git", "-C", RULER_SOURCE, "checkout", "--detach", RULER_REVISION,
        ], check=True)
    if os.path.exists(os.path.join(RULER_SOURCE, ".git")):
        revision = subprocess.check_output(
            ["git", "-C", RULER_SOURCE, "rev-parse", "HEAD"], text=True,
        ).strip()
    else:
        with open(revision_path) as handle:
            revision = handle.read().strip()
    if not os.path.isfile(prepare):
        raise SystemExit(f"[ruler] missing generator: {prepare}")
    if revision != RULER_REVISION:
        raise SystemExit(
            f"[ruler] source revision {revision!r} != pinned {RULER_REVISION}"
        )

    out_dir = os.path.join(DATASETS_ROOT, "ruler")
    out_path = os.path.join(out_dir, "niah_122880.jsonl")
    provenance_path = os.path.join(out_dir, "generator.json")
    if os.path.isfile(out_path) and os.path.isfile(provenance_path) and not force:
        print(f"[ruler] present, skipping ({out_path})")
        return

    raw_dir = os.path.join(out_dir, "raw_122880")
    os.makedirs(raw_dir, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    for task in RULER_TASKS:
        command = [
            sys.executable,
            prepare,
            "--save_dir", raw_dir,
            "--benchmark", "synthetic",
            "--task", task,
            "--tokenizer_path", tokenizer_path,
            "--tokenizer_type", "hf",
            "--max_seq_length", str(RULER_MAX_SEQ_LENGTH),
            "--num_samples", str(RULER_NUM_SAMPLES),
            "--random_seed", str(RULER_SEED),
            "--model_template_type", "base",
        ]
        print(f"[ruler] generating {task}")
        subprocess.run(command, check=True, env=env)
        task_path = os.path.join(raw_dir, task, "validation.jsonl")
        if not os.path.isfile(task_path):
            raise SystemExit(f"[ruler] generator did not create {task_path}")
        with open(task_path) as handle:
            count = sum(bool(line.strip()) for line in handle)
        if count != RULER_NUM_SAMPLES:
            raise SystemExit(
                f"[ruler] {task} generated {count}, expected {RULER_NUM_SAMPLES}"
            )

    temporary = f"{out_path}.tmp.{os.getpid()}"
    shard_paths = [
        os.path.join(
            out_dir,
            f"niah_122880.shard_{index}_of_{RULER_NUM_SHARDS}.jsonl",
        )
        for index in range(RULER_NUM_SHARDS)
    ]
    shard_temporaries = [f"{path}.tmp.{os.getpid()}" for path in shard_paths]
    shard_handles = [open(path, "w") for path in shard_temporaries]
    smoke_paths = [
        os.path.join(
            out_dir,
            f"niah_122880.smoke_{RULER_SMOKE_LIMIT}.shard_{index}_of_"
            f"{RULER_NUM_SHARDS}.jsonl",
        )
        for index in range(RULER_NUM_SHARDS)
    ]
    smoke_temporaries = [f"{path}.tmp.{os.getpid()}" for path in smoke_paths]
    smoke_handles = [open(path, "w") for path in smoke_temporaries]
    smoke_base, smoke_remainder = divmod(RULER_SMOKE_LIMIT, len(RULER_TASKS))
    smoke_source_indices = {}
    for position, task in enumerate(RULER_TASKS):
        take = smoke_base + (position < smoke_remainder)
        if take == 1:
            selected = {0}
        else:
            selected = {
                index * (RULER_NUM_SAMPLES - 1) // (take - 1)
                for index in range(take)
            }
        smoke_source_indices[task] = selected
    bench_index = 0
    smoke_index = 0
    source_files = {}
    try:
        with open(temporary, "w") as output:
            for task in RULER_TASKS:
                task_path = os.path.join(raw_dir, task, "validation.jsonl")
                source_files[task] = {
                    "path": os.path.relpath(task_path, REPO_ROOT),
                    "sha256": sha256_file(task_path),
                    "rows": RULER_NUM_SAMPLES,
                }
                with open(task_path) as source:
                    for source_index, line in enumerate(source):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        row.update({
                            "bench_index": bench_index,
                            "task": task,
                            "source_index": source_index,
                            "generator_revision": revision,
                        })
                        rendered = json.dumps(row, ensure_ascii=False) + "\n"
                        output.write(rendered)
                        shard_handles[bench_index % RULER_NUM_SHARDS].write(rendered)
                        if source_index in smoke_source_indices[task]:
                            smoke_handles[
                                smoke_index % RULER_NUM_SHARDS
                            ].write(rendered)
                            smoke_index += 1
                        bench_index += 1
            output.flush()
            os.fsync(output.fileno())
        for handle in shard_handles:
            handle.flush()
            os.fsync(handle.fileno())
        for handle in smoke_handles:
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in shard_handles + smoke_handles:
            handle.close()
    os.replace(temporary, out_path)
    for temporary_path, shard_path in zip(
        shard_temporaries, shard_paths, strict=True
    ):
        os.replace(temporary_path, shard_path)
    for temporary_path, shard_path in zip(
        smoke_temporaries, smoke_paths, strict=True
    ):
        os.replace(temporary_path, shard_path)
    if bench_index != len(RULER_TASKS) * RULER_NUM_SAMPLES:
        raise SystemExit(f"[ruler] combined row count is {bench_index}")
    if smoke_index != RULER_SMOKE_LIMIT:
        raise SystemExit(f"[ruler] smoke row count is {smoke_index}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    provenance = {
        "generator": "NVIDIA/RULER scripts/data/prepare.py",
        "generator_revision": revision,
        "random_seed": RULER_SEED,
        "tokenizer_path": (os.path.realpath(tokenizer_path)
                           if os.path.isdir(tokenizer_path) else tokenizer_path),
        "tokenizer_vocab_sha256": _tokenizer_vocab_sha256(tokenizer),
        "max_seq_length": RULER_MAX_SEQ_LENGTH,
        "num_samples_per_task": RULER_NUM_SAMPLES,
        "tasks": list(RULER_TASKS),
        "source_files": source_files,
        "evaluation_shards": [
            {
                "path": os.path.relpath(path, REPO_ROOT),
                "sha256": sha256_file(path),
                "rows": bench_index // RULER_NUM_SHARDS,
            }
            for path in shard_paths
        ],
        "smoke_shards": [
            {
                "path": os.path.relpath(path, REPO_ROOT),
                "sha256": sha256_file(path),
                "rows": (
                    RULER_SMOKE_LIMIT + RULER_NUM_SHARDS - 1 - index
                ) // RULER_NUM_SHARDS,
            }
            for index, path in enumerate(smoke_paths)
        ],
        "combined": {
            "path": os.path.relpath(out_path, REPO_ROOT),
            "sha256": sha256_file(out_path),
            "rows": bench_index,
        },
    }
    os.makedirs(os.path.dirname(provenance_path), exist_ok=True)
    with open(provenance_path, "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[ruler] wrote {bench_index} prompts -> {out_path}")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def _report(label: str, got: str, expected: str) -> bool:
    if not expected:
        print(f"  {label}: {got} (no expectation recorded)")
        return True
    ok = got == expected
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: {got}{'' if ok else f' != {expected}'}")
    return ok


def _jsonl_identity(path: str) -> tuple[int, str]:
    """Count non-empty rows and hash one large JSONL in a single pass."""
    digest = hashlib.sha256()
    rows = 0
    with open(path, "rb") as handle:
        for line in handle:
            digest.update(line)
            rows += bool(line.strip())
    return rows, digest.hexdigest()


def step_verify() -> bool:
    """Hash every staged evaluation input and validate its registered size."""
    manifest = load_manifest()
    print("[verify] checking staged artifacts against datasets/MANIFEST.json")
    ok, missing = True, 0

    if os.path.isfile(os.path.join(LONGBENCH_PRO_DIR, "dataset_info.json")):
        from datasets import load_from_disk

        dataset = load_from_disk(LONGBENCH_PRO_DIR)
        if hasattr(dataset, "values"):
            values = list(dataset.values())
            if len(values) != 1:
                raise SystemExit(
                    f"[verify] LongBench-Pro has {len(values)} splits, expected one"
                )
            dataset = values[0]
        spec = _expected(manifest, "datasets/longbench_pro")
        ok &= _report(
            "LongBench-Pro rows", str(len(dataset)), str(spec.get("n_rows", ""))
        )
        ok &= _report(
            "LongBench-Pro content sha256",
            content_hash_rows(dataset),
            spec.get("content_sha256"),
        )
    else:
        print("  MISS datasets/longbench_pro")
        missing += 1

    for rel, path in [
        ("datasets/mrcr/_filtered_le34000.jsonl",
         os.path.join(MRCR_DIR, "_filtered_le34000.jsonl")),
        ("datasets/mrcr/_filtered_le98048.jsonl",
         os.path.join(MRCR_DIR, "_filtered_le98048.jsonl")),
        ("datasets/aime24/aime24.parquet",
         os.path.join(DATASETS_ROOT, "aime24", "aime24.parquet")),
        ("datasets/aime25/aime25.parquet",
         os.path.join(DATASETS_ROOT, "aime25", "aime25.parquet")),
        ("datasets/gpqa/gpqa_diamond.parquet",
         os.path.join(DATASETS_ROOT, "gpqa", "gpqa_diamond.parquet")),
        ("datasets/lcb_v6/prompts.jsonl",
         os.path.join(DATASETS_ROOT, "lcb_v6", "prompts.jsonl")),
        ("datasets/ruler/niah_122880.jsonl",
         os.path.join(DATASETS_ROOT, "ruler", "niah_122880.jsonl")),
    ]:
        spec = _expected(manifest, rel)
        if not os.path.exists(path):
            print(f"  MISS {rel}")
            missing += 1
            continue
        if path.endswith(".parquet"):
            import pandas as pd

            rows = len(pd.read_parquet(path))
            ok &= _report(f"{rel} rows", str(rows), str(spec.get("n_rows", "")))
            if spec.get("content_sha256"):
                ok &= _report(
                    f"{rel} content sha256",
                    content_hash_parquet(path),
                    spec["content_sha256"],
                )
            else:
                ok &= _report(f"{rel} sha256", sha256_file(path), spec.get("sha256"))
        else:
            rows, digest = _jsonl_identity(path)
            ok &= _report(f"{rel} rows", str(rows), str(spec.get("n_rows", "")))
            ok &= _report(f"{rel} sha256", digest, spec.get("sha256"))

    ruler_provenance = os.path.join(DATASETS_ROOT, "ruler", "generator.json")
    if not os.path.isfile(ruler_provenance):
        print("  MISS datasets/ruler/generator.json")
        missing += 1
    else:
        spec = _expected(manifest, "datasets/ruler/generator.json")
        ok &= _report(
            "datasets/ruler/generator.json sha256",
            sha256_file(ruler_provenance),
            spec.get("sha256"),
        )
        with open(ruler_provenance) as handle:
            provenance = json.load(handle)
        for identity in (
            provenance.get("evaluation_shards", [])
            + provenance.get("smoke_shards", [])
        ):
            path = os.path.join(REPO_ROOT, identity["path"])
            if not os.path.isfile(path):
                print(f"  MISS {identity['path']}")
                missing += 1
                continue
            rows, digest = _jsonl_identity(path)
            ok &= _report(
                f"{identity['path']} rows", str(rows), str(identity.get("rows", ""))
            )
            ok &= _report(
                f"{identity['path']} sha256", digest, identity.get("sha256")
            )
    if missing:
        print(f"[verify] {missing} artifact(s) not staged — run --steps eval")
    return ok and missing == 0


def step_long_context_verify() -> bool:
    """Verify the long-context model, tokenizer, and MRCR cache."""
    config_path = os.path.join(LONG_CONTEXT_MODEL_DIR, "config.json")
    index_path = os.path.join(
        LONG_CONTEXT_MODEL_DIR, "model.safetensors.index.json"
    )
    cache_path = os.path.join(MRCR_DIR, "_filtered_le229120.jsonl")
    if not os.path.isfile(config_path):
        print(f"  MISS {config_path}")
        return False
    with open(config_path) as handle:
        model_config = json.load(handle)
    native_context = int(model_config.get("max_position_embeddings", 0))
    if native_context < 262_144:
        print(f"  FAIL native context: {native_context} < 262144")
        return False
    manifest = load_manifest()
    model_spec = manifest["upstream"].get(LONG_CONTEXT_MODEL_REPO, {})
    ok = True
    ok &= _report(
        "long-context model config sha256",
        sha256_file(config_path),
        model_spec.get("config_sha256"),
    )
    if not os.path.isfile(index_path):
        print(f"  MISS {index_path}")
        return False
    ok &= _report(
        "long-context model index sha256",
        sha256_file(index_path),
        model_spec.get("model_index_sha256"),
    )
    vocab_hash = verify_long_context_tokenizer()
    ok &= _report(
        "long-context tokenizer vocabulary sha256",
        vocab_hash,
        model_spec.get("tokenizer_vocab_sha256"),
    )
    if not os.path.isfile(cache_path):
        print(f"  MISS {cache_path}")
        return False
    count = 0
    maximum = 0
    with open(cache_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            maximum = max(maximum, int(row["prompt_tokens"]))
    spec = _expected(load_manifest(), "datasets/mrcr/_filtered_le229120.jsonl")
    expected_rows = spec.get("n_rows")
    ok &= expected_rows in (None, count) and maximum <= LONG_CONTEXT_MRCR_BUDGET
    print(
        f"  {'OK  ' if ok else 'FAIL'} long-context MRCR: "
        f"rows={count}, max_prompt_tokens={maximum}"
    )
    expected_hash = spec.get("sha256")
    if expected_hash:
        ok &= _report("long-context MRCR sha256", sha256_file(cache_path), expected_hash)
    return ok


def main() -> int:
    default_tokenizer = os.environ.get("BITDECODING_TOKENIZER") or os.environ.get(
        "MODEL_PATH") or LONG_CONTEXT_MODEL_REPO
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps", nargs="+", default=["eval"],
        choices=list(STEPS) + ["all", "eval", "long-context"],
        help="'eval' = all seven paper datasets; "
             "'long-context' = native-262k checkpoint and cache",
    )
    parser.add_argument("--tokenizer", default=default_tokenizer,
                        help="Qwen3 tokenizer for the MRCR length filter")
    parser.add_argument(
        "--long-context-mrcr-budget", type=int, default=LONG_CONTEXT_MRCR_BUDGET,
        help="long-context evaluation MRCR prompt cap (the 131k fallback uses 98048)",
    )
    parser.add_argument("--force", action="store_true",
                        help="rebuild artifacts that already exist")
    args = parser.parse_args()

    aliases = {
        "all": list(STEPS),
        "eval": list(EVAL_STEPS),
        "long-context": list(LONG_CONTEXT_STEPS),
    }
    steps = []
    for s in args.steps:
        steps.extend(aliases.get(s, [s]))
    seen, ordered = set(), []
    for s in STEPS:
        if s in steps and s not in seen:
            seen.add(s)
            ordered.append(s)

    for step in ordered:
        if step == "long-context-model":
            step_long_context_model(args.force)
        elif step == "longbench-pro":
            step_longbench_pro(args.force)
        elif step == "mrcr":
            step_mrcr(args.tokenizer, args.force)
        elif step == "long-context-mrcr":
            step_long_context_mrcr(args.force, args.long_context_mrcr_budget)
        elif step == "aime24":
            step_aime24(args.force)
        elif step == "aime25":
            step_aime25(args.force)
        elif step == "gpqa":
            step_gpqa(args.force)
        elif step == "lcb":
            step_lcb(args.force)
        elif step == "ruler":
            step_ruler(args.force, args.tokenizer)
        elif step == "verify":
            if not step_verify():
                print("\n[verify] MISMATCH — staged data differs from MANIFEST.json.")
                return 1
        elif step == "long-context-verify":
            if not step_long_context_verify():
                print("\n[long-context-verify] MISMATCH")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
