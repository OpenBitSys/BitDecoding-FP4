#!/usr/bin/env python3
"""Reproduce the paper's 64K/TP1 or 128K/TP2 serving throughput sweep."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmark")]
METHODS = ("bf16", "fp8", "nvfp4", "bitdecoding-nvfp4")
WORKLOADS = {"64k": (65536, 1), "128k": (131072, 2)}
MODEL_REVISIONS = {
    "Qwen/Qwen3-4B-Thinking-2507": "768f209d9ea81521153ed38c47d515654e938aea",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
}
from evaluation.config import (
    RECIPE as ACCURACY_RECIPE,
    needs_tau_calibration,
    probability_environment,
    recipe_additional_config as custom_additional_config,
)

RECIPE = copy.deepcopy(ACCURACY_RECIPE)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_prompts(tokenizer, corpus: list[dict], count: int, length: int) -> list[dict]:
    """Pack the same shuffled MRCR messages for both context lengths."""
    marker = "BODY_MARKER_45f24ec9"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    prefix, suffix = template.split(marker)
    prefix = tokenizer.encode(prefix, add_special_tokens=False)
    suffix = tokenizer.encode(suffix, add_special_tokens=False)
    budget = 131072 - len(prefix) - len(suffix)
    if not corpus or length <= len(prefix) + len(suffix):
        raise ValueError("empty corpus or insufficient context for the chat template")
    order = list(range(len(corpus)))
    random.Random(42).shuffle(order)
    cache, prompts, cursor = {}, [], 0
    for index in range(count):
        body = []
        while len(body) < budget:
            source = order[cursor % len(order)]
            cursor += 1
            if source not in cache:
                text = "\n\n".join(
                    message["role"] + ":\n" + message["content"]
                    for message in corpus[source]["messages"]
                ) + "\n\n"
                cache[source] = tokenizer.encode(text, add_special_tokens=False)
                if not cache[source]:
                    raise ValueError(f"corpus row {source} contains no text tokens")
            body.extend(cache[source])
        ids = prefix + body[:length - len(prefix) - len(suffix)] + suffix
        prompts.append({"prompt_token_ids": ids, "sampling_seed": 42 + index})
    return prompts


def decode_rates(short: list[float], long: list[float], count: int,
                 short_tokens: int, long_tokens: int, tp: int) -> dict:
    if (not short or len(short) != len(long) or count < 1 or tp < 1
            or not 0 < short_tokens < long_tokens
            or any(not math.isfinite(t) or t <= 0 for t in short + long)):
        raise ValueError("invalid paired timings or output lengths")
    elapsed = statistics.median(long) - statistics.median(short)
    differences = [b - a for a, b in zip(short, long, strict=True)]
    if elapsed <= 0 or any(value <= 0 for value in differences):
        raise ValueError("long generations must take longer than their paired baselines")
    per_request = (long_tokens - short_tokens) / elapsed
    return {
        "decode_seconds": elapsed,
        "tokens_per_second_per_request": per_request,
        "tokens_per_second_per_gpu": count * per_request / tp,
        "short_median_seconds": statistics.median(short),
        "long_median_seconds": statistics.median(long),
        "paired_differences_seconds": differences,
    }


class ThroughputWorker:
    """Synchronize every TP rank without instrumenting the attention kernels."""

    def throughput_sync(self) -> None:
        import torch

        torch.cuda.synchronize()

    def throughput_state(self) -> dict:
        import torch
        from nvfp4_decode_kernel.p_scale import P_SMALL_GROUP_AMAX, resolve_p_scale_environment

        torch.cuda.synchronize()
        config = self.vllm_config
        model = self.get_model()
        layers = [module for module in model.modules()
                  if hasattr(module, "attn_backend") and hasattr(module, "kv_cache_dtype")]
        return {
            "rank": self.rank,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "kv_cache_blocks": self.model_runner.kv_cache_config.num_blocks,
            "cache_dtypes": dict(Counter(str(layer.kv_cache_dtype) for layer in layers)),
            "backends": dict(Counter(layer.attn_backend.get_name() for layer in layers)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "probability_scaling": {"mode": resolve_p_scale_environment()[0], "tau": P_SMALL_GROUP_AMAX},
            "engine": {
                "tp": config.parallel_config.tensor_parallel_size,
                "max_model_len": config.model_config.max_model_len,
                "max_num_seqs": config.scheduler_config.max_num_seqs,
                "block_size": config.cache_config.block_size,
                "gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
                "enable_prefix_caching": config.cache_config.enable_prefix_caching,
                "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
                "enforce_eager": config.model_config.enforce_eager,
                "enable_flashinfer_autotune": config.kernel_config.enable_flashinfer_autotune,
                "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
            },
        }


class AdmissionObserver:
    """Count resident requests and preemptions without changing scheduling."""

    def __init__(self, llm):
        core = llm.llm_engine.engine_core
        while hasattr(core, "engine_core"):
            core = core.engine_core
        self.scheduler = core.scheduler
        self.original = self.scheduler.schedule
        self.reset()

    def reset(self) -> None:
        self.max_running = 0
        self.preemptions = {}

    def schedule(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        self.max_running = max(self.max_running, len(self.scheduler.running))
        for key, request in self.scheduler.requests.items():
            self.preemptions[key] = max(self.preemptions.get(key, 0), request.num_preemptions)
        return result

    def __enter__(self):
        self.scheduler.schedule = self.schedule
        return self

    def __exit__(self, *_):
        self.scheduler.schedule = self.original


def measure(llm, rows: list[dict], observer, short_tokens: int,
            long_tokens: int, warmup: int, repeats: int, clock=time.perf_counter) -> dict:
    from vllm import SamplingParams, TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=row["prompt_token_ids"]) for row in rows]
    parameters = {
        length: [SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
            min_tokens=length, max_tokens=length, ignore_eos=True,
            seed=row["sampling_seed"],
        ) for row in rows]
        for length in (short_tokens, long_tokens)
    }
    calls = []

    def run(length, measured):
        observer.reset()
        llm.collective_rpc("throughput_sync")
        start = clock()
        outputs = llm.generate(prompts, parameters[length], use_tqdm=False)
        llm.collective_rpc("throughput_sync")
        seconds = clock() - start
        if len(outputs) != len(rows):
            raise RuntimeError("not every submitted request completed")
        for row, output in zip(rows, outputs, strict=True):
            if (list(output.prompt_token_ids) != row["prompt_token_ids"]
                    or len(output.outputs) != 1
                    or len(output.outputs[0].token_ids) != length):
                raise RuntimeError("generated prompt or output length differs from the workload")
        calls.append({
            "output_tokens": length, "measured": measured, "seconds": seconds,
            "max_resident_requests": observer.max_running,
            "preemptions": sum(observer.preemptions.values()),
        })
        return seconds

    for _ in range(warmup):
        run(short_tokens, False)
        run(long_tokens, False)
    short, long = [], []
    for repeat in range(repeats):
        order = (short_tokens, long_tokens) if repeat % 2 == 0 else (long_tokens, short_tokens)
        times = {length: run(length, True) for length in order}
        short.append(times[short_tokens])
        long.append(times[long_tokens])
    return {"short_seconds": short, "long_seconds": long, "calls": calls}


def engine_options(job: dict, *, calibration=False) -> dict:
    options = {
        "model": job["model"], "dtype": "bfloat16", "trust_remote_code": True,
        "tensor_parallel_size": 1 if calibration else job["tp"],
        "max_model_len": job["input_tokens"] + 2048,
        "max_num_seqs": 1 if calibration else job["concurrency"],
        "max_num_batched_tokens": job["input_tokens"] + 2048,
        "block_size": 128, "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": False, "enable_chunked_prefill": False,
        "enforce_eager": False, "enable_flashinfer_autotune": False,
    }
    if not calibration:
        options["worker_extension_cls"] = "bench_e2e_throughput.ThroughputWorker"
    elif needs_tau_calibration(job["recipe"]):
        options["enforce_eager"] = True
    return options


def build_engine(job: dict):
    options = engine_options(job)
    method = job["method"]
    if method == "bitdecoding-nvfp4":
        from nvfp4_vllm import build_llm

        recipe = job["recipe"]
        return build_llm(
            additional_config=custom_additional_config(
                recipe, key_smoothing=job["key_smoothing"]),
            **options,
        )
    options["kv_cache_dtype"] = {"bf16": "auto", "fp8": "fp8", "nvfp4": "nvfp4"}[method]
    from evaluation.vllm_runner import VllmRunner

    options["attention_config"] = {"backend": "FLASHINFER"}
    return VllmRunner._build_standard(
        options, {"kv_cache_dtype": options["kv_cache_dtype"]},
        worker_extension_cls=options["worker_extension_cls"],
    )


def shutdown(llm) -> None:
    for owner in (llm, llm.llm_engine, llm.llm_engine.engine_core):
        close = getattr(owner, "shutdown", None)
        if callable(close):
            close()
            return


def run_worker(job: dict) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    recipe = job["recipe"]
    custom = job["kind"] == "calibrate" or job["method"] == "bitdecoding-nvfp4"
    environment = (probability_environment(recipe, calibration=job["kind"] == "calibrate")
                   if custom else {"NVFP4_P_SCALE_MODE": None, "NVFP4_P_SCALE_TAU": None})
    for key, value in environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.environ.pop("NVFP4_P_SCALE_DIVISOR", None)
    if job["kind"] == "calibrate":
        from nvfp4_vllm import build_llm, calibrate_key_smoothing

        llm = build_llm(additional_config=custom_additional_config(recipe),
                        **engine_options(job, calibration=True))
        try:
            multiplier = (recipe["underflow_aware_probability_scaling"]["c"]
                          if needs_tau_calibration(recipe) else None)
            result = calibrate_key_smoothing(
                llm, context_tokens=job["input_tokens"] + 2047,
                save_to=job["key_smoothing"], probability_multiplier=multiplier,
            )
            if result is None or (job["key_smoothing"] and not Path(job["key_smoothing"]).is_file()):
                raise RuntimeError("key-smoothing calibration did not produce a sidecar")
            save(Path(job["result"]), result)
        finally:
            shutdown(llm)
        return

    rows = [json.loads(line) for line in Path(job["inputs"]).read_text().splitlines()]
    rows = rows[:job["concurrency"]]
    short_tokens, long_tokens = 128, 2048
    llm = build_engine(job)
    try:
        workers = llm.collective_rpc("throughput_state")
        expected_backend = "CUSTOM" if job["method"] == "bitdecoding-nvfp4" else "FLASHINFER"
        expected_dtype = {"bf16": "auto", "fp8": "fp8"}.get(job["method"], "nvfp4")
        layers = json.loads((Path(job["model"]) / "config.json").read_text())["num_hidden_layers"]
        if (len(workers) != job["tp"]
                or len({worker["rank"] for worker in workers}) != job["tp"]
                or any(worker["compute_capability"][0] != 10
                       or worker["backends"] != {expected_backend: layers}
                       or worker["cache_dtypes"] != {expected_dtype: layers} for worker in workers)):
            raise RuntimeError("GPU or loaded attention layers differ from the requested method")
        expected_engine = {
            "tp": job["tp"], "max_model_len": job["input_tokens"] + 2048,
            "max_num_seqs": job["concurrency"], "block_size": 128,
            "gpu_memory_utilization": 0.90, "enable_prefix_caching": False,
            "enable_chunked_prefill": False, "enforce_eager": False,
            "enable_flashinfer_autotune": False,
        }
        for worker in workers:
            if (any(worker["engine"][key] != value for key, value in expected_engine.items())
                    or worker["engine"]["cudagraph_mode"] in ("None", "NONE", "CUDAGraphMode.NONE")):
                raise RuntimeError("resolved engine settings differ from the paper workload")
            if job["method"] == "bitdecoding-nvfp4":
                environment = probability_environment(recipe)
                if worker["probability_scaling"]["mode"] != environment["NVFP4_P_SCALE_MODE"]:
                    raise RuntimeError("worker probability scaling differs from the recipe")
                if environment["NVFP4_P_SCALE_TAU"] is not None and (
                    worker["probability_scaling"]["tau"] != float(environment["NVFP4_P_SCALE_TAU"])
                ):
                    raise RuntimeError("worker did not load the calibrated tau")
        with AdmissionObserver(llm) as observer:
            timing = measure(llm, rows, observer, short_tokens, long_tokens,
                             job["warmup"], job["repeats"])
        rates = decode_rates(timing["short_seconds"], timing["long_seconds"], len(rows),
                             short_tokens, long_tokens, job["tp"])
        timed = [call for call in timing["calls"] if call["measured"]]
        resident = max(call["max_resident_requests"] for call in timed)
        preemptions = sum(call["preemptions"] for call in timed)
        save(Path(job["result"]), {
            **job, **rates, **timing, "baseline_tokens": short_tokens,
            "output_tokens": long_tokens, "max_resident_requests": resident,
            "preemptions": preemptions, "queued": resident < len(rows) or preemptions > 0,
            "workers": llm.collective_rpc("throughput_state"),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "vllm", "flashinfer-python", "nvidia-cutlass-dsl")},
        })
    finally:
        shutdown(llm)


def launch(job: dict, directory: Path) -> None:
    job_path = directory / "job.json"
    save(job_path, job)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmark"), environment.get("PYTHONPATH", "")])
    with (directory / "run.log").open("w") as log:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 "--worker", str(job_path)], env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"benchmark failed; see {directory / 'run.log'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Qwen3 checkpoint directory or Hugging Face model ID")
    parser.add_argument("--workload", choices=WORKLOADS, default="64k")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated cache methods")
    parser.add_argument("--concurrency", default="1,8,16,32,64")
    parser.add_argument("--prompts", type=Path, help="MRCR messages JSONL; downloaded if omitted")
    parser.add_argument("--out", type=Path, help="empty output directory")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(json.loads(args.worker.read_text()))
        return
    methods = args.methods.split(",")
    try:
        concurrency = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        parser.error("--concurrency must contain comma-separated integers")
    if (not methods or set(methods) - set(METHODS) or len(methods) != len(set(methods))
            or not concurrency or any(c < 1 or c > 64 for c in concurrency)
            or len(concurrency) != len(set(concurrency)) or args.warmup < 0 or args.repeats < 1):
        parser.error(f"use distinct methods from {METHODS}, concurrency 1..64, and valid repeat counts")
    if args.prompts is not None and not args.prompts.is_file():
        parser.error(f"prompt corpus does not exist: {args.prompts}")
    name = args.model.rstrip("/").split("/")[-1]
    output = (args.out or ROOT / "_local/benchmark/e2e" / name / args.workload).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output directory is not empty: {output}; choose another --out")
    model = Path(args.model).expanduser()
    if not model.is_dir():
        from huggingface_hub import snapshot_download

        model = Path(snapshot_download(args.model, revision=MODEL_REVISIONS.get(args.model)))
    model = model.resolve()
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") not in {"qwen3", "qwen3_moe"} or config.get("quantization_config"):
        parser.error("the paper workload requires an unquantized Qwen3 checkpoint")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    corpus_path = args.prompts or ROOT / "datasets/mrcr/_filtered_le98048.jsonl"
    if not corpus_path.exists():
        from huggingface_hub import snapshot_download
        from tools.prepare_data import build_mrcr_cache

        snapshot_download("openai/mrcr", repo_type="dataset",
                          local_dir=str(ROOT / "datasets/mrcr"),
                          allow_patterns=[f"{n}needle/*.parquet" for n in (2, 4, 8)])
        build_mrcr_cache(str(model), budget=98048, out_path=str(corpus_path),
                         manifest_key="datasets/mrcr/_filtered_le98048.jsonl")
    corpus = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    length, tp = WORKLOADS[args.workload]
    prompts = make_prompts(tokenizer, corpus, max(concurrency), length)
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "prompts.jsonl"
    with input_path.open("w") as stream:
        for row in prompts:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    base = {
        "model": str(model), "workload": args.workload, "input_tokens": length, "tp": tp,
        "inputs": str(input_path), "input_sha256": digest(input_path),
        "corpus_sha256": digest(corpus_path), "warmup": args.warmup, "repeats": args.repeats,
        "recipe": copy.deepcopy(RECIPE),
        "key_smoothing": (str(output / "key_smoothing.safetensors")
                          if RECIPE["outlier_aware_key_quantization"]["key_smoothing"] else None),
    }
    manifest = {**base, "methods": methods, "concurrency": concurrency,
                                     "requested_recipe": copy.deepcopy(RECIPE),
                                     "model_source": args.model,
                                     "model_revision": MODEL_REVISIONS.get(args.model),
                                     "model_config_sha256": digest(model / "config.json"),
                                     "benchmark_sha256": digest(Path(__file__)),
                                     "sampling": {"temperature": 0.6, "top_p": 0.95,
                                                  "top_k": 20, "min_p": 0.0, "seed": "42 + index"}}
    save(output / "manifest.json", manifest)
    if "bitdecoding-nvfp4" in methods and (base["key_smoothing"] or needs_tau_calibration(RECIPE)):
        print("Calibrating key smoothing and probability scaling before timing...", flush=True)
        launch({**base, "kind": "calibrate", "result": str(output / "calibration.json")},
               output / "calibration")
        if needs_tau_calibration(RECIPE):
            calibration = json.loads((output / "calibration.json").read_text())
            tau = calibration["probability_scaling"]["tau"]
            base["recipe"]["underflow_aware_probability_scaling"]["tau"] = tau
            print(f"Calibrated tau={tau:.9g}", flush=True)
        save(output / "manifest.json", manifest)
    results = []
    for count in concurrency:
        for method in methods:
            directory = output / f"c{count}" / method
            result_path = directory / "result.json"
            print(f"{name} {args.workload} TP{tp} concurrency={count} {method}", flush=True)
            launch({**base, "kind": "measure", "method": method, "concurrency": count,
                    "result": str(result_path)}, directory)
            result = json.loads(result_path.read_text())
            fields = ("method", "concurrency", "tp", "input_tokens", "output_tokens", "baseline_tokens",
                      "tokens_per_second_per_gpu", "tokens_per_second_per_request", "decode_seconds",
                      "max_resident_requests", "preemptions", "queued")
            results.append({field: result[field] for field in fields})
            with (output / "throughput.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
            print(f"  {result['tokens_per_second_per_gpu']:.2f} tokens/s/GPU, "
                  f"{result['tokens_per_second_per_request']:.2f} tokens/s/request", flush=True)
    print(f"Results: {output / 'throughput.csv'}")


if __name__ == "__main__":
    main()
