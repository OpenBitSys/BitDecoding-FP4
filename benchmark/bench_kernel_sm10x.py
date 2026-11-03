#!/usr/bin/env python3
"""Decode attention kernel speedup over BF16 KV at long context on SM100 / SM103.

- BF16 KV: FlashInfer trtllm-gen BF16 decode, what vLLM runs for BF16 decode
  on SM100.
- TRT-LLM-NVFP4: FlashInfer trtllm-gen NVFP4 decode with an FP8 query.
- BitDecoding-NVFP4: this repository's ``fp4_decode`` with a two-term FP4
  query over FP4 pages, at the fastest of up to four candidate split counts.

    python benchmark/bench_kernel_sm10x.py
    python benchmark/bench_kernel_sm10x.py --models qwen3-4b --batches 1,64
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: E402
from flashinfer.fp4_quantization import nvfp4_kv_quantize  # noqa: E402
from nvfp4_decode_kernel import _quantize, fp4_decode, split_partial_count  # noqa: E402
from nvfp4_decode_kernel._decode import split_k_heuristic  # noqa: E402
from nvfp4_vllm.builder import decode_num_splits  # noqa: E402

PAGE_SIZE = 128
HEAD_DIM = 128
HEADS_Q = 32
# name -> (title, KV heads)
MODELS = {"qwen3-4b": ("Qwen3-4B", 8), "qwen3-30b-a3b": ("Qwen3-30B-A3B", 4)}
SIDES = {
    "bf16": "BF16 KV",
    "trtllm_nvfp4": "TRT-LLM-NVFP4",
    "bitdecoding_nvfp4": "BitDecoding-NVFP4",
}
FP8 = torch.float8_e4m3fn
E4M3_MAX = 448.0
# vLLM's FlashInfer workspace size.
WORKSPACE_BYTES = 394 * 1024 * 1024
# Bound temporary allocations while preparing the benchmark's quantized cache.
QUANTIZE_CHUNK_PAGES = 32768
TRTLLM_CHUNK_PAGES = 512


def amax(x: torch.Tensor) -> float:
    lo, hi = torch.aminmax(x)
    return max(-lo.item(), hi.item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def quantize_pages(quantizer, pages: torch.Tensor):
    """This repository's FP4 pages and scales, one launch per chunk of pages."""
    count = pages.shape[0]
    data = scales = None
    for start in range(0, count, QUANTIZE_CHUNK_PAGES):
        stop = min(start + QUANTIZE_CHUNK_PAGES, count)
        chunk_data, chunk_scales = quantizer(pages[start:stop])
        if data is None:
            data = torch.empty((count, *chunk_data.shape[1:]), dtype=chunk_data.dtype,
                               device=pages.device)
            # The scales are a strided view in the kernel's swizzled layout;
            # only the page stride changes with the page count.
            storage = torch.empty(count * chunk_scales.stride(0), dtype=torch.uint8,
                                  device=pages.device)
            scales = storage.as_strided((count, *chunk_scales.shape[1:]),
                                        chunk_scales.stride()).view(chunk_scales.dtype)
        data[start:stop].copy_(chunk_data)
        scales[start:stop].copy_(chunk_scales)
    return data, scales


def trtllm_nvfp4_pages(pages: torch.Tensor, heads_kv: int):
    """trtllm-gen NVFP4 pages: HND layout, linear E4M3 scales, a global descale.

    trtllm-gen dequantizes to FP8 before its MMAs, so a block's ``e2m1 * scale``
    has to fit in E4M3; a descale of ``amax / 448`` keeps block scales at or
    below 448 / 6.
    """
    num_pages = pages.shape[0]
    descale = amax(pages) / E4M3_MAX
    global_scale = torch.tensor([descale], dtype=torch.float32, device=pages.device)
    data = torch.empty(num_pages, heads_kv, PAGE_SIZE, HEAD_DIM // 2, dtype=torch.uint8,
                       device=pages.device)
    scales = torch.empty(num_pages, heads_kv, PAGE_SIZE, HEAD_DIM // 16, dtype=FP8,
                         device=pages.device)
    for start in range(0, num_pages, TRTLLM_CHUNK_PAGES):
        stop = min(start + TRTLLM_CHUNK_PAGES, num_pages)
        block = pages[start:stop].permute(0, 2, 1, 3).contiguous()
        block_data, block_scales = nvfp4_kv_quantize(block.reshape(-1, HEAD_DIM), global_scale)
        data[start:stop].copy_(block_data.view(-1, heads_kv, PAGE_SIZE, HEAD_DIM // 2))
        scales[start:stop].copy_(
            block_scales.view(-1, heads_kv, PAGE_SIZE, HEAD_DIM // 16).view(FP8))
    return data, scales, descale


def split_candidates(batch: int, heads_kv: int, seqlen: int, sms: int) -> list[int]:
    tiles, pages = batch * heads_kv, seqlen // PAGE_SIZE
    candidates = {
        decode_num_splits(batch, heads_kv, sms),  # what the vLLM backend picks
        split_k_heuristic(batch, heads_kv, pages, sms=sms),
        max(1, sms // tiles),  # one wave of CTAs
    }
    # The fewest splits whose last wave of CTAs is at least 97% full, keeping
    # at least 64 pages per split.
    for splits in range(1, 65):
        if pages // splits < 64:
            break
        ctas = tiles * splits
        if ctas / (-(-ctas // sms) * sms) >= 0.97:
            candidates.add(splits)
            break
    return sorted(candidates)


def build_sides(batch: int, heads_kv: int, seqlen: int, splits: list[int], sms: int,
                device) -> dict:
    """name -> (zero-argument kernel call, its output as FP32 for the cosine check)."""
    pages_per_row = seqlen // PAGE_SIZE
    num_pages = batch * pages_per_row
    softmax_scale = HEAD_DIM**-0.5
    query = torch.randn(batch, HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device=device).mul_(0.3)
    k = torch.randn(num_pages, PAGE_SIZE, heads_kv, HEAD_DIM, dtype=torch.bfloat16,
                    device=device).mul_(0.3)
    v = torch.randn_like(k).mul_(0.3)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch, pages_per_row)
    seq_lens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)

    k_fp4, k_scales = quantize_pages(_quantize.quantize_key_pages, k)
    v_fp4, v_scales = quantize_pages(_quantize.quantize_value_pages, v)
    q0, s0, q1, s1 = _quantize.quantize_query(query, heads_kv=heads_kv, query_terms=2)

    k_trt, k_trt_scales, k_descale = trtllm_nvfp4_pages(k, heads_kv)
    v_trt, v_trt_scales, v_descale = trtllm_nvfp4_pages(v, heads_kv)
    q_descale = amax(query) / E4M3_MAX
    query_fp8 = (query.float() / q_descale).to(FP8)
    # trtllm-gen NVFP4 only writes E4M3. Attention output is a convex
    # combination of V rows, so |out| <= amax(V) and this descale cannot saturate.
    out_descale = amax(v) / E4M3_MAX

    k_hnd = k.permute(0, 2, 1, 3).contiguous()
    del k
    v_hnd = v.permute(0, 2, 1, 3).contiguous()
    del v

    def trtllm(query, kv_cache, out, **scales):
        workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

        def run():
            trtllm_batch_decode_with_kv_cache(
                query=query, kv_cache=kv_cache, workspace_buffer=workspace,
                block_tables=page_table, seq_lens=seq_lens, max_seq_len=seqlen, out=out,
                kv_layout="HND", backend="trtllm-gen", q_len_per_req=1, **scales)
            return out

        return run

    bf16_out = torch.empty(batch, HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    trt_out = torch.empty(batch, HEADS_Q, HEAD_DIM, dtype=FP8, device=device)
    sides = {
        "bf16": (
            trtllm(query, (k_hnd, v_hnd), bf16_out,
                   bmm1_scale=softmax_scale, bmm2_scale=1.0),
            lambda out: out.float(),
        ),
        "trtllm_nvfp4": (
            trtllm(query_fp8, (k_trt, v_trt), trt_out,
                   bmm1_scale=q_descale * k_descale * softmax_scale,
                   bmm2_scale=v_descale / out_descale,
                   kv_cache_sf=(k_trt_scales, v_trt_scales)),
            lambda out: out.float() * out_descale,
        ),
    }

    def bitdecoding(num_splits: int):
        out = torch.empty(batch, HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device=device)
        scratch = {}
        if num_splits > 1:
            partials = split_partial_count(batch, heads_kv, num_splits, residual=False, sms=sms)
            scratch = {
                "split_partial_scratch": torch.empty(
                    partials * batch * HEADS_Q * HEAD_DIM, dtype=torch.float32, device=device),
                "split_lse_scratch": torch.empty(
                    partials * batch * HEADS_Q, dtype=torch.float32, device=device),
            }

        def run():
            return fp4_decode(
                key_pages_fp4=k_fp4, key_scales=k_scales,
                value_pages_fp4=v_fp4, value_scales=v_scales,
                fp4_page_table=page_table, seqused_fp4=seq_lens,
                query_fp4=q0, query_scales=s0, query_fp4_second=q1, query_scales_second=s1,
                query_terms=2, softmax_scale=softmax_scale, trusted_metadata=True,
                num_splits=num_splits, out=out, **scratch)

        return run

    for num_splits in splits:
        sides[f"bitdecoding_nvfp4_s{num_splits}"] = (bitdecoding(num_splits), lambda out: out.float())
    return sides


def capture(run, calls: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            run()
    return graph


def time_us(graph: torch.cuda.CUDAGraph, calls: int, args) -> float:
    """Median microseconds per call over ``args.repeats`` timed batches of replays."""
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / (args.iters * calls))
    return statistics.median(samples)


def run_case(model: str, batch: int, args, device, sms: int) -> dict:
    title, heads_kv = MODELS[model]
    torch.manual_seed(args.seed + batch)
    torch.cuda.reset_peak_memory_stats(device)
    splits = split_candidates(batch, heads_kv, args.seqlen, sms)
    sides = build_sides(batch, heads_kv, args.seqlen, splits, sms, device)

    # The first call compiles each kernel and is the accuracy sample.
    outputs = {}
    for name, (run, as_float) in sides.items():
        outputs[name] = as_float(run()).clone()
        torch.cuda.synchronize()
    cosines = {name: cosine(out, outputs["bf16"]) for name, out in outputs.items() if name != "bf16"}

    graphs = {name: capture(run, args.calls_per_graph) for name, (run, _) in sides.items()}
    names = list(graphs)
    samples = {name: [] for name in names}
    for round_index in range(args.rounds):
        for name in names if round_index % 2 == 0 else names[::-1]:
            samples[name].append(time_us(graphs[name], args.calls_per_graph, args))
    measured = {name: statistics.median(values) for name, values in samples.items()}

    by_split = {s: measured[f"bitdecoding_nvfp4_s{s}"] for s in splits}
    best_split = min(by_split, key=by_split.get)
    us = {
        "bf16": measured["bf16"],
        "trtllm_nvfp4": measured["trtllm_nvfp4"],
        "bitdecoding_nvfp4": by_split[best_split],
    }
    row = {
        "model": model, "title": title, "batch": batch, "seqlen": args.seqlen,
        "heads_q": HEADS_Q, "heads_kv": heads_kv,
        "us": us,
        "speedup": {side: us["bf16"] / value for side, value in us.items()},
        "bitdecoding_splits": best_split,
        "bitdecoding_us_by_splits": by_split,
        "cosine_vs_bf16": {
            "trtllm_nvfp4": cosines["trtllm_nvfp4"],
            "bitdecoding_nvfp4": cosines[f"bitdecoding_nvfp4_s{best_split}"],
        },
        "samples_us": samples,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    print(f"{title:<14} batch {batch:>3} | BF16 KV {us['bf16']:8.1f} us"
          f" | TRT-LLM-NVFP4 {us['trtllm_nvfp4']:8.1f} us ({row['speedup']['trtllm_nvfp4']:.2f}x)"
          f" | BitDecoding-NVFP4 {us['bitdecoding_nvfp4']:8.1f} us"
          f" ({row['speedup']['bitdecoding_nvfp4']:.2f}x, {best_split} splits)", flush=True)
    return row


def summary(rows: list[dict]) -> str:
    lines = ["", "Speedup over BF16 KV"]
    for model in dict.fromkeys(row["model"] for row in rows):
        model_rows = [row for row in rows if row["model"] == model]
        lines.append(f"  {MODELS[model][0]}")
        lines.append("    " + f"{'batch':<20}" + "".join(f"{row['batch']:>8}" for row in model_rows))
        for side in ("trtllm_nvfp4", "bitdecoding_nvfp4"):
            lines.append("    " + f"{SIDES[side]:<20}"
                         + "".join(f"{row['speedup'][side]:>8.2f}" for row in model_rows))
    return "\n".join(lines)


def provenance(device, args) -> dict:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sms": props.multi_processor_count,
        "packages": {name: version(name) for name in
                     ("torch", "flashinfer-python", "nvidia-cutlass-dsl")},
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(MODELS),
                        help=f"comma-separated, from {', '.join(MODELS)}")
    parser.add_argument("--batches", default="1,16,32,64")
    parser.add_argument("--seqlen", type=int, default=131072,
                        help="context length per row, a multiple of 128")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30, help="graph replays before timing")
    parser.add_argument("--iters", type=int, default=20, help="graph replays per timed sample")
    parser.add_argument("--repeats", type=int, default=5, help="timed samples per round")
    parser.add_argument("--calls-per-graph", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "_local/benchmark/kernel_sm10x.json")
    args = parser.parse_args()

    models = args.models.split(",")
    unknown = [model for model in models if model not in MODELS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    if args.seqlen % PAGE_SIZE:
        parser.error("--seqlen must be a multiple of 128")
    batches = [int(value) for value in args.batches.split(",")]

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise SystemExit(f"needs an SM100-family GPU (B200 / B300); "
                         f"{torch.cuda.get_device_name(device)} is sm_{major}{minor}")
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    result = {**provenance(device, args), "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for model in models:
        for batch in batches:
            result["rows"].append(run_case(model, batch, args, device, sms))
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            gc.collect()
            torch.cuda.empty_cache()

    print(summary(result["rows"]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
