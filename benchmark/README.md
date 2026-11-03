# Benchmark

## Kernel benchmark

Compares BF16, TRT-LLM NVFP4, and BitDecoding NVFP4 decode latency using
synthetic inputs at 128K context:

```bash
REPO=/path/to/BitDecoding-FP4
CUDA_VISIBLE_DEVICES=0 "$REPO/_local/envs/bd_fp4/bin/python" "$REPO/benchmark/bench_kernel_sm10x.py"
```

> Defaults: Qwen3-4B and Qwen3-30B-A3B shapes at batch sizes 1, 16, 32, and 64; override with `--models` and `--batches`.

## E2E serving benchmark

Compares BF16, FP8, NVFP4, and BitDecoding NVFP4 decode throughput in vLLM at
1, 8, 16, 32, and 64 concurrent requests:

```bash
CUDA_VISIBLE_DEVICES=0 "$REPO/_local/envs/bd_fp4/bin/python" "$REPO/benchmark/bench_e2e_throughput.py" \
  --model Qwen/Qwen3-4B-Thinking-2507  # or Qwen/Qwen3-30B-A3B-Thinking-2507
```

> Defaults: 64K-token prompts on one GPU; for 128K on two GPUs, add `--workload 128k` and set `CUDA_VISIBLE_DEVICES=0,1`.
