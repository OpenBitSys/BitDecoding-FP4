# Accuracy evaluation

Evaluates LongBench-Pro, MRCR, RULER-NIAH, AIME 2024/2025, LiveCodeBench v6, and GPQA-Diamond.

## Datasets

```bash
cd /path/to/BitDecoding-FP4
PY="$PWD/_local/envs/bd_fp4/bin/python"
export MODEL_PATH=/path/to/Qwen3-4B-Thinking-2507
"$PY" tools/prepare_data.py --steps eval --tokenizer "$MODEL_PATH"
```

Downloads the datasets into `datasets/` and prepares RULER and LiveCodeBench under `_local/`.

## Run

Evaluate at 128K context:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m evaluation.run_eval --model-path "$MODEL_PATH" \
  --modes bitdecoding-nvfp4 --thinking --out-dir _local/accuracy
```

* Edit `EVALUATION` in [config.py](config.py) for context length, batch size, and dataset settings

* edit `RECIPE` for the paper's three methods. Key smoothing and τ are calibrated together and saved under `_local/accuracy/calibration/`.


LiveCodeBench scoring runs separately:

```bash
"$PY" tools/score_lcb.py --run-dir _local/accuracy --mode bitdecoding-nvfp4
```
