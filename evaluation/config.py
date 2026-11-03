"""Static configuration for the BitDecoding-FP4 evaluation.

One in-process vLLM engine is stood up per KV-cache mode and every requested
benchmark is scored against it. The modes include:

  * ``bf16``              -> BF16 KV baseline (``kv_cache_dtype="auto"``)
  * ``fp8``               -> vLLM FP8 e4m3 KV cache with unit scales
  * ``turboquant-4bit-nc`` -> vLLM TurboQuant 4-bit keys and values
  * ``turboquant-4bit-nc-all-layers`` -> the same preset on every layer
  * ``nvfp4``             -> official vLLM NVFP4 KV cache
  * ``bitdecoding-nvfp4`` -> this repository's BitDecoding-NVFP4 recipe
  * ``bitdecoding-mxfp4`` -> MXFP4 scales with the same serving recipe

``bitdecoding-nvfp4`` is the method: CUSTOM backend, 16-token BF16 sink,
K centering, ``{4,6}`` block scales for K with the default ``amax6`` for V
and Q, two-term query, and Hadamard rotation. It
calibrates static key smoothing and the probability threshold together before
scoring, or reuses the matching calibration saved with the evaluation output.

``EVALUATION`` defines the default 128K run. Dataset paths are relative to
the repository's ``datasets/`` directory, populated by ``tools/prepare_data.py``.
"""
from __future__ import annotations

import copy
import json
import math
import os

EVALUATION = {
    "max_model_len": 131_072,
    "max_num_seqs": 1,
    "longbench_prompt_budget": 98_048,
    "longbench_excluded_buckets": ["256k"],
    "mrcr_cache": "mrcr/_filtered_le98048.jsonl",
}

RECIPE = {
    "dual_residual_query_quantization": True,
    "underflow_aware_probability_scaling": {
        "enabled": True,
        "c": 8,
        "tau": "calibrate",
    },
    "outlier_aware_key_quantization": {
        "key_smoothing": True,
        "hadamard_rotation": True,
        "adaptive_block_scale_search": True,
    },
}


def recipe_additional_config(recipe: dict, *, key_smoothing: str | None = None) -> dict:
    keys = recipe["outlier_aware_key_quantization"]
    return {
        "nvfp4_key_smoothing": key_smoothing if keys["key_smoothing"] else None,
        "nvfp4_scale_mode": {
            "key": "four_six" if keys["adaptive_block_scale_search"] else "amax6",
            "value": "amax6", "query": "amax6",
        },
        "nvfp4_query_terms": 2 if recipe["dual_residual_query_quantization"] else 1,
        "nvfp4_rotation": "hadamard" if keys["hadamard_rotation"] else "none",
    }


def needs_tau_calibration(recipe: dict) -> bool:
    probability = recipe["underflow_aware_probability_scaling"]
    return probability["enabled"] and probability["tau"] == "calibrate"


def probability_environment(recipe: dict, *, calibration: bool = False) -> dict:
    probability = recipe["underflow_aware_probability_scaling"]
    if not probability["enabled"]:
        return {"NVFP4_P_SCALE_MODE": "amax6", "NVFP4_P_SCALE_TAU": None}
    if probability["c"] not in (1, 8):
        raise ValueError("the probability kernel supports c=1 or c=8")
    tau = probability["tau"]
    if tau == "calibrate":
        if not calibration:
            raise ValueError("tau must be calibrated before evaluation or throughput measurement")
        tau = None
    elif type(tau) not in (int, float) or not math.isfinite(tau) or not 0 <= tau <= 448:
        raise ValueError("tau must be 'calibrate' or a finite number between 0 and 448")
    return {
        "NVFP4_P_SCALE_MODE": "amax2_or_amax6_c8" if probability["c"] == 8 else "amax2_or_amax6",
        "NVFP4_P_SCALE_TAU": None if tau is None else repr(float(tau)),
    }


# Repo root == parent of this ``evaluation/`` package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Use a Hub model ID or a local checkpoint path.
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507")
# Evaluation datasets (staged by tools/prepare_data.py). Override with BITDECODING_DATA_DIR.
DATA_DIR = os.environ.get("BITDECODING_DATA_DIR", os.path.join(REPO_ROOT, "datasets"))
# Launchers can override these defaults for individual evaluation shards.
MAX_MODEL_LEN = int(os.environ.get("BITDECODING_MAX_MODEL_LEN", EVALUATION["max_model_len"]))
LONGBENCH_PROMPT_BUDGET = int(
    os.environ.get("BITDECODING_LONGBENCH_PROMPT_BUDGET", EVALUATION["longbench_prompt_budget"])
)

# Use the same scheduler and CUDA graph settings across cache modes.
PARITY_ENGINE_KWARGS = {
    "max_num_seqs": int(os.environ.get(
        "BITDECODING_LONG_CONTEXT_MAX_NUM_SEQS", EVALUATION["max_num_seqs"]
    )),
    "block_size": 128,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "max_num_batched_tokens": max(40_960, MAX_MODEL_LEN),
    "pipeline_parallel_size": 1,
    "enforce_eager": False,
}
LONG_CONTEXT_ENGINE_KWARGS = dict(PARITY_ENGINE_KWARGS)

# Shared calibration files; the accuracy runner creates missing key means.
SIDECAR_DIR = os.environ.get(
    "BITDECODING_SIDECAR_DIR", os.path.join(REPO_ROOT, "_local", "sidecars")
)
KEY_SHIFT_SIDECAR = os.environ.get(
    "BITDECODING_KEY_SHIFT_SIDECAR",
    os.path.join(SIDECAR_DIR, "key_shift.safetensors"),
)

# Windows above this threshold use the long-context sampling settings.
SHORT_CONTEXT_MAX_MODEL_LEN = 40_960

KV_MODES = {
    "bf16": {
        "kv_cache_dtype": "auto",
        "label": "BF16 KV (baseline)",
    },
    "fp8": {
        "kv_cache_dtype": "fp8",
        "label": "vLLM FP8 e4m3 KV (unit scales)",
    },
    "turboquant-4bit-nc": {
        "kv_cache_dtype": "turboquant_4bit_nc",
        "label": "vLLM TurboQuant 4-bit NC",
    },
    "turboquant-4bit-nc-all-layers": {
        "kv_cache_dtype": "turboquant_4bit_nc",
        "turboquant_all_layers": True,
        "label": "vLLM TurboQuant 4-bit NC (all layers)",
    },
    "nvfp4": {
        "kv_cache_dtype": "nvfp4",
        "label": "NVFP4 KV cache (official vLLM)",
    },
    "bitdecoding-nvfp4": {
        "kv_cache_dtype": "nvfp4",
        "engine_kwargs": PARITY_ENGINE_KWARGS,
        "custom_backend": True,
        "sink_tokens": 16,
        "scale_mode": recipe_additional_config(RECIPE)["nvfp4_scale_mode"],
        "key_scale_geometry": os.environ.get(
            "BITDECODING_KEY_SCALE_GEOMETRY", "local"
        ),
        "query_terms": recipe_additional_config(RECIPE)["nvfp4_query_terms"],
        "rotation": os.environ.get("BITDECODING_CUSTOM_ROTATION", recipe_additional_config(RECIPE)["nvfp4_rotation"]),
        "key_shift": (KEY_SHIFT_SIDECAR
                      if RECIPE["outlier_aware_key_quantization"]["key_smoothing"] else None),
        "label": "BitDecoding-NVFP4",
    },
    "bitdecoding-mxfp4": {
        "kv_cache_dtype": "nvfp4",
        "engine_kwargs": PARITY_ENGINE_KWARGS,
        "custom_backend": True,
        "sink_tokens": 16,
        "scale_mode": "mxfp4_search",
        "key_scale_geometry": "local",
        "query_terms": recipe_additional_config(RECIPE)["nvfp4_query_terms"],
        "rotation": "hadamard",
        "key_shift": (KEY_SHIFT_SIDECAR
                      if RECIPE["outlier_aware_key_quantization"]["key_smoothing"] else None),
        "label": "BitDecoding-MXFP4",
    },
}


def model_family(model_path=None):
    """Read the model family from its config, falling back to the checkpoint name."""
    try:
        with open(os.path.join(model_path or MODEL_PATH, "config.json")) as handle:
            value = json.load(handle)
        if "glm" in json.dumps(value).lower():
            return "glm"
    except (OSError, TypeError, json.JSONDecodeError):
        pass
    return "glm" if "glm" in str(model_path or "").lower() else "qwen"


def chat_template_kwargs(model_path, thinking):
    return {"enable_thinking": thinking}


def resolve_mode(name: str, model_path=None) -> dict:
    """A copy of ``KV_MODES[name]`` with long-context engine overlays applied."""
    mode = copy.deepcopy(KV_MODES[name])
    seqs = int(os.environ.get(
        "BITDECODING_LONG_CONTEXT_MAX_NUM_SEQS", EVALUATION["max_num_seqs"]
    ))
    kwargs = dict(LONG_CONTEXT_ENGINE_KWARGS)
    kwargs.update(mode.get("engine_kwargs") or {})
    # Disable FP8 autotuning to avoid FA2 paths without FP8 prefill support.
    if name == "fp8":
        kwargs["attention_backend"] = "FLASHINFER"
        kwargs["enable_flashinfer_autotune"] = False
    elif name not in {
        "turboquant-4bit-nc",
        "turboquant-4bit-nc-all-layers",
        "bitdecoding-nvfp4",
        "bitdecoding-mxfp4",
    }:
        kwargs["attention_backend"] = "FLASHINFER"
    kwargs["max_num_seqs"] = seqs
    kwargs["max_num_batched_tokens"] = max(
        int(kwargs.get("max_num_batched_tokens") or 0), MAX_MODEL_LEN
    )
    mode["engine_kwargs"] = kwargs
    return mode

# Qwen thinking-mode sampling; use identical per-row seeds across cache modes.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
MODEL_THINKING_SAMPLING = {
    "qwen": THINKING_SAMPLING,
    "glm": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0},
}

# ``max_new_tokens`` may be this word instead of a number, meaning "whatever is
# left of the window once this row's prompt is in it". One constant cannot say
# that for a benchmark whose prompts run from 8k to 28.6k: set for the longest
# prompt it truncates the median row, set for the median row it overruns the
# window on the longest. Every arm sees the same prompts, so every arm gets the
# same per-row budget and the arms stay paired.
FILL_WINDOW = "fill_window"
# Slack for the chat template's own tokens, which ``prompt_tokens`` counts
# before the template is applied and so does not include.
GENERATION_MARGIN = 256

BENCH_DEFAULTS = {
    "longbench_pro": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 4096},
    "mrcr": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 4096},
    "aime": {**THINKING_SAMPLING, "max_new_tokens": 38_912},
    "aime25": {**THINKING_SAMPLING, "max_new_tokens": 38_912},
    "lcb": {**THINKING_SAMPLING, "max_new_tokens": 39_168},
    "ruler_niah": {**THINKING_SAMPLING, "max_new_tokens": 8_192},
    "gpqa": {**THINKING_SAMPLING, "max_new_tokens": 81_920},
}

# Thinking-mode sampling for the greedy benchmarks. Selected by --thinking
# rather than by redefining the greedy defaults above.
THINKING_BENCH_OVERRIDES = {
    "longbench_pro": {**THINKING_SAMPLING, "max_new_tokens": FILL_WINDOW},
    "mrcr": {**THINKING_SAMPLING, "max_new_tokens": FILL_WINDOW},
    "ruler_niah": {**THINKING_SAMPLING, "max_new_tokens": 8_192},
    "gpqa": {**THINKING_SAMPLING, "max_new_tokens": 81_920},
}

def configure_data_dir(data_dir: str) -> None:
    """Resolve dataset paths and their staging checks from one root."""
    global DATA_DIR, LONGBENCH_PRO_DIR, MRCR_FILTERED_CACHE
    global AIME24_PARQUET, AIME25_PARQUET, LCB_PROMPTS, RULER_NIAH_PROMPTS
    global GPQA_PATH, BENCH_REQUIREMENTS

    DATA_DIR = os.path.abspath(os.path.expanduser(data_dir))
    LONGBENCH_PRO_DIR = os.path.join(DATA_DIR, "longbench_pro")
    MRCR_FILTERED_CACHE = os.environ.get(
        "BITDECODING_MRCR_FILTERED_CACHE",
        os.path.join(DATA_DIR, EVALUATION["mrcr_cache"]),
    )
    AIME24_PARQUET = os.path.join(DATA_DIR, "aime24", "aime24.parquet")
    AIME25_PARQUET = os.path.join(DATA_DIR, "aime25", "aime25.parquet")
    LCB_PROMPTS = os.path.join(DATA_DIR, "lcb_v6", "prompts.jsonl")
    RULER_NIAH_PROMPTS = os.path.join(DATA_DIR, "ruler", "niah_122880.jsonl")
    GPQA_PATH = os.environ.get(
        "BITDECODING_GPQA_PATH", os.path.join(DATA_DIR, "gpqa", "gpqa_diamond.parquet")
    )
    BENCH_REQUIREMENTS = {
        "longbench_pro": [("LongBench-Pro dataset", LONGBENCH_PRO_DIR, "dir")],
        "mrcr": [("MRCR filtered cache", MRCR_FILTERED_CACHE, "file")],
        "aime": [("AIME 2024 parquet", AIME24_PARQUET, "file")],
        "aime25": [("AIME 2025 parquet", AIME25_PARQUET, "file")],
        "lcb": [("LiveCodeBench v6 prompts", LCB_PROMPTS, "file")],
        "ruler_niah": [("RULER NIAH prompts", RULER_NIAH_PROMPTS, "file")],
        "gpqa": [("GPQA Diamond data", GPQA_PATH, "file")],
    }


configure_data_dir(DATA_DIR)
BENCH_DEPENDENCIES = {}

LONG_CONTEXT_BENCH_OVERRIDES = {
    "aime": {**THINKING_SAMPLING, "max_new_tokens": 81_920},
    "aime25": {**THINKING_SAMPLING, "max_new_tokens": 81_920},
    "lcb": {**THINKING_SAMPLING, "max_new_tokens": 81_920},
    "longbench_pro": {**THINKING_SAMPLING, "max_new_tokens": FILL_WINDOW},
    "mrcr": {**THINKING_SAMPLING, "max_new_tokens": FILL_WINDOW},
    "ruler_niah": {**THINKING_SAMPLING, "max_new_tokens": 8_192},
    "gpqa": {**THINKING_SAMPLING, "max_new_tokens": 81_920},
}


def _model_available() -> bool:
    """A local snapshot must hold a config.json; a bare Hub id vLLM downloads.

    A directory that is actually there is local whatever it is spelled like:
    ``_local/models/Qwen3-8B`` is neither absolute nor ``./``-prefixed, and
    reading it as a Hub id rejected a checkpoint that was sitting right there.
    A path that does not exist still falls through to the Hub-id shape, so a
    mistyped local path is reported as missing rather than downloaded.
    """
    if (os.path.isdir(MODEL_PATH) or os.path.isabs(MODEL_PATH)
            or MODEL_PATH.startswith((".", "~"))):
        return os.path.isfile(os.path.join(MODEL_PATH, "config.json"))
    return MODEL_PATH.count("/") == 1


def verify_prestaging(benchmarks=None) -> list:
    """Return a list of human-readable problems; empty list == ready to run.

    ``benchmarks`` restricts the check to what will actually be read. Passing
    nothing keeps the old behaviour of demanding everything.
    """
    problems = []
    if not _model_available():
        problems.append(f"model checkpoint missing at {MODEL_PATH} (set MODEL_PATH)")
    names = list(BENCH_REQUIREMENTS) if benchmarks is None else list(benchmarks)
    for name in names:
        for description, path, kind in BENCH_REQUIREMENTS.get(name, []):
            present = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
            if not present:
                problems.append(f"{description} missing at {path}")
    return problems


def verify_dependencies(benchmarks) -> list:
    """Problems with the packages the requested benchmarks score with."""
    import importlib.util

    problems = []
    for name in benchmarks:
        for module, purpose in BENCH_DEPENDENCIES.get(name, []):
            if importlib.util.find_spec(module) is None:
                problems.append(
                    f"benchmark {name!r} scores with {module!r} ({purpose}), "
                    "which is not installed"
                )
    return problems


def verify_sidecars(modes) -> list:
    """Problems with the calibration sidecars the requested modes name."""
    problems = []
    for mode in modes:
        entry = KV_MODES.get(mode, {})
        for key in ("key_shift",):
            path = entry.get(key)
            if isinstance(path, str) and path != "auto" and not os.path.isfile(path):
                problems.append(
                    f"mode {mode!r} wants the {key} sidecar at {path}, which is "
                    "not there (run the accuracy evaluation to calibrate it)"
                )
    return problems
