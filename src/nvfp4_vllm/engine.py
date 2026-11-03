"""Build a vLLM engine with the paper's static key smoothing."""

from __future__ import annotations

import os
from typing import Any

from .backend import PAGE_SIZE
from .calibrate import WORKER_EXTENSION_CLS, calibrate_key_shift
from .guards import DEFAULT_SLOTS
from .runtime import KEY_SHIFT_ENV


REQUIRED_ENGINE_KWARGS: dict[str, Any] = {
    "kv_cache_dtype": "nvfp4",
    "attention_config": {"backend": "CUSTOM"},
    "block_size": PAGE_SIZE,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
}


def build_llm(
    model: str, *, additional_config: dict[str, Any] | None = None, **kwargs: Any
):
    """Build the FP4 backend with options in ``additional_config``.

    ``nvfp4_key_smoothing`` accepts ``"auto"`` (the default), a calibration
    file path, or ``None`` to disable smoothing. Automatic calibration measures
    the static per-head key mean before serving requests.
    """
    additional = dict(additional_config or {})
    smoothing = additional.setdefault("nvfp4_key_smoothing", "auto")
    if smoothing is not None:
        smoothing = os.fsdecode(smoothing)
    additional["nvfp4_key_smoothing"] = smoothing
    # Source checkouts may not have the installed plugin entry point.
    from . import register as register_custom_backend

    register_custom_backend()

    # CuTe may initialize the CUDA driver before PyTorch detects it. Spawn
    # workers so they do not inherit that driver state through fork.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM

    contradicted = {
        name: kwargs[name]
        for name, required in REQUIRED_ENGINE_KWARGS.items()
        if name in kwargs and kwargs[name] != required
    }
    if contradicted:
        wanted = {name: REQUIRED_ENGINE_KWARGS[name] for name in contradicted}
        raise ValueError(
            f"this backend requires {wanted}, and was given {contradicted}; "
            "an engine holding only some of these settings still starts, and "
            "serves from a different cache"
        )

    engine = {**REQUIRED_ENGINE_KWARGS, **kwargs, "model": model,
              "additional_config": additional}
    # Reserve one BF16 tail slot per concurrent request.
    engine.setdefault("max_num_seqs", DEFAULT_SLOTS)
    # With chunking disabled, the entire prompt must fit in one batch.
    if engine.get("max_model_len") is not None:
        engine.setdefault(
            "max_num_batched_tokens",
            max(engine["max_model_len"], engine["max_num_seqs"]),
        )

    # Workers read the calibration path during engine construction.
    if smoothing is None or smoothing == "auto":
        os.environ.pop(KEY_SHIFT_ENV, None)
    else:
        os.environ[KEY_SHIFT_ENV] = smoothing

    supplied = engine.get("worker_extension_cls")
    if supplied is None:
        # Also provides the smoothing toggle for saved calibrations.
        engine["worker_extension_cls"] = WORKER_EXTENSION_CLS
    elif smoothing == "auto" and supplied != WORKER_EXTENSION_CLS:
        raise ValueError(
            f"calibrating key smoothing needs worker_extension_cls="
            f"{WORKER_EXTENSION_CLS!r}, and vLLM accepts one; either pass a "
            "sidecar path instead or mix that class into your own"
        )

    llm = LLM(**engine)
    if smoothing == "auto" and calibrate_key_shift(llm) is None:
        raise RuntimeError(
            "the engine came up without the NVFP4 cache, so there was no K "
            "mean to measure"
        )
    return llm
