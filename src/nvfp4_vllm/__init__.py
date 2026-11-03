"""Out-of-tree NVFP4 KV cache backend for vLLM V1."""


def register() -> None:
    """Entry point for the ``vllm.general_plugins`` group."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "nvfp4_vllm.backend.NVFP4Backend",
    )


# All three are imported lazily so that ``register`` stays free of torch and
# vLLM, which the plugin entry point is loaded too early to depend on.


def build_llm(model, **kwargs):
    """See ``nvfp4_vllm.engine.build_llm``."""
    from .engine import build_llm as _build

    return _build(model, **kwargs)


def calibrate_key_smoothing(llm, **kwargs):
    """Calibrate static key smoothing, optionally fitting the probability threshold."""
    from .calibrate import calibrate_key_shift as _calibrate

    return _calibrate(llm, **kwargs)


def set_key_smoothing_enabled(llm, enabled: bool):
    """Enable or disable an engine's calibrated static key smoothing."""
    from .calibrate import set_key_shift_enabled as _set

    return _set(llm, enabled)


__all__ = [
    "register",
    "build_llm",
    "calibrate_key_smoothing",
    "set_key_smoothing_enabled",
]
