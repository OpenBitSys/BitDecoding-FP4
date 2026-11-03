"""Engine configuration the NVFP4 KV cache path refuses to run under.

These checks cannot live in ``AttentionBackend.validate_configuration``:
``_cached_get_attn_backend`` memoizes on ``AttentionSelectorConfig``, which
carries none of the fields below, so a second engine in the same process would
reuse the first one's verdict and never be validated. ``NVFP4Impl.__init__``
runs once per layer per engine and is not cached.
"""

from __future__ import annotations

from typing import NamedTuple

from nvfp4_decode_kernel.scale_mode import DEFAULT_SCALE_MODE, parse_scale_mode_spec, scale_format, scale_rule
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionImpl, AttentionType

from .control import MAX_SUPPORTED_SLOTS


NVFP4 = "nvfp4"

# Which block-scale candidates the quantizers search, read from vLLM's
# ``additional_config`` — the field vLLM documents for out-of-tree backend
# options, and the only place a value like this can come from that is neither
# an environment switch nor a fork. It joins the engine's config hash, so two
# engines that disagree about it cannot share a compiled artifact. Holds a
# mode name, or a per-tensor mapping of them; see ``resolve_scale_mode``.
SCALE_MODE_KEY = "nvfp4_scale_mode"

# Whether K derives its eight stored scales independently per 16 channels or
# repeats one norm-derived scale across the whole 128-channel vector. Both use
# the identical FP4 payload and E4M3 scale slots.
KEY_SCALE_GEOMETRY_KEY = "nvfp4_key_scale_geometry"
KEY_SCALE_GEOMETRIES = ("local", "gaussian_global")

# How many independently scaled NVFP4 terms represent each decode query.
# This is fixed before profiling because the second term owns persistent
# scratch buffers that must be charged before vLLM sizes the KV cache.
QUERY_TERMS_KEY = "nvfp4_query_terms"

# The shared orthogonal basis applied to post-RoPE Q and cache-bound K.
# ``none`` preserves the existing path exactly; ``hadamard`` uses one
# normalized Sylvester matrix for every layer and head.
ROTATION_KEY = "nvfp4_rotation"

# vLLM runs at most ``max_num_seqs`` requests concurrently, and each running
# request needs one BF16 tail slot. The slot table itself is sized from
# ``max_num_seqs``; this is the width v1 is validated at.
DEFAULT_SLOTS = 32
MAX_SLOTS = MAX_SUPPORTED_SLOTS


class UnsupportedConfigError(ValueError):
    """An engine configuration the NVFP4 path cannot serve correctly."""


class ScaleModes(NamedTuple):
    """One block-scale mode per quantized tensor."""

    key: str
    value: str
    query: str


def resolve_scale_mode(vllm_config: VllmConfig) -> ScaleModes:
    """The block-scale modes this engine quantizes K, V and Q with.

    A bare name gives all three the same one, and that is the only shape a
    served engine should use: a candidate set that differs by tensor makes a
    quality result unattributable, because the clipping the extra candidates
    trade for accuracy acts differently on a key than on a query.

    A ``{"key": ..., "value": ..., "query": ...}`` mapping sets them apart,
    which exists to answer *which* tensor a quality difference came from once
    the uniform arms have found one. All three names are required — a partial
    mapping would leave the rest at a default that is nobody's intent, and the
    resulting arm would look like a valid one. The precedent is ``num_splits``:
    a typed argument, one value in production, every branch reachable from a
    public entry point.

    Unset means ``amax6``, which is what every cache this repository has
    written so far holds. An unknown name is refused here rather than at the
    first quantized token, because by then the engine has loaded a model. So
    is a mapping that mixes NVFP4 and MXFP4: Q meets K in one MMA and P meets
    V in another, and the kernel runs both in one format.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return ScaleModes(*(DEFAULT_SCALE_MODE,) * 3)
    requested = additional.get(SCALE_MODE_KEY, DEFAULT_SCALE_MODE)

    if isinstance(requested, dict):
        if set(requested) != set(ScaleModes._fields):
            raise UnsupportedConfigError(
                f"additional_config[{SCALE_MODE_KEY!r}] as a mapping needs "
                f"exactly {sorted(ScaleModes._fields)}, got {sorted(requested)}"
            )
        modes = ScaleModes(**requested)
    else:
        modes = ScaleModes(*(requested,) * 3)

    for tensor, mode in zip(ScaleModes._fields, modes):
        try:
            scale_rule(mode)
        except (ValueError, TypeError) as error:
            raise UnsupportedConfigError(
                f"additional_config[{SCALE_MODE_KEY!r}] for {tensor}: {error}"
            ) from None
    formats = {scale_format(mode) for mode in modes}
    if len(formats) != 1:
        raise UnsupportedConfigError(
            f"additional_config[{SCALE_MODE_KEY!r}] mixes {sorted(formats)}; "
            "key, value and query must share one scale format"
        )
    return modes


def check_mxfp4_options(
    scale_modes: ScaleModes,
    *,
    key_scale_geometry: str,
) -> None:
    """Refuse the NVFP4-only cache options under an MXFP4 scale mode.

    Gaussian-global K scaling uses E4M3 scales and has no MXFP4 kernel.
    """
    if scale_format(scale_modes.key) != "mxfp4":
        return
    unsupported = {
        KEY_SCALE_GEOMETRY_KEY: key_scale_geometry != "local",
    }
    named = sorted(key for key, enabled in unsupported.items() if enabled)
    if named:
        raise UnsupportedConfigError(
            f"MXFP4 scale modes do not support {named} yet"
        )


def resolve_key_scale_geometry(vllm_config: VllmConfig) -> str:
    """How the K write path shares its already-stored scale slots."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return "local"
    requested = additional.get(KEY_SCALE_GEOMETRY_KEY, "local")
    if requested not in KEY_SCALE_GEOMETRIES:
        raise UnsupportedConfigError(
            f"additional_config[{KEY_SCALE_GEOMETRY_KEY!r}] must be one of "
            f"{list(KEY_SCALE_GEOMETRIES)}, got {requested!r}"
        )
    return requested


def resolve_query_terms(vllm_config: VllmConfig) -> int:
    """How many NVFP4 terms represent each decode query."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return 1
    requested = additional.get(QUERY_TERMS_KEY, 1)
    if type(requested) is not int or requested not in (1, 2):
        raise UnsupportedConfigError(
            f"additional_config[{QUERY_TERMS_KEY!r}] must be 1 or 2, got "
            f"{requested!r}"
        )
    return requested


def resolve_rotation(vllm_config: VllmConfig) -> str:
    """Which shared basis post-RoPE Q and cache-bound K use."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return "none"
    requested = additional.get(ROTATION_KEY, "none")
    if requested not in ("none", "hadamard", "block_hadamard_64"):
        raise UnsupportedConfigError(
            f"additional_config[{ROTATION_KEY!r}] must be 'none', "
            f"'hadamard', or 'block_hadamard_64', got {requested!r}"
        )
    return requested


def check_supported(vllm_config: VllmConfig) -> None:
    """Raise unless the engine is configured for the NVFP4 KV cache path.

    A no-op unless ``kv_cache_dtype`` is ``nvfp4``. With any other cache dtype
    the backend is a plain FlashAttention pass-through: none of the machinery
    these constraints protect is in play.
    """
    cache_config = vllm_config.cache_config
    if cache_config is None or cache_config.cache_dtype != NVFP4:
        return

    scheduler_config = vllm_config.scheduler_config
    if scheduler_config is not None and scheduler_config.max_num_seqs > MAX_SLOTS:
        raise UnsupportedConfigError(
            f"max_num_seqs={scheduler_config.max_num_seqs} exceeds the "
            f"{MAX_SLOTS} BF16 tail slots the NVFP4 path preallocates. Pass "
            f"max_num_seqs<={MAX_SLOTS}, or use a BF16 KV cache."
        )

    if cache_config.enable_prefix_caching:
        raise UnsupportedConfigError(
            "prefix caching is not supported by the NVFP4 KV cache: a cache "
            "hit resumes a sequence mid-page, and prefill over an FP4 prefix "
            "is not implemented. Pass enable_prefix_caching=False."
        )

    # Everything below keeps one invariant: a step either runs a whole prompt
    # from nothing, or extends a sequence by exactly one token. The control
    # plane reports a violation through ``error_code``, but only after the
    # tokens have gone somewhere wrong, so the configuration is refused first.
    if scheduler_config is not None and scheduler_config.enable_chunked_prefill:
        raise UnsupportedConfigError(
            "chunked prefill is not supported by the NVFP4 KV cache: the "
            "second chunk of a split prompt resumes mid-page, which needs "
            "prefill over an FP4 prefix. Pass enable_chunked_prefill=False; "
            "vLLM then requires max_num_batched_tokens >= max_model_len so "
            "that no prompt has to be split."
        )

    if scheduler_config is not None and scheduler_config.long_prefill_token_threshold:
        raise UnsupportedConfigError(
            "long_prefill_token_threshold="
            f"{scheduler_config.long_prefill_token_threshold} splits long "
            "prompts even with chunked prefill disabled. Leave it at 0."
        )

    if vllm_config.speculative_config is not None:
        raise UnsupportedConfigError(
            "speculative decoding is not supported by the NVFP4 KV cache: it "
            "verifies several draft tokens per step, and the decode kernel "
            "attends one query token per sequence."
        )

    if cache_config.kv_cache_dtype_skip_layers:
        raise UnsupportedConfigError(
            "kv_cache_dtype_skip_layers="
            f"{cache_config.kv_cache_dtype_skip_layers} leaves some layers on "
            "a different cache dtype. That splits the model into KV cache "
            "groups with their own metadata builders, so the slot table would "
            "advance more than once per step, and it makes vLLM zero freshly "
            "allocated blocks. Leave it empty."
        )

    if cache_config.kv_offloading_size is not None:
        raise UnsupportedConfigError(
            "KV offloading is not supported by the NVFP4 KV cache: it copies "
            "pages by address and size, which does not describe the packed "
            "FP4 layout. Leave kv_offloading_size unset."
        )

    parallel_config = vllm_config.parallel_config
    if parallel_config is not None and parallel_config.use_ubatching:
        raise UnsupportedConfigError(
            "microbatching is not supported by the NVFP4 KV cache: each "
            "ubatch builds its own attention metadata, so the slot table "
            "would advance twice per step and the second half of the batch "
            "would extend tails the first half had already extended. Leave "
            "enable_dbo unset and ubatch_size at 0."
        )

    if parallel_config is not None and parallel_config.pipeline_parallel_size > 1:
        raise UnsupportedConfigError(
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size} "
            "is not supported by the NVFP4 KV cache: page promotion is driven "
            "from the last attention layer, which a pipeline stage does not "
            "own. Pass pipeline_parallel_size=1."
        )


def check_layer_supported(impl: AttentionImpl) -> None:
    """Raise unless this attention layer is one the decode kernel can serve.

    Separate from ``check_supported`` because these come from the model rather
    than the engine configuration. Each of them is a modifier the decode kernel
    does not implement, so serving the layer would return plausible attention
    computed against the wrong mask or the wrong scores — wrong quietly, which
    is the failure mode worth spending a check on.
    """
    if impl.attn_type != AttentionType.DECODER:
        raise UnsupportedConfigError(
            f"attention type {impl.attn_type} is not supported by the NVFP4 "
            "KV cache, which is a decoder cache."
        )
    if impl.kv_sharing_target_layer_name is not None:
        raise UnsupportedConfigError(
            "cross-layer KV sharing is not supported by the NVFP4 KV cache: "
            "the BF16 tail is indexed by layer, so a layer that reads another "
            "layer's cache would read its own empty tail."
        )
    # The two above each fail for their own reason and say so. These four fail
    # for one reason, so they are reported together rather than one per run.
    unsupported = {
        "sliding window": impl.sliding_window != (-1, -1),
        "logit soft capping": bool(impl.logits_soft_cap),
        "ALiBi": impl.alibi_slopes is not None,
        "attention sinks": impl.sinks is not None,
    }
    named = [name for name, present in unsupported.items() if present]
    if named:
        raise UnsupportedConfigError(
            f"{', '.join(named)} is not supported by the NVFP4 decode kernel, "
            "which computes unmodified scaled dot-product attention over the "
            "whole cache."
        )
