"""Calibrate static key smoothing and optionally the probability threshold.

Model-generated text is tiled across the context window to sample post-RoPE
keys at representative positions. Subtracting their per-channel mean reduces
quantization error without changing softmax attention.

Calibration completes before serving so every cached key uses the same mean.
Named ``collective_rpc`` methods run calibration in each worker and save the
sidecar there, avoiding tensor serialization through the engine client.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch


# vLLM mixes this into the worker, so ``collective_rpc`` can name its methods.
# An engine that wants to calibrate must be built with
# ``worker_extension_cls="nvfp4_vllm.calibrate.NVFP4WorkerExtension"``.
WORKER_EXTENSION_CLS = "nvfp4_vllm.calibrate.NVFP4WorkerExtension"


SEEDS = (
    "Explain how a hash table resolves collisions, and when each strategy "
    "is preferable.",
    "Summarise the causes of the 1929 financial crash for a reader who "
    "knows no economics.",
    "Write a short story about a lighthouse keeper who stops receiving "
    "supply ships.",
    "Derive the quadratic formula from completing the square, showing every "
    "step.",
    "Describe how photosynthesis converts light into chemical energy, at the "
    "level of a first-year undergraduate.",
    "Compare pessimistic and optimistic concurrency control in databases.",
    "Translate the idea of technical debt into terms a non-engineer manager "
    "would act on.",
    "Walk through debugging a service that is slow only under high load.",
)
TOKENS_PER_SEED = 256
# Calibration spans this many positions, clamped to the engine's window.
CONTEXT_TOKENS = 32_768


def _runtimes(model: torch.nn.Module) -> list[Any]:
    """The distinct LayerRuntime objects this model's attention layers share."""
    from .impl import NVFP4Impl

    seen: list[Any] = []
    for module in model.modules():
        impl = getattr(module, "impl", None)
        runtime = getattr(impl, "runtime", None)
        if isinstance(impl, NVFP4Impl) and runtime is not None:
            if not any(runtime is other for other in seen):
                seen.append(runtime)
    return seen


class NVFP4WorkerExtension:
    """Named RPC methods for calibration and cache inspection.

    Workers save calibration tensors locally and return only summary metadata.
    """

    def nvfp4_layer_cache_contract(self) -> dict[str, dict[str, int]]:
        """Attest every loaded attention layer through a named safe RPC."""
        layer_cache_dtypes = Counter()
        layer_attention_backends = Counter()
        for module in self.get_model().modules():
            layer_dtype = getattr(module, "kv_cache_dtype", None)
            backend = getattr(module, "attn_backend", None)
            if layer_dtype is None or backend is None:
                continue
            layer_cache_dtypes[str(layer_dtype)] += 1
            layer_attention_backends[str(backend.get_name())] += 1
        return {
            "layer_cache_dtype_counts": dict(layer_cache_dtypes),
            "layer_attention_backend_counts": dict(layer_attention_backends),
        }

    def nvfp4_key_shift_is_active(self) -> bool:
        return all(
            runtime.active_key_shift is not None
            for runtime in _runtimes(self.get_model())
        )

    def nvfp4_set_key_shift_enabled(self, enabled: bool) -> int:
        runtimes = _runtimes(self.get_model())
        for runtime in runtimes:
            runtime.set_key_shift_enabled(enabled)
        return len(runtimes)

    def nvfp4_begin_key_shift_measurement(self) -> int:
        runtimes = _runtimes(self.get_model())
        for runtime in runtimes:
            runtime.begin_key_shift_measurement()
        return len(runtimes)

    def nvfp4_begin_probability_calibration(
        self, tokens: int, multiplier: float, samples: int
    ) -> None:
        from .probability_calibration import ProbabilityCalibration

        if hasattr(self, "_probability_calibration"):
            raise RuntimeError("probability calibration is already active")
        collector = ProbabilityCalibration(tokens, multiplier, samples)
        collector.install(self.get_model())
        self._probability_calibration = collector

    def nvfp4_end_probability_calibration(self, complete: bool = True) -> list[dict] | None:
        collector = self._probability_calibration
        try:
            return collector.finish() if complete else None
        finally:
            collector.close()
            del self._probability_calibration

    def nvfp4_finish_key_shift_measurement(
        self, save_to: str | None = None
    ) -> dict[str, Any] | None:
        runtimes = _runtimes(self.get_model())
        if not runtimes:
            return None
        tokens = runtimes[0].key_shift_tokens
        shift = runtimes[0].finish_key_shift_measurement()
        for other in runtimes[1:]:
            other.finish_key_shift_measurement()

        if save_to is not None:
            from safetensors.torch import save_file

            writer = True
            if torch.distributed.is_initialized():
                from vllm.distributed import get_tp_group

                tp = get_tp_group()
                if tp.world_size > 1:
                    total_heads = (
                        self.vllm_config.model_config.get_total_num_kv_heads()
                    )
                    if total_heads != shift.shape[1] * tp.world_size:
                        raise ValueError(
                            "key-shift export requires unreplicated, evenly "
                            "partitioned KV heads"
                        )
                    counts = tp.all_gather(
                        torch.tensor([tokens], dtype=torch.int64, device=shift.device),
                        dim=0,
                    )
                    if not bool((counts == tokens).all()):
                        raise ValueError(
                            "TP workers disagree on key-shift measurement token counts"
                        )
                    # vLLM partitions unreplicated KV heads contiguously in
                    # TP-rank order. Gather the original-basis means, before
                    # the loader applies its rotation and dtype conversion.
                    shift = tp.all_gather(shift, dim=1)
                    writer = tp.rank_in_group == 0
            if writer:
                save_file(
                    {"key_shift": shift.cpu()},
                    save_to,
                    metadata={"schema": "1", "source": "calibrate_key_shift"},
                )
        return {
            "tokens": int(tokens),
            "shape": list(shift.shape),
            "mean_abs": float(shift.abs().mean()),
            "saved_to": save_to,
        }


def set_key_shift_enabled(llm, enabled: bool) -> bool:
    """Turn the shift on or off, for measuring one arm against the other.

    The vector survives being turned off, so a single loaded model can answer
    a benchmark both ways. Returns whether the engine had a shift to switch.

    Safe only between requests. Keys already cached were written under
    whichever setting was in force at the time, and a sequence spanning a flip
    would see two constants in one softmax. Calling this between two
    ``llm.generate`` calls satisfies that, since those are synchronous and the
    cache is dropped with the requests that filled it.
    """
    switched = llm.collective_rpc(
        "nvfp4_set_key_shift_enabled", args=(enabled,)
    )
    llm.reset_prefix_cache()
    return any(switched)


def _long_prompt(llm, texts: list[str], context_tokens: int) -> list[int]:
    """One prompt of about ``context_tokens`` tokens, tiled from ``texts``.

    Tiling repeats the sample rather than extending it, which is fine for a
    mean: the point of the long prompt is to visit high positions, and every
    position gets visited once whatever sits there.
    """
    if not any(text.strip() for text in texts):
        raise ValueError(
            "the model returned no calibration text to tile, so the mean "
            "would be measured over separators"
        )
    unit = llm.get_tokenizer().encode(
        "\n\n".join(texts) + "\n\n", add_special_tokens=False
    )
    repeats = -(-context_tokens // len(unit))
    return (unit * repeats)[:context_tokens]


def calibrate_key_shift(
    llm,
    *,
    seeds: tuple[str, ...] = SEEDS,
    tokens_per_seed: int = TOKENS_PER_SEED,
    context_tokens: int = CONTEXT_TOKENS,
    save_to: str | None = None,
    probability_multiplier: float | None = None,
    probability_samples: int = 8,
) -> dict[str, Any] | None:
    """Calibrate static key smoothing from a long prefill.

    Returns what was measured -- token count, shape, mean magnitude -- or
    ``None`` if the engine is not running the NVFP4 cache and there was
    nothing to calibrate.

    ``save_to`` writes a reusable sidecar. Pass its path as
    ``additional_config["nvfp4_key_smoothing"]`` when building another engine
    for the same checkpoint. Unreplicated TP head slices are gathered in rank
    order and only TP rank zero writes the global sidecar. Export requires one
    TP group, without pipeline/data parallelism. Without saving, each worker
    keeps its own mean.

    With ``probability_multiplier`` set, also fit the probability threshold
    tau from BF16 Q/K samples in the same long prefill. This requires an eager
    calibration engine so attention hooks execute. The returned
    ``probability_scaling`` report uses leave-one-query-position-out checks.
    """
    from vllm import SamplingParams, TokensPrompt

    if probability_multiplier is not None and not llm.llm_engine.vllm_config.model_config.enforce_eager:
        raise ValueError("probability calibration requires an eager engine")

    try:
        found = llm.collective_rpc("nvfp4_begin_key_shift_measurement")
    except AttributeError as error:
        raise RuntimeError(
            "the engine was built without "
            f'worker_extension_cls="{WORKER_EXTENSION_CLS}", so the K mean '
            "cannot be read out of the worker"
        ) from error
    if not any(found):
        return None
    if save_to is not None:
        config = llm.llm_engine.vllm_config
        parallel = config.parallel_config
        tp_size = parallel.tensor_parallel_size
        heads = config.model_config.get_total_num_kv_heads()
        if (
            not all(found)
            or len(found) != tp_size
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or heads < tp_size
            or heads % tp_size
        ):
            raise ValueError(
                "key-shift export requires one TP group with unreplicated, "
                "evenly partitioned KV heads and no pipeline/data parallelism"
            )

    sampled = llm.generate(
        list(seeds),
        SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=tokens_per_seed, seed=42
        ),
        use_tqdm=False,
    )

    budget = min(
        context_tokens, llm.llm_engine.model_config.max_model_len - 1
    )
    if probability_multiplier is not None:
        llm.collective_rpc("nvfp4_begin_probability_calibration",
                           args=(budget, probability_multiplier, probability_samples))
    try:
        llm.generate(
            TokensPrompt(
                prompt_token_ids=_long_prompt(
                    llm,
                    [output.prompt + output.outputs[0].text for output in sampled],
                    budget,
                )
            ),
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )
    except BaseException:
        if probability_multiplier is not None:
            llm.collective_rpc("nvfp4_end_probability_calibration", args=(False,))
        raise
    probability_report = None
    if probability_multiplier is not None:
        from .probability_calibration import fit_probability_threshold

        samples = llm.collective_rpc("nvfp4_end_probability_calibration")
        if len(samples) != len(found) or any(
            [row["position"] for row in worker] != [row["position"] for row in samples[0]]
            for worker in samples
        ):
            raise ValueError("TP workers disagree on probability calibration samples")
        probability_report = fit_probability_threshold(samples, probability_multiplier)

    measurements = llm.collective_rpc(
        "nvfp4_finish_key_shift_measurement", args=(save_to,)
    )
    measured = measurements[0]
    if save_to is not None and len(found) > 1:
        if len(measurements) != len(found) or any(
            item is None
            or item["shape"] != measured["shape"]
            or item["tokens"] != measured["tokens"]
            or item["saved_to"] != save_to
            for item in measurements
        ):
            raise ValueError("TP workers disagree on the exported key-shift measurement")
    if probability_report is not None:
        measured = {**measured, "probability_scaling": probability_report}

    # Every key cached during the measurement was stored unshifted, and the
    # next request must not read one. With prefix caching refused -- which the
    # NVFP4 engine configuration requires anyway -- a finished request leaves
    # nothing behind, but saying so costs nothing and does not depend on that.
    llm.reset_prefix_cache()
    return measured
