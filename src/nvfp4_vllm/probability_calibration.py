"""Fit the probability-scale threshold from bounded calibration Q/K samples."""

from __future__ import annotations

import math
import struct


def _e4m3_value(code: int) -> float:
    exponent, mantissa = code >> 3, code & 7
    return mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)


def threshold_for_code(code: int) -> float:
    """Largest FP32 cut matching E4M3 round-to-nearest-even codes <= code."""
    if code == -1:
        return 0.0
    if not 0 <= code < 126:
        raise ValueError("threshold code must be in [-1, 125]")
    midpoint = (_e4m3_value(code) + _e4m3_value(code + 1)) / 2
    if code % 2:
        bits = struct.unpack("!I", struct.pack("!f", midpoint))[0]
        return struct.unpack("!f", struct.pack("!I", bits - 1))[0]
    return midpoint


def _combine(samples: list[dict]) -> dict:
    total = {}
    for sample in samples:
        for code, bucket in sample["codes"].items():
            target = total.setdefault(int(code), {"groups": 0, "2": 0.0, "6": 0.0})
            target["groups"] += bucket["groups"]
            for divisor in ("2", "6"):
                target[divisor] += bucket["error_sum"][divisor]
    return total


def _error(buckets: dict, cut: int) -> float:
    return math.fsum(bucket["2" if code <= cut else "6"] for code, bucket in buckets.items())


def _best_cut(buckets: dict) -> int:
    return min(range(-1, max(buckets) + 1), key=lambda code: (_error(buckets, code), code))


def fit_probability_threshold(worker_samples: list[list[dict]], multiplier: float) -> dict:
    """Fit tau on all samples and report leave-one-query-position-out error."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("probability multiplier must be positive and finite")
    samples = {}
    for worker in worker_samples:
        for sample in worker:
            samples.setdefault(sample["position"], []).append(sample)
    if len(samples) < 2:
        raise ValueError("tau calibration needs at least two query positions")
    grouped = {position: _combine(values) for position, values in samples.items()}
    for buckets in grouped.values():
        if not buckets or any(
            not 0 <= code < 126 or bucket["groups"] < 1
            or any(not math.isfinite(bucket[d]) or bucket[d] < 0 for d in ("2", "6"))
            for code, bucket in buckets.items()
        ):
            raise ValueError("invalid probability calibration statistics")
    total = _combine([sample for worker in worker_samples for sample in worker])
    cut = _best_cut(total)
    folds = []
    for position, held in sorted(grouped.items()):
        training = {
            code: {key: value - held.get(code, {}).get(key, 0)
                   for key, value in bucket.items()}
            for code, bucket in total.items()
        }
        training_cut = _best_cut(training)
        folds.append({"position": position, "tau": threshold_for_code(training_cut),
                      "squared_error": _error(held, training_cut) / multiplier**2})
    return {
        "tau": threshold_for_code(cut), "c": multiplier,
        "rule": "max(p) <= tau / c", "threshold_code": cut,
        "samples": len(samples), "groups": sum(bucket["groups"] for bucket in total.values()),
        "squared_error": _error(total, cut) / multiplier**2,
        "amax6_squared_error": math.fsum(bucket["6"] for bucket in total.values()) / multiplier**2,
        "validation": "leave_one_query_position_out", "folds": folds,
        "source": "BF16 Q/K, reverse-page online softmax, 16-value probability groups",
    }


class ProbabilityCalibration:
    """Observe one unchunked prefill and retain only per-code error sums."""

    def __init__(self, tokens: int, multiplier: float, samples: int = 8):
        if tokens < 256 or not 2 <= samples <= tokens - 128:
            raise ValueError("tau calibration needs at least 256 tokens and 2..tokens-128 samples")
        if not math.isfinite(multiplier) or not 0 < multiplier <= 448:
            raise ValueError("probability multiplier must be finite and between 0 and 448")
        self.tokens = tokens
        self.multiplier = multiplier
        self.positions = [128 + i * (tokens - 129) // (samples - 1) for i in range(samples)]
        self.samples = {position: {"position": position, "codes": {}} for position in self.positions}
        self.handles = []
        self.seen = set()
        self.expected_layers = 0

    def install(self, model) -> None:
        from vllm.model_executor.layers.attention.attention import Attention

        try:
            for module in model.modules():
                if isinstance(module, Attention):
                    self.handles.append(module.register_forward_pre_hook(self.observe, with_kwargs=True))
            self.expected_layers = len(self.handles)
            if not self.expected_layers:
                raise ValueError("no attention layers found for probability calibration")
        except BaseException:
            self.close()
            raise

    def observe(self, module, args, kwargs) -> None:
        import torch
        from nvfp4_decode_kernel.reference import online_probability_scale_statistics

        if id(module) in self.seen:
            raise RuntimeError("tau calibration requires one complete prefill per attention layer")
        query = args[0] if args else kwargs["query"]
        key = args[1] if len(args) > 1 else kwargs["key"]
        if query.shape[0] < self.tokens or key.shape[0] < self.tokens:
            raise ValueError("tau calibration requires an unchunked prefill")
        heads, kv_heads, dim = module.num_heads, module.num_kv_heads, module.head_size
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or heads % kv_heads:
            raise ValueError("tau calibration requires BF16 Q/K and grouped-query attention")
        query = query[:self.tokens].reshape(self.tokens, heads, dim)
        key = key[:self.tokens].reshape(self.tokens, kv_heads, dim).float()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            for position in self.positions:
                selected = query[position].float().reshape(kv_heads, heads // kv_heads, dim)
                scores = torch.einsum("hrd,thd->hrt", selected, key[:position + 1]).reshape(heads, position + 1)
                stats = online_probability_scale_statistics(
                    scores, fp4_tokens=(position // 128) * 128, head_dim=dim,
                    probability_multiplier=self.multiplier,
                )
                target = self.samples[position]["codes"]
                for code, bucket in stats["by_amax_e4m3_code"].items():
                    merged = target.setdefault(code, {"groups": 0, "error_sum": {"2": 0.0, "6": 0.0}})
                    merged["groups"] += bucket["groups"]
                    for divisor in ("2", "6"):
                        merged["error_sum"][divisor] += bucket["error_sum"][divisor]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        self.seen.add(id(module))

    def finish(self) -> list[dict]:
        if len(self.seen) != self.expected_layers:
            raise RuntimeError("not every attention layer supplied probability calibration samples")
        return list(self.samples.values())

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
