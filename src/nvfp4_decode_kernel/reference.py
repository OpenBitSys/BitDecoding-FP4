"""PyTorch reference operations for the kernel's NVFP4 quantization.

This mirrors the CuTeDSL quantizers arithmetically, not merely closely. E2M1
rounding breaks ties towards the even code as ``cvt.rn.satfinite.e2m1x2.f32``
does, a scale is the reciprocal multiply the kernel performs rather than a
division, and the squared error a candidate is judged by accumulates through
fused multiply-add in element order. Each of those is a last-place difference
on its own, but they feed an argmin whose losers are sometimes a hair behind,
so any one of them diverging turns into the two implementations storing
different bytes.
"""

from __future__ import annotations

import math

import torch

from .p_scale import P_SMALL_GROUP_AMAX
from .scale_mode import DEFAULT_SCALE_MODE, candidates


# Magnitudes an E2M1 code can take, indexed by the code's low three bits.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Midpoint ``m`` separates codes ``m`` and ``m + 1``, so round-to-nearest-even
# sends a tie down at an even midpoint and up at an odd one.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

# E4M3's largest finite value. A scale is clamped here before being cast,
# because ``cvt.rn.satfinite.e4m3x2.f32`` saturates and Torch does not agree
# with it across versions — 2.11 returns NaN for anything that rounds past the
# top, 2.13 saturates. Clamping first reproduces the instruction on either: a
# non-negative value that would round past 448 is exactly one that saturates
# there, and a scale is never negative.
_E4M3_MAX = 448.0

_PROBABILITY_SCALE_MODES = {
    "two_level": (6.0,),
    "amax2_or_amax6_c8": (2.0, 6.0),
    "amax2": (2.0,),
    "amax3": (3.0,),
    "amax4": (4.0,),
    # Map a group's maximum to each useful interior/top E2M1 rung. Keep the
    # production amax/6 choice first so it wins exact ties.
    "e2m1_max_grid": (6.0, 4.0, 3.0, 2.0),
}


def _e2m1_boundaries(device: torch.device) -> torch.Tensor:
    """Midpoints arranged so that ``bucketize`` breaks ties towards even.

    ``bucketize`` sends a value equal to a boundary into the lower bucket, so
    the odd midpoints are moved one float below the tie to send it the other
    way. One float, in float32, because that is the width the comparison
    happens in — anything wider rounds back onto the tie.
    """
    midpoints = torch.tensor(
        _E2M1_MIDPOINTS, dtype=torch.float32, device=device
    )
    below = torch.nextafter(midpoints, torch.zeros_like(midpoints))
    odd = torch.arange(midpoints.numel(), device=device) % 2 == 1
    return torch.where(odd, below, midpoints)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round values to the nearest representable E2M1 value, ties to even.

    Magnitudes past 6 saturate there rather than overflowing, which is the
    ``satfinite`` half of the instruction's name and what the zero-scale guard
    in :func:`nvfp4_quantize` relies on.
    """
    magnitudes = torch.tensor(
        _E2M1_MAGNITUDES, dtype=torch.float32, device=x.device
    )
    indices = torch.bucketize(
        x.abs(), _e2m1_boundaries(x.device), right=False, out_int32=True
    )
    return torch.sign(x) * magnitudes[indices]


def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """The four-bit code each value rounds to, sign in bit 3.

    Same bucketing as :func:`round_e2m1`, which is the point: a writer that
    derived the code independently could disagree with the reconstruction in
    the one place ties are broken. Negative values that round to zero get code
    8, the negative zero the hardware conversion also emits; it reads back as
    the same magnitude, so a byte comparison against a path that emitted 0
    there is comparing an encoding rather than a value.
    """
    indices = torch.bucketize(
        x.abs(), _e2m1_boundaries(x.device), right=False, out_int32=True
    )
    return (indices | torch.where(x < 0, 8, 0)).to(torch.uint8)


def _accumulate_squares(difference: torch.Tensor) -> torch.Tensor:
    """A group's squared error, accumulated the way the kernel accumulates it.

    The kernel spells each step ``fma.rn.f32``: one rounding per element,
    taken in element order. Both halves of that matter. A reduction free to
    reassociate lands a last place away, and so does squaring into float32
    before adding, and either is enough to swap two candidates that are
    otherwise level.

    Float64 reproduces the instruction rather than approximating it. The
    product of two float32 values is exact in float64, and rounding the
    float64 sum back to float32 agrees with a single rounding of the exact
    value, because float64's 53 bits clear the 2p+2 that makes the double
    rounding harmless at p=24.
    """
    # An MX group is owned by two lanes, each accumulating sixteen values.
    # Keep that single final addition in the reference as well as in K/V's
    # one-thread quantizer, so all three select the same near-tie exponent.
    if difference.shape[-1] == 32:
        return _accumulate_squares(difference[..., :16]) + _accumulate_squares(
            difference[..., 16:]
        )
    total = torch.zeros_like(difference[..., :1])
    wide = difference.double()
    for element in range(difference.shape[-1]):
        term = wide[..., element : element + 1]
        total = (total.double() + term * term).float()
    return total


def _apply_candidate(
    groups: torch.Tensor,
    amax: torch.Tensor,
    divisor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One candidate's scale and its round trip."""
    # The kernel holds 1/divisor as a float32 constant and multiplies by it, so
    # dividing here would differ in the last place on some amax and, through
    # E4M3's three mantissa bits, occasionally in the byte that gets stored.
    reciprocal = torch.tensor(1.0 / divisor, dtype=torch.float32).item()
    scale = (
        (amax * reciprocal).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn).float()
    )
    # The kernel's guard against a scale that flushed to zero. Every nonzero
    # element is then thrown past the top of the grid, where E2M1 saturates at
    # 6, and multiplying by the zero scale brings the group back to zero. Zero
    # is what the decode path reads back in that case, so it is what a
    # reference has to produce.
    return scale, round_e2m1(groups * (1.0 / (scale + 1.0e-30))) * scale


def nvfp4_quantize(
    x: torch.Tensor,
    group: int = 16,
    scale_mode: str = DEFAULT_SCALE_MODE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize groups on the last axis, each with the scale that costs least.

    Returns the round-tripped values, the E4M3 scale every group settled on,
    and which candidate that was as an index into ``scale_mode``'s tuple.

    No candidate is screened for range. E4M3 saturates at 448 and flushes below
    ``2**-9``, so a divisor too small for the group leaves a scale that cannot
    represent it and an error that says as much. Judging every candidate by the
    scale it would actually store lets the argmin drop those without a branch,
    and it never drops one that would have won.
    """
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    if x.shape[-1] % group != 0:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by group {group}"
        )

    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    amax = groups.abs().amax(dim=-1, keepdim=True)

    best_scale, best_values, best_error, winner = None, None, None, None
    divisors = candidates(scale_mode)
    for index, divisor in enumerate(divisors):
        scale, values = _apply_candidate(groups, amax, divisor)
        if len(divisors) == 1:
            return (
                values.reshape_as(x),
                scale.squeeze(-1).to(torch.float8_e4m3fn),
                torch.zeros_like(amax.squeeze(-1), dtype=torch.int64),
            )
        # An E2M1 magnitude carries two significant bits and an E4M3 scale
        # four, so their product is exact in float32 and this subtraction is
        # the kernel's ``fma(magnitude, scale, -x)`` to the bit.
        error = _accumulate_squares(values - groups)
        if best_error is None:
            best_scale, best_values, best_error = scale, values, error
            winner = torch.zeros_like(error, dtype=torch.int64)
            continue
        # Strictly better, so a tie leaves the earlier candidate in place.
        better = error < best_error
        best_scale = torch.where(better, scale, best_scale)
        best_values = torch.where(better, values, best_values)
        best_error = torch.where(better, error, best_error)
        winner = torch.where(better, torch.full_like(winner, index), winner)

    return (
        best_values.reshape_as(x),
        best_scale.squeeze(-1).to(torch.float8_e4m3fn),
        winner.squeeze(-1),
    )


def nvfp4_round_trip(
    x: torch.Tensor,
    group: int = 16,
    scale_mode: str = DEFAULT_SCALE_MODE,
) -> torch.Tensor:
    """Quantize and dequantize groups on the last axis with NVFP4."""
    values, _, _ = nvfp4_quantize(x, group=group, scale_mode=scale_mode)
    return values


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """The last axis times the normalized Sylvester Hadamard, as the kernels do.

    FP32 butterflies ``(a, c) -> (a + c, a - c)`` over the index bits, lowest
    first, then one multiply by ``1 / sqrt(n)``. That is the query quantizer's
    and the K writer's order and rounding, so their FP32 results match this
    bit for bit; a matrix product sums in another order and does not.
    """
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    leading = x.shape[:-1]
    values = x.float()
    stride = 1
    while stride < n:
        pairs = values.reshape(*leading, n // (2 * stride), 2, stride)
        low, high = pairs[..., 0, :], pairs[..., 1, :]
        values = torch.stack((low + high, low - high), dim=-2)
        stride *= 2
    inverse = torch.tensor(
        1.0 / math.sqrt(n), dtype=torch.float32, device=values.device
    )
    return values.reshape(*leading, n) * inverse


def nvfp4_two_term(
    x: torch.Tensor,
    group: int = 16,
    scale_mode: str = DEFAULT_SCALE_MODE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two NVFP4 reconstructions whose sum approximates ``x``."""
    first = nvfp4_round_trip(x, group=group, scale_mode=scale_mode)
    second = nvfp4_round_trip(
        x.float() - first,
        group=group,
        scale_mode=scale_mode,
    )
    return first, second


# OCP MXFP4: 32 values share one E8M0 scale, a power of two 2**e stored as the
# byte e + 127 (255 is NaN). E2M1's largest value is 1.5 * 2**2, so the OCP
# rule e = floor(log2(amax)) - 2 maps amax into [4, 8) and saturates what lands
# above 6.
MX_GROUP = 32
MX_SCALE_RULES = ("ocp", "round_up", "search")
_E2M1_MAX_EXPONENT = 2
_E8M0_MIN_EXPONENT = -127
_E8M0_MAX_EXPONENT = 127


def _mx_round_trip(groups: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """``groups`` rounded on the power-of-two scale ``2**exponent``."""
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    return round_e2m1(groups / scale) * scale


def mxfp4_quantize(
    x: torch.Tensor, rule: str = "ocp"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize groups of 32 on the last axis to MXFP4.

    ``rule`` picks each group's exponent: ``ocp`` is the OCP specification's
    ``floor(log2(amax)) - 2``, ``round_up`` the smallest exponent that keeps
    the group maximum at or below 6, and ``search`` whichever of ``ocp`` and
    ``ocp + 1`` reconstructs the group with less squared error (ties keep
    ``ocp``). ``round_up`` is always one of those two, so ``search`` is never
    worse than either fixed rule. An all-zero group gets the smallest scale,
    because E8M0 has no zero.

    Returns the round-tripped values, the E8M0 bytes, and which exponent won
    (0 for ``ocp``, 1 for ``ocp + 1``).
    """
    if rule not in MX_SCALE_RULES:
        raise ValueError(f"rule must be one of {MX_SCALE_RULES}, got {rule!r}")
    if x.shape[-1] % MX_GROUP:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by {MX_GROUP}"
        )
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // MX_GROUP, MX_GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    # frexp gives amax = m * 2**k with m in [0.5, 1), so floor(log2(amax)) is
    # k - 1 exactly, with no logarithm to round.
    _, exponent = torch.frexp(amax)
    ocp = (exponent.to(torch.int32) - 1 - _E2M1_MAX_EXPONENT).clamp(
        _E8M0_MIN_EXPONENT, _E8M0_MAX_EXPONENT
    )
    ocp = torch.where(amax > 0, ocp, torch.full_like(ocp, _E8M0_MIN_EXPONENT))
    ocp_values = _mx_round_trip(groups, ocp)
    if rule == "ocp":
        winner = torch.zeros_like(ocp, dtype=torch.int64)
    else:
        doubled = (ocp + 1).clamp(max=_E8M0_MAX_EXPONENT)
        doubled_values = _mx_round_trip(groups, doubled)
        if rule == "round_up":
            ocp_scale = torch.ldexp(torch.ones_like(amax), ocp)
            choose = amax > 6.0 * ocp_scale
        else:
            choose = _accumulate_squares(doubled_values - groups) < _accumulate_squares(
                ocp_values - groups
            )
        winner = choose.to(torch.int64)
        ocp = torch.where(choose, doubled, ocp)
        ocp_values = torch.where(choose, doubled_values, ocp_values)
    codes = (ocp + 127).to(torch.uint8)
    return ocp_values.reshape_as(x), codes.squeeze(-1), winner.squeeze(-1)


def mxfp4_round_trip(x: torch.Tensor, rule: str = "ocp") -> torch.Tensor:
    """Quantize and dequantize groups of 32 on the last axis with MXFP4."""
    values, _, _ = mxfp4_quantize(x, rule)
    return values


def mxfp4_probability_quantize(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """P-specific adjacent-exponent search, including a group-mass penalty.

    Reproduce the warp's butterfly reduction and final FMA when comparing
    candidates. Return reconstructed values and one E8M0 byte per group.
    """
    baseline, codes, _ = mxfp4_quantize(probability, "round_up")
    groups = probability.float().reshape(*probability.shape[:-1], -1, 32)
    exponent = codes.int().unsqueeze(-1) - 127
    smaller = (exponent - 1).clamp(min=-127)
    rival = _mx_round_trip(groups, smaller)

    def cost(values):
        difference = values - groups
        squared, mass = difference.square(), difference
        for _ in range(5):
            squared = squared[..., 0::2] + squared[..., 1::2]
            mass = mass[..., 0::2] + mass[..., 1::2]
        return ((mass / 32).double() * mass.double() + squared.double()).float()

    baseline = baseline.reshape_as(groups)
    choose = cost(rival) < cost(baseline)
    values = torch.where(choose, rival, baseline).reshape_as(probability)
    selected = torch.where(choose, smaller, exponent).squeeze(-1)
    return values, (selected + 127).to(torch.uint8)


FP4_FORMATS = ("nvfp4", "mxfp4")


def fp4_round_trip(x: torch.Tensor, fmt: str, scale_mode: str) -> torch.Tensor:
    """Round trip on the last axis: NVFP4 groups of 16 with ``scale_mode`` as an
    NVFP4 candidate set, or MXFP4 groups of 32 with ``scale_mode`` as an MX rule."""
    if fmt == "nvfp4":
        return nvfp4_round_trip(x, scale_mode=scale_mode)
    if fmt == "mxfp4":
        return mxfp4_round_trip(x, scale_mode)
    raise ValueError(f"fmt must be one of {FP4_FORMATS}, got {fmt!r}")


def fp4_probability_round_trip(
    probability: torch.Tensor, fmt: str, scale_mode: str = "amax6"
) -> torch.Tensor:
    """The decode kernel's P operand: NVFP4 per :func:`nvfp4_probability_quantize`,
    or MXFP4 with adjacent-exponent element/mass-error search."""
    if fmt == "nvfp4":
        values, _ = nvfp4_probability_quantize(
            probability, group=16, scale_mode=scale_mode
        )
        return values
    if fmt == "mxfp4":
        return mxfp4_probability_quantize(probability)[0]
    raise ValueError(f"fmt must be one of {FP4_FORMATS}, got {fmt!r}")


def nvfp4_probability_quantize(
    x: torch.Tensor,
    *,
    group: int = 16,
    scale_mode: str = "amax6",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the decode kernel's P-specific scale/code ordering.

    The online softmax normalizes P with the unrounded FP32 ``amax/divisor``
    before converting the normalized value to E2M1.  Only the scale consumed
    by the P×V MMA is rounded to E4M3.  This differs slightly from cached K/V,
    whose code is selected using the already-rounded stored scale.
    """
    if group <= 0 or x.shape[-1] % group:
        raise ValueError("P group must be positive and divide the last axis")
    if scale_mode == "amax2_or_amax6_c8":
        values, winner = nvfp4_probability_quantize(
            x.float() * 8.0, group=group, scale_mode="amax2_or_amax6"
        )
        return values * 0.125, winner
    if scale_mode == "two_level":
        tile_size = min(128, x.shape[-1])
        if x.shape[-1] % tile_size:
            raise ValueError("two_level P needs complete 128-value tiles")
        tiles = x.float().reshape(*x.shape[:-1], -1, tile_size)
        peak = tiles.abs().amax(dim=-1, keepdim=True)
        outer = peak / (448.0 * 6.0)
        amplified = torch.where(peak > 0, tiles / peak.clamp_min(torch.finfo(torch.float32).tiny) * (448.0 * 6.0), 0.0)
        values, winner = nvfp4_probability_quantize(amplified, group=group, scale_mode="amax6")
        return (values * outer).reshape_as(x), winner.reshape(*x.shape[:-1], -1)
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    amax = groups.abs().amax(dim=-1, keepdim=True)

    if scale_mode == "amax2_or_amax6":
        low_amax = amax <= P_SMALL_GROUP_AMAX
        reciprocal = torch.where(
            low_amax,
            torch.tensor(1.0 / 2.0, dtype=torch.float32, device=x.device),
            torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device),
        )
        raw_scale = amax * reciprocal
        stored_scale = raw_scale.clamp(max=_E4M3_MAX).to(
            torch.float8_e4m3fn
        ).float()
        codes = round_e2m1(groups * (1.0 / (raw_scale + 1.0e-30)))
        values = codes * stored_scale
        winner = torch.where(
            low_amax.squeeze(-1),
            torch.full_like(amax.squeeze(-1), 3, dtype=torch.int64),
            torch.zeros_like(amax.squeeze(-1), dtype=torch.int64),
        )
        return values.reshape_as(x), winner

    best_values = best_error = best_winner = None
    divisors = probability_scale_candidates(scale_mode)
    for index, divisor in enumerate(divisors):
        reciprocal = torch.tensor(1.0 / divisor, dtype=torch.float32).item()
        raw_scale = amax * reciprocal
        stored_scale = raw_scale.clamp(max=_E4M3_MAX).to(
            torch.float8_e4m3fn
        ).float()
        codes = round_e2m1(groups * (1.0 / (raw_scale + 1.0e-30)))
        values = codes * stored_scale
        if len(divisors) == 1:
            winner = torch.zeros_like(amax.squeeze(-1), dtype=torch.int64)
            return values.reshape_as(x), winner
        error = _accumulate_squares(values - groups)
        if best_error is None:
            best_values = values
            best_error = error
            best_winner = torch.zeros_like(error, dtype=torch.int64)
            continue
        better = error < best_error
        best_values = torch.where(better, values, best_values)
        best_error = torch.where(better, error, best_error)
        best_winner = torch.where(
            better, torch.full_like(best_winner, index), best_winner
        )
    return best_values.reshape_as(x), best_winner.squeeze(-1)


def probability_scale_candidates(scale_mode: str) -> tuple[float, ...]:
    """Candidate divisors supported by the ephemeral P quantizer.

    Cached K/V keep their existing scale-mode vocabulary. P additionally
    supports fixed interior E2M1 rungs and an offline oracle over those rungs;
    none of these modes changes the P payload or scale layout.
    """
    if scale_mode in _PROBABILITY_SCALE_MODES:
        return _PROBABILITY_SCALE_MODES[scale_mode]
    return candidates(scale_mode)


def value_token_axis_round_trip(
    values: torch.Tensor,
    *,
    fp4_tokens: int,
    sink_tokens: int = 16,
    page_size: int = 128,
    scale_mode: str = "four_six",
    fmt: str = "nvfp4",
) -> torch.Tensor:
    """Round V with the cache's real token-axis grouping.

    ``values`` is ``[tokens, heads, head_dim]``. Each complete cache page is
    viewed as ``[heads, head_dim, tokens]`` before grouping, because token is
    the K dimension of the P×V MMA. The permanent sink and the incomplete
    recent tail remain BF16 and are restored after the page round trip.
    """
    if values.ndim != 3:
        raise ValueError(
            f"values must have shape [tokens,heads,dim], got {values.shape}"
        )
    tokens = values.shape[0]
    if (
        type(fp4_tokens) is not int
        or fp4_tokens < 0
        or fp4_tokens > tokens
        or fp4_tokens % page_size
    ):
        raise ValueError(
            "fp4_tokens must be a page-aligned integer inside the sequence"
        )
    if sink_tokens < 0 or sink_tokens > tokens:
        raise ValueError("sink_tokens must lie inside the sequence")

    rounded = values.float().clone()
    if fp4_tokens:
        pages = values[:fp4_tokens].reshape(
            -1, page_size, values.shape[1], values.shape[2]
        ).permute(0, 2, 3, 1)
        rounded_pages = fp4_round_trip(pages, fmt, scale_mode).permute(
            0, 3, 1, 2
        )
        rounded[:fp4_tokens] = rounded_pages.reshape_as(
            rounded[:fp4_tokens]
        )
    rounded[:sink_tokens] = values[:sink_tokens].float()
    rounded[fp4_tokens:] = values[fp4_tokens:].float()
    return rounded


def online_softmax_pv(
    scores: torch.Tensor,
    values: torch.Tensor,
    *,
    fp4_tokens: int,
    quantize_probabilities: bool,
    sink_tokens: int = 16,
    page_size: int = 128,
    softmax_scale: float | None = None,
    probability_format: str = "nvfp4",
    rescale_threshold: float = 0.0,
) -> torch.Tensor:
    """Accumulate P×V in the decode kernel's online block order.

    The incomplete BF16 tail is consumed first, complete FP4 pages from newest
    to oldest, and the permanent BF16 sink last. On FP4 pages, P is optionally
    rounded as ``probability_format``'s P operand
    (:func:`fp4_probability_round_trip`), NVFP4 with fixed ``amax/6``. The denominator
    always uses the unquantized exponentials, matching the kernel.

    ``rescale_threshold`` is the kernel's ``RESCALE_THRESHOLD``: a running
    maximum that would grow by at most that many powers of two is kept, so
    P is formed against the stale maximum. Zero always rescales.
    """
    if scores.ndim != 2 or values.ndim != 3:
        raise ValueError("scores must be [heads,tokens] and values [tokens,heads,dim]")
    heads, tokens = scores.shape
    if values.shape[:2] != (tokens, heads):
        raise ValueError(
            f"score/value shape mismatch: {scores.shape} versus {values.shape}"
        )
    if (
        type(fp4_tokens) is not int
        or fp4_tokens < 0
        or fp4_tokens > tokens
        or fp4_tokens % page_size
    ):
        raise ValueError(
            "fp4_tokens must be a page-aligned integer inside the sequence"
        )
    if sink_tokens < 0 or sink_tokens > tokens:
        raise ValueError("sink_tokens must lie inside the sequence")
    scale = float(softmax_scale or values.shape[-1] ** -0.5)
    scaled_scores = scores.float() * scale
    output = torch.zeros(
        heads, values.shape[-1], device=values.device, dtype=torch.float32
    )
    running_max = torch.full(
        (heads,), -torch.inf, device=values.device, dtype=torch.float32
    )
    running_sum = torch.zeros_like(running_max)

    def accumulate(
        block_scores: torch.Tensor,
        block_values: torch.Tensor,
        *,
        fp4_probability: bool,
    ) -> None:
        nonlocal output, running_max, running_sum
        block_max = torch.maximum(running_max, block_scores.amax(dim=-1))
        if rescale_threshold:
            growth = (block_max - running_max) * math.log2(math.e)
            block_max = torch.where(
                growth <= rescale_threshold, running_max, block_max
            )
        rescale = torch.exp(running_max - block_max)
        rescale = torch.where(
            torch.isfinite(rescale), rescale, torch.zeros_like(rescale)
        )
        finite_max = torch.where(
            torch.isfinite(block_max), block_max, torch.zeros_like(block_max)
        )
        probability = torch.exp(block_scores - finite_max[:, None])
        running_sum = running_sum * rescale + probability.sum(dim=-1)
        output = output * rescale[:, None]
        if fp4_probability:
            used_probability = fp4_probability_round_trip(
                probability, probability_format
            )
        else:
            used_probability = probability.to(torch.bfloat16).float()
        output = output + torch.einsum(
            "ht,thd->hd", used_probability, block_values.float()
        )
        running_max = block_max

    tail_start = sink_tokens if fp4_tokens == 0 else fp4_tokens
    if tail_start < tokens:
        accumulate(
            scaled_scores[:, tail_start:],
            values[tail_start:],
            fp4_probability=False,
        )

    for start in range(fp4_tokens - page_size, -1, -page_size):
        block_scores = scaled_scores[:, start : start + page_size].clone()
        if start < sink_tokens:
            invalid = torch.arange(
                start, start + page_size, device=scores.device
            ) < sink_tokens
            block_scores[:, invalid] = -torch.inf
        accumulate(
            block_scores,
            values[start : start + page_size],
            fp4_probability=quantize_probabilities,
        )

    if sink_tokens:
        accumulate(
            scaled_scores[:, :sink_tokens],
            values[:sink_tokens],
            fp4_probability=False,
        )
    return output / running_sum.clamp_min(1.0e-30)[:, None]


def online_softmax_pv_factors(
    scores: torch.Tensor,
    bf16_values: torch.Tensor,
    fp4_values: torch.Tensor,
    *,
    fp4_tokens: int,
    sink_tokens: int = 16,
    page_size: int = 128,
    softmax_scale: float | None = None,
    probability_scale_mode: str = "amax6",
) -> dict[str, torch.Tensor]:
    """Isolate P and V errors in one production-order online-softmax pass.

    The four returned outputs share the exact same unquantized softmax
    denominator.  Complete prefix pages use either BF16 exponentials or the
    kernel's fixed-amax/6 NVFP4 P operand, crossed with BF16 or token-axis
    NVFP4 V.  Sink and incomplete-tail blocks remain BF16 in every arm.
    """
    if scores.ndim != 2 or bf16_values.ndim != 3 or fp4_values.ndim != 3:
        raise ValueError(
            "scores must be [heads,tokens] and values [tokens,heads,dim]"
        )
    heads, tokens = scores.shape
    expected = (tokens, heads)
    if bf16_values.shape[:2] != expected or fp4_values.shape != bf16_values.shape:
        raise ValueError("score/value shapes do not agree")
    if (
        type(fp4_tokens) is not int
        or fp4_tokens < 0
        or fp4_tokens > tokens
        or fp4_tokens % page_size
    ):
        raise ValueError(
            "fp4_tokens must be a page-aligned integer inside the sequence"
        )
    if sink_tokens < 0 or sink_tokens > tokens:
        raise ValueError("sink_tokens must lie inside the sequence")

    scale = float(softmax_scale or bf16_values.shape[-1] ** -0.5)
    scaled_scores = scores.float() * scale
    outputs = {
        name: torch.zeros(
            heads,
            bf16_values.shape[-1],
            device=bf16_values.device,
            dtype=torch.float32,
        )
        for name in ("reference", "p_only", "v_only", "p_and_v")
    }
    running_max = torch.full(
        (heads,), -torch.inf, device=scores.device, dtype=torch.float32
    )
    running_sum = torch.zeros_like(running_max)

    def accumulate(start: int, end: int, *, fp4_block: bool) -> None:
        nonlocal running_max, running_sum
        block_scores = scaled_scores[:, start:end]
        if fp4_block and start < sink_tokens:
            block_scores = block_scores.clone()
            invalid = torch.arange(start, end, device=scores.device) < sink_tokens
            block_scores[:, invalid] = -torch.inf
        block_max = torch.maximum(running_max, block_scores.amax(dim=-1))
        rescale = torch.exp(running_max - block_max)
        rescale = torch.where(
            torch.isfinite(rescale), rescale, torch.zeros_like(rescale)
        )
        finite_max = torch.where(
            torch.isfinite(block_max), block_max, torch.zeros_like(block_max)
        )
        probability = torch.exp(block_scores - finite_max[:, None])
        running_sum = running_sum * rescale + probability.sum(dim=-1)
        for name in outputs:
            outputs[name] *= rescale[:, None]

        bf16_probability = probability.to(torch.bfloat16).float()
        fp4_probability = (
            nvfp4_probability_quantize(
                probability,
                group=16,
                scale_mode=probability_scale_mode,
            )[0]
            if fp4_block
            else bf16_probability
        )
        bf16_block = bf16_values[start:end].float()
        fp4_block_values = (
            fp4_values[start:end].float() if fp4_block else bf16_block
        )
        outputs["reference"] += torch.einsum(
            "ht,thd->hd", bf16_probability, bf16_block
        )
        outputs["p_only"] += torch.einsum(
            "ht,thd->hd", fp4_probability, bf16_block
        )
        outputs["v_only"] += torch.einsum(
            "ht,thd->hd", bf16_probability, fp4_block_values
        )
        outputs["p_and_v"] += torch.einsum(
            "ht,thd->hd", fp4_probability, fp4_block_values
        )
        running_max = block_max

    tail_start = sink_tokens if fp4_tokens == 0 else fp4_tokens
    if tail_start < tokens:
        accumulate(tail_start, tokens, fp4_block=False)
    for start in range(fp4_tokens - page_size, -1, -page_size):
        accumulate(start, start + page_size, fp4_block=True)
    if sink_tokens:
        accumulate(0, sink_tokens, fp4_block=False)

    denominator = running_sum.clamp_min(1.0e-30)[:, None]
    return {name: output / denominator for name, output in outputs.items()}


def online_probability_scale_statistics(
    scores: torch.Tensor,
    *,
    fp4_tokens: int,
    head_dim: int,
    sink_tokens: int = 16,
    page_size: int = 128,
    probability_multiplier: float = 1.0,
) -> dict:
    """Aggregate exact P-scale errors by the E4M3 code of each group maximum.

    This vectorizes the same reverse-page online-softmax order used by
    :func:`online_softmax_pv_factors`.  It is an offline diagnostic: the
    returned code buckets are used to ask whether a cheap amax-code lookup can
    approximate the expensive four-round-trip per-group oracle.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [heads, tokens]")
    heads, tokens = scores.shape
    if fp4_tokens <= 0 or fp4_tokens > tokens or fp4_tokens % page_size:
        raise ValueError("fp4_tokens must be positive, page aligned, and in range")
    if page_size % 16:
        raise ValueError("page size must be divisible by the P group size")
    if not 0 <= sink_tokens <= fp4_tokens:
        raise ValueError("sink tokens must lie inside the FP4 prefix")
    if not math.isfinite(probability_multiplier) or not 0 < probability_multiplier <= 448:
        raise ValueError("probability_multiplier must be finite and between 0 and 448")

    scaled = scores.float() * float(head_dim**-0.5)
    pages = scaled[:, :fp4_tokens].reshape(heads, -1, page_size).clone()
    if sink_tokens:
        pages[:, 0, :sink_tokens] = -torch.inf
    page_max = pages.amax(dim=-1)
    suffix_max = torch.flip(
        torch.cummax(torch.flip(page_max, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    if fp4_tokens < tokens:
        tail_max = scaled[:, fp4_tokens:].amax(dim=-1, keepdim=True)
        suffix_max = torch.maximum(suffix_max, tail_max)
    finite_max = torch.where(
        torch.isfinite(suffix_max), suffix_max, torch.zeros_like(suffix_max)
    )
    probability = torch.exp(pages - finite_max[..., None])
    probability = torch.where(
        torch.isfinite(pages), probability, torch.zeros_like(probability)
    )
    groups = (probability * probability_multiplier).reshape(-1, 16)
    maximum = groups.amax(dim=-1, keepdim=True)
    maximum_code = maximum.squeeze(-1).to(torch.float8_e4m3fn).view(torch.uint8)

    divisors = probability_scale_candidates("e2m1_max_grid")
    errors = []
    for divisor in divisors:
        raw_scale = maximum * float(1.0 / divisor)
        stored_scale = raw_scale.clamp(max=_E4M3_MAX).to(
            torch.float8_e4m3fn
        ).float()
        codes = round_e2m1(groups / raw_scale.clamp_min(1.0e-30))
        errors.append((codes * stored_scale - groups).square().sum(dim=-1))
    error = torch.stack(errors, dim=-1)
    winner = error.argmin(dim=-1)
    oracle_error = error.gather(-1, winner[:, None]).squeeze(-1)

    bucket_count = torch.bincount(maximum_code.long(), minlength=256)
    winner_count = torch.bincount(
        maximum_code.long() * len(divisors) + winner,
        minlength=256 * len(divisors),
    ).reshape(256, len(divisors))
    error_sum = torch.zeros(
        256, len(divisors), device=error.device, dtype=torch.float64
    )
    error_sum.scatter_add_(
        0,
        maximum_code.long()[:, None].expand_as(error),
        error.double(),
    )
    oracle_sum = torch.zeros(256, device=error.device, dtype=torch.float64)
    oracle_sum.scatter_add_(0, maximum_code.long(), oracle_error.double())

    counts = bucket_count.cpu().tolist()
    winners = winner_count.cpu().tolist()
    sums = error_sum.cpu().tolist()
    oracle = oracle_sum.cpu().tolist()
    by_code = {
        str(code): {
            "groups": int(counts[code]),
            "winner_counts": {
                str(int(divisor)): int(winners[code][index])
                for index, divisor in enumerate(divisors)
            },
            "error_sum": {
                str(int(divisor)): float(sums[code][index])
                for index, divisor in enumerate(divisors)
            },
            "oracle_error_sum": float(oracle[code]),
        }
        for code in range(256)
        if counts[code]
    }
    return {
        "group_count": int(groups.shape[0]),
        "divisors": [int(value) for value in divisors],
        "winner_counts": {
            str(int(divisor)): int((winner == index).sum().item())
            for index, divisor in enumerate(divisors)
        },
        "error_sum": {
            str(int(divisor)): float(error[:, index].double().sum().item())
            for index, divisor in enumerate(divisors)
        },
        "oracle_error_sum": float(oracle_error.double().sum().item()),
        "by_amax_e4m3_code": by_code,
    }
