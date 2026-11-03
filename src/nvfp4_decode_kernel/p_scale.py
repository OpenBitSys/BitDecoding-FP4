"""Configuration for the ephemeral NVFP4 softmax-probability operand."""

from __future__ import annotations

import os
import math
from collections.abc import Mapping


# Fallback threshold in scaled-probability units; calibration overrides it
# through NVFP4_P_SCALE_TAU. Using amax/2 for small groups keeps their E4M3
# scales farther from zero than amax/6. The default 21/1024 lies halfway
# between the adjacent E4M3 values 20/1024 and 22/1024.
DEFAULT_P_SCALE_TAU = 21.0 / 1024.0


def resolve_p_scale_tau(environment: Mapping[str, str] | None = None) -> float:
    """Threshold in scaled-probability units: choose /2 when c * max(p) <= tau."""
    env = os.environ if environment is None else environment
    tau = float(env.get("NVFP4_P_SCALE_TAU", DEFAULT_P_SCALE_TAU))
    if not math.isfinite(tau) or not 0 <= tau <= 448:
        raise ValueError("NVFP4_P_SCALE_TAU must be finite and between 0 and 448")
    return tau


# Read before kernel import: the comparison is embedded in its generated PTX.
P_SMALL_GROUP_AMAX = resolve_p_scale_tau()

P_SCALE_MODES: dict[str, tuple[float, ...]] = {
    "amax6": (6.0,),
    "two_level": (6.0,),
    "amax4": (4.0,),
    "amax3": (3.0,),
    "amax2": (2.0,),
    "e2m1_max_grid": (6.0, 4.0, 3.0, 2.0),
    "amax2_or_amax6": (2.0, 6.0),
    "amax2_or_amax6_c8": (2.0, 6.0),
}

DEFAULT_P_SCALE_MODE = "amax2_or_amax6_c8"

# Accept the older configuration name for ``amax2_or_amax6``.
_LEGACY_P_SCALE_MODES = {"e4m3_amax_code10_threshold": "amax2_or_amax6"}


def p_scale_is_amax2_or_amax6(mode: str) -> bool:
    """Whether ``mode`` selects amax/2 for small groups and amax/6 otherwise."""
    return _LEGACY_P_SCALE_MODES.get(mode, mode) in {
        "amax2_or_amax6", "amax2_or_amax6_c8",
    }


def resolve_p_scale_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[str, float, bool]:
    """Return ``(mode, fixed_divisor, searches_grid)`` with contradiction checks.

    With neither variable set the mode is ``DEFAULT_P_SCALE_MODE``.
    ``NVFP4_P_SCALE_DIVISOR`` selects the divisor for fixed modes. Adaptive
    modes return six as a constructor placeholder; a separate compile-time
    flag selects the actual divisor.
    """
    env = os.environ if environment is None else environment
    mode = env.get("NVFP4_P_SCALE_MODE")
    divisor_text = env.get("NVFP4_P_SCALE_DIVISOR")
    if mode is None and divisor_text is None:
        mode = DEFAULT_P_SCALE_MODE
    if mode is None:
        try:
            divisor = float(divisor_text)
        except ValueError as error:
            raise ValueError(
                "NVFP4_P_SCALE_DIVISOR must be 2, 3, 4, or 6, got "
                f"{divisor_text!r}"
            ) from error
        if divisor not in (2.0, 3.0, 4.0, 6.0):
            raise ValueError(
                "NVFP4_P_SCALE_DIVISOR must be 2, 3, 4, or 6, got "
                f"{divisor_text!r}"
            )
        return f"amax{int(divisor)}", divisor, False

    mode = _LEGACY_P_SCALE_MODES.get(mode, mode)
    if mode not in P_SCALE_MODES:
        raise ValueError(
            f"NVFP4_P_SCALE_MODE must be one of {sorted(P_SCALE_MODES)}, "
            f"got {mode!r}"
        )
    divisors = P_SCALE_MODES[mode]
    grid = mode == "e2m1_max_grid"
    divisor = divisors[0] if len(divisors) == 1 else 6.0
    if divisor_text is not None and float(divisor_text) != divisor:
        raise ValueError(
            f"NVFP4_P_SCALE_MODE={mode!r} contradicts "
            f"NVFP4_P_SCALE_DIVISOR={divisor_text!r}"
        )
    return mode, divisor, grid
