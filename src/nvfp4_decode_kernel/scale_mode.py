"""The block-scale candidate sets the quantization entries share.

A group's scale is its largest magnitude divided by one of these, rounded to
E4M3. One candidate is what every version of this cache has stored; the others
exist because ``amax/6``'s widest rung, the gap between 4 and 6, is four times
the grid's own step and swallows whatever second-largest element lands in it.

Order inside a tuple is tie-break order, earliest wins. That makes ``amax/6``
the answer whenever nothing beats it, so a larger set is a strict improvement
on a smaller one rather than a different answer, and it makes the extra
candidates byte-identical to the set without them on every group they do not
win. Both properties are what lets the three modes be compared as one variable.
"""

from __future__ import annotations


SCALE_MODES: dict[str, tuple[float, ...]] = {
    "amax6": (6.0,),
    "four_six": (6.0, 4.0),
    "four_six_sixhalf": (6.0, 4.0, 6.5),
}

DEFAULT_SCALE_MODE = "amax6"


def parse_scale_mode_spec(spec: str) -> str | dict[str, str]:
    """Parse a CLI scale mode without importing the attention implementation."""
    if "=" not in spec:
        return spec
    modes = dict(part.split("=", 1) for part in spec.split(",") if part)
    fields = {"key", "value", "query"}
    if set(modes) != fields:
        raise ValueError(
            f"a per-tensor scale mode needs exactly {sorted(fields)}, got {sorted(modes)}"
        )
    return modes

# MXFP4 modes: 32 values per power-of-two (E8M0) scale. The value is the
# exponent rule ``reference.mxfp4_quantize`` and ``_block_scale`` apply.
MX_SCALE_MODES: dict[str, str] = {
    "mxfp4_ocp": "ocp",
    "mxfp4_round_up": "round_up",
    "mxfp4_search": "search",
}


def scale_format(scale_mode: str) -> str:
    """``"mxfp4"`` for an MX mode, ``"nvfp4"`` for a candidate set."""
    if scale_mode in MX_SCALE_MODES:
        return "mxfp4"
    candidates(scale_mode)
    return "nvfp4"


def sf_vec_size(scale_mode: str) -> int:
    """How many values share one scale under ``scale_mode``."""
    return 32 if scale_format(scale_mode) == "mxfp4" else 16


def candidates(scale_mode: str) -> tuple[float, ...]:
    """The divisors an NVFP4 ``scale_mode`` names, in tie-break order."""
    try:
        return SCALE_MODES[scale_mode]
    except KeyError:
        raise ValueError(
            f"unknown scale_mode {scale_mode!r}, expected one of "
            f"{sorted(SCALE_MODES) + sorted(MX_SCALE_MODES)}"
        ) from None


def scale_rule(scale_mode: str) -> tuple[float, ...] | str:
    """What a quantizer specializes on: NVFP4 divisors, or an MX exponent rule."""
    if scale_mode in MX_SCALE_MODES:
        return MX_SCALE_MODES[scale_mode]
    return candidates(scale_mode)
