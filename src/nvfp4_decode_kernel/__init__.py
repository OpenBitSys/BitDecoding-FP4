"""Public API for the NVFP4 paged-decode kernel."""

import os

os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")

from .interface import (
    RESIDUAL_ROW_TILE,
    DecodeTailWrite,
    KernelNotAvailableError,
    fp4_decode,
    split_partial_count,
)

__all__ = [
    "RESIDUAL_ROW_TILE",
    "DecodeTailWrite",
    "KernelNotAvailableError",
    "fp4_decode",
    "split_partial_count",
]
