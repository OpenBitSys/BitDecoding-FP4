"""Interpretation of a vLLM KV cache block as an NVFP4 page.

vLLM hands out a flat byte page per block and never looks inside it. The four
things that have to live there — packed K, K scales, packed V, V scales — are
laid out one after another *within* each block rather than as four contiguous
region arrays spanning all blocks. Block-major costs each region a page stride,
which the quantizer and the decode kernel both accept, and in exchange a block
stays the self-contained byte range that vLLM believes it is. Anything that
copies a block by address, which today the guardrails forbid but tomorrow may
not, then keeps working.

An MXFP4 page has the same packed regions and half the scale bytes. Its scale
regions keep the NVFP4 size, the tail unused, so a block is the same size in
either format and an engine sizes its cache identically for both.
"""

from __future__ import annotations

import functools

import torch

from nvfp4_decode_kernel.quantize_kv_kernel import (
    quantize_key_pages,
    quantize_value_pages,
)
from nvfp4_decode_kernel.scale_mode import DEFAULT_SCALE_MODE


PAGE_SIZE = 128


def _probe_regions(
    num_kv_heads: int, head_dim: int, device: str, scale_mode: str
) -> tuple[torch.Tensor, ...]:
    probe = torch.zeros(
        2,
        PAGE_SIZE,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    return (
        *quantize_key_pages(probe, scale_mode=scale_mode),
        *quantize_value_pages(probe, scale_mode=scale_mode),
    )


@functools.lru_cache(maxsize=None)
def region_specs(
    num_kv_heads: int,
    head_dim: int,
    device: str,
    scale_mode: str = DEFAULT_SCALE_MODE,
) -> tuple[tuple, ...]:
    """Ask the quantizers what one page of each region actually looks like.

    The scale factors use a swizzled layout, so their shape and strides are not
    something to rederive here; running the real quantizer on a throwaway page
    is the only way to stay correct if that layout ever changes. A region's
    size is always the NVFP4 one, which an MXFP4 region fits inside.
    """
    regions = _probe_regions(num_kv_heads, head_dim, device, scale_mode)
    sizes = [
        region.stride()[0] * region.element_size()
        for region in _probe_regions(
            num_kv_heads, head_dim, device, DEFAULT_SCALE_MODE
        )
    ]
    return tuple(
        (
            tuple(region.shape[1:]),
            tuple(region.stride()[1:]),
            region.dtype,
            size,
        )
        for region, size in zip(regions, sizes)
    )


def page_bytes(num_kv_heads: int, head_dim: int, device: str) -> int:
    """Total bytes one block must provide for all four regions."""
    return sum(spec[3] for spec in region_specs(num_kv_heads, head_dim, device))


def carve(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    scale_mode: str = DEFAULT_SCALE_MODE,
) -> tuple[torch.Tensor, ...]:
    """View one layer's block array as packed K, K scales, packed V, V scales.

    Each returned view keeps the quantizer's exact layout inside a page and
    replaces only its page stride with the block pitch, which is what both the
    quantizer's indexed writes and the decode kernel's page strides expect.
    """
    if kv_cache.ndim != 2 or kv_cache.element_size() != 1:
        raise ValueError(
            f"expected a [blocks, bytes] byte cache, got {tuple(kv_cache.shape)}"
            f" of {kv_cache.dtype}"
        )
    blocks, pitch = kv_cache.shape
    if kv_cache.stride() != (pitch, 1):
        raise ValueError("the block array must be contiguous")

    specs = region_specs(num_kv_heads, head_dim, str(kv_cache.device), scale_mode)
    required = sum(spec[3] for spec in specs)
    if pitch != required:
        raise ValueError(
            f"a block of {pitch} bytes cannot hold {required} bytes of NVFP4 "
            "page; the backend's page size and this layout disagree"
        )

    flat = kv_cache.view(torch.uint8)
    views, offset = [], kv_cache.storage_offset()
    for shape, strides, dtype, region_bytes in specs:
        if dtype.itemsize != 1:
            # Offsets and strides below are byte counts taken from the probe,
            # which as_strided would read as element counts for a wider dtype.
            raise ValueError(f"region dtype {dtype} is not byte-sized")
        typed = flat if dtype is torch.uint8 else flat.view(dtype)
        views.append(
            typed.as_strided(
                (blocks,) + shape, (pitch,) + strides, offset
            )
        )
        offset += region_bytes
    return tuple(views)
