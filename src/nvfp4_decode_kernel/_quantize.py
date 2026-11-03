"""NVFP4 query quantization."""

from __future__ import annotations

import torch

from .interface import DecodeTailWrite
from .quantize_kv_kernel import quantize_key_pages, quantize_value_pages
from .quantize_q_kernel import quantize_decode_q_to_padded_fp4
from .scale_mode import DEFAULT_SCALE_MODE, sf_vec_size


PAGE_SIZE = 128


def quantize_query(
    query: torch.Tensor,
    *,
    row_indices: torch.Tensor | None = None,
    query_padded_out: torch.Tensor | None = None,
    heads_kv: int | None = None,
    query_fp4_scratch: torch.Tensor | None = None,
    query_scales_scratch: torch.Tensor | None = None,
    query_fp4_second_scratch: torch.Tensor | None = None,
    query_scales_second_scratch: torch.Tensor | None = None,
    query_terms: int = 1,
    scale_mode: str = DEFAULT_SCALE_MODE,
    rotation: str = "none",
    tail_write: DecodeTailWrite | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Quantize BF16 decode rows with the CuTeDSL query kernel.

    Outputs may come from the caller instead of a fresh allocation, and
    may be wider than this batch. Rows are dim 0 of each, so the prefix slice
    taken here leaves every inner stride alone and the scale view still comes
    out with the layout the decode kernel insists on. A supplied scale buffer
    must arrive zeroed: the layout reserves 128 scale slots per KV head and
    the kernel writes only the ``heads // heads_kv`` of them that carry a
    query head. Nothing clears the rest, so a buffer that starts out zeroed
    is byte-for-byte what a fresh allocation would have produced, and one
    that does not stays wrong for as long as it is reused. Which slots those
    are follows ``heads // heads_kv`` while the storage shape does not, so a
    scale buffer also belongs to one ``(heads, heads_kv, head_dim)`` triple
    for its whole life: carrying it across a ``heads_kv`` change leaves the
    previous geometry's scales sitting in slots the new one needs zeroed.
    With ``query_terms=2``, the extra pair encodes the residual left by the
    first pair and the function returns ``(q0, s0, q1, s1)``. The default
    remains the original two-value return. ``rotation="hadamard"`` rotates
    each head first, and ``tail_write`` also places each row's new K/V token
    in the BF16 tail; see :func:`quantize_decode_q_to_padded_fp4`.
    """
    if query_terms not in (1, 2):
        raise ValueError(f"query_terms must be 1 or 2, got {query_terms!r}")
    if query_terms == 1 and (
        query_fp4_second_scratch is not None
        or query_scales_second_scratch is not None
    ):
        raise ValueError(
            "second query scratch requires query_terms=2"
        )
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        raise ValueError("query must be a BF16 CUDA tensor")
    if query.ndim != 3 or query.stride(-1) != 1:
        raise ValueError("query must have shape [rows, heads, head_dim]")
    if row_indices is not None and (
        row_indices.dtype is not torch.int32
        or not row_indices.is_cuda
        or not row_indices.is_contiguous()
        or row_indices.ndim != 1
    ):
        raise ValueError(
            "row_indices must be contiguous 1-D INT32 CUDA"
        )

    _, heads, head_dim = query.shape
    heads_kv = heads if heads_kv is None else heads_kv
    if heads_kv < 1 or heads % heads_kv:
        raise ValueError("query heads must be divisible by heads_kv")
    rows = query.shape[0] if row_indices is None else row_indices.shape[0]
    atom_k = 4 * sf_vec_size(scale_mode)
    if head_dim % atom_k != 0:
        raise ValueError(f"query head_dim must be divisible by {atom_k}")

    query_fp4 = (
        torch.zeros(
            rows,
            1,
            heads,
            head_dim // 2,
            dtype=torch.uint8,
            device=query.device,
        )
        if query_fp4_scratch is None
        else query_fp4_scratch[:rows]
    ).view(torch.float4_e2m1fn_x2)

    rest_k = head_dim // atom_k
    scale_storage = (
        torch.zeros(
            rows,
            1,
            heads,
            rest_k,
            32,
            4,
            4,
            dtype=torch.uint8,
            device=query.device,
        )
        if query_scales_scratch is None
        else query_scales_scratch[:rows]
    )
    native_scales = scale_storage.permute(
        0, 2, 1, 3, 4, 5, 6
    ).permute(
        4, 5, 2, 6, 3, 1, 0
    )
    query_scales = native_scales.as_strided(
        (
            32,
            4,
            1,
            4,
            rest_k,
            heads_kv,
            rows,
        ),
        native_scales.stride(),
    )

    query_fp4_second = None
    query_scales_second = None
    if query_terms == 2:
        query_fp4_second = (
            torch.zeros(
                rows,
                1,
                heads,
                head_dim // 2,
                dtype=torch.uint8,
                device=query.device,
            )
            if query_fp4_second_scratch is None
            else query_fp4_second_scratch[:rows]
        ).view(torch.float4_e2m1fn_x2)
        scale_storage_second = (
            torch.zeros(
                rows,
                1,
                heads,
                rest_k,
                32,
                4,
                4,
                dtype=torch.uint8,
                device=query.device,
            )
            if query_scales_second_scratch is None
            else query_scales_second_scratch[:rows]
        )
        native_scales_second = scale_storage_second.permute(
            0, 2, 1, 3, 4, 5, 6
        ).permute(
            4, 5, 2, 6, 3, 1, 0
        )
        query_scales_second = native_scales_second.as_strided(
            (
                32,
                4,
                1,
                4,
                rest_k,
                heads_kv,
                rows,
            ),
            native_scales_second.stride(),
        )

    quantize_decode_q_to_padded_fp4(
        query,
        query_fp4,
        query_scales,
        query_padded_out,
        query_fp4_out_2=query_fp4_second,
        query_scales_out_2=query_scales_second,
        row_indices=row_indices,
        heads_kv=heads_kv,
        scale_mode=scale_mode,
        rotation=rotation,
        tail_write=tail_write,
    )

    if query_terms == 2:
        assert query_fp4_second is not None
        assert query_scales_second is not None
        return (
            query_fp4,
            query_scales,
            query_fp4_second,
            query_scales_second,
        )
    return query_fp4, query_scales
