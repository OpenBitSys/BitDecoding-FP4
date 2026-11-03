"""Composition layer between the public API and private kernel operations."""

from __future__ import annotations

import torch

from .interface import RESIDUAL_ROW_TILE, DecodeTailWrite
from .scale_mode import DEFAULT_SCALE_MODE, scale_format, sf_vec_size


def _padded_query_buffer(
    query: torch.Tensor,
    rows: int,
    scratch: torch.Tensor | None,
) -> torch.Tensor:
    """The BF16 query padded to the residual MMA's row tile.

    Deliberately not cached here. A framework that sizes its KV cache from the
    memory left over after a profiling run — vLLM does — would find a buffer
    grown on a later call coming out of memory already promised to the cache.
    Leaving the allocation to the caller lets it happen during profiling, at
    the price of the contract ``fp4_decode`` documents: zeroed once, because
    the quantizer writes ``[row, 0, head, :]`` and nothing writes the rest.
    """
    heads, head_dim = query.shape[1], query.shape[2]
    if scratch is None:
        return torch.zeros(
            rows,
            RESIDUAL_ROW_TILE,
            heads,
            head_dim,
            dtype=torch.bfloat16,
            device=query.device,
        )
    if (
        scratch.dtype is not torch.bfloat16
        or scratch.device != query.device
        or scratch.ndim != 4
        or scratch.shape[0] < rows
        or tuple(scratch.shape[1:]) != (RESIDUAL_ROW_TILE, heads, head_dim)
        or not scratch.is_contiguous()
    ):
        raise ValueError(
            "query_padded_scratch must be a contiguous BF16 tensor on the "
            f"query's device shaped [at least {rows}, {RESIDUAL_ROW_TILE}, "
            f"{heads}, {head_dim}]"
        )
    return scratch[:rows]


def _check_quantizer_scratch(
    scratch: torch.Tensor | None,
    name: str,
    rows: int,
    trailing: tuple[int, ...],
    device: torch.device,
) -> None:
    """Vet a caller-owned buffer for one of the quantizer's two outputs.

    Same bargain as ``_padded_query_buffer``, and made for the same reason: a
    framework that sizes its KV cache around what profiling left free wants
    every decode buffer to exist before that measurement, and wants the step
    itself to neither allocate nor fill. Only the row count may exceed what
    this batch needs, since rows are dim 0 and everything the decode kernel
    checks about the scale layout is an inner stride.

    Both buffers reach kernels compiled with ``assumed_align=16``, so the
    requirement is really on the address rather than on the layout. Demanding
    storage offset zero buys that without an alignment test: offset zero is
    the start of a PyTorch allocation, which the caching allocator hands out
    aligned to 512 bytes, far past the 16 the kernels assume. A contiguous
    view onto the middle of somebody else's tensor carries no such guarantee.
    """
    if scratch is None:
        return
    if (
        scratch.dtype is not torch.uint8
        or scratch.device != device
        or scratch.ndim != 1 + len(trailing)
        or scratch.shape[0] < rows
        or tuple(scratch.shape[1:]) != trailing
        or not scratch.is_contiguous()
        or scratch.storage_offset() != 0
    ):
        extents = ", ".join(str(extent) for extent in trailing)
        raise ValueError(
            f"{name} must be a contiguous UINT8 tensor on the query's device "
            f"shaped [at least {rows}, {extents}], starting at storage "
            "offset zero"
        )


def fp4_decode_impl(
    query: torch.Tensor | None,
    key_pages_fp4: torch.Tensor,
    key_scales: torch.Tensor,
    value_pages_fp4: torch.Tensor,
    value_scales: torch.Tensor,
    fp4_page_table: torch.Tensor,
    seqused_fp4: torch.Tensor,
    *,
    query_fp4: torch.Tensor | None = None,
    query_scales: torch.Tensor | None = None,
    query_fp4_second: torch.Tensor | None = None,
    query_scales_second: torch.Tensor | None = None,
    residual_key_pages_bf16: torch.Tensor | None = None,
    residual_value_pages_bf16: torch.Tensor | None = None,
    residual_page_ids: torch.Tensor | None = None,
    seqused_residual: torch.Tensor | None = None,
    seqused_sink: torch.Tensor | None = None,
    has_bf16: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    query_row_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    trusted_metadata: bool = False,
    num_splits: int = 1,
    query_padded_scratch: torch.Tensor | None = None,
    query_fp4_scratch: torch.Tensor | None = None,
    query_scales_scratch: torch.Tensor | None = None,
    query_fp4_second_scratch: torch.Tensor | None = None,
    query_scales_second_scratch: torch.Tensor | None = None,
    split_partial_scratch: torch.Tensor | None = None,
    split_lse_scratch: torch.Tensor | None = None,
    query_terms: int = 1,
    scale_mode: str = DEFAULT_SCALE_MODE,
    query_rotation: str = "none",
    tail_write: DecodeTailWrite | None = None,
) -> torch.Tensor:
    """Prepare either query contract and run the shared decode core."""
    from ._decode import decode_fp4, decode_fp4_split
    from ._quantize import quantize_query

    # Any count partitions the pages (ceil division per split), and the
    # combine compiles one kernel per power-of-two tier above it. Checked
    # before any quantization so a bad value costs nothing.
    if not isinstance(num_splits, int) or num_splits < 1:
        raise ValueError(
            f"num_splits must be a positive integer, got {num_splits!r}"
        )
    if query_terms not in (1, 2):
        raise ValueError(f"query_terms must be 1 or 2, got {query_terms!r}")
    fmt = scale_format(scale_mode)

    has_bf16_query = query is not None
    has_fp4_query = any(
        tensor is not None
        for tensor in (
            query_fp4,
            query_scales,
            query_fp4_second,
            query_scales_second,
        )
    )
    if has_bf16_query == has_fp4_query:
        raise ValueError(
            "supply either BF16 query or query_fp4 + query_scales"
        )

    if has_bf16_query:
        assert query is not None
        if any(
            tensor is not None
            for tensor in (
                query_fp4,
                query_scales,
                query_fp4_second,
                query_scales_second,
            )
        ):
            raise ValueError(
                "BF16 query is mutually exclusive with pre-quantized query "
                "terms"
            )
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5
        rows = (
            query.shape[0]
            if query_row_indices is None
            else query_row_indices.shape[0]
        )
        query_padded_bf16 = (
            _padded_query_buffer(query, rows, query_padded_scratch)
            if residual_key_pages_bf16 is not None
            else None
        )
        heads, head_dim = query.shape[1], query.shape[2]
        _check_quantizer_scratch(
            query_fp4_scratch,
            "query_fp4_scratch",
            rows,
            (1, heads, head_dim // 2),
            query.device,
        )
        _check_quantizer_scratch(
            query_scales_scratch,
            "query_scales_scratch",
            rows,
            (1, heads, head_dim // (4 * sf_vec_size(scale_mode)), 32, 4, 4),
            query.device,
        )
        if query_terms == 1 and (
            query_fp4_second_scratch is not None
            or query_scales_second_scratch is not None
        ):
            raise ValueError(
                "second query scratch requires query_terms=2"
            )
        if query_terms == 2:
            _check_quantizer_scratch(
                query_fp4_second_scratch,
                "query_fp4_second_scratch",
                rows,
                (1, heads, head_dim // 2),
                query.device,
            )
            _check_quantizer_scratch(
                query_scales_second_scratch,
                "query_scales_second_scratch",
                rows,
                (1, heads, head_dim // (4 * sf_vec_size(scale_mode)), 32, 4, 4),
                query.device,
            )
        quantized_query = quantize_query(
            query,
            row_indices=query_row_indices,
            query_padded_out=query_padded_bf16,
            heads_kv=key_pages_fp4.shape[2],
            query_fp4_scratch=query_fp4_scratch,
            query_scales_scratch=query_scales_scratch,
            query_fp4_second_scratch=query_fp4_second_scratch,
            query_scales_second_scratch=query_scales_second_scratch,
            query_terms=query_terms,
            scale_mode=scale_mode,
            rotation=query_rotation,
            tail_write=tail_write,
        )
        if query_terms == 2:
            (
                query_fp4,
                query_scales,
                query_fp4_second,
                query_scales_second,
            ) = quantized_query
        else:
            query_fp4, query_scales = quantized_query
            query_fp4_second = None
            query_scales_second = None
    else:
        if query_fp4 is None or query_scales is None:
            raise ValueError(
                "query_fp4 and query_scales must be provided together"
            )
        has_second_pair = (
            query_fp4_second is not None
            and query_scales_second is not None
        )
        if (query_fp4_second is None) != (query_scales_second is None):
            raise ValueError(
                "query_fp4_second and query_scales_second must be provided "
                "together"
            )
        if query_terms == 2 and not has_second_pair:
            raise ValueError(
                "query_terms=2 needs query_fp4_second and "
                "query_scales_second"
            )
        if query_terms == 1 and has_second_pair:
            raise ValueError(
                "second pre-quantized query pair requires query_terms=2"
            )
        if query_row_indices is not None:
            raise ValueError(
                "query_row_indices applies only to the BF16 query path"
            )
        for name, scratch in (
            ("query_padded_scratch", query_padded_scratch),
            ("query_fp4_scratch", query_fp4_scratch),
            ("query_scales_scratch", query_scales_scratch),
            ("query_fp4_second_scratch", query_fp4_second_scratch),
            (
                "query_scales_second_scratch",
                query_scales_second_scratch,
            ),
        ):
            if scratch is not None:
                raise ValueError(
                    f"{name} applies only to the BF16 query path"
                )
        if scale_mode != DEFAULT_SCALE_MODE:
            # The caller already chose one when they quantized the query
            # themselves, and this argument would name a second.
            raise ValueError("scale_mode applies only to the BF16 query path")
        if query_rotation != "none":
            raise ValueError(
                "query_rotation applies only to the BF16 query path"
            )
        if tail_write is not None:
            raise ValueError("tail_write applies only to the BF16 query path")
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        query_padded_bf16 = None

    # A residual reaches split-K only when every one of its arguments is
    # present. Keying off a single argument would let a partially supplied
    # residual be dropped there instead of raising the all-or-none error that
    # the non-split path enforces.
    residual_arguments = (
        residual_key_pages_bf16,
        residual_value_pages_bf16,
        residual_page_ids,
        seqused_residual,
    )
    complete_residual = all(
        argument is not None for argument in residual_arguments
    )
    splittable = (
        all(argument is None for argument in residual_arguments)
        and has_bf16 is None
    ) or (complete_residual and query_padded_bf16 is not None)
    # Only a scatter by index forces the single-tile path; the split path can
    # write into a caller's buffer, it just cannot reorder rows on the way.
    scatter_by_index = out_indices is not None
    # A split the dispatcher cannot serve is refused rather than served with
    # one tile, so that a caller asking for the split path either gets it or
    # hears why not.
    if num_splits > 1 and not splittable:
        raise ValueError(
            "num_splits > 1 needs either no residual argument at all or a "
            "complete residual on the BF16 query path; this combination "
            "runs on the single-tile path only"
        )
    if num_splits > 1 and scatter_by_index:
        raise ValueError(
            "num_splits > 1 cannot be combined with out_indices, because the "
            "split path's combine writes the batch's rows in order"
        )
    if num_splits > 1:
        # Keep the single public entry's full contract checks when dispatching
        # to the split implementation. Trusted metadata skips only the
        # intentional device-value scans, not host shape/dtype validation.
        decode_fp4(
            query_fp4=query_fp4,
            query_scales=query_scales,
            query_fp4_second=query_fp4_second,
            query_scales_second=query_scales_second,
            query_padded_bf16=query_padded_bf16,
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            fp4_page_table=fp4_page_table,
            seqused_fp4=seqused_fp4,
            residual_key_pages_bf16=residual_key_pages_bf16,
            residual_value_pages_bf16=residual_value_pages_bf16,
            residual_page_ids=residual_page_ids,
            seqused_residual=seqused_residual,
            seqused_sink=seqused_sink,
            has_bf16=has_bf16,
            softmax_scale=softmax_scale,
            out=out,
            scale_format=fmt,
            out_indices=out_indices,
            trusted_metadata=trusted_metadata,
            validate_only=True,
        )
        return decode_fp4_split(
            query_fp4=query_fp4,
            query_scales=query_scales,
            query_fp4_second=query_fp4_second,
            query_scales_second=query_scales_second,
            key_pages_fp4=key_pages_fp4,
            key_scales=key_scales,
            value_pages_fp4=value_pages_fp4,
            value_scales=value_scales,
            fp4_page_table=fp4_page_table,
            seqused_fp4=seqused_fp4,
            softmax_scale=softmax_scale,
            num_splits=num_splits,
            query_padded_bf16=query_padded_bf16,
            residual_key_pages_bf16=residual_key_pages_bf16,
            residual_value_pages_bf16=residual_value_pages_bf16,
            residual_page_ids=residual_page_ids,
            seqused_residual=seqused_residual,
            seqused_sink=seqused_sink,
            has_bf16=has_bf16,
            out=out,
            split_partial_scratch=split_partial_scratch,
            split_lse_scratch=split_lse_scratch,
            scale_format=fmt,
        )

    return decode_fp4(
        query_fp4=query_fp4,
        query_scales=query_scales,
        query_fp4_second=query_fp4_second,
        query_scales_second=query_scales_second,
        query_padded_bf16=query_padded_bf16,
        key_pages_fp4=key_pages_fp4,
        key_scales=key_scales,
        value_pages_fp4=value_pages_fp4,
        value_scales=value_scales,
        fp4_page_table=fp4_page_table,
        seqused_fp4=seqused_fp4,
        residual_key_pages_bf16=residual_key_pages_bf16,
        residual_value_pages_bf16=residual_value_pages_bf16,
        residual_page_ids=residual_page_ids,
        seqused_residual=seqused_residual,
        seqused_sink=seqused_sink,
        has_bf16=has_bf16,
        softmax_scale=softmax_scale,
        out=out,
        scale_format=fmt,
        out_indices=out_indices,
        trusted_metadata=trusted_metadata,
    )
