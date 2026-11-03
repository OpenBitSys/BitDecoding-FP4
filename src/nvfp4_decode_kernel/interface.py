"""Framework-independent interface for paged NVFP4 and MXFP4 decode."""

from __future__ import annotations

from typing import NamedTuple

import torch

from .scale_mode import DEFAULT_SCALE_MODE


# Rows of the tile the residual MMA reads the BF16 query through. Only row 0 of
# each tile holds a real query, since decode has one token per row.
RESIDUAL_ROW_TILE = 128
# Leading tokens retained in the BF16 cache. FP4 keeps its original page origin
# and page zero's duplicate is masked during decode.
SINK_TOKENS = 16


class KernelNotAvailableError(RuntimeError):
    """Raised when the kernel implementation has not been installed."""


def split_partial_count(
    rows: int, heads_kv: int, num_splits: int, *, residual: bool, sms: int
) -> int:
    """How many partial outputs per row a decode at ``num_splits`` writes.

    It is ``num_splits``, one per partition of the FP4 pages, plus one when a
    residual is fused and the tail and the sink get a split of their own. They
    get one only while the launch still fits on the device's ``sms`` SMs in a
    single wave with it, ``rows * heads_kv * (num_splits + 1) <= sms``: the
    extra CTA is worth having because it runs beside the FP4 splits and takes
    the BF16 blocks off every one of them, not after them. Otherwise every
    split runs the BF16 blocks and only split 0 counts them.

    ``split_partial_scratch`` and ``split_lse_scratch`` are sized by this.
    """
    if num_splits <= 1:
        return num_splits
    own_split = residual and rows * heads_kv * (num_splits + 1) <= sms
    return num_splits + int(own_split)


class DecodeTailWrite(NamedTuple):
    """This step's K/V token of each decode row, bound for the BF16 tail.

    ``key`` and ``value`` are ``[>= rows, heads_kv, head_dim]`` BF16 with row
    ``i`` the token of decode row ``i``, at position ``seq_lens[i] - 1``. It
    lands at ``position - seqused_fp4[i]`` of the rolling tail, and also at
    ``RESIDUAL_ROW_TILE + position`` while the position is below
    ``sink_tokens``. ``tail_key`` / ``tail_value`` are one layer's
    ``[slots, RESIDUAL_ROW_TILE + sink_tokens, heads_kv, head_dim]``, the
    slot is ``row_to_slot[i]``, and a negative slot writes nothing. K is
    stored in the cache's basis: Hadamard-rotated when ``rotate_key``, then
    ``key_shift`` (FP32 ``[heads_kv, head_dim]``, already in the rotated
    basis) subtracted, then rounded to BF16 once.
    """

    key: torch.Tensor
    value: torch.Tensor
    tail_key: torch.Tensor
    tail_value: torch.Tensor
    row_to_slot: torch.Tensor
    seq_lens: torch.Tensor
    seqused_fp4: torch.Tensor
    rotate_key: bool
    key_shift: torch.Tensor | None
    sink_tokens: int


def fp4_decode(
    query: torch.Tensor | None = None,
    key_pages_fp4: torch.Tensor | None = None,
    key_scales: torch.Tensor | None = None,
    value_pages_fp4: torch.Tensor | None = None,
    value_scales: torch.Tensor | None = None,
    fp4_page_table: torch.Tensor | None = None,
    seqused_fp4: torch.Tensor | None = None,
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
    """Run paged FP4 decode attention.

    Args:
        query: Optional BF16 query in ``[rows, heads_q, head_dim]`` layout. It may
            contain more rows than the current decode batch when
            ``query_row_indices`` is provided. Supply either this argument or
            both ``query_fp4`` and ``query_scales``.
        query_fp4: Optional pre-quantized packed E2M1 query from
            ``quantize_query``.
        query_scales: Optional E4M3 scale-factor bytes paired with
            ``query_fp4``.
        query_fp4_second: Optional residual-query term paired with
            ``query_scales_second``. Both are required when using the
            pre-quantized two-term path.
        key_pages_fp4: Packed E2M1 K pages.
        key_scales: E4M3 (NVFP4) or E8M0 (MXFP4) scale factors for ``key_pages_fp4``.
        value_pages_fp4: Packed E2M1 V pages.
        value_scales: E4M3 (NVFP4) or E8M0 (MXFP4) scale factors for ``value_pages_fp4``.
        fp4_page_table: INT32 logical-to-physical FP4 page mapping, shaped
            ``[rows, max_pages_per_row]``.
        seqused_fp4: INT32 FP4 token count for each row, shaped ``[rows]``.
            Every value must be a multiple of the page size. With an attention
            sink this remains the prefix on the original page grid; page zero
            may therefore contain a masked copy of the sink.
        residual_key_pages_bf16: Optional physical BF16 K-page cache.
        residual_value_pages_bf16: Optional physical BF16 V-page cache.
        residual_page_ids: Optional INT32 physical BF16 page ID per row.
        seqused_residual: Optional INT32 valid BF16 token count per row.
        seqused_sink: Optional INT32 valid BF16 attention-sink token count per
            row. When supplied, the sink occupies rows
            ``[RESIDUAL_ROW_TILE, RESIDUAL_ROW_TILE + SINK_TOKENS)`` of the
            same physical page selected by ``residual_page_ids``. The first
            ``seqused_sink`` positions of FP4 page zero are treated as
            duplicates and masked. Before page zero is promoted, the same
            prefix is masked in the rolling BF16 page instead.
        has_bf16: Optional BOOL indicator of whether each row owns an active
            BF16 residual page. It must agree with ``seqused_residual > 0``.
        softmax_scale: Score scale. The implementation uses
            ``head_dim**-0.5`` when omitted.
        query_row_indices: Optional INT32 mapping from compact decode rows to
            rows in a full 3-D query tensor.
        out: Optional BF16 output tensor in ``[rows, heads_q, head_dim]``
            layout, written in place instead of a fresh allocation. Row i of
            the batch lands in row i unless ``out_indices`` says otherwise.
        out_indices: Optional INT32 mapping from compact decode rows to rows
            in ``out``. Costs split-K, which cannot reorder rows on its way
            out, so pass it only when the rows really do have to move.
        num_splits: How many key partitions the decode is split across, as a
            positive integer. One CTA serves one row and KV head per
            split, so the count is what fills the device when the batch alone
            cannot. The right value depends on the batch and the SM count,
            neither of which this function knows, hence the default of one;
            ``nvfp4_vllm`` chooses it per step from the decode row count.
            Never derive it from the page table: under CUDA graph capture the
            table is frozen at the model's maximum length, and a count read
            from it would be baked into every later replay.

            The count partitions the FP4 pages. A fused residual's tail and
            sink are merged once, either by a split of their own launched
            beside them or by split 0; ``split_partial_count`` says which.

            A value above one is honoured only where the split path can serve
            it: no residual argument at all, or a complete residual on the
            BF16 query path, and never together with ``out_indices``, whose
            reordering the combine cannot perform.
            Asking for a split anywhere else raises, because a request quietly
            downgraded to one tile would look exactly like a split that ran.
        query_padded_scratch: Optional BF16 buffer shaped ``[at least rows,
            RESIDUAL_ROW_TILE, heads_q, head_dim]`` that holds the query padded
            to the residual MMA's row tile. Only the first row of each tile is
            written, so a caller-owned buffer has to be zeroed once and never
            again. Supplying one avoids reallocating and rezeroing it on every
            call, which matters when a residual is present on every layer of
            every step.
        query_fp4_scratch: Optional UINT8 buffer shaped ``[at least rows, 1,
            heads_q, head_dim // 2]`` that receives the packed E2M1 query. The
            quantizer rewrites every byte of it, so whatever it arrives
            holding is irrelevant.
        query_scales_scratch: Optional UINT8 buffer shaped ``[at least rows,
            1, heads_q, head_dim // (4 * sf_vec_size), 32, 4, 4]`` that receives
            the query scale factors. ``sf_vec_size`` is 16 for NVFP4 and
            32 for MXFP4. Unlike ``query_fp4_scratch`` this one carries
            the same obligation as ``query_padded_scratch``, and for a
            sharper reason: the layout reserves ``32 * 4`` scale slots per KV
            head and only the ``heads_q // heads_kv`` of them that carry a
            query head are ever written. Nothing clears the others, so a
            buffer zeroed once is byte-for-byte what an internal allocation
            would have produced, while one handed over uninitialized feeds the
            MMA whatever it found for as long as it is reused. No shape check
            can catch that.

            For the same reason a buffer belongs to one ``(heads_q, heads_kv,
            head_dim)`` triple for its whole life. Its shape says nothing
            about ``heads_kv``, so the shape check cannot notice the reuse,
            but ``heads_q // heads_kv`` is exactly what picks the slots that
            get written: at ``heads_kv == 1`` query head ``h`` lands in slot
            ``h`` of the single KV head's tile, and at ``heads_kv == 8`` in
            slot ``h % (heads_q // 8)`` of tile ``h // (heads_q // 8)``.
            Handing the first buffer to the second geometry leaves the slots
            only the first one wrote still holding its scales, where the
            second needs zeros.
        split_partial_scratch: Optional contiguous 1-D FP32 buffer, starting
            at storage offset zero, that holds the split path's partial
            outputs: at least ``split_partial_count(...) * rows * heads_q *
            head_dim`` elements, which with a residual can be one partial per
            row more than ``num_splits``. Unlike the query scratch it carries
            no obligation to be zeroed, then or ever: every split writes its
            LSE, an empty one -inf, and the combine adds a partial output only
            under a positive weight. Ignored at one split.
        split_lse_scratch: The same for the split path's log-sum-exps: at
            least ``split_partial_count(...) * rows * heads_q`` elements.
            Supplying both
            means a split decode allocates and fills nothing, which matters
            under CUDA graph capture, where both would otherwise be replayed
            on every layer of every step.
        scale_mode: Which block-scale candidates the query quantizer searches;
            see :mod:`.scale_mode`. Applies to the BF16 query path only — on
            the pre-quantized path the caller has already chosen, and naming a
            second mode here is refused rather than ignored. It does not
            describe the pages, which were written by whatever mode was in
            force then; nothing in the stored format records which.
        query_terms: One or two independently scaled NVFP4 query terms. Two
            quantizes the residual of the first term and accumulates both QK
            products before softmax.
        query_rotation: ``"hadamard"`` multiplies each BF16 query head by the
            normalized Sylvester Hadamard inside the quantizer, in FP32, and
            quantizes every term from those FP32 values; the BF16 query the
            residual MMA reads is the same rotation rounded once. The pages
            must then hold keys in the same basis. BF16 query path only, like
            ``scale_mode``.
        tail_write: Optional :class:`DecodeTailWrite`. The query quantizer
            then also copies each decode row's new K/V token into the BF16
            tail from extra CTAs of its own launch, so the decode launch that
            follows reads it with no separate write kernel ahead of them. It
            stores K as ``KVCacheWriter`` does, byte for byte, and is not a
            quantization: whole FP4 pages still come only from the serving
            code. BF16 query path only.

    Returns:
        A compact BF16 output for the selected query rows. It is a view of
        ``out`` when one was supplied rather than a fresh allocation: the
        whole of ``out`` when ``out_indices`` scattered the rows across it,
        and ``out[:rows]`` otherwise, because the rows past the batch belong
        to whoever else shares the buffer.
    """
    try:
        from ._kernel import fp4_decode_impl
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}._kernel":
            raise
        raise KernelNotAvailableError(
            "NVFP4 kernel implementation is unavailable."
        ) from error

    required = {
        "key_pages_fp4": key_pages_fp4,
        "key_scales": key_scales,
        "value_pages_fp4": value_pages_fp4,
        "value_scales": value_scales,
        "fp4_page_table": fp4_page_table,
        "seqused_fp4": seqused_fp4,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"missing required decode arguments: {', '.join(missing)}"
        )

    return fp4_decode_impl(
        query=query,
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
        residual_key_pages_bf16=residual_key_pages_bf16,
        residual_value_pages_bf16=residual_value_pages_bf16,
        residual_page_ids=residual_page_ids,
        seqused_residual=seqused_residual,
        seqused_sink=seqused_sink,
        has_bf16=has_bf16,
        softmax_scale=softmax_scale,
        query_row_indices=query_row_indices,
        out=out,
        out_indices=out_indices,
        trusted_metadata=trusted_metadata,
        num_splits=num_splits,
        query_padded_scratch=query_padded_scratch,
        query_fp4_scratch=query_fp4_scratch,
        query_scales_scratch=query_scales_scratch,
        query_fp4_second_scratch=query_fp4_second_scratch,
        query_scales_second_scratch=query_scales_second_scratch,
        split_partial_scratch=split_partial_scratch,
        split_lse_scratch=split_lse_scratch,
        query_terms=query_terms,
        scale_mode=scale_mode,
        query_rotation=query_rotation,
        tail_write=tail_write,
    )
