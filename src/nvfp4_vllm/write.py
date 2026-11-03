"""Placing this step's K/V into the NVFP4 cache and the BF16 tail.

The first 16 tokens are copied into the permanent BF16 sink while the ordinary
vLLM page grid is preserved. Until page zero fills, those tokens are also in
the rolling BF16 page; after promotion they have an FP4 copy that decode masks.
The duplication avoids moving every later page boundary, which would change
page-local FP4-P rounding. The final partial page is copied into the rolling
BF16 tail, and the two BF16 regions share one slot.

Which tokens fall on which side is decided entirely on the device. The work
table and the layer's write (a ``KVCacheWriter`` launch) read the control
plane's per-row answers and the block table, so no length or offset is ever
brought back to the host, and every launch shape comes from host integers vLLM
already has.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from nvfp4_decode_kernel.quantize_kv_kernel import KVCacheWriter

from .control import PAGE_SIZE


@triton.jit
def _work_table_kernel(
    query_start_loc_ptr,
    seqused_fp4_ptr,
    row_to_slot_ptr,
    block_table_ptr,
    block_table_stride,
    source_tokens_ptr,
    destination_pages_ptr,
    num_reqs,
    columns_per_row,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Say, for each slot of the quantizer's grid, what to read and where.

    The grid is a fixed rectangle of rows by page columns, so most slots in a
    given step do nothing; those get a destination of -1, which the quantizer
    treats as no work. A row contributes pages only if it is a fresh prefill,
    because only then are the tokens of its full pages present in this step's
    activations. A decode row's earlier pages were quantized when they were
    written and must not be touched again.
    """
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    within = columns < columns_per_row

    start = tl.load(query_start_loc_ptr + row)
    end = tl.load(query_start_loc_ptr + row + 1)
    slot = tl.load(row_to_slot_ptr + row)
    full_pages = tl.load(seqused_fp4_ptr + row) // PAGE

    row_writes = (end - start > 1) & (slot >= 0)
    active = within & row_writes & (columns < full_pages)

    blocks = tl.load(
        block_table_ptr + row * block_table_stride + columns,
        mask=active,
        other=-1,
    )
    out = row * columns_per_row + columns
    tl.store(
        source_tokens_ptr + out,
        start + columns * PAGE,
        mask=within,
    )
    tl.store(
        destination_pages_ptr + out,
        tl.where(active, blocks, -1),
        mask=within,
    )


@triton.jit
def _tail_reset_kernel(
    tail_key_ptr,
    tail_value_ptr,
    tail_layer_stride,
    tail_slot_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    row_to_slot_ptr,
    num_reqs,
    WIDTH: tl.constexpr,
    TOKENS: tl.constexpr,
):
    """Clear the tail page a brand new request is about to move into.

    Attention reads a row's whole tail page and masks the positions past its
    length by weighting them zero, so those positions have to hold a finite
    number: zero times a NaN is a NaN, and one poisoned lane takes the row's
    entire output with it. The buffer is allocated zeroed for that reason, but
    a slot outlives the request that filled it, and the next occupant inherits
    whatever the last one left past its own length. Clearing on arrival is what
    keeps the allocation-time invariant true for every later tenant.

    A request is on its first step when the sequence is no longer than what it
    brought, which catches a single-token prompt that the batch reordering
    filed among the decodes, and a recycled block id that let a new request
    match some earlier tenant's slot. Both keep a stale page otherwise.
    """
    layer = tl.program_id(0)
    row = tl.program_id(1)

    slot = tl.load(row_to_slot_ptr + row, mask=row < num_reqs, other=-1)
    start = tl.load(query_start_loc_ptr + row, mask=row < num_reqs, other=0)
    end = tl.load(query_start_loc_ptr + row + 1, mask=row < num_reqs, other=0)
    seq_len = tl.load(seq_lens_ptr + row, mask=row < num_reqs, other=0)

    if (slot >= 0) & (seq_len <= end - start):
        lanes = tl.arange(0, WIDTH)
        zero = tl.zeros([WIDTH], dtype=tail_key_ptr.dtype.element_ty)
        for position in tl.range(0, TOKENS):
            destination = (
                layer * tail_layer_stride
                + slot * tail_slot_stride
                + position * WIDTH
                + lanes
            )
            tl.store(tail_key_ptr + destination, zero)
            tl.store(tail_value_ptr + destination, zero)


def reset_new_request_tails(
    *,
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    row_to_slot: torch.Tensor,
) -> None:
    """Clear every layer's tail page for the rows starting a request this step.

    Takes the tail for all layers at once and runs before any of them writes,
    rather than once per layer, because a slot's history has to end for the
    whole model at the same moment; a later layer clearing its own page would
    also have to do it after that layer had already read one.

    The grid covers every row whether or not it is new, since which rows are
    new is a device-side answer and asking the host would cost the very
    synchronization the control plane exists to avoid. One program owns a
    whole page so a row that is not new exits once instead of once per token;
    a new row clears the page in an inner loop.
    """
    num_layers, _, tokens, heads, head_dim = tail_key.shape
    _tail_reset_kernel[(num_layers, seq_lens.shape[0])](
        tail_key,
        tail_value,
        tail_key.stride(0),
        tail_key.stride(1),
        query_start_loc,
        seq_lens,
        row_to_slot,
        seq_lens.shape[0],
        WIDTH=heads * head_dim,
        TOKENS=tokens,
        num_warps=4,
    )


class PageWorkTable:
    """Per-step description of which full pages this step must quantize.

    Built once per step by the metadata builder and read by every layer. It is
    a dense rectangle rather than a compacted list because compacting would
    need a count the host does not have, and the rectangle is small: rows times
    pages per row, both bounded by the scheduler's limits.
    """

    def __init__(
        self, *, max_num_seqs: int, max_num_batched_tokens: int, device
    ) -> None:
        self.max_num_seqs = max_num_seqs
        # Chunked prefill is refused, so a prefill row's whole prompt arrives
        # in one step and its full pages cannot outnumber the step's tokens.
        self.max_columns = max_num_batched_tokens // PAGE_SIZE + 1
        capacity = max_num_seqs * self.max_columns
        self.source_tokens = torch.empty(
            capacity, dtype=torch.int32, device=device
        )
        self.destination_pages = torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        )

    def build(
        self,
        *,
        query_start_loc: torch.Tensor,
        seqused_fp4: torch.Tensor,
        row_to_slot: torch.Tensor,
        block_table: torch.Tensor,
        num_reqs: int,
        max_query_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A step where no row runs more than one token has no full pages to
        # write, and an empty table leaves every layer's write without page
        # CTAs, or without a launch when no tail token is its to write either.
        if max_query_len <= 1:
            return self.source_tokens[:0], self.destination_pages[:0]

        columns = min(
            max_query_len // PAGE_SIZE + 1,
            self.max_columns,
            block_table.shape[1],
        )
        items = num_reqs * columns
        _work_table_kernel[(num_reqs,)](
            query_start_loc,
            seqused_fp4,
            row_to_slot,
            block_table,
            block_table.stride(0),
            self.source_tokens,
            self.destination_pages,
            num_reqs,
            columns,
            PAGE=PAGE_SIZE,
            BLOCK=triton.next_power_of_2(columns),
            num_warps=4,
        )
        return self.source_tokens[:items], self.destination_pages[:items]


def write_kv(
    *,
    writer: KVCacheWriter,
    regions: tuple,
    key: torch.Tensor,
    value: torch.Tensor,
    source_tokens: torch.Tensor,
    destination_pages: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    seqused_fp4: torch.Tensor,
    row_to_slot: torch.Tensor,
    layer_index: int,
    decode_rows: int,
    first_row: int,
) -> None:
    """Place one layer's K/V for this step: full pages quantized, rest tailed.

    One launch of the runtime's writer (see ``LayerRuntime.writer_for``) into
    this layer's ``regions`` and its slice of the shared tail: rows
    ``[:128]`` of a slot are the rolling tail and rows ``[128:144]`` the
    permanent sink. ``key`` is what the writer puts into the cache's basis on
    the way in, which it does for the pages and the tail alike, so a key
    reaches the cache with the same bytes whichever of them stores it.

    ``key`` and ``value`` may run past this step's tokens (vLLM pads them);
    the writer reads only the tokens the work table and the rows name. Rows
    before ``first_row`` are left out of the tail: the decode prefix passes
    its own tokens to the query quantizer instead, which writes them with the
    same arithmetic in its own launch.
    """
    writer(
        key,
        value,
        regions,
        source_tokens,
        destination_pages,
        query_start_loc,
        seq_lens,
        seqused_fp4,
        row_to_slot,
        layer_index,
        decode_rows,
        first_row,
    )
