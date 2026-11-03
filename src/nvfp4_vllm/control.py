"""Per-step bookkeeping for the BF16 tail slots, resolved entirely on device.

A sequence keeps its first 16 tokens in a permanent BF16 attention sink. The
FP4 cache keeps vLLM's original 128-token page grid, including a quantized copy
of those tokens in page zero; decode masks that copy so the sink contributes
exactly once. Keeping the old grid matters because FP4 P and online softmax are
page-local and therefore not invariant to moving every page boundary.

A suffix whose length is not a multiple of the page size is a partial page
that cannot be quantized, because V is packed along the token axis a page at a
time. Those trailing tokens stay in a rolling BF16 buffer until the page
fills. The sink and rolling page share one buffer per running request: a
*slot*.

vLLM never learns that these buffers exist, so it never tells us when one is
dead. It cannot: ``CommonAttentionMetadata`` carries no request identity, and
``finished_req_ids`` lives in ``SchedulerOutput``, which no attention backend
sees. Slot lifetime therefore has to be inferred from what does arrive each
step, which is three device tensors.

The inference keys on ``block_table[row, 0]``, the physical block holding
logical page 0. It is fixed for the life of a request and unique among live
requests, whereas the batch row index is not: ``InputBatch.condense()`` slides
the last live row into the hole a finished request leaves.

Slots are reclaimed lazily. A slot is never released because its key is absent
from a step: the scheduler can leave a live request out of a batch when its
token budget runs out, and releasing that request's tail would silently lose
the only copy of its most recent keys and values. Instead a slot is taken only
when a new key needs one, and the victim is the one whose owner has been unseen
longest. The stronger failure mode this replaces is still detected: a row that
has computed tokens but matches no slot is reported through ``error_code``.

A row is live when its sequence length is nonzero and its key is not vLLM's
null block. Neither test is redundant. Padding rows, which vLLM appends to
reach a captured width for full CUDA graphs, get a zero sequence length but
keep whatever block table entry ``condense()`` left behind, so counting rows
would hand a slot to a request that finished several steps ago. The startup
profile run goes the other way: it reports a full batch of long sequences over
an untouched block table, so every row keys on block 0, which the block pool
reserves as its null block and never allocates. Those rows are not requests.

Everything runs in a single CTA. With a slot table this small the whole
matching problem is a ``[rows, slots]`` comparison held in registers, and one
CTA can mutate the table without grid-wide synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from nvfp4_decode_kernel.interface import SINK_TOKENS


PAGE_SIZE = 128
TAIL_BUFFER_TOKENS = PAGE_SIZE + SINK_TOKENS


def split_sequence_length(
    sequence_length: int,
    *,
    sink_tokens: int = SINK_TOKENS,
    page_size: int = PAGE_SIZE,
) -> tuple[int, int, int]:
    """Describe the BF16 sink over the unchanged FP4 page partition.

    The sink is duplicated in page zero once that page is promoted. Decode
    masks the FP4 copy, so unlike disjoint storage classes these three lengths
    need not sum to the logical sequence length.
    """
    if sequence_length < 0:
        raise ValueError(
            f"sequence_length must be nonnegative, got {sequence_length}"
        )
    if sink_tokens < 0:
        raise ValueError(f"sink_tokens must be nonnegative, got {sink_tokens}")
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")

    sink = min(sequence_length, sink_tokens)
    fp4 = (
        ((sequence_length - 1) // page_size) * page_size
        if sequence_length
        else 0
    )
    return sink, fp4, sequence_length - fp4

# Largest supported slot table. BF16 sink/tail storage grows with slot count,
# while the single-CTA control kernel compares a square matrix of rows and
# slots. Larger tables cause excessive register spilling and require a
# different control-kernel layout.
MAX_SUPPORTED_SLOTS = 128

FREE_KEY = -1
# vLLM's block pool takes block 0 out of the free list at startup and hands it
# to no one, so a row keyed on it is not a request. Dummy batches read it
# straight out of the zero-initialized block table.
NULL_BLOCK = 0
# Marks a row or token that owns no tail slot. A real slot index would be
# indistinguishable from a live entry, so the write path would extend some
# other request's tail with tokens that are not its own.
INACTIVE_ROW = -1

# Sticky bit flags in ``error_code``. Nothing on the hot path reads them;
# reading one costs a host synchronization. They exist so that a violated
# assumption is attributable after the fact rather than silent.
ERR_CONTINUATION_PREFILL = 1
ERR_SLOT_LOST = 2
ERR_STALE_SLOT_HISTORY = 4
ERR_NO_FREE_SLOT = 8
ERR_DUPLICATE_KEY = 16
ERR_PROMOTION_COLUMN = 32

ERROR_NAMES = {
    ERR_CONTINUATION_PREFILL: "a prompt arrived split across steps",
    ERR_SLOT_LOST: "a row with history matched no slot",
    ERR_STALE_SLOT_HISTORY: "a matched slot's length does not continue",
    ERR_NO_FREE_SLOT: "a row needing a slot found none free",
    ERR_DUPLICATE_KEY: "two live rows share a physical block",
    ERR_PROMOTION_COLUMN: "a promotion column is past the block table",
}

_TOKEN_BLOCK = 256

# Triton cannot close over plain Python globals, only over constexpr ones.
_FREE = tl.constexpr(FREE_KEY)
_NULL_BLOCK = tl.constexpr(NULL_BLOCK)
_INACTIVE = tl.constexpr(INACTIVE_ROW)
_E_CONTINUATION_PREFILL = tl.constexpr(ERR_CONTINUATION_PREFILL)
_E_SLOT_LOST = tl.constexpr(ERR_SLOT_LOST)
_E_STALE_SLOT_HISTORY = tl.constexpr(ERR_STALE_SLOT_HISTORY)
_E_NO_FREE_SLOT = tl.constexpr(ERR_NO_FREE_SLOT)
_E_DUPLICATE_KEY = tl.constexpr(ERR_DUPLICATE_KEY)
_E_PROMOTION_COLUMN = tl.constexpr(ERR_PROMOTION_COLUMN)


@triton.jit
def _control_kernel(
    block_table_ptr,
    block_table_stride,
    block_table_columns,
    seq_lens_ptr,
    query_start_loc_ptr,
    slot_keys_ptr,
    slot_last_seq_ptr,
    slot_last_seen_ptr,
    step_ptr,
    row_to_slot_ptr,
    token_to_slot_ptr,
    seqused_sink_ptr,
    seqused_fp4_ptr,
    seqused_residual_ptr,
    promotion_source_tokens_ptr,
    promotion_pages_ptr,
    error_code_ptr,
    num_reqs,
    num_actual_tokens,
    NUM_SLOTS: tl.constexpr,
    SINK: tl.constexpr,
    TAIL_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)

    idx = tl.arange(0, BLOCK)
    in_batch = idx < num_reqs
    is_slot = idx < NUM_SLOTS

    # Column 0 of the block table, which is a strided read: vLLM owns the
    # tensor and reshaping it per step would cost an allocation and a copy.
    key = tl.load(block_table_ptr + idx * block_table_stride, mask=in_batch, other=_FREE)
    seq = tl.load(seq_lens_ptr + idx, mask=in_batch, other=0)
    q_begin = tl.load(query_start_loc_ptr + idx, mask=in_batch, other=0)
    q_end = tl.load(query_start_loc_ptr + idx + 1, mask=in_batch, other=0)
    query_len = q_end - q_begin

    # ``num_reqs`` only bounds how far the input tensors may be read. A row is
    # a request only if vLLM gave it both a length and a block: padding rows
    # keep a stale block table entry under a zeroed length, and the rows of a
    # dummy batch keep a nonzero length over the null block.
    live = in_batch & (seq > 0) & (key != _NULL_BLOCK)
    # Tokens already in the cache before this step. seq_lens is written as
    # num_computed_tokens + num_scheduled_tokens, so it already counts the
    # tokens this step will store.
    computed = seq - query_len

    slot_key = tl.load(slot_keys_ptr + idx, mask=is_slot, other=_FREE)
    slot_last_seq = tl.load(slot_last_seq_ptr + idx, mask=is_slot, other=-1)
    slot_last_seen = tl.load(slot_last_seen_ptr + idx, mask=is_slot, other=-1)

    # A dead row reads FREE_KEY, which is also an unused slot's key, so the
    # liveness of the row has to be part of the match.
    #
    # Reduced with min rather than a sum of matching indices: two live rows
    # carrying the same key is a broken invariant, not an impossible state, and
    # a sum would answer with an index outside the table. Min answers with a
    # real slot, and the duplicate itself is reported below.
    match = (key[:, None] == slot_key[None, :]) & live[:, None] & is_slot[None, :]
    matched_slot = tl.min(tl.where(match, idx[None, :], NUM_SLOTS), axis=1)
    has_match = matched_slot < NUM_SLOTS
    owns_match = match & (idx[None, :] == matched_slot[:, None])
    matched_last_seq = tl.sum(tl.where(owns_match, slot_last_seq[None, :], 0), axis=1)
    claimed = tl.sum(owns_match.to(tl.int32), axis=0) > 0

    # Rows without a slot take one of the slots no row claimed this step,
    # oldest first. Ranking both sides and pairing the ranks avoids a serial
    # scan over the table.
    needs_slot = live & (has_match == 0)
    need_rank = tl.cumsum(needs_slot.to(tl.int32), axis=0) - 1
    free = is_slot & (claimed == 0)
    older = free[None, :] & (
        (slot_last_seen[None, :] < slot_last_seen[:, None])
        | (
            (slot_last_seen[None, :] == slot_last_seen[:, None])
            & (idx[None, :] < idx[:, None])
        )
    )
    free_rank = tl.sum(older.to(tl.int32), axis=1)
    pick = (
        needs_slot[:, None]
        & free[None, :]
        & (free_rank[None, :] == need_rank[:, None])
    )
    allocated = tl.sum(tl.where(pick, idx[None, :], 0), axis=1)
    got_slot = tl.sum(pick.to(tl.int32), axis=1) > 0

    slot = tl.where(has_match, matched_slot, allocated)
    slot = tl.where(live, slot, _INACTIVE)

    errors = 0
    # A prompt split across steps would arrive with history and more than one
    # query token. Guarded against at configuration time; reported here so a
    # regression is attributable.
    errors += tl.where(
        tl.sum((live & (query_len > 1) & (computed > 0)).to(tl.int32), axis=0) > 0,
        _E_CONTINUATION_PREFILL,
        0,
    )
    # A row that has already computed tokens but matches no slot has lost its
    # tail. This is exactly the failure mode eager reclamation would cause.
    errors += tl.where(
        tl.sum((needs_slot & (computed > 0)).to(tl.int32), axis=0) > 0,
        _E_SLOT_LOST,
        0,
    )
    # A matched slot whose recorded length does not continue into this step
    # means the block id was recycled, and only a brand new request can be
    # given a recycled block, so it should have no history to continue.
    errors += tl.where(
        tl.sum(
            (has_match & (matched_last_seq != computed) & (computed > 0)).to(tl.int32),
            axis=0,
        )
        > 0,
        _E_STALE_SLOT_HISTORY,
        0,
    )
    # Unreachable through the caller, which refuses more rows than slots: a
    # needy row always has an unclaimed slot waiting. This checks the rank
    # pairing above rather than the input.
    errors += tl.where(
        tl.sum((needs_slot & (got_slot == 0)).to(tl.int32), axis=0) > 0,
        _E_NO_FREE_SLOT,
        0,
    )
    # Two live rows sharing a physical block would have to share a tail, which
    # no pair of live requests can legitimately need. Everything above answers
    # in range regardless, but the answer is meaningless.
    same_key = (key[:, None] == key[None, :]) & live[:, None] & live[None, :]
    errors += tl.where(
        tl.max(tl.sum(same_key.to(tl.int32), axis=1), axis=0) > 1,
        _E_DUPLICATE_KEY,
        0,
    )

    # Slot state follows its owner. Slots nobody claimed keep their old key so
    # a request that sits out a step can still find its tail. Min again, so a
    # slot two rows landed on takes one of them rather than their sum.
    owns = (slot[:, None] == idx[None, :]) & live[:, None] & is_slot[None, :]
    owned = tl.sum(owns.to(tl.int32), axis=0) > 0
    owner_row = tl.min(tl.where(owns, idx[:, None], BLOCK), axis=0)
    sole_owner = owns & (idx[:, None] == owner_row[None, :])
    owner_key = tl.sum(tl.where(sole_owner, key[:, None], 0), axis=0)
    owner_seq = tl.sum(tl.where(sole_owner, seq[:, None], 0), axis=0)
    tl.store(slot_keys_ptr + idx, tl.where(owned, owner_key, slot_key), mask=is_slot)
    tl.store(
        slot_last_seq_ptr + idx,
        tl.where(owned, owner_seq, slot_last_seq),
        mask=is_slot,
    )
    tl.store(
        slot_last_seen_ptr + idx,
        tl.where(owned, step, slot_last_seen),
        mask=is_slot,
    )

    tl.store(row_to_slot_ptr + idx, slot, mask=is_slot)

    # The first SINK tokens stay in the fixed BF16 sink, while FP4 retains the
    # original vLLM page grid. Page zero eventually contains a duplicate copy
    # that decode masks. Not shifting the grid keeps page-local FP4-P rounding
    # and online-softmax order unchanged for the rest of the sequence.
    # Dead rows are zeroed so graph padding attends to nothing.
    sink = tl.where(live, tl.minimum(seq, SINK), 0)
    live_seq = tl.where(live, seq, 0)
    seq_safe = tl.where(live_seq > 0, live_seq, 1)
    fp4 = tl.where(
        live_seq > 0, ((seq_safe - 1) // PAGE) * PAGE, 0
    )
    tl.store(seqused_sink_ptr + idx, sink, mask=is_slot)
    tl.store(seqused_fp4_ptr + idx, fp4, mask=is_slot)
    tl.store(seqused_residual_ptr + idx, live_seq - fp4, mask=is_slot)

    # Where promotion would read and write, were this the step that filled the
    # tail. Derived here because everything it needs is already in registers:
    # one more gather beats a second kernel launch on a path taken every step.
    #
    # ``column`` is the logical page the tail is filling, and a tail that is
    # exactly a page long is that page, complete. The other rows answer -1,
    # which is what tells the quantizer to leave them alone.
    column = fp4 // PAGE
    crossing = live & (seq > 0) & (seq % PAGE == 0)
    # Reaching past the block table would gather the next row's block id and
    # promote a whole page onto some other request's cache. Structurally that
    # cannot happen — vLLM allocated the block holding the row's last token —
    # but it is exactly the kind of damage nothing downstream could attribute.
    in_bounds = column < block_table_columns
    errors += tl.where(
        tl.sum((crossing & (in_bounds == 0)).to(tl.int32), axis=0) > 0,
        _E_PROMOTION_COLUMN,
        0,
    )
    tl.store(
        promotion_source_tokens_ptr + idx,
        tl.where(live, slot * TAIL_TOKENS, 0),
        mask=is_slot,
    )
    tl.store(
        promotion_pages_ptr + idx,
        tl.load(
            block_table_ptr + idx * block_table_stride + column,
            mask=crossing & in_bounds,
            other=_INACTIVE,
        ),
        mask=is_slot,
    )

    tl.atomic_or(error_code_ptr, errors)

    # Bounded by the tokens actually present, so a steady-state decode step
    # runs one iteration.
    for base in tl.range(0, num_actual_tokens, TOKEN_BLOCK):
        offs = base + tl.arange(0, TOKEN_BLOCK)
        in_range = offs < num_actual_tokens
        belongs = (
            (offs[:, None] >= q_begin[None, :])
            & (offs[:, None] < q_end[None, :])
            & live[None, :]
        )
        tl.store(
            token_to_slot_ptr + offs,
            tl.sum(tl.where(belongs, slot[None, :], 0), axis=1),
            mask=in_range,
        )


@dataclass(frozen=True)
class ControlOutputs:
    """Views into the control plane's scratch, valid until the next step.

    Nothing here may be read on the host during a decode step; every tensor
    stays on device and is consumed by later kernels.
    """

    row_to_slot: torch.Tensor
    token_to_slot: torch.Tensor
    # Full width so a captured decode sees one stable-shape sink-length tensor.
    seqused_sink: torch.Tensor
    seqused_fp4: torch.Tensor
    seqused_residual: torch.Tensor
    # Full width like seqused_sink: promotion launches over the whole table so
    # that its shape does not follow the batch.
    promotion_source_tokens: torch.Tensor
    promotion_pages: torch.Tensor
    error_code: torch.Tensor


class ControlPlaneError(RuntimeError):
    """An invariant the slot table reported through ``error_code``."""


class ControlPlane:
    """Slot table and per-step scratch for one engine.

    All buffers are allocated once at the high-water mark. Reallocating inside
    a step would both cost a synchronization and move the addresses a captured
    graph would have baked in.
    """

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        device: torch.device | str,
        page_size: int = PAGE_SIZE,
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        if max_num_seqs > MAX_SUPPORTED_SLOTS:
            raise ValueError(
                f"max_num_seqs={max_num_seqs} exceeds the validated "
                f"{MAX_SUPPORTED_SLOTS}-slot control-plane ceiling"
            )
        if max_num_batched_tokens < max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({max_num_batched_tokens}) must cover "
                f"one token per row ({max_num_seqs})"
            )

        self.num_slots = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.page_size = page_size
        self.device = torch.device(device)
        # Triton indexes the table with tl.arange, which needs a power of two.
        self._block = triton.next_power_of_2(max_num_seqs)

        def _slots(dtype: torch.dtype, fill) -> torch.Tensor:
            return torch.full(
                (self.num_slots,), fill, dtype=dtype, device=self.device
            )

        self.slot_keys = _slots(torch.int32, FREE_KEY)
        # -1 is not a reachable computed-token count, so a slot handed out for
        # the first time never looks like a continuation of anything.
        self.slot_last_seq = _slots(torch.int32, -1)
        # int64 so a long-running server cannot wrap the LRU ordering.
        self.slot_last_seen = _slots(torch.int64, -1)
        self.step = torch.zeros(1, dtype=torch.int64, device=self.device)

        self.row_to_slot = _slots(torch.int32, INACTIVE_ROW)
        self.seqused_sink = _slots(torch.int32, 0)
        self.seqused_fp4 = _slots(torch.int32, 0)
        self.seqused_residual = _slots(torch.int32, 0)
        self.promotion_source_tokens = _slots(torch.int32, 0)
        self.promotion_pages = _slots(torch.int32, INACTIVE_ROW)
        self.error_code = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.token_to_slot = torch.full(
            (max_num_batched_tokens,),
            INACTIVE_ROW,
            dtype=torch.int32,
            device=self.device,
        )

    def reset(self) -> None:
        """Forget every slot. Only valid when no request is in flight."""
        self.slot_keys.fill_(FREE_KEY)
        self.slot_last_seq.fill_(-1)
        self.slot_last_seen.fill_(-1)
        self.step.zero_()
        self.error_code.zero_()

    def raise_for_errors(self) -> None:
        """Read the sticky error word and raise if the kernel reported anything.

        Reading it costs a host synchronization, so nothing on the normal path
        calls this — see ``NVFP4_DEBUG``. The flags are sticky for the life of
        the engine, so a caller that reads only at the end still learns
        everything that went wrong, just not when.
        """
        code = int(self.error_code.item())
        if not code:
            return
        named = [
            text for bit, text in ERROR_NAMES.items() if code & bit
        ]
        unknown = code & ~sum(ERROR_NAMES)
        if unknown:
            named.append(f"unrecognized flags {unknown:#x}")
        raise ControlPlaneError(
            f"the slot control plane reported error_code {code}: "
            + "; ".join(named)
        )

    def prepare(
        self,
        *,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        num_actual_tokens: int,
    ) -> ControlOutputs:
        """Advance the slot table by one step and return this step's mapping.

        ``block_table`` is vLLM's ``[rows, max_blocks]`` tensor; only column 0
        is read, as the identity of the row's request. ``seq_lens`` must
        already include the tokens this step will write, as vLLM's model runner
        computes it, and a row whose length is zero is treated as padding.
        ``num_reqs`` and ``num_actual_tokens`` are host values the caller
        already has; nothing is read back from the device.
        """
        if block_table.ndim != 2:
            raise ValueError(f"block_table must be 2-D, got {block_table.ndim}-D")
        if num_reqs > self.num_slots:
            raise ValueError(
                f"{num_reqs} rows exceeds the {self.num_slots} tail slots"
            )
        if num_actual_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"{num_actual_tokens} tokens exceeds the "
                f"{self.max_num_batched_tokens} the scratch was sized for"
            )

        _control_kernel[(1,)](
            block_table,
            block_table.stride(0),
            block_table.shape[1],
            seq_lens,
            query_start_loc,
            self.slot_keys,
            self.slot_last_seq,
            self.slot_last_seen,
            self.step,
            self.row_to_slot,
            self.token_to_slot,
            self.seqused_sink,
            self.seqused_fp4,
            self.seqused_residual,
            self.promotion_source_tokens,
            self.promotion_pages,
            self.error_code,
            num_reqs,
            num_actual_tokens,
            NUM_SLOTS=self.num_slots,
            SINK=SINK_TOKENS,
            TAIL_TOKENS=TAIL_BUFFER_TOKENS,
            PAGE=self.page_size,
            BLOCK=self._block,
            TOKEN_BLOCK=_TOKEN_BLOCK,
            num_warps=(
                32 if self._block == 128 else max(4, self._block // 8)
            ),
        )

        return ControlOutputs(
            row_to_slot=self.row_to_slot[:num_reqs],
            token_to_slot=self.token_to_slot[:num_actual_tokens],
            seqused_sink=self.seqused_sink,
            seqused_fp4=self.seqused_fp4[:num_reqs],
            seqused_residual=self.seqused_residual[:num_reqs],
            promotion_source_tokens=self.promotion_source_tokens,
            promotion_pages=self.promotion_pages,
            error_code=self.error_code,
        )
