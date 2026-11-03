"""Per-step attention metadata for the NVFP4 KV cache.

Carries FlashAttention's own metadata unchanged and appends what the control
plane resolved for this step. Subclassing rather than wrapping keeps every
non-NVFP4 layer working: until the decode kernel is wired in, the layers still
run FlashAttention, and they read these objects through the base class's
fields.

Most of the fields below are device tensors written by the control kernel.
Reading any of those on the host costs a synchronization, so nothing outside a
kernel should. The plain ints are the exception, and are marked as such: they
come from CPU-side batch descriptions, never from the device.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from .control import ControlOutputs


# kw_only because the base class ends in defaulted fields, which would
# otherwise forbid adding required ones.
@dataclass(kw_only=True)
class NVFP4Metadata(FlashAttentionMetadata):
    """FlashAttention metadata plus this step's tail-slot assignment."""

    row_to_slot: torch.Tensor
    """``[num_reqs]`` int32. Tail slot owning each batch row, -1 if the row is
    padding."""

    token_to_slot: torch.Tensor
    """``[num_actual_tokens]`` int32. Tail slot each token belongs to, so the
    KV write can find its destination without a per-row loop."""

    seqused_sink: torch.Tensor
    """``[max_num_seqs]`` int32. Tokens of each sequence held permanently in
    the BF16 sink. Full width and zero for graph-padding rows."""

    seqused_fp4: torch.Tensor
    """``[num_reqs]`` int32. Prefix covered by whole FP4 pages on vLLM's
    original page grid. Page zero includes a masked copy of the BF16 sink."""

    seqused_residual: torch.Tensor
    """``[num_reqs]`` int32. Tokens of each sequence that live in the BF16
    rolling tail. Together with ``seqused_fp4`` this sums to the sequence
    length; ``seqused_sink`` is a duplicated precision overlay."""

    promotion_source_tokens: torch.Tensor
    """``[max_num_seqs]`` int32. First token of each row's tail, as an index
    into one layer's tail buffer."""

    promotion_pages: torch.Tensor
    """``[max_num_seqs]`` int32. Block that a row's tail must be quantized into
    before the next step, -1 for a row whose tail did not fill a page.

    Full width rather than ``num_reqs`` wide, unlike the fields above: this
    pair is the promotion launch's grid, and a grid that followed the batch
    would put a host branch back on a path that has none."""

    decode_prefix_rows: int
    """How many rows of this batch emit exactly one token. vLLM has already
    moved them to the front, so they are rows ``[0, decode_prefix_rows)``. A
    host int, not a device tensor: it comes from ``query_start_loc_cpu``, so
    reading it is free.

    Not the base class's ``num_decode_reqs``, which counts something else
    (rows that carry decode context under decode context parallelism) and is
    left at zero outside that feature."""

    decode_prefix_tokens: int
    """Where the prefill tokens begin. Equal to ``decode_prefix_rows`` while
    every decode row emits one token, but derived rather than assumed."""

    prefill_query_start_loc: torch.Tensor
    """``[num_prefills + 1]`` int32. ``query_start_loc`` rebased on the first
    prefill token, which is what FlashAttention needs once the decode prefix
    has been sliced off. Built here rather than per layer because every layer
    of the step would otherwise recompute the same subtraction on the
    device."""

    decode_page_columns: int
    """How many page-table columns the decode kernel has to walk. The kernel
    reads the table's width as the longest row it could see, which is the
    length it validates ``seqused_fp4`` against and the extent its tile
    scheduler is sized for, so a table left at the model's capacity makes a
    short batch look long.

    The batch's longest row, not the decodes' own: the per-row lengths live on
    the device, and vLLM's host-side copy is materialized by a transfer that
    would cost the step a synchronization. A prefill sharing the step can only
    widen this, never narrow it wrongly. The final page on the original vLLM
    grid is always the rolling tail rather than an FP4 page."""

    decode_num_splits: int
    """How many key partitions the decode kernel splits into, from
    ``builder.decode_num_splits``. A host int chosen from the decode row count
    alone, so a graph captured at a width keeps the count right for every
    batch it later replays."""

    source_tokens: torch.Tensor
    """``[work]`` int32. First token of each full page this step must quantize,
    as an index into the step's flattened K/V."""

    destination_pages: torch.Tensor
    """``[work]`` int32. Block each of those pages belongs in, -1 where the
    grid slot has no work. Paired with ``source_tokens`` and shared by every
    layer, since which pages exist does not depend on the layer."""

    @classmethod
    def from_flash(
        cls,
        base: FlashAttentionMetadata,
        outputs: ControlOutputs,
        *,
        source_tokens: torch.Tensor,
        destination_pages: torch.Tensor,
        decode_prefix_rows: int,
        decode_prefix_tokens: int,
        prefill_query_start_loc: torch.Tensor,
        decode_page_columns: int,
        decode_num_splits: int,
    ) -> "NVFP4Metadata":
        """Carry every FlashAttention field across and append the slot fields.

        Copied field by field rather than mutated in place: the base object is
        what FlashAttention's own builder returned, and a metadata object is
        read by every layer in the step.
        """
        return cls(
            **{field.name: getattr(base, field.name) for field in fields(base)},
            row_to_slot=outputs.row_to_slot,
            token_to_slot=outputs.token_to_slot,
            seqused_sink=outputs.seqused_sink,
            seqused_fp4=outputs.seqused_fp4,
            seqused_residual=outputs.seqused_residual,
            promotion_source_tokens=outputs.promotion_source_tokens,
            promotion_pages=outputs.promotion_pages,
            decode_prefix_rows=decode_prefix_rows,
            decode_prefix_tokens=decode_prefix_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            decode_page_columns=decode_page_columns,
            decode_num_splits=decode_num_splits,
            source_tokens=source_tokens,
            destination_pages=destination_pages,
        )
