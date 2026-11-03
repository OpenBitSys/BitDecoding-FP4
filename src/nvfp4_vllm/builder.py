"""Assign BF16 tail slots once per scheduler step, before attention layers run.

The control plane's counter tracks scheduler steps. Microbatching would build
metadata more than once per step and is rejected by ``guards``.

CUDA graph capture uses null-block rows, which receive no slots. The first
real ``build`` after capture resets the table. Startup profiling also uses
null-block rows; abandoned warmup slots are reclaimed lazily.
"""

from __future__ import annotations

import os

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

from .control import PAGE_SIZE, ControlPlane, split_sequence_length
from .guards import NVFP4, check_supported
from .metadata import NVFP4Metadata
from .write import PageWorkTable


# Read the control plane's sticky error word after each step for debugging.
# Disabled by default because reading it synchronizes the host and device.
DEBUG_ENV = "NVFP4_DEBUG"

# The largest power-of-two split count. A count this high already leaves an 8k
# context two pages per split; filling the SMs it leaves idle can take a count
# up to one short of twice it.
MAX_DECODE_SPLITS = 32


def decode_num_splits(rows: int, num_kv_heads: int, sms: int) -> int:
    """Choose key partitions from batch width, KV heads and available SMs.

    Start with a power of two, then fill otherwise idle SMs while reserving
    one CTA per row and KV head for the BF16 tail and sink. Stay below the
    next power of two to retain the same combine tier.

    The choice uses no sequence lengths, so it remains valid when a captured
    CUDA graph is replayed at a different context length.
    """
    if rows < 1:
        return 1
    ctas_per_split = rows * num_kv_heads
    budget = min(sms // ctas_per_split, MAX_DECODE_SPLITS)
    splits = 1
    while splits * 2 <= budget:
        splits *= 2
    filled = sms // ctas_per_split - 1
    return filled if splits < filled < 2 * splits else splits


def decode_split(
    common_attn_metadata: CommonAttentionMetadata,
) -> tuple[int, int]:
    """Where this step's decode prefix ends, in rows and in tokens.

    The one place this module looks at a batch on the host, and it is allowed
    to because ``query_start_loc_cpu`` is something the model runner already
    materialized, so reading it costs no synchronization. What it guards is the
    premise the whole read path rests on — that the reordering actually
    happened. Without it, a vLLM that stopped reordering would have the decode
    kernel attend to prefill rows' caches with decode rows' queries, which is
    wrong everywhere and loud nowhere.

    A batch dispatched to a full CUDA graph is padded out to the captured
    width with rows carrying no token at all. Those rows stay inside the
    returned prefix, because the kernels have to keep the shape the graph was
    captured at, and the control plane gives them no slot and no length. They
    are only ever a suffix: vLLM pads a uniform decode batch and nothing else,
    so a zero-length row among the real ones would mean something other than
    padding put it there.
    """
    num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
        common_attn_metadata, decode_threshold=1
    )
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    query_lens = query_start_loc[1 : common_attn_metadata.num_reqs + 1] - (
        query_start_loc[: common_attn_metadata.num_reqs]
    )
    decode_lens = query_lens[:num_decodes]
    # Where the padding starts, if there is any. Comparing only below it also
    # decides that the zero-length rows are contiguous at the end, since a zero
    # anywhere earlier falls inside the compared range.
    scheduled = int((decode_lens > 0).sum())
    # The last condition is the one that can fire. vLLM counts the prefix by
    # scanning one-token rows from the front, so an unsorted batch gives a
    # prefix that is too short rather than a wrong one, and what gives it away
    # is a one-token row left behind among the prefills.
    if not bool((decode_lens[:scheduled] == 1).all()) or not bool(
        (query_lens[num_decodes:] > 1).all()
    ):
        raise ValueError(
            "the NVFP4 read path needs one-token requests at the front of the "
            f"batch, but query lengths are {query_lens.tolist()} with a decode "
            f"prefix of {num_decodes}"
        )
    return num_decodes, num_decode_tokens


class NVFP4MetadataBuilder(FlashAttentionMetadataBuilder):
    """FlashAttention's builder, extended with this step's slot assignment.

    A pass-through under any other cache dtype: the backend is selected per
    engine, not per cache dtype, so this class is also what a BF16 run gets.

    ``supports_update_block_table`` stays inherited. It is what keeps a model
    with several KV cache groups to one ``build()`` per step, and the base
    implementation shallow-copies the metadata object, so the slot fields
    below survive the copy.

    Under NVFP4 it also asks vLLM to sort decode requests to the front of the
    batch; see ``decode_split``.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """How much of a batch this backend can have captured.

        FlashAttention answers ``ALWAYS`` on FA3, which claims a mixed
        prefill+decode batch can go into one graph. That is true of it and not
        of us: an NVFP4 prefill runs varlen FlashAttention over a token count
        that changes every step, so only the uniform one-token decode batches
        can be frozen. vLLM reads this before any builder is constructed, so
        the cache dtype has to come from the configuration rather than from
        ``self.nvfp4``.
        """
        cache_config = vllm_config.cache_config
        if cache_config is None or cache_config.cache_dtype != NVFP4:
            return super().get_cudagraph_support(vllm_config, kv_cache_spec)
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        cache_config = vllm_config.cache_config
        self.nvfp4 = cache_config is not None and cache_config.cache_dtype == NVFP4
        # Raised once capture has asked for metadata, and read by the first
        # ``build`` that is a real step again. ``capturing`` distinguishes the
        # builds capture drives from that step, since both reach ``build``.
        self.built_for_capture = False
        self.capturing = False
        if not self.nvfp4:
            self.plane = None
            return

        # Read once. Per step this has to be a bool test, not a dict lookup.
        self.debug = os.environ.get(DEBUG_ENV) == "1"

        # Ask the model runner to move one-token requests to the front of the
        # batch. That makes the decode rows a contiguous prefix, so the decode
        # kernel takes slices of this step's arrays instead of a compacted copy
        # of them, and its output is a slice of vLLM's. Left unset above, so a
        # BF16 run through this backend keeps FlashAttention's batch order.
        self._init_reorder_batch_threshold(1)

        # Already enforced when the layers were constructed. Repeated here so a
        # builder is safe to construct on its own, and so the control plane is
        # never sized from a configuration it cannot serve.
        check_supported(vllm_config)

        scheduler_config = vllm_config.scheduler_config
        self.plane = ControlPlane(
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            device=device,
        )
        # Which full pages this step writes depends only on the batch, so it is
        # resolved here alongside the slot assignment rather than once per
        # layer.
        self.work_table = PageWorkTable(
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            device=device,
        )
        # Asked once: the split count is chosen per step, and a device
        # property query on that path would be a host call per step.
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashAttentionMetadata:
        """Metadata for a batch being recorded into a graph, not served.

        Nothing here differs from a normal build; the point is the record that
        it happened, so that ``build`` can throw away whatever capture left in
        the slot table. Capture is expected to leave nothing — every row of a
        capture batch keys on the null block and is not live — but this is the
        one moment in an engine's life when no request exists to lose, so the
        cheap thing to do is not rely on that.
        The base implementation routes straight back into ``build``, so the
        flag is raised only once that has returned. Raising it first would
        have ``build`` consume it on the way through and leave it low when
        capture ends, which is the one moment it has to be high.
        """
        self.capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self.capturing = False
            if self.nvfp4:
                self.built_for_capture = True

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if not self.nvfp4:
            return base

        if self.built_for_capture and not self.capturing:
            # First step after capture, and only that one: the builds capture
            # itself makes arrive through here too, and clearing the table
            # between them would just be undone by the next one. The table
            # goes back to the state it had before the engine warmed up,
            # including the step counter each captured size advanced.
            self.built_for_capture = False
            self.plane.reset()

        # seq_lens is read as num_computed_tokens + num_scheduled_tokens, and
        # the FP4/BF16 page split is derived from it, so the other reading would
        # move every page boundary by one step. Nothing asserts it here: the
        # host copies the model runner computes it from are not on
        # CommonAttentionMetadata, and the reading is self-reporting anyway. If
        # it ever flips, a fresh prefill arrives with a zero length, is taken
        # for padding, gets no slot, and returns one step later with history it
        # cannot place, which is ERR_SLOT_LOST.
        outputs = self.plane.prepare(
            block_table=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )

        if self.debug:
            # Placed after prepare rather than before, so the step that broke
            # the invariant is the step that raises. The flags are sticky, so
            # every later step would raise too, but the first one is the one
            # with a batch worth looking at.
            self.plane.raise_for_errors()

        source_tokens, destination_pages = self.work_table.build(
            query_start_loc=common_attn_metadata.query_start_loc,
            seqused_fp4=outputs.seqused_fp4,
            row_to_slot=outputs.row_to_slot,
            block_table=common_attn_metadata.block_table_tensor,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
        )

        decode_rows, decode_tokens = decode_split(common_attn_metadata)
        query_start_loc = common_attn_metadata.query_start_loc
        if decode_tokens:
            # Only a mixed batch pays for this. A step of nothing but prefills
            # already starts at zero, which is the shape a prefill step
            # normally has.
            query_start_loc = query_start_loc[decode_rows:] - decode_tokens

        _, max_fp4, _ = split_sequence_length(
            common_attn_metadata.max_seq_len
        )
        return NVFP4Metadata.from_flash(
            base,
            outputs,
            source_tokens=source_tokens,
            destination_pages=destination_pages,
            decode_prefix_rows=decode_rows,
            decode_prefix_tokens=decode_tokens,
            prefill_query_start_loc=query_start_loc,
            decode_page_columns=max(1, max_fp4 // PAGE_SIZE),
            decode_num_splits=decode_num_splits(
                decode_rows, self.num_kv_heads, self.sms
            ),
        )
