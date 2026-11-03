"""SM100 CuTeDSL kernel for hybrid FP4/BF16 paged decode."""

import enum
import math
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial
from dataclasses import dataclass
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, Float8E4M3FN, Float4E2M1FN
from cutlass.cute.experimental import iket
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import MmaMXF4NVF4Op, MmaMXF4Op
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from ._fa4.block_scaled_layout import make_smem_layout_sfa, make_smem_layout_sfb
import cutlass.utils.blockscaled_layout as blockscaled_utils
from ._fa4.paged_kv import PagedKVManager
from ._fa4 import utils as utils
from ._fa4 import copy_utils
from ._fa4 import pipeline as pipeline
from ._fa4.mask import AttentionMask
# from ._fa4.softmax import SoftmaxSm100, apply_score_mod_inner
from ._fa4.softmax import (
    SoftmaxSm100,
    _round_probability_code,
    _round_probability_scale,
    _amax2_or_amax6_probability_scale,
    _mxfp4_probability_exponent,
    apply_score_mod_inner,
)
from ._fa4.seqlen_info import SeqlenInfoQK
from ._fa4.block_info import BlockInfo
from ._fa4.block_sparsity import BlockSparseTensors
from ._fa4.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from ._fa4.pack_gqa import PackGQA
from ._fa4 import mma_sm100_desc as sm100_desc
from ._fa4 import blackwell_helpers as sm100_utils
from ._fa4.blackwell_helpers import packed_float_to_ue4m3, packed_float_to_e2m1
from ._fa4.fast_math import FastDivmod
from ._block_scale import inverse_power_of_two, round_up_exponent
from ._fa4.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)

def _sf_layout_with_page_stride(layout, page_stride):
    """Override the page pitch of a K/V block-scale layout.

    ``tile_to_shape`` derives every stride from the shape, which assumes the
    scale factors of consecutive pages sit back to back. When a page is a region
    of a wider cache page the pitch is larger, and it is the only stride that
    changes: the swizzle inside a page is fixed by the MMA. The page axis is
    mode 3, whose stride is ``(0, pitch)``.

    Reconstructing the layout drops the divisibility facts ``tile_to_shape``
    attached to the dynamic strides, so they are reapplied; the SFB TMA atom
    needs the 512-byte atom alignment to remain known.
    """
    if page_stride is None:
        return layout

    def assume_atom_aligned(value):
        if isinstance(value, int):
            return value
        return cute.assume(value, divby=512)

    stride = layout.stride
    return cute.make_layout(
        layout.shape,
        stride=(
            (stride[0][0], assume_atom_aligned(stride[0][1])),
            stride[1],
            (stride[2][0], assume_atom_aligned(stride[2][1])),
            (stride[3][0], assume_atom_aligned(page_stride)),
        ),
    )


# A rescale whose factor is within 2**-RESCALE_THRESHOLD of one is skipped:
# the correction pass over O costs more than the exponent headroom it buys.
RESCALE_THRESHOLD = 8.0


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # Publishes the per-warp partial row maxima between the four softmax warps
    # of one warpgroup, once per KV block when S is transposed. The warpgroups
    # own disjoint query rows and must not synchronise with each other, so each
    # takes its own barrier; there is one slot per possible row group.
    SoftmaxReduce = enum.auto()
    SoftmaxReduce1 = enum.auto()
    SoftmaxReduce2 = enum.auto()
    SoftmaxReduce3 = enum.auto()
#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class SplitRole(enum.Enum):
    """Which blocks one CTA of the launch merges.

    ``ALL`` is every CTA when there is no split of its own for the BF16 blocks:
    the tail, the CTA's FP4 pages and the sink, each as far as it is fused.
    With one, the FP4 splits are ``PAGES``, their pages alone, and the last
    split is ``BF16``, the tail and the sink alone. Each role is a separately
    specialised body of every warp role, chosen once per CTA.
    """

    ALL = enum.auto()
    PAGES = enum.auto()
    BF16 = enum.auto()


# How many ways the transposed softmax may split the query rows of a tile. The
# reduction takes one named barrier per group, counting up from SoftmaxReduce,
# so this many consecutive slots must exist in the enum above.
MAX_SOFTMAX_ROW_GROUPS = 4
assert (
    int(NamedBarrierFwd.SoftmaxReduce) + MAX_SOFTMAX_ROW_GROUPS - 1
    == int(NamedBarrierFwd.SoftmaxReduce3)
)

# A group runs one cross-warp reduction per KV tile — a butterfly, a named
# barrier and a round trip through shared memory — and that cost is per group,
# not per row, so it does not shrink as the group is handed fewer rows. Below
# this many rows a group spends more on its own reduction than the shorter
# serial chain saves: measured at batch 16, seqlen 128k, splitting four query
# rows four ways is 10 percent slower than splitting them two ways, while
# splitting eight and thirty-two rows four ways wins 5 and 10 percent.
MIN_ROWS_PER_SOFTMAX_GROUP = 2


class FP4DecodeKernel:
    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        sf_dtype: Optional[Type[cutlass.Numeric]] = None,
        sf_vec_size: Optional[int] = None,
        fused_residual_first_block: bool = False,
        fused_sink_block: bool = False,
        bf16_split: bool = False,
        residual_source: str = "contiguous",
        use_out_indices: bool = False,
        qhead_per_kvhead_O: cutlass.Constexpr[int] = None,
        seqlen_q_static_one: bool = False,
        transpose_s: bool = False,
        query_terms: cutlass.Constexpr[int] = 1,
        p_scale_divisor: cutlass.Constexpr[float] = 6.0,
        p_scale_grid_search: cutlass.Constexpr[bool] = False,
        p_scale_amax2_or_amax6: cutlass.Constexpr[bool] = False,
        p_scale_two_level: cutlass.Constexpr[bool] = False,
        p_scale_rescale_log2: cutlass.Constexpr[int] = 0,
    ):
        assert (sf_vec_size, sf_dtype) in (
            (16, cutlass.Float8E4M3FN), (32, cutlass.Float8E8M0FNU),
        ), "scales must be NVFP4 (16, E4M3) or MXFP4 (32, E8M0)"
        self.is_mxfp4 = sf_vec_size == 32
        assert not self.is_mxfp4 or (
            p_scale_divisor == 6.0 and not p_scale_grid_search
            and not p_scale_amax2_or_amax6 and not p_scale_two_level
            and p_scale_rescale_log2 == 0
        ), "MXFP4 P uses its own power-of-two scale search"
        assert query_terms in (1, 2)
        self.query_terms = query_terms
        assert p_scale_divisor in (2.0, 3.0, 4.0, 6.0)
        self.p_scale_divisor = p_scale_divisor
        self.p_scale_grid_search = p_scale_grid_search
        self.p_scale_amax2_or_amax6 = p_scale_amax2_or_amax6
        self.p_scale_two_level = p_scale_two_level
        if p_scale_rescale_log2 not in (0, 3):
            raise ValueError("P exponent offset must be zero or log2(8)")
        if p_scale_rescale_log2 and (not p_scale_amax2_or_amax6 or p_scale_two_level or p_scale_grid_search):
            raise ValueError("P exponent offset requires the amax2_or_amax6 ladder")
        self.p_scale_rescale_log2 = p_scale_rescale_log2
        assert not p_scale_two_level or (p_scale_divisor == 6.0 and not p_scale_grid_search and not p_scale_amax2_or_amax6)
        assert not (p_scale_grid_search and p_scale_amax2_or_amax6)
        self.fused_residual_first_block = fused_residual_first_block
        self.fused_sink_block = fused_sink_block
        assert not fused_sink_block or fused_residual_first_block, (
            "the BF16 sink reuses the residual cache and requires its prologue"
        )
        # The residual and the sink are single blocks outside the page range
        # split-k divides, so under split-k exactly one split may count them.
        # With bf16_split that is a split of their own, launched after the
        # FP4 splits: mO carries one more partial than there are page
        # partitions, the FP4 splits divide the pages among themselves alone,
        # and none of them waits for, loads or holds a BF16 block.
        # Without it every split runs both blocks and all but split 0 with a
        # length of zero, which the zero-length contract already makes
        # contribute nothing; split 0 is the one the scheduler always runs,
        # even when it holds no FP4 page.
        self.bf16_split = bf16_split
        assert not bf16_split or (is_split_kv and fused_residual_first_block), (
            "a split of its own for the BF16 blocks needs split-k and a residual"
        )
        self.residual_split_idx = 0
        assert residual_source in ("contiguous", "paged_bf16"), (
            f"unknown residual_source={residual_source!r}; "
            f"expected 'contiguous' or 'paged_bf16'"
        )
        self.residual_source = residual_source
        self.use_out_indices = use_out_indices
        self.seqlen_q_static_one = seqlen_q_static_one
        # Hold S as (kv position, query row) instead of (query row, kv position).
        # A tensor-memory load hands a thread one row and the same columns as
        # every other lane, so only the row axis can be narrowed per thread.
        # Decode leaves almost all of the query axis empty, and putting it in
        # the columns is what lets a thread exponentiate qhead_per_kvhead values
        # instead of a whole row of keys. The row reductions become cross-thread
        # in exchange. Resolved against the quantization flags in __call__.
        self.transpose_s_requested = transpose_s
        self.transpose_s = False
        self.tmem_sfpv_offset = None
        self.transposed_query_rows = 0
        self.stacked_query_terms = False
        self.narrow_pv = False
        self.pv_rows = 0
        self.softmax_row_groups = 1
        self.softmax_rows_per_group = 0
        self.softmax_scatters_rows = False
        self.softmax_red_slots = 0
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        # A second Q stage only earns its keep when a CTA carries two Q tiles.
        # With seqlen_q statically one the packed M extent is qhead_per_kvhead
        # rows under PackGQA and a single row without it, either way at most one
        # m_block_size, so stage 1 would TMA an out-of-range Q, get zeros, and
        # run a full softmax, TMEM round trip and PV MMA over them.
        if seqlen_q_static_one:
            assert qhead_per_kvhead <= m_block_size
        if is_split_kv or seqlen_q_static_one:
            self.q_stage = 1
        elif getattr(type(self), "_force_q_stage_1", False):
            self.q_stage = 1
        else:
            self.q_stage = 2
        assert self.q_stage in [1, 2]
        assert self.query_terms == 1 or self.q_stage == 1
        # 2 Q tile per CTA
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.qhead_per_kvhead_O = (
            qhead_per_kvhead_O if qhead_per_kvhead_O is not None else qhead_per_kvhead
        )
        assert self.qhead_per_kvhead_O >= self.qhead_per_kvhead, (
            "qhead_per_kvhead_O must be >= qhead_per_kvhead (pad up, never down)"
        )
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        # A single query position packs onto a flat head axis (see the PackGQA
        # layouts in __call__), which tiles for any ratio up to m_block_size.
        # Only the (head, seqlen) packing needs the tile to be a whole number
        # of query groups.
        if pack_gqa and not seqlen_q_static_one:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
            assert m_block_size % self.qhead_per_kvhead_O == 0, (
                "For PackGQA with separate O-axis, m_block_size must be divisible by qhead_per_kvhead_O"
            )
        # The MMA fixes the M extent at 128 rows while decode fills only
        # qhead_per_kvhead of them, and a softmax thread owns one row, so the
        # rows that carry no query still cost a full row of exp2. Repeating each
        # query row across the tile puts a real row under every thread, which is
        # what lets the column range a thread owns shrink by this factor. The
        # copies of one query row are consecutive, so a group of them lands
        # inside a single warp and the row reductions stay warp-local.
        # The split path writes FP32 partials and an LSE per row for the combine
        # kernel to consume, both keyed by the unreplicated row order, so it is
        # left alone until those two writers learn the same row selection the
        # unsplit epilogue uses.
        self.q_replicate = 1
        if (
            pack_gqa
            and seqlen_q_static_one
            and not is_split_kv
            and getattr(type(self), "_enable_q_replicate", False)
        ):
            self.q_replicate = m_block_size // self.qhead_per_kvhead
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 2
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv)
        )
        # BF16 blocks merged ahead of and behind a CTA's own FP4 pages hold
        # their tiles until the tile's last MMA, and O is staged only once that
        # MMA has completed, so on a CTA that owns one tile O takes the tiles'
        # bytes instead of widening sQ or holding its own.
        self.overlap_sO_bf16 = fused_residual_first_block and not bf16_split
        if self.overlap_sO_bf16:
            assert not is_persistent, (
                "O staged in the BF16 tiles needs one tile per CTA"
            )
            self.overlap_sO_sQ = False
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        # self.tmem_o_offset = self.tmem_s_offset
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_s_to_p_offset = self.n_block_size // 2

        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # e.g., 64, 192

        self.tmem_p_bf16_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # same as FP4 but aliased temporally — see comment above.

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        # The count a softmax warpgroup was tuned at. A CTA wide enough that it
        # cannot afford this narrows it; see _assign_warp_roles.
        if self.head_dim_padded < 96:
            self.num_regs_softmax_tuned = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        else:
            # self.num_regs_softmax_tuned = 192 if self.is_causal or self.is_local else 184
            self.num_regs_softmax_tuned = 216
            # self.num_regs_softmax_tuned = 176
            # self.num_regs_correction = 96
            # self.num_regs_correction = 80
            # self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
            self.num_regs_correction = 48
            # self.num_regs_other = 32
            # self.num_regs_other = 64
            # self.num_regs_other = 80
            self.num_regs_other = 24
            # self.num_regs_other = 96 if self.is_causal or self.is_local else 80
            # self.num_regs_other = 64 if self.is_causal or self.is_local else 80
        self.num_regs_empty = 24
        # Two softmax warpgroups is the layout every configuration starts from:
        # one per Q stage. A transposed decode that splits its query rows four
        # ways asks for four and re-runs this once __call__ has resolved that.
        self._assign_warp_roles(2)
        self.buffer_align_bytes = 1024
        
        # Scale factor parameters for block-scaled quantization (FP4)
        self.sf_dtype = sf_dtype
        self.sf_vec_size = sf_vec_size
        self.mma_inst_bits_k = 256
        self.mma_inst_tile_k = self.head_dim_padded // (self.mma_inst_bits_k // 4)
        self.sf_groups_per_mma_k = self.mma_inst_bits_k // 4 // self.sf_vec_size

    # An SM holds this many 32-bit registers, handed out in blocks of eight per
    # thread, and the kernel runs one CTA per SM.
    REGISTER_FILE_PER_SM = 65536
    REGISTER_ALLOC_GRANULARITY = 8
    WARPS_PER_WARPGROUP = 4

    def _assign_warp_roles(self, num_softmax_warpgroups: int):
        """Lay out the warp roles for a CTA with this many softmax warpgroups.

        Each softmax warpgroup is a contiguous run of four warps starting on a
        128-thread boundary, which is what makes ``thread_idx % 128`` the kv
        position inside the tile that the transposed softmax indexes by. The
        remaining roles follow: one correction warpgroup, then the mma,
        epilogue, load and idle warps that share the last warpgroup.

        Widening the CTA costs registers, so the softmax warpgroups split
        whatever the correction and service warpgroups leave of the CTA's
        register allocation rather than each keeping the count a narrower CTA
        could afford.
        """
        warps = self.WARPS_PER_WARPGROUP
        threads_per_warpgroup = warps * cute.arch.WARP_SIZE
        self.softmax_warp_ids = tuple(
            tuple(range(g * warps, (g + 1) * warps))
            for g in range(num_softmax_warpgroups)
        )
        self.softmax0_warp_ids = self.softmax_warp_ids[0]  # stage 0
        self.softmax1_warp_ids = self.softmax_warp_ids[1]  # stage 1
        service = num_softmax_warpgroups * warps
        self.correction_warp_ids = tuple(range(service, service + warps))
        self.mma_warp_id = service + warps
        self.epilogue_warp_ids = (service + warps + 1,)
        self.load_warp_ids = (service + warps + 2,)
        self.empty_warp_ids = (service + warps + 3,)
        self.threads_per_cta = cute.arch.WARP_SIZE * (service + 2 * warps)

        if not self.use_tma_KV:
            self.load_warp_ids = self.load_warp_ids + self.empty_warp_ids
            self.empty_warp_ids = ()
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q:  # fallback
            self.epilogue_warp_ids = (service + warps + 1, service + warps + 2)

        # setmaxnreg only redistributes the registers the CTA was given at
        # launch, which is threads_per_cta times the largest per-thread count
        # that fits the register file, rounded down to the allocation
        # granularity. A CTA of 768 threads loses that rounding twice over: the
        # per-thread count drops to 80 and the CTA's pool with it, so the
        # arithmetic has to be done against the pool and not against the whole
        # register file. Over-subscribe it and setmaxnreg.inc waits for
        # registers no one will release.
        regs_at_launch = (
            self.REGISTER_FILE_PER_SM
            // self.threads_per_cta
            // self.REGISTER_ALLOC_GRANULARITY
            * self.REGISTER_ALLOC_GRANULARITY
        )
        softmax_pool = regs_at_launch * self.threads_per_cta - threads_per_warpgroup * (
            self.num_regs_correction + max(self.num_regs_other, self.num_regs_empty)
        )
        affordable = (
            softmax_pool
            // (threads_per_warpgroup * num_softmax_warpgroups)
            // self.REGISTER_ALLOC_GRANULARITY
            * self.REGISTER_ALLOC_GRANULARITY
        )
        self.num_regs_softmax = min(self.num_regs_softmax_tuned, affordable)

    def _softmax_row_group_count(self) -> int:
        """How many ways the transposed softmax splits this tile's query rows.

        A decode carries one Q tile, so only the first softmax warpgroup has a
        stage to run. Spreading the query rows over more groups shortens the
        serial chain every thread runs — its butterfly, its exponentials and
        its packing — and the groups write disjoint rows of P and of the
        scales. Two groups is what any tile with more than one query row
        already split into; going past two is worth it only while every group
        keeps enough rows to amortize its own reduction.

        Plain Python, evaluated while ``__call__`` is traced, so the count and
        everything derived from it are compile-time constants.
        """
        # The BF16 steps share one P tile between the groups, and nothing
        # orders one group's stores against another's clear, so each clears
        # only rows no group writes (see zero_transposed_p_bf16). A second Q
        # stage gives the second warpgroup its own work.
        if self.q_stage != 1:
            return 1
        if self.transposed_query_rows < 2:
            return 1
        return min(
            MAX_SOFTMAX_ROW_GROUPS,
            max(2, self.transposed_query_rows // MIN_ROWS_PER_SOFTMAX_GROUP),
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """
        self.acc_stage = 1
        # The epilogue indexes sO by q tile, so a deeper ring than there are q
        # tiles is memory no one can name. Decode runs a single q tile, where
        # the second buffer was costing 32KB of output, or 64KB once the split
        # path widens it to fp32 partials.
        self.epi_stage = self.q_stage
        # Compute kv_stage from SMEM budget.
        # Blackwell: 228KB per SM, 227KB optin per block.
        # K and V alias when same dtype or when K is smaller (FP4 K in BF16 V).
        smem_budget = 227 * 1024
        align = self.buffer_align_bytes  # 128B struct field alignment
        def align_up(x, a): return (x + a - 1) // a * a
        # Fixed fields (not scaled by kv_stage): mbar, tmem_holding_buf, sScale, sO, sQ, SFQ, SFP
        # mbar_total depends on kv_stage but is small (~40 barriers * 8B = 320B); use upper bound
        smem_mbar = 512  # generous upper bound for mbarrier storage
        smem_tmem = 4  # Int32
        smem_sScale = align_up(max(self.q_stage * self.m_block_size * 2, self.m_block_size * 4) * 4, align)  # Float32
        smem_q_per_stage = self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_o_per_stage = self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        # SFQ/SFP are per q_stage, SF layout cosize: m_block * head_dim / sf_vec_size
        sfq_per_stage = self.m_block_size * self.head_dim_padded // self.sf_vec_size
        sfp_per_stage = self.m_block_size * self.head_dim_v_padded // self.sf_vec_size
        sfq_bytes = (
            sfq_per_stage * self.q_stage * self.query_terms if self.quant_qk else 0
        )
        # When the two overlap, sQ is widened to cover sO rather than sO being
        # dropped, so the budget has to charge the wider of the two. The Q side
        # is every query term's tile and its scale factors, which are dead by
        # the time sO is written, just like the tiles.
        if self.overlap_sO_sQ:
            smem_sO = 0
            smem_sQ = align_up(
                max(
                    smem_q_per_stage * self.q_stage * self.query_terms + sfq_bytes,
                    smem_o_per_stage * self.epi_stage,
                ),
                align,
            )
            smem_sSFQ = 0
        else:
            smem_sO = align_up(smem_o_per_stage * self.epi_stage, align)
            smem_sQ = align_up(
                smem_q_per_stage * self.q_stage * self.query_terms,
                align,
            )
            smem_sSFQ = align_up(sfq_bytes, align)
        smem_sSFP = align_up(sfp_per_stage * self.q_stage, align) if self.quant_pv else 0
        # A transposed P reaches the tensor core from shared memory, and the
        # cross-warp row maximum needs one float per softmax warp per query row,
        # double buffered so a single named barrier per KV block suffices.
        if self.transpose_s:
            smem_sP = align_up(
                self.m_block_size * self.n_block_size * self.v_dtype.width // 8, align
            )
            smem_sRed = align_up(self.softmax_red_slots * 4, align)
        else:
            smem_sP = 0
            smem_sRed = 0
        if self.fused_residual_first_block:
            smem_sK_bf16 = align_up(self.m_block_size * self.head_dim_padded * 2, align)
            smem_sV_bf16 = align_up(self.m_block_size * self.head_dim_v_padded * 2, align)
            smem_sQ_bf16 = align_up(self.m_block_size * self.head_dim_padded * 2, align)
        else:
            smem_sK_bf16 = 0
            smem_sV_bf16 = 0
            smem_sQ_bf16 = 0
        smem_bf16_tiles = smem_sK_bf16 + smem_sV_bf16 + smem_sQ_bf16
        if self.overlap_sO_bf16:
            smem_bf16_tiles = max(smem_bf16_tiles, smem_sO)
            smem_sO = 0
        # On a split of their own the BF16 tiles share their bytes with the
        # K/V stages (see SharedStorage), so they are charged further down.
        smem_fixed = smem_mbar + smem_tmem + smem_sScale + smem_sO + smem_sQ + smem_sSFQ + smem_sSFP + smem_sP + smem_sRed
        if not self.bf16_split:
            smem_fixed += smem_bf16_tiles
        # Per-kv_stage fields: sK (or aliased), sV, SFK, SFV
        smem_k_per_stage = self.m_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_v_per_stage = self.m_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        if self.v_dtype == self.k_dtype or self.k_dtype.width < self.v_dtype.width:
            smem_kv_per_stage = max(smem_k_per_stage, smem_v_per_stage)
        else:
            smem_kv_per_stage = smem_k_per_stage + smem_v_per_stage
        # SF layout cosize per stage: n_block * head_dim / sf_vec_size (MMA-tiled)
        sfk_per_stage = self.n_block_size * self.head_dim_padded // self.sf_vec_size
        sfv_per_stage = self.n_block_size * self.head_dim_v_padded // self.sf_vec_size
        if self.quant_qk:
            smem_kv_per_stage += sfk_per_stage
        if self.quant_pv:
            smem_kv_per_stage += sfv_per_stage
        # Add per-stage padding for swizzle/layout cosize inflation (~128B per staged field)
        num_kv_staged_fields = 2  # sK + sV (or 1 if aliased, but sV still has cosize overhead)
        if self.quant_qk:
            num_kv_staged_fields += 1  # sSFK
        if self.quant_pv:
            num_kv_staged_fields += 1  # sSFV
        smem_kv_per_stage += num_kv_staged_fields * 128
        self.kv_stage = (smem_budget - smem_fixed) // smem_kv_per_stage
        if self.quant_qk and self.quant_pv:
            # Depth buys latency hiding only until the mainloop stops waiting on
            # KV. Past that the extra buffers cost cosize and issue slots: on the
            # non-split path a depth sweep is flat from 8 to 10 and 1% worse at
            # 12 and beyond, so spending the whole SMEM budget (which lands at
            # 14) is a loss. The split path carries an FP32 partial-output buffer
            # and cannot fit more than 9 stages anyway.
            KV_STAGE_FP4_SPLITK_CAP = 8
            KV_STAGE_FP4_CAP = 10
            cap = KV_STAGE_FP4_SPLITK_CAP if self.is_split_kv else KV_STAGE_FP4_CAP
            self.kv_stage = min(self.kv_stage, cap)
        if self.bf16_split:
            # The tiles overlay the stages' K/V data, so only what they need
            # past it adds to the budget.
            smem_kv_data_per_stage = max(smem_k_per_stage, smem_v_per_stage)

            def smem_total(stages):
                overhang = smem_bf16_tiles - stages * smem_kv_data_per_stage
                return smem_fixed + stages * smem_kv_per_stage + max(0, overhang)

            while smem_total(self.kv_stage) > smem_budget:
                self.kv_stage -= 1
        assert self.kv_stage >= 2, (
            f"kv_stage={self.kv_stage} < 2: FP4 mainloop requires K/V double-buffer"
        )
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        # self.uneven_kv_smem = (
            # self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        # )
        self.uneven_kv_smem = False
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ,  # cute.Tensor or cute.Pointer (b, s_q, h, d)
        mK,  # cute.Tensor or cute.Pointer (b_k, s_k, h_k, d)
        mV,  # cute.Tensor or cute.Pointer (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        # For pointer-based Q/K: separate shapes to handle cross-attention (seqlen_q != seqlen_k)
        q_ptr_shape: tuple = (),
        k_ptr_shape: tuple = (),
        # For pointer-based V (FP4 K-major): full headdim shape (b, s, h, d)
        v_ptr_shape: tuple = (),
        # Distance between consecutive K/V pages, in FP4 elements for the packed
        # data and in bytes for the scale factors. None means the pages are
        # densely packed, which is what the shapes alone imply. A larger stride
        # lets one page be a region of a wider cache page that also holds the
        # scale factors and the other of K/V.
        k_page_stride=None,
        v_page_stride=None,
        k_sf_page_stride=None,
        v_sf_page_stride=None,
        compute_sp1: cutlass.Constexpr[bool] = False,
        mResidualQ: Optional[cute.Tensor] = None,
        mResidualK: Optional[cute.Tensor] = None,
        mResidualV: Optional[cute.Tensor] = None,
        mResidualSeqUsedK: Optional[cute.Tensor] = None,
        mResidualKCache: Optional[cute.Tensor] = None,
        mResidualVCache: Optional[cute.Tensor] = None,
        mResidualBlockIds: Optional[cute.Tensor] = None,
        mSinkSeqUsedK: Optional[cute.Tensor] = None,
        mOutIndices: Optional[cute.Tensor] = None,
        mQSecond=None,
        mSFQSecond: Optional[cute.Tensor] = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        For FP4, mQ/mK can be cute.Pointer with q/k_ptr_shape providing (b, s, h, d).
        The kernel builds tensors from the pointer using make_ordered_layout.
        """
        # Build Q/K tensors from pointer/tensor + shape
        # For pointers: mQ is a Pointer, .iterator not needed
        # For tensors: mQ is a Tensor, use .iterator to extract pointer
        q_iter = mQ.iterator if hasattr(mQ, 'iterator') else mQ
        q_second_iter = (
            mQSecond.iterator if hasattr(mQSecond, "iterator") else mQSecond
        )
        k_iter = mK.iterator if hasattr(mK, 'iterator') else mK
        mQ = cute.make_tensor(q_iter, cute.make_ordered_layout(
            q_ptr_shape, order=tuple(range(len(q_ptr_shape) - 1, -1, -1))
        ))
        mQSecond = cute.make_tensor(
            q_second_iter,
            cute.make_ordered_layout(
                q_ptr_shape,
                order=tuple(range(len(q_ptr_shape) - 1, -1, -1)),
            ),
        )
        if const_expr(k_page_stride is None):
            mK = cute.make_tensor(k_iter, cute.make_ordered_layout(
                k_ptr_shape, order=tuple(range(len(k_ptr_shape) - 1, -1, -1))
            ))
        else:
            # Same row-major (b, s, h, d) layout the ordered form produces, with
            # the page stride supplied instead of derived. The innermost stride
            # stays a literal 1 so the major mode is still statically known.
            _, _, k_h, k_d = k_ptr_shape
            mK = cute.make_tensor(k_iter, cute.make_layout(
                k_ptr_shape, stride=(k_page_stride, k_h * k_d, k_d, 1)
            ))
        # FP4 K-major V: build from pointer with explicit (b, s, h, d) shape and
        # K-major strides (S*H*D, 1, S, S*H). The host transposes V's underlying
        # buffer so that seqlen has stride 1 in the FP4 byte buffer.
        if const_expr(len(v_ptr_shape) > 0):
            v_iter = mV.iterator if hasattr(mV, 'iterator') else mV
            # K-major V: nvfp4_quantize on `v.permute(0,2,3,1).reshape(b*h*d, s)`
            # produces an FP4 byte buffer of physical shape (b, h, d, s/2) row-major,
            # where each int8 byte holds two seqlen-adjacent FP4 in the high/low
            # nibble. Logically the V tensor has FP4 shape (b, s, h, d) with element
            # strides (h*d*s, 1, h*d, d) — order=(3, 0, 2, 1):
            #   s (dim 1) order 0 → stride 1
            #   d (dim 3) order 1 → stride s
            #   h (dim 2) order 2 → stride s*d
            #   b (dim 0) order 3 → stride s*d*h
            # Int64 shape so make_ordered_layout produces Int64 strides
            # (tma_partition for SFV requires Int64).
            from cutlass import Int64
            v_b, v_s, v_h, v_d = v_ptr_shape
            v_shape = (Int64(v_b), Int64(v_s), Int64(v_h), Int64(v_d))
            if const_expr(v_page_stride is None):
                mV = cute.make_tensor(v_iter, cute.make_ordered_layout(
                    v_shape, order=(3, 0, 2, 1),
                ))
            else:
                # The order=(3, 0, 2, 1) strides written out, with the page
                # stride supplied. Seqlen keeps a literal stride of 1 so V stays
                # statically K-major.
                mV = cute.make_tensor(v_iter, cute.make_layout(
                    v_shape,
                    stride=(
                        Int64(v_page_stride),
                        1,
                        Int64(v_s) * Int64(v_d),
                        Int64(v_s),
                    ),
                ))
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        self.compute_sp1 = const_expr(compute_sp1)
        if const_expr(mSFQ is None):
            assert self.q_dtype.width >= 8
            assert const_expr(mSFK is None), "Must provide both QK sfs or None"
        self.quant_qk = const_expr(mSFQ is not None)
        self.quant_pv = const_expr(mSFV is not None)
        assert not (not self.quant_qk and self.quant_pv)

        # The transposed layout only pays off where the query axis is short and
        # statically known, and it reroutes P through shared memory, so it is
        # confined to the pure FP4 decode shape. Everything else keeps the
        # untransposed path. The query count is padded to a power of two below,
        # because it becomes the repetition count of a single tcgen05 load.
        # The conditions are combined with all() over a list rather than a chain
        # of `and`. Preprocessing rewrites each short-circuit operator into a
        # nested conditional that carries a copy of the continuation, so an
        # n-term chain in a traced function grows the transformed AST by 2**n;
        # sixteen terms here took this function from 14k nodes to over 4M and
        # its preprocessing from 0.01s to more than ten minutes. Every term is a
        # plain attribute read, so evaluating all of them eagerly is free.
        self.transpose_s = const_expr(
            all(
                [
                    self.transpose_s_requested,
                    self.quant_qk,
                    self.quant_pv,
                    self.pack_gqa,
                    self.seqlen_q_static_one,
                    not self.compute_sp1,
                    not self.is_causal,
                    not self.is_local,
                    # self.use_block_sparsity is not derived until much later in
                    # this function, so read its source argument directly.
                    blocksparse_tensors is None,
                    self.score_mod is None,
                    self.qhead_per_kvhead <= 128,
                ]
            )
        )
        assert not self.p_scale_two_level or self.transpose_s, "two_level P requires transposed decode"
        assert not self.p_scale_rescale_log2 or self.transpose_s, (
            "rescaled P requires transposed decode"
        )
        assert self.query_terms == 1 or self.transpose_s, (
            "two-term query decode requires the transposed score path"
        )
        assert not self.is_mxfp4 or self.transpose_s, (
            "MXFP4 decode requires the transposed score path"
        )
        if const_expr(self.transpose_s):
            # Replication addresses the same padding by repeating rows, which
            # the transposed layout makes pointless.
            self.q_replicate = 1
            # Transposing swaps the two leading modes of the QK tiler. Requiring
            # them equal keeps that tuple, and every partition derived from it,
            # unchanged, so only the operand roles move.
            assert self.m_block_size == self.n_block_size, (
                "transpose_s requires a square QK tile"
            )
            # PackGQA with a statically-one query length fills qhead_per_kvhead
            # rows of the M tile; transposed, they are the live columns of S.
            # The count becomes the repetition of a single tcgen05 load, which
            # must be a power of two, so it is rounded up. The extra columns
            # carry the zero rows TMA fills past the flat head axis: their
            # scores are finite and their rows of O are never stored.
            self.transposed_query_rows = 1 << (self.qhead_per_kvhead - 1).bit_length()
            # Two independently scaled terms occupy disjoint query rows of
            # one narrow QK instruction. The logical softmax row count stays
            # unchanged; its FP32 addition joins the two score columns.
            self.stacked_query_terms = (
                self.query_terms == 2
                and self.q_stage == 1
                and self.qhead_per_kvhead in (4, 8, 16)
            )
            # PV accumulates O^T: V is the A operand over the head dimension and
            # P the B operand over the live query rows, rounded up to the eight
            # an MMA needs. Padding the rows to the 128 of an A operand instead
            # spends sixteen times the tensor-core energy at a ratio of eight,
            # which under the power cap costs clock. Each correction thread then
            # holds one head-dimension lane of every live row, so the row count
            # is bounded by the correction warps' registers; the two-level P
            # scale accumulates rows of O per thread and keeps the wide tile.
            self.pv_rows = max(8, self.transposed_query_rows)
            self.narrow_pv = all(
                [
                    self.transposed_query_rows <= 16,
                    self.q_stage == 1,
                    not self.p_scale_two_level,
                    learnable_sink is None,
                    self.head_dim_v_padded == self.m_block_size,
                    not self.use_correction_warps_for_epi,
                ]
            )
            self.softmax_row_groups = self._softmax_row_group_count()
            # transposed_query_rows is a power of two, so the group count
            # always divides it evenly.
            self.softmax_rows_per_group = (
                self.transposed_query_rows // self.softmax_row_groups
            )
            # The row maximum reduce-scatters a thread's rows across lanes when
            # it holds more than one and each lane's row still has the lanes of
            # a whole scale group to itself. The grid-search P scale reduces an
            # error per row over the group and keeps the row-per-thread path.
            self.softmax_scatters_rows = (
                2 <= self.softmax_rows_per_group <= self.sf_vec_size
                and not self.p_scale_grid_search
            )
            # One softmax warpgroup per row group, except that the CTA never
            # narrows below the two warpgroups the Q stages need.
            self._assign_warp_roles(max(2, self.softmax_row_groups))
            # Two buffers per warp per row, so a single named barrier per KV
            # block suffices for the row reductions. Splitting the rows across
            # groups moves slots between the groups without changing the total.
            self.softmax_red_slots = (
                2 * len(self.softmax0_warp_ids) * self.transposed_query_rows
            )
            # P and its scale factors normally alias the tensor memory S was
            # read out of, which is why the QK of the next block has to wait
            # behind this block's PV. Transposed, P lives in shared memory, so
            # only the scale factors still alias, and a decode leaves 128
            # tensor-memory columns free above O to move them into. Once they
            # are out of the way the only thing the next QK waits on is the
            # softmax having drained S, which it signals as soon as the load
            # completes rather than at the end of the step.
            self.tmem_sfpv_offset = self.tmem_total
        if const_expr(self.fused_sink_block):
            assert self.transpose_s, (
                "the two-island BF16 path requires transposed scores"
            )
        # Each CTA picks its role once, from its split, so it must own exactly
        # one tile for its whole life. Its BF16 tiles share the bytes of sK,
        # which therefore has to hold V as well.
        if const_expr(self.bf16_split):
            assert (
                self.transpose_s
                and not self.is_persistent
                and self.v_dtype == self.k_dtype
            ), (
                "a split of its own for the BF16 blocks needs transposed "
                "scores, one tile per CTA, and K and V of one dtype"
            )

        # Assume all strides are divisible by 128 bits except the last stride
        def _assume_strides(t):
            divby = 128 // t.element_type.width
            return tuple(
                s if isinstance(s, int) else cute.assume(s, divby=divby)
                for s in t.stride[:-1]
            ) + (t.stride[-1],)
        mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=_assume_strides(t)))
            for t in (mV, mO)
        ]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)) # (s_q, d, h, b)
        mQSecond = cute.make_tensor(
            mQSecond.iterator,
            cute.select(mQSecond.layout, mode=Q_layout_transpose),
        )
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        # For FP4 block-scaled MMA, B (V) must be K-major (K=seqlen contiguous).
        # Skip V transpose when V is already K-major (headdim contiguous = N contiguous after transpose).
        # Without transpose: mV = (s, d, h, b) → mode 0=s(K), mode 1=d(N) with d contiguous → K-major: NO!
        # With transpose:    mV = (d, s, h, b) → mode 0=d(N), mode 1=s(K) with d contiguous → MN-major
        # For FP4 we need K-major: mode 1 (K=seqlen) contiguous. Need V physically transposed on host.
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        # FP4 block-scaled MMA requires K-major B; standard MMA uses MN-major
        if const_expr(mSFV is not None):
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mV must be K-major for FP4 block-scaled MMA")
        else:
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
                raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype == cutlass.Int8):
            assert self.q_dtype == self.k_dtype
            self.q_dtype = self.k_dtype = cutlass.Float4E2M1FN
            if const_expr(mSFV is not None):
                self.v_dtype = cutlass.Float4E2M1FN

        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(mSFV is not None and self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype} (V quantization requires matching dtype)")
        self._setup_attributes()
        self.use_tma_O = (
            self.arch >= 90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not self.seqlen_q_static_one
        )
        # This can be tuned
        self.e2e_freq = 16
        if const_expr(
            self.head_dim_padded > 64 and not self.is_causal and not self.is_local and self.pack_gqa
        ):
            self.e2e_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10

        use_2cta_instrs = self.mma_tiler_qk[0] == 256
        assert use_2cta_instrs == False, "Two-CTA instructions not supported yet"
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # the intermediate tensor p is from tmem & mK-major. Transposed, softmax
        # holds P in registers rather than tensor memory, and OperandSource only
        # selects the source of A, so P has to travel through shared memory.
        p_source = (
            tcgen05.OperandSource.SMEM
            if const_expr(self.transpose_s)
            else tcgen05.OperandSource.TMEM
        )
        p_major_mode = tcgen05.OperandMajorMode.K
        
        # Transposing S makes K the A operand and Q the B operand; both are
        # K-major over the head dimension either way, so only the roles move.
        qk_a_major = self.k_major_mode if const_expr(self.transpose_s) else self.q_major_mode
        qk_b_major = self.q_major_mode if const_expr(self.transpose_s) else self.k_major_mode

        # Use block-scaled MMA for PV only if V is being quantized (mSFV is provided)
        if const_expr(self.quant_qk):
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                qk_a_major,
                qk_b_major,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                qk_a_major,
                qk_b_major,
                self.qk_acc_dtype,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )

        if const_expr(self.quant_pv):
            tiled_mma_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )
        else:
            tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.pv_acc_dtype,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]
        
        # Which builder a tensor uses follows its operand role, not its name.
        make_q_smem_layout = (
            sm100_utils_basic.make_smem_layout_b
            if const_expr(self.transpose_s)
            else sm100_utils_basic.make_smem_layout_a
        )
        make_k_smem_layout = (
            sm100_utils_basic.make_smem_layout_a
            if const_expr(self.transpose_s)
            else sm100_utils_basic.make_smem_layout_b
        )
        make_sfq_smem_layout = (
            make_smem_layout_sfb if const_expr(self.transpose_s) else make_smem_layout_sfa
        )
        make_sfk_smem_layout = (
            make_smem_layout_sfa if const_expr(self.transpose_s) else make_smem_layout_sfb
        )

        # ((Atom_Inst_M, Atom_Inst_K), MMA_M, MMA_K, STAGE)
        sQ_layout = make_q_smem_layout(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = make_k_smem_layout(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )

        sfv_smem_layout_staged = None
        sfp_smem_layout_staged = None
        # # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
        sfq_smem_layout_staged = make_sfq_smem_layout(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.sf_vec_size,
            self.q_stage,
            mma_tile_inst_k=self.mma_inst_tile_k,
        )
        sfk_smem_layout_staged = make_sfk_smem_layout(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.sf_vec_size,
            self.kv_stage,
            mma_tile_inst_k=self.mma_inst_tile_k,
        )
        # Create P scale factor layout for P*V operation (P is the A matrix)
        if const_expr(self.quant_pv):
            sfp_smem_layout_staged = make_smem_layout_sfa(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                self.q_stage,
                mma_tile_inst_k=self.mma_inst_tile_k,
            )

            sfv_smem_layout_staged = make_smem_layout_sfb(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                self.kv_stage,
                mma_tile_inst_k=self.mma_inst_tile_k,
            )
        
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )


        mQ_shape_unpacked = mQ.shape

        def packed_rows(t, head_mode: int, heads: int):
            """Shape and stride of the packed row mode of a (sq, ..., h, ...) tensor.

            With a single query position the (head, seqlen) mode collapses to
            the head axis alone. A flat axis of ``heads`` rows tiles by the
            M tile for any ratio, with TMA zero-filling the rows past it; the
            hierarchical mode only tiles when the tile is a whole number of
            query groups.
            """
            if const_expr(self.seqlen_q_static_one):
                return heads, t.stride[head_mode]
            return (heads, t.shape[0]), (t.stride[head_mode], t.stride[0])

        if const_expr(self.pack_gqa):
            q_rows_shape, q_rows_stride = packed_rows(mQ, 2, self.qhead_per_kvhead)
            shape_Q_packed = (
                q_rows_shape, # qhead_per_kvhead, or (qhead_per_kvhead, sq)
                mQ.shape[1], # d
                mK.shape[2], # h_k
                *mQ.shape[3:], # b
            )
            stride_Q_packed = (
                q_rows_stride,
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            mQSecond = cute.make_tensor(
                mQSecond.iterator,
                cute.make_layout(shape_Q_packed, stride=stride_Q_packed),
            )
            o_rows_shape, o_rows_stride = packed_rows(mO, 2, self.qhead_per_kvhead_O)
            shape_O_packed = (
                o_rows_shape,
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                o_rows_stride,
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead_O,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                lse_rows_shape, lse_rows_stride = packed_rows(
                    mLSE, 1, self.qhead_per_kvhead
                )
                shape_LSE_packed = (
                    lse_rows_shape,
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    lse_rows_stride,
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        # Add scale factor copy bytes to Q/K/V since they use the same barrier
        if const_expr(self.quant_qk):
            self.tma_copy_bytes["Q"] += cute.size_in_bytes(mSFQ.element_type, cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]))
            self.tma_copy_bytes["K"] += cute.size_in_bytes(mSFK.element_type, cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]))
            self.tma_copy_bytes["Q"] *= self.query_terms
        if const_expr(self.quant_pv):
            self.tma_copy_bytes["V"] += cute.size_in_bytes(mSFV.element_type, cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]))

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        mQ_shape = mQ.shape
        # The FP4 Q tile is the B operand once S is transposed.
        make_q_tma_atom = (
            cute.nvgpu.make_tiled_tma_atom_B
            if const_expr(self.transpose_s)
            else cute.nvgpu.make_tiled_tma_atom_A
        )
        tma_atom_Q, mQ = make_q_tma_atom(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_Q_second, mQ_second_tma = make_q_tma_atom(
            tma_load_op,
            mQSecond,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_K = None
        tma_atom_V = None
        mK_shape = mK.shape
        mV_shape = mV.shape
        if const_expr(self.use_tma_KV):
            # TMA load for K — the A operand once S is transposed.
            make_k_tma_atom = (
                cute.nvgpu.make_tiled_tma_atom_A
                if const_expr(self.transpose_s)
                else cute.nvgpu.make_tiled_tma_atom_B
            )
            tma_atom_K, mK = make_k_tma_atom(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )

        tiled_mma_qk_bf16 = None
        tiled_mma_pv_bf16 = None
        sQ_bf16_layout = None
        sK_bf16_layout = None
        sV_bf16_layout = None
        tP_bf16_layout = None
        tma_atom_Q_bf16 = None
        tma_atom_K_bf16 = None
        tma_atom_V_bf16 = None
        mResidualQ_t = None
        mResidualK_t = None
        mResidualV_t = None
        if const_expr(self.fused_residual_first_block):
            if const_expr(self.residual_source == "contiguous"):
                mResK_src = mResidualK
                mResV_src = mResidualV
            else:
                # paged_bf16 — caller must supply both cache tensors.
                mResK_src = mResidualKCache
                mResV_src = mResidualVCache
            # mResK_src has logical shape (B-or-num_blocks, S=128, H_kv, D) bf16.
            # Apply the same transpose pattern as the BF16 reference kernel:
            # (b, s, h, d) -> (s, d, h, b) via mode=[1, 3, 2, 0].
            mResK = cute.make_tensor(
                mResK_src.iterator,
                cute.select(mResK_src.layout, mode=[1, 3, 2, 0]),
            )
            mResV = cute.make_tensor(
                mResV_src.iterator,
                cute.select(mResV_src.layout, mode=[1, 3, 2, 0]),
            )
            # BF16 V is MN-major (not transposed): (s, d, h, b) -> (d, s, h, b)
            mResV = cute.make_tensor(
                mResV.iterator,
                cute.select(mResV.layout, mode=[1, 0, 2, 3]),
            )
            # BF16 GEMMs use a non-blockscaled tiled MMA (standard SM100 BF16 tcgen05).
            # We re-use the existing q_major_mode/k_major_mode/v_major_mode
            # since the residual K/V are constructed with the same logical layout
            # transposes as the BF16 reference kernel.
            res_k_dtype = mResK.element_type   # cutlass.BFloat16
            res_v_dtype = mResV.element_type   # cutlass.BFloat16
            res_k_major = cutlass.utils.LayoutEnum.from_tensor(mResK).mma_major_mode()
            res_v_major = cutlass.utils.LayoutEnum.from_tensor(mResV).mma_major_mode()
            # The residual follows the FP4 blocks' orientation so that both write
            # the same tensor-memory tile the same way round and one softmax
            # reads them. Transposed, K is the A operand and Q the B operand.
            tiled_mma_qk_bf16 = sm100_utils_basic.make_trivial_tiled_mma(
                res_k_dtype,
                res_k_major if const_expr(self.transpose_s) else self.q_major_mode,
                self.q_major_mode if const_expr(self.transpose_s) else res_k_major,
                self.qk_acc_dtype,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )
            tiled_mma_pv_bf16 = sm100_utils_basic.make_trivial_tiled_mma(
                res_v_dtype,
                p_major_mode,
                res_v_major,
                self.pv_acc_dtype,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )
            # SMEM layouts for BF16 residual K/V (single stage — block 0 only).
            make_k_bf16_smem_layout = (
                sm100_utils_basic.make_smem_layout_a
                if const_expr(self.transpose_s)
                else sm100_utils_basic.make_smem_layout_b
            )
            sK_bf16_layout = make_k_bf16_smem_layout(
                tiled_mma_qk_bf16,
                self.mma_tiler_qk,
                res_k_dtype,
                1,  # 1 stage suffices: residual block 0 is loaded once per CTA-tile.
            )
            sV_bf16_layout = sm100_utils_basic.make_smem_layout_b(
                tiled_mma_pv_bf16,
                self.mma_tiler_pv,
                res_v_dtype,
                1,
            )
            tP_bf16_layout = sm100_utils_basic.make_smem_layout_a(
                tiled_mma_pv_bf16,
                self.mma_tiler_pv,
                res_v_dtype,  # BFloat16
                1,
            )
            # TMA atoms for BF16 residual K/V.
            make_k_bf16_tma_atom = (
                cute.nvgpu.make_tiled_tma_atom_A
                if const_expr(self.transpose_s)
                else cute.nvgpu.make_tiled_tma_atom_B
            )
            tma_atom_K_bf16, mResidualK_t = make_k_bf16_tma_atom(
                tma_load_op,
                mResK,
                cute.select(sK_bf16_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk_bf16,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_V_bf16, mResidualV_t = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mResV,
                cute.select(sV_bf16_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv_bf16,
                self.cluster_layout_vmnk.shape,
            )
            mResQ = cute.make_tensor(
                mResidualQ.iterator,
                cute.select(mResidualQ.layout, mode=[1, 3, 2, 0]),
            )
            if const_expr(self.pack_gqa):
                res_rows_shape, res_rows_stride = packed_rows(
                    mResQ, 2, self.qhead_per_kvhead
                )
                shape_ResQ_packed = (
                    res_rows_shape,
                    mResQ.shape[1],
                    mK.shape[2],
                    *mResQ.shape[3:],
                )
                stride_ResQ_packed = (
                    res_rows_stride,
                    mResQ.stride[1],
                    mResQ.stride[2] * self.qhead_per_kvhead,
                    *mResQ.stride[3:],
                )
                mResQ = cute.make_tensor(
                    mResQ.iterator,
                    cute.make_layout(shape_ResQ_packed, stride=stride_ResQ_packed),
                )
            res_q_dtype = mResQ.element_type   # cutlass.BFloat16
            make_q_bf16_smem_layout = (
                sm100_utils_basic.make_smem_layout_b
                if const_expr(self.transpose_s)
                else sm100_utils_basic.make_smem_layout_a
            )
            sQ_bf16_layout = make_q_bf16_smem_layout(
                tiled_mma_qk_bf16,
                self.mma_tiler_qk,
                res_q_dtype,
                1,
            )
            make_q_bf16_tma_atom = (
                cute.nvgpu.make_tiled_tma_atom_B
                if const_expr(self.transpose_s)
                else cute.nvgpu.make_tiled_tma_atom_A
            )
            tma_atom_Q_bf16, mResidualQ_t = make_q_bf16_tma_atom(
                tma_load_op,
                mResQ,
                cute.select(sQ_bf16_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk_bf16,
                self.cluster_layout_vmnk.shape,
            )

        # TMA load for scale factors
        tma_atom_sfq = None
        tma_tensor_sfq = None
        tma_atom_sfk = None
        tma_tensor_sfk = None
        tma_atom_sfv = None
        tma_tensor_sfv = None
        if const_expr(self.quant_qk):
            sfq_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, mQ_shape_unpacked, (2, 1, 3, 4))
            sfq_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )
            mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)
            mSFQSecond = cute.make_tensor(mSFQSecond.iterator, sfq_layout)
            if const_expr(self.pack_gqa):
                shape_SFQ_packed = (
                    mSFQ.shape[0],                       # ((32,4), rest_m=1) unchanged
                    mSFQ.shape[1],                       # K modes unchanged
                    (mSFQ.shape[2][0], mK_shape[2]),     # (1, h_kv=8) — use pre-TMA shape
                    mSFQ.shape[3],                       # (1, b=1)
                )
                stride_SFQ_packed = (
                    mSFQ.stride[0],                      # ((16,4), rest_m_stride) unchanged
                    mSFQ.stride[1],                      # K strides unchanged
                    mSFQ.stride[2],                      # (0, h_sf_stride=1024) unchanged
                    mSFQ.stride[3],                      # batch unchanged
                )
                mSFQ = cute.make_tensor(
                    mSFQ.iterator,
                    cute.make_layout(shape_SFQ_packed, stride=stride_SFQ_packed),
                )
                mSFQSecond = cute.make_tensor(
                    mSFQSecond.iterator,
                    cute.make_layout(
                        shape_SFQ_packed, stride=stride_SFQ_packed
                    ),
                )
            tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_A(
                sfq_op,
                mSFQ,
                cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            (
                tma_atom_sfq_second,
                tma_tensor_sfq_second,
            ) = cute.nvgpu.make_tiled_tma_atom_A(
                sfq_op,
                mSFQSecond,
                cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        if const_expr(self.quant_qk):
            # Setup TMA load for SFK (scale factor for K, like SFB).
            sfk_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )
            sfk_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, mK_shape, (2, 1, 3, 4))
            sfk_layout = _sf_layout_with_page_stride(sfk_layout, k_sf_page_stride)
            mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)

            # For SFB, compute mma_inst_shape_mnk_sfb: (M // (2 if use_2cta_instrs else 1), round_up(N, 128), K)
            mma_inst_shape_mnk_qk = (
                self.mma_tiler_qk[0],
                self.mma_tiler_qk[1],
                self.mma_inst_bits_k // self.k_dtype.width,
            )

            mma_inst_shape_mnk_sfb_qk = (
                mma_inst_shape_mnk_qk[0] // (2 if use_2cta_instrs else 1),
                cute.round_up(mma_inst_shape_mnk_qk[1], 128),
                mma_inst_shape_mnk_qk[2],
            )
        
            mma_tiler_sfb_qk = (
                mma_inst_shape_mnk_sfb_qk[0],
                mma_inst_shape_mnk_sfb_qk[1],
                mma_inst_shape_mnk_sfb_qk[2] * self.mma_inst_tile_k,
            )
            # For SFB, we need a separate tiled_mma_sfb with CtaGroup.ONE
            tiled_mma_sfb_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.k_dtype,
                self.k_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                mma_inst_shape_mnk_sfb_qk[:2],
            )
            cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_qk.thr_id.shape,),
            )
            tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.make_tiled_tma_atom_B(
                sfk_op,
                mSFK,
                cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]),
                mma_tiler_sfb_qk,
                tiled_mma_sfb_qk,
                cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
    
        if const_expr(self.quant_pv):
            # Setup TMA load for SFV (scale factor for V, like SFB)
            sfv_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_pv.thr_id
            )
            # Setup scale factor tensor layout
            sfv_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, mV_shape, (2, 1, 3, 4))
            sfv_layout = _sf_layout_with_page_stride(sfv_layout, v_sf_page_stride)
            mSFV = cute.make_tensor(mSFV.iterator, sfv_layout)
            # For SFB, compute mma_inst_shape_mnk_sfb: (M // (2 if use_2cta_instrs else 1), round_up(N, 128), K)
            mma_inst_shape_mnk_pv = ( # the same processed by one tcgen05.mma instruction
                self.mma_tiler_pv[0],
                self.mma_tiler_pv[1],
                self.mma_inst_bits_k // self.v_dtype.width,
            )
            use_2cta_instrs = self.mma_tiler_pv[0] == 256
            mma_inst_shape_mnk_sfb_pv = (
                mma_inst_shape_mnk_pv[0] // (2 if use_2cta_instrs else 1),
                cute.round_up(mma_inst_shape_mnk_pv[1], 128),
                mma_inst_shape_mnk_pv[2],
            )
            mma_tiler_sfb_pv = (
                mma_inst_shape_mnk_sfb_pv[0],
                mma_inst_shape_mnk_sfb_pv[1],
                mma_inst_shape_mnk_sfb_pv[2] * self.mma_inst_tile_k,
            )
            # For SFB, we need a separate tiled_mma_sfb with CtaGroup.ONE
            tiled_mma_sfb_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                mma_inst_shape_mnk_sfb_pv[:2],
                p_source,
            )
            cluster_layout_sfb_pv_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_pv.thr_id.shape,),
            )
            tma_atom_sfv, tma_tensor_sfv = cute.nvgpu.make_tiled_tma_atom_B(
                sfv_op,
                mSFV,
                cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]),
                mma_tiler_sfb_pv,
                tiled_mma_sfb_pv,
                cluster_layout_sfb_pv_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            # For sub-byte dtypes (FP4), width=4 rounds to 0 under integer division;
            # clamp so size_one_head * element_size in the scheduler's L2 swizzle calc
            # doesn't become 0 and divide by zero in the else branch of the ifexp.
            element_size=max(self.k_dtype.width // 8, 1),
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + self.q_stage
        self.mbar_O_full_offset = self.mbar_S_full_offset + self.q_stage
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + self.q_stage
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + self.q_stage
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + self.epi_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.epi_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + self.q_stage
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        # QK and PV SF tmem load wait for softmax t2r store
        self.mbar_sfqk_load_offset = self.mbar_P_full_2_offset + self.q_stage
        # The PV GEMM has read the shared-memory P and its scale factors.
        self.mbar_P_consumed_offset = self.mbar_sfqk_load_offset + self.q_stage
        self.mbar_q_fp4_ready_offset = self.mbar_P_consumed_offset + self.q_stage
        self.mbar_residual_kv_full_offset = self.mbar_q_fp4_ready_offset + self.q_stage
        # `self.fused_residual_first_block` is a constant (set in __init__) so
        # the branch is selected at trace time; no const_expr wrapper needed
        # since these are plain Python int assignments.
        _residual_kv_slots = 3 if self.fused_residual_first_block else 0
        self.mbar_residual_kv_empty_offset = self.mbar_residual_kv_full_offset + _residual_kv_slots
        self.mbar_bf16_S_full_offset = self.mbar_residual_kv_empty_offset + _residual_kv_slots
        _bf16_phase_slots = 3 if self.fused_residual_first_block else 0
        self.mbar_bf16_P_full_offset = self.mbar_bf16_S_full_offset + (1 if self.fused_residual_first_block else 0)
        self.mbar_bf16_P_full_2_offset = self.mbar_bf16_P_full_offset + (1 if self.fused_residual_first_block else 0)
        self.mbar_bf16_PV_done_offset = self.mbar_bf16_P_full_2_offset + (1 if self.fused_residual_first_block else 0)
        self.mbar_total = self.mbar_bf16_PV_done_offset + (1 if self.fused_sink_block else 0)
        # self.mbar_total = self.mbar_P_full_2_offset + self.q_stage
        self.mbar_p_split = lambda k: (k // 4 * 3 if cutlass.const_expr(self.v_dtype.width >= 8) else k // 2 )
        sO_size = (
            cute.cosize(sO_layout)
            if const_expr(not (self.overlap_sO_sQ or self.overlap_sO_bf16))
            else 1
        )
        # The query terms and their scale factors live in sQ only while QK
        # runs and sO only after the last MMA, so an overlapped region holds
        # the larger of the two, not one O tile per term. The scale factors
        # follow the Q tiles there, which leaves sSFQ a placeholder.
        q_terms_size = self.query_terms * cute.cosize(sQ_layout)
        sfq_terms_size = (
            self.query_terms * cute.cosize(sfq_smem_layout_staged)
            if const_expr(self.quant_qk)
            else 0
        )
        sQ_size = (
            cutlass.max(
                q_terms_size + sfq_terms_size * 8 // self.q_dtype.width,
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
            if const_expr(self.overlap_sO_sQ)
            else q_terms_size
        )

        # Calculate scale factor shared memory sizes
        # Use size 1 as minimum to avoid alignment issues when size is 0
        sfq_smem_size = (
            sfq_terms_size
            if const_expr(self.quant_qk and not self.overlap_sO_sQ)
            else 1
        )
        sfk_smem_size = cute.cosize(sfk_smem_layout_staged) if const_expr(self.quant_qk) else 1
        sfp_smem_size = cute.cosize(sfp_smem_layout_staged) if const_expr(self.quant_pv) else 1
        sfv_smem_size = cute.cosize(sfv_smem_layout_staged) if const_expr(self.quant_pv) else 1

        # P leaves tensor memory under the transposed layout, so the same tile
        # the MMA reads as its A operand needs real storage. sSoftmaxRed carries
        # the per-warp partial row maxima between the four softmax warps.
        sP_smem_size = cute.cosize(tP_layout) if const_expr(self.transpose_s) else 1
        softmax_red_size = (
            self.softmax_red_slots if const_expr(self.transpose_s) else 1
        )

        sK_bf16_smem_size = (
            cute.cosize(sK_bf16_layout) if const_expr(self.fused_residual_first_block) else 1
        )
        sV_bf16_smem_size = (
            cute.cosize(sV_bf16_layout) if const_expr(self.fused_residual_first_block) else 1
        )
        # Transposed, the residual's P also reaches the tensor core from shared
        # memory. It is the same 128x128 of bf16 as the residual Q tile and the
        # QK GEMM, whose completion softmax waits on before writing P, is
        # exactly when that tile dies, so the two share the bytes.
        sQ_bf16_smem_size = (
            max(cute.cosize(sQ_bf16_layout), cute.cosize(tP_bf16_layout))
            if const_expr(self.fused_residual_first_block)
            else 1
        )

        @cute.struct
        class BF16Tiles:
            sK_bf16: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, sK_bf16_smem_size],
                self.buffer_align_bytes,
            ]
            sV_bf16: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, sV_bf16_smem_size],
                self.buffer_align_bytes,
            ]
            sQ_bf16: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, sQ_bf16_smem_size],
                self.buffer_align_bytes,
            ]

        # K reuses V's buffer when K is smaller (FP4 K in BF16 V), or same dtype
        k_aliases_v = self.k_dtype.width < self.v_dtype.width
        kv_stages_storage = cute.struct.Align[
            cute.struct.MemRange[self.k_dtype, 1] if const_expr(k_aliases_v) else cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
            1 if const_expr(k_aliases_v) else self.buffer_align_bytes,
        ]

        # A CTA on the split of the BF16 blocks holds no FP4 K/V stage and the
        # others hold no BF16 tile, so with that split the two share one
        # region in place of sK.
        @cute.union
        class KVStagesOrBF16Tiles:
            stages: kv_stages_storage
            bf16: BF16Tiles

        @cute.union
        class BF16TilesOrO:
            bf16: BF16Tiles
            o: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(sO_layout)],
                self.buffer_align_bytes,
            ]

        o_storage = (
            BF16TilesOrO
            if const_expr(self.overlap_sO_bf16)
            else cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
        )

        # Every block the tensor core reads its operands out of the K/V stages,
        # P and Q while TMA refills the stages. The softmax's cross-warp row
        # reduction goes through sSoftmaxRed on its critical path every block,
        # and in the same half of shared memory as that traffic its loads wait
        # behind it: with sSoftmaxRed, P and the stages in one half,
        # Qwen3-30B's merged decode ran 8% slower and Qwen3-4B's 3% slower.
        # So the streamed buffers come first, O (or the BF16 tiles), cold
        # through the main loop, carries what follows into the other half,
        # and sSoftmaxRed is last.
        @cute.struct
        class SharedStorage:
            sK: KVStagesOrBF16Tiles if const_expr(self.bf16_split) else kv_stages_storage
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)] if not const_expr(self.v_dtype == self.k_dtype) else cute.struct.MemRange[self.k_dtype, 1],
                self.buffer_align_bytes if not const_expr(self.v_dtype == self.k_dtype) else 1,
            ]
            # Scale factor shared memory (if block-scaled quantization is used)
            sSFK: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfk_smem_size],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfv_smem_size],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, sP_smem_size],
                self.buffer_align_bytes if const_expr(self.transpose_s) else 1,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfq_smem_size],
                self.buffer_align_bytes,
            ]
            sO: o_storage
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, max(self.q_stage * self.m_block_size * 2, self.m_block_size * 4)], self.buffer_align_bytes]
            sSFP: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfp_smem_size],
                self.buffer_align_bytes,
            ]
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            sSoftmaxRed: cute.struct.Align[
                cute.struct.MemRange[Float32, softmax_red_size],
                self.buffer_align_bytes if const_expr(self.transpose_s) else 1,
            ]

        # Remove scale factors to avoid OOM. Seems I can't set their size to 0
        @cute.struct
        class SharedStorageBF16:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, max(self.q_stage * self.m_block_size * 2, self.m_block_size * 4)], self.buffer_align_bytes]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage if const_expr(self.quant_qk) or const_expr(self.quant_pv) else SharedStorageBF16
        
        # Verify shared memory fits within budget
        total_smem_bytes = self.shared_storage.size_in_bytes()
        assert total_smem_bytes <= 227 * 1024, (
            f"SharedStorage {total_smem_bytes // 1024}KB exceeds 227KB limit. "
            f"Reduce kv_stage (currently {self.kv_stage})."
        )
        print(f"Total shared memory used: {total_smem_bytes / 1024:.2f} KB")
        sO_bytes = (
            cute.size_in_bytes(self.o_dtype, sO_layout)
            if const_expr(not (self.overlap_sO_sQ or self.overlap_sO_bf16))
            else 0
        )
        sQ_bytes = cute.size_in_bytes(self.q_dtype, sQ_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, sK_layout)
        sV_bytes = cute.size_in_bytes(self.v_dtype, sV_layout)
        sfq_bytes = cute.size_in_bytes(cutlass.Uint8, sfq_smem_layout_staged)
        sfk_bytes = cute.size_in_bytes(cutlass.Uint8, sfk_smem_layout_staged)
        sfp_bytes = cute.size_in_bytes(cutlass.Uint8, sfp_smem_layout_staged) if const_expr(sfp_smem_layout_staged is not None) else 0
        sfv_bytes = cute.size_in_bytes(cutlass.Uint8, sfv_smem_layout_staged) if const_expr(sfv_smem_layout_staged is not None) else 0
        print(f"sO_size: {sO_bytes / 1024:.2f} KB")
        print(f"sQ_size: {sQ_bytes / 1024:.2f} KB")
        print(f"sK_size: {sK_bytes / 1024:.2f} KB")
        if const_expr(self.v_dtype != self.k_dtype):
            print(f"sV_size: {sV_bytes / 1024:.2f} KB")
        print(f"sfq_smem_size: {sfq_bytes / 1024:.2f} KB")
        print(f"sfk_smem_size: {sfk_bytes / 1024:.2f} KB")
        print(f"sfp_smem_size: {sfp_bytes / 1024:.2f} KB")
        print(f"sfv_smem_size: {sfv_bytes / 1024:.2f} KB")
        
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale = None
        else:
            # NB: If a users passes in a score mod, we want to apply the score-mod in the sm_scaled qk
            # But in the original base 10. We hijack softmax_scale_log2 to just be the change of base
            # and correctly apply the softmax_scale prior to score_mod in the softmax step
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmod.create(seqlen_q)
            seqlen_k_divmod = FastDivmod.create(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mQ_second_tma,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_Q_second,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            tma_atom_sfq,
            tma_tensor_sfq,
            tma_atom_sfq_second,
            tma_tensor_sfq_second,
            tma_atom_sfk,
            tma_tensor_sfk,
            tma_atom_sfv,
            tma_tensor_sfv,
            sfq_smem_layout_staged,
            sfk_smem_layout_staged,
            sfp_smem_layout_staged,
            sfv_smem_layout_staged,
            sQ_bf16_layout,
            tiled_mma_qk_bf16,
            mResidualK_t,
            mResidualV_t,
            mResidualSeqUsedK,
            mSinkSeqUsedK,
            tma_atom_K_bf16,
            tma_atom_V_bf16,
            sK_bf16_layout,
            sV_bf16_layout,
            tiled_mma_pv_bf16,
            mResidualQ_t,
            tma_atom_Q_bf16,
            tP_bf16_layout,
            mResidualBlockIds,
            mOutIndices,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mQSecond: cute.Tensor,
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Q_second: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        tma_atom_sfq: Optional[cute.CopyAtom] = None,
        tma_tensor_sfq: Optional[cute.Tensor] = None,
        tma_atom_sfq_second: Optional[cute.CopyAtom] = None,
        tma_tensor_sfq_second: Optional[cute.Tensor] = None,
        tma_atom_sfk: Optional[cute.CopyAtom] = None,
        tma_tensor_sfk: Optional[cute.Tensor] = None,
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        tma_tensor_sfv: Optional[cute.Tensor] = None,
        sfq_smem_layout_staged: Optional[cute.Layout] = None,
        sfk_smem_layout_staged: Optional[cute.Layout] = None,
        sfp_smem_layout_staged: Optional[cute.Layout] = None,
        sfv_smem_layout_staged: Optional[cute.Layout] = None,
        sQ_bf16_layout: Optional[cute.ComposedLayout] = None,
        tiled_mma_qk_bf16: Optional[cute.TiledMma] = None,
        mResidualK: Optional[cute.Tensor] = None,
        mResidualV: Optional[cute.Tensor] = None,
        mResidualSeqUsedK: Optional[cute.Tensor] = None,
        mSinkSeqUsedK: Optional[cute.Tensor] = None,
        tma_atom_K_bf16: Optional[cute.CopyAtom] = None,
        tma_atom_V_bf16: Optional[cute.CopyAtom] = None,
        sK_bf16_layout: Optional[cute.ComposedLayout] = None,
        sV_bf16_layout: Optional[cute.ComposedLayout] = None,
        tiled_mma_pv_bf16: Optional[cute.TiledMma] = None,
        mResidualQ: Optional[cute.Tensor] = None,
        tma_atom_Q_bf16: Optional[cute.CopyAtom] = None,
        tP_bf16_layout: Optional[cute.ComposedLayout] = None,
        mResidualBlockIds: Optional[cute.Tensor] = None,
        mOutIndices: Optional[cute.Tensor] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        iket.range_push("pro_init")
        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)
            if const_expr(tma_atom_sfq is not None):
                cpasync.prefetch_descriptor(tma_atom_sfq)
            if const_expr(tma_atom_sfk is not None):
                cpasync.prefetch_descriptor(tma_atom_sfk)
            if const_expr(tma_atom_sfv is not None):
                cpasync.prefetch_descriptor(tma_atom_sfv)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Use the first N warps to initialize barriers
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_full_offset + i, 1
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 2:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i,
                    cute.arch.WARP_SIZE
                    * len(self.softmax0_warp_ids)
                    * self.softmax_row_groups,
                )
        if warp_idx == 3:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE
                    )
        if const_expr(not self.use_correction_warps_for_epi) and warp_idx == 4:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
        if warp_idx == 5:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                    cute.arch.WARP_SIZE
                    * (
                        len(self.softmax0_warp_ids) * self.softmax_row_groups
                        + len(self.correction_warp_ids)
                    ),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id])
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 6:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,
                    cute.arch.WARP_SIZE *  len(self.softmax0_warp_ids),
                )
        if warp_idx == 7:
            # Every softmax warp in the CTA arrives here once it leaves its
            # role, whether or not it was handed a row group, and so does every
            # correction warp.
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * (
                    sum(len(ids) for ids in self.softmax_warp_ids)
                    + len(self.correction_warp_ids)
                ),
            )
        if warp_idx == 8:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_sfqk_load_offset + i,
                    cute.arch.WARP_SIZE
                    * len(self.softmax0_warp_ids)
                    * self.softmax_row_groups,
                )
            if const_expr(self.fused_residual_first_block):
                for i in cutlass.range_constexpr(3):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_residual_kv_full_offset + i, 1
                    )
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_residual_kv_empty_offset + i,
                        len([self.mma_warp_id]),
                    )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_bf16_S_full_offset, 1
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_bf16_P_full_offset,
                    cute.arch.WARP_SIZE
                    * len(self.softmax0_warp_ids)
                    * self.softmax_row_groups,
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_bf16_P_full_2_offset,
                    cute.arch.WARP_SIZE
                    * len(self.softmax0_warp_ids)
                    * self.softmax_row_groups,
                )
                if const_expr(self.fused_sink_block):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_bf16_PV_done_offset, 1
                    )
        if warp_idx == 9:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_q_fp4_ready_offset + i,
                    1,  # one arrive: the load warp (after cast)
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_consumed_offset + i, len([self.mma_warp_id])
                )
        iket.range_pop()
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        iket.range_push("pro_sync")
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)
        iket.range_pop()
        iket.range_push("pro_setup")

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_terms = tuple(
            cute.make_tensor(
                sQ.iterator + term * cute.cosize(sQ_layout),
                sQ.layout,
            )
            for term in range(self.query_terms)
        )
        sQ_bf16 = None
        sQ_uint8 = None
        # Byte view for moving complete FP4 rows: a thread owns whole bytes,
        # never a single nibble. Stacking and replication share this layout.
        if const_expr(self.q_replicate > 1 or self.stacked_query_terms):
            uint8_swizzle = cute.make_swizzle(2, 4, 3)
            # FP4 atom_K (per kH) = head_dim_padded//2 elements; Uint8
            # atom_K_byte = atom_K_FP4 // 2 = head_dim_padded // 4.
            atom_K_b_size = self.head_dim_padded // 4
            # M-stride in Uint8 bytes = head_dim_padded // 2 (one byte per 2
            # FP4 elements; one row spans head_dim_padded FP4 = head_dim/2 byte).
            row_bytes = self.head_dim_padded // 2
            kH_stride_bytes = atom_K_b_size  # one kH atom = atom_K_b_size bytes
            stage_stride_bytes = self.m_block_size * row_bytes
            uint8_outer = cute.make_layout(
                ((self.m_block_size, atom_K_b_size), 1, 2, self.q_stage),
                stride=(((row_bytes, 1)), 0, kH_stride_bytes, stage_stride_bytes),
            )
            sQ_uint8 = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, uint8_swizzle, cutlass.Uint8),
                uint8_outer,
            )
        # (MMA, MMA_K, MMA_D, PIPE)
        # K and V share physical SMEM:
        # - same dtype: V aliases K (V uses K's base pointer)
        # - FP4 K + BF16 V: K aliases V (K uses V's base pointer, stride scaled by dtype ratio)
        kv_stages = storage.sK.stages if const_expr(self.bf16_split) else storage.sK
        bf16_tiles = (
            storage.sK.bf16
            if const_expr(self.bf16_split)
            else (storage.sO.bf16 if const_expr(self.overlap_sO_bf16) else None)
        )
        if const_expr(self.v_dtype == self.k_dtype):
            sK = kv_stages.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        elif const_expr(self.k_dtype.width < self.v_dtype.width):
            # K aliases V's buffer — K is smaller, fits inside V's stage.
            # K's stage stride must match V's stage stride in bytes so they align.
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stride_sK_aligned = const_expr(stride_sV * self.v_dtype.width // self.k_dtype.width)
            sK_outer_aligned = cute.make_layout(
                sK_layout.outer.shape,
                stride=(*sK_layout.outer.stride[:-1], stride_sK_aligned),
            )
            sK = storage.sV.get_tensor(sK_outer_aligned, swizzle=sK_layout.inner, dtype=self.k_dtype)
        else:
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            
        if const_expr(self.overlap_sO_bf16):
            sO = storage.sO.o.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        elif const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(
            cute.make_layout(max(self.q_stage * self.m_block_size * 2, self.m_block_size * 4))
        )

        # Get scale factor shared memory tensors if they exist
        sSFV = None
        sSFP = None
        sSFQ = None
        sSFK = None
        if const_expr(self.quant_qk):
            if const_expr(self.overlap_sO_sQ):
                sSFQ = storage.sQ.get_tensor(
                    sfq_smem_layout_staged, dtype=self.sf_dtype
                )
                q_terms_bytes = const_expr(
                    self.query_terms * cute.cosize(sQ_layout) * self.q_dtype.width // 8
                )
                sSFQ = cute.make_tensor(sSFQ.iterator + q_terms_bytes, sSFQ.layout)
            else:
                sSFQ = storage.sSFQ.get_tensor(sfq_smem_layout_staged)
            sSFQ_terms = tuple(
                cute.make_tensor(
                    sSFQ.iterator
                    + term * cute.cosize(sfq_smem_layout_staged),
                    sSFQ.layout,
                )
                for term in range(self.query_terms)
            )
            sSFK = storage.sSFK.get_tensor(sfk_smem_layout_staged)
        else:
            sSFQ_terms = (None,)
        if const_expr(sfp_smem_layout_staged is not None):
            sSFP = storage.sSFP.get_tensor(sfp_smem_layout_staged)
        if const_expr(sfv_smem_layout_staged is not None):
            sSFV = storage.sSFV.get_tensor(sfv_smem_layout_staged)

        sK_bf16 = None
        sV_bf16 = None
        sP_bf16 = None
        sP_bf16_mk = None
        if const_expr(self.fused_residual_first_block):
            sK_bf16 = bf16_tiles.sK_bf16.get_tensor(sK_bf16_layout.outer, swizzle=sK_bf16_layout.inner)
            sV_bf16 = bf16_tiles.sV_bf16.get_tensor(sV_bf16_layout.outer, swizzle=sV_bf16_layout.inner)
            sQ_bf16 = bf16_tiles.sQ_bf16.get_tensor(sQ_bf16_layout.outer, swizzle=sQ_bf16_layout.inner)
            if const_expr(self.transpose_s):
                sP_bf16 = bf16_tiles.sQ_bf16.get_tensor(
                    tP_bf16_layout.outer, swizzle=tP_bf16_layout.inner
                )
                # The softmax writes one query row at a time down its own kv
                # column, so it wants the plain (query row, kv) view of the
                # operand layout rather than its atom decomposition.
                sP_bf16_mk = cute.composition(
                    sP_bf16[None, None, None, 0],
                    cute.make_layout((self.m_block_size, self.n_block_size)),
                )

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)


        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(2)
        )
        tOtOs = tuple(
            cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
            for stage in range(self.q_stage)
        )

        # P in shared memory: the MMA A-operand tile, plus a byte view for the
        # softmax warps, which own single nibbles and can only store whole bytes.
        sP = None
        sP_uint8 = None
        if const_expr(self.transpose_s):
            sP = storage.sP.get_tensor(tP_layout.outer, swizzle=tP_layout.inner)
            sP_uint8 = cute.make_tensor(
                cute.recast_ptr(sP.iterator, cute.make_swizzle(2, 4, 3), cutlass.Uint8),
                cute.make_layout(
                    (
                        (self.m_block_size, self.n_block_size // 4),
                        1,
                        2,
                        self.acc_stage,
                    ),
                    stride=(
                        (self.n_block_size // 2, 1),
                        0,
                        self.n_block_size // 4,
                        self.m_block_size * self.n_block_size // 2,
                    ),
                ),
            )
            sSoftmaxRed = storage.sSoftmaxRed.get_tensor(
                cute.make_layout(self.softmax_red_slots)
            )
        else:
            sSoftmaxRed = None

        if const_expr(self.transpose_s):
            tOrP = thr_mma_pv.make_fragment_A(sP)[None, None, None, 0]
            tOrPs = [tOrP, tOrP]
        else:
            tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
            tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

            tOrPs = [
                cute.make_tensor(
                    tOrP.iterator
                    + self.qk_acc_dtype.width // tOrP._dtype.width * self.tmem_p_offset[stage],
                    tOrP.layout,
                )
                for stage in range(2)
            ]
        tOrPs_bf16 = (None, None)
        if const_expr(self.fused_residual_first_block):
            thr_mma_pv_bf16 = tiled_mma_pv_bf16.get_slice(0)
            if const_expr(self.transpose_s):
                tOrP_bf16 = thr_mma_pv_bf16.make_fragment_A(sP_bf16)[None, None, None, 0]
                if const_expr(self.fused_sink_block):
                    # The sink occupies one 16-token BF16 MMA-K block. Keep the
                    # original operand layout and expose only that first block.
                    tOrP_sink_bf16 = cute.make_tensor(
                        tOrP_bf16.iterator,
                        cute.make_layout(
                            (
                                tOrP_bf16.layout.shape[0],
                                tOrP_bf16.layout.shape[1],
                                (1, 1),
                            ),
                            stride=tOrP_bf16.layout.stride,
                        ),
                    )
                    tOrPs_bf16 = (tOrP_bf16, tOrP_sink_bf16)
                else:
                    tOrPs_bf16 = (tOrP_bf16, tOrP_bf16)
            else:
                tP_bf16 = cute.make_tensor(tStS.iterator, tP_bf16_layout.outer)
                tOrP_bf16 = thr_mma_pv_bf16.make_fragment_A(tP_bf16)[None, None, None, 0]
                tOrPs_bf16 = tuple(
                    cute.make_tensor(
                        tOrP_bf16.iterator
                        + self.qk_acc_dtype.width // tOrP_bf16._dtype.width
                        * self.tmem_p_bf16_offset[stage],
                        tOrP_bf16.layout,
                    )
                    for stage in range(2)
                )
        # Setup scale factor TMEM tensors and S2T copy operations
        # Use the TMEM region immediately following the accumulator (O tensor)

        # sf_tmem_ptr = cute.make_ptr(self.sf_dtype, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)

        align = 16  # Required for tcgen05.cp 4x32dp128bit.
        # find_tmem_tensor_col_offset returns u32 columns; convert to sf_dtype elements
        sf_dtype_per_u32 = 32 // self.sf_dtype.width
        tCtSFQ_terms = [[None] * self.q_stage]
        tCtSFKs = [None] * self.q_stage
        if const_expr(self.quant_qk):
            if const_expr(self.q_stage == 2):
                sfq_tmem_col_offsets = [self.tmem_s_offset[self.q_stage - 1 - stage] for stage in range(self.q_stage)]
            else:
                sfq_tmem_col_offsets = [self.tmem_s_offset[1]]  # the unused stage-1 S region
            sfq_tmem_ptrs = [cute.recast_ptr(
                            cute.make_ptr(Float32, sfq_tmem_col_offsets[stage], # shuffle to minimize dependency
                            mem_space=cute.AddressSpace.tmem, assumed_align=align),
                            dtype=self.sf_dtype) for stage in range(self.q_stage)
                            ]

            # (MMA, MMA_M, MMA_K) — Q is the B operand once S is transposed.
            make_tmem_layout_sfq = (
                blockscaled_utils.make_tmem_layout_sfb
                if const_expr(self.transpose_s)
                else blockscaled_utils.make_tmem_layout_sfa
            )
            make_tmem_layout_sfk = (
                blockscaled_utils.make_tmem_layout_sfa
                if const_expr(self.transpose_s)
                else blockscaled_utils.make_tmem_layout_sfb
            )
            tCtSFQ_layout = make_tmem_layout_sfq(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                cute.slice_(sfq_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFQ_probe = cute.make_tensor(
                sfq_tmem_ptrs[0], tCtSFQ_layout
            )
            sfq_col_offset = tcgen05.find_tmem_tensor_col_offset(
                tCtSFQ_probe
            )
            sfq_offset = math.ceil(sfq_col_offset * sf_dtype_per_u32 / align) * align
            tCtSFQ_terms = [
                [
                    cute.make_tensor(
                        sfq_tmem_ptrs[stage] + term * sfq_offset,
                        tCtSFQ_layout,
                    )
                    for stage in range(self.q_stage)
                ]
                for term in range(self.query_terms)
            ]

            # K scale factors follow every independently scaled query term.
            sfk_tmem_ptrs = [
                sfq_tmem_ptrs[stage] + self.query_terms * sfq_offset
                for stage in range(self.q_stage)
            ]

            # (MMA, MMA_N, MMA_K)
            tCtSFK_layout = make_tmem_layout_sfk(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                cute.slice_(sfk_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFKs = [cute.make_tensor(sfk_tmem_ptrs[stage], tCtSFK_layout) for stage in range(self.q_stage)]
        
        # Setup SFP and SFV TMEM tensors. These normally sit in the columns S
        # was read out of, which costs the next QK a wait on this block's PV.
        # Where tmem_sfpv_offset says there is room, they get their own columns
        # instead and the two GEMMs stop contending.
        tCtSFPs = [None] * self.q_stage
        tCtSFVs = [None] * self.q_stage
        if const_expr(self.quant_pv):
            if const_expr(self.tmem_sfpv_offset is not None):
                sfpv_col_offsets = [self.tmem_sfpv_offset] * self.q_stage
            else:
                sfpv_col_offsets = [self.tmem_s_offset[stage] for stage in range(self.q_stage)]
            sfp_tmem_ptrs = [cute.recast_ptr(
                            cute.make_ptr(Float32, sfpv_col_offsets[stage],
                            mem_space=cute.AddressSpace.tmem, assumed_align=align),
                            dtype=self.sf_dtype) for stage in range(self.q_stage)]
            # A narrow PV swaps the operands, so P's scales are the B operand's
            # and V's the A operand's, as K's and Q's are on a transposed QK.
            # The shared-memory layouts are the same either way round for a
            # 128 x 128 tile; only the tensor-memory placement follows the role.
            make_tmem_layout_sfp = (
                blockscaled_utils.make_tmem_layout_sfb
                if const_expr(self.narrow_pv)
                else blockscaled_utils.make_tmem_layout_sfa
            )
            make_tmem_layout_sfv = (
                blockscaled_utils.make_tmem_layout_sfa
                if const_expr(self.narrow_pv)
                else blockscaled_utils.make_tmem_layout_sfb
            )
            # (MMA, MMA_M, MMA_K) 
            tCtSFP_layout = make_tmem_layout_sfp(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                cute.slice_(sfp_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFPs = [cute.make_tensor(sfp_tmem_ptrs[stage], tCtSFP_layout) for stage in range(self.q_stage)]
            
            # Make SFV tmem tensor
            sfp_offset = math.ceil(tcgen05.find_tmem_tensor_col_offset(tCtSFPs[0]) * sf_dtype_per_u32 / align) * align
            sfv_tmem_ptrs = [sfp_tmem_ptrs[stage] + sfp_offset for stage in range(self.q_stage)]

            # (MMA, MMA_N, MMA_K) for P*V operation (V is the B matrix)
            tCtSFV_layout = make_tmem_layout_sfv(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                cute.slice_(sfv_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFVs = [cute.make_tensor(sfv_tmem_ptrs[stage], tCtSFV_layout) for stage in range(self.q_stage)]
            if const_expr(self.tmem_sfpv_offset is not None):
                sfpv_cols = math.ceil(
                    (sfp_offset + tcgen05.find_tmem_tensor_col_offset(tCtSFVs[0]) * sf_dtype_per_u32)
                    / sf_dtype_per_u32
                )
                assert self.tmem_sfpv_offset + sfpv_cols <= self.tmem_alloc_cols, (
                    f"SFP/SFV need {sfpv_cols} columns at {self.tmem_sfpv_offset}, "
                    f"past the {self.tmem_alloc_cols}-column capacity"
                )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=(
                1
                if const_expr(self.seqlen_q_static_one)
                else mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)
        is_bf16_cta = False
        if const_expr(self.bf16_split):
            is_bf16_cta = self.is_bf16_split(
                TileSchedulerCls().initial_work_tile_info().tile_idx[3], num_splits
            )
        iket.range_pop()

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if const_expr(len(self.empty_warp_ids) > 1):
            if warp_idx == self.empty_warp_ids[1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        assert len(self.empty_warp_ids) <= 2

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            role_lifetime = iket.range_start("load_life")
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            iket.range_push("load_main")
            self.run_split_role(is_bf16_cta, partial(
                self.load,
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mQSecond,
                mK,
                mV,
                sQ_terms,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_Q_second,
                tma_atom_K,
                tma_atom_V,
                tma_atom_sfq,
                tma_tensor_sfq,
                tma_atom_sfq_second,
                tma_tensor_sfq_second,
                tma_atom_sfk,
                tma_tensor_sfk,
                sSFQ_terms,
                sSFK,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                tma_atom_sfv,
                tma_tensor_sfv,
                sfv_smem_layout_staged,
                sSFV,
                sQ_bf16=sQ_bf16,
                sQ_uint8=sQ_uint8,
                mResidualK_t=mResidualK,
                mResidualV_t=mResidualV,
                mResidualQ_t=mResidualQ,
                tma_atom_K_bf16=tma_atom_K_bf16,
                tma_atom_V_bf16=tma_atom_V_bf16,
                tma_atom_Q_bf16=tma_atom_Q_bf16,
                sK_bf16=sK_bf16,
                sV_bf16=sV_bf16,
                tiled_mma_qk_bf16=tiled_mma_qk_bf16,
                tiled_mma_pv_bf16=tiled_mma_pv_bf16,
                mResidualBlockIds=mResidualBlockIds,
            ))
            iket.range_pop()
            iket.range_end(role_lifetime)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            role_lifetime = iket.range_start("mma_life")
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            self.run_split_role(is_bf16_cta, partial(
                self.mma,
                tiled_mma_qk,
                tiled_mma_pv,
                sQ_terms,
                sK,
                sV,
                tStSs,
                tOtOs,
                tOrPs,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                sSFQ_terms,
                sSFK,
                sSFV,
                sSFP,
                tCtSFQ_terms,
                tCtSFKs,
                tCtSFPs,
                tCtSFVs,
                tiled_mma_qk_bf16=tiled_mma_qk_bf16,
                tiled_mma_pv_bf16=tiled_mma_pv_bf16,
                sK_bf16=sK_bf16,
                sV_bf16=sV_bf16,
                sQ_bf16=sQ_bf16,
                mResidualSeqUsedK=mResidualSeqUsedK,
                tOrPs_bf16=tOrPs_bf16,
                sP=sP,
                sP_bf16=sP_bf16,
            ))

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)
            iket.range_end(role_lifetime)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                role_lifetime = iket.range_start("epi_life")
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                iket.range_push("epi_main")
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    mOutIndices=mOutIndices,
                )
                iket.range_pop()
                iket.range_end(role_lifetime)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0]:
            role_lifetime = iket.range_start("softmax_life")
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                blocksparse_tensors=blocksparse_tensors,
                sSFP=sSFP,
                mResidualSeqUsedK=mResidualSeqUsedK,
                mSinkSeqUsedK=mSinkSeqUsedK,
                sP_uint8=sP_uint8,
                sSoftmaxRed=sSoftmaxRed,
                sP_bf16=sP_bf16,
                sP_bf16_mk=sP_bf16_mk,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                # Compute tCtSFP/tStSi unconditionally so they have stable types.
                # When q_stage=1 we always read [0]; when q_stage=2 we pick by stage.
                if const_expr(self.quant_pv):
                    if const_expr(self.q_stage == 2):
                        tCtSFP = tCtSFPs[0] if stage == 0 else tCtSFPs[1]
                    else:
                        tCtSFP = tCtSFPs[0]
                else:
                    tCtSFP = None
                if const_expr(self.q_stage == 2):
                    s_off = self.tmem_s_offset[0] if stage == 0 else self.tmem_s_offset[1]
                else:
                    s_off = self.tmem_s_offset[0]
                tStSi_local = cute.make_tensor(tStS.iterator + s_off, tStS.layout)
                if const_expr(self.softmax_row_groups > 1):
                    # Every group drives the same single Q stage over its own
                    # query rows; specialising on the group keeps its row
                    # offset, its scratch base and its barrier id compile-time
                    # constants.
                    softmax_warpgroup = warp_idx // len(self.softmax0_warp_ids)
                    for group in cutlass.range_constexpr(self.softmax_row_groups):
                        if softmax_warpgroup == group:
                            self.run_split_role(is_bf16_cta, partial(
                                softmax_loop,
                                stage=0,
                                tStSi=tStSi_local,
                                tCtSFP=tCtSFP,
                                row_group=group,
                            ))
                elif const_expr(self.q_stage == 2) or stage == 0:
                    self.run_split_role(is_bf16_cta, partial(
                        softmax_loop,
                        stage=stage,
                        tStSi=tStSi_local,
                        tCtSFP=tCtSFP, # need to copy P sf to tmem after exp
                    ))
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                    softmax_loop(stage=0, tStSi=tStSi, tCtSFP=tCtSFPs[0])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                    softmax_loop(stage=1, tStSi=tStSi, tCtSFP=tCtSFPs[1])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            iket.range_end(role_lifetime)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            role_lifetime = iket.range_start("corr_life")
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            iket.range_push("corr_main")
            self.run_split_role(is_bf16_cta, partial(
                self.correction_loop,
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtOs,
                sScale,
                mO,
                mLSE,
                sO,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            ))
            iket.range_pop()
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            iket.range_end(role_lifetime)

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mQSecond: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ_terms: Tuple[cute.Tensor, ...],
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Q_second: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_sfq: Optional[cute.CopyAtom],
        tma_tensor_sfq: Optional[cute.Tensor],
        tma_atom_sfq_second: Optional[cute.CopyAtom],
        tma_tensor_sfq_second: Optional[cute.Tensor],
        tma_atom_sfk: Optional[cute.CopyAtom],
        tma_tensor_sfk: Optional[cute.Tensor],
        sSFQ_terms: Tuple[Optional[cute.Tensor], ...],
        sSFK: Optional[cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        tma_tensor_sfv: Optional[cute.Tensor] = None,
        sfv_smem_layout_staged: Optional[cute.ComposedLayout] = None,
        sSFV: Optional[cute.Tensor] = None,
        sQ_bf16: Optional[cute.Tensor] = None,
        sQ_uint8: Optional[cute.Tensor] = None,
        mResidualK_t: Optional[cute.Tensor] = None,
        mResidualV_t: Optional[cute.Tensor] = None,
        mResidualQ_t: Optional[cute.Tensor] = None,
        tma_atom_K_bf16: Optional[cute.CopyAtom] = None,
        tma_atom_V_bf16: Optional[cute.CopyAtom] = None,
        tma_atom_Q_bf16: Optional[cute.CopyAtom] = None,
        sK_bf16: Optional[cute.Tensor] = None,
        sV_bf16: Optional[cute.Tensor] = None,
        tiled_mma_qk_bf16: Optional[cute.TiledMma] = None,
        tiled_mma_pv_bf16: Optional[cute.TiledMma] = None,
        mResidualBlockIds: Optional[cute.Tensor] = None,
        role: cutlass.Constexpr = SplitRole.ALL,
    ):
        loads_pages = const_expr(role != SplitRole.BF16)
        loads_bf16 = const_expr(
            self.fused_residual_first_block and role != SplitRole.PAGES
        )
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        residual_kv_empty_producer_phase = Int32(1)
        residual_k_empty_producer_phase = Int32(1)
        residual_v_empty_producer_phase = Int32(1)
        residual_q_empty_producer_phase = Int32(1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            # mQ: [s, d, h, b]
            mQ_cur = seqlen.offset_batch_Q(
                mQ, batch_idx, dim=3
            )[None, None, head_idx]
            gQ = cute.local_tile(
                mQ_cur,
                cute.select(self.mma_tiler_qk, mode=[0, 2]),
                (None, 0),
            )
            mQ_second_cur = seqlen.offset_batch_Q(
                mQSecond, batch_idx, dim=3
            )[None, None, head_idx]
            gQ_second = cute.local_tile(
                mQ_second_cur,
                cute.select(self.mma_tiler_qk, mode=[0, 2]),
                (None, 0),
            )

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgQ_second = thr_mma_qk.partition_A(gQ_second)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                tSgQ,
                sQ_terms[0],
            )
            load_Q_second_fn = load_Q_fn
            if const_expr(self.query_terms == 2):
                (
                    load_Q_second_fn,
                    _,
                    _,
                ) = copy_utils.tma_get_copy_fn(
                    tma_atom_Q_second,
                    0,
                    cute.make_layout(1),
                    tSgQ_second,
                    sQ_terms[1],
                )

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmod.create(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_SFQ_fn = None
            load_SFQ_second_fn = None
            if const_expr(self.quant_qk):
                tma_tensor_sfq_cur = seqlen.offset_batch_Q(
                    tma_tensor_sfq, batch_idx, dim=3
                )[None, None, head_idx]
                gSFQ = cute.local_tile(
                    tma_tensor_sfq_cur,
                    cute.select(self.mma_tiler_qk, mode=[0, 2]),
                    (None, 0),
                )
                tSgSFQ = thr_mma_qk.partition_A(gSFQ)
                load_SFQ_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_sfq,
                    0,
                    cute.make_layout(1),
                    tSgSFQ,
                    sSFQ_terms[0],
                    filter_zeros=True,
                )
                load_SFQ_second_fn = load_SFQ_fn
                if const_expr(self.query_terms == 2):
                    tma_tensor_sfq_second_cur = seqlen.offset_batch_Q(
                        tma_tensor_sfq_second, batch_idx, dim=3
                    )[None, None, head_idx]
                    gSFQ_second = cute.local_tile(
                        tma_tensor_sfq_second_cur,
                        cute.select(self.mma_tiler_qk, mode=[0, 2]),
                        (None, 0),
                    )
                    tSgSFQ_second = thr_mma_qk.partition_A(
                        gSFQ_second
                    )
                    (
                        load_SFQ_second_fn,
                        _,
                        _,
                    ) = copy_utils.tma_get_copy_fn(
                        tma_atom_sfq_second,
                        0,
                        cute.make_layout(1),
                        tSgSFQ_second,
                        sSFQ_terms[1],
                        filter_zeros=True,
                    )
            
            # Partition SFK similar to K - index batch and head first like mK_cur
            tKsSFK = None
            tKgSFK = None
            if const_expr(self.quant_qk):
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, batch_idx]
                        gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                    else:
                        tma_tensor_sfk_cur = cute.domain_offset((seqlen.offset_k, 0), tma_tensor_sfk[None, None, head_idx_kv])
                        gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                else:
                    tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, None]
                    gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
                tSgSFK = thr_mma_qk.partition_B(gSFK)
                # Group only the first 3 modes (static MMA modes) to avoid grouping dynamic Rest modes
                tKsSFK, tKgSFK = cpasync.tma_partition(
                    tma_atom_sfk,
                    0, 
                    cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK, 0, 3),
                )
                tKsSFK = cute.filter_zeros(tKsSFK)
                tKgSFK = cute.filter_zeros(tKgSFK)
            
            # Partition SFV similar to V
            tVsSFV, tVgSFV = None, None
            if const_expr(self.quant_pv):
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        tma_tensor_sfv_cur = tma_tensor_sfv[None, None, head_idx_kv, batch_idx]
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                    else:
                        tma_tensor_sfv_cur = cute.domain_offset((0, seqlen.offset_k), tma_tensor_sfv[None, None, head_idx_kv])
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                else:
                    tma_tensor_sfv_cur = tma_tensor_sfv[None, None, head_idx_kv, None]
                    gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
                tOgSFV = thr_mma_pv.partition_B(gSFV)
                # Group only the first 3 modes (static MMA modes) to avoid grouping dynamic Rest modes
                tVsSFV, tVgSFV = cpasync.tma_partition(
                    tma_atom_sfv,
                    0,  
                    cute.make_layout(1),
                    cute.group_modes(sSFV, 0, 3),
                    cute.group_modes(tOgSFV, 0, 3),
                )
                tVsSFV = cute.filter_zeros(tVsSFV)
                tVgSFV = cute.filter_zeros(tVgSFV)

            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
                load_SFQ_fn=load_SFQ_fn,
                load_Q_second_fn=load_Q_second_fn,
                load_SFQ_second_fn=load_SFQ_second_fn,
            )
            replicate_Q = None
            if const_expr(self.q_replicate > 1):
                # Replication gets its own pass over the tile between the TMA
                # and the first MMA, signalling mbar_q_fp4_ready when the tile
                # is whole.
                replicate_Q = partial(
                    self.replicate_Q_fp4_rows,
                    sQ_uint8,
                    sSFQ_terms[0],
                    mbar_ptr + self.mbar_load_q_full_offset,
                    mbar_ptr + self.mbar_q_fp4_ready_offset,
                    phase=q_producer_phase ^ 1,  # waits for the just-issued TMA
                    tidx=tidx,
                )
            stack_Q = None
            if const_expr(self.stacked_query_terms):
                stack_Q = partial(
                    self.stack_Q_fp4_terms,
                    sQ_uint8,
                    sQ_terms[1],
                    sSFQ_terms[0],
                    sSFQ_terms[1],
                    mbar_ptr + self.mbar_load_q_full_offset,
                    mbar_ptr + self.mbar_q_fp4_ready_offset,
                    phase=q_producer_phase ^ 1,
                    tidx=tidx,
                )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
                tma_atom_sf=tma_atom_sfk,
                tXgSF=tKgSFK,
                tXsSF=tKsSFK,
            )

            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
                tma_atom_sf=tma_atom_sfv,
                tXgSF=tVgSFV,
                tXsSF=tVsSFV,
            )

            if const_expr(not self.use_block_sparsity):
                iket.range_push("load_seqinfo")
                n_block_min, n_block_max = self.split_block_range(
                    block_info, seqlen, m_block, split_idx, num_splits
                )
                iket.range_pop()
                has_work = self.tile_has_work(
                    n_block_min, n_block_max, split_idx, num_splits
                )
                # A split with no kv block runs no MMA, so nothing would release
                # the BF16 tiles the sink reload waits on: load only for work.
                if has_work:
                    if const_expr(loads_bf16):
                        if const_expr(self.residual_source == "paged_bf16"):
                            # Graph padding keeps row_to_slot=-1 so it cannot own or
                            # write a tail. TMA still needs a legal source page before
                            # the zero residual/sink lengths mask the row away.
                            residual_kv_idx = cutlass.max(
                                mResidualBlockIds[batch_idx], Int32(0)
                            )
                        else:
                            residual_kv_idx = batch_idx
                        # BF16 residual K/V layout: (s=128, d, h_kv, b-or-num_blocks).
                        # Slice batch/block + head, local_tile to BLOCK_SIZE (=128),
                        # partition via the BF16 tiled MMAs, then tma_partition with
                        # the BF16 atoms.
                        thr_mma_qk_bf16_slc = tiled_mma_qk_bf16.get_slice(0)
                        thr_mma_pv_bf16_slc = tiled_mma_pv_bf16.get_slice(0)
                        mResK_cur = mResidualK_t[None, None, head_idx_kv, residual_kv_idx]
                        mResV_cur = mResidualV_t[None, None, head_idx_kv, residual_kv_idx]
                        gResK = cute.local_tile(
                            mResK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
                        )
                        gResV = cute.local_tile(
                            mResV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
                        )
                        tSgResK = thr_mma_qk_bf16_slc.partition_B(gResK)
                        tOgResV = thr_mma_pv_bf16_slc.partition_B(gResV)
                        tKsK_bf16, tKgResK = cpasync.tma_partition(
                            tma_atom_K_bf16,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sK_bf16, 0, 3),
                            cute.group_modes(tSgResK, 0, 3),
                        )
                        tVsV_bf16, tVgResV = cpasync.tma_partition(
                            tma_atom_V_bf16,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sV_bf16, 0, 3),
                            cute.group_modes(tOgResV, 0, 3),
                        )
                        mResQ_cur = mResidualQ_t[None, None, head_idx, batch_idx]
                        gResQ = cute.local_tile(
                            mResQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
                        )
                        tSgResQ = thr_mma_qk_bf16_slc.partition_A(gResQ)
                        tQsQ_bf16, tQgResQ = cpasync.tma_partition(
                            tma_atom_Q_bf16,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sQ_bf16, 0, 3),
                            cute.group_modes(tSgResQ, 0, 3),
                        )
                        # BF16 K/V/Q are loaded into dedicated single-stage SMEM tiles
                        # (sK_bf16 / sV_bf16 / sQ_bf16) using dedicated mbarriers
                        # (mbar_residual_kv_full + 0 = K, +1 = V, +2 = Q). The FP4
                        # mainloop's kv pipeline is unchanged. For subsequent work
                        # tiles (M>1), wait on the empty barrier so the MMA warp
                        # has finished consuming the previous tile's data before
                        # the load warp overwrites the SMEM.
                        bf16_full_mbar = mbar_ptr + self.mbar_residual_kv_full_offset
                        bf16_empty_mbar = mbar_ptr + self.mbar_residual_kv_empty_offset
                        if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                            iket.range_push("load_wait_res")
                            cute.arch.mbarrier_wait(
                                bf16_empty_mbar + 0,
                                residual_k_empty_producer_phase
                                if const_expr(self.fused_sink_block)
                                else residual_kv_empty_producer_phase,
                            )
                            cute.arch.mbarrier_wait(
                                bf16_empty_mbar + 1,
                                residual_v_empty_producer_phase
                                if const_expr(self.fused_sink_block)
                                else residual_kv_empty_producer_phase,
                            )
                            cute.arch.mbarrier_wait(
                                bf16_empty_mbar + 2,
                                residual_q_empty_producer_phase
                                if const_expr(self.fused_sink_block)
                                else residual_kv_empty_producer_phase,
                            )
                            iket.range_pop()
                            # Slot 0: BF16 K
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bf16_full_mbar + 0,
                                    cute.cosize(sK_bf16) * 2,  # bf16 = 2 bytes
                                )
                            cute.copy(
                                tma_atom_K_bf16,
                                tKgResK[None, 0],
                                tKsK_bf16[None, 0],
                                tma_bar_ptr=bf16_full_mbar + 0,
                            )
                            # Slot 1: BF16 V
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bf16_full_mbar + 1,
                                    cute.cosize(sV_bf16) * 2,
                                )
                            cute.copy(
                                tma_atom_V_bf16,
                                tVgResV[None, 0],
                                tVsV_bf16[None, 0],
                                tma_bar_ptr=bf16_full_mbar + 1,
                            )
                            sQ_bf16_stage_elems = const_expr(cute.cosize(sQ_bf16))
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bf16_full_mbar + 2,
                                    sQ_bf16_stage_elems * 2,
                                )
                            cute.copy(
                                tma_atom_Q_bf16,
                                tQgResQ[None, m_block],
                                tQsQ_bf16[None, 0],
                                tma_bar_ptr=bf16_full_mbar + 2,
                            )
                            if const_expr(self.fused_sink_block):
                                residual_k_empty_producer_phase ^= 1
                                residual_v_empty_producer_phase ^= 1
                                residual_q_empty_producer_phase ^= 1
                            else:
                                residual_kv_empty_producer_phase ^= 1
                    if const_expr(loads_pages):
                        if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                            load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0 + SFQ0
                        n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                        iket.range_push("load_pageidx")
                        page_idx = (
                            mPageTable[batch_idx, n_block_first]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block_first)
                        iket.range_pop()
                        iket.range_push("load_issue_k0")
                        load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0 + SFK0
                        iket.range_pop()
                        kv_producer_state.advance()
                        if const_expr(self.q_stage == 2) and (const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE):
                            load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1 + SFQ1
                        if const_expr(self.q_replicate > 1):
                            for qs in cutlass.range_constexpr(self.q_stage):
                                replicate_Q(stage=qs)
                        if const_expr(self.stacked_query_terms):
                            # One warp owns both the byte copy and its single
                            # ready-barrier arrival, including non-TMA KV loads.
                            if tidx < cute.arch.WARP_SIZE:
                                stack_Q(stage=0)
                        q_producer_phase ^= 1
                        load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0 + SFV0
                        kv_producer_state.advance()
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block = n_block_max - 2 - i
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None and self.use_tma_KV)
                                else None
                            )
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki + SFKi
                            kv_producer_state.advance()
                            load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi + SFVi
                            kv_producer_state.advance()
                    # The sink is merged after the last page, so its operands
                    # are loaded behind the pages: they stream while the tail's
                    # GEMMs run, and the sink's own load hides under the last
                    # blocks. K can be replaced once the tail QK has issued; V
                    # and Q/P stay resident until the tail PV consumes them, so
                    # each goes through its own empty barrier.
                    if const_expr(loads_bf16 and self.fused_sink_block):
                        if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                            cute.arch.mbarrier_wait(
                                bf16_empty_mbar + 0,
                                residual_k_empty_producer_phase,
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bf16_full_mbar + 0,
                                    cute.cosize(sK_bf16) * 2,
                                )
                            cute.copy(
                                tma_atom_K_bf16,
                                tKgResK[None, 1],
                                tKsK_bf16[None, 0],
                                tma_bar_ptr=bf16_full_mbar + 0,
                            )
                            residual_k_empty_producer_phase ^= 1

                            cute.arch.mbarrier_wait(
                                bf16_empty_mbar + 1,
                                residual_v_empty_producer_phase,
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bf16_full_mbar + 1,
                                    cute.cosize(sV_bf16) * 2,
                                )
                            cute.copy(
                                tma_atom_V_bf16,
                                tVgResV[None, 1],
                                tVsV_bf16[None, 0],
                                tma_bar_ptr=bf16_full_mbar + 1,
                            )
                            residual_v_empty_producer_phase ^= 1

                            cute.arch.mbarrier_wait(
                                bf16_empty_mbar + 2,
                                residual_q_empty_producer_phase,
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bf16_full_mbar + 2,
                                    cute.cosize(sQ_bf16) * 2,
                                )
                            cute.copy(
                                tma_atom_Q_bf16,
                                tQgResQ[None, m_block],
                                tQsQ_bf16[None, 0],
                                tma_bar_ptr=bf16_full_mbar + 2,
                            )
                            residual_q_empty_producer_phase ^= 1

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                )


            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    def qk_gemm(self, gemm, tCrK: cute.Tensor, sK_cur: cute.Tensor):
        """Issue one QK GEMM with the K tile in whichever operand slot it holds.

        K is the B operand normally and the A operand once S is transposed. The
        Q side is already bound into ``gemm``.
        """
        if const_expr(self.transpose_s):
            gemm(tCrA=tCrK, sA=sK_cur)
        else:
            gemm(tCrB=tCrK, sB=sK_cur)

    def release_shared_p(self, mbar_ptr: cute.Pointer, stage: int):
        """Signal once the PV GEMM just issued has read P from shared memory.

        Transposed, P and its scale factors sit in one shared-memory tile that
        the softmax rewrites for every block, and the next block's QK is issued
        ahead of this block's PV, so the S it signals says nothing about this
        PV. Every FP4 PV signals here, which keeps one completion per block.
        """
        if const_expr(self.transpose_s):
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_P_consumed_offset + stage)

    def pv_gemm(
        self,
        gemm,
        tOrVi: cute.Tensor,
        sV_cur: cute.Tensor,
        zero_init,
        mbar_ptr: cute.Pointer,
        stage: int,
        phase: Int32,
    ):
        """Issue one PV GEMM, splitting the wait on P only where P is in TMEM.

        With P in tensor memory the instruction can start on its first half and
        wait for the rest mid-flight. A shared-memory A operand has no such
        entry point, so the whole tile must already be published; the caller's
        wait on ``mbar_P_full_O_rescaled`` covers it and the second-half barrier
        is unused.
        """
        if const_expr(self.transpose_s):
            gemm(tCrB=tOrVi, sB=sV_cur, zero_init=zero_init)
        else:
            gemm(
                tCrB=tOrVi,
                sB=sV_cur,
                zero_init=zero_init,
                mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                mbar_phase=phase,
            )

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ_terms: Tuple[cute.Tensor, ...],
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        # In smem
        sSFQ_terms: Tuple[Optional[cute.Tensor], ...],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        # In tmem - per-stage scale factors
        tCtSFQ_terms: Tuple[Tuple[cute.Tensor, ...], ...],
        tCtSFKs: Tuple[cute.Tensor, ...],
        tCtSFPs: Tuple[cute.Tensor, ...],
        tCtSFVs: Tuple[cute.Tensor, ...],
        tiled_mma_qk_bf16: Optional[cute.TiledMma] = None,
        tiled_mma_pv_bf16: Optional[cute.TiledMma] = None,
        sK_bf16: Optional[cute.Tensor] = None,
        sV_bf16: Optional[cute.Tensor] = None,
        sQ_bf16: Optional[cute.Tensor] = None,
        mResidualSeqUsedK: Optional[cute.Tensor] = None,
        tOrPs_bf16: Tuple[Optional[cute.Tensor], Optional[cute.Tensor]] = (None, None),
        sP: Optional[cute.Tensor] = None,
        sP_bf16: Optional[cute.Tensor] = None,
        role: cutlass.Constexpr = SplitRole.ALL,
    ):
        runs_pages = const_expr(role != SplitRole.BF16)
        runs_tail = const_expr(
            self.fused_residual_first_block and role != SplitRole.PAGES
        )
        runs_sink = const_expr(self.fused_sink_block and role != SplitRole.PAGES)
        if const_expr(self.fused_residual_first_block):
            tOrV_bf16 = tiled_mma_pv_bf16.make_fragment_B(sV_bf16)
            if const_expr(self.transpose_s):
                tSrK_bf16 = tiled_mma_qk_bf16.make_fragment_A(sK_bf16)
                tSrQ_bf16 = tiled_mma_qk_bf16.make_fragment_B(sQ_bf16)
            else:
                tSrK_bf16 = tiled_mma_qk_bf16.make_fragment_B(sK_bf16)
                tSrQ_bf16 = tiled_mma_qk_bf16.make_fragment_A(sQ_bf16)
            tSrQ_bf16_stages = (tSrQ_bf16[None, None, None, 0], tSrQ_bf16[None, None, None, 0])
        else:
            tSrK_bf16 = None
            tOrV_bf16 = None
            tSrQ_bf16 = None
            tSrQ_bf16_stages = (None, None)
        if const_expr(self.transpose_s):
            tSrQ_terms = tuple(
                tiled_mma_qk.make_fragment_B(sQ_term)
                for sQ_term in sQ_terms
            )
            tSrK = tiled_mma_qk.make_fragment_A(sK)
        else:
            tSrQ_terms = tuple(
                tiled_mma_qk.make_fragment_A(sQ_term)
                for sQ_term in sQ_terms
            )
            tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        tSrQ_term_stages = tuple(
            (
                (
                    tSrQ[None, None, None, 0],
                    tSrQ[None, None, None, 1],
                )
                if const_expr(self.q_stage == 2)
                else (
                    tSrQ[None, None, None, 0],
                    tSrQ[None, None, None, 0],
                )
            )
            for tSrQ in tSrQ_terms
        )

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        if const_expr(self.stacked_query_terms):
            # Retain the full-width operand layouts for TMA and BF16 paths.
            # This op consumes their populated prefix and writes exactly the
            # two score ranges that the FP4 softmax joins.
            assert qk_mma_op.shape_mnk == (128, 128, 64)
            narrow_shape = (128, 2 * self.transposed_query_rows, 64)
            if const_expr(self.is_mxfp4):
                qk_mma_op = MmaMXF4Op(
                    narrow_shape, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM
                )
            else:
                qk_mma_op = MmaMXF4NVF4Op(
                    self.sf_dtype,
                    narrow_shape,
                    tcgen05.CtaGroup.ONE,
                    tcgen05.OperandSource.SMEM,
                )
        # Q is bound here because it is fixed for the whole tile; K arrives per
        # KV stage and is supplied at the call site through ``qk_gemm``. Which
        # operand each one is depends on whether S is transposed, so bind Q by
        # keyword and let the caller name the K side the same way either way.
        def _qk_scales(term, stage):
            if const_expr(self.transpose_s):
                return dict(
                    tScaleA=tCtSFKs[stage],
                    tScaleB=tCtSFQ_terms[term][stage],
                )
            return dict(
                tScaleA=tCtSFQ_terms[term][stage],
                tScaleB=tCtSFKs[stage],
            )

        def _make_gemm_si(term, stage):
            common = dict(zero_init=(term == 0))
            if const_expr(self.transpose_s):
                common.update(
                    tCrB=tSrQ_term_stages[term][stage],
                    sB=sQ_terms[term][None, None, None, stage],
                )
                positional = ()
            else:
                positional = (tSrQ_term_stages[term][stage],)
                common.update(
                    sA=sQ_terms[term][None, None, None, stage]
                )
            if const_expr(self.quant_qk):
                return partial(
                    sm100_utils.gemm_ptx_partial_fp4,
                    qk_mma_op,
                    self.tmem_s_offset[stage],
                    *positional,
                    **common,
                    **_qk_scales(term, stage),
                )
            return partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op,
                self.tmem_s_offset[stage],
                *positional,
                **common,
            )

        gemm_Si_terms = [
            [
                _make_gemm_si(term, stage)
                for stage in range(self.q_stage)
            ]
            for term in range(self.query_terms)
        ]

        def _make_gemm_pi(stage):
            acc_offset = self.tmem_o_offset[stage if self.q_stage == 2 else 0]
            if const_expr(self.narrow_pv):
                # O^T = V^T P^T. Both operands keep their shared-memory tiles
                # and K-major layouts; the instruction reads the first pv_rows
                # rows of P, the only ones softmax fills.
                narrow_shape = (self.m_block_size, self.pv_rows, 64)
                if const_expr(self.is_mxfp4):
                    narrow_pv_op = MmaMXF4Op(
                        narrow_shape, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM
                    )
                else:
                    narrow_pv_op = MmaMXF4NVF4Op(
                        self.sf_dtype,
                        narrow_shape,
                        tcgen05.CtaGroup.ONE,
                        tcgen05.OperandSource.SMEM,
                    )

                def gemm_pv_narrow(tCrB, sB, zero_init):
                    sm100_utils.gemm_ptx_partial_fp4(
                        narrow_pv_op,
                        acc_offset,
                        tCrB,
                        tOrPs[stage],
                        sA=sB,
                        sB=sP[None, None, None, 0],
                        tScaleA=tCtSFVs[stage],
                        tScaleB=tCtSFPs[stage],
                        zero_init=zero_init,
                    )

                return gemm_pv_narrow
            if const_expr(self.transpose_s):
                # A from shared memory: no TMEM address, and the split wait the
                # TMEM path uses is not available here.
                return partial(
                    sm100_utils.gemm_ptx_partial_fp4,
                    pv_mma_op,
                    acc_offset,
                    tOrPs[stage],
                    sA=sP[None, None, None, 0],
                    tScaleA=tCtSFPs[stage],
                    tScaleB=tCtSFVs[stage],
                )
            if const_expr(self.quant_pv):
                return partial(
                    sm100_utils.gemm_ptx_partial_fp4,
                    pv_mma_op,
                    acc_offset,
                    tOrPs[stage],
                    sA=None,
                    tScaleA=tCtSFPs[stage],
                    tScaleB=tCtSFVs[stage],
                    pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])),
                    tA_addr=self.tmem_p_offset[stage],
                )
            return partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                acc_offset,
                tOrPs[stage],
                sA=None,
                pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])),
            )

        gemm_Pi = [_make_gemm_pi(stage) for stage in range(self.q_stage)]
        if const_expr(self.fused_residual_first_block):
            qk_mma_op_bf16 = tiled_mma_qk_bf16.op
            pv_mma_op_bf16 = tiled_mma_pv_bf16.op
            def _make_gemm_si_bf16(stage):
                # Q is bound here and K supplied at the call site, mirroring the
                # FP4 builders; transposed, Q is the B operand.
                if const_expr(self.transpose_s):
                    return partial(
                        sm100_utils.gemm_ptx_partial,
                        qk_mma_op_bf16,
                        self.tmem_s_offset[stage],
                        tCrB=tSrQ_bf16_stages[stage],
                        sB=sQ_bf16[None, None, None, 0],
                        zero_init=True,
                    )
                return partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op_bf16,
                    self.tmem_s_offset[stage],
                    tSrQ_bf16_stages[stage],
                    sA=sQ_bf16[None, None, None, 0],
                    zero_init=True,
                )

            gemm_Si_bf16 = [_make_gemm_si_bf16(stage) for stage in range(self.q_stage)]
            gemm_S_sink_bf16 = (
                _make_gemm_si_bf16(1)
                if const_expr(self.fused_sink_block)
                else None
            )
            def _make_gemm_pi_bf16(stage, operand=None):
                tOrP_bf16 = tOrPs_bf16[stage] if operand is None else operand
                if const_expr(self.narrow_pv):
                    # The FP4 blocks accumulate O^T, so the residual's PV does
                    # too: V^T is an MN-major A operand over the head dimension.
                    narrow_pv_op_bf16 = tcgen05.MmaF16BF16Op(
                        pv_mma_op_bf16.a_dtype,
                        pv_mma_op_bf16.acc_dtype,
                        (self.m_block_size, self.pv_rows, 16),
                        tcgen05.CtaGroup.ONE,
                        tcgen05.OperandSource.SMEM,
                        pv_mma_op_bf16.b_major_mode,
                        pv_mma_op_bf16.a_major_mode,
                    )
                    acc_offset = self.tmem_o_offset[stage if self.q_stage == 2 else 0]
                    k_blocks = cute.size(tOrP_bf16.shape[2])

                    def gemm_pv_bf16_narrow(tCrB, sB, zero_init):
                        # The instruction count follows the A operand, so V is
                        # cut to as many K blocks as this P carries: one for
                        # the sink.
                        layout = tCrB.layout
                        tCrA = tCrB
                        if const_expr(k_blocks != cute.size(layout.shape[2])):
                            assert k_blocks == 1
                            tCrA = cute.make_tensor(
                                tCrB.iterator,
                                cute.make_layout(
                                    (layout.shape[0], layout.shape[1], 1),
                                    stride=(layout.stride[0], layout.stride[1], 0),
                                ),
                            )
                        sm100_utils.gemm_ptx_partial(
                            narrow_pv_op_bf16,
                            acc_offset,
                            tCrA,
                            tOrP_bf16,
                            sA=sB,
                            sB=sP_bf16[None, None, None, 0],
                            zero_init=zero_init,
                        )

                    return gemm_pv_bf16_narrow
                if const_expr(self.transpose_s):
                    # A from shared memory: no split wait, as on the FP4 side.
                    return partial(
                        sm100_utils.gemm_ptx_partial,
                        pv_mma_op_bf16,
                        self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                        tOrP_bf16,
                        sA=sP_bf16[None, None, None, 0],
                    )
                return partial(
                    sm100_utils.gemm_ptx_partial,
                    pv_mma_op_bf16,
                    self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                    tOrP_bf16,
                    sA=None,
                    pre_mbar_tiles=self.mbar_p_split(cute.size(tOrP_bf16.shape[2])),
                )

            gemm_Pi_bf16 = [_make_gemm_pi_bf16(stage) for stage in range(self.q_stage)]
            gemm_P_sink_bf16 = (
                _make_gemm_pi_bf16(0, tOrPs_bf16[1])
                if const_expr(self.fused_sink_block)
                else None
            )
        else:
            gemm_Si_bf16 = None
            gemm_S_sink_bf16 = None
            gemm_Pi_bf16 = None
            gemm_P_sink_bf16 = None

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        residual_kv_full_phase = Int32(0)
        residual_k_full_phase = Int32(0)
        residual_v_full_phase = Int32(0)
        residual_q_full_phase = Int32(0)
        bf16_p_full_phase = Int32(0)
        residual_kv_empty_phase = Int32(1)

        # Partition for S2T copy of SFQ/SFK - change addr per q_stage to avoid overwriting
        if const_expr(self.quant_qk):
            tiled_copy_s2t_sfq_terms_staged = [
                [
                    self.mainloop_s2t_copy_and_partition(
                        sSFQ_terms[term],
                        tCtSFQ_terms[term][stage],
                    )
                    for stage in range(self.q_stage)
                ]
                for term in range(self.query_terms)
            ]
            tiled_copy_s2t_sfk_staged = [
                self.mainloop_s2t_copy_and_partition(sSFK, tCtSFKs[stage])
                for stage in range(self.q_stage)
            ]
            (
                tiled_copy_s2t_sfq_0,
                tCsSFQ_compact_s2t_0,
                _,
            ) = tiled_copy_s2t_sfq_terms_staged[0][0]
            if const_expr(self.query_terms == 2 and not self.stacked_query_terms):
                (
                    tiled_copy_s2t_sfq_1,
                    tCsSFQ_compact_s2t_1,
                    _,
                ) = tiled_copy_s2t_sfq_terms_staged[1][0]
            else:
                tiled_copy_s2t_sfq_1 = tiled_copy_s2t_sfq_0
                tCsSFQ_compact_s2t_1 = tCsSFQ_compact_s2t_0
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, _ = tiled_copy_s2t_sfk_staged[0]
        else:
            # Dummy values when quant_qk is False - these won't be used
            tiled_copy_s2t_sfq_terms_staged = []
            tiled_copy_s2t_sfk_staged = []
            tiled_copy_s2t_sfq_0 = None
            tCsSFQ_compact_s2t_0 = None
            tiled_copy_s2t_sfq_1 = None
            tCsSFQ_compact_s2t_1 = None
            tiled_copy_s2t_sfk = None
            tCsSFK_compact_s2t = None

        if const_expr(self.quant_pv):
            tiled_copy_s2t_sfv_staged = [
                self.mainloop_s2t_copy_and_partition(sSFV, tCtSFVs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, _ = tiled_copy_s2t_sfv_staged[0]
            # S2T copy setup for SFP (P scale factors, computed by softmax warp via R2S)
            tiled_copy_s2t_sfp_staged = [
                self.mainloop_s2t_copy_and_partition(sSFP, tCtSFPs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, _ = tiled_copy_s2t_sfp_staged[0]
        else:
            tiled_copy_s2t_sfv_staged = []
            tiled_copy_s2t_sfv = None
            tCsSFV_compact_s2t = None
            tiled_copy_s2t_sfp_staged = []
            tiled_copy_s2t_sfp = None
            tCsSFP_compact_s2t = None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()


        mma_sfqk_producer_phase = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = self.split_block_range(
                    block_info, seqlen, m_block, split_idx, num_splits
                )
                block_iter_count = n_block_max - n_block_min
                process_tile = self.tile_has_work(
                    n_block_min, n_block_max, split_idx, num_splits
                )

            if process_tile:
                if const_expr(runs_tail):
                    if const_expr(self.fused_sink_block):
                        # The rolling residual seeds O before the FP4 pages.
                        # The permanent sink is deliberately deferred until the
                        # page loop has accumulated O at page-local FP4-P scale.
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_residual_kv_full_offset + 0,
                            residual_k_full_phase,
                        )
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_residual_kv_full_offset + 2,
                            residual_q_full_phase,
                        )
                        self.qk_gemm(
                            gemm_Si_bf16[0],
                            tSrK_bf16[None, None, None, 0],
                            sK_bf16[None, None, None, 0],
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(
                                mbar_ptr + self.mbar_bf16_S_full_offset
                            )
                            cute.arch.mbarrier_arrive(
                                mbar_ptr
                                + self.mbar_residual_kv_empty_offset
                                + 0
                            )
                        residual_k_full_phase ^= 1
                        residual_q_full_phase ^= 1

                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_residual_kv_full_offset + 1,
                            residual_v_full_phase,
                        )
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_bf16_P_full_offset,
                            bf16_p_full_phase,
                        )
                        gemm_Pi_bf16[0](
                            tCrB=tOrV_bf16[None, None, None, 0],
                            sB=sV_bf16[None, None, None, 0],
                            zero_init=True,
                        )
                        # PV consumes the aliased P/V tiles asynchronously; let
                        # the Tensor Core completion release the producer slots.
                        with cute.arch.elect_one():
                            tcgen05.commit(
                                mbar_ptr
                                + self.mbar_residual_kv_empty_offset
                                + 1
                            )
                            tcgen05.commit(
                                mbar_ptr
                                + self.mbar_residual_kv_empty_offset
                                + 2
                            )
                        residual_v_full_phase ^= 1
                        bf16_p_full_phase ^= 1
                    else:
                        # Wait for K and Q (V is consumed later in the PV phase, but
                        # the QK GEMM only needs K + Q here).
                        iket.range_push("mma_res_wait_k")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_residual_kv_full_offset + 0,
                            residual_kv_full_phase,
                        )
                        iket.range_pop()
                        iket.range_push("mma_res_wait_q")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_residual_kv_full_offset + 2,
                            residual_kv_full_phase,
                        )
                        iket.range_pop()
                        self.qk_gemm(
                            gemm_Si_bf16[0],
                            tSrK_bf16[None, None, None, 0],
                            sK_bf16[None, None, None, 0],
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_bf16_S_full_offset)
                        # Wait for V (consumed in BF16 GEMM_PV below).
                        iket.range_push("mma_res_wait_v")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_residual_kv_full_offset + 1,
                            residual_kv_full_phase,
                        )
                        iket.range_pop()
                        iket.range_push("mma_res_wait_p")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_bf16_P_full_offset,
                            Int32(0),
                        )
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_bf16_P_full_2_offset,
                            Int32(0),
                        )
                        iket.range_pop()
                        # BF16 PV: P_BF16 (TMEM) × V_BF16 (SMEM) → O TMEM (zero_init=True).
                        gemm_Pi_bf16[0](
                            tCrB=tOrV_bf16[None, None, None, 0],
                            sB=sV_bf16[None, None, None, 0],
                            zero_init=True,
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                mbar_ptr + self.mbar_residual_kv_empty_offset + 0,
                            )
                            cute.arch.mbarrier_arrive(
                                mbar_ptr + self.mbar_residual_kv_empty_offset + 1,
                            )
                            cute.arch.mbarrier_arrive(
                                mbar_ptr + self.mbar_residual_kv_empty_offset + 2,
                            )
                        residual_kv_full_phase ^= 1
                if const_expr(runs_pages):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        iket.range_push("mma_wait_qk")
                        if const_expr(self.q_replicate > 1 or self.stacked_query_terms):
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_q_fp4_ready_offset + stage,
                                mma_q_consumer_phase,
                            )
                        else:
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_load_q_full_offset + stage,
                                mma_q_consumer_phase,
                            )
                        # 2. wait for K0
                        if const_expr(stage == 0):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        iket.range_pop()
                        tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                        # We don't need to acquire empty S0 / S1.
                        # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                        # are empty. For subsequent iterations, the wait happened at the end
                        # of the while loop.
                    
                        # Copy SFQ 
                        # only tmem changes per q_stage.
                        if const_expr(self.quant_qk):
                            sm100_utils.tcgen05_after_thread_sync()
                            iket.range_push("mma_wait_sf")
                            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + stage, mma_sfqk_producer_phase)
                            iket.range_pop()
                            _, _, tCtSFQ_compact_s2t_0 = (
                                tiled_copy_s2t_sfq_terms_staged[0][stage]
                            )
                            cute.copy(
                                tiled_copy_s2t_sfq_0,
                                tCsSFQ_compact_s2t_0[
                                    None, None, None, None, stage
                                ],
                                tCtSFQ_compact_s2t_0,
                            )
                            if const_expr(self.query_terms == 2 and not self.stacked_query_terms):
                                _, _, tCtSFQ_compact_s2t_1 = (
                                    tiled_copy_s2t_sfq_terms_staged[1][stage]
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfq_1,
                                    tCsSFQ_compact_s2t_1[
                                        None, None, None, None, stage
                                    ],
                                    tCtSFQ_compact_s2t_1,
                                )

                            _, _, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[stage]
                            tCsSFK_compact_s2t_staged = tCsSFK_compact_s2t[None, None, None, None, mma_kv_consumer_state.index]
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t_staged,
                                tCtSFK_compact_s2t,
                            )

                        # 3. gemm
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                        sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(
                                sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                            )

                        self.qk_gemm(
                            gemm_Si_terms[0][stage],
                            tSrKi,
                            sK_cur,
                        )
                        if const_expr(self.query_terms == 2 and not self.stacked_query_terms):
                            self.qk_gemm(
                                gemm_Si_terms[1][stage],
                                tSrKi,
                                sK_cur,
                            )

                        # 4. release S0 / S1
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                    mma_q_consumer_phase ^= 1
                    if const_expr(self.quant_qk):
                        mma_sfqk_producer_phase ^= 1
                    # 5. release K0
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM (Q1 * K0 -> S1)
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop

                    block_loop_count = block_iter_count - 1
                    # A tail ahead of the pages has already written O.
                    O_should_accumulate = runs_tail
                    for i in cutlass.range(block_loop_count, unroll=1):
                        # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                        # 1. wait for V0
                        iket.range_push("mma_wait_v")
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        iket.range_pop()
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        tOrVi = tOrV[None, None, None, Vi_index]
                        for stage in cutlass.range_constexpr(self.q_stage):
                            def issue_pv():
                                # 2. acquire corrected O0/O1_partial and P0 / P1
                                # For the first iteration in this work tile, waiting for O0/O1_partial
                                # means that the correction warps has finished reading tO during
                                # the last iteration of the previous work tile has finished.
                                iket.range_push("mma_wait_p")
                                cute.arch.mbarrier_wait(
                                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                                    P_full_O_rescaled_phase,
                                )
                                iket.range_pop()
                                # No need for mbar.wait because it depends on the same Si as the prev qk mma
                                if const_expr(self.quant_pv):
                                    # S2T copy SFP: sSFP (smem) -> tCtSFPs (tmem)
                                    _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
                                    tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
                                    cute.copy(
                                        tiled_copy_s2t_sfp,
                                        tCsSFP_compact_s2t_cur,
                                        tCtSFP_compact_s2t,
                                    )
                                    # S2T copy SFV: sSFV (smem) -> tCtSFVs (tmem)
                                    _, _, tCtSFV_compact_s2t = tiled_copy_s2t_sfv_staged[stage]
                                    tCsSFV_compact_s2t_staged = tCsSFV_compact_s2t[None, None, None, None, Vi_index]
                                    cute.copy(
                                        tiled_copy_s2t_sfv,
                                        tCsSFV_compact_s2t_staged,
                                        tCtSFV_compact_s2t,
                                    )

                                sV_cur = sV[None, None, None, Vi_index]
                                if const_expr(self.uneven_kv_smem):
                                    sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)

                                self.pv_gemm(
                                    gemm_Pi[stage],
                                    tOrVi,
                                    sV_cur,
                                    self.p_scale_two_level or not O_should_accumulate,
                                    mbar_ptr,
                                    stage,
                                    P_full_O_rescaled_phase,
                                )
                                self.release_shared_p(mbar_ptr, stage)

                                if const_expr(stage == self.q_stage - 1):
                                    pipeline_kv.consumer_release(mma_kv_release_state)
                                    mma_kv_release_state.advance()
                                # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                            def issue_qk():
                                # GEMM_QK0i (Q0 * Ki -> S0)
                                # 1. wait for Ki
                                if const_expr(stage == 0):
                                    mma_kv_consumer_state.advance()
                                    iket.range_push("mma_wait_k")
                                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                                    iket.range_pop()
                                Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                                # The wait below is what makes overwriting Si safe: the
                                # softmax arrives on it once its tensor-memory load has
                                # retired, which also implies the previous QK finished.
                                if const_expr(self.quant_qk):
                                    sm100_utils.tcgen05_after_thread_sync()
                                    iket.range_push("mma_wait_sf")
                                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + stage, mma_sfqk_producer_phase)
                                    iket.range_pop()
                                    # One Q stage keeps its scales in the unused
                                    # S stage throughout the page loop. The first
                                    # page loads them again for every new tile;
                                    # the sink reuses those columns only after the loop.
                                    # Two Q stages overwrite each other's scales.
                                    if const_expr(self.q_stage == 2):
                                        _, _, tCtSFQ_compact_s2t_0 = (
                                            tiled_copy_s2t_sfq_terms_staged[0][stage]
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfq_0,
                                            tCsSFQ_compact_s2t_0[
                                                None, None, None, None, stage
                                            ],
                                            tCtSFQ_compact_s2t_0,
                                        )
                                        if const_expr(self.query_terms == 2 and not self.stacked_query_terms):
                                            _, _, tCtSFQ_compact_s2t_1 = (
                                                tiled_copy_s2t_sfq_terms_staged[1][stage]
                                            )
                                            cute.copy(
                                                tiled_copy_s2t_sfq_1,
                                                tCsSFQ_compact_s2t_1[
                                                    None, None, None, None, stage
                                                ],
                                                tCtSFQ_compact_s2t_1,
                                            )
                                    _, _, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[stage]
                                    tCsSFK_compact_s2t_staged = tCsSFK_compact_s2t[None, None, None, None, mma_kv_consumer_state.index]
                                    cute.copy(
                                        tiled_copy_s2t_sfk,
                                        tCsSFK_compact_s2t_staged,
                                        tCtSFK_compact_s2t,
                                    )

                                sK_cur = sK[None, None, None, Ki_index]
                                if const_expr(self.uneven_kv_smem):
                                    sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)

                                self.qk_gemm(
                                    gemm_Si_terms[0][stage],
                                    tSrK[None, None, None, Ki_index],
                                    sK_cur,
                                )
                                if const_expr(self.query_terms == 2 and not self.stacked_query_terms):
                                    self.qk_gemm(
                                        gemm_Si_terms[1][stage],
                                        tSrK[None, None, None, Ki_index],
                                        sK_cur,
                                    )
                                # Release Si. The commit fires once every MMA this
                                # warp issued before it has retired, so it belongs
                                # next to the QK: leaving it after the PV would make
                                # the softmax wait for a GEMM it does not read.
                                with cute.arch.elect_one():
                                    tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)

                            if const_expr(self.tmem_sfpv_offset is not None):
                                # Nothing the next QK overwrites is still live. P is in
                                # shared memory here and its scale factors have tensor
                                # memory of their own, so S is the only thing left in
                                # these columns and the softmax released it at the top
                                # of its step. Issuing the QK before waiting on P is
                                # what lets it run underneath the softmax instead of
                                # after it, which is the whole of the block's exposed
                                # GEMM latency.
                                issue_qk()
                                issue_pv()
                            else:
                                # P and its scale factors share S's columns, so the
                                # next QK cannot be issued until this PV has consumed
                                # them.
                                issue_pv()
                                issue_qk()
                        # 4. release Ki
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        P_full_O_rescaled_phase ^= 1
                        if const_expr(self.quant_qk):
                            mma_sfqk_producer_phase ^= 1
                        O_should_accumulate = True
                    # End of seqlen_kv loop

                    # release Q0 & Q1
                    with cute.arch.elect_one():
                        for stage in cutlass.range_constexpr(self.q_stage):
                            tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    iket.range_push("mma_wait_v")
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    iket.range_pop()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected Oi_partial and Pi
                        iket.range_push("mma_wait_p")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase
                        )
                        iket.range_pop()
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)

                        if const_expr(self.quant_pv):
                            # S2T copy SFP: sSFP (smem) -> tCtSFPs (tmem)
                            _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
                            tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_compact_s2t_cur,
                                tCtSFP_compact_s2t,
                            )
                            # S2T copy SFV: sSFV (smem) -> tCtSFVs (tmem)
                            _, _, tCtSFV_compact_s2t = tiled_copy_s2t_sfv_staged[stage]
                            tCsSFV_compact_s2t_staged = tCsSFV_compact_s2t[None, None, None, None, Vi_index]
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_compact_s2t_staged,
                                tCtSFV_compact_s2t,
                            )
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        _zi_post = self.p_scale_two_level or not O_should_accumulate
                        self.pv_gemm(
                            gemm_Pi[stage],
                            tOrVi,
                            sV_cur,
                            _zi_post,
                            mbar_ptr,
                            stage,
                            P_full_O_rescaled_phase,
                        )
                        self.release_shared_p(mbar_ptr, stage)
                        # 4. release accumulated O0_partial
                        # We do need O_full here since for the last tile, by the time the softmax warp
                        # has signaled to the correction warps, the softmax warp has just finished compute
                        # the row sum of the current tile. It does not guarantee that the 1st tile
                        # of the next work tile has been computed yet.
                        if const_expr(not runs_sink):
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        if const_expr(self.transpose_s and self.is_persistent):
                            # No later page observes this tile's final P phase.
                            # Close it before another persistent tile can reuse
                            # the barrier and overwrite the shared P tile.
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_P_consumed_offset + stage,
                                P_full_O_rescaled_phase,
                            )
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                    P_full_O_rescaled_phase ^= 1
                    # 5. release Vi_end
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)
                elif const_expr(not runs_sink):
                    # The tail's PV is the last GEMM of a split with no page
                    # and no sink.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset)

                if const_expr(runs_sink):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_residual_kv_full_offset + 0,
                        residual_k_full_phase,
                    )
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_residual_kv_full_offset + 2,
                        residual_q_full_phase,
                    )
                    self.qk_gemm(
                        gemm_S_sink_bf16,
                        tSrK_bf16[None, None, None, 0],
                        sK_bf16[None, None, None, 0],
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset)
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            + self.mbar_residual_kv_empty_offset
                            + 0
                        )
                    residual_k_full_phase ^= 1
                    residual_q_full_phase ^= 1

                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_residual_kv_full_offset + 1,
                        residual_v_full_phase,
                    )
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_bf16_P_full_offset,
                        bf16_p_full_phase,
                    )
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset,
                        P_full_O_rescaled_phase,
                    )
                    gemm_P_sink_bf16(
                        tCrB=tOrV_bf16[None, None, None, 0],
                        sB=sV_bf16[None, None, None, 0],
                        zero_init=self.p_scale_two_level and runs_pages,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            mbar_ptr + self.mbar_O_full_offset
                        )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            + self.mbar_residual_kv_empty_offset
                            + 1
                        )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr
                            + self.mbar_residual_kv_empty_offset
                            + 2
                        )
                    residual_v_full_phase ^= 1
                    bf16_p_full_phase ^= 1
                    P_full_O_rescaled_phase ^= 1


            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop


    @cute.jit
    def softmax_warpgroup_reduce(
        self,
        vals: cute.Tensor,
        sSoftmaxRed: cute.Tensor,
        tidx: Int32,
        red_buf: Int32,
        op: Callable,
        intra_warp: cutlass.Constexpr[bool] = True,
        row_group: cutlass.Constexpr[int] = 0,
    ) -> Int32:
        """Reduce one value per query row across all softmax threads, in place.

        Transposed, a row of S runs along the thread axis, so a row reduction is
        a warp butterfly followed by an exchange between the four softmax warps.
        The scratch is double buffered by ``red_buf`` so a single named barrier
        per call is enough: a warp can only overwrite the buffer it read two
        calls ago, and the intervening barrier already ordered that read.

        ``intra_warp`` is for callers that already hold the warp-wide result
        because they needed one of the butterfly's intermediate stages anyway.
        """
        num_warps = const_expr(len(self.softmax0_warp_ids))
        rows = const_expr(self.softmax_rows_per_group)
        if const_expr(intra_warp):
            for j in cutlass.range_constexpr(rows):
                vals[j] = utils.warp_reduce(vals[j], op)
        base = const_expr(row_group * 2 * num_warps * rows) + red_buf * (num_warps * rows)
        if tidx % cute.arch.WARP_SIZE == 0:
            warp = tidx // cute.arch.WARP_SIZE
            for j in cutlass.range_constexpr(rows):
                sSoftmaxRed[base + warp * rows + j] = vals[j]
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.SoftmaxReduce) + row_group,
            number_of_threads=num_warps * cute.arch.WARP_SIZE,
        )
        for j in cutlass.range_constexpr(rows):
            acc = sSoftmaxRed[base + j]
            for w in cutlass.range_constexpr(1, num_warps):
                acc = op(acc, sSoftmaxRed[base + w * rows + j])
            vals[j] = acc
        return red_buf ^ 1

    @cute.jit
    def softmax_wait_corr(
        self, mbar_ptr: cute.Pointer, stage: int | Int32, si_corr_producer_phase: Int32
    ) -> Int32:
        """Wait until correction has taken the factors last published in
        sScale, and return the producer phase for the next wait."""
        iket.range_push("sm_wait_corr")
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )
        iket.range_pop()
        return si_corr_producer_phase ^ 1

    def scatter_lane(self, row: int) -> int:
        """The lane within a cluster of ``softmax_rows_per_group`` that ends a
        reduce-scatter holding ``row``: the row's bits in reverse order, since
        the first butterfly step splits the rows in halves."""
        steps = self.softmax_rows_per_group.bit_length() - 1
        return int(format(row, f"0{steps}b")[::-1], 2) if steps else 0

    @cute.jit
    def scatter_row(self, tidx: Int32) -> Int32:
        """The row a lane holds after the reduce-scatter, ``scatter_lane``
        inverted."""
        rows = const_expr(self.softmax_rows_per_group)
        row = Int32(0)
        for i in cutlass.range_constexpr(rows.bit_length() - 1):
            row += ((tidx >> i) & 1) * (rows >> (i + 1))
        return row

    @cute.jit
    def softmax_warpgroup_reduce_scattered(
        self,
        val: Float32,
        own_row: Int32,
        block_max: cute.Tensor,
        sSoftmaxRed: cute.Tensor,
        tidx: Int32,
        red_buf: Int32,
        row_group: cutlass.Constexpr[int] = 0,
    ) -> Int32:
        """``softmax_warpgroup_reduce`` for a warp maximum reduce-scattered over
        lanes: the first lanes of each warp hold one row each, and every thread
        reads every row back, so ``block_max`` ends as the unscattered reduce
        leaves it. Same scratch, slots and barrier."""
        num_warps = const_expr(len(self.softmax0_warp_ids))
        rows = const_expr(self.softmax_rows_per_group)
        base = const_expr(row_group * 2 * num_warps * rows) + red_buf * (num_warps * rows)
        if tidx % cute.arch.WARP_SIZE < rows:
            warp = tidx // cute.arch.WARP_SIZE
            sSoftmaxRed[base + warp * rows + own_row] = val
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.SoftmaxReduce) + row_group,
            number_of_threads=num_warps * cute.arch.WARP_SIZE,
        )
        for j in cutlass.range_constexpr(rows):
            acc = sSoftmaxRed[base + j]
            for w in cutlass.range_constexpr(1, num_warps):
                acc = cute.arch.fmax(acc, sSoftmaxRed[base + w * rows + j])
            block_max[j] = acc
        return red_buf ^ 1

    @cute.jit
    def split_block_range(
        self,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32,
        num_splits: Int32,
    ) -> Tuple[Int32, Int32]:
        """The FP4 page blocks a split walks, ``[n_block_min, n_block_max)``.

        With a split of its own for the BF16 blocks, ``num_splits`` counts it
        too: the pages are divided among the others exactly as a launch with
        no BF16 block would divide them, and the last split is handed none.
        Every warp role must ask here, since a role that saw a different range
        would wait on barriers the others never arrive on.
        """
        if const_expr(not self.bf16_split):
            return block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, num_splits - 1
        )
        if split_idx == num_splits - 1:
            n_block_max = n_block_min
        return n_block_min, n_block_max

    @cute.jit
    def is_bf16_split(self, split_idx: Int32, num_splits: Int32):
        """Whether this is the split that merges the tail and the sink."""
        if const_expr(self.bf16_split):
            return split_idx == num_splits - 1
        return split_idx == self.residual_split_idx

    @cute.jit
    def tile_has_work(
        self,
        n_block_min: Int32,
        n_block_max: Int32,
        split_idx: Int32,
        num_splits: Int32,
    ):
        """Whether a work tile has anything for the pipeline to do.

        Without split-k every tile runs, including one whose fp4 length is
        zero: its single mainloop step is fully masked and the epilogue still
        owes an output row. Split-k drops empty splits instead, except for the
        one that owns the residual block, which has work whether or not any fp4
        block landed in it.

        Every warp asks this question separately and they have to agree, so
        they all ask it here.
        """
        if const_expr(not self.is_split_kv):
            return True
        has_work = n_block_min < n_block_max
        if const_expr(self.fused_residual_first_block):
            if self.is_bf16_split(split_idx, num_splits):
                has_work = cutlass.Boolean(True)
        return has_work

    @cute.jit
    def run_split_role(self, is_bf16_cta, role_body: Callable):
        """Run one warp role's body, specialised for what this CTA merges.

        ``role_body`` takes the :class:`SplitRole` as a compile-time keyword,
        so each CTA runs one of two separately compiled bodies when the BF16
        blocks have a split of their own, and the one combined body otherwise.
        A CTA owns a single tile, which is what keeps every barrier phase of
        one body independent of the other's.
        """
        if const_expr(self.bf16_split):
            if is_bf16_cta:
                role_body(role=SplitRole.BF16)
            else:
                role_body(role=SplitRole.PAGES)
        else:
            role_body(role=SplitRole.ALL)

    @cute.jit
    def zero_transposed_p(self, sP_uint8: cute.Tensor, sSFP: cute.Tensor, tidx: Int32):
        """Zero the shared P tile and its scale factors once per CTA.

        Only the first ``transposed_query_rows`` rows are ever written, but the
        MMA reads all 128 and an undefined E4M3 scale factor can decode to NaN.
        Zeroing the whole region is a permutation-invariant write, so the
        swizzle can be ignored and the buffers filled as flat words.
        """
        num_threads = const_expr(len(self.softmax0_warp_ids) * cute.arch.WARP_SIZE)
        p_words = const_expr(self.m_block_size * self.n_block_size // 2 // 4)
        sfp_words = const_expr(
            self.m_block_size * self.head_dim_v_padded // self.sf_vec_size * self.q_stage // 4
        )
        p_flat = cute.make_tensor(
            cute.recast_ptr(sP_uint8.iterator, dtype=Int32), cute.make_layout(p_words)
        )
        sfp_flat = cute.make_tensor(
            cute.recast_ptr(sSFP.iterator, dtype=Int32), cute.make_layout(sfp_words)
        )
        for i in cutlass.range_constexpr(p_words // num_threads):
            p_flat[tidx + i * num_threads] = Int32(0)
        for i in cutlass.range_constexpr(max(1, sfp_words // num_threads)):
            if tidx + i * num_threads < sfp_words:
                sfp_flat[tidx + i * num_threads] = Int32(0)

    @cute.jit
    def zero_transposed_p_bf16(
        self,
        sP_bf16: cute.Tensor,
        sP_bf16_mk: cute.Tensor,
        tidx: Int32,
        row_group: cutlass.Constexpr[int] = 0,
    ):
        """Zero the shared BF16 P tile of the residual block.

        The FP4 tile is cleared once per CTA, but this one shares its bytes
        with the residual Q tile and so cannot be cleared until the QK GEMM
        has consumed it. As above the write is permutation invariant, so the
        swizzle can be ignored and the buffer filled as flat words.

        Several row groups wake on the same S and nothing orders one group's
        clear against another's stores, so then each clears only its share of
        the rows past ``transposed_query_rows``: a group stores every kv
        position of its own rows, and no group stores into those.
        """
        groups = const_expr(self.softmax_row_groups)
        if const_expr(groups == 1):
            num_threads = const_expr(len(self.softmax0_warp_ids) * cute.arch.WARP_SIZE)
            words = const_expr(self.m_block_size * self.n_block_size * 2 // 4)
            flat = cute.make_tensor(
                cute.recast_ptr(sP_bf16.iterator, dtype=Int32), cute.make_layout(words)
            )
            for i in cutlass.range_constexpr(words // num_threads):
                flat[tidx + i * num_threads] = Int32(0)
        else:
            share = const_expr(
                (self.m_block_size - self.transposed_query_rows) // groups
            )
            first = const_expr(self.transposed_query_rows + row_group * share)
            for i in cutlass.range_constexpr(share):
                sP_bf16_mk[first + i, tidx] = cutlass.BFloat16(0.0)

    @cute.jit
    def softmax_residual_step_transposed(
        self,
        softmax_scale_log2: Float32,
        mbar_ptr: cute.Pointer,
        thr_tmem_load: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        frg_shape,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        sP_bf16: cute.Tensor,
        sP_bf16_mk: cute.Tensor,
        sSoftmaxRed: cute.Tensor,
        red_buf: Int32,
        tidx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        mResidualSeqUsedK: Optional[cute.Tensor],
        mSinkSeqUsedK: Optional[cute.Tensor] = None,
        seqused_fp4: Int32 = Int32(1),
        row_group: cutlass.Constexpr[int] = 0,
        pages_follow: cutlass.Constexpr[bool] = True,
    ) -> Int32:
        """The residual block in the same orientation as the FP4 blocks.

        Structurally ``softmax_step_transposed`` with the quantization replaced
        by a plain bf16 store, since P feeds a bf16 MMA and carries no scale
        factors. It runs before any FP4 block and seeds the online softmax, so
        ``row_max`` and ``row_sum`` leave here holding the residual's
        contribution and the FP4 blocks merge onto them with is_first=False.
        ``pages_follow`` is whether this CTA has FP4 blocks to run after it.
        """
        rows = const_expr(self.softmax_rows_per_group)
        row_base = const_expr(row_group * rows)

        iket.range_push("sm_res_wait_s")
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_bf16_S_full_offset, Int32(0))
        iket.range_pop()

        iket.range_push("sm_res_comp")
        # Safe here and only here: the QK GEMM has just released the Q tile
        # these bytes belong to. With one row group the row-maximum exchange
        # below carries the barrier that separates this clear from the stores
        # at the end; with more the groups clear disjoint rows that none of
        # them stores into.
        self.zero_transposed_p_bf16(sP_bf16, sP_bf16_mk, tidx, row_group=row_group)

        tSrS = cute.make_rmem_tensor(frg_shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS)
        cute.arch.fence_view_async_tmem_load()
        if const_expr(mSinkSeqUsedK is not None and pages_follow):
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset)

        # Transposed, thread ``tidx`` owns kv position ``tidx``, so the residual
        # length is one predicate. A split that does not own the residual sees a
        # length of zero, masks every position, and contributes nothing.
        seqused_k = Int32(self.n_block_size)
        residual_start = Int32(0)
        if const_expr(mResidualSeqUsedK is not None):
            seqused_k = mResidualSeqUsedK[batch_idx]
            if const_expr(self.is_split_kv and not self.bf16_split):
                if split_idx != self.residual_split_idx:
                    seqused_k = Int32(0)
        if const_expr(mSinkSeqUsedK is not None):
            sink_length = mSinkSeqUsedK[batch_idx]
            if seqused_fp4 == Int32(0):
                residual_start = sink_length
        if (tidx < residual_start) or (tidx >= seqused_k):
            for j in cutlass.range_constexpr(rows):
                tSrS[j] = -Float32.inf

        block_max = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        for j in cutlass.range_constexpr(rows):
            block_max[j] = utils.warp_reduce(tSrS[j], cute.arch.fmax)
        red_buf = self.softmax_warpgroup_reduce(
            block_max,
            sSoftmaxRed,
            tidx,
            red_buf,
            cute.arch.fmax,
            intra_warp=False,
            row_group=row_group,
        )
        max_scaled = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        for j in cutlass.range_constexpr(rows):
            row_max[j] = block_max[j]
            safe = block_max[j] if block_max[j] != -Float32.inf else 0.0
            max_scaled[j] = safe * softmax_scale_log2
            if const_expr(self.p_scale_rescale_log2):
                max_scaled[j] -= Float32(self.p_scale_rescale_log2)

        for j in cutlass.range_constexpr(rows):
            tSrS[j] = cute.arch.exp2(tSrS[j] * softmax_scale_log2 - max_scaled[j])
            row_sum[j] = tSrS[j]

        tSrP = cute.make_rmem_tensor(frg_shape, cutlass.BFloat16)
        tSrP.store(tSrS.load().to(cutlass.BFloat16))
        for j in cutlass.range_constexpr(rows):
            sP_bf16_mk[row_base + j, tidx] = tSrP[j]
        # The MMA reads P through the async proxy while these are generic-proxy
        # stores, so the mbarrier below needs the fence to make them visible.
        cute.arch.fence_view_async_shared()
        iket.range_pop()

        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_bf16_P_full_offset)
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_bf16_P_full_2_offset)
        return red_buf

    @cute.jit
    def softmax_sink_step_transposed(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        red_buf: Int32,
        softmax_scale_log2: Float32,
        mbar_ptr: cute.Pointer,
        thr_tmem_load: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStS_sink_t2r: cute.Tensor,
        frg_shape,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        sScale: cute.Tensor,
        sP_bf16: cute.Tensor,
        sP_bf16_mk: cute.Tensor,
        sSoftmaxRed: cute.Tensor,
        stage: int | Int32,
        tidx: Int32,
        batch_idx: Int32,
        mSinkSeqUsedK: cute.Tensor,
        split_idx: Int32,
        row_group: cutlass.Constexpr[int] = 0,
        waits_for_previous: cutlass.Constexpr[bool] = True,
    ) -> Tuple[Int32, Int32, Int32]:
        """Merge the permanent BF16 sink after every FP4 page.

        ``waits_for_previous`` as for ``softmax_step_transposed``: false only
        when nothing in the tile published before it.

        The sink's probabilities are kept in BF16, so its large score can
        rescale the FP32 output accumulator without forcing the FP4 page P
        tiles to quantize against the sink maximum.
        """
        rows = const_expr(self.softmax_rows_per_group)
        row_base = const_expr(row_group * rows)

        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_S_full_offset + stage,
            mma_si_consumer_phase,
        )
        tSrS = cute.make_rmem_tensor(frg_shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_sink_t2r, tSrS)
        cute.arch.fence_view_async_tmem_load()

        sink_length = mSinkSeqUsedK[batch_idx]
        if const_expr(self.is_split_kv and not self.bf16_split):
            # One sink, one owner, as with the tail: every other split merges
            # a length of zero, so the combine sees the sink exactly once.
            if split_idx != self.residual_split_idx:
                sink_length = Int32(0)
        if tidx >= sink_length:
            for j in cutlass.range_constexpr(rows):
                tSrS[j] = -Float32.inf

        block_max = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        for j in cutlass.range_constexpr(rows):
            block_max[j] = utils.warp_reduce(tSrS[j], cute.arch.fmax)
        red_buf = self.softmax_warpgroup_reduce(
            block_max,
            sSoftmaxRed,
            tidx,
            red_buf,
            cute.arch.fmax,
            intra_warp=False,
            row_group=row_group,
        )

        acc_scale = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        row_max_safe = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        for j in cutlass.range_constexpr(rows):
            row_max_old = row_max[j]
            row_max_new = cute.arch.fmax(row_max_old, block_max[j])
            safe = row_max_new if row_max_new != -Float32.inf else 0.0
            acc_scale_ = (row_max_old - safe) * softmax_scale_log2
            scale = utils.exp2f(acc_scale_)
            if const_expr(RESCALE_THRESHOLD > 0.0):
                if acc_scale_ >= -RESCALE_THRESHOLD:
                    row_max_new = row_max_old
                    safe = row_max_old
                    scale = Float32(1.0)
            row_max[j] = row_max_new
            row_max_safe[j] = safe
            acc_scale[j] = scale

        if const_expr(waits_for_previous):
            si_corr_producer_phase = self.softmax_wait_corr(
                mbar_ptr, stage, si_corr_producer_phase
            )
        if tidx == 0:
            for j in cutlass.range_constexpr(rows):
                sScale[row_base + j + stage * self.m_block_size] = acc_scale[j]
                if const_expr(self.p_scale_two_level):
                    sScale[row_base + j + stage * self.m_block_size + self.m_block_size * 2] = 1.0
        cute.arch.mbarrier_arrive(
            mbar_ptr + self.mbar_softmax_corr_full_offset + stage
        )

        max_scaled = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        for j in cutlass.range_constexpr(rows):
            max_scaled[j] = row_max_safe[j] * softmax_scale_log2
            if const_expr(self.p_scale_rescale_log2):
                max_scaled[j] -= Float32(self.p_scale_rescale_log2)
            tSrS[j] = cute.arch.exp2(tSrS[j] * softmax_scale_log2 - max_scaled[j])
            row_sum[j] = row_sum[j] * acc_scale[j] + tSrS[j]

        tSrP = cute.make_rmem_tensor(frg_shape, cutlass.BFloat16)
        tSrP.store(tSrS.load().to(cutlass.BFloat16))
        for j in cutlass.range_constexpr(rows):
            sP_bf16_mk[row_base + j, tidx] = tSrP[j]
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            mbar_ptr + self.mbar_bf16_P_full_offset
        )
        cute.arch.mbarrier_arrive(
            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
        )
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase, red_buf

    @cute.jit
    def softmax_loop_transposed(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        thr_mma_qk: cute.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        sSFP: cute.Tensor,
        sP_uint8: cute.Tensor,
        sSoftmaxRed: cute.Tensor,
        tidx: Int32,
        row_group: cutlass.Constexpr[int] = 0,
        mResidualSeqUsedK: Optional[cute.Tensor] = None,
        mSinkSeqUsedK: Optional[cute.Tensor] = None,
        sP_bf16: Optional[cute.Tensor] = None,
        sP_bf16_mk: Optional[cute.Tensor] = None,
        role: cutlass.Constexpr = SplitRole.ALL,
    ):
        """Softmax over an S tile held as (kv position, query row).

        Thread ``tidx`` owns kv position ``tidx`` of the block and the
        ``transposed_query_rows`` columns that carry a query, so it runs that
        many exponentials instead of a whole row of keys. The row maximum is
        along kv and therefore crosses threads; the row sum is deferred to the
        end of the tile because the running correction factor is the same in
        every thread and factors out of the accumulation.
        """
        runs_pages = const_expr(role != SplitRole.BF16)
        runs_tail = const_expr(
            self.fused_residual_first_block and role != SplitRole.PAGES
        )
        runs_sink = const_expr(self.fused_sink_block and role != SplitRole.PAGES)
        rows = const_expr(self.softmax_rows_per_group)
        row_base = const_expr(row_group * rows)
        # Only the leading columns of S carry a query; the rest of the MMA tile
        # is padding that the tensor-memory load simply never reads. A column of
        # S is a tensor-memory column, so a warpgroup reaches its own rows by
        # advancing the iterator past the group before it.
        tStS_group = cute.make_tensor(tStSi.iterator + row_base, tStSi.layout)
        tStS_live = cute.composition(tStS_group, cute.make_layout((self.m_block_size, rows)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS_live = cute.composition(tScS, cute.make_layout((self.m_block_size, rows)))
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(rows)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_live).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStS_live)
        tStS_second_t2r = None
        if const_expr(self.stacked_query_terms):
            tStS_second_group = cute.make_tensor(
                tStSi.iterator + self.transposed_query_rows + row_base,
                tStSi.layout,
            )
            tStS_second_live = cute.composition(
                tStS_second_group, cute.make_layout((self.m_block_size, rows))
            )
            tStS_second_t2r = thr_tmem_load.partition_S(tStS_second_live)
        tStS_sink_t2r = None
        if const_expr(self.fused_sink_block):
            sink_offset = self.tmem_s_offset[1] - self.tmem_s_offset[0]
            tStS_sink_group = cute.make_tensor(
                tStSi.iterator + sink_offset + row_base, tStSi.layout
            )
            tStS_sink_live = cute.composition(
                tStS_sink_group,
                cute.make_layout((self.m_block_size, rows)),
            )
            tStS_sink_t2r = thr_tmem_load.partition_S(tStS_sink_live)
        frg_shape = thr_tmem_load.partition_D(tScS_live).shape

        if const_expr(runs_pages):
            self.zero_transposed_p(sP_uint8, sSFP, tidx)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        red_buf = Int32(0)

        # The first FP4 QK may overwrite S once this arrives. Behind a tail
        # with a sink that is only after the tail's S has been read, so the
        # residual step arrives instead.
        if const_expr(runs_pages and not (runs_tail and self.fused_sink_block)):
            if stage == 1:
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + 0)
            if const_expr(self.q_stage == 1) and stage == 0:
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + 0)

        row_max = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        row_sum = cute.make_rmem_tensor(cute.make_layout(rows), Float32)

        softmax_step = partial(
            self.softmax_step_transposed,
            softmax_scale_log2=softmax_scale_log2,
            mbar_ptr=mbar_ptr,
            thr_tmem_load=thr_tmem_load,
            tStS_t2r=tStS_t2r,
            tStS_second_t2r=tStS_second_t2r,
            frg_shape=frg_shape,
            row_max=row_max,
            row_sum=row_sum,
            sScale=sScale,
            sSFP=sSFP,
            sP_uint8=sP_uint8,
            sSoftmaxRed=sSoftmaxRed,
            stage=stage,
            tidx=tidx,
            row_group=row_group,
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            iket.range_push("sm_seqinfo")
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = self.split_block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )
            iket.range_pop()
            # A split can be handed no kv block at all. Such a tile must stay out
            # of the correction handshake entirely, as on the untransposed path,
            # or the two warp groups fall out of step.
            has_work = self.tile_has_work(
                n_block_min, n_block_max, split_idx, num_splits
            )
            if has_work:
                row_max.fill(-Float32.inf)
                row_sum.fill(0.0)

                iket.range_push("sm_wait_corr")
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage,
                    si_corr_producer_phase,
                )
                iket.range_pop()
                si_corr_producer_phase ^= 1

                seeded = runs_tail
                if const_expr(runs_tail):
                    if const_expr(self.fused_sink_block):
                        red_buf = self.softmax_residual_step_transposed(
                            softmax_scale_log2=softmax_scale_log2,
                            mbar_ptr=mbar_ptr,
                            thr_tmem_load=thr_tmem_load,
                            tStS_t2r=tStS_t2r,
                            frg_shape=frg_shape,
                            row_max=row_max,
                            row_sum=row_sum,
                            sP_bf16=sP_bf16,
                            sP_bf16_mk=sP_bf16_mk,
                            sSoftmaxRed=sSoftmaxRed,
                            red_buf=red_buf,
                            tidx=tidx,
                            batch_idx=batch_idx,
                            mSinkSeqUsedK=mSinkSeqUsedK,
                            split_idx=split_idx,
                            mResidualSeqUsedK=mResidualSeqUsedK,
                            seqused_fp4=seqlen.seqlen_k,
                            row_group=row_group,
                            pages_follow=runs_pages,
                        )
                    else:
                        red_buf = self.softmax_residual_step_transposed(
                            softmax_scale_log2=softmax_scale_log2,
                            mbar_ptr=mbar_ptr,
                            thr_tmem_load=thr_tmem_load,
                            tStS_t2r=tStS_t2r,
                            frg_shape=frg_shape,
                            row_max=row_max,
                            row_sum=row_sum,
                            sP_bf16=sP_bf16,
                            sP_bf16_mk=sP_bf16_mk,
                            sSoftmaxRed=sSoftmaxRed,
                            red_buf=red_buf,
                            tidx=tidx,
                            batch_idx=batch_idx,
                            split_idx=split_idx,
                            mResidualSeqUsedK=mResidualSeqUsedK,
                            row_group=row_group,
                        )

                if const_expr(runs_pages):
                    sink_length = Int32(0)
                    if const_expr(self.fused_sink_block):
                        sink_length = mSinkSeqUsedK[batch_idx]
                    mma_si_consumer_phase, si_corr_producer_phase, red_buf = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        red_buf,
                        n_block_max - 1,
                        seqlen_k=seqlen.seqlen_k,
                        is_first=not seeded,
                        mask_seqlen=True,
                        sink_length=sink_length,
                        mask_sink=True,
                        waits_for_previous=False,
                    )
                    # Blocks run from the last page down, so page zero, whose
                    # sink duplicate is masked, can only be the first block or
                    # the last; the steps between carry no sink test.
                    peel_last = const_expr(self.fused_sink_block)
                    for n_tile in cutlass.range(
                        n_block_max - 1 - n_block_min - int(peel_last), unroll=1
                    ):
                        mma_si_consumer_phase, si_corr_producer_phase, red_buf = softmax_step(
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            red_buf,
                            n_block_max - 2 - n_tile,
                            seqlen_k=seqlen.seqlen_k,
                            is_first=False,
                            mask_seqlen=False,
                            sink_length=sink_length,
                            follows_pv=True,
                        )
                    if const_expr(peel_last):
                        if n_block_max - 1 > n_block_min:
                            mma_si_consumer_phase, si_corr_producer_phase, red_buf = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                red_buf,
                                n_block_min,
                                seqlen_k=seqlen.seqlen_k,
                                is_first=False,
                                mask_seqlen=False,
                                sink_length=sink_length,
                                follows_pv=True,
                                mask_sink=True,
                            )

                if const_expr(runs_sink):
                    (
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        red_buf,
                    ) = self.softmax_sink_step_transposed(
                        mma_si_consumer_phase=mma_si_consumer_phase,
                        si_corr_producer_phase=si_corr_producer_phase,
                        red_buf=red_buf,
                        softmax_scale_log2=softmax_scale_log2,
                        mbar_ptr=mbar_ptr,
                        thr_tmem_load=thr_tmem_load,
                        tStS_t2r=tStS_t2r,
                        tStS_sink_t2r=tStS_sink_t2r,
                        frg_shape=frg_shape,
                        row_max=row_max,
                        row_sum=row_sum,
                        sScale=sScale,
                        sP_bf16=sP_bf16,
                        sP_bf16_mk=sP_bf16_mk,
                        sSoftmaxRed=sSoftmaxRed,
                        stage=stage,
                        tidx=tidx,
                        batch_idx=batch_idx,
                        mSinkSeqUsedK=mSinkSeqUsedK,
                        split_idx=split_idx,
                        row_group=row_group,
                        waits_for_previous=runs_pages,
                    )

                red_buf = self.softmax_warpgroup_reduce(
                    row_sum,
                    sSoftmaxRed,
                    tidx,
                    red_buf,
                    lambda a, b: a + b,
                    row_group=row_group,
                )
                if const_expr(runs_pages or runs_sink):
                    si_corr_producer_phase = self.softmax_wait_corr(
                        mbar_ptr, stage, si_corr_producer_phase
                    )
                if tidx == 0:
                    for j in cutlass.range_constexpr(rows):
                        sScale[row_base + j + stage * self.m_block_size] = row_sum[j]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        for j in cutlass.range_constexpr(rows):
                            sScale[
                                row_base + j + stage * self.m_block_size + self.m_block_size * 2
                            ] = row_max[j]
                cute.arch.mbarrier_arrive(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + stage
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def softmax_step_transposed(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        red_buf: Int32,
        n_block: Int32,
        softmax_scale_log2: Float32,
        mbar_ptr: cute.Pointer,
        thr_tmem_load: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStS_second_t2r: Optional[cute.Tensor],
        frg_shape,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        sScale: cute.Tensor,
        sSFP: cute.Tensor,
        sP_uint8: cute.Tensor,
        sSoftmaxRed: cute.Tensor,
        stage: int | Int32,
        tidx: Int32,
        seqlen_k: Int32,
        is_first: cutlass.Constexpr[bool],
        mask_seqlen: cutlass.Constexpr[bool],
        sink_length: Int32,
        row_group: cutlass.Constexpr[int] = 0,
        follows_pv: cutlass.Constexpr[bool] = False,
        mask_sink: cutlass.Constexpr[bool] = False,
        waits_for_previous: cutlass.Constexpr[bool] = True,
    ) -> Tuple[Int32, Int32, Int32]:
        """One KV block of the transposed softmax.

        The tensor-memory load gives thread ``tidx`` kv position ``tidx``, which
        is the invariant the whole step rests on: masking is one predicate, a
        scale group of 16 kv positions is 16 lanes of one warp, and the two FP4
        nibbles that share a byte sit in neighbouring lanes.

        ``follows_pv`` is every block of a tile but its first: the previous
        block's PV may still be reading the P tile this step rewrites.
        ``mask_sink`` is a block that may be page zero. The test sits on the
        softmax's critical path, which paces the whole loop, so the steps that
        cannot reach page zero leave it out.

        ``waits_for_previous`` is every step but the tile's first to publish:
        correction must have taken the previous step's factors from sScale
        before this step overwrites them. The wait sits just before the
        overwrite rather than at the end of the step that published, so
        correction has this step's row maximum to wake in.
        """
        rows = const_expr(self.softmax_rows_per_group)
        row_base = const_expr(row_group * rows)
        bytes_per_row = const_expr(self.n_block_size // 2)
        atom_k_bytes = const_expr(self.n_block_size // 4)

        iket.range_push("sm_wait_s")
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        iket.range_pop()
        tSrS = cute.make_rmem_tensor(frg_shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS)
        if const_expr(self.stacked_query_terms):
            tSrS_second = cute.make_rmem_tensor(frg_shape, self.qk_acc_dtype)
            cute.copy(thr_tmem_load, tStS_second_t2r, tSrS_second)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(
            mbar_ptr + self.mbar_sfqk_load_offset + (self.q_stage - 1 - stage)
        )
        if const_expr(self.stacked_query_terms):
            for j in cutlass.range_constexpr(rows):
                tSrS[j] = tSrS[j] + tSrS_second[j]

        if const_expr(mask_seqlen):
            # A row whose fp4 length is zero still runs one step, at block -1.
            # Folding that back onto block 0 puts the limit at the start of the
            # sequence, so every position is masked, as the untransposed mask
            # does for the same edge.
            n_block_masked = n_block
            if n_block_masked < 0:
                n_block_masked = Int32(0)
            if n_block_masked * self.n_block_size + tidx >= seqlen_k:
                for j in cutlass.range_constexpr(rows):
                    tSrS[j] = -Float32.inf
        if const_expr(self.fused_sink_block and mask_sink):
            # Page zero retains a quantized duplicate of the permanent BF16
            # sink so that every later page keeps its original vLLM boundary.
            # Suppress only that prefix; all other FP4 positions are unchanged.
            if (n_block == Int32(0)) and (tidx < sink_length):
                for j in cutlass.range_constexpr(rows):
                    tSrS[j] = -Float32.inf

        iket.range_push("sm_rowmax")
        # A scale group is sf_vec_size consecutive kv positions, hence that many
        # consecutive lanes, so the butterfly passes through the group maximum
        # on its way to the warp maximum. That intermediate is what the FP4
        # packing below needs, because exp2 is monotonic: the largest
        # exponential of a group is the exponential of the group's largest
        # score. Carrying it out costs one exp2 per row and removes a second
        # butterfly of the same depth.
        group_steps = const_expr(self.sf_vec_size.bit_length() - 1)
        warp_steps = const_expr(cute.arch.WARP_SIZE.bit_length() - 1)
        block_max = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        group_max_score = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        scatter = const_expr(self.softmax_scatters_rows)
        # Past the first block a row's maximum only moves when it grows by
        # more than the rescale threshold, which once it settles it almost
        # never does. Each thread applies the skip test below to its own
        # scores and one barrier ORs the verdicts. The test is monotonic in
        # the score, so it passes on the row maximum exactly when it passes on
        # every score of the row, and the cross-warp maximum is reduced only
        # when some row will take it.
        votes = const_expr(not is_first and RESCALE_THRESHOLD > 0.0 and not self.p_scale_two_level)
        grows = cutlass.Boolean(True)
        if const_expr(votes):
            own_grows = Int32(0)
            for j in cutlass.range_constexpr(rows):
                grown = (
                    row_max[j] - cute.arch.fmax(row_max[j], tSrS[j])
                ) * softmax_scale_log2
                own_grows = Int32(
                    cutlass.select_(grown >= -RESCALE_THRESHOLD, own_grows, Int32(1))
                )
            # Every thread gets the same verdict back; voting on it tells the
            # compiler so, which keeps divergence handling out of the branches
            # it decides.
            grows = cute.arch.vote_any_sync(
                utils.barrier_or(
                    own_grows,
                    barrier_id=int(NamedBarrierFwd.SoftmaxReduce) + row_group,
                    number_of_threads=len(self.softmax0_warp_ids) * cute.arch.WARP_SIZE,
                )
                != Int32(0)
            )
        if const_expr(scatter):
            # Reduce-scatter: each of the first log2(rows) butterfly steps hands
            # half of the rows still held to the partner lane and keeps the
            # other half, so a lane ends with one row, the one ``scatter_row``
            # names, and the steps cost rows - 1 shuffles in all rather than
            # one per row each. The rest of the butterfly then runs on that
            # row alone, through its group maximum to its warp maximum.
            scatter_steps = const_expr(rows.bit_length() - 1)
            held = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
            for j in cutlass.range_constexpr(rows):
                held[j] = tSrS[j]
            for i in cutlass.range_constexpr(scatter_steps):
                half = const_expr(rows >> (i + 1))
                upper = ((tidx >> i) & 1) == 1
                for k in cutlass.range_constexpr(half):
                    send = Float32(cutlass.select_(upper, held[k], held[k + half]))
                    keep = Float32(cutlass.select_(upper, held[k + half], held[k]))
                    held[k] = cute.arch.fmax(
                        keep, cute.arch.shuffle_sync_bfly(send, offset=1 << i)
                    )
            val = held[0]
            for i in cutlass.range_constexpr(scatter_steps, group_steps):
                val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
            own_group_max_score = val
            own_row = self.scatter_row(tidx)
            if grows:
                for i in cutlass.range_constexpr(group_steps, warp_steps):
                    val = cute.arch.fmax(
                        val, cute.arch.shuffle_sync_bfly(val, offset=1 << i)
                    )
                red_buf = self.softmax_warpgroup_reduce_scattered(
                    val,
                    own_row,
                    block_max,
                    sSoftmaxRed,
                    tidx,
                    red_buf,
                    row_group=row_group,
                )
        else:
            for j in cutlass.range_constexpr(rows):
                val = tSrS[j]
                for i in cutlass.range_constexpr(group_steps):
                    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
                group_max_score[j] = val
                block_max[j] = val
            if grows:
                for j in cutlass.range_constexpr(rows):
                    val = block_max[j]
                    for i in cutlass.range_constexpr(group_steps, warp_steps):
                        val = cute.arch.fmax(
                            val, cute.arch.shuffle_sync_bfly(val, offset=1 << i)
                        )
                    block_max[j] = val
                red_buf = self.softmax_warpgroup_reduce(
                    block_max,
                    sSoftmaxRed,
                    tidx,
                    red_buf,
                    cute.arch.fmax,
                    intra_warp=False,
                    row_group=row_group,
                )
        acc_scale = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        row_max_safe = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        # Once the maximum settles every row skips its rescale, and the skip
        # needs only the vote: the factor is 1 and the maximum stays. A block
        # where some row grows takes the full update below, which applies the
        # same test row by row against the reduced maximum.
        if not grows:
            for j in cutlass.range_constexpr(rows):
                row_max_safe[j] = row_max[j]
                acc_scale[j] = Float32(1.0)
        else:
            for j in cutlass.range_constexpr(rows):
                if const_expr(is_first):
                    row_max_new = block_max[j]
                    row_max_safe[j] = row_max_new if row_max_new != -Float32.inf else 0.0
                    acc_scale[j] = 0.0
                else:
                    row_max_old = row_max[j]
                    row_max_new = cute.arch.fmax(row_max_old, block_max[j])
                    safe = row_max_new if row_max_new != -Float32.inf else 0.0
                    acc_scale_ = (row_max_old - safe) * softmax_scale_log2
                    # The skip decision is made on a maximum every thread agrees
                    # on, so it stays uniform across the warpgroup, and a
                    # skipped rescale costs no exponential.
                    scale = Float32(1.0)
                    if const_expr(RESCALE_THRESHOLD > 0.0):
                        if acc_scale_ >= -RESCALE_THRESHOLD:
                            row_max_new = row_max_old
                            safe = row_max_old
                        else:
                            scale = utils.exp2f(acc_scale_)
                    else:
                        scale = utils.exp2f(acc_scale_)
                    row_max_safe[j] = safe
                    acc_scale[j] = scale
                row_max[j] = row_max_new
        # The shift is the same for every score of a row, so scaling it once
        # leaves each exponential a single fused multiply-add. A reduce-
        # scattered tile holds an even number of rows, so its arithmetic runs
        # on pairs of rows in the packed FP32x2 units; one rounding per lane,
        # as the scalar instructions round.
        max_scaled = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        neg_max_scaled = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        if const_expr(scatter):
            for j in cutlass.range_constexpr(0, rows, 2):
                max_scaled[j], max_scaled[j + 1] = utils.mul_packed_f32x2(
                    (row_max_safe[j], row_max_safe[j + 1]),
                    (softmax_scale_log2, softmax_scale_log2),
                )
                neg_max_scaled[j], neg_max_scaled[j + 1] = utils.mul_packed_f32x2(
                    (row_max_safe[j], row_max_safe[j + 1]),
                    (-softmax_scale_log2, -softmax_scale_log2),
                )
        else:
            for j in cutlass.range_constexpr(rows):
                max_scaled[j] = row_max_safe[j] * softmax_scale_log2
        if const_expr(self.p_scale_rescale_log2):
            for j in cutlass.range_constexpr(rows):
                max_scaled[j] -= Float32(self.p_scale_rescale_log2)
                if const_expr(scatter):
                    neg_max_scaled[j] += Float32(self.p_scale_rescale_log2)
        iket.range_pop()

        if const_expr(waits_for_previous):
            si_corr_producer_phase = self.softmax_wait_corr(
                mbar_ptr, stage, si_corr_producer_phase
            )
        # Every thread holds the same correction factors, so one publishes them.
        if const_expr(not is_first):
            if tidx == 0:
                for j in cutlass.range_constexpr(rows):
                    sScale[row_base + j + stage * self.m_block_size] = acc_scale[j]
        if const_expr(self.p_scale_two_level):
            if tidx == 0:
                for j in cutlass.range_constexpr(rows):
                    outer = cute.arch.exp2(
                        block_max[j] * softmax_scale_log2 - max_scaled[j]
                        - Float32(math.log2(448.0 * 6.0))
                    )
                    sScale[row_base + j + stage * self.m_block_size + self.m_block_size * 2] = outer
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        amplified = None
        if const_expr(self.p_scale_two_level):
            amplified = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
            for j in cutlass.range_constexpr(rows):
                peak = Float32(cutlass.select_(block_max[j] == -Float32.inf, 0.0, block_max[j]))
                amplified[j] = cute.arch.exp2(
                    (tSrS[j] - peak) * softmax_scale_log2
                    + Float32(math.log2(448.0 * 6.0))
                )
        iket.range_push("sm_exp")
        if const_expr(scatter):
            for j in cutlass.range_constexpr(0, rows, 2):
                arg0, arg1 = utils.fma_packed_f32x2(
                    (tSrS[j], tSrS[j + 1]),
                    (softmax_scale_log2, softmax_scale_log2),
                    (neg_max_scaled[j], neg_max_scaled[j + 1]),
                )
                tSrS[j] = cute.arch.exp2(arg0)
                tSrS[j + 1] = cute.arch.exp2(arg1)
                if const_expr(is_first):
                    row_sum[j] = tSrS[j]
                    row_sum[j + 1] = tSrS[j + 1]
                else:
                    row_sum[j], row_sum[j + 1] = utils.fma_packed_f32x2(
                        (row_sum[j], row_sum[j + 1]),
                        (acc_scale[j], acc_scale[j + 1]),
                        (tSrS[j], tSrS[j + 1]),
                    )
        else:
            for j in cutlass.range_constexpr(rows):
                tSrS[j] = cute.arch.exp2(tSrS[j] * softmax_scale_log2 - max_scaled[j])
                if const_expr(is_first):
                    row_sum[j] = tSrS[j]
                else:
                    row_sum[j] = row_sum[j] * acc_scale[j] + tSrS[j]
        iket.range_pop()

        if const_expr(self.p_scale_two_level):
            for j in cutlass.range_constexpr(rows):
                peak = Float32(cutlass.select_(block_max[j] == -Float32.inf, 0.0, block_max[j]))
                max_scaled[j] = peak * softmax_scale_log2 - Float32(math.log2(448.0 * 6.0))
                tSrS[j] = amplified[j]
        iket.range_push("sm_pquant")
        group_max = cute.make_rmem_tensor(cute.make_layout(rows), Float32)
        group_exponent = cute.make_rmem_tensor(cute.make_layout(rows), Int32)
        own_scale = Float32(0.0)
        own_exponent = Int32(0)
        if const_expr(scatter):
            # Every lane of a scale group would compute the same scale for a
            # row, so the lane that holds the row's group maximum computes it
            # once and the group's lanes fetch its reciprocal from there.
            own_max_scaled = max_scaled[0]
            for j in cutlass.range_constexpr(1, rows):
                own_max_scaled = Float32(
                    cutlass.select_(own_row == j, max_scaled[j], own_max_scaled)
                )
            maximum = cute.arch.exp2(
                own_group_max_score * softmax_scale_log2 - own_max_scaled
            )
            if const_expr(self.is_mxfp4):
                own_exponent = round_up_exponent(maximum)
            else:
                if const_expr(self.p_scale_amax2_or_amax6):
                    own_scale = _amax2_or_amax6_probability_scale(maximum)
                else:
                    own_scale = maximum * Float32(1.0 / self.p_scale_divisor)
                own_inverse = cute.arch.rcp_approx(cute.arch.fmax(own_scale, 1e-20))
            cluster = tidx & (cute.arch.WARP_SIZE - rows)
            if const_expr(self.is_mxfp4):
                # Each row still has one P value in every lane. Preserve the
                # scattered maximum/exponential arithmetic, but reduce each
                # candidate's error over all 32 values of that row's group.
                for j in cutlass.range_constexpr(rows):
                    exponent = Int32(cute.arch.shuffle_sync(
                        own_exponent, cluster | self.scatter_lane(j)
                    ))
                    group_exponent[j] = _mxfp4_probability_exponent(tSrS[j], exponent)
                    tSrS[j] = tSrS[j] * inverse_power_of_two(group_exponent[j])
                # Do not overwrite any scatter owner until all original
                # exponents have been fetched by every row above.
                for j in cutlass.range_constexpr(rows):
                    own_exponent = Int32(cutlass.select_(
                        own_row == j, group_exponent[j], own_exponent
                    ))
            else:
                for j in cutlass.range_constexpr(0, rows, 2):
                    tSrS[j], tSrS[j + 1] = utils.mul_packed_f32x2(
                        (tSrS[j], tSrS[j + 1]),
                        (
                            cute.arch.shuffle_sync(own_inverse, cluster | self.scatter_lane(j)),
                            cute.arch.shuffle_sync(
                                own_inverse, cluster | self.scatter_lane(j + 1)
                            ),
                        ),
                    )
        for j in cutlass.range_constexpr(0 if scatter else rows):
            # Same expression as the exponentials above, so this is the value
            # the largest lane of the group holds, to the bit.
            maximum = cute.arch.exp2(
                group_max_score[j] * softmax_scale_log2 - max_scaled[j]
            )
            if const_expr(self.is_mxfp4):
                group_exponent[j] = _mxfp4_probability_exponent(
                    tSrS[j], round_up_exponent(maximum)
                )
            elif const_expr(self.p_scale_amax2_or_amax6):
                group_max[j] = _amax2_or_amax6_probability_scale(maximum)
            elif const_expr(self.p_scale_grid_search):
                best_scale = maximum * Float32(1.0 / 6.0)
                best_stored = _round_probability_scale(best_scale)
                best_code = _round_probability_code(
                    tSrS[j]
                    * cute.arch.rcp_approx(
                        cute.arch.fmax(best_scale, 1e-20)
                    )
                )
                difference = best_code * best_stored - tSrS[j]
                best_error = difference * difference
                for i in cutlass.range_constexpr(group_steps):
                    best_error += cute.arch.shuffle_sync_bfly(
                        best_error, offset=1 << i
                    )
                rival_divisors = (4.0, 3.0, 2.0)
                for candidate in cutlass.range_constexpr(
                    len(rival_divisors)
                ):
                    divisor = rival_divisors[candidate]
                    rival_scale = maximum * Float32(1.0 / divisor)
                    rival_stored = _round_probability_scale(rival_scale)
                    rival_code = _round_probability_code(
                        tSrS[j]
                        * cute.arch.rcp_approx(
                            cute.arch.fmax(rival_scale, 1e-20)
                        )
                    )
                    difference = rival_code * rival_stored - tSrS[j]
                    rival_error = difference * difference
                    for i in cutlass.range_constexpr(group_steps):
                        rival_error += cute.arch.shuffle_sync_bfly(
                            rival_error, offset=1 << i
                        )
                    better = rival_error < best_error
                    best_scale = Float32(
                        cutlass.select_(better, rival_scale, best_scale)
                    )
                    best_error = Float32(
                        cutlass.select_(better, rival_error, best_error)
                    )
                group_max[j] = best_scale
            else:
                group_max[j] = maximum * Float32(
                    1.0 / self.p_scale_divisor
                )
        for j in cutlass.range_constexpr(0 if scatter else rows):
            # An approximate reciprocal is one instruction where the exact
            # division is a sequence, and P carries three mantissa bits.
            if const_expr(self.is_mxfp4):
                tSrS[j] = tSrS[j] * inverse_power_of_two(group_exponent[j])
            else:
                tSrS[j] = tSrS[j] * cute.arch.rcp_approx(cute.arch.fmax(group_max[j], 1e-20))

        # The S this step read was signalled before the previous block's PV
        # was even issued, so that PV is waited for here, as late as possible.
        # It completes once per block and runs at most one block behind, so
        # the S phase of the block before names it.
        if const_expr(follows_pv):
            iket.range_push("sm_wait_pv")
            cute.arch.mbarrier_wait(
                mbar_ptr + self.mbar_P_consumed_offset + stage,
                mma_si_consumer_phase ^ 1,
            )
            iket.range_pop()

        # K-major P wants kv contiguous, so the nibbles for kv = tidx and
        # tidx + 1 share a byte. One butterfly per query row brings the odd
        # lane's value over and the even lane stores the merged byte.
        k_byte = tidx // 2
        for j in cutlass.range_constexpr(rows):
            row = row_base + j
            partner = cute.arch.shuffle_sync_bfly(tSrS[j], offset=1)
            if tidx % 2 == 0:
                packed = sm100_utils.packed_float_to_e2m1x2(partner, tSrS[j])
                sP_uint8[(row, k_byte % atom_k_bytes), 0, k_byte // atom_k_bytes, 0] = (
                    cutlass.Uint8(packed & 0xFF)
                )

        sSFP_u8 = cute.recast_tensor(sSFP[None, None, None, stage], cutlass.Uint8)
        if const_expr(scatter):
            # The first lanes of a group hold one row each, so each stores the
            # scale of its own row.
            if tidx % self.sf_vec_size < rows:
                k_group = tidx // self.sf_vec_size
                row = row_base + own_row
                if const_expr(self.is_mxfp4):
                    sf_byte = own_exponent + Int32(127)
                else:
                    sf_byte = packed_float_to_ue4m3(
                        own_scale, Float32(0.0), Float32(0.0), Float32(0.0)
                    )
                sSFP_u8[
                    (((row % 32, row // 32), 0),
                     (0, k_group % self.sf_groups_per_mma_k)),
                    0, k_group // self.sf_groups_per_mma_k
                ] = cutlass.Uint8(sf_byte & 0xFF)
        elif tidx % self.sf_vec_size == 0:
            k_group = tidx // self.sf_vec_size
            for j in cutlass.range_constexpr(rows):
                row = row_base + j
                if const_expr(self.is_mxfp4):
                    sf_byte = group_exponent[j] + Int32(127)
                else:
                    sf_byte = packed_float_to_ue4m3(
                        group_max[j], Float32(0.0), Float32(0.0), Float32(0.0)
                    )
                sSFP_u8[
                    (((row % 32, row // 32), 0),
                     (0, k_group % self.sf_groups_per_mma_k)),
                    0, k_group // self.sf_groups_per_mma_k
                ] = cutlass.Uint8(sf_byte & 0xFF)
        # The MMA reads P and its scale factors through the async proxy while
        # these are generic-proxy stores; the mbarrier below only orders generic
        # traffic, so the fence is what makes them visible.
        cute.arch.fence_view_async_shared()
        iket.range_pop()

        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase, red_buf

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tCtSFP: Optional[Tuple[cute.Tensor, ...]] = None,
        sSFP: Optional[cute.Tensor] = None,
        mResidualSeqUsedK: Optional[cute.Tensor] = None,
        mSinkSeqUsedK: Optional[cute.Tensor] = None,
        sP_uint8: Optional[cute.Tensor] = None,
        sSoftmaxRed: Optional[cute.Tensor] = None,
        sP_bf16: Optional[cute.Tensor] = None,
        sP_bf16_mk: Optional[cute.Tensor] = None,
        row_group: cutlass.Constexpr[int] = 0,
        role: cutlass.Constexpr = SplitRole.ALL,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )

        tStScale = cute.composition(tStSi, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)
        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
            Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(tidx)

        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16 if const_expr(not self.quant_pv) else 8)),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        if const_expr(self.transpose_s):
            self.softmax_loop_transposed(
                stage=stage,
                softmax_scale_log2=softmax_scale_log2,
                thr_mma_qk=thr_mma_qk,
                tStSi=tStSi,
                sScale=sScale,
                mLSE=mLSE,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                TileSchedulerCls=TileSchedulerCls,
                sSFP=sSFP,
                sP_uint8=sP_uint8,
                sSoftmaxRed=sSoftmaxRed,
                tidx=tidx,
                row_group=row_group,
                mResidualSeqUsedK=mResidualSeqUsedK,
                mSinkSeqUsedK=mSinkSeqUsedK,
                sP_bf16=sP_bf16,
                sP_bf16_mk=sP_bf16_mk,
                role=role,
            )
            return

        thr_tmem_store_bf16 = thr_tmem_store
        tStP_bf16_r2t = tStP_r2t
        if const_expr(self.fused_residual_first_block):
            tilePlikeFP32_bf16 = self.mma_tiler_qk[1] // 32 * 16  # BF16=16 bits
            tStP_bf16_layout = cute.composition(
                tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32_bf16))
            )
            tStP_bf16 = cute.make_tensor(
                tStSi.iterator + self.tmem_s_to_p_offset, tStP_bf16_layout
            )
            tmem_store_bf16_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
                Float32,
            )
            thr_tmem_store_bf16 = tcgen05.make_tmem_copy(
                tmem_store_bf16_atom, tStP_bf16
            ).get_slice(tidx)
            tStP_bf16_r2t = thr_tmem_store_bf16.partition_D(tStP_bf16)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
        
        if stage == 1 and const_expr(self.quant_qk):
            sfqk_stage = 0
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)
        if const_expr(self.q_stage == 1) and stage == 0 and const_expr(self.quant_qk):
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + 0)
        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg
        
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = self.split_block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )

            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            shared_mask_kwargs = dict(
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )
            block_mask_mod = self.mask_mod if const_expr(self.use_block_sparsity) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=block_mask_mod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=RESCALE_THRESHOLD,
                # rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0, # (Wenxuan) disable skipping rescale until FP4 precision is verified
                softmax_scale=softmax_scale,
                quant_pv=self.quant_pv,
                compute_sp1=self.compute_sp1,
                kahan=self.fused_residual_first_block,
                p_scale_divisor=self.p_scale_divisor,
                p_scale_grid_search=self.p_scale_grid_search,
                p_scale_amax2_or_amax6=self.p_scale_amax2_or_amax6,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = self.tile_has_work(
                    n_block_min, n_block_max, split_idx, num_splits
                )

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=self.q_stage * m_block + stage,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                mask_fn=partial(mask_fn, mask_seqlen=False),
                tCtSFP=tCtSFP,
                sSFP=sSFP,
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                iket.range_push("sm_wait_corr")
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
                )
                iket.range_pop()
                si_corr_producer_phase ^= 1

            if const_expr(self.fused_residual_first_block) and stage == 0 and has_work:
                mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                    self.softmax_residual_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        softmax=softmax,
                        mbar_ptr=mbar_ptr,
                        thr_mma_qk=thr_mma_qk,
                        thr_tmem_load=thr_tmem_load,
                        thr_tmem_store_bf16=thr_tmem_store_bf16,
                        thr_tmem_store_scale=thr_tmem_store_scale,
                        tStS_t2r=tStS_t2r,
                        tStP_bf16_r2t=tStP_bf16_r2t,
                        sScale=sScale,
                        stage=stage,
                        batch_idx=batch_idx,
                        m_block=self.q_stage * m_block + stage,
                        mResidualSeqUsedK=mResidualSeqUsedK,
                        thread_idx=tidx,
                        split_idx=split_idx,
                    )
                )

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,
                    self.mbar_P_full_2_offset,
                    self.q_stage,
                    Int32(stage),
                )
                if not empty_tile:
                    if const_expr(self.fused_residual_first_block):
                        sScale[tidx + stage * self.m_block_size] = (
                            softmax.row_sum[0] + softmax.row_sum_comp[0]
                        )
                    else:
                        sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
            else:
                if has_work:
                    if const_expr(self.fused_residual_first_block) and stage == 0:
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                            n_block_max - 1,
                            is_first=False,
                            mask_fn=partial(mask_fn, mask_seqlen=True),
                        )
                    else:
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                            n_block_max - 1,
                            is_first=True,
                            mask_fn=partial(mask_fn, mask_seqlen=True),
                        )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block
                    )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    if const_expr(self.fused_residual_first_block):
                        sScale[tidx + stage * self.m_block_size] = (
                            softmax.row_sum[0] + softmax.row_sum_comp[0]
                        )
                    else:
                        sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + utils.log2f(softmax.row_sum[0])) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop
        
    @cute.jit
    def _quant_fp4(self, 
                   # src
                   tSrP_f32: cute.Tensor,
                   tSrPSF_f32: cute.Tensor,
                   # dst
                   tSrP: cute.Tensor,
                   tSrPSF: cute.Tensor,
                   ):
        tSrP_f32_frag = cute.logical_divide(tSrP_f32, cute.make_layout(self.sf_vec_size))
        assert cute.size(tSrP_f32_frag, mode=[1]) == cute.size(tSrPSF_f32)
        tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(self.sf_vec_size))
        tSrPSF_u32_view = cute.recast_tensor(tSrPSF, cute.Int32)

        # Process in groups of 4 for UE4M3 conversion
        assert cute.size(tSrPSF_f32) % 4 == 0
        for i in cutlass.range_constexpr(0, cute.size(tSrPSF_f32) // 4, unroll=1):
        # for i in cutlass.range_constexpr(0, 2):
            # Pack 4 FP32 values into UE4M3 format
            packed_ue4m3 = packed_float_to_ue4m3(
                tSrPSF_f32[i * 4],
                tSrPSF_f32[i * 4 + 1], 
                tSrPSF_f32[i * 4 + 2],
                tSrPSF_f32[i * 4 + 3]
            )
            tSrPSF_u32_view[i] = packed_ue4m3
    
        # Quantize main tensor to E2M1 format (8 values per uint32_t)
        # Process in groups of 8 for E2M1 conversion
        for i in cutlass.range_constexpr(0, cute.size(tSrP_frag, mode=[1])):
            tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, i], cute.Int32)
            for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                packed_e2m1 = packed_float_to_e2m1(
                    tSrP_f32_frag[k * 8, i],
                    tSrP_f32_frag[k * 8 + 1, i],
                    tSrP_f32_frag[k * 8 + 2, i],
                    tSrP_f32_frag[k * 8 + 3, i],
                    tSrP_f32_frag[k * 8 + 4, i],
                    tSrP_f32_frag[k * 8 + 5, i],
                    tSrP_f32_frag[k * 8 + 6, i],
                    tSrP_f32_frag[k * 8 + 7, i]
                )
                tSrP_u32_view[k] = packed_e2m1

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        tCtSFP: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        # P size when in FP32
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))

        # Wait for Si
        iket.range_push("sm_wait_s")
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        iket.range_pop()
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        
        # unblock sfqk load
        cute.arch.fence_view_async_tmem_load()
        sfqk_stage = self.q_stage - 1 - stage
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)

        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                aux_tensors,
                fastdiv_mods,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block) 

        iket.range_push("sm_rowmax")
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        iket.range_pop()
        tSrPSF_f32 = None
        tSrPSF = None

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_rmem_tensor(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
            )
        tSrP_r2t_f32 = cute.make_rmem_tensor(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype),
            tSrS_t2r.layout, # shape of S owned by this thread
        )


        if const_expr(self.quant_pv):
            iket.range_push("sm_exp")
            # Exp2 with softmax scale and sp1 scaling
            softmax.apply_exp2_convert(
                tSrS_t2r,
                e2e=mask_fn is None and self.head_dim_padded <= 128,
                e2e_freq=self.e2e_freq,
            )
            # update_row_sum BEFORE scale_groupwise so it uses original P values
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
            iket.range_pop()
            # Everything below is the FP4-only P re-quantization: a segmented
            # group max, a groupwise rescale, the FP4 convert, and the scale
            # factor R2S. The BF16 path reaches the TMEM store directly from
            # apply_exp2_convert, so this range is the extra cost FP4 pays on
            # the softmax critical path.
            iket.range_push("sm_pquant")
            tSrPSF_f32 = softmax.compute_group_max(tSrS_t2r, sf_size=self.sf_vec_size)
            tSrPSF = cute.make_rmem_tensor(tSrPSF_f32.layout, cute.Float8E4M3FN)
            softmax.scale_groupwise(tSrS_t2r, tSrPSF_f32, sf_size=self.sf_vec_size)
            self._quant_fp4(tSrS_t2r, tSrPSF_f32, tSrP_r2t, tSrPSF)
            # R2S: Copy tSrPSF (registers) to sSFP (shared memory).
            # The SFP smem layout is BlockScaledBasicChunk(16) tile_to_shape((M=128, K=128)),
            # giving byte offsets:
            #   byte = (m%32)*16 + ((m//32)%4)*4 + (k_block%4) + (k_block//4)*512
            # Each softmax thread holds 1 M row (lane_id within warp = row in [0,32),
            # warp_id within softmax warpgroup = row block in [0,4)) and 8 K groups.
            # The first 4 K groups land at +0,+1,+2,+3 within the row's atom; the
            # next 4 land at +512,+513,+514,+515 (rest_k stride).
            if const_expr(sSFP is not None):
                thread_idx = thr_tmem_load.thr_idx
                lane_id = thread_idx % 32
                warp_id = thread_idx // 32
                base_offset = lane_id * 16 + (warp_id % 4) * 4
                sfp_thread_layout = cute.make_layout((4, 2), stride=(1, 512))
                sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
                sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
                tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(4))
                cute.autovec_copy(tSrPSF_2d, sSFP_thread)
                # These stores go through the generic proxy, but the MMA warp
                # reads sSFP with tcgen05.cp, an async-proxy access. The
                # mbarrier arrive below orders generic-proxy traffic only, so
                # without this fence the S2T copy can still observe the
                # previous n_block's scale factors.
                cute.arch.fence_view_async_shared()
            iket.range_pop()
        else:
            # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
            softmax.apply_exp2_convert(
                tSrS_t2r,
                tSrP_r2t,
                e2e=mask_fn is None and self.head_dim_padded <= 128,
                e2e_freq=self.e2e_freq,
            )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2]))):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready (and SFP is in SMEM)
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2])), cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        iket.range_push("sm_wait_corr")
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )
        iket.range_pop()

        if const_expr(not self.quant_pv):
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def softmax_residual_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        thr_mma_qk: cute.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store_bf16: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_bf16_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        m_block: Int32,
        mResidualSeqUsedK: cute.Tensor,
        thread_idx: Int32,
        split_idx: Int32 = Int32(0),
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Compute the BF16 residual softmax step.

        BF16 variant of softmax_step for the residual block. Reads BF16-derived
        FP32 scores from TMEM (written by BF16 GEMM_QK in MMA warp), applies
        mResidualSeqUsedK mask, computes row_max, exp2, row_sum (is_first=True
        so initializes the softmax state from this block alone), casts P→BF16,
        stores P_BF16 to TMEM at tmem_p_bf16_offset (separate from FP4 P).

        After this step, softmax.row_max / softmax.row_sum hold the BF16-block
        contribution. The FP4 mainloop's first softmax_step then runs with
        is_first=False so the online-softmax merge folds in the FP4 blocks on
        top of the BF16 seed.
        """
        tilePlikeFP32_bf16 = self.mma_tiler_qk[1] // Float32.width * 16  # BF16=16 bits
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScP_bf16 = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32_bf16)))

        iket.range_push("sm_res_wait_s")
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_bf16_S_full_offset, Int32(0))
        iket.range_pop()
        iket.range_push("sm_res_comp")
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        cute.arch.fence_view_async_tmem_load()

        # Apply mResidualSeqUsedK mask: tokens at j >= seqused_k[batch_idx] → -inf.
        # Values are in [0, BLOCK_SIZE]. A zero-length row still participates in
        # the residual pipeline barriers, but below it writes P=0 and leaves the
        # reset online-softmax state (-inf max, zero sum) untouched. The first
        # FP4 tile can therefore merge with is_first=False without NaNs.
        seqused_k = Int32(self.n_block_size)
        if const_expr(mResidualSeqUsedK is not None):
            seqused_k = mResidualSeqUsedK[batch_idx]
            if const_expr(self.is_split_kv and not self.bf16_split):
                # One block, one owner: the other splits run the same pipeline
                # over a length of zero and contribute nothing, so the combine
                # sees the residual exactly once.
                if split_idx != self.residual_split_idx:
                    seqused_k = Int32(0)
            tScS_t2r = thr_tmem_load.partition_D(tScS)
            for i in cutlass.range_constexpr(cute.size(tSrS_t2r)):
                n_coord = tScS_t2r[i][1]  # N coordinate
                if n_coord >= seqused_k:
                    tSrS_t2r[i] = -Float32.inf

        # Build BF16 P fragment in registers (cast destination).
        tSrP_bf16_r2t_view = cute.make_rmem_tensor(
            thr_tmem_store_bf16.partition_S(tScP_bf16).shape, Float32
        )
        # The BF16 P fragment is the same logical shape as tSrS_t2r but typed
        # as BFloat16. We use the f32 register and cast inside apply_exp2_convert.
        tSrP_bf16 = cute.make_tensor(
            cute.recast_ptr(tSrP_bf16_r2t_view.iterator, dtype=cutlass.BFloat16),
            tSrS_t2r.layout,
        )

        if seqused_k > Int32(0):
            # Compute row_max with is_first=True (initializes softmax state).
            row_max, acc_scale = softmax.update_row_max(
                tSrS_t2r.load(), is_first=True
            )
            # In-place compute (S - row_max) * scale_log2.
            softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
            # exp2 in FP32 + cast to BF16 destination.
            softmax.apply_exp2_convert(
                tSrS_t2r,
                tSrP_bf16,
                e2e=False,
            )
            softmax.update_row_sum(
                tSrS_t2r.load(), acc_scale, is_first=True
            )
        else:
            # The MMA/PV pipeline must still observe its normal producer
            # arrivals, so publish an all-zero P tile instead of skipping it.
            tSrP_bf16_r2t_view.fill(Float32(0.0))

        # Store BF16 P to TMEM (split into two halves to overlap with MMA).
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_bf16_r2t.shape[2]))):
            cute.copy(thr_tmem_store_bf16, tSrP_bf16_r2t_view[None, None, i], tStP_bf16_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_bf16_P_full_offset)
        for i in cutlass.range_constexpr(
            self.mbar_p_split(cute.size(tStP_bf16_r2t.shape[2])),
            cute.size(tStP_bf16_r2t.shape[2]),
        ):
            cute.copy(thr_tmem_store_bf16, tSrP_bf16_r2t_view[None, None, i], tStP_bf16_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_bf16_P_full_2_offset)
        iket.range_pop()


        # Phase tracking: mma_si_consumer_phase is unchanged because we
        # didn't consume mbar_S_full; si_corr_producer_phase is unchanged
        # because we didn't consume mbar_softmax_corr_empty.
        return mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        role: cutlass.Constexpr = SplitRole.ALL,
    ):
        runs_pages = const_expr(role != SplitRole.BF16)
        runs_tail = const_expr(
            self.fused_residual_first_block and role != SplitRole.PAGES
        )
        runs_sink = const_expr(self.fused_sink_block and role != SplitRole.PAGES)
        two_level = const_expr(self.p_scale_two_level and runs_pages)
        accumulated = None
        previous_outer = Float32(1.0)
        if const_expr(two_level):
            assert self.q_stage == 1
            accumulated = cute.make_rmem_tensor(cute.make_layout(self.head_dim_v_padded), Float32)
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        # O^T is a 128-lane FP32 accumulator like S, so S's tensor-memory
        # layout at O's columns addresses it; column r is query row r.
        tOtOs_corr = tOtOs
        if const_expr(self.narrow_pv):
            tOtOs_corr = (
                cute.make_tensor(tStS.iterator + self.tmem_o_offset[0], tStS.layout),
            )
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
            self.qk_acc_dtype,
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # The free arrival that lets the mma warp past its first PV. Only a tile
        # starting on an fp4 block gets it; see correction_consume_scale.
        if const_expr(not runs_tail):
            for stage_init in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_arrive(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage_init
                )

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        pv_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = self.split_block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = self.tile_has_work(
                    n_block_min, n_block_max, split_idx, num_splits
                )

            if has_work:
                if const_expr(two_level):
                    accumulated.fill(0.0)
                    previous_outer = Float32(1.0)
                if const_expr(runs_pages):
                    # The first signal from softmax needs no correction unless a
                    # residual seeded the online softmax, in which case it carries
                    # the factor that takes O from the residual's row max to the
                    # first fp4 block's.
                    iket.range_push("corr_wait_sm")
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 0, softmax_corr_consumer_phase
                    )
                    if const_expr(runs_tail):
                        previous_outer = self.correction_consume_scale(
                            thr_mma_pv, tOtOs_corr, sScale, tidx, 0, mbar_ptr,
                            accumulated=accumulated, previous_outer=previous_outer,
                            tScS=tScS,
                        )
                    elif const_expr(two_level):
                        previous_outer = sScale[tidx + self.m_block_size * 2] if tidx < self.transposed_query_rows else Float32(0.0)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
                    if const_expr(self.q_stage == 2):
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + 1, softmax_corr_consumer_phase
                        )
                        if const_expr(runs_tail):
                            self.correction_consume_scale(
                                thr_mma_pv, tOtOs_corr, sScale, tidx, 1, mbar_ptr,
                                tScS=tScS,
                            )
                    iket.range_pop()
                    softmax_corr_consumer_phase ^= 1

                    tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                    # The sink's factor comes over the same handshake as a
                    # page's, so it takes one more step of this loop. A second
                    # copy of the rescale after the loop, never mind whether it
                    # runs, compiles every block of the loop longer: 2.5%
                    # slower at batch 32 on Qwen3-4B. A tile whose fp4 length is
                    # zero still runs its first step.
                    correction_steps = total_block_count - 1
                    if const_expr(runs_sink):
                        correction_steps = cutlass.max(correction_steps, Int32(0)) + 1
                    for i in cutlass.range(correction_steps, unroll=1):
                        for stage in cutlass.range_constexpr(self.q_stage):
                            # wait for S0 / S1
                            iket.range_push("corr_wait_sm")
                            cute.arch.mbarrier_wait(
                                mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                                softmax_corr_consumer_phase,
                            )
                            iket.range_pop()
                            previous_outer = self.correction_consume_scale(
                                thr_mma_pv, tOtOs_corr, sScale, tidx, stage, mbar_ptr,
                                previous_pv_phase=pv_consumer_phase ^ (i & 1),
                                accumulated=accumulated, previous_outer=previous_outer,
                                tScS=tScS,
                            )
                            if const_expr(self.q_stage == 2):
                                cute.arch.mbarrier_arrive(
                                    mbar_ptr + self.mbar_softmax_corr_empty_offset + (1 - stage)
                                )
                            else:
                                cute.arch.mbarrier_arrive(
                                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage
                                )
                        softmax_corr_consumer_phase ^= 1
                    if const_expr(self.transpose_s):
                        # P has one completion per FP4 page, including the
                        # zero-page placeholder. The correction handshake also
                        # signals the final row sum, so its phase cannot track
                        # P across tiles of a persistent CTA.
                        pv_consumer_phase ^= cutlass.max(total_block_count, Int32(1)) & 1
                if const_expr(runs_sink and not runs_pages):
                    iket.range_push("corr_wait_sm")
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset,
                        softmax_corr_consumer_phase,
                    )
                    iket.range_pop()
                    self.correction_consume_scale(
                        thr_mma_pv, tOtOs_corr, sScale, tidx, 0, mbar_ptr,
                        tScS=tScS,
                    )
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset
                    )
                    softmax_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + 1
                    )
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                # Over the BF16 tiles the O_full wait below suffices: it follows
                # the tile's last MMA, and that MMA is their last reader.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (self.q_stage * m_block + stage) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    iket.range_push("corr_wait_sm")
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        softmax_corr_consumer_phase,
                    )
                    iket.range_pop()
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    if const_expr(self.transpose_s):
                        # A zero sum marks the row as empty further down, which
                        # is what the rows past the live query rows are.
                        row_sum = (
                            sScale[tidx + stage * self.m_block_size]
                            if tidx < self.transposed_query_rows
                            else Float32(0.0)
                        )
                    else:
                        row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[tidx + stage * self.m_block_size + self.m_block_size * 2]
                        if const_expr(self.transpose_s):
                            row_max = (
                                row_max
                                if tidx < self.transposed_query_rows
                                else -Float32.inf
                            )
                    else:
                        row_max = None
                    # A narrow PV leaves every live row's lane of O^T in every
                    # thread, so each one needs all the row sums, and has to
                    # read them before softmax is free to overwrite sScale.
                    row_scales = None
                    if const_expr(self.narrow_pv):
                        row_scales = cute.make_rmem_tensor(
                            cute.make_layout(self.transposed_query_rows), Float32
                        )
                        for row in cutlass.range_constexpr(self.transposed_query_rows):
                            live_sum = sScale[row + stage * self.m_block_size]
                            row_scales[row] = cute.arch.rcp_approx(
                                live_sum
                                if not (live_sum == 0.0 or live_sum != live_sum)
                                else 1.0
                            )
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(2.0 ** self.p_scale_rescale_log2)
                            else:
                                row_sum += utils.exp2f(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2
                                    + Float32(self.p_scale_rescale_log2)
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    iket.range_push("corr_wait_o")
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase
                    )
                    iket.range_pop()
                    if const_expr(not self.use_correction_warps_for_epi):
                        iket.range_push("corr_wait_epi")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage, corr_epi_producer_phase
                        )
                        iket.range_pop()
                    if const_expr(two_level):
                        self.correction_accumulate_tile(
                            thr_mma_pv, tOtOs[stage], tidx, accumulated,
                            previous_outer, Float32(1.0), store=True,
                        )
                    if const_expr(self.narrow_pv):
                        self.correction_epilogue_rows(
                            tOtOs_corr[0], tScS, tidx, row_scales, sO[None, None, stage]
                        )
                    else:
                        self.correction_epilogue(
                            thr_mma_pv,
                            tOtOs[stage],
                            tidx,
                            stage,
                            m_block,
                            seqlen.seqlen_q,
                            scale,
                            sO[None, None, stage],
                            mO_cur,
                            gO,
                            gmem_tiled_copy_O,
                        )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + stage)
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them. Withheld on a residual-seeded
                    # tile for the same reason as the pre-loop arrival; see
                    # correction_consume_scale.
                    if const_expr(not runs_tail):
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
                        )

                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # WARNING: we need some code before the const_expr, see https://github.com/NVIDIA/cutlass/issues/2781
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                else:
                    gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (self.q_stage * m_block + stage,)
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + utils.log2f(row_sum)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if const_expr(self.p_scale_rescale_log2):
                        lse -= Float32(self.p_scale_rescale_log2 * LN2)
                    effective_seqlen_q_lse = (
                        Int32(1)
                        if const_expr(self.seqlen_q_static_one)
                        else seqlen.seqlen_q
                    )
                    seqlen_q = (
                        effective_seqlen_q_lse
                        if const_expr(not self.pack_gqa)
                        else effective_seqlen_q_lse * self.qhead_per_kvhead
                    )
                    if tidx < seqlen_q - (self.q_stage * m_block + stage) * self.m_block_size:
                        # This actually just works with PackGQA too
                        gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)
        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    def _o_transposed_copies(self, tOtT: cute.Tensor, tScS: cute.Tensor, tidx: Int32):
        """Tensor-memory copies over eight rows of O^T at a time.

        A thread gets one head-dimension lane and eight consecutive columns,
        which are query rows; fragment element j is column j.
        """
        chunk = 8
        tile = cute.composition(tOtT, cute.make_layout((self.m_block_size, chunk)))
        coords = cute.composition(tScS, cute.make_layout((self.m_block_size, chunk)))
        load = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(chunk)), self.pv_acc_dtype
            ),
            tile,
        ).get_slice(tidx)
        store = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(chunk)), self.pv_acc_dtype
            ),
            tile,
        ).get_slice(tidx)
        return (
            chunk,
            load,
            store,
            load.partition_S(tile),
            store.partition_D(tile),
            load.partition_D(coords).shape,
        )

    @cute.jit
    def correction_rescale_rows(
        self,
        tOtT: cute.Tensor,
        tScS: cute.Tensor,
        tidx: Int32,
        sScale: cute.Tensor,
        stage: int,
    ):
        """Rescale O^T by each live row's factor; see ``narrow_pv``.

        The columns past the live rows hold the product of P's zero rows and
        are left alone.
        """
        chunk, load, store, source, destination, frg_shape = self._o_transposed_copies(
            tOtT, tScS, tidx
        )
        for c in cutlass.range_constexpr(self.pv_rows // chunk):
            fragment = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            cute.copy(load, cute.make_tensor(source.iterator + c * chunk, source.layout), fragment)
            for j in cutlass.range_constexpr(chunk):
                if const_expr(c * chunk + j < self.transposed_query_rows):
                    fragment[j] = fragment[j] * sScale[c * chunk + j + stage * self.m_block_size]
            cute.copy(
                store, fragment, cute.make_tensor(destination.iterator + c * chunk, destination.layout)
            )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue_rows(
        self,
        tOtT: cute.Tensor,
        tScS: cute.Tensor,
        tidx: Int32,
        row_scales: cute.Tensor,
        sO: cute.Tensor,
    ):
        """Normalize O^T and store it into the row-major O tile.

        Each thread writes its head-dimension lane of every live row, so the
        epilogue warp copies sO out exactly as it does the wide tile.
        """
        chunk, load, _, source, _, frg_shape = self._o_transposed_copies(tOtT, tScS, tidx)
        for c in cutlass.range_constexpr(self.pv_rows // chunk):
            fragment = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            cute.copy(load, cute.make_tensor(source.iterator + c * chunk, source.layout), fragment)
            for j in cutlass.range_constexpr(chunk):
                if const_expr(c * chunk + j < self.transposed_query_rows):
                    fragment[j] = fragment[j] * row_scales[c * chunk + j]
            converted = cute.make_rmem_tensor(frg_shape, self.o_dtype)
            converted.store(fragment.load().to(self.o_dtype))
            for j in cutlass.range_constexpr(chunk):
                if const_expr(c * chunk + j < self.transposed_query_rows):
                    sO[c * chunk + j, tidx] = converted[j]
        cute.arch.fence_view_async_tmem_load()
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def correction_accumulate_tile(
        self, thr_mma: cute.ThrMma, tOtO: cute.Tensor, tidx: Int32,
        accumulated: cute.Tensor, outer: Float32, rescale: Float32,
        store: cutlass.Constexpr[bool] = False,
    ):
        """Merge one PV tile after undoing its FP32 outer scale exactly once.

        The running numerator remains in the ordinary online-softmax basis.
        Each FP4 MMA zero-initializes TMEM; the correction handshake prevents
        its reuse until this load completes. BF16 tail and sink use outer=1.
        """
        width = const_expr(16)
        identity = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        tile = cute.composition(tOtO, cute.make_layout((self.m_block_size, width)))
        coordinates = cute.composition(identity, cute.make_layout((self.m_block_size, width)))
        loader = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(width)), Float32,
        ), tile).get_slice(tidx)
        source = loader.partition_S(tile)
        fragment = cute.make_rmem_tensor(loader.partition_D(coordinates).shape, Float32)
        writer = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(width)), Float32,
        ), tile).get_slice(tidx)
        destination = writer.partition_D(tile)
        for i in cutlass.range_constexpr(self.head_dim_v_padded // width):
            src = cute.make_tensor(source.iterator + i * width, source.layout)
            cute.copy(loader, src, fragment)
            for j in cutlass.range_constexpr(cute.size(fragment)):
                contribution = Float32(0.0)
                if outer > 0.0:
                    contribution = fragment[j] * outer
                accumulated[i * width + j] = (accumulated[i * width + j] + contribution) * rescale
                if const_expr(store):
                    fragment[j] = accumulated[i * width + j]
            if const_expr(store):
                dst = cute.make_tensor(destination.iterator + i * width, destination.layout)
                cute.copy(writer, fragment, dst)
        cute.arch.fence_view_async_tmem_load()
        if const_expr(store):
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_consume_scale(
        self,
        thr_mma_pv: cute.ThrMma,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        tidx: Int32,
        stage: int,
        mbar_ptr: cute.Pointer,
        previous_pv_phase: Optional[Int32] = None,
        accumulated: Optional[cute.Tensor] = None,
        previous_outer: Float32 = Float32(1.0),
        tScS: Optional[cute.Tensor] = None,
    ):
        """Take one correction factor from softmax and release the mma warp.

        Transposed, softmax publishes one factor per live query row rather than
        one per thread, so the O rows past that bound have no factor and no
        consumer.

        Every step of the loop runs this. The first step of a tile does too,
        but only when a fused residual seeded the online softmax: the
        residual's PV has already written O by then, which makes that first
        factor a real one rather than the no-op it is on a tile whose first
        block zero-initializes O. The arrival matters as much as the rescale —
        it is what holds the mma warp's first fp4 PV until O has been scaled,
        and a residual-seeded tile has no free pre-loop arrival standing in for
        it.

        With a narrow PV ``tOtOs`` holds the O^T view instead, every thread
        holds a lane of every live row, and all of them read every factor.
        """
        if const_expr(self.narrow_pv):
            min_scale = sScale[stage * self.m_block_size]
            for row in cutlass.range_constexpr(1, self.transposed_query_rows):
                min_scale = cutlass.min(min_scale, sScale[row + stage * self.m_block_size])
            if min_scale < 1.0:
                if const_expr(previous_pv_phase is not None):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_consumed_offset + stage,
                        previous_pv_phase,
                    )
                self.correction_rescale_rows(tOtOs[0], tScS, tidx, sScale, stage)
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
            return Float32(1.0)
        if const_expr(self.transpose_s):
            scale = (
                sScale[tidx + stage * self.m_block_size]
                if tidx < self.transposed_query_rows
                else Float32(1.0)
            )
        else:
            scale = sScale[tidx + stage * self.m_block_size]
        next_outer = Float32(1.0)
        if const_expr(accumulated is not None):
            if const_expr(previous_pv_phase is not None):
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_consumed_offset + stage, previous_pv_phase)
            self.correction_accumulate_tile(
                thr_mma_pv, tOtOs[0], tidx, accumulated, previous_outer, scale,
            )
            next_outer = sScale[tidx + self.m_block_size * 2] if tidx < self.transposed_query_rows else Float32(0.0)
        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
        if const_expr(accumulated is None) and should_rescale:
            # Transposed QK can finish before the preceding PV is issued.
            # Its softmax signal therefore does not make O safe to rescale.
            # Wait only when touching O; a unit factor needs no PV dependency.
            if const_expr(self.transpose_s and previous_pv_phase is not None):
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_P_consumed_offset + stage,
                    previous_pv_phase,
                )
            self.correction_rescale(
                thr_mma_pv, tOtOs[stage if self.q_stage == 2 else 0], tidx, scale
            )
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        return next_outer

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor, # tmem
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor(softmax denominator) to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0]).get_slice(
            tidx
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOrO_frg_cvt = cute.make_rmem_tensor(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
        # fence view async shared
        cute.arch.fence_proxy("async.shared", space="cta")
        if const_expr(self.use_correction_warps_for_epi):
            assert(not self.use_tma_O)
            assert(gmem_tiled_copy_O is not None)
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                              number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE)
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=mO_cur.shape[1])
            # This output path does not support packed GQA.
            assert not self.pack_gqa
            pack_gqa = PackGQA(
                self.m_block_size,
                self.head_dim_v_padded,
                self.check_hdim_v_oob,
                self.qhead_per_kvhead,
            )
        
            # load acc O from smem to rmem for wider vectorization
            tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)
            effective_seqlen_q_corr = (
                Int32(1)
                if const_expr(self.seqlen_q_static_one)
                else seqlen_q
            )
            if const_expr(not self.pack_gqa):
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < effective_seqlen_q_corr
                        - (self.q_stage * m_block + stage) * self.m_block_size
                        - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur,
                    tOrO,
                    gmem_tiled_copy_O,
                    tidx,
                    self.q_stage * m_block + stage,
                    effective_seqlen_q_corr,
                )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mOutIndices: Optional[cute.Tensor] = None,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = self.split_block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )

            if self.tile_has_work(n_block_min, n_block_max, split_idx, num_splits):
                if const_expr(self.use_out_indices):
                    mO_cur = seqlen.offset_batch_O_via_indices(
                        mO, batch_idx, mOutIndices, dim=3,
                    )[None, None, head_idx]
                elif const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        iket.range_push("epi_wait_corr")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        iket.range_pop()
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=self.q_stage * m_block + stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    if const_expr(not self.use_out_indices and not self.seqlen_q_static_one):
                        assert not self.pack_gqa
                    pack_gqa = PackGQA(
                        self.m_block_size,
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        self.qhead_per_kvhead,
                    )
                    # With seqlen_q statically one, PackGQA folds exactly
                    # qhead_per_kvhead rows, and since that is at most one
                    # m_block_size there is a single m_block and every reachable
                    # row lives in Q stage 0. Both the row bound and the dead
                    # stage are therefore compile-time constants.
                    reachable_rows = None
                    if const_expr(self.pack_gqa and self.seqlen_q_static_one):
                        assert self.qhead_per_kvhead <= self.m_block_size
                        # Replication spreads the same heads over the whole tile,
                        # one every q_replicate rows, so the last reachable row
                        # moves to the end of the tile.
                        reachable_rows = self.qhead_per_kvhead * self.q_replicate
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        iket.range_push("epi_wait_corr")
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        iket.range_pop()
                        # 2. copy O0 / O1 to gmem
                        stage_is_live = (
                            reachable_rows is None
                            or stage * self.m_block_size < reachable_rows
                        )
                        if const_expr(stage_is_live):
                            if const_expr(not self.pack_gqa):
                                # load acc O from smem to rmem for wider vectorization
                                tOrO = cute.make_fragment_like(
                                    tOsO[None, None, None, 0], self.o_dtype
                                )
                                cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                                effective_seqlen_q_nonpack = (
                                    Int32(1)
                                    if const_expr(self.seqlen_q_static_one)
                                    else seqlen.seqlen_q
                                )
                                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                                    if (
                                        t0OcO[0, rest_m, 0][0]
                                        < effective_seqlen_q_nonpack
                                        - (self.q_stage * m_block + stage) * self.m_block_size
                                        - tOcO[0][0]
                                    ):
                                        cute.copy(
                                            gmem_tiled_copy_O,
                                            tOrO[None, rest_m, None],
                                            tOgO[
                                                None,
                                                rest_m,
                                                None,
                                                self.q_stage * m_block + stage,
                                            ],
                                            pred=tOpO[None, rest_m, None]
                                            if const_expr(self.check_hdim_v_oob)
                                            else None,
                                        )
                            else:
                                effective_seqlen_q = (
                                    Int32(1)
                                    if const_expr(self.use_out_indices or self.seqlen_q_static_one)
                                    else seqlen.seqlen_q
                                )
                                pack_gqa.store_O(
                                    mO_cur,
                                    tOsO[None, None, None, stage],
                                    gmem_tiled_copy_O,
                                    tidx,
                                    self.q_stage * m_block + stage,
                                    effective_seqlen_q,
                                    max_rows=reachable_rows,
                                    stage_from_smem=True,
                                    row_stride=self.q_replicate,
                                )
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

                epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
        load_SFQ_fn: Optional[Callable] = None,
        load_Q_second_fn: Optional[Callable] = None,
        load_SFQ_second_fn: Optional[Callable] = None,
    ):
        iket.range_push("load_wait_q")
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        iket.range_pop()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        load_Q_fn(
            src_idx=block,
            dst_idx=stage,
            tma_bar_ptr=mbar_full_ptr + stage,
        )
        if const_expr(load_SFQ_fn is not None):
            load_SFQ_fn(
                src_idx=block,
                dst_idx=stage,
                tma_bar_ptr=mbar_full_ptr + stage,
            )
        if const_expr(self.query_terms == 2):
            load_Q_second_fn(
                src_idx=block,
                dst_idx=stage,
                tma_bar_ptr=mbar_full_ptr + stage,
            )
            if const_expr(load_SFQ_second_fn is not None):
                load_SFQ_second_fn(
                    src_idx=block,
                    dst_idx=stage,
                    tma_bar_ptr=mbar_full_ptr + stage,
                )

    @cute.jit
    def stack_Q_fp4_terms(
        self,
        sQ_uint8: cute.Tensor,
        sQ_second: cute.Tensor,
        sSFQ: cute.Tensor,
        sSFQ_second: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_ready_ptr: cute.Pointer,
        stage: int,
        phase: Int32,
        tidx: Int32,
    ):
        """Put the second term after the first term's live rows in SMEM.

        Both full query tiles and both scale tiles have already arrived by
        the time the load barrier completes. Their sizes and TMA transaction
        counts stay unchanged. Only padding in the first tile is replaced;
        the caller-owned scratch and its zero padding are never modified.
        One load warp copies complete bytes and scales, fences its writes for
        the Tensor Core/S2T consumers, then publishes one ready arrival.
        """
        assert self.q_stage == 1
        assert self.head_dim_padded == 128
        cute.arch.mbarrier_wait(mbar_full_ptr + stage, phase)
        rows = const_expr(self.transposed_query_rows)
        # Pointer swizzles act on byte addresses, so the word view retains
        # the byte view's SW(2,4,3); only its layout strides change units.
        # Each aligned word preserves eight packed nibbles. Two rows fill a
        # warp, with sixteen lanes moving the sixteen words of each row.
        q_word_layout = cute.make_layout(
            (self.m_block_size, 16), stride=(16, 1)
        )
        q_word_swizzle = cute.make_swizzle(2, 4, 3)
        src_q = cute.make_tensor(
            cute.recast_ptr(sQ_second.iterator, q_word_swizzle, cutlass.Uint32),
            q_word_layout,
        )
        dst_q = cute.make_tensor(
            cute.recast_ptr(sQ_uint8.iterator, q_word_swizzle, cutlass.Uint32),
            q_word_layout,
        )
        for row_pair in cutlass.range_constexpr(rows // 2):
            row = 2 * row_pair + tidx // 16
            word = tidx % 16
            dst_q[rows + row, word] = src_q[row, word]
        src_sf = cute.recast_tensor(
            sSFQ_second[None, None, None, stage], cutlass.Uint8
        )
        dst_sf = cute.recast_tensor(
            sSFQ[None, None, None, stage], cutlass.Uint8
        )
        # The scale layout is unchanged; only lane ownership changes, from
        # one row's lanes to a warp covering several rows at once. A row
        # holds head_dim / sf_vec_size scales, sf_groups_per_mma_k per MMA K
        # step: eight and four for NVFP4, four and two for MXFP4.
        scale_count = const_expr(
            rows * self.head_dim_padded // self.sf_vec_size
        )
        for scale_group in cutlass.range_constexpr(
            -(-scale_count // cute.arch.WARP_SIZE)
        ):
            scale_index = scale_group * cute.arch.WARP_SIZE + tidx
            if const_expr(scale_count % cute.arch.WARP_SIZE):
                # Fewer scales than lanes (MXFP4 at four rows): a spare lane
                # repeats another lane's copy, the same byte to the same
                # address.
                scale_index = scale_index % scale_count
            row = scale_index % rows
            scale = scale_index // rows
            mma_k = scale // self.sf_groups_per_mma_k
            k_inst = scale % self.sf_groups_per_mma_k
            dst = rows + row
            dst_sf[(((dst % 32, dst // 32), 0), (0, k_inst)), 0, mma_k] = (
                src_sf[(((row % 32, row // 32), 0), (0, k_inst)), 0, mma_k]
            )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar_ready_ptr + stage)

    @cute.jit
    def replicate_Q_fp4_rows(
        self,
        sQ_uint8: cute.Tensor,
        sSFQ: cute.Tensor,
        mbar_full_ptr: cute.Pointer,   # mbar_load_q_full (FP4 TMA arrival)
        mbar_ready_ptr: cute.Pointer,  # mbar_q_fp4_ready
        stage: int,
        phase: Int32,
        tidx: Int32,
    ):
        """Repeat each query row across its block of the Q tile, in place.

        The TMA lands ``qhead_per_kvhead`` real rows and zero-fills the rest.
        This rewrites row ``m`` from row ``m // q_replicate`` so every row of the
        tile carries a query, which is what lets a softmax thread own a short
        column range instead of a whole row.

        Source rows are also destinations — row 1 feeds the second block and is
        itself overwritten from row 0 — so the order matters. Each thread owns a
        fixed set of byte columns for the whole tile, which removes any
        cross-thread hazard, and writes block 0 last, after every other block
        has consumed the source rows that live inside it.
        """
        cute.arch.mbarrier_wait(mbar_full_ptr + stage, phase)

        replicate = const_expr(self.q_replicate)
        real_rows = const_expr(self.m_block_size // replicate)
        bytes_per_row = const_expr(self.head_dim_padded // 2)  # two FP4 per byte
        sf_groups_per_row = const_expr(self.head_dim_padded // self.sf_vec_size)
        num_threads = const_expr(len(self.load_warp_ids) * cute.arch.WARP_SIZE)
        assert bytes_per_row % num_threads == 0

        sQ_u8_stg = sQ_uint8[None, None, None, stage]
        sSFQ_u8_stg = cute.recast_tensor(sSFQ[None, None, None, stage], cutlass.Uint8)
        cols_per_thread = const_expr(bytes_per_row // num_threads)

        # Descending so that block 0, the one holding every source row, is
        # written after all of them have been read.
        for r_rev in cutlass.range_constexpr(real_rows):
            r = const_expr(real_rows - 1 - r_rev)
            for c in cutlass.range_constexpr(cols_per_thread):
                k_byte = tidx * cols_per_thread + c
                atom_k = k_byte % 32
                kH = k_byte // 32
                val = sQ_u8_stg[(r, atom_k), 0, kH]
                for i in cutlass.range_constexpr(replicate):
                    sQ_u8_stg[(r * replicate + i, atom_k), 0, kH] = val
            if tidx < sf_groups_per_row:
                mma_k = tidx // 4
                k_inst = tidx % 4
                sf_val = sSFQ_u8_stg[(((r % 32, r // 32), 0), (0, k_inst)), 0, mma_k]
                for i in cutlass.range_constexpr(replicate):
                    m = r * replicate + i
                    sSFQ_u8_stg[(((m % 32, m // 32), 0), (0, k_inst)), 0, mma_k] = sf_val

        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar_ready_ptr + stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        tma_atom_sf: Optional[cute.CopyAtom] = None,
        tXgSF: Optional[cute.Tensor] = None,
        tXsSF: Optional[cute.Tensor] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        iket.range_push("load_wait_kv")
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)
        iket.range_pop()

        if const_expr(self.use_tma_KV):
            assert (
                tXgX is not None and
                tXsX is not None and
                tma_atom is not None
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V],
                )
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)
        else:
            assert paged_kv_manager is not None
            paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)
        
        # Load scale factor for K or V if provided (uses same barrier as K/V)
        if const_expr(tma_atom_sf is not None and tXgSF is not None and tXsSF is not None):
            tXsSF_cur = tXsSF[None, stage]
            # After tma_partition with rank-1 grouping, tXgSF has structure: ((atom_v, rest_v), RestL)
            if const_expr(page_idx is None):
                tXgSF_cur = tXgSF[None, block]
            else:
                tXgSF_cur = tXgSF[None, 0, page_idx]
            cute.copy(tma_atom_sf, tXgSF_cur, tXsSF_cur, tma_bar_ptr=mbar_full_ptr + stage)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tCtSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr, use_k_bytes: bool = True):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        if self.use_tma_KV:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids)
            )
            return cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=load_kv_mbar_ptr,
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"] if use_k_bytes else self.tma_copy_bytes["V"],
            )
        else:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids) * cute.arch.WARP_SIZE
            )
            return cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                barrier_storage=load_kv_mbar_ptr,
            )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical = q_physical // self.qhead_per_kvhead
            head_offset = q_physical - q_idx_logical * self.qhead_per_kvhead
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = seqlen_q_divmod.divmod(q_idx_logical)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )
