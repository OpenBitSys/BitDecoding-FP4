# Copyright (c) 2025, Tri Dao.

import math
import operator

import cutlass
import cutlass.cute as cute

from . import utils as utils


class PackGQA:
    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        head_dim_padded: cutlass.Constexpr[int],
        check_hdim_oob: cutlass.Constexpr[bool],
        qhead_per_kvhead: cutlass.Constexpr[bool],
    ):
        self.m_block_size = m_block_size
        self.head_dim_padded = head_dim_padded
        self.check_hdim_oob = check_hdim_oob
        self.qhead_per_kvhead = qhead_per_kvhead

    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
        row_stride: cutlass.Constexpr[int] = 1,
    ):
        """Address of each row this thread will move.

        ``row_stride`` describes a tile whose query rows have been repeated, so
        that row ``r * row_stride`` carries head ``r``. It is only meaningful
        with a single query position, where the block index is zero and the
        sequence coordinate is always zero.

        ``tensor`` is indexed by either the hierarchical (head, seqlen) row
        mode or, for a single query position, a flat head axis; the flat form
        takes the head index alone.
        """
        num_ptr_per_thread = cute.ceil_div(cute.size(cRows), threads_per_row)
        tPrPtr = cute.make_rmem_tensor(num_ptr_per_thread, cutlass.Int64)
        for i in cutlass.range_constexpr(num_ptr_per_thread):
            row = i * num_threads + cRows[tidx % threads_per_row][0]
            if cutlass.const_expr(row_stride > 1):
                h_idx = row // row_stride
                m_idx = 0
            else:
                idx = block * self.m_block_size + row
                m_idx = idx // self.qhead_per_kvhead
                h_idx = idx - m_idx * self.qhead_per_kvhead
            if cutlass.const_expr(cute.rank(tensor.shape[0]) == 1):
                tPrPtr[i] = utils.elem_pointer(tensor, (h_idx,)).toint()
            else:
                tPrPtr[i] = utils.elem_pointer(tensor, ((h_idx, m_idx),)).toint()
        return tPrPtr

    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        sQ: cute.Tensor,  # (m_block_size, head_dim_padded)
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=mQ.shape[1])
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrQPtr = self.compute_ptr(mQ[None, 0], tQcQ_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_ptr_i64 = utils.shuffle_sync(
                tPrQPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            q_gmem_ptr = cute.make_ptr(
                mQ.element_type, q_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0QcQ[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tQcQ_row[0][0]
            ):
                mQ_cur = cute.make_tensor(q_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tQsQ.shape[0][0])
                mQ_cur_copy = cute.tiled_divide(mQ_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        mQ_cur_copy[None, ki],
                        tQsQ[None, m, k],
                        pred=tQpQ[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            # We don't need to clear the sQ smem tiles since we'll only write out the valid outputs

    @cute.jit
    def store_LSE(
        self,
        mLSE: cute.Tensor,  # (qhead_per_kvhead, seqlen_q)
        tLSErLSE: cute.Tensor,  # (m_block_size, head_dim_padded)
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = thr_mma.partition_C(caccO)
        taccOcO_row = utils.make_acc_tensor_mn_view(taccOcO)[None, 0]
        assert cute.size(tLSErLSE) == cute.size(taccOcO_row)
        threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        assert cute.size(tLSErLSE) <= threads_per_row
        num_threads = tiled_mma.size
        tPrLSEPtr = self.compute_ptr(mLSE, taccOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tLSErLSE)):
            lse_ptr_i64 = utils.shuffle_sync(
                tPrLSEPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            row = block * self.m_block_size + taccOcO_row[m][0]
            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0 and row < seqlen * self.qhead_per_kvhead:
                mLSE_copy = cute.make_tensor(lse_gmem_ptr, (1,))
                mLSE_copy[0] = tLSErLSE[m]

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads according to gmem_tiled_copy
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        max_rows: cutlass.Constexpr = None,
        stage_from_smem: cutlass.Constexpr[bool] = False,
        row_stride: cutlass.Constexpr[int] = 1,
    ):
        """Store one output tile, one row-step at a time.

        ``max_rows`` bounds, at compile time, how many rows of this tile the
        runtime predicate below can admit; the caller must guarantee
        ``seqlen * qhead_per_kvhead - block * m_block_size <= max_rows``. Row
        steps past that bound would store nothing, so the loop simply stops.

        ``stage_from_smem`` makes ``tOrO`` a shared-memory tensor of the same
        partitioned shape instead of a register fragment, staged into registers
        one row-step at a time. Materializing the whole tile up front costs 256
        registers per thread, which spills when this runs on a single warp that
        has already dropped to a small register budget.

        ``row_stride`` says the tile's query rows were repeated on the way in,
        so head ``r`` is at row ``r * row_stride`` and the rows in between are
        duplicates that must not be stored. Every row of the tile is then
        reachable, so the caller's ``max_rows`` must cover the whole tile.
        """
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(
            mO[None, 0], tOcO_row, tidx, block, threads_per_row, num_threads, row_stride
        )
        num_row_steps = cute.size(tOrO.shape[1])
        if cutlass.const_expr(max_rows is not None):
            rows_per_step = num_threads // threads_per_row
            num_row_steps = min(num_row_steps, -(-max_rows // rows_per_step))
        for m in cutlass.range_constexpr(num_row_steps):
            o_ptr_i64 = utils.shuffle_sync(
                tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            o_gmem_ptr = cute.make_ptr(
                mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if cutlass.const_expr(stage_from_smem):
                tOrO_m = cute.make_fragment_like(tOrO[None, m, None], mO.element_type)
                cute.autovec_copy(tOrO[None, m, None], tOrO_m)
            else:
                tOrO_m = tOrO[None, m, None]
            if cutlass.const_expr(row_stride > 1):
                row_is_live = (t0OcO[0, m, 0][0] + tOcO_row[0][0]) % row_stride == 0
            else:
                row_is_live = (
                    t0OcO[0, m, 0][0]
                    < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]
                )
            if row_is_live:
                mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tOrO.shape[0][0])
                mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO_m[None, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
