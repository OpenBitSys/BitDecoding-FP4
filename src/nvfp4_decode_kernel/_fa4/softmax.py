# Copyright (c) 2025, Tri Dao.

from typing import Optional
import math
import operator
import struct
from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from . import utils as utils
from .cute_dsl_utils import ParamsBase
from cutlass import const_expr

from .._block_scale import _fma, _widen_e2m1_byte, power_of_two, inverse_power_of_two
from ..p_scale import P_SMALL_GROUP_AMAX
from .blackwell_helpers import (
    packed_float_to_e2m1x2,
    packed_float_to_ue4m3,
)


@cute.jit
def _round_probability_scale(value: Float32) -> Float32:
    rounded = cute.make_rmem_tensor(
        cute.make_layout(4), cutlass.Float8E4M3FN
    )
    cute.recast_tensor(rounded, cutlass.Int32)[0] = packed_float_to_ue4m3(
        value, Float32(0.0), Float32(0.0), Float32(0.0)
    )
    return Float32(rounded.load().to(Float32)[0])


@cute.jit
def _round_probability_code(value: Float32) -> Float32:
    packed = packed_float_to_e2m1x2(Float32(0.0), value)
    widened = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Float16)
    cute.recast_tensor(widened, cutlass.Int32)[0] = _widen_e2m1_byte(
        packed, 0
    )
    return Float32(widened[0])


@cute.jit
def _mxfp4_probability_exponent(probability: Float32, exponent: cutlass.Int32) -> cutlass.Int32:
    """One P value per lane; the full warp shares one E8M0 scale.

    Compare round-up with the adjacent smaller scale. The signed error sum
    penalizes lost or added group mass without changing the softmax denominator.
    Strict comparison retains round-up on ties, including fully masked groups.
    """
    smaller = cutlass.Int32(cutlass.select_(exponent > -127, exponent - 1, exponent))
    best_cost = Float32(0.0)
    rival_cost = Float32(0.0)
    for candidate in cutlass.range_constexpr(2):
        chosen = exponent if cutlass.const_expr(candidate == 0) else smaller
        code = _round_probability_code(probability * inverse_power_of_two(chosen))
        difference = _fma(code, power_of_two(chosen), -probability)
        squared = difference * difference
        mass = difference
        for step in cutlass.range_constexpr(5):
            squared += cute.arch.shuffle_sync_bfly(squared, offset=1 << step)
            mass += cute.arch.shuffle_sync_bfly(mass, offset=1 << step)
        cost = _fma(mass * Float32(1.0 / 32.0), mass, squared)
        if cutlass.const_expr(candidate == 0):
            best_cost = cost
        else:
            rival_cost = cost
    return cutlass.Int32(cutlass.select_(rival_cost < best_cost, smaller, exponent))


def _ptx_f32(value: float) -> str:
    return "0f" + struct.pack(">f", value).hex()


_AMAX2_OR_AMAX6_PTX = (
    "{\n\t"
    ".reg .pred low;\n\t"
    f"setp.le.f32 low, $1, {_ptx_f32(P_SMALL_GROUP_AMAX)};\n\t"
    f"mul.rn.f32 $0, $1, {_ptx_f32(1.0 / 6.0)};\n\t"
    f"@low mul.rn.f32 $0, $1, {_ptx_f32(1.0 / 2.0)};\n\t"
    "}\n"
)


@cute.jit
def _amax2_or_amax6_probability_scale(
    maximum: Float32, *, loc=None, ip=None
) -> Float32:
    """Return amax/2 when ``maximum <= P_SMALL_GROUP_AMAX``, else amax/6."""
    # Keep the predicate and alternate multiply inside one inline block so the
    # compiler only has to keep the returned scale live. Materializing the
    # selected reciprocal in DSL IR crossed a register-allocation boundary in
    # the 128K kernel and cost more than the arithmetic itself.
    scale = llvm.inline_asm(
        T.f32(),
        [Float32(maximum).ir_value(loc=loc, ip=ip)],
        _AMAX2_OR_AMAX6_PTX,
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(scale)


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)

                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_max_prev = row_max[r]
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
                # row_scale[r] = utils.exp2f(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = utils.exp2f((row_max_prev - row_max_cur) * scale_log2)

                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_max[r] = row_max_cur
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        if cutlass.const_expr(sink_val is not None and isinstance(sink_val, cute.Tensor)):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                LOG2_E = math.log2(math.e)
                row_sum[r] += utils.exp2f(sink_val_cur * LOG2_E - row_max[r] * scale_log2)

            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + utils.log2f(row_sum_cur)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = utils.make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0

    fp8_scalexfp4_scale_log2: cutlass.Constexpr[float] = -11.392317422778762 # log2f(fp8_scalexfp4_scale=1.0 / (448 * 6))
    fp4_scale_log2: cutlass.Constexpr[float] = -2.584962500721156 # log2f(fp4_scale=1 / 6)
    quant_pv: cutlass.Constexpr[bool] = False
    compute_sp1: cutlass.Constexpr[bool] = False
    p_scale_divisor: cutlass.Constexpr[float] = 6.0
    p_scale_grid_search: cutlass.Constexpr[bool] = False
    p_scale_amax2_or_amax6: cutlass.Constexpr[bool] = False
    # Optional Kahan-compensated FP32 accumulation for row_sum.
    # Compensation register tracks lost low-order bits across the
    # online-softmax rescale + add steps. A single Float32 per softmax
    # thread; lives in registers, with no TMEM or SMEM cost.
    kahan: cutlass.Constexpr[bool] = False
    row_sum_comp: Optional[cute.Tensor] = None

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
        quant_pv: cutlass.Constexpr[bool] = False,
        compute_sp1: cutlass.Constexpr[bool] = False,
        p_scale_divisor: cutlass.Constexpr[float] = 6.0,
        p_scale_grid_search: cutlass.Constexpr[bool] = False,
        p_scale_amax2_or_amax6: cutlass.Constexpr[bool] = False,
        kahan: cutlass.Constexpr[bool] = False,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        row_sum_comp = cute.make_rmem_tensor(num_rows, Float32) if cutlass.const_expr(kahan) else None

        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
            quant_pv=quant_pv,
            compute_sp1=compute_sp1,
            p_scale_divisor=p_scale_divisor,
            p_scale_grid_search=p_scale_grid_search,
            p_scale_amax2_or_amax6=p_scale_amax2_or_amax6,
            kahan=kahan,
            row_sum_comp=row_sum_comp,
        )

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)
        if cutlass.const_expr(self.kahan):
            self.row_sum_comp.fill(0.0)

    @cute.jit
    def update_row_max(self, acc_S_row: cute.TensorSSA, is_first: int) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = utils.exp2f(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def compute_group_max(self, acc_S_row: cute.Tensor, sf_size: cutlass.Constexpr[int] = 16) -> cute.Tensor:
        """Per-group max for FP4 P quantization.

        Two bugs in the original code:
        1. No cross-thread reduce. The tcgen05 t2r load distributes a K group of 16
           elements across multiple lanes; each thread sees only its slice. The
           per-thread fmax_reduce returns the local max, not the per-group max,
           so SF_p stored to TMEM is wrong for most positions.
        2. Returned plain max instead of max/6, so scale_groupwise produced
           P/max ∈ [0, 1]. FP4 quantization in [0, 1] only ever uses nibbles
           {0, 0.5, 1.0} — 3 levels of precision per element, vs the full 8
           ({0, 0.5, 1, 1.5, 2, 3, 4, 6}) when scaling to [0, 6].

        Width-32 warp_reduce is empirically the best (cos jumps from 0.62 → 0.90).
        The exact lane sharing depends on the tcgen05 partition; over-reducing to
        the full warp is conservative (every group sees the warp-wide max) and
        loses some per-group adaptivity but correctness dominates.
        """
        acc_S_row_frag = cute.logical_divide(acc_S_row, cute.make_layout(sf_size))
        num_frags = cute.size(acc_S_row_frag, mode=[1])
        acc_S_row_group_max = cute.make_rmem_tensor(cute.make_layout(num_frags), Float32)
        for i in cutlass.range_constexpr(num_frags):
            # Each thread holds the entire K group (16 elements) for one M row of
            # the t2r tile, so per-thread fmax_reduce IS the per-group max — no
            # cross-lane reduce needed.
            values = acc_S_row_frag[None, i].load()
            maximum = self._compute_row_max(values)
            if cutlass.const_expr(self.p_scale_amax2_or_amax6):
                acc_S_row_group_max[i] = (
                    _amax2_or_amax6_probability_scale(maximum)
                )
            elif cutlass.const_expr(self.p_scale_grid_search):
                best_scale = maximum * Float32(1.0 / 6.0)
                best_stored = _round_probability_scale(best_scale)
                best_inverse = Float32(1.0) / cute.arch.fmax(
                    best_scale, 1e-20
                )
                best_error = Float32(0.0)
                for element in cutlass.range_constexpr(sf_size):
                    code = _round_probability_code(
                        Float32(values[element]) * best_inverse
                    )
                    difference = _fma(
                        code, best_stored, -Float32(values[element])
                    )
                    best_error = _fma(difference, difference, best_error)
                rival_divisors = (4.0, 3.0, 2.0)
                for candidate in cutlass.range_constexpr(
                    len(rival_divisors)
                ):
                    divisor = rival_divisors[candidate]
                    rival_scale = maximum * Float32(1.0 / divisor)
                    rival_stored = _round_probability_scale(rival_scale)
                    rival_inverse = Float32(1.0) / cute.arch.fmax(
                        rival_scale, 1e-20
                    )
                    rival_error = Float32(0.0)
                    for element in cutlass.range_constexpr(sf_size):
                        code = _round_probability_code(
                            Float32(values[element]) * rival_inverse
                        )
                        difference = _fma(
                            code, rival_stored, -Float32(values[element])
                        )
                        rival_error = _fma(
                            difference, difference, rival_error
                        )
                    better = rival_error < best_error
                    best_scale = Float32(
                        cutlass.select_(better, rival_scale, best_scale)
                    )
                    best_error = Float32(
                        cutlass.select_(better, rival_error, best_error)
                    )
                acc_S_row_group_max[i] = best_scale
            else:
                acc_S_row_group_max[i] = (
                    maximum * Float32(1.0 / self.p_scale_divisor)
                )
        return acc_S_row_group_max

    @cute.jit
    def scale_groupwise(self, acc_S_row: cute.Tensor, group_max: cute.Tensor, sf_size: cutlass.Constexpr[int] = 16):
        """Normalize P by per-group max before FP4 quantization.

        After this: P_fp4 * SFP ≈ (P/gmax) * gmax = P.
        """
        acc_S_row_frag = cute.logical_divide(acc_S_row, cute.make_layout(sf_size))
        for g in cutlass.range_constexpr(cute.size(group_max)):
            denominator = cute.arch.fmax(group_max[g], 1e-20)
            # The alternate transposed-S implementation already uses the
            # hardware reciprocal here. E2M1 has only two significant bits,
            # so its approximation does not buy representable precision, and
            # using it for the threshold candidate offsets the one predicate
            # that chooses /2 versus /6 in each hot P group.
            inv_gmax = (
                cute.arch.rcp_approx(denominator)
                if cutlass.const_expr(self.p_scale_amax2_or_amax6)
                else Float32(1.0) / denominator
            )
            for j in cutlass.range(0, sf_size, 2, unroll_full=True):
                acc_S_row_frag[j, g], acc_S_row_frag[j + 1, g] = utils.mul_packed_f32x2(
                    (acc_S_row_frag[j, g], acc_S_row_frag[j + 1, g]),
                    (inv_gmax, inv_gmax),
                )

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        if cutlass.const_expr(self.kahan):
            # Kahan-compensated row_sum update.
            # Captures lost bits across rescale + add. Equivalent to:
            #   row_sum_new = row_sum * row_scale + sum(acc_S_row_exp)
            # but tracks the rounding error in row_sum_comp so the
            # epilogue (`row_sum + row_sum_comp` before division into o)
            # recovers the lost low-order bits.
            #
            # is_first=True: row_sum = row_scale = 0; comp stays at 0.
            # is_first=False: standard Kahan around the scalar+vec sum:
            #   contrib = sum(acc_S_row_exp)
            #   init    = row_sum * row_scale
            #   comp    = comp * row_scale         # carry comp through rescale
            #   y       = contrib - comp
            #   t       = init + y
            #   comp    = (t - init) - y
            #   row_sum = t
            contrib = self._compute_row_sum(acc_S_row_exp, init_val=None)
            if cutlass.const_expr(is_first):
                # First block: row_sum = 0, comp stays 0. Skip Kahan to
                # match the legacy is_first=True semantics exactly.
                self.row_sum[0] = contrib
                self.row_sum_comp[0] = Float32(0.0)
            else:
                init = self.row_sum[0] * row_scale
                comp_rescaled = self.row_sum_comp[0] * row_scale
                y = contrib - comp_rescaled
                t = init + y
                self.row_sum_comp[0] = (t - init) - y
                self.row_sum[0] = t
        else:
            init_val = self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
            # self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=self.row_sum[0] * row_scale)
            self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)
            # tmp = self._compute_row_sum(acc_S_row_exp)

    @cute.jit
    def update_row_sum_sage(
        self, acc_S_row_exp: cute.Tensor, acc_S_row_group_max_exp: cute.Tensor, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        vec_size = const_expr(cute.size(acc_S_row_exp) // cute.size(acc_S_row_group_max_exp))
        acc_S_row_exp_frag = cute.logical_divide(acc_S_row_exp, cute.make_layout(vec_size))
        acc_S_group_sum = cute.make_rmem_tensor(acc_S_row_group_max_exp.layout, dtype=Float32)

        num_groups = cute.size(acc_S_row_exp_frag, mode=[1])
        assert num_groups % 2 == 0
        for i in cutlass.range_constexpr(0, num_groups, 2):
            acc_S_group_sum[i], acc_S_group_sum[i + 1] = utils.mul_packed_f32x2(
                (self._compute_row_sum(acc_S_row_exp_frag[None, i].load()), self._compute_row_sum(acc_S_row_exp_frag[None, i + 1].load())),
                (acc_S_row_group_max_exp[i], acc_S_row_group_max_exp[i + 1]),
            )
        self.row_sum[0]= self._compute_row_sum(acc_S_group_sum.load(), init_val=init_val)

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_group_max: Optional[cute.Tensor] = None,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        
        row_max_scaled = Float32(0.0)
        if const_expr(self.compute_sp1) and const_expr(self.quant_pv):
            row_max_scaled = row_max * self.scale_log2 + self.fp8_scalexfp4_scale_log2
        else:
            row_max_scaled = row_max * self.scale_log2

        if const_expr(acc_S_row_group_max is not None):
            if const_expr(self.compute_sp1) and const_expr(self.quant_pv):
                row_max_scaled += self.fp4_scale_log2 # / 6 to scale to fp4 max
            for i in cutlass.range(0, cute.size(acc_S_row_group_max.shape), 2, unroll_full=True):
                acc_S_row_group_max[i], acc_S_row_group_max[i + 1] = utils.fma_packed_f32x2(
                    (acc_S_row_group_max[i], acc_S_row_group_max[i + 1]),
                    (self.scale_log2, self.scale_log2),
                    (-row_max_scaled, -row_max_scaled),
                )
            vec_size = cute.size(acc_S_row) // cute.size(acc_S_row_group_max)
            acc_S_row_frag = cute.logical_divide(acc_S_row, cute.make_layout(vec_size))
            for j in cutlass.range_constexpr(0, cute.size(acc_S_row_frag, mode=[1])):
                for i in cutlass.range(0, cute.size(acc_S_row_frag, mode=[0]), 2, unroll_full=True):
                    bias = -row_max_scaled - acc_S_row_group_max[j]
                    acc_S_row_frag[i, j], acc_S_row_frag[i + 1, j] = utils.fma_packed_f32x2(
                        (acc_S_row_frag[i, j], acc_S_row_frag[i + 1, j]),
                        (self.scale_log2, self.scale_log2),
                        (bias, bias),
                    )
        else:
            for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
                acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
                    (acc_S_row[i], acc_S_row[i + 1]),
                    (self.scale_log2, self.scale_log2),
                    (-row_max_scaled, -row_max_scaled),
                )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: Optional[cute.Tensor] = None,
        e2e: cutlass.Constexpr[bool] = False,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        frg_tile = min(32, cute.size(acc_S_row))
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j] = utils.exp2f(acc_S_row_frg[k, j])
                # acc_S_row_frg[k + 1, j] = utils.exp2f(acc_S_row_frg[k + 1, j])
                if cutlass.const_expr(not e2e):
                    acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                    acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
                else:
                    if cutlass.const_expr(
                        k % e2e_freq < e2e_freq - e2e_res or j >= frg_cnt - e2e_frg_limit
                    ):
                        acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                        acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
                    else:
                        # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.e2e_asm2(acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
            if cutlass.const_expr(acc_S_row_converted is not None):
                acc_S_row_converted_frg = cute.logical_divide(
                    acc_S_row_converted, cute.make_layout(frg_tile)
                )
                acc_S_row_converted_frg[None, j].store(
                    acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
                )

    @cute.jit
    def apply_sage_sp1(self, P_row: cute.Tensor):
        row_max = self._compute_row_max(P_row.load())
        sp1 = row_max * self.sp1_scale
        inv_sp1 = 1.0 / sp1
        P_row.store(P_row.load() * inv_sp1)
        return sp1

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        # for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
        #     acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
        #         (acc_S_row[i], acc_S_row[i + 1]),
        #         (self.scale_log2, self.scale_log2),
        #         (minus_row_max_scaled, minus_row_max_scaled),
        #     )
        #     acc_S_row[i] = cute.arch.exp2(acc_S_row[i])
        #     acc_S_row[i + 1] = cute.arch.exp2(acc_S_row[i + 1])

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                #     utils.fma_packed_f32x2(
                #         (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                #         (self.scale_log2, self.scale_log2),
                #         (minus_row_max_scaled, minus_row_max_scaled),
                #     )
                # )
                # acc_S_row_frg[k, j] = utils.exp2f(acc_S_row_frg[k, j])
                # acc_S_row_frg[k + 1, j] = utils.exp2f(acc_S_row_frg[k + 1, j])
                acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )


@cute.jit
def floor_if_packed(
    q_idx,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Convert q_idx to packed format for Pack-GQA."""
    if cutlass.const_expr(qhead_per_kvhead == 1):
        return q_idx
    return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
):
    """Shared implementation for applying score modification.

    Args:
        score_tensor: The scores to modify (acc_S for flash_fwd, tSrS_t2r for sm100)
        index_tensor: Index positions (tScS for flash_fwd, tScS_t2r for sm100)
        score_mod: The score modification function to apply
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        constant_q_idx: If provided, use this constant for all q_idx values
                       If None, compute q_idx per-element
        qhead_per_kvhead_packgqa: Pack-GQA replication factor. Divide q_idx by this
                                  when greater than 1 so score mods see logical heads.
    """
    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # SSA values for batch (constant across all elements)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))

    # Handle q_idx based on whether it's constant
    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    # since a thread my process multiple query head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            score_vec[j] = score_tensor[i + j] * softmax_scale

            # Extract head offset from packed q_idx for Pack-GQA
            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][0]
                # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            # If we will do loads we mod, in order to not read OOB
            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(index_tensor[i + j][0], qhead_per_kvhead)
                    _, q_idx_wrapped = seqlen_q_divmod.divmod(q_idx_floored)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = seqlen_k_divmod.divmod(index_tensor[i + j][1])
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][0], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][1]

        # Convert to SSA for score_mod call
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()
        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            # NB we do not apply Pack-GQA division here, as constant_q_idx is assumed to already be logical
            q_idx_const = constant_q_idx
            q_idx_ssa = utils.scalar_to_ssa(q_idx_const, cutlass.Int32).broadcast_to((vec_size,))

        # Compute head_idx_ssa: per-element for Pack-GQA with non-constant q_idx, constant otherwise
        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        post_mod_scores = score_mod(
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            aux_tensors=aux_args,
        )

        # Write back modified scores
        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]
