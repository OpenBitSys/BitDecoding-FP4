"""LSE-weighted reduction of split-K decode partials into the BF16 output."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32

LOG2_E = math.log2(math.e)


class SplitKCombine:
    """One CTA per decode row and query head, one thread per head-dim element.

    A thread loads every split's LSE and partial output for its element at
    once, so the reduction waits on L2 once whatever the split count. Walking
    the splits through a short pipeline instead costs a round trip every few
    splits, which at batch 1, where few CTAs each carry 32 or more splits, was
    most of the combine's time.

    A split that was handed no work wrote LSE -inf and left its partial output
    as the scratch held it, so a partial is summed only under a positive
    weight: NaN in an unwritten slot never reaches the output.
    """

    def __init__(self, head_dim: int, num_splits: int):
        self.head_dim = head_dim
        self.num_splits = num_splits

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # mO_partial (splits, rows, 1, heads, head_dim), mLSE_partial
        # (splits, rows, 1, heads), mO (rows, 1, heads, head_dim).
        self.kernel(mO_partial, mLSE_partial, mO).launch(
            grid=(mO.shape[2], mO.shape[0], 1),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
    ):
        dim, _, _ = cute.arch.thread_idx()
        head, row, _ = cute.arch.block_idx()
        weight = cute.make_rmem_tensor(cute.make_layout(self.num_splits), Float32)
        partial = cute.make_rmem_tensor(cute.make_layout(self.num_splits), Float32)
        for s in cutlass.range_constexpr(self.num_splits):
            weight[s] = mLSE_partial[s, row, 0, head]
            partial[s] = mO_partial[s, row, 0, head, dim]

        lse_max = weight[0]
        for s in cutlass.range_constexpr(1, self.num_splits):
            lse_max = cute.arch.fmax(lse_max, weight[s])
        # Every split empty: every weight is then exp2(-inf) = 0.
        lse_max = 0.0 if lse_max == -Float32.inf else lse_max
        weight_sum = Float32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            weight[s] = cute.math.exp2(
                weight[s] * LOG2_E - (lse_max * LOG2_E), fastmath=True
            )
            weight_sum += weight[s]
        inverse = (
            0.0 if (weight_sum == 0.0 or weight_sum != weight_sum) else 1.0 / weight_sum
        )

        # Split order, so a given set of partials always sums to the same bits.
        out = Float32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            scale = weight[s] * inverse
            if scale > 0.0:
                out = out + scale * partial[s]
        mO[row, 0, head, dim] = cutlass.BFloat16(out)


def combine_split_partials(out_partial, lse_partial, out=None):
    """Reduce ``out_partial`` ``(splits, rows, 1, heads, head_dim)`` FP32 under
    ``lse_partial`` ``(splits, rows, 1, heads)`` FP32 into BF16
    ``(rows, 1, heads, head_dim)``, written to ``out`` when given."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    num_splits, rows, seqlen, heads, head_dim = out_partial.shape
    if seqlen != 1:
        raise ValueError(f"the split-K combine serves one query position, got {seqlen}")
    if out is None:
        out = torch.empty(
            rows, seqlen, heads, head_dim,
            dtype=torch.bfloat16, device=out_partial.device,
        )
    # One kernel per split count: every split is a register and an exponential
    # per thread, so a padded tier would spend that on slots no split fills.
    # Serving launches one count per captured width.
    key = (head_dim, num_splits)

    def to_cute(tensor, leading_dim):
        return from_dlpack(
            tensor.detach(), assumed_align=16, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=leading_dim)

    cute_args = (
        to_cute(out_partial, 4),
        to_cute(lse_partial, 3),
        to_cute(out, 3),
    )
    compiled = combine_split_partials.compile_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            SplitKCombine(head_dim, num_splits),
            *cute_args,
            cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )
        combine_split_partials.compile_cache[key] = compiled
    # Nothing orders this against the attention kernel that wrote the partials
    # except sharing its stream, and the TVM FFI environment's is the default
    # one. That went unnoticed until a host that keeps its work off the default
    # stream — vLLM does — turned the read into a race against the write.
    compiled(
        *cute_args,
        cuda.CUstream(torch.cuda.current_stream(out_partial.device).cuda_stream),
    )
    return out


combine_split_partials.compile_cache = {}
