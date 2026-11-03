"""CuTeDSL decode-query quantization."""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T

from ._block_scale import (
    GROUPS_PER_ATOM,
    _pack_e4m3,
    _float_bits,
    _mx_round_trip,
    group_residual,
    ocp_exponent,
    power_of_two,
    round_up_exponent,
    select_group_scale,
)
from .interface import DecodeTailWrite
from .scale_mode import DEFAULT_SCALE_MODE, scale_rule, sf_vec_size


PAGE_SIZE = 128
SF_VEC_SIZE = 16
THREADS = 128
QUERY_ROTATIONS = ("none", "hadamard")


@cute.jit
def _group_slice(vector: cute.Tensor, group, width: cutlass.Constexpr[int]):
    """One group's contiguous run of ``width`` elements of ``vector``.

    The host checks that every head vector starts on a 16-byte boundary, so a
    group of sixteen BF16 values or eight packed FP4 bytes is aligned to its own
    size and each copy is one or two vector accesses.
    """
    alignment = const_expr(min(16, width * vector.element_type.width // 8))
    return cute.make_tensor(
        (vector.iterator + group * width).align(alignment),
        cute.make_layout(width),
    )


@cute.jit
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b`` rounded on its own, never contracted into a later add."""
    product = llvm.inline_asm(
        T.f32(),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(product)


@cute.jit
def _sub_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    difference = llvm.inline_asm(
        T.f32(),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(difference)


@cute.jit
def _hadamard(
    values: cute.Tensor,
    group: Int32,
    lanes_per_head: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
):
    """Multiply one head by the normalized Sylvester Hadamard, in place.

    The ``±1`` butterflies of :func:`_butterflies`, then the single
    ``1 / sqrt(head_dim)`` multiply.
    """
    _butterflies(values, group, lanes_per_head)
    inverse = Float32(1.0 / math.sqrt(head_dim))
    for j in cutlass.range_constexpr(SF_VEC_SIZE):
        values[j] = Float32(values[j]) * inverse


@cute.jit
def _butterflies(
    values: cute.Tensor,
    group: Int32,
    lanes_per_head: cutlass.Constexpr[int],
):
    """One head times the unnormalized Sylvester Hadamard, in place.

    Channel ``SF_VEC_SIZE * group + j`` is register ``j`` of the head's lane
    ``group``. The low four index bits are butterflied in registers and the
    rest across the head's aligned lanes, lowest bit first, which is the order
    the reference reproduces. Each stage maps ``(a, c)`` at indices with the
    bit clear and set to ``(a + c, a - c)`` in FP32.
    """
    for bit in cutlass.range_constexpr(4):
        stride = 1 << bit
        for j in cutlass.range_constexpr(SF_VEC_SIZE):
            if const_expr((j & stride) == 0):
                low = values[j]
                high = values[j + stride]
                values[j] = low + high
                values[j + stride] = low - high
    for bit in cutlass.range_constexpr(int(math.log2(lanes_per_head))):
        mask = 1 << bit
        upper = (group & mask) != 0
        for j in cutlass.range_constexpr(SF_VEC_SIZE):
            own = Float32(values[j])
            partner = cute.arch.shuffle_sync_bfly(own, offset=mask)
            values[j] = Float32(
                cutlass.select_(upper, partner - own, own + partner)
            )


@cute.jit
def _write_tail_token(
    key: cute.Tensor,
    value: cute.Tensor,
    key_shift: cute.Tensor,
    tail_key: cute.Tensor,
    tail_value: cute.Tensor,
    token: Int32,
    slot: Int32,
    position: Int32,
    offset: Int32,
    item: Int32,
    heads_kv: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    rotate: cutlass.Constexpr[bool],
    shift: cutlass.Constexpr[bool],
    sink_tokens: cutlass.Constexpr[int],
):
    """One 16-element group of one token's new K or V, into the tail.

    ``token`` indexes ``key`` and ``value``; ``position`` is its place in the
    sequence and ``offset`` its place in the rolling page (negative: not in
    it). It is stored at ``offset`` when that is not negative and at
    ``PAGE_SIZE + position`` in the sink below ``sink_tokens``, both from this
    one call. K is rotated by FP32 butterflies, scaled by one separately
    rounded multiply, shifted by one separately rounded subtract of the FP32
    (or exactly widened BF16) shift, and rounded to BF16 once: the arithmetic
    of the K page writer, ``quantize_kv_kernel._cache_basis``.

    K groups fill whole warps before V starts, so the butterfly shuffles never
    see a lane that took the V side; K lanes past the last head clamp their
    head and only skip the store.
    """
    groups_per_head = const_expr(head_dim // SF_VEC_SIZE)
    items = const_expr(heads_kv * groups_per_head)
    key_span = const_expr((items + 31) // 32 * 32)

    if item < key_span:
        active = item < items
        head = Int32(
            cutlass.select_(active, item // groups_per_head, heads_kv - 1)
        )
        group = item % groups_per_head
        loaded = cute.make_rmem_tensor(
            cute.make_layout(SF_VEC_SIZE), cutlass.BFloat16
        )
        cute.autovec_copy(
            _group_slice(key[token, head, None], group, SF_VEC_SIZE), loaded
        )
        if const_expr(rotate or shift):
            values = cute.make_rmem_tensor(
                cute.make_layout(SF_VEC_SIZE), Float32
            )
            values.store(loaded.load().to(Float32))
            if const_expr(rotate):
                _butterflies(values, group, groups_per_head)
                inverse = Float32(1.0 / math.sqrt(head_dim))
                for j in cutlass.range_constexpr(SF_VEC_SIZE):
                    values[j] = _mul_rn(values[j], inverse)
            if const_expr(shift):
                centre = cute.make_rmem_tensor(
                    cute.make_layout(SF_VEC_SIZE), key_shift.element_type
                )
                cute.autovec_copy(
                    _group_slice(key_shift[head, None], group, SF_VEC_SIZE),
                    centre,
                )
                for j in cutlass.range_constexpr(SF_VEC_SIZE):
                    values[j] = _sub_rn(values[j], Float32(centre[j]))
            loaded.store(values.load().to(cutlass.BFloat16))
        if active:
            if slot >= 0:
                if offset >= 0:
                    cute.autovec_copy(
                        loaded,
                        _group_slice(
                            tail_key[slot, offset, head, None],
                            group,
                            SF_VEC_SIZE,
                        ),
                    )
                if position < sink_tokens:
                    cute.autovec_copy(
                        loaded,
                        _group_slice(
                            tail_key[slot, PAGE_SIZE + position, head, None],
                            group,
                            SF_VEC_SIZE,
                        ),
                    )
    else:
        v_index = item - key_span
        if v_index < items:
            v_head = v_index // groups_per_head
            v_group = v_index % groups_per_head
            v_loaded = cute.make_rmem_tensor(
                cute.make_layout(SF_VEC_SIZE), cutlass.BFloat16
            )
            cute.autovec_copy(
                _group_slice(value[token, v_head, None], v_group, SF_VEC_SIZE),
                v_loaded,
            )
            if slot >= 0:
                if offset >= 0:
                    cute.autovec_copy(
                        v_loaded,
                        _group_slice(
                            tail_value[slot, offset, v_head, None],
                            v_group,
                            SF_VEC_SIZE,
                        ),
                    )
                if position < sink_tokens:
                    cute.autovec_copy(
                        v_loaded,
                        _group_slice(
                            tail_value[slot, PAGE_SIZE + position, v_head, None],
                            v_group,
                            SF_VEC_SIZE,
                        ),
                    )


@cute.jit
def _scale_word(scale: Float32, slot: Int32) -> Int32:
    """The atom's four E4M3 scales as one word, gathered from its four lanes.

    Byte ``slot`` of the word belongs to group ``slot`` of the atom, and the
    atom's groups sit in four consecutive lanes, so two butterfly ORs give
    every one of them the whole word; lane ``slot == 0`` stores it.
    """
    byte = _pack_e4m3(scale, scale, scale, scale) & 0xFF
    word = Int32(byte << (slot * 8))
    word = word | cute.arch.shuffle_sync_bfly(word, offset=1)
    word = word | cute.arch.shuffle_sync_bfly(word, offset=2)
    return word


@cute.jit
def _store_term(
    scale: Float32,
    low: Int32,
    high: Int32,
    query_fp4: cute.Tensor,
    query_scales: cute.Tensor,
    row: Int32,
    head: Int32,
    group: Int32,
    active,
    qhead_per_kvhead: cutlass.Constexpr[int],
    is_mxfp4: cutlass.Constexpr[bool] = False,
):
    words = cute.make_rmem_tensor(cute.make_layout(2), Int32)
    words[0] = low
    words[1] = high
    if const_expr(is_mxfp4):
        # Two lanes share one scale; eight consecutive lanes make an atom.
        # The FP32 exponent bits already encode E8M0, including byte zero
        # for the single subnormal scale 2**-127.
        slot = group % (2 * GROUPS_PER_ATOM)
        byte = (_float_bits(scale) >> 23) & 0xFF
        word = Int32(byte << ((slot // 2) * 8))
        for bit in cutlass.range_constexpr(3):
            word = word | cute.arch.shuffle_sync_bfly(word, offset=1 << bit)
        scale_atom = group // (2 * GROUPS_PER_ATOM)
    else:
        slot = group % GROUPS_PER_ATOM
        word = _scale_word(scale, slot)
        scale_atom = group // GROUPS_PER_ATOM
    if active:
        destination = query_fp4[row, 0, head, None]
        cute.autovec_copy(
            words,
            cute.make_tensor(
                cute.recast_ptr(
                    _group_slice(destination, group, SF_VEC_SIZE // 2).iterator,
                    dtype=Int32,
                ),
                cute.make_layout(2),
            ),
        )
        if slot == 0:
            packed_row = head % qhead_per_kvhead
            scale_tile = query_scales[
                packed_row % 32,
                packed_row // 32,
                0,
                None,
                scale_atom,
                head // qhead_per_kvhead,
                row,
            ]
            scale_word = cute.make_tensor(
                cute.recast_ptr(scale_tile.iterator, dtype=Int32),
                cute.make_layout(1),
            )
            scale_word[0] = word


@cute.jit
def _select_mx_query_group(values: cute.Tensor, rule: cutlass.Constexpr[str]):
    """Two adjacent lanes choose one exponent, each holding sixteen values."""
    maximum = Float32(0.0)
    for i in cutlass.range_constexpr(SF_VEC_SIZE):
        value = Float32(values[i])
        maximum = cute.arch.fmax(maximum, cute.arch.fmax(value, -value))
    maximum = cute.arch.fmax(
        maximum, cute.arch.shuffle_sync_bfly(maximum, offset=1)
    )
    if const_expr(rule == "round_up"):
        exponent = round_up_exponent(maximum)
    else:
        exponent = ocp_exponent(maximum)
    words, error = _mx_round_trip(values, Int32(0), exponent, SF_VEC_SIZE)
    if const_expr(rule == "search"):
        error = error + cute.arch.shuffle_sync_bfly(error, offset=1)
        rival = Int32(cutlass.select_(exponent < 127, exponent + 1, exponent))
        other, other_error = _mx_round_trip(values, Int32(0), rival, SF_VEC_SIZE)
        other_error = other_error + cute.arch.shuffle_sync_bfly(other_error, offset=1)
        take = other_error < error
        exponent = Int32(cutlass.select_(take, rival, exponent))
        for word in cutlass.range_constexpr(2):
            words[word] = Int32(cutlass.select_(take, other[word], words[word]))
    return power_of_two(exponent), words[0], words[1]


@cute.kernel
def _quantize_query_kernel(
    query: cute.Tensor,
    row_indices: cute.Tensor,
    query_fp4: cute.Tensor,
    query_scales: cute.Tensor,
    query_fp4_second: cute.Tensor,
    query_scales_second: cute.Tensor,
    query_padded: cute.Tensor,
    key: cute.Tensor,
    value: cute.Tensor,
    key_shift: cute.Tensor,
    tail_key: cute.Tensor,
    tail_value: cute.Tensor,
    row_to_slot: cute.Tensor,
    seq_lens: cute.Tensor,
    seqused_fp4: cute.Tensor,
    heads: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    use_row_indices: cutlass.Constexpr[bool],
    write_padded: cutlass.Constexpr[bool],
    write_second: cutlass.Constexpr[bool],
    rotate: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr[int],
    divisors: cutlass.Constexpr,
    query_chunks: cutlass.Constexpr[int],
    write_tail: cutlass.Constexpr[bool],
    rotate_key: cutlass.Constexpr[bool],
    shift_key: cutlass.Constexpr[bool],
    sink_tokens: cutlass.Constexpr[int],
):
    """Each row's query heads, then optionally the row's new K/V token.

    The first ``query_chunks`` CTAs of a row quantize its query; the rest
    write its K/V token into the BF16 tail. Neither side reads what the other
    writes, so they run side by side with no synchronization.
    """
    thread, _, _ = cute.arch.thread_idx()
    row, chunk, _ = cute.arch.block_idx()
    if const_expr(write_tail):
        if chunk >= query_chunks:
            position = seq_lens[row] - 1
            _write_tail_token(
                key, value, key_shift, tail_key, tail_value,
                row, row_to_slot[row], position, position - seqused_fp4[row],
                (chunk - query_chunks) * THREADS + thread,
                heads // qhead_per_kvhead, rest_k * 64, rotate_key, shift_key,
                sink_tokens,
            )
        else:
            _quantize_query_group(
                query, row_indices, query_fp4, query_scales,
                query_fp4_second, query_scales_second, query_padded,
                row, chunk * THREADS + thread,
                heads, rest_k, use_row_indices, write_padded, write_second,
                rotate, qhead_per_kvhead, divisors,
            )
    else:
        _quantize_query_group(
            query, row_indices, query_fp4, query_scales,
            query_fp4_second, query_scales_second, query_padded,
            row, chunk * THREADS + thread,
            heads, rest_k, use_row_indices, write_padded, write_second,
            rotate, qhead_per_kvhead, divisors,
        )


@cute.jit
def _quantize_query_group(
    query: cute.Tensor,
    row_indices: cute.Tensor,
    query_fp4: cute.Tensor,
    query_scales: cute.Tensor,
    query_fp4_second: cute.Tensor,
    query_scales_second: cute.Tensor,
    query_padded: cute.Tensor,
    row: Int32,
    item: Int32,
    heads: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    use_row_indices: cutlass.Constexpr[bool],
    write_padded: cutlass.Constexpr[bool],
    write_second: cutlass.Constexpr[bool],
    rotate: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr[int],
    divisors: cutlass.Constexpr,
):
    """One thread per 16-element scale group of one row's query heads.

    NVFP4's scale, codes and residual depend on one lane's sixteen values.
    MXFP4 shares the maximum and candidate costs with the adjacent lane;
    Hadamard and the atom's packed scale word also cross lanes. Threads past the
    last head load a clamped head and take part in those shuffles, so every
    lane of a warp reaches them; only their stores are skipped.
    """
    groups_per_head = const_expr(rest_k * GROUPS_PER_ATOM)
    active = item < heads * groups_per_head
    head = Int32(cutlass.select_(active, item // groups_per_head, heads - 1))
    group = item % groups_per_head

    source_row = row
    if const_expr(use_row_indices):
        source_row = row_indices[row]
    loaded = cute.make_rmem_tensor(
        cute.make_layout(SF_VEC_SIZE), cutlass.BFloat16
    )
    cute.autovec_copy(
        _group_slice(query[source_row, head, None], group, SF_VEC_SIZE),
        loaded,
    )
    values = cute.make_rmem_tensor(cute.make_layout(SF_VEC_SIZE), Float32)
    values.store(loaded.load().to(Float32))

    if const_expr(rotate):
        _hadamard(values, group, groups_per_head, groups_per_head * SF_VEC_SIZE)

    if const_expr(write_padded):
        padded = loaded
        if const_expr(rotate):
            padded = cute.make_rmem_tensor(
                cute.make_layout(SF_VEC_SIZE), cutlass.BFloat16
            )
            padded.store(values.load().to(cutlass.BFloat16))
        if active:
            cute.autovec_copy(
                padded,
                _group_slice(query_padded[row, 0, head, None], group, SF_VEC_SIZE),
            )

    if const_expr(isinstance(divisors, str)):
        scale, low, high = _select_mx_query_group(values, divisors)
    else:
        scale, low, high = select_group_scale(values, 0, SF_VEC_SIZE, divisors)
    _store_term(
        scale, low, high, query_fp4, query_scales,
        row, head, group, active, qhead_per_kvhead, isinstance(divisors, str),
    )
    if const_expr(write_second):
        residual = cute.make_rmem_tensor(
            cute.make_layout(SF_VEC_SIZE), Float32
        )
        group_residual(values, residual, 0, scale, low, high, SF_VEC_SIZE)
        if const_expr(isinstance(divisors, str)):
            scale, low, high = _select_mx_query_group(residual, divisors)
        else:
            scale, low, high = select_group_scale(
                residual, 0, SF_VEC_SIZE, divisors
            )
        _store_term(
            scale, low, high, query_fp4_second, query_scales_second,
            row, head, group, active, qhead_per_kvhead, isinstance(divisors, str),
        )


@cute.jit
def _launch_quantize_query(
    query_ptr: cute.Pointer,
    row_indices_ptr: cute.Pointer,
    query_fp4_ptr: cute.Pointer,
    query_scales_ptr: cute.Pointer,
    query_fp4_second_ptr: cute.Pointer,
    query_scales_second_ptr: cute.Pointer,
    query_padded_ptr: cute.Pointer,
    query_shape: Tuple[Int32, Int32, Int32],
    query_strides: Tuple[Int32, Int32, Int32],
    row_indices_shape: Tuple[Int32],
    fp4_shape: Tuple[Int32, Int32, Int32, Int32],
    scales_shape: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    scales_strides: Tuple[
        Int32, Int32, Int32, Int32, Int32, Int32, Int32
    ],
    padded_shape: Tuple[Int32, Int32, Int32, Int32],
    key_ptr: cute.Pointer,
    value_ptr: cute.Pointer,
    key_shift_ptr: cute.Pointer,
    tail_key_ptr: cute.Pointer,
    tail_value_ptr: cute.Pointer,
    row_to_slot_ptr: cute.Pointer,
    seq_lens_ptr: cute.Pointer,
    seqused_fp4_ptr: cute.Pointer,
    key_shape: Tuple[Int32, Int32, Int32],
    key_strides: Tuple[Int32, Int32, Int32],
    value_strides: Tuple[Int32, Int32, Int32],
    tail_shape: Tuple[Int32, Int32, Int32, Int32],
    tail_strides: Tuple[Int32, Int32, Int32, Int32],
    rows: Int32,
    heads: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    use_row_indices: cutlass.Constexpr[bool],
    write_padded: cutlass.Constexpr[bool],
    write_second: cutlass.Constexpr[bool],
    rotate: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr[int],
    divisors: cutlass.Constexpr,
    write_tail: cutlass.Constexpr[bool],
    rotate_key: cutlass.Constexpr[bool],
    shift_key: cutlass.Constexpr[bool],
    sink_tokens: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    query = cute.make_tensor(
        query_ptr,
        cute.make_layout(query_shape, stride=query_strides),
    )
    row_indices = cute.make_tensor(
        row_indices_ptr,
        cute.make_ordered_layout(row_indices_shape, order=(0,)),
    )
    query_fp4 = cute.make_tensor(
        query_fp4_ptr,
        cute.make_ordered_layout(
            fp4_shape,
            order=tuple(range(len(fp4_shape) - 1, -1, -1)),
        ),
    )
    query_scales = cute.make_tensor(
        query_scales_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    query_fp4_second = cute.make_tensor(
        query_fp4_second_ptr,
        cute.make_ordered_layout(
            fp4_shape,
            order=tuple(range(len(fp4_shape) - 1, -1, -1)),
        ),
    )
    query_scales_second = cute.make_tensor(
        query_scales_second_ptr,
        cute.make_layout(scales_shape, stride=scales_strides),
    )
    query_padded = cute.make_tensor(
        query_padded_ptr,
        cute.make_ordered_layout(
            padded_shape,
            order=tuple(range(len(padded_shape) - 1, -1, -1)),
        ),
    )

    key = cute.make_tensor(
        key_ptr, cute.make_layout(key_shape, stride=key_strides)
    )
    value = cute.make_tensor(
        value_ptr, cute.make_layout(key_shape, stride=value_strides)
    )
    key_shift = cute.make_tensor(
        key_shift_ptr,
        cute.make_ordered_layout(
            (key_shape[1], key_shape[2]), order=(1, 0)
        ),
    )
    tail_key = cute.make_tensor(
        tail_key_ptr, cute.make_layout(tail_shape, stride=tail_strides)
    )
    tail_value = cute.make_tensor(
        tail_value_ptr, cute.make_layout(tail_shape, stride=tail_strides)
    )
    row_to_slot = cute.make_tensor(
        row_to_slot_ptr, cute.make_ordered_layout((rows,), order=(0,))
    )
    seq_lens = cute.make_tensor(
        seq_lens_ptr, cute.make_ordered_layout((rows,), order=(0,))
    )
    seqused_fp4 = cute.make_tensor(
        seqused_fp4_ptr, cute.make_ordered_layout((rows,), order=(0,))
    )

    query_chunks = const_expr(
        (heads * rest_k * GROUPS_PER_ATOM + THREADS - 1) // THREADS
    )
    # K's groups padded to whole warps, then V's.
    tail_items = const_expr(heads // qhead_per_kvhead * rest_k * GROUPS_PER_ATOM)
    tail_chunks = const_expr(
        ((tail_items + 31) // 32 * 32 + tail_items + THREADS - 1) // THREADS
        if write_tail
        else 0
    )
    _quantize_query_kernel(
        query,
        row_indices,
        query_fp4,
        query_scales,
        query_fp4_second,
        query_scales_second,
        query_padded,
        key,
        value,
        key_shift,
        tail_key,
        tail_value,
        row_to_slot,
        seq_lens,
        seqused_fp4,
        const_expr(heads),
        const_expr(rest_k),
        const_expr(use_row_indices),
        const_expr(write_padded),
        const_expr(write_second),
        const_expr(rotate),
        const_expr(qhead_per_kvhead),
        const_expr(divisors),
        const_expr(query_chunks),
        const_expr(write_tail),
        const_expr(rotate_key),
        const_expr(shift_key),
        const_expr(sink_tokens),
    ).launch(
        grid=(rows, query_chunks + tail_chunks, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


_launch_quantize_query.compile_cache = {}


def quantize_decode_q_to_padded_fp4(
    query: torch.Tensor,
    query_fp4_out: torch.Tensor,
    query_scales_out: torch.Tensor,
    query_padded_out: Optional[torch.Tensor] = None,
    *,
    query_fp4_out_2: Optional[torch.Tensor] = None,
    query_scales_out_2: Optional[torch.Tensor] = None,
    row_indices: Optional[torch.Tensor] = None,
    heads_kv: Optional[int] = None,
    scale_mode: str = DEFAULT_SCALE_MODE,
    rotation: str = "none",
    tail_write: Optional[DecodeTailWrite] = None,
) -> None:
    """Quantize selected BF16 query rows into preallocated outputs.

    ``scale_mode`` names the block-scale candidates; see :mod:`.scale_mode`.
    ``rotation="hadamard"`` first multiplies each head by the normalized
    Sylvester Hadamard in FP32; both terms are quantized from those FP32
    values, and the padded copy holds them rounded to BF16. Both arguments
    join the compile cache key, so switching costs one compilation and nothing
    per call. ``tail_write`` also places each row's new K/V token in the BF16
    tail from extra CTAs of the same launch; see :class:`DecodeTailWrite`.
    """
    divisors = scale_rule(scale_mode)
    scale_group_size = sf_vec_size(scale_mode)
    if rotation not in QUERY_ROTATIONS:
        raise ValueError(
            f"rotation must be one of {QUERY_ROTATIONS}, got {rotation!r}"
        )
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        raise ValueError("query must be a BF16 CUDA tensor")
    if query.ndim != 3 or query.stride(-1) != 1:
        raise ValueError("query must have shape [rows, heads, head_dim]")
    if (
        query.data_ptr() % 16
        or query.stride(0) % 8
        or query.stride(1) % 8
    ):
        raise ValueError(
            "every query head vector must start on a 16-byte boundary: "
            "the data pointer must be 16-byte aligned and the row and head "
            "strides multiples of 8 elements"
        )

    source_rows, heads, head_dim = query.shape
    heads_kv = heads if heads_kv is None else heads_kv
    if heads_kv < 1 or heads % heads_kv:
        raise ValueError("query heads must be divisible by heads_kv")
    qhead_per_kvhead = heads // heads_kv
    rows = query_fp4_out.shape[0]
    if head_dim % (GROUPS_PER_ATOM * scale_group_size) != 0:
        raise ValueError(
            f"query head_dim must be divisible by {GROUPS_PER_ATOM * scale_group_size}"
        )
    rotate = rotation == "hadamard"
    if rotate and (
        head_dim & (head_dim - 1) or head_dim // SF_VEC_SIZE > 32
    ):
        raise ValueError(
            "the fused Hadamard needs a power-of-two head_dim of at most "
            f"{32 * SF_VEC_SIZE}, got {head_dim}"
        )
    if row_indices is None:
        if source_rows != rows:
            raise ValueError("query rows must match output rows")
    elif (
        row_indices.dtype is not torch.int32
        or not row_indices.is_cuda
        or row_indices.device != query.device
        or not row_indices.is_contiguous()
        or row_indices.shape != (rows,)
    ):
        raise ValueError(
            "row_indices must be contiguous INT32 CUDA with shape [rows]"
        )

    expected_fp4_shape = (
        rows,
        1,
        heads,
        head_dim // 2,
    )
    expected_scales_shape = (
        32,
        4,
        1,
        4,
        head_dim // (GROUPS_PER_ATOM * scale_group_size),
        heads_kv,
        rows,
    )
    write_second = (
        query_fp4_out_2 is not None or query_scales_out_2 is not None
    )
    if (query_fp4_out_2 is None) != (query_scales_out_2 is None):
        raise ValueError(
            "query_fp4_out_2 and query_scales_out_2 must be provided together"
        )
    if (
        tuple(query_fp4_out.shape) != expected_fp4_shape
        or query_fp4_out.dtype is not torch.float4_e2m1fn_x2
        or not query_fp4_out.is_cuda
        or query_fp4_out.device != query.device
        or not query_fp4_out.is_contiguous()
    ):
        raise ValueError(
            f"query_fp4_out must have shape {expected_fp4_shape}"
        )
    if (
        tuple(query_scales_out.shape) != expected_scales_shape
        or query_scales_out.dtype is not torch.uint8
        or not query_scales_out.is_cuda
        or query_scales_out.device != query.device
    ):
        raise ValueError(
            f"query_scales_out must have shape {expected_scales_shape}"
        )
    if write_second:
        assert query_fp4_out_2 is not None
        assert query_scales_out_2 is not None
        if (
            tuple(query_fp4_out_2.shape) != expected_fp4_shape
            or query_fp4_out_2.dtype is not torch.float4_e2m1fn_x2
            or not query_fp4_out_2.is_cuda
            or query_fp4_out_2.device != query.device
            or not query_fp4_out_2.is_contiguous()
        ):
            raise ValueError(
                f"query_fp4_out_2 must have shape {expected_fp4_shape}"
            )
        if (
            tuple(query_scales_out_2.shape) != expected_scales_shape
            or query_scales_out_2.dtype is not torch.uint8
            or not query_scales_out_2.is_cuda
            or query_scales_out_2.device != query.device
        ):
            raise ValueError(
                f"query_scales_out_2 must have shape {expected_scales_shape}"
            )

    write_padded = query_padded_out is not None
    if write_padded:
        if (
            query_padded_out.ndim != 4
            or query_padded_out.shape[0] < rows
            or tuple(query_padded_out.shape[1:])
            != (PAGE_SIZE, heads, head_dim)
            or query_padded_out.dtype is not torch.bfloat16
            or not query_padded_out.is_cuda
            or query_padded_out.device != query.device
            or not query_padded_out.is_contiguous()
        ):
            raise ValueError(
                "query_padded_out must have shape "
                f"[at least {rows}, {PAGE_SIZE}, {heads}, {head_dim}]"
            )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    query_ptr = make_ptr(
        cutlass.BFloat16,
        query.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    if row_indices is None:
        row_indices_ptr = make_ptr(
            cutlass.Int32,
            query.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
    else:
        row_indices_ptr = make_ptr(
            cutlass.Int32,
            row_indices.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
    fp4_ptr = make_ptr(
        cutlass.Uint8,
        query_fp4_out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scales_ptr = make_ptr(
        cutlass.Uint8,
        query_scales_out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    fp4_second_ptr = make_ptr(
        cutlass.Uint8,
        (
            query_fp4_out_2.data_ptr()
            if query_fp4_out_2 is not None
            else query_fp4_out.data_ptr()
        ),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scales_second_ptr = make_ptr(
        cutlass.Uint8,
        (
            query_scales_out_2.data_ptr()
            if query_scales_out_2 is not None
            else query_scales_out.data_ptr()
        ),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    if query_padded_out is None:
        padded_ptr = query_ptr
        padded_shape = (
            Int32(rows),
            Int32(1),
            Int32(heads),
            Int32(head_dim),
        )
    else:
        padded_ptr = make_ptr(
            cutlass.BFloat16,
            query_padded_out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        padded_shape = tuple(
            Int32(value) for value in query_padded_out.shape
        )

    write_tail = tail_write is not None
    if write_tail:
        tail_args = _tail_write_arguments(
            tail_write, rows, heads_kv, head_dim, query.device, row_indices
        )
        rotate_key = bool(tail_write.rotate_key)
        shift_key = tail_write.key_shift is not None
        sink_tokens = int(tail_write.sink_tokens)
    else:
        # Never dereferenced: every use sits under the write_tail constexpr.
        tail_args = (
            query_ptr, query_ptr,
            make_ptr(cutlass.Float32, query.data_ptr(), cute.AddressSpace.gmem,
                     assumed_align=16),
            query_ptr, query_ptr,
            row_indices_ptr, row_indices_ptr, row_indices_ptr,
            (Int32(rows), Int32(heads_kv), Int32(head_dim)),
            (Int32(0), Int32(0), Int32(1)),
            (Int32(0), Int32(0), Int32(1)),
            (Int32(1), Int32(1), Int32(heads_kv), Int32(head_dim)),
            (Int32(0), Int32(0), Int32(0), Int32(1)),
        )
        rotate_key = shift_key = False
        sink_tokens = 0

    query_shape = tuple(Int32(value) for value in query.shape)
    query_strides = tuple(Int32(value) for value in query.stride())
    row_indices_shape = (Int32(rows),)
    fp4_shape = tuple(Int32(value) for value in expected_fp4_shape)
    scales_shape = tuple(Int32(value) for value in expected_scales_shape)
    scales_strides = tuple(
        Int32(value) for value in query_scales_out.stride()
    )
    # This counts four-lane groups, independently of the stored scale atoms.
    # MX still gives each lane sixteen values; its scale tile has half as
    # many atoms, but its Hadamard and tail-write geometry stay unchanged.
    rest_k = head_dim // 64
    use_row_indices = row_indices is not None

    compile_args = (
        query_ptr,
        row_indices_ptr,
        fp4_ptr,
        scales_ptr,
        fp4_second_ptr,
        scales_second_ptr,
        padded_ptr,
        query_shape,
        query_strides,
        row_indices_shape,
        fp4_shape,
        scales_shape,
        scales_strides,
        padded_shape,
        *tail_args,
        Int32(rows),
        heads,
        rest_k,
        use_row_indices,
        write_padded,
        write_second,
        rotate,
        qhead_per_kvhead,
        divisors,
        write_tail,
        rotate_key,
        shift_key,
        sink_tokens,
        stream,
    )
    compile_key = (
        torch.cuda.current_device(),
        heads,
        rest_k,
        use_row_indices,
        write_padded,
        write_second,
        rotate,
        qhead_per_kvhead,
        divisors,
        write_tail,
        rotate_key,
        shift_key,
        sink_tokens,
    )
    compiled = _launch_quantize_query.compile_cache.get(compile_key)
    if compiled is None:
        compiled = cute.compile(_launch_quantize_query, *compile_args)
        _launch_quantize_query.compile_cache[compile_key] = compiled

    compiled(
        query_ptr,
        row_indices_ptr,
        fp4_ptr,
        scales_ptr,
        fp4_second_ptr,
        scales_second_ptr,
        padded_ptr,
        query_shape,
        query_strides,
        row_indices_shape,
        fp4_shape,
        scales_shape,
        scales_strides,
        padded_shape,
        *tail_args,
        Int32(rows),
        stream,
    )


def _tail_write_arguments(
    tail_write: DecodeTailWrite,
    rows: int,
    heads_kv: int,
    head_dim: int,
    device: torch.device,
    row_indices: Optional[torch.Tensor],
) -> tuple:
    """Check a :class:`DecodeTailWrite` and turn it into launch arguments."""
    if row_indices is not None:
        raise ValueError("tail_write needs the query rows in batch order")
    key, value = tail_write.key, tail_write.value
    for name, tensor in (("key", key), ("value", value)):
        if (
            tensor.dtype is not torch.bfloat16
            or tensor.device != device
            or tensor.ndim != 3
            or tensor.shape[0] < rows
            or tuple(tensor.shape[1:]) != (heads_kv, head_dim)
            or tensor.stride(2) != 1
            or tensor.data_ptr() % 16
            or tensor.stride(0) % 8
            or tensor.stride(1) % 8
        ):
            raise ValueError(
                f"tail_write.{name} must be BF16 [>= {rows}, {heads_kv}, "
                f"{head_dim}] with 16-byte aligned head vectors"
            )
    tail_key, tail_value = tail_write.tail_key, tail_write.tail_value
    tail_shape = (PAGE_SIZE + tail_write.sink_tokens, heads_kv, head_dim)
    for name, tensor in (("tail_key", tail_key), ("tail_value", tail_value)):
        if (
            tensor.dtype is not torch.bfloat16
            or tensor.device != device
            or tensor.ndim != 4
            or tuple(tensor.shape[1:]) != tail_shape
            or tensor.stride() != tail_key.stride()
            or tensor.stride(3) != 1
            or tensor.data_ptr() % 16
            or any(stride % 8 for stride in tensor.stride()[:3])
        ):
            raise ValueError(
                f"tail_write.{name} must be BF16 [slots, *{tail_shape}] with "
                "the tail key's strides and 16-byte aligned head vectors"
            )
    for name in ("row_to_slot", "seq_lens", "seqused_fp4"):
        tensor = getattr(tail_write, name)
        if (
            tensor.dtype is not torch.int32
            or tensor.device != device
            or not tensor.is_contiguous()
            or tensor.ndim != 1
            or tensor.shape[0] < rows
        ):
            raise ValueError(
                f"tail_write.{name} must be contiguous INT32 with >= {rows} rows"
            )
    shift = tail_write.key_shift
    if shift is not None and (
        shift.dtype is not torch.float32
        or shift.device != device
        or tuple(shift.shape) != (heads_kv, head_dim)
        or not shift.is_contiguous()
    ):
        raise ValueError(
            f"tail_write.key_shift must be contiguous FP32 [{heads_kv}, {head_dim}]"
        )
    if tail_write.rotate_key and head_dim & (head_dim - 1):
        raise ValueError("the K rotation needs a power-of-two head_dim")

    def pointer(dtype, tensor, align):
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                        assumed_align=align)

    return (
        pointer(cutlass.BFloat16, key, 16),
        pointer(cutlass.BFloat16, value, 16),
        pointer(cutlass.Float32, key if shift is None else shift, 16),
        pointer(cutlass.BFloat16, tail_key, 16),
        pointer(cutlass.BFloat16, tail_value, 16),
        pointer(cutlass.Int32, tail_write.row_to_slot, 4),
        pointer(cutlass.Int32, tail_write.seq_lens, 4),
        pointer(cutlass.Int32, tail_write.seqused_fp4, 4),
        tuple(Int32(value) for value in key.shape),
        tuple(Int32(value) for value in key.stride()),
        tuple(Int32(value) for value in value.stride()),
        tuple(Int32(value) for value in tail_key.shape),
        tuple(Int32(value) for value in tail_key.stride()),
    )
