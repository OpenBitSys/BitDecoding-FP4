"""CuTeDSL K/V page quantization."""

import math
from typing import Tuple

import torch
import tvm_ffi

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ._block_scale import (
    GROUPS_PER_ATOM,
    _load_vector,
    gaussian_global_scale,
    quantize_atom,
    quantize_atom_with_scale,
)
from .quantize_q_kernel import _mul_rn, _sub_rn, _write_tail_token
from .scale_mode import DEFAULT_SCALE_MODE, scale_format, scale_rule, sf_vec_size


PAGE_SIZE = 128
HEAD_DIM = 128
SF_VEC_SIZE = 16
KEY_SCALE_GEOMETRIES = ("local", "gaussian_global")


@cute.jit
def _resolve_work(
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    work: Int32,
) -> Tuple[Int32, Int32]:
    """Map a grid slot to the token it reads and the page it writes.

    The two ends are independent, which is what a paged cache needs — the
    tokens of one page are somewhere in a flat activation buffer and the page
    they belong to is wherever the block table says. A negative destination
    marks a grid slot with no work, so the launch shape can be fixed while the
    batch varies.
    """
    return source_tokens[work], destination_pages[work]


@cute.jit
def _rebase(
    tensor: cute.Tensor,
    bases,
    region: cutlass.Constexpr[int],
    layer: Int32,
) -> cute.Tensor:
    """The same page layout, based wherever this layer's copy happens to be.

    A caller with one destination passes no table and gets the tensor back
    untouched. A caller with one destination per layer passes their base
    addresses, because independently allocated destinations share only their
    interior layout — their addresses are not a stride apart, so no single
    tensor can reach all of them. Row ``region`` of the table is this tensor's.
    """
    if const_expr(bases is None):
        return tensor
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint8,
            bases[region, layer],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        tensor.layout,
    )


@cute.jit
def _quantize_vector(
    values: cute.Tensor,
    destination: cute.Tensor,
    scales: cute.Tensor,
    page: Int32,
    lane: Int32,
    warp: Int32,
    head: Int32,
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    rule: cutlass.Constexpr,
    use_gaussian_global_scale: cutlass.Constexpr[bool],
):
    """Quantize one thread's vector to E2M1 with one scale per group.

    K and V reduce to this once their page slicing is done: K walks the head
    dimension of one token, V walks one head dimension across the page. The
    scale tile is addressed identically in both, which is what keeps the two
    directions numerically the same.

    ``values`` holds the vector in registers.
    """
    vector_scale = None
    if const_expr(use_gaussian_global_scale):
        vector_scale = gaussian_global_scale(values)
    for k_atom in cutlass.range_constexpr(rest_k):
        if const_expr(use_gaussian_global_scale):
            packed_scale = quantize_atom_with_scale(
                values,
                destination,
                k_atom,
                sf_vec_size,
                vector_scale,
            )
        else:
            packed_scale = quantize_atom(
                values, destination, k_atom, sf_vec_size, rule
            )
        scale_tile = scales[page, lane, warp, 0, None, k_atom, head]
        scale_word = cute.make_tensor(
            cute.recast_ptr(scale_tile.iterator, dtype=cutlass.Int32),
            cute.make_layout(1),
        )
        scale_word[0] = packed_scale


@cute.jit
def _cache_basis(
    values: cute.Tensor,
    shift_vector: cute.Tensor,
    rotate: cutlass.Constexpr[bool],
) -> cute.Tensor:
    """One token's K head as the cache stores it: rotated, then centered.

    The exact BF16 values are widened to FP32 and butterflied ``(a, c) ->
    (a + c, a - c)`` over the index bits, lowest first; one separately rounded
    multiply by the FP32 ``1 / sqrt(128)`` and one separately rounded subtract
    of the shift (already in the rotated basis, BF16 widened exactly) follow,
    and the result is rounded to BF16 once. A butterfly's result depends only
    on its operands' order, not on which thread holds them, so this is the
    decode quantizer's ``_write_tail_token`` arithmetic element for element.
    A contracted FMA in place of either rounded step moves 7 in 307,200 values
    by one BF16 step.
    """
    work = cute.make_rmem_tensor(cute.make_layout(HEAD_DIM), cutlass.Float32)
    work.store(values.load().to(cutlass.Float32))
    if const_expr(rotate):
        for bit in cutlass.range_constexpr(int(math.log2(HEAD_DIM))):
            stride = 1 << bit
            for j in cutlass.range_constexpr(HEAD_DIM):
                if const_expr((j & stride) == 0):
                    low = work[j]
                    high = work[j + stride]
                    work[j] = low + high
                    work[j + stride] = low - high
        inverse = cutlass.Float32(1.0 / math.sqrt(HEAD_DIM))
        for j in cutlass.range_constexpr(HEAD_DIM):
            work[j] = _mul_rn(work[j], inverse)
    if const_expr(shift_vector is not None):
        centre = cute.make_fragment_like(shift_vector)
        cute.autovec_copy(shift_vector, centre)
        for j in cutlass.range_constexpr(HEAD_DIM):
            work[j] = _sub_rn(work[j], cutlass.Float32(centre[j]))
    rounded = cute.make_rmem_tensor(cute.make_layout(HEAD_DIM), cutlass.BFloat16)
    rounded.store(work.load().to(cutlass.BFloat16))
    return rounded


@cute.jit
def _key_page(
    key: cute.Tensor,
    key_shift: cute.Tensor,
    key_pages_fp4: cute.Tensor,
    key_scales: cute.Tensor,
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    destination_bases: cute.Tensor,
    item: Int32,
    layer: Int32,
    source_offset: Int32,
    heads: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    rule: cutlass.Constexpr,
    use_gaussian_global_scale: cutlass.Constexpr[bool],
    rotate: cutlass.Constexpr[bool],
):
    """One head of one K page, one token's head vector per thread."""
    thread, _, _ = cute.arch.thread_idx()
    lane = thread % 32
    warp = thread // 32
    head = item % heads
    source_token, page = _resolve_work(
        source_tokens, destination_pages, item // heads
    )
    if page >= 0:
        source = key[source_token + source_offset + thread, head, None]
        values = _load_vector(source)
        if const_expr(rotate or key_shift is not None):
            shift_vector = None
            if const_expr(key_shift is not None):
                shift_vector = key_shift[layer, head, None]
            values = _cache_basis(values, shift_vector, rotate)
        _quantize_vector(
            values,
            _rebase(key_pages_fp4, destination_bases, 0, layer)[
                page, thread, head, None
            ],
            _rebase(key_scales, destination_bases, 1, layer),
            page,
            lane,
            warp,
            head,
            sf_vec_size,
            rest_k,
            rule,
            use_gaussian_global_scale,
        )


@cute.jit
def _value_page(
    value_pages: cute.Tensor,
    value_pages_fp4: cute.Tensor,
    value_scales: cute.Tensor,
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    destination_bases: cute.Tensor,
    item: Int32,
    layer: Int32,
    source_offset: Int32,
    heads: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    rest_k: cutlass.Constexpr[int],
    rule: cutlass.Constexpr,
):
    """One head of one V page, one channel across the page per thread."""
    thread, _, _ = cute.arch.thread_idx()
    head = item % heads
    source_token, page = _resolve_work(
        source_tokens, destination_pages, item // heads
    )
    if page >= 0:
        source = value_pages[source_token + source_offset, None, head, thread]
        _quantize_vector(
            _load_vector(source),
            _rebase(value_pages_fp4, destination_bases, 2, layer)[
                page, head, thread, None
            ],
            _rebase(value_scales, destination_bases, 3, layer),
            page,
            thread % 32,
            thread // 32,
            head,
            sf_vec_size,
            rest_k,
            rule,
            const_expr(False),
        )


@cute.jit
def _tail_token(
    key: cute.Tensor,
    value: cute.Tensor,
    key_shift: cute.Tensor,
    tail_key: cute.Tensor,
    tail_value: cute.Tensor,
    query_start_loc: cute.Tensor,
    seq_lens: cute.Tensor,
    seqused_fp4: cute.Tensor,
    row_to_slot: cute.Tensor,
    item: Int32,
    layer: Int32,
    decode_rows: Int32,
    first_row: Int32,
    heads: cutlass.Constexpr[int],
    rotate: cutlass.Constexpr[bool],
    sink_tokens: cutlass.Constexpr[int],
    tail_chunks: cutlass.Constexpr[int],
):
    """One token into the BF16 rolling page and/or sink.

    A one-token row in ``[first_row, decode_rows)`` gets one CTA for its
    token. Every later row gets ``PAGE_SIZE + sink_tokens`` candidates: one
    per rolling position past its FP4 prefix, then one per sink position, a
    sink candidate standing down where the rolling side already covers its
    token so that a token in both regions is written by one CTA, to both.
    Candidates past the row's length, before this step's tokens, or in a row
    with no slot exit. ``tail_chunks`` CTAs share a token when its groups
    outnumber one CTA's threads.
    """
    thread, _, _ = cute.arch.thread_idx()
    candidates = const_expr(PAGE_SIZE + sink_tokens)
    candidate = item // tail_chunks
    decode_count = decode_rows - first_row
    is_decode = candidate < decode_count
    later = candidate - decode_count
    row = Int32(
        cutlass.select_(
            is_decode, first_row + candidate, decode_rows + later // candidates
        )
    )
    within = later % candidates
    start = query_start_loc[row]
    seq_len = seq_lens[row]
    quantized = seqused_fp4[row]
    slot = row_to_slot[row]
    computed = seq_len - (query_start_loc[row + 1] - start)
    rolling = within < PAGE_SIZE
    position = Int32(
        cutlass.select_(
            is_decode,
            seq_len - 1,
            cutlass.select_(rolling, quantized + within, within - PAGE_SIZE),
        )
    )
    if (
        (slot >= 0)
        & (position < seq_len)
        & (position >= computed)
        & (is_decode | rolling | (position < quantized))
    ):
        shift = None
        if const_expr(key_shift is not None):
            shift = key_shift[layer, None, None]
        _write_tail_token(
            key,
            value,
            shift,
            tail_key[layer, None, None, None, None],
            tail_value[layer, None, None, None, None],
            start + position - computed,
            slot,
            position,
            position - quantized,
            (item % tail_chunks) * 128 + thread,
            heads,
            HEAD_DIM,
            rotate,
            key_shift is not None,
            sink_tokens,
        )


@cute.kernel
def _write_kv_kernel(
    key: cute.Tensor,
    value: cute.Tensor,
    value_pages: cute.Tensor,
    key_pages_fp4: cute.Tensor,
    key_scales: cute.Tensor,
    value_pages_fp4: cute.Tensor,
    value_scales: cute.Tensor,
    source_tokens: cute.Tensor,
    destination_pages: cute.Tensor,
    query_start_loc: cute.Tensor,
    seq_lens: cute.Tensor,
    seqused_fp4: cute.Tensor,
    row_to_slot: cute.Tensor,
    tail_key: cute.Tensor,
    tail_value: cute.Tensor,
    key_shift: cute.Tensor,
    destination_bases: cute.Tensor,
    layer_index: Int32,
    decode_rows: Int32,
    first_row: Int32,
    key_end: Int32,
    value_end: Int32,
    source_layer_stride: Int32,
    heads: cutlass.Constexpr[int],
    key_sf_vec_size: cutlass.Constexpr[int],
    key_rest_k: cutlass.Constexpr[int],
    key_rule: cutlass.Constexpr,
    value_sf_vec_size: cutlass.Constexpr[int],
    value_rest_k: cutlass.Constexpr[int],
    value_rule: cutlass.Constexpr,
    use_gaussian_global_scale: cutlass.Constexpr[bool],
    rotate: cutlass.Constexpr[bool],
    sink_tokens: cutlass.Constexpr[int],
    tail_chunks: cutlass.Constexpr[int],
):
    """K pages, then V pages, then tail tokens, on one flat CTA axis.

    The roles share no data and need no synchronization between CTAs. A
    per-layer launch (``destination_bases``) takes the layer from the grid,
    a one-layer launch from ``layer_index``, which also selects that layer's
    shift and tail.
    """
    item, _, grid_layer = cute.arch.block_idx()
    layer = layer_index
    if const_expr(destination_bases is not None):
        layer = grid_layer
    source_offset = grid_layer * source_layer_stride
    if item < key_end:
        if const_expr(key_pages_fp4 is not None):
            _key_page(
                key,
                key_shift,
                key_pages_fp4,
                key_scales,
                source_tokens,
                destination_pages,
                destination_bases,
                item,
                layer,
                source_offset,
                heads,
                key_sf_vec_size,
                key_rest_k,
                key_rule,
                use_gaussian_global_scale,
                rotate,
            )
    elif item < value_end:
        if const_expr(value_pages_fp4 is not None):
            _value_page(
                value_pages,
                value_pages_fp4,
                value_scales,
                source_tokens,
                destination_pages,
                destination_bases,
                item - key_end,
                layer,
                source_offset,
                heads,
                value_sf_vec_size,
                value_rest_k,
                value_rule,
            )
    else:
        if const_expr(tail_key is not None):
            _tail_token(
                key,
                value,
                key_shift,
                tail_key,
                tail_value,
                query_start_loc,
                seq_lens,
                seqused_fp4,
                row_to_slot,
                item - value_end,
                layer,
                decode_rows,
                first_row,
                heads,
                rotate,
                sink_tokens,
                tail_chunks,
            )


class _KVWrite:
    """The compile-time half of :class:`KVCacheWriter`: one launch configuration."""

    def __init__(
        self,
        *,
        heads: int,
        write_key: bool,
        write_value: bool,
        write_tail: bool,
        key_scale_mode: str,
        value_scale_mode: str,
        scale_geometry: str,
        rotate: bool,
        sink_tokens: int,
    ) -> None:
        self.heads = heads
        self.write_key = write_key
        self.write_value = write_value
        self.write_tail = write_tail
        self.key_sf_vec_size = sf_vec_size(key_scale_mode)
        self.key_rest_k = scale_atoms(HEAD_DIM, key_scale_mode)
        self.key_rule = scale_rule(key_scale_mode)
        self.value_sf_vec_size = sf_vec_size(value_scale_mode)
        self.value_rest_k = scale_atoms(PAGE_SIZE, value_scale_mode)
        self.value_rule = scale_rule(value_scale_mode)
        self.use_gaussian_global_scale = scale_geometry == "gaussian_global"
        self.rotate = rotate
        self.sink_tokens = sink_tokens
        # A token's K groups padded to whole warps, then its V groups.
        groups = heads * (HEAD_DIM // SF_VEC_SIZE)
        self.tail_chunks = ((groups + 31) // 32 * 32 + groups + 127) // 128
        self.smem = 0

    @cute.jit
    def __call__(
        self,
        key,
        value,
        key_pages_fp4,
        key_scales,
        value_pages_fp4,
        value_scales,
        source_tokens: cute.Tensor,
        destination_pages: cute.Tensor,
        query_start_loc,
        seq_lens,
        seqused_fp4,
        row_to_slot,
        tail_key,
        tail_value,
        key_shift,
        destination_bases,
        layer_index: Int32,
        decode_rows: Int32,
        first_row: Int32,
        stream,
    ):
        layers = Int32(1)
        if const_expr(destination_bases is not None):
            layers = destination_bases.shape[1]
        page_items = self.heads * source_tokens.shape[0]
        key_end = Int32(0)
        if const_expr(self.write_key):
            key_end = page_items
        value_end = key_end
        if const_expr(self.write_value):
            value_end = key_end + page_items
        tail_items = Int32(0)
        if const_expr(self.write_tail):
            tail_items = (
                (decode_rows - first_row)
                + (seq_lens.shape[0] - decode_rows) * (PAGE_SIZE + self.sink_tokens)
            ) * self.tail_chunks
        # One destination per layer means one source per layer too, stacked in
        # that order along the token axis.
        tokens = key
        if const_expr(key is None):
            tokens = value
        source_layer_stride = tokens.shape[0] // layers
        # V reads one channel down a page: a (page, token, head, dim) view whose
        # page axis has a one-token stride, so the leading index is a token
        # offset and a work item can start anywhere. The axes overlap, which is
        # harmless for a read.
        value_pages = None
        if const_expr(value is not None):
            value_pages = cute.make_tensor(
                value.iterator,
                cute.make_layout(
                    (source_tokens.shape[0], PAGE_SIZE, self.heads, HEAD_DIM),
                    stride=(value.stride[0], value.stride[0], HEAD_DIM, 1),
                ),
            )
        _write_kv_kernel(
            key,
            value,
            value_pages,
            key_pages_fp4,
            _as_bytes(key_scales),
            value_pages_fp4,
            _as_bytes(value_scales),
            source_tokens,
            destination_pages,
            query_start_loc,
            seq_lens,
            seqused_fp4,
            row_to_slot,
            tail_key,
            tail_value,
            key_shift,
            destination_bases,
            layer_index,
            decode_rows,
            first_row,
            key_end,
            value_end,
            source_layer_stride,
            const_expr(self.heads),
            const_expr(self.key_sf_vec_size),
            const_expr(self.key_rest_k),
            const_expr(self.key_rule),
            const_expr(self.value_sf_vec_size),
            const_expr(self.value_rest_k),
            const_expr(self.value_rule),
            const_expr(self.use_gaussian_global_scale),
            const_expr(self.rotate),
            const_expr(self.sink_tokens),
            const_expr(self.tail_chunks),
            # The per-layer form is promotion's; a profile tells it from the
            # write path's launches by name.
            _name_prefix="seal" if destination_bases is not None else None,
        ).launch(
            grid=(value_end + tail_items, 1, layers),
            block=(128, 1, 1),
            smem=self.smem,
            stream=stream,
        )


@cute.jit
def _as_bytes(scales):
    if const_expr(scales is None):
        return None
    return cute.make_tensor(
        cute.recast_ptr(scales.iterator, dtype=cutlass.Uint8), scales.layout
    )


_SCALE_TYPES = {
    torch.uint8: cutlass.Uint8,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
}
_compiled_writers: dict[tuple, object] = {}


def _compile_writer(
    *,
    heads: int,
    key_scales_dtype: torch.dtype | None,
    value_scales_dtype: torch.dtype | None,
    key_scale_mode: str,
    value_scale_mode: str,
    scale_geometry: str,
    write_tail: bool,
    shift_dtype: torch.dtype | None,
    rotate: bool,
    per_layer: bool,
    sink_tokens: int,
):
    """The compiled launch for one configuration; every page interior static.

    TVM-FFI checks each call's tensors against these declarations (dtype,
    rank, device, alignment, every static extent and stride, and equality of
    shared symbols), which is the per-call validation: only token pitches,
    block pitches, and counts are free. Shapes are 32-bit, pitches 64-bit,
    since a block index times a block pitch passes 2^31 at 14563 blocks.
    """
    cache_key = (
        torch.cuda.current_device(),
        heads,
        key_scales_dtype,
        value_scales_dtype,
        key_scale_mode,
        value_scale_mode,
        scale_geometry,
        write_tail,
        shift_dtype,
        rotate,
        per_layer,
        sink_tokens,
    )
    compiled = _compiled_writers.get(cache_key)
    if compiled is not None:
        return compiled

    def fake(dtype, shape, stride, align):
        return make_fake_tensor(dtype, shape, stride, assumed_align=align)

    sym32, sym64 = cute.sym_int32, cute.sym_int64
    write_key = key_scales_dtype is not None
    write_value = value_scales_dtype is not None

    def token_major():
        return fake(
            cutlass.BFloat16,
            (sym32(), heads, HEAD_DIM),
            (sym64(divisibility=8), HEAD_DIM, 1),
            16,
        )

    blocks, layers, work = sym32(), sym32(), sym32()

    def regions(packed_shape, scales_dtype, rest_k):
        packed = fake(
            cutlass.Uint8,
            (blocks,) + packed_shape,
            (sym64(divisibility=16),) + _dense_strides(packed_shape),
            16,
        )
        scales = fake(
            _SCALE_TYPES[scales_dtype],
            (blocks, 32, 4, 1, 4, rest_k, heads),
            (sym64(divisibility=16), 16, 4, rest_k * 512, 1, 512, rest_k * 512),
            16,
        )
        return packed, scales

    key = token_major() if write_key or write_tail else None
    value = token_major() if write_value or write_tail else None
    key_regions = (None, None)
    if write_key:
        key_regions = regions(
            (PAGE_SIZE, heads, HEAD_DIM // 2),
            key_scales_dtype,
            scale_atoms(HEAD_DIM, key_scale_mode),
        )
    value_regions = (None, None)
    if write_value:
        value_regions = regions(
            (heads, HEAD_DIM, PAGE_SIZE // 2),
            value_scales_dtype,
            scale_atoms(PAGE_SIZE, value_scale_mode),
        )
    rows, row_vectors, tails = sym32(), (None,) * 4, (None, None)
    if write_tail:
        row_vectors = (
            fake(Int32, (sym32(),), (1,), 4),
            *(fake(Int32, (rows,), (1,), 4) for _ in range(3)),
        )
        tail_shape = (layers, sym32(), PAGE_SIZE + sink_tokens, heads, HEAD_DIM)
        tail_strides = (sym64(), sym64()) + _dense_strides(tail_shape[2:])
        tails = (
            fake(cutlass.BFloat16, tail_shape, tail_strides, 16),
            fake(cutlass.BFloat16, tail_shape, tail_strides, 16),
        )
    shift = None
    if shift_dtype is not None:
        shift = fake(
            {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16}[
                shift_dtype
            ],
            (layers, heads, HEAD_DIM),
            (heads * HEAD_DIM, HEAD_DIM, 1),
            16,
        )
    bases = (
        fake(cutlass.Int64, (4, layers), (sym64(), 1), 8) if per_layer else None
    )
    launch = _KVWrite(
        heads=heads,
        write_key=write_key,
        write_value=write_value,
        write_tail=write_tail,
        key_scale_mode=key_scale_mode,
        value_scale_mode=value_scale_mode,
        scale_geometry=scale_geometry,
        rotate=rotate,
        sink_tokens=sink_tokens,
    )
    compiled = cute.compile(
        launch,
        key,
        value,
        *key_regions,
        *value_regions,
        fake(Int32, (work,), (1,), 4),
        fake(Int32, (work,), (1,), 4),
        *row_vectors,
        *tails,
        shift,
        bases,
        Int32(0),
        Int32(0),
        Int32(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _compiled_writers[cache_key] = compiled
    return compiled


def _validate_destination(
    pages_fp4: torch.Tensor,
    scales: torch.Tensor,
    *,
    heads: int,
    inner_shape: tuple[int, ...],
    rest_k: int,
) -> None:
    """Reject a destination whose page interior is not the quantizer's layout.

    Only the page pitch is free. Everything inside a page is the layout the
    decode kernel reads back, so a mismatch there is silent corruption.
    """
    if (
        pages_fp4.dtype is not torch.uint8
        or not pages_fp4.is_cuda
        or tuple(pages_fp4.shape[1:]) != inner_shape
        or pages_fp4.stride()[1:] != _dense_strides(inner_shape)
        or pages_fp4.stride(0) % 16
    ):
        raise ValueError(
            f"packed destination must be uint8 CUDA pages of {inner_shape}, "
            "dense within a page, 16-byte page pitch"
        )
    if (
        scales.dtype not in _SCALE_TYPES
        or scales.device != pages_fp4.device
        or tuple(scales.shape[1:]) != (32, 4, 1, 4, rest_k, heads)
        or scales.stride()[1:] != (16, 4, rest_k * 512, 1, 512, rest_k * 512)
        or scales.stride(0) % 16
    ):
        raise ValueError(
            "scale destination must use the quantizer's own scale layout"
        )
    if pages_fp4.shape[0] != scales.shape[0]:
        raise ValueError("packed and scale destinations must have equal pages")


class KVCacheWriter:
    """K/V into FP4 pages and the BF16 tail, one launch per call.

    Constructed once with everything that does not change between steps, and
    all of it is checked here: the scale modes, geometry, the tail buffers
    ``[layers, slots, 128 + sink_tokens, heads, 128]``, the shift
    ``[layers, heads, 128]`` (FP32, or BF16, already in the rotated basis),
    and the ``[4, layers]`` base-address table of a per-layer writer. A call
    passes only what changes: the source K/V, a layer's four regions (as
    :meth:`bind_regions` returned them, or as tensors, which TVM-FFI then
    converts on every call), the step's work table and row
    metadata, the layer index, the decode row count and the first row whose
    tail token this writes. TVM-FFI checks each call's tensors against the
    layouts compiled in.

    The roles present are those given: K pages with K regions, V pages with V
    regions, tail tokens with tail buffers. A per-layer writer (one with
    ``destination_bases``) quantizes the same work items for every layer at
    once from a source holding the layers back to back, each into the
    destination its row of the table names; it writes no tail.

    ``rotate_key`` and ``key_shift`` put raw K into the cache's basis on the
    way in, for the pages and the tail alike, with the arithmetic of
    :func:`_cache_basis`. A work item reads the 128 tokens from its source
    token and writes its destination page; a negative page skips it, which is
    how a fixed launch shape covers a varying batch.
    """

    def __init__(
        self,
        *,
        regions: tuple,
        key_scale_mode: str = DEFAULT_SCALE_MODE,
        value_scale_mode: str = DEFAULT_SCALE_MODE,
        key_scale_geometry: str = "local",
        rotate_key: bool = False,
        key_shift: torch.Tensor | None = None,
        tail_key: torch.Tensor | None = None,
        tail_value: torch.Tensor | None = None,
        sink_tokens: int = 0,
        destination_bases: torch.Tensor | None = None,
    ) -> None:
        key_packed, key_scales, value_packed, value_scales = regions
        any_packed = key_packed if key_packed is not None else value_packed
        if any_packed is None:
            raise ValueError("a writer needs K or V regions")
        heads = any_packed.shape[2 if key_packed is not None else 1]
        self.heads = heads
        self.key_scale_mode = key_scale_mode
        self.value_scale_mode = value_scale_mode
        rules = (scale_rule(key_scale_mode), scale_rule(value_scale_mode))
        if any(isinstance(rule, str) for rule in rules) and key_scale_geometry != "local":
            raise ValueError(
                "MXFP4 pages support local scales only"
            )
        if key_scale_geometry not in KEY_SCALE_GEOMETRIES:
            raise ValueError(
                f"unknown key_scale_geometry {key_scale_geometry!r}, expected "
                f"one of {list(KEY_SCALE_GEOMETRIES)}"
            )
        if key_packed is None and (key_shift is not None or rotate_key):
            raise ValueError("the K transform needs K regions")
        if (tail_key is None) != (tail_value is None):
            raise ValueError("tail_key and tail_value come together")
        if tail_key is not None and destination_bases is not None:
            raise ValueError("a per-layer writer writes no tail")
        device = any_packed.device
        layers = None

        def layer_axis(tensor: torch.Tensor, name: str, inner: tuple, dtypes):
            nonlocal layers
            if (
                tensor.dtype not in dtypes
                or tensor.device != device
                or not tensor.is_contiguous()
                or tuple(tensor.shape[1:]) != inner
            ):
                raise ValueError(
                    f"{name} must be contiguous {' or '.join(map(str, dtypes))} "
                    f"[layers, *{inner}] on {device}"
                )
            if layers is not None and tensor.shape[0] != layers:
                raise ValueError(
                    f"{name} has {tensor.shape[0]} layers, not {layers}"
                )
            layers = tensor.shape[0]

        if destination_bases is not None:
            if (
                destination_bases.dtype is not torch.int64
                or destination_bases.device != device
                or not destination_bases.is_contiguous()
                or destination_bases.ndim != 2
                or destination_bases.shape[0] != 4
            ):
                raise ValueError(
                    "destination_bases must be a contiguous int64 [4, layers] "
                    "table of K packed, K scale, V packed and V scale addresses"
                )
            layers = destination_bases.shape[1]
        if tail_key is not None:
            row = (PAGE_SIZE + sink_tokens, heads, HEAD_DIM)
            for name, tensor in (("tail_key", tail_key), ("tail_value", tail_value)):
                if (
                    tensor.dtype is not torch.bfloat16
                    or tensor.device != device
                    or not tensor.is_contiguous()
                    or tensor.ndim != 5
                    or tuple(tensor.shape[2:]) != row
                    or tensor.shape != tail_key.shape
                ):
                    raise ValueError(
                        f"{name} must be contiguous BF16 [layers, slots, "
                        f"{PAGE_SIZE + sink_tokens}, {heads}, {HEAD_DIM}] on {device}"
                    )
            layers = tail_key.shape[0]
        if key_shift is not None:
            layer_axis(
                key_shift,
                "key_shift",
                (heads, HEAD_DIM),
                (torch.float32, torch.bfloat16),
            )
        self._regions_dtypes = (
            None if key_scales is None else key_scales.dtype,
            None if value_scales is None else value_scales.dtype,
        )
        self.bind_regions(*regions)
        self._compiled = _compile_writer(
            heads=heads,
            key_scales_dtype=self._regions_dtypes[0],
            value_scales_dtype=self._regions_dtypes[1],
            key_scale_mode=key_scale_mode,
            value_scale_mode=value_scale_mode,
            scale_geometry=key_scale_geometry,
            write_tail=tail_key is not None,
            shift_dtype=None if key_shift is None else key_shift.dtype,
            rotate=bool(rotate_key),
            per_layer=destination_bases is not None,
            sink_tokens=sink_tokens,
        )
        self.writes_tail = tail_key is not None
        self.layers = layers if destination_bases is not None else None
        # Converted once: a tvm_ffi.Tensor crosses into the compiled call as a
        # pointer, where a torch tensor is converted on every call.
        self._constants = tuple(
            None if tensor is None else tvm_ffi.from_dlpack(tensor)
            for tensor in (
                tail_key,
                tail_value,
                key_shift,
                destination_bases,
            )
        )

    def bind_regions(
        self,
        key_pages_fp4: torch.Tensor | None,
        key_scales: torch.Tensor | None,
        value_pages_fp4: torch.Tensor | None,
        value_scales: torch.Tensor | None,
    ) -> tuple:
        """Check one layer's four regions once and keep them call-ready."""
        dtypes = (
            None if key_scales is None else key_scales.dtype,
            None if value_scales is None else value_scales.dtype,
        )
        if dtypes != self._regions_dtypes:
            raise ValueError(
                f"regions carry scale dtypes {dtypes}, this writer was built for "
                f"{self._regions_dtypes}"
            )
        if key_pages_fp4 is not None:
            _validate_destination(
                key_pages_fp4,
                key_scales,
                heads=self.heads,
                inner_shape=(PAGE_SIZE, self.heads, HEAD_DIM // 2),
                rest_k=scale_atoms(HEAD_DIM, self.key_scale_mode),
            )
        if value_pages_fp4 is not None:
            _validate_destination(
                value_pages_fp4,
                value_scales,
                heads=self.heads,
                inner_shape=(self.heads, HEAD_DIM, PAGE_SIZE // 2),
                rest_k=scale_atoms(PAGE_SIZE, self.value_scale_mode),
            )
        return tuple(
            None if region is None else tvm_ffi.from_dlpack(region)
            for region in (key_pages_fp4, key_scales, value_pages_fp4, value_scales)
        )

    def __call__(
        self,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        regions: tuple,
        source_tokens: torch.Tensor,
        destination_pages: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        seqused_fp4: torch.Tensor | None = None,
        row_to_slot: torch.Tensor | None = None,
        layer: int = 0,
        decode_rows: int = 0,
        first_row: int = 0,
    ) -> None:
        """Write this call's pages and tail tokens; launch nothing if there are none.

        ``key`` and ``value`` are ``[tokens, heads, 128]`` BF16, dense within a
        token; a per-layer writer's holds the layers back to back. Rows
        ``[0, decode_rows)`` of the row metadata are one-token rows; rows
        before ``first_row`` get no tail write.
        """
        if not destination_pages.shape[0] and (
            not self.writes_tail or seq_lens.shape[0] == first_row
        ):
            return
        if self.layers is not None:
            tokens = (key if key is not None else value).shape[0]
            if tokens % self.layers:
                # Rounding the per-layer stride down would shift every layer
                # but the first by a few tokens: a plausible-looking cache,
                # quietly out of step.
                raise ValueError(
                    f"a source of {tokens} tokens does not divide into "
                    f"{self.layers} layers"
                )
        tail_key, tail_value, key_shift, bases = self._constants
        self._compiled(
            key,
            value,
            *regions,
            source_tokens,
            destination_pages,
            query_start_loc,
            seq_lens,
            seqused_fp4,
            row_to_slot,
            tail_key,
            tail_value,
            key_shift,
            bases,
            layer,
            decode_rows,
            first_row,
        )


def scale_atoms(extent: int, scale_mode: str) -> int:
    """How many four-group scale atoms cover ``extent`` values under ``scale_mode``."""
    return extent // (GROUPS_PER_ATOM * sf_vec_size(scale_mode))


def scale_dtype(scale_mode: str) -> torch.dtype:
    """E4M3 for an NVFP4 mode, E8M0 for an MX mode."""
    if scale_format(scale_mode) == "mxfp4":
        return torch.float8_e8m0fnu
    return torch.float8_e4m3fn


def _validate_pages(pages: torch.Tensor, name: str) -> tuple[int, int]:
    if (
        pages.dtype is not torch.bfloat16
        or not pages.is_cuda
        or not pages.is_contiguous()
        or pages.ndim != 4
    ):
        raise ValueError(
            f"{name} must be contiguous BF16 CUDA with shape "
            "[pages, 128, heads, 128]"
        )
    page_count, page_size, heads, head_dim = pages.shape
    if page_count < 1 or page_size != PAGE_SIZE or head_dim != HEAD_DIM:
        raise ValueError(
            f"{name} must have shape [pages, 128, heads, 128]"
        )
    return page_count, heads


def _allocate_scales(
    page_count: int,
    heads: int,
    rest_m: int,
    rest_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    atom_bytes = 512
    m_block_stride = rest_k * atom_bytes
    head_stride = rest_m * m_block_stride
    page_stride = heads * head_stride
    shape = (page_count, 32, 4, rest_m, 4, rest_k, heads)
    strides = (
        page_stride,
        16,
        4,
        m_block_stride,
        1,
        atom_bytes,
        head_stride,
    )
    storage = torch.empty(
        page_count * page_stride,
        dtype=torch.uint8,
        device=device,
    )
    return storage, storage.as_strided(shape, strides)


def _dense_strides(shape: tuple) -> tuple:
    """Row-major strides. The leading extent, which may be symbolic, is not used."""
    strides = [1]
    for extent in reversed(shape[1:]):
        strides.append(strides[-1] * extent)
    return tuple(reversed(strides))


def _identity_work(page_count: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Work item ``w`` reads source page ``w`` and writes destination page ``w``."""
    pages = torch.arange(page_count, dtype=torch.int32, device=device)
    return pages * PAGE_SIZE, pages


def quantize_key_pages(
    key_pages: torch.Tensor,
    scale_mode: str = DEFAULT_SCALE_MODE,
    scale_geometry: str = "local",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 K pages into freshly allocated E2M1 and their scales."""
    page_count, heads = _validate_pages(key_pages, "key_pages")
    rest_k = scale_atoms(HEAD_DIM, scale_mode)
    key_pages_fp4 = torch.empty(
        page_count,
        PAGE_SIZE,
        heads,
        HEAD_DIM // 2,
        dtype=torch.uint8,
        device=key_pages.device,
    )
    _, key_scales = _allocate_scales(
        page_count,
        heads,
        1,
        rest_k,
        key_pages.device,
    )
    regions = (key_pages_fp4, key_scales, None, None)
    writer = KVCacheWriter(
        regions=regions,
        key_scale_mode=scale_mode,
        key_scale_geometry=scale_geometry,
    )
    writer(
        key_pages.reshape(-1, heads, HEAD_DIM),
        None,
        regions,
        *_identity_work(page_count, key_pages.device),
    )
    return key_pages_fp4, key_scales.view(scale_dtype(scale_mode))


def quantize_value_pages(
    value_pages: torch.Tensor,
    scale_mode: str = DEFAULT_SCALE_MODE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 V pages into freshly allocated E2M1 and their scales."""
    page_count, heads = _validate_pages(value_pages, "value_pages")
    rest_m = HEAD_DIM // 128
    page_rest_k = scale_atoms(PAGE_SIZE, scale_mode)
    value_pages_fp4 = torch.empty(
        page_count,
        heads,
        HEAD_DIM,
        PAGE_SIZE // 2,
        dtype=torch.uint8,
        device=value_pages.device,
    )
    _, value_scales = _allocate_scales(
        page_count,
        heads,
        rest_m,
        page_rest_k,
        value_pages.device,
    )
    regions = (None, None, value_pages_fp4, value_scales)
    writer = KVCacheWriter(regions=regions, value_scale_mode=scale_mode)
    writer(
        None,
        value_pages.reshape(-1, heads, HEAD_DIM),
        regions,
        *_identity_work(page_count, value_pages.device),
    )
    return value_pages_fp4, value_scales.view(scale_dtype(scale_mode))
