"""The block-scale search both quantizers run, and the conversions it needs.

A group of sixteen gets one E4M3 scale. Dividing its largest magnitude by six
puts that magnitude on the top rung of the E2M1 ladder; dividing by less
sacrifices it to place the rest of the group better. Which trade wins is a
property of the group, so this tries each divisor `scale_mode` names, keeps
the one whose round trip costs least, and stores that one. With a single
divisor the search collapses and the result is byte for byte what the
one-divisor quantizer stored.

Two arithmetic choices must match `reference.py` exactly:

- A candidate is judged by its reconstruction, read back through
  ``cvt.rn.f16x2.e2m1x2`` from the very bytes that would be stored. Nothing
  re-derives the rounding, so nothing can drift from it.
- The squared error accumulates through explicit ``fma.rn.f32``, in element
  order. Written as ``error + difference * difference`` the backend contracts
  it into an FMA anyway; spelling it out fixes the arithmetic against a
  backend that someday would not, and the last place it settles decides which
  candidate wins a group where two are level.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T


GROUPS_PER_ATOM = 4
GAUSSIAN_E2M1_NORM_SCALE = 0.04305219637254294


@cute.jit
def _load_vector(source: cute.Tensor) -> cute.Tensor:
    """Bring one thread's whole vector into registers in one vectorized copy."""
    values = cute.make_fragment_like(source)
    cute.autovec_copy(source, values)
    return values


@cute.jit
def _pack_e4m3(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b16 lo;\n\t"
        ".reg .b16 hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t"
        "}\n",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(packed)


@cute.jit
def _pack_e2m1(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    f4: Float32,
    f5: Float32,
    f6: Float32,
    f7: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
            Float32(f4).ir_value(loc=loc, ip=ip),
            Float32(f5).ir_value(loc=loc, ip=ip),
            Float32(f6).ir_value(loc=loc, ip=ip),
            Float32(f7).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b8 byte0;\n\t"
        ".reg .b8 byte1;\n\t"
        ".reg .b8 byte2;\n\t"
        ".reg .b8 byte3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n\t"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(packed)


@cute.jit
def _widen_e2m1_byte(
    packed: Int32,
    byte: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> Int32:
    """One byte of two E2M1 nibbles as an FP16 pair, low nibble in the low half."""
    widened = llvm.inline_asm(
        T.i32(),
        [Int32(packed).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 shifted;\n\t"
        ".reg .b8 nibbles;\n\t"
        ".reg .b32 pair;\n\t"
        f"shr.b32 shifted, $1, {8 * byte};\n\t"
        "cvt.u8.u32 nibbles, shifted;\n\t"
        "cvt.rn.f16x2.e2m1x2 pair, nibbles;\n\t"
        "mov.b32 $0, pair;\n\t"
        "}\n",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(widened)


@cute.jit
def _unpack_e2m1(packed: Int32):
    """The eight magnitudes a packed word encodes, in element order.

    FP16 holds every E2M1 value exactly, and so does FP32, so widening twice
    is lossless and what comes back is the grid point the store committed to.
    """
    halves = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float16)
    words = cute.recast_tensor(halves, cutlass.Int32)
    for byte in cutlass.range_constexpr(4):
        words[byte] = _widen_e2m1_byte(packed, byte)
    magnitudes = [Float32(0.0)] * 8
    for index in cutlass.range_constexpr(8):
        magnitudes[index] = Float32(halves[index])
    return magnitudes


@cute.jit
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b + c`` with one rounding, whatever the backend would have done."""
    fused = llvm.inline_asm(
        T.f32(),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
        ],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(fused)


@cute.jit
def _sqrt(value: Float32, *, loc=None, ip=None) -> Float32:
    """One correctly rounded FP32 square root."""
    result = llvm.inline_asm(
        T.f32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        "sqrt.rn.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(result)


@cute.jit
def gaussian_global_scale(values: cute.Tensor) -> Float32:
    """One Gaussian-optimal E2M1 scale for a complete 128-channel vector."""
    energy = Float32(0.0)
    for element in cutlass.range_constexpr(128):
        value = Float32(values[element])
        energy = _fma(value, value, energy)
    raw = _sqrt(energy) * Float32(GAUSSIAN_E2M1_NORM_SCALE)
    rounded = cute.make_rmem_tensor(
        cute.make_layout(GROUPS_PER_ATOM), cutlass.Float8E4M3FN
    )
    cute.recast_tensor(rounded, cutlass.Int32)[0] = _pack_e4m3(
        raw, raw, raw, raw
    )
    loaded = rounded.load().to(Float32)
    return Float32(loaded[0])


@cute.jit
def _round_trip(
    source: cute.Tensor,
    start: Int32,
    scale: Float32,
    sf_vec_size: cutlass.Constexpr[int],
):
    """One candidate over one group: the two words it stores and what they cost.

    A scale that flushed to zero throws every nonzero element past the top of
    the grid, where E2M1 saturates at six; multiplying back by the zero scale
    returns the group as zeros, which is what the decode path would read. The
    error then equals the group's own energy, so the candidate loses to any
    that can represent anything at all, and wins only where there is nothing
    to represent.
    """
    inverse = Float32(1.0) / (scale + Float32(1.0e-30))
    scaled = [Float32(0.0)] * 16
    for element in cutlass.range_constexpr(sf_vec_size):
        scaled[element] = Float32(source[start + element]) * inverse

    low = _pack_e2m1(
        scaled[0],
        scaled[1],
        scaled[2],
        scaled[3],
        scaled[4],
        scaled[5],
        scaled[6],
        scaled[7],
    )
    high = _pack_e2m1(
        scaled[8],
        scaled[9],
        scaled[10],
        scaled[11],
        scaled[12],
        scaled[13],
        scaled[14],
        scaled[15],
    )

    magnitudes = _unpack_e2m1(low) + _unpack_e2m1(high)
    error = Float32(0.0)
    for element in cutlass.range_constexpr(sf_vec_size):
        difference = _fma(
            magnitudes[element],
            scale,
            -Float32(source[start + element]),
        )
        error = _fma(difference, difference, error)
    return low, high, error


@cute.jit
def _candidate_scale(maximum: Float32, divisor: cutlass.Constexpr) -> Float32:
    """``maximum / divisor`` as the E4M3 scale it would be stored as.

    The kernel multiplies by a reciprocal constant where a reader might divide,
    and through E4M3's three mantissa bits the two disagree on some amax. The
    conversion is per element, so packing one value four times stores the byte
    that packing four different groups' values would have stored for it.
    """
    scaled = maximum * Float32(1.0 / divisor)
    rounded = cute.make_rmem_tensor(
        cute.make_layout(GROUPS_PER_ATOM), cutlass.Float8E4M3FN
    )
    cute.recast_tensor(rounded, cutlass.Int32)[0] = _pack_e4m3(
        scaled, scaled, scaled, scaled
    )
    loaded = rounded.load().to(Float32)
    return Float32(loaded[0])


@cute.jit
def select_group_scale(
    values: cute.Tensor,
    start,
    sf_vec_size: cutlass.Constexpr[int],
    divisors: cutlass.Constexpr,
):
    """Choose one group's scale and return it with the exact RTN words it won.

    A group's result depends on its own sixteen values and nothing else, which
    is what lets one thread own one group as well as one thread own a vector.

    ``divisors`` is in tie-break order: a candidate replaces the incumbent
    only on a strictly lower error, so the first divisor wins a tie and a
    longer tuple can only improve on a shorter prefix of itself.
    """
    maximum = Float32(0.0)
    for element in cutlass.range_constexpr(sf_vec_size):
        value = Float32(values[start + element])
        maximum = cute.arch.fmax(maximum, cute.arch.fmax(value, -value))

    scale = _candidate_scale(maximum, divisors[0])
    low, high, error = _round_trip(
        values, start, scale, sf_vec_size
    )
    for index in cutlass.range_constexpr(1, len(divisors)):
        rival = _candidate_scale(maximum, divisors[index])
        rival_low, rival_high, rival_error = _round_trip(
            values, start, rival, sf_vec_size
        )
        better = rival_error < error
        scale = Float32(cutlass.select_(better, rival, scale))
        low = Int32(cutlass.select_(better, rival_low, low))
        high = Int32(cutlass.select_(better, rival_high, high))
        error = Float32(cutlass.select_(better, rival_error, error))
    return scale, low, high


@cute.jit
def select_scales(
    values: cute.Tensor,
    k_atom: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    divisors: cutlass.Constexpr,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Choose four scales and return them with the exact RTN words they won.

    The quantizer stores ``words`` directly, preserving the reconstruction
    used to select each scale and the candidate tie order.
    """
    winners = cute.make_rmem_tensor(
        cute.make_layout(GROUPS_PER_ATOM), Float32
    )
    words = cute.make_rmem_tensor(
        cute.make_layout(2 * GROUPS_PER_ATOM), Int32
    )
    for group in cutlass.range_constexpr(GROUPS_PER_ATOM):
        group_index = k_atom * GROUPS_PER_ATOM + group
        scale, low, high = select_group_scale(
            values, group_index * sf_vec_size, sf_vec_size, divisors
        )
        winners[group] = scale
        words[2 * group] = low
        words[2 * group + 1] = high

    return winners, words


@cute.jit
def _store_atom(
    words: cute.Tensor,
    destination: cute.Tensor,
    k_atom: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
):
    """Store one atom's packed words (eight for NVFP4, sixteen for MXFP4) in one vector copy."""
    atom_byte_offset = k_atom * GROUPS_PER_ATOM * (sf_vec_size // 2)
    # The validated K/V layouts place every vector, and therefore every
    # atom within it, on a 16-byte boundary. Restoring that alignment
    # after the dynamic slice lets autovec_copy issue one coalesced store.
    atom_pointer = (destination.iterator + atom_byte_offset).align(16)
    atom_destination = cute.make_tensor(
        cute.recast_ptr(atom_pointer, dtype=cutlass.Int32),
        cute.make_layout(GROUPS_PER_ATOM * sf_vec_size // 8),
    )
    cute.autovec_copy(words, atom_destination)


MX_SF_VEC_SIZE = 32


@cute.jit
def _float_bits(value: Float32, *, loc=None, ip=None) -> Int32:
    bits = llvm.inline_asm(
        T.i32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(bits)


@cute.jit
def _bits_float(bits: Int32, *, loc=None, ip=None) -> Float32:
    value = llvm.inline_asm(
        T.f32(),
        [Int32(bits).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(value)


@cute.jit
def power_of_two(exponent: Int32) -> Float32:
    """``2**exponent`` for an E8M0 exponent; 2**-127 is the one subnormal."""
    normal = (exponent + Int32(127)) << Int32(23)
    return _bits_float(
        Int32(cutlass.select_(exponent < Int32(-126), Int32(1 << 22), normal))
    )


@cute.jit
def ocp_exponent(amax: Float32) -> Int32:
    """``floor(log2(amax)) - 2`` clamped to E8M0, read from the float's exponent field.

    A zero or subnormal amax has field 0 and lands on the clamp at -127, which
    is where the reference puts every group too small to represent.
    """
    exponent = ((_float_bits(amax) >> Int32(23)) & Int32(0xFF)) - Int32(129)
    return Int32(cutlass.select_(exponent < Int32(-127), Int32(-127), exponent))


@cute.jit
def round_up_exponent(amax: Float32) -> Int32:
    """The smallest E8M0 exponent whose scale keeps ``amax`` at or below six.

    ``amax > 6 * 2**ocp`` holds exactly when the mantissa is above 1.5, and
    adding 0x3FFFFF to the bits carries into the exponent field exactly then,
    so the rule costs an add and a shift on the softmax path. A zero, negative
    zero or subnormal amax lands on the clamp at -127.
    """
    exponent = ((_float_bits(amax) + Int32(0x3FFFFF)) >> Int32(23)) - Int32(129)
    return Int32(cutlass.select_(exponent < Int32(-127), Int32(-127), exponent))


@cute.jit
def inverse_power_of_two(exponent: Int32) -> Float32:
    """``2**-exponent`` for an E8M0 exponent below 127, always a normal float."""
    return _bits_float((Int32(127) - exponent) << Int32(23))


@cute.jit
def _mx_round_trip(
    source: cute.Tensor, start: Int32, exponent: Int32,
    width: cutlass.Constexpr[int] = 32,
):
    """An MX group, or one lane's half, on a shared power-of-two scale."""
    scale = power_of_two(exponent)
    inverse = power_of_two(Int32(0) - exponent)
    words = [Int32(0)] * (width // 8)
    for word in cutlass.range_constexpr(width // 8):
        base = start + 8 * word
        words[word] = _pack_e2m1(
            Float32(source[base]) * inverse,
            Float32(source[base + 1]) * inverse,
            Float32(source[base + 2]) * inverse,
            Float32(source[base + 3]) * inverse,
            Float32(source[base + 4]) * inverse,
            Float32(source[base + 5]) * inverse,
            Float32(source[base + 6]) * inverse,
            Float32(source[base + 7]) * inverse,
        )
    # Match Q's two-lane group: FMA within each half, then one FP32 sum.
    errors = [Float32(0.0)] * (width // 16)
    for half in cutlass.range_constexpr(width // 16):
        for word in cutlass.range_constexpr(2):
            index = 2 * half + word
            magnitudes = _unpack_e2m1(words[index])
            for element in cutlass.range_constexpr(8):
                difference = _fma(
                    magnitudes[element],
                    scale,
                    -Float32(source[start + 8 * index + element]),
                )
                errors[half] = _fma(difference, difference, errors[half])
    error = errors[0]
    if cutlass.const_expr(width == 32):
        error = error + errors[1]
    return words, error


@cute.jit
def mx_quantize_atom(
    values: cute.Tensor,
    destination: cute.Tensor,
    k_atom: cutlass.Constexpr[int],
    rule: cutlass.Constexpr[str],
    residual: cute.Tensor = None,
) -> Int32:
    """Quantize four 32-value groups to MXFP4 into ``destination``; return their E8M0 bytes.

    ``rule`` is ``ocp``, ``round_up`` or ``search``, decided exactly as
    ``reference.mxfp4_quantize`` decides it: ``round_up`` takes the doubled
    exponent when the group maximum exceeds six OCP scales
    (:func:`round_up_exponent`), ``search`` when the doubled exponent's squared
    error is strictly lower. With ``residual`` it also writes the exact FP32
    reconstruction error from the stored words and scale.
    """
    words = cute.make_rmem_tensor(
        cute.make_layout(GROUPS_PER_ATOM * MX_SF_VEC_SIZE // 8), Int32
    )
    packed = Int32(0)
    for group in cutlass.range_constexpr(GROUPS_PER_ATOM):
        start = (k_atom * GROUPS_PER_ATOM + group) * MX_SF_VEC_SIZE
        maximum = Float32(0.0)
        for element in cutlass.range_constexpr(MX_SF_VEC_SIZE):
            value = Float32(values[start + element])
            maximum = cute.arch.fmax(maximum, cute.arch.fmax(value, -value))
        if cutlass.const_expr(rule == "round_up"):
            exponent = round_up_exponent(maximum)
        else:
            exponent = ocp_exponent(maximum)
        group_words, error = _mx_round_trip(values, start, exponent)
        if cutlass.const_expr(rule == "search"):
            doubled = Int32(
                cutlass.select_(exponent < Int32(127), exponent + Int32(1), exponent)
            )
            doubled_words, doubled_error = _mx_round_trip(values, start, doubled)
            take = doubled_error < error
            exponent = Int32(cutlass.select_(take, doubled, exponent))
            for word in cutlass.range_constexpr(4):
                group_words[word] = Int32(
                    cutlass.select_(take, doubled_words[word], group_words[word])
                )
        for word in cutlass.range_constexpr(4):
            words[4 * group + word] = group_words[word]
        if cutlass.const_expr(residual is not None):
            scale = power_of_two(exponent)
            for word in cutlass.range_constexpr(4):
                magnitudes = _unpack_e2m1(group_words[word])
                for element in cutlass.range_constexpr(8):
                    index = start + 8 * word + element
                    residual[index] = _fma(
                        -magnitudes[element], scale, Float32(values[index])
                    )
        packed = packed | ((exponent + Int32(127)) << Int32(8 * group))
    _store_atom(words, destination, k_atom, MX_SF_VEC_SIZE)
    return packed


@cute.jit
def quantize_atom_with_scale(
    values: cute.Tensor,
    destination: cute.Tensor,
    k_atom: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    scale: Float32,
) -> Int32:
    """Quantize four groups with one already-rounded shared E4M3 scale."""
    words = cute.make_rmem_tensor(
        cute.make_layout(2 * GROUPS_PER_ATOM), Int32
    )
    for group in cutlass.range_constexpr(GROUPS_PER_ATOM):
        start = (k_atom * GROUPS_PER_ATOM + group) * sf_vec_size
        low, high, _ = _round_trip(
            values, start, scale, sf_vec_size
        )
        words[2 * group] = low
        words[2 * group + 1] = high
    _store_atom(words, destination, k_atom, sf_vec_size)
    return _pack_e4m3(scale, scale, scale, scale)


@cute.jit
def quantize_atom(
    values: cute.Tensor,
    destination: cute.Tensor,
    k_atom: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
    scale_rule: cutlass.Constexpr,
) -> Int32:
    """Quantize four groups into ``destination``; return their packed scales.

    ``scale_rule`` is ``scale_mode.scale_rule``: NVFP4 divisors, or an MX
    exponent rule, which hands the atom to :func:`mx_quantize_atom`.

    The caller stores the returned word wherever its scale layout puts this
    atom. It is packed last rather than first because which scale each group
    keeps is not known until every candidate has been priced.
    """
    if cutlass.const_expr(isinstance(scale_rule, str)):
        assert sf_vec_size == MX_SF_VEC_SIZE
        return mx_quantize_atom(values, destination, k_atom, scale_rule)
    divisors = scale_rule
    winners, words = select_scales(values, k_atom, sf_vec_size, divisors)
    _store_atom(words, destination, k_atom, sf_vec_size)

    # Re-rounding a value that came out of E4M3 returns it, so this reproduces
    # the byte the winning candidate was judged on rather than approaching it
    # a second time.
    return _pack_e4m3(winners[0], winners[1], winners[2], winners[3])


@cute.jit
def group_residual(
    values: cute.Tensor,
    residual: cute.Tensor,
    start,
    scale: Float32,
    low: Int32,
    high: Int32,
    sf_vec_size: cutlass.Constexpr[int],
):
    """Write ``values - decode(low, high) * scale`` for one group, exactly.

    An E2M1 magnitude carries two significant bits and an E4M3 scale four, so
    their product is exact in FP32 and the FMA rounds only the difference.
    """
    magnitudes = _unpack_e2m1(low) + _unpack_e2m1(high)
    for element in cutlass.range_constexpr(sf_vec_size):
        residual[start + element] = _fma(
            -magnitudes[element],
            scale,
            Float32(values[start + element]),
        )
