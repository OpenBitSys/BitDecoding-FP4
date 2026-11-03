"""GPU state one model's NVFP4 layers share for their whole lifetime.

Three things have to outlive a step and be shared by every layer: the BF16
buffer holding each sequence's permanent sink and partial rolling page, the
views that reinterpret a layer's block array as NVFP4 regions, and the buffers
a layer quantizes its query into. The metadata builder cannot own them, because
it has no reference to the attention layers, so they are attached to the
layers and created the first time one of them runs.

That first time is deliberately the profile run, before the KV cache exists.
vLLM sizes the cache from whatever memory is still free once profiling ends, so
anything allocated later would come out of memory already promised to the
cache. Allocating during profiling makes the cache exactly that much smaller,
which is the honest accounting.

Carving the cache views cannot happen that early, since during profiling the
cache is a placeholder with no storage, so the views appear on the first real
write instead.
"""

from __future__ import annotations

import os

import torch
from nvfp4_decode_kernel import RESIDUAL_ROW_TILE, split_partial_count
from nvfp4_decode_kernel.quantize_kv_kernel import KVCacheWriter
from nvfp4_decode_kernel.scale_mode import DEFAULT_SCALE_MODE, sf_vec_size

from . import layout
from .builder import decode_num_splits
from .control import SINK_TOKENS, TAIL_BUFFER_TOKENS
from .rotation import block_hadamard_64, hadamard


PAGE_SIZE = 128

# Workers load the key-smoothing mean from this safetensors path. Its tensor
# is stored as ``key_shift`` with shape ``[num_layers, num_kv_heads, head_dim]``.
# When unset, no saved mean is loaded.
KEY_SHIFT_ENV = "NVFP4_KEY_SHIFT"


def load_key_shift(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rotation: torch.Tensor | None = None,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
) -> torch.Tensor | None:
    """A constant subtracted from post-RoPE K on its way into the cache.

    Post-RoPE K carries large per-channel offsets, and those offsets are what
    sets the group maximum that the block scale is derived from. Removing them
    before quantizing shrinks the scale, so every element in the group gets a
    finer step.

    A full-model sidecar is sliced along KV heads for tensor parallelism.
    Already rank-local sidecars keep their shape. Head replication is not
    inferred: only exactly ``num_kv_heads * tensor_parallel_size`` full-model
    heads are accepted.

    The sidecar is stored in the original basis. If ``rotation`` is present,
    this loader changes the constant to ``c R^T`` once, in float64, and rounds
    it once into ``dtype``.

    Nothing adds the constant back. Attention scores become ``q·k - q·mu``,
    and ``q·mu`` is the same for every key a given query sees, so softmax is
    unchanged. That argument only holds if *every* stored key is shifted, which
    is why the subtraction sits at the single point all three write paths pass
    through rather than inside the quantizer.

    Rounding ``mu`` was the obvious thing to be afraid of and measures as
    nothing: the invariance above holds for any constant, however coarse, and
    against captured post-RoPE K the bfloat16 constant removes 55.80% of the
    NVFP4 error where the float32 one removes 55.75%. The step it is rounded
    against is four-bit; a 2e-3 relative error on the constant disappears into
    it. So ``dtype`` is whatever the write path subtracts in for free: the
    cache's own for a shift applied as one BF16 ``sub``, float32 where the K
    writer already works in float32.
    """
    path = os.environ.get(KEY_SHIFT_ENV)
    if not path:
        return None
    if tensor_parallel_size < 1 or not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError(
            "invalid tensor-parallel coordinates: "
            f"rank={tensor_parallel_rank}, size={tensor_parallel_size}"
        )

    from safetensors.torch import load_file

    tensors = load_file(path)
    if "key_shift" not in tensors:
        raise ValueError(
            f"{path} has no 'key_shift' tensor, found {sorted(tensors)}"
        )
    shift = tensors["key_shift"]
    expected = (num_layers, num_kv_heads, head_dim)
    global_shape = (num_layers, num_kv_heads * tensor_parallel_size, head_dim)
    if tensor_parallel_size > 1 and tuple(shift.shape) == global_shape:
        # vLLM partitions unreplicated KV heads in contiguous rank order.
        start = tensor_parallel_rank * num_kv_heads
        shift = shift[:, start:start + num_kv_heads, :].contiguous()
    if tuple(shift.shape) != expected:
        raise ValueError(
            f"{path} holds a key_shift of {tuple(shift.shape)}, this model "
            f"needs {expected} (rank-local) or {global_shape} (full-model)"
        )
    return _active_shift(shift.to(device), rotation, dtype)


def _active_shift(
    shift: torch.Tensor, rotation: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor:
    """An original-basis mean as the write path subtracts it, rounded once."""
    if rotation is None:
        return shift.to(dtype)
    exact = torch.matmul(
        shift.double(), rotation.to(device=shift.device).double().T
    )
    return exact.to(dtype)


class LayerRuntime:
    """The tail buffers, cache views, and query scratch one model shares."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        query_terms: int = 1,
        rotation: str = "none",
        key_scale_mode: str = DEFAULT_SCALE_MODE,
        value_scale_mode: str = DEFAULT_SCALE_MODE,
        key_scale_geometry: str = "local",
    ) -> None:
        if query_terms not in (1, 2):
            raise ValueError(f"query_terms must be 1 or 2, got {query_terms!r}")
        if rotation not in ("none", "hadamard", "block_hadamard_64"):
            raise ValueError(
                "rotation must be 'none', 'hadamard', or "
                f"'block_hadamard_64', got {rotation!r}"
            )
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.query_terms = query_terms
        self.rotation_name = rotation
        self.key_scale_mode = key_scale_mode
        self.value_scale_mode = value_scale_mode
        if key_scale_geometry not in ("local", "gaussian_global"):
            raise ValueError(
                "key_scale_geometry must be 'local' or 'gaussian_global', "
                f"got {key_scale_geometry!r}"
            )
        self.key_scale_geometry = key_scale_geometry
        rotations = {
            "none": None,
            "hadamard": hadamard,
            "block_hadamard_64": block_hadamard_64,
        }
        constructor = rotations[rotation]
        self.rotation = (
            None if constructor is None else constructor(head_dim, device)
        )
        # ``hadamard`` is applied inside the query quantizer and the K writer,
        # in FP32 butterflies; any other rotation stays a BF16 GEMM, with this
        # execution view kept so that no layer pays a cast and an allocation
        # for it. The FP32 matrix remains canonical for sidecar transforms.
        self.fused_rotation = rotation == "hadamard"
        self._rotation_by_dtype: dict[torch.dtype, torch.Tensor] = {}
        if self.rotation is not None:
            self._rotation_by_dtype = {torch.float32: self.rotation}
            if not self.fused_rotation:
                self._rotation_by_dtype[torch.bfloat16] = self.rotation.to(
                    torch.bfloat16
                )
        # Held separately from the vector so that turning the shift off keeps
        # what it cost to measure. A/B-ing the two arms in one process is the
        # reason to want that.
        self.key_shift_enabled = True
        # Set only while measuring a shift, which cannot happen at the same
        # time as applying one: the sum has to be taken over the K the model
        # actually produces. Each layer accumulates its own row; the token
        # count is shared, since every layer sees the same tokens.
        self.key_shift_sum: torch.Tensor | None = None
        self.key_shift_tokens = 0

        # One allocation each, indexed by layer, so promotion can take the
        # layer as a grid dimension instead of looping in Python. Zeroed
        # rather than empty because attention reads a whole tail page back and
        # the positions past the sequence must not be denormals or NaNs.
        shape = (
            num_layers,
            num_slots,
            TAIL_BUFFER_TOKENS,
            num_kv_heads,
            head_dim,
        )
        self.tail_key = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.tail_value = torch.zeros(
            shape, dtype=torch.bfloat16, device=device
        )
        # The same storage seen as one run of tokens, which is what promotion
        # quantizes out of: layer ``l`` slot ``s`` starts at token
        # ``l * num_slots * TAIL_BUFFER_TOKENS
        # + s * TAIL_BUFFER_TOKENS``.
        # Loaded once the tail exists, because the shift is subtracted from K
        # on its way into that tail and carrying a second dtype would put a
        # cast in the write path.
        tp_size, tp_rank = 1, 0
        if os.environ.get(KEY_SHIFT_ENV) and torch.distributed.is_initialized():
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
        self.key_shift = load_key_shift(
            num_layers,
            num_kv_heads,
            head_dim,
            device,
            self.key_shift_dtype,
            rotation=self.rotation,
            tensor_parallel_size=tp_size,
            tensor_parallel_rank=tp_rank,
        )
        self.tail_key_tokens = self.tail_key.view(-1, num_kv_heads, head_dim)
        self.tail_value_tokens = self.tail_value.view(
            -1, num_kv_heads, head_dim
        )

        # Zeroed once here and never again, which is the contract fp4_decode
        # asks of a caller-owned buffer: the quantizer writes row 0 of each
        # tile and nothing writes the rest. Shared by every layer, since a
        # layer is done with it before the next one runs.
        self.query_padded = torch.zeros(
            num_slots,
            RESIDUAL_ROW_TILE,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        # The quantizer's two outputs, sized and shared the same way, so that
        # quantizing a query allocates and fills nothing. The packed query is
        # rewritten whole every call and what it holds now does not matter.
        # The scales need the zeroing more than query_padded does: the layout
        # gives each KV head 32 * 4 row slots and each four-scale k-atom a
        # four-byte scale word, and only the num_heads // num_kv_heads slots
        # that carry a query head are ever written. The guards keep Q in the
        # format K is in, so K's mode sizes the k-atoms.
        self.query_fp4 = torch.zeros(
            num_slots,
            1,
            num_heads,
            head_dim // 2,
            dtype=torch.uint8,
            device=device,
        )
        self.query_scales = torch.zeros(
            num_slots,
            1,
            num_heads,
            head_dim // (4 * sf_vec_size(key_scale_mode)),
            32,
            4,
            4,
            dtype=torch.uint8,
            device=device,
        )
        self.query_fp4_second = (
            torch.zeros_like(self.query_fp4) if query_terms == 2 else None
        )
        self.query_scales_second = (
            torch.zeros_like(self.query_scales) if query_terms == 2 else None
        )
        # The split path's partial outputs and LSEs, sized for the widest
        # split launch any decode batch this engine can form asks for, and
        # shared like the query scratch. fp4_decode needs them neither zeroed
        # nor refilled, so empty is the honest initial state. A runtime on the
        # CPU runs no decode kernel and gets none.
        sms = (
            torch.cuda.get_device_properties(device).multi_processor_count
            if torch.device(device).type == "cuda"
            else 0
        )
        # Every decode carries the BF16 tail, so a split launch may hold one
        # more partial per row than it has splits.
        split_rows = max(
            (
                rows
                * split_partial_count(
                    rows, num_kv_heads, splits, residual=True, sms=sms
                )
                for rows in range(1, num_slots + 1)
                if (splits := decode_num_splits(rows, num_kv_heads, sms)) > 1
            ),
            default=0,
        )
        self.split_partial = (
            torch.empty(
                split_rows * num_heads * head_dim,
                dtype=torch.float32,
                device=device,
            )
            if split_rows
            else None
        )
        self.split_lse = (
            torch.empty(split_rows * num_heads, dtype=torch.float32, device=device)
            if split_rows
            else None
        )

        self._views: dict[int, tuple[torch.Tensor, ...]] = {}
        self._by_layer: dict[int, tuple[torch.Tensor, ...]] = {}
        self._bases: torch.Tensor | None = None
        self._expected_layer = 0
        self._writer: KVCacheWriter | None = None
        self._writer_inputs: tuple = ()
        self._bound: dict[int, tuple] = {}
        self._promotion: tuple | None = None
        self._promotion_inputs: tuple = ()

    def rotation_for(self, dtype: torch.dtype) -> torch.Tensor | None:
        """The shared rotation in an operand's dtype, allocated only once."""
        if self.rotation is None:
            return None
        matrix = self._rotation_by_dtype.get(dtype)
        if matrix is None:
            matrix = self.rotation.to(dtype=dtype)
            self._rotation_by_dtype[dtype] = matrix
        return matrix

    @property
    def key_shift_dtype(self) -> torch.dtype:
        """What the shift is held in: see :func:`load_key_shift`."""
        return torch.float32 if self.fused_rotation else self.tail_key.dtype

    @property
    def query_rotation(self) -> str:
        """The rotation ``fp4_decode``'s quantizer applies to the BF16 query."""
        return "hadamard" if self.fused_rotation else "none"

    def transform_query(self, query: torch.Tensor) -> torch.Tensor:
        """The part of Q's basis change the quantizer does not do itself."""
        if self.rotation is None or self.fused_rotation:
            return query
        return torch.matmul(query, self.rotation_for(query.dtype).T)

    def key_write(
        self, layer_index: int, key: torch.Tensor
    ) -> tuple[torch.Tensor, bool, torch.Tensor | None]:
        """The K a cache write takes and what it still has to apply to it.

        The fused rotation and every shift are left to the writers, which do
        them in the same pass that copies K; only an unfused rotation is
        applied here, as a GEMM.
        """
        shift = self.active_key_shift
        if self.rotation is not None and not self.fused_rotation:
            return self.transform_key_for_cache(layer_index, key), False, None
        return (
            key,
            self.fused_rotation,
            None if shift is None else shift[layer_index],
        )

    @property
    def decode_writes_tail(self) -> bool:
        """Whether the decode prefix's quantizer writes its own tail tokens.

        It does whenever the writer would have rotated (or not) and shifted
        the raw K itself, with an FP32 shift; an unfused rotation hands the
        writer an already rotated K that the decode path never sees.
        """
        shift = self.active_key_shift
        return (self.rotation is None or self.fused_rotation) and (
            shift is None or shift.dtype == torch.float32
        )

    def writer_for(
        self, layer_index: int, kv_cache: torch.Tensor
    ) -> tuple[KVCacheWriter, tuple]:
        """The write path's writer and this layer's regions, bound for it.

        Built on first use and again whenever the shift
        in force changes, since the writer holds it; a layer's regions are
        checked and bound once, when its cache is first carved.
        """
        bound = self._bound.get(layer_index)
        if bound is None or bound[0] is not kv_cache:
            carved = self.views(layer_index, kv_cache)
            writer = self._current_writer(carved)
            bound = (kv_cache, writer.bind_regions(*carved))
            self._bound[layer_index] = bound
        return self._current_writer(None), bound[1]

    def _current_writer(self, carved) -> KVCacheWriter:
        unfused = self.rotation is not None and not self.fused_rotation
        inputs = (
            None if unfused else self.active_key_shift,
        )
        writer = self._writer
        if writer is not None and all(
            a is b for a, b in zip(inputs, self._writer_inputs)
        ):
            return writer
        (shift,) = inputs
        writer = KVCacheWriter(
            regions=carved if carved is not None else self._by_layer[min(self._by_layer)],
            key_scale_mode=self.key_scale_mode,
            value_scale_mode=self.value_scale_mode,
            key_scale_geometry=self.key_scale_geometry,
            rotate_key=self.fused_rotation,
            key_shift=shift,
            tail_key=self.tail_key,
            tail_value=self.tail_value,
            sink_tokens=SINK_TOKENS,
        )
        self._writer, self._writer_inputs = writer, inputs
        return writer

    def transform_key_for_cache(
        self, layer_index: int, key: torch.Tensor
    ) -> torch.Tensor:
        """Cache-bound K in the cache's basis, as a tensor of its own.

        Only for what the writers do not fuse: a shift alone, or a rotation
        applied as a GEMM. The fused Hadamard exists only inside the writers.
        """
        if self.fused_rotation:
            raise ValueError(
                "the fused Hadamard is applied by the cache writers, not here"
            )
        shift = self.active_key_shift
        rotation = self.rotation_for(key.dtype)
        if rotation is None:
            return key if shift is None else torch.sub(key, shift[layer_index])
        if shift is None:
            return torch.matmul(key, rotation.T)

        # baddbmm broadcasts one per-head shift across all tokens and fuses
        # ``k R^T - c R^T`` into the GEMM epilogue.
        return torch.baddbmm(
            shift[layer_index],
            key,
            rotation.T.expand(key.shape[0], -1, -1),
            beta=-1.0,
        )

    @property
    def active_key_shift(self) -> torch.Tensor | None:
        """What the write path should subtract, or None to store K as it is."""
        return self.key_shift if self.key_shift_enabled else None

    def set_key_shift_enabled(self, enabled: bool) -> None:
        """Turn the shift on or off between requests.

        Only between them. Keys already in the cache were stored under
        whichever setting was in force when they were written, so a sequence
        that spans a flip has two different constants in one softmax. The
        caller has to know the cache is drained; nothing here can check it.
        """
        if enabled and self.key_shift is None:
            raise ValueError(
                "no shift has been loaded or measured, so there is nothing "
                "to enable"
            )
        self.key_shift_enabled = enabled

    def begin_key_shift_measurement(self) -> None:
        """Measure the K mean instead of removing it, until told to stop.

        Any shift in force is dropped first. Measuring against a shifted K
        would give the residual mean rather than the mean, and the two are not
        interchangeable: the second measurement would be near zero and would
        look like there was nothing to remove.
        """
        self.key_shift = None
        self.key_shift_sum = torch.zeros(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float64,
            device=self.tail_key.device,
        )
        self.key_shift_tokens = 0

    def finish_key_shift_measurement(self) -> torch.Tensor:
        """Freeze what was measured and start subtracting it.

        Accumulated in float64. The returned value is the mean rounded once
        into the cache's dtype, in the original basis, for sidecar storage; the
        active value is the same mean rotated in float64 and rounded once into
        :attr:`key_shift_dtype`.
        """
        if self.key_shift_sum is None or self.key_shift_tokens == 0:
            raise ValueError(
                "no tokens were measured, so there is no mean to subtract; "
                "begin_key_shift_measurement must precede a forward pass"
            )
        mean = self.key_shift_sum / self.key_shift_tokens
        measured = mean.to(self.tail_key.dtype)
        self.key_shift = _active_shift(
            measured if self.rotation is None else mean,
            self.rotation,
            self.key_shift_dtype,
        )
        self.key_shift_enabled = True
        self.key_shift_sum = None
        self.key_shift_tokens = 0
        # The sidecar contract is always the model's original post-RoPE basis.
        return measured

    def views(
        self, layer_index: int, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """The four NVFP4 regions of one layer's block array.

        Carving is only address arithmetic, but it builds Python objects and
        this runs once per layer per step, so the result is kept. A layer's
        cache tensor is allocated once and never moves, which is what makes its
        address usable as the key.

        The layer index is recorded alongside, because promotion needs every
        layer's address in one table and this is the only place that learns
        which cache a layer actually uses. Taking it from here rather than
        from vLLM's module registry means promotion writes into the tensor the
        write path wrote into, under the index the write path used, without
        either of those having to be what vLLM believes.
        """
        key = kv_cache.data_ptr()
        carved = self._views.get(key)
        if carved is None:
            carved = layout.carve(
                kv_cache, self.num_kv_heads, self.head_dim, self.key_scale_mode
            )
            self._views[key] = carved
        if self._by_layer.get(layer_index) is not carved:
            self._by_layer[layer_index] = carved
            self._bases = None
        return carved

    @property
    def destination_bases(self) -> torch.Tensor:
        """``[4, num_layers]`` int64. Where each layer's four regions start.

        Built on first use and kept, because it is a host-to-device copy and
        promotion runs every step. Every layer must have written its cache by
        then, which the execution order below guarantees: promotion fires
        after the last layer, and each one carves its cache before it writes.
        """
        if self._bases is None:
            missing = [
                index
                for index in range(self.num_layers)
                if index not in self._by_layer
            ]
            if missing:
                raise ValueError(
                    f"layers {missing} have not written their KV cache, so "
                    "promotion has no address to send their pages to"
                )
            self._bases = torch.tensor(
                [
                    [
                        self._by_layer[layer][region].data_ptr()
                        for layer in range(self.num_layers)
                    ]
                    for region in range(4)
                ],
                dtype=torch.int64,
                device=self.tail_key.device,
            )
        return self._bases

    def promotion_writer(self) -> tuple[KVCacheWriter, tuple]:
        """The writer that seals filled tail pages for every layer at once.

        The write path's kernel with its page roles only, over the base table:
        the tail already holds K in the cache's basis, so it neither rotates
        nor shifts. Built on first use and again when a layer's cache is
        carved anew (a new table).
        Layer 0's regions give the layout; the table gives every address.
        """
        inputs = (
            self.destination_bases,
        )
        promotion = self._promotion
        if promotion is not None and all(
            a is b for a, b in zip(inputs, self._promotion_inputs)
        ):
            return promotion
        (bases,) = inputs
        regions = self._by_layer[0]
        writer = KVCacheWriter(
            regions=regions,
            key_scale_mode=self.key_scale_mode,
            value_scale_mode=self.value_scale_mode,
            key_scale_geometry=self.key_scale_geometry,
            destination_bases=bases,
        )
        self._promotion = (writer, writer.bind_regions(*regions))
        self._promotion_inputs = inputs
        return self._promotion

    def expect_layer(self, layer_index: int) -> None:
        """Refuse a step whose layers do not run in the order they are indexed.

        Promotion fires after the highest index, which is the last layer only
        if construction order is execution order. That holds for a decoder
        stack, and nothing in vLLM promises it. If it stopped holding, a page
        would be sealed into FP4 before some layer had written its share of
        the boundary token, and what reached the user would be a model that is
        slightly and inexplicably worse. Two integers a layer is cheaper than
        ever having to debug that.

        The count carries across steps rather than resetting at layer 0, so
        that a step which ran only some of its layers is caught by the next
        step's first layer arriving early. It resynchronizes when it complains,
        because the other way to leave it mid-stack is for something else to
        raise partway through a step, and a stale count would then answer every
        later step with a complaint about layer order instead of about whatever
        actually broke.
        """
        if layer_index != self._expected_layer:
            expected = self._expected_layer
            self._expected_layer = (layer_index + 1) % self.num_layers
            raise ValueError(
                f"layer {layer_index} ran where layer {expected} was expected; "
                "promotion assumes layers run in index order"
            )
        self._expected_layer = (layer_index + 1) % self.num_layers
