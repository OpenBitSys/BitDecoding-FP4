"""In-process vLLM engine wrapper for the NVFP4 KV-cache evaluation.

One :class:`VllmRunner` == one loaded model with a fixed ``kv_cache_dtype``.
The official vLLM NVFP4 KV cache is selected purely via
``kv_cache_dtype="nvfp4"`` (served by FlashInfer's trtllm-gen path); the BF16
baseline uses ``kv_cache_dtype="auto"``.  The repository CUSTOM backend is
selected by a bound mode in :mod:`evaluation.config`, which carries every
scheduler/cache argument that backend requires.

A mode that sets ``custom_backend`` is built through
:func:`nvfp4_vllm.build_llm` rather than through ``LLM`` directly. That
function owns the five settings the write path cannot work without and refuses
to be contradicted, which is the difference between an engine that fails to
start and one that starts and quietly serves from a different cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import List

from . import config


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        child for child in path.rglob("*") if child.is_file()
    )
    for child in files:
        relative = child.name if path.is_file() else child.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class VllmRunner:
    def __init__(self, model_path=None, *, kv_cache_dtype="auto",
                 max_model_len=None, gpu_memory_utilization=0.9,
                 tensor_parallel_size=1, enforce_eager=None, distributed_executor_backend=None,
                 attention_backend=None, max_num_seqs=None, block_size=None,
                 enable_prefix_caching=None, enable_chunked_prefill=None,
                 max_num_batched_tokens=None, pipeline_parallel_size=None,
                 scale_mode=None, rope_scaling=None,
                 logits_processors=None):
        from transformers import AutoTokenizer

        self.model_path = model_path or config.MODEL_PATH
        self.tensor_parallel_size = tensor_parallel_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if isinstance(kv_cache_dtype, dict):
            mode = dict(kv_cache_dtype)
        else:
            mode = {"kv_cache_dtype": str(kv_cache_dtype)}
        mode_kwargs = dict(mode.get("engine_kwargs", {}))
        kv_cache_dtype = str(mode["kv_cache_dtype"])
        self.kv_cache_dtype = kv_cache_dtype
        custom_backend = bool(mode.get("custom_backend"))
        mode_rope_scaling = mode.get("rope_scaling")
        if rope_scaling is not None and mode_rope_scaling is not None:
            if rope_scaling != mode_rope_scaling:
                raise ValueError("caller rope_scaling contradicts the named mode")
        selected_rope_scaling = self._validate_rope_scaling(
            mode_rope_scaling if mode_rope_scaling is not None else rope_scaling
        )

        self._check_sink(mode.get("sink_tokens"), custom_backend)

        if enforce_eager is None:
            enforce_eager = bool(mode_kwargs.get("enforce_eager", False))

        llm_kwargs = dict(
            model=self.model_path,
            dtype="bfloat16",
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=max_model_len or config.MAX_MODEL_LEN,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        if logits_processors is not None:
            llm_kwargs["logits_processors"] = logits_processors
        if selected_rope_scaling is not None:
            llm_kwargs["hf_overrides"] = {
                "rope_scaling": selected_rope_scaling,
            }
        optional_engine_kwargs = {
            "distributed_executor_backend": distributed_executor_backend,
            "attention_backend": attention_backend,
            "max_num_seqs": max_num_seqs,
            "block_size": block_size,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
            "max_num_batched_tokens": max_num_batched_tokens,
            "pipeline_parallel_size": pipeline_parallel_size,
        }
        llm_kwargs.update(
            {key: value for key, value in optional_engine_kwargs.items()
             if value is not None}
        )
        if mode_kwargs:
            # A mode's requirements win over generic caller overrides. In
            # particular, a CUSTOM mode must never be detached from CUSTOM.
            # Only unchunked prefill needs the entire prompt in one batch.
            if not mode_kwargs.get("enable_chunked_prefill", False):
                mode_kwargs["max_num_batched_tokens"] = max(
                    mode_kwargs["max_num_batched_tokens"],
                    llm_kwargs["max_model_len"],
                )
            llm_kwargs.update(mode_kwargs)

        # Submit at most one scheduler-width of requests at a time.  The
        # CUSTOM backend supports pure prefill and pure decode batches, but on
        # SM103 the bundled FA4 prefill path can abort when a heterogeneous
        # replacement prompt is JIT-compiled into an active decode batch.
        # Use the same grouping for every cache mode.
        self._request_batch_size = llm_kwargs.get("max_num_seqs")

        if custom_backend:
            self.llm = self._build_custom(llm_kwargs, mode, scale_mode)
        else:
            llm_kwargs.update(
                self._custom_backend_kwargs(
                    llm_kwargs, scale_mode=scale_mode
                )
            )
            self.llm = self._build_standard(llm_kwargs, mode)

        self.requested_rope_scaling = selected_rope_scaling


    @staticmethod
    def _validate_rope_scaling(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("rope_scaling must be a dictionary")
        result = copy.deepcopy(value)
        rope_type = result.get("rope_type", result.get("type"))
        if rope_type != "yarn":
            raise ValueError("only a static YaRN rope_scaling override is supported")
        factor = result.get("factor")
        original = result.get("original_max_position_embeddings")
        if not isinstance(factor, (int, float)) or factor <= 1:
            raise ValueError("YaRN factor must be greater than one")
        if not isinstance(original, int) or original <= 0:
            raise ValueError(
                "YaRN original_max_position_embeddings must be positive"
            )
        result.pop("type", None)
        result["rope_type"] = "yarn"
        return result

    def engine_contract(self) -> dict:
        """Requested and resolved positional settings for run manifests."""
        from nvfp4_decode_kernel.p_scale import (
            P_SMALL_GROUP_AMAX,
            p_scale_is_amax2_or_amax6,
            resolve_p_scale_environment,
        )
        p_scale_mode, p_scale_divisor, p_scale_grid_search = resolve_p_scale_environment()
        p_scale_tau = P_SMALL_GROUP_AMAX
        resolved = None
        max_model_len = None
        cache_dtype = None
        kv_cache_memory_bytes = None
        cache_skip_layers = None
        configured_backend = None
        block_size = None
        max_num_seqs = None
        max_num_batched_tokens = None
        enable_prefix_caching = None
        enable_chunked_prefill = None
        enable_flashinfer_autotune = None
        additional_config = None
        layer_cache_dtypes = Counter()
        layer_attention_backends = Counter()
        worker_layer_contracts = None
        worker_layer_contract_error = None
        resolved_executor_backend = None
        resolved_tp_size = self.tensor_parallel_size
        try:
            vllm_config = self.llm.llm_engine.vllm_config
            model_config = vllm_config.model_config
            parallel = getattr(vllm_config, "parallel_config", None)
            if parallel is not None:
                resolved_executor_backend = getattr(parallel, "distributed_executor_backend", None)
                resolved_tp_size = parallel.tensor_parallel_size
            max_model_len = int(model_config.max_model_len)
            hf_config = model_config.hf_config
            resolved = getattr(hf_config, "rope_scaling", None)
            if resolved is None:
                resolved = getattr(hf_config, "rope_parameters", None)
            cache = vllm_config.cache_config
            kv_cache_memory_bytes = getattr(cache, "kv_cache_memory_bytes", None)
            cache_dtype = str(cache.cache_dtype)
            cache_skip_layers = list(getattr(cache, "kv_cache_dtype_skip_layers", []) or [])
            block_size = int(cache.block_size)
            enable_prefix_caching = bool(cache.enable_prefix_caching)
            scheduler = vllm_config.scheduler_config
            max_num_seqs = int(scheduler.max_num_seqs)
            max_num_batched_tokens = int(scheduler.max_num_batched_tokens)
            enable_chunked_prefill = bool(scheduler.enable_chunked_prefill)
            enable_flashinfer_autotune = bool(
                getattr(vllm_config.kernel_config, "enable_flashinfer_autotune", False)
            )
            selected = getattr(vllm_config.attention_config, "backend", None)
            additional_config = copy.deepcopy(
                getattr(vllm_config, "additional_config", None)
            )
            configured_backend = (
                getattr(selected, "value", None) or str(selected)
                if selected is not None else None
            )
        except AttributeError:
            pass
        try:
            core = self.llm.llm_engine.engine_core
            while hasattr(core, "engine_core"):
                core = core.engine_core
            model = core.model_executor.driver_worker.worker.model_runner.model
            for module in model.modules():
                layer_dtype = getattr(module, "kv_cache_dtype", None)
                backend = getattr(module, "attn_backend", None)
                if layer_dtype is None or backend is None:
                    continue
                layer_cache_dtypes[str(layer_dtype)] += 1
                layer_attention_backends[str(backend.get_name())] += 1
        except (AttributeError, TypeError):
            pass
        if not layer_cache_dtypes or not layer_attention_backends:
            try:
                worker_layer_contracts = self.llm.collective_rpc(
                    "nvfp4_layer_cache_contract"
                )
                if not worker_layer_contracts:
                    raise RuntimeError("worker RPC returned no contracts")
                canonical = worker_layer_contracts[0]
                if any(item != canonical for item in worker_layer_contracts[1:]):
                    raise RuntimeError(
                        "tensor-parallel workers reported different layer contracts: "
                        f"{worker_layer_contracts}"
                    )
                layer_cache_dtypes.clear()
                layer_attention_backends.clear()
                layer_cache_dtypes.update(
                    canonical.get("layer_cache_dtype_counts") or {}
                )
                layer_attention_backends.update(
                    canonical.get("layer_attention_backend_counts") or {}
                )
            except Exception as error:
                worker_layer_contract_error = (
                    f"{type(error).__name__}: {error}"
                )
        probability_format = None
        probability_mode = None
        if layer_attention_backends.get("CUSTOM"):
            from nvfp4_decode_kernel.scale_mode import scale_format

            mode = (additional_config or {}).get("nvfp4_scale_mode", "amax6")
            if isinstance(mode, dict):
                mode = mode["key"]
            probability_format = scale_format(mode)
            probability_mode = (
                "adjacent_sse_mass" if probability_format == "mxfp4" else p_scale_mode
            )
        is_mxfp4 = probability_format == "mxfp4"
        return {
            "requested_rope_scaling": copy.deepcopy(self.requested_rope_scaling),
            "resolved_rope_scaling": copy.deepcopy(resolved),
            "resolved_max_model_len": max_model_len,
            "tensor_parallel_size": resolved_tp_size,
            "distributed_executor_backend": resolved_executor_backend,
            "resolved_cache_dtype": cache_dtype,
            **({"kv_cache_memory_bytes": kv_cache_memory_bytes} if kv_cache_memory_bytes is not None else {}),
            "kv_cache_dtype_skip_layers": cache_skip_layers,
            "configured_attention_backend": configured_backend,
            "layer_cache_dtype_counts": dict(layer_cache_dtypes),
            "layer_attention_backend_counts": dict(layer_attention_backends),
            "worker_layer_contracts": worker_layer_contracts,
            "worker_layer_contract_error": worker_layer_contract_error,
            "block_size": block_size,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enable_flashinfer_autotune": enable_flashinfer_autotune,
            "probability_scale_format": probability_format,
            "probability_scale_mode": probability_mode,
            "nvfp4_p_rescale_factor": (None if probability_format != "nvfp4" else 8 if p_scale_mode == "amax2_or_amax6_c8" else 1),
            "nvfp4_p_two_level": (probability_format == "nvfp4" and p_scale_mode == "two_level"),
            "nvfp4_p_scale_mode": None if is_mxfp4 else p_scale_mode,
            "nvfp4_p_scale_tau": None if is_mxfp4 else p_scale_tau,
            "nvfp4_p_scale_divisor": None if is_mxfp4 else p_scale_divisor,
            "nvfp4_p_scale_grid_search": None if is_mxfp4 else p_scale_grid_search,
            "nvfp4_p_scale_amax2_or_amax6": (
                None if is_mxfp4 else p_scale_is_amax2_or_amax6(p_scale_mode)
            ),
            "additional_config": additional_config,
        }

    def shutdown(self) -> str | None:
        """Release the vLLM engine, including a multiprocess EngineCore."""
        engine = getattr(self.llm, "llm_engine", None)
        # The shutdown entry point may be on the engine or its core client.
        for name, owner in (
            ("llm", self.llm),
            ("llm_engine", engine),
            ("llm_engine.engine_core", getattr(engine, "engine_core", None)),
        ):
            shutdown = getattr(owner, "shutdown", None)
            if callable(shutdown):
                shutdown()
                return name
        return None

    @staticmethod
    def _check_sink(expected, custom_backend: bool) -> None:
        """Check that the loaded kernel's sink matches the evaluation mode."""
        if expected is None:
            return
        if not custom_backend:
            raise ValueError("sink_tokens is a property of the CUSTOM backend")
        from nvfp4_decode_kernel.interface import SINK_TOKENS
        import nvfp4_decode_kernel

        if SINK_TOKENS != expected:
            raise SystemExit(
                f"this arm needs SINK_TOKENS={expected}, but the kernel on "
                f"the path has {SINK_TOKENS} "
                f"({nvfp4_decode_kernel.__file__}). Point PYTHONPATH at the "
                "checkout that matches the arm."
            )

    @staticmethod
    def _build_custom(llm_kwargs, mode, scale_mode):
        """The CUSTOM backend, through the constructor that owns its contract."""
        from nvfp4_vllm import build_llm

        engine = dict(llm_kwargs)
        model = engine.pop("model")
        # build_llm sets attention_config; vLLM refuses to be told both.
        engine.pop("attention_backend", None)
        engine.pop("kv_cache_dtype", None)

        requested = VllmRunner._custom_additional_config(mode, scale_mode)
        if requested:
            print(f"[custom backend] additional_config={requested}", flush=True)
            additional = dict(engine.get("additional_config") or {})
            additional.update(requested)
            engine["additional_config"] = additional
        additional = dict(engine.get("additional_config") or {})
        additional["nvfp4_key_smoothing"] = mode.get("key_shift")
        return build_llm(model, **{**engine, "additional_config": additional})

    @staticmethod
    def _build_standard(llm_kwargs, mode, *, worker_extension_cls=None):
        """Build vLLM, optionally suppressing its TurboQuant boundary skip.

        Upstream vLLM automatically keeps the first and last two transformer
        layers on native KV cache for every TurboQuant dtype. The all-layers
        evaluation arm deliberately removes that policy for the duration of
        engine construction. Restoring the static method in ``finally`` keeps
        a later boundary-protected engine in the same Python process honest.
        The run manifest independently refuses the arm unless all loaded
        layers attest TurboQuant cache and the TURBOQUANT backend.

        A benchmark can supply a worker extension for timing synchronization;
        evaluation retains the calibration and cache-inspection extension.
        """
        from vllm import LLM

        from nvfp4_vllm.calibrate import WORKER_EXTENSION_CLS

        expected_extension = worker_extension_cls or WORKER_EXTENSION_CLS
        supplied_extension = llm_kwargs.get("worker_extension_cls")
        if supplied_extension not in (None, expected_extension):
            raise ValueError(
                "standard evaluation requires worker_extension_cls="
                f"{expected_extension}, got {supplied_extension}"
            )
        llm_kwargs["worker_extension_cls"] = expected_extension

        if not mode.get("turboquant_all_layers"):
            return LLM(**llm_kwargs)
        if mode.get("kv_cache_dtype") != "turboquant_4bit_nc":
            raise ValueError(
                "turboquant_all_layers requires kv_cache_dtype="
                "'turboquant_4bit_nc'"
            )
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        # At the pinned vLLM a TurboQuant layer's KV spec reports no quant
        # mode, so the runner shapes its cache as a skipped layer's, asking
        # the TurboQuant backend for cache dtype "auto", which it refuses;
        # Qwen3-30B reaches that in the CUDA-graph memory profile. With no
        # layer skipped, "auto" can only mean the preset. Both patches live
        # in this process, so the engine must run in it too
        # (VLLM_ENABLE_V1_MULTIPROCESSING=0).
        preset = mode["kv_cache_dtype"]
        boundary = TurboQuantConfig.__dict__["get_boundary_skip_layers"]
        cache_shape = TurboQuantAttentionBackend.__dict__["get_kv_cache_shape"]

        def preset_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=preset
        ):
            if cache_dtype_str == "auto":
                cache_dtype_str = preset
            return cache_shape.__func__(
                num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
            )

        TurboQuantConfig.get_boundary_skip_layers = staticmethod(
            lambda model_config, n=2: []
        )
        TurboQuantAttentionBackend.get_kv_cache_shape = staticmethod(preset_cache_shape)
        try:
            return LLM(**llm_kwargs)
        finally:
            TurboQuantConfig.get_boundary_skip_layers = boundary
            TurboQuantAttentionBackend.get_kv_cache_shape = cache_shape

    @staticmethod
    def _custom_additional_config(mode, scale_mode):
        """Translate a named evaluation arm into the backend's public keys."""
        selected_scale = (
            scale_mode if scale_mode is not None else mode.get("scale_mode")
        )
        requested = {
            "nvfp4_scale_mode": selected_scale,
            "nvfp4_key_scale_geometry": mode.get("key_scale_geometry"),
            "nvfp4_query_terms": mode.get("query_terms"),
            "nvfp4_rotation": mode.get("rotation"),
        }
        return {key: value for key, value in requested.items() if value is not None}

    @staticmethod
    def _custom_backend_kwargs(llm_kwargs, *, scale_mode):
        """Apply block-scale overrides only to the CUSTOM backend."""
        if scale_mode is None:
            return {}
        if llm_kwargs.get("attention_backend") != "CUSTOM":
            raise ValueError(
                "--scale-mode is an option of the CUSTOM "
                "NVFP4 backend; pass --modes bitdecoding-nvfp4"
            )
        return {"additional_config": {"nvfp4_scale_mode": scale_mode}}

    def tokenize(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def chat(self, conversations, *, temperature=0.0, top_p=1.0, top_k=None,
             min_p=None, max_new_tokens=1024, seed=None,
             enable_thinking=None) -> List[dict]:
        """Run one chat completion per conversation. Returns per-request dicts
        with ``text``, ``prompt_tokens``, ``gen_tokens`` and ``finish_reason``.

        ``seed`` may be one value for the batch, or one per conversation. The
        per-conversation form is for benchmarks that sample the same prompt
        more than once: under a shared seed those repeats decode identically,
        which looks like a valid run and silently measures Avg@1.

        ``max_new_tokens`` takes the same two forms. The per-row form lets a
        long-context benchmark give each row whatever the window has left
        after its own prompt, instead of every row sharing a budget that has
        to be small enough for the longest one.

        ``enable_thinking`` is passed to the chat template. Qwen3 defaults to
        thinking, so this is stated rather than relied upon: the default is
        a property of the checkpoint's template and could change under us.
        """
        from vllm import SamplingParams

        extra = {}
        if top_k is not None:
            extra["top_k"] = top_k
        if min_p is not None:
            extra["min_p"] = min_p

        def params(one_seed, one_budget):
            return SamplingParams(
                temperature=temperature, top_p=top_p,
                max_tokens=one_budget, seed=one_seed, **extra,
            )

        def spread(value, what):
            if not isinstance(value, (list, tuple)):
                return [value] * len(conversations), False
            if len(value) != len(conversations):
                raise ValueError(
                    f"got {len(value)} {what} for {len(conversations)} "
                    "conversations"
                )
            return list(value), True

        seeds, per_row_seed = spread(seed, "seeds")
        budgets, per_row_budget = spread(max_new_tokens, "token budgets")
        if per_row_seed or per_row_budget:
            sp = [params(s, b) for s, b in zip(seeds, budgets)]
        else:
            sp = params(seed, max_new_tokens)
        chat_kwargs = {}
        if enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = config.chat_template_kwargs(
                getattr(self, "model_path", None), enable_thinking
            )
        batch_size = getattr(self, "_request_batch_size", None)
        batch_size = int(batch_size or len(conversations) or 1)
        outputs = []
        for start in range(0, len(conversations), batch_size):
            stop = min(start + batch_size, len(conversations))
            batch_params = sp[start:stop] if isinstance(sp, list) else sp
            outputs.extend(
                self.llm.chat(
                    conversations[start:stop], batch_params,
                    use_tqdm=True, **chat_kwargs,
                )
            )
        results = []
        for out in outputs:
            comp = out.outputs[0]
            results.append({
                "text": comp.text,
                "prompt_tokens": len(out.prompt_token_ids),
                "prompt_token_sha256": hashlib.sha256(json.dumps(
                    list(out.prompt_token_ids), separators=(",", ":")
                ).encode()).hexdigest(),
                "gen_tokens": len(comp.token_ids),
                "output_token_ids": list(comp.token_ids),
                "finish_reason": comp.finish_reason,
            })
        return results

    def generate(self, prompts, *, temperature=0.0, top_p=1.0, top_k=None,
                 min_p=None, max_new_tokens=1024, seed=None) -> List[dict]:
        """Run raw text completions without applying a chat template.

        Seed and token budget accept the same scalar-or-per-row forms as
        :meth:`chat`.
        """
        from vllm import SamplingParams

        extra = {}
        if top_k is not None:
            extra["top_k"] = top_k
        if min_p is not None:
            extra["min_p"] = min_p

        def params(one_seed, one_budget):
            return SamplingParams(
                temperature=temperature, top_p=top_p,
                max_tokens=one_budget, seed=one_seed, **extra,
            )

        def spread(value, what):
            if not isinstance(value, (list, tuple)):
                return [value] * len(prompts), False
            if len(value) != len(prompts):
                raise ValueError(
                    f"got {len(value)} {what} for {len(prompts)} prompts"
                )
            return list(value), True

        seeds, per_row_seed = spread(seed, "seeds")
        budgets, per_row_budget = spread(max_new_tokens, "token budgets")
        if per_row_seed or per_row_budget:
            sp = [params(s, b) for s, b in zip(seeds, budgets)]
        else:
            sp = params(seed, max_new_tokens)

        batch_size = getattr(self, "_request_batch_size", None)
        batch_size = int(batch_size or len(prompts) or 1)
        outputs = []
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            batch_params = sp[start:stop] if isinstance(sp, list) else sp
            outputs.extend(
                self.llm.generate(
                    prompts[start:stop], batch_params, use_tqdm=True,
                )
            )

        results = []
        for out in outputs:
            comp = out.outputs[0]
            results.append({
                "text": comp.text,
                "prompt_tokens": len(out.prompt_token_ids),
                "prompt_token_sha256": hashlib.sha256(json.dumps(
                    list(out.prompt_token_ids), separators=(",", ":")
                ).encode()).hexdigest(),
                "gen_tokens": len(comp.token_ids),
                "output_token_ids": list(comp.token_ids),
                "finish_reason": comp.finish_reason,
            })
        return results
