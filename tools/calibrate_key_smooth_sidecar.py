"""Calibrate static key smoothing and optional probability threshold tau.

The accuracy runner invokes this in a separate process before loading the
scoring engine. Tensor-parallel workers export one global key mean, which
serving workers slice when loading the saved calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="checkpoint to measure the mean on"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="destination for the calibrated key mean",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=10240,
        help="engine context; match the rollout the shift will serve",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=32768,
        help=(
            "requested self-generated context; calibration clamps this to "
            "max_model_len - 1"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--probability-multiplier", type=float, default=None,
                        help="also calibrate tau during the key-smoothing prefill, using this c")
    parser.add_argument(
        "--scale-mode", default=None,
        help="scale rule, or a JSON mapping with key/value/query rules",
    )
    parser.add_argument("--query-terms", type=int, choices=(1, 2), default=None)
    parser.add_argument("--rotation", choices=("none", "hadamard"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_path.exists():
        raise SystemExit(
            f"{args.output_path} exists; a sidecar in use by a run must not be "
            "rewritten underneath it"
        )

    # The measurement has to happen under the engine that will serve it: the
    # mean is post-RoPE, so it depends on the positions visited, and those
    # depend on the window. TP workers export one global-head sidecar that
    # the runtime slices for each serving rank.
    from nvfp4_vllm import build_llm, calibrate_key_smoothing

    scale_mode = args.scale_mode
    if scale_mode is not None and scale_mode.lstrip().startswith("{"):
        scale_mode = json.loads(scale_mode)
    additional_config = {
        key: value for key, value in {
            "nvfp4_scale_mode": scale_mode,
            "nvfp4_query_terms": args.query_terms,
            "nvfp4_rotation": args.rotation,
        }.items() if value is not None
    }
    additional_config["nvfp4_key_smoothing"] = None
    llm = build_llm(
        args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max(40_960, args.max_model_len),
        additional_config=additional_config,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    probability = ({"probability_multiplier": args.probability_multiplier}
                   if args.probability_multiplier is not None else {})
    measured = calibrate_key_smoothing(
        llm,
        context_tokens=args.context_tokens,
        save_to=str(args.output_path),
        **probability,
    )
    if measured is None:
        raise SystemExit(
            "the engine came up without the NVFP4 cache, so there was no mean "
            "to measure"
        )

    provenance = args.output_path.with_suffix(".json")
    temporary = provenance.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "model_path": os.path.realpath(args.model_path),
                "max_model_len": args.max_model_len,
                "context_tokens": args.context_tokens,
                "tensor_parallel_size": args.tensor_parallel_size,
                "additional_config": additional_config,
                **measured,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(provenance)
    print(f"[key smoothing] {measured}", flush=True)
    print(f"[key smoothing] provenance {provenance}", flush=True)


if __name__ == "__main__":
    main()
