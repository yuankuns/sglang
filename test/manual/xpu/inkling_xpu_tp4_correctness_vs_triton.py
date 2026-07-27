#!/usr/bin/env python3
"""Compare TP4 Inkling XPU and Triton attention outputs on dummy data."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TP_SIZE = 4
DEFAULT_XPU_AFFINITY_MASK = "0,1,2,3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_tp4_smoke"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate checkpoint")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--watchdog-timeout", type=int, default=1800)
    parser.add_argument(
        "--xpu-affinity-mask",
        default=os.environ.get("ZE_AFFINITY_MASK", DEFAULT_XPU_AFFINITY_MASK),
    )
    parser.add_argument(
        "--backend-worker",
        choices=("intel_xpu", "triton"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_new_tokens < 2:
        raise ValueError(
            "Use at least two generated tokens to cover prefill and decode"
        )

    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ["ZE_AFFINITY_MASK"] = args.xpu_affinity_mask
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE", "1")

    import torch

    device_count = torch.xpu.device_count()
    if device_count < TP_SIZE:
        raise RuntimeError(
            f"TP{TP_SIZE} requires at least {TP_SIZE} accessible XPU devices; "
            f"PyTorch found {device_count}"
        )

    from inkling_xpu_correctness_vs_torch_native import (
        compare_outputs,
        run_engine,
    )
    from inkling_xpu_offline_engine_smoke import (
        ReducedInklingSpec,
        prepare_sgl_kernel_overlay,
        write_fake_inkling_checkpoint,
    )
    from inkling_xpu_offline_engine_tp4_smoke import (
        TP4_HIDDEN_SIZE,
        TP4_INTERMEDIATE_SIZE,
        TP4_LOCAL_LAYER_IDS,
        TP4_NUM_HEADS,
        TP4_NUM_KV_HEADS,
        TP4_NUM_LAYERS,
    )

    spec = ReducedInklingSpec(
        hidden_size=TP4_HIDDEN_SIZE,
        intermediate_size=TP4_INTERMEDIATE_SIZE,
        num_layers=TP4_NUM_LAYERS,
        num_heads=TP4_NUM_HEADS,
        num_kv_heads=TP4_NUM_KV_HEADS,
        local_layer_ids=TP4_LOCAL_LAYER_IDS,
    )

    input_ids = list(range(3, 3 + args.prompt_len))
    if args.backend_worker:
        if args.output_file is None:
            raise ValueError("--output-file is required for a backend worker")
        overlay = prepare_sgl_kernel_overlay()
        print(f"Using sgl_kernel overlay: {overlay}", flush=True)
        output_ids, logprobs = run_engine(
            args.model_dir,
            args.backend_worker,
            input_ids,
            args.max_new_tokens,
            tp_size=TP_SIZE,
            watchdog_timeout=args.watchdog_timeout,
        )
        torch.save(
            {"output_ids": output_ids, "logprobs": logprobs},
            args.output_file,
        )
        return

    write_fake_inkling_checkpoint(
        args.model_dir,
        force=args.force,
        spec=spec,
        tp_size=TP_SIZE,
    )
    overlay = prepare_sgl_kernel_overlay()
    print(f"Using sgl_kernel overlay: {overlay}", flush=True)

    results = {}
    with tempfile.TemporaryDirectory(prefix="inkling_tp4_compare_") as temp_dir:
        for backend in ("intel_xpu", "triton"):
            output_file = Path(temp_dir) / f"{backend}.pt"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model-dir",
                str(args.model_dir),
                "--prompt-len",
                str(args.prompt_len),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--watchdog-timeout",
                str(args.watchdog_timeout),
                "--xpu-affinity-mask",
                args.xpu_affinity_mask,
                "--backend-worker",
                backend,
                "--output-file",
                str(output_file),
            ]
            subprocess.run(command, check=True)
            results[backend] = torch.load(output_file, weights_only=True)

    xpu_ids = results["intel_xpu"]["output_ids"]
    xpu_logprobs = results["intel_xpu"]["logprobs"]
    triton_ids = results["triton"]["output_ids"]
    triton_logprobs = results["triton"]["logprobs"]

    if xpu_ids != triton_ids:
        raise AssertionError(
            f"Generated token IDs differ: intel_xpu={xpu_ids}, triton={triton_ids}"
        )

    summaries = compare_outputs(
        xpu_logprobs,
        triton_logprobs,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not all(
        math.isfinite(metric) for summary in summaries for metric in summary.values()
    ):
        raise AssertionError(f"Non-finite comparison summary: {summaries}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "input_ids": input_ids,
                "output_ids": xpu_ids,
                "rtol": args.rtol,
                "atol": args.atol,
                "per_step": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
