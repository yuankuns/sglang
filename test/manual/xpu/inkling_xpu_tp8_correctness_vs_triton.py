#!/usr/bin/env python3
"""Compare the six-layer TP8 Inkling XPU and Triton outputs on dummy data.

The fused attention prologue is disabled so both backends receive identically
preprocessed Q/K/V tensors and this test isolates the attention implementation.
Layers 0..4 use local/SWA attention and layer 5 uses global attention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TP_SIZE = 8
DEFAULT_XPU_AFFINITY_MASK = "0,1,2,3,4,5,6,7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_tp8_6layer"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate checkpoint")
    parser.add_argument(
        "--allow-large-checkpoint",
        action="store_true",
        help="Allow materializing the approximately 126 GiB fake checkpoint",
    )
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
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

    if not args.allow_large_checkpoint:
        raise RuntimeError(
            "The official-width reduced model is approximately 126 GiB in BF16; "
            "pass --allow-large-checkpoint to materialize it."
        )

    if args.max_new_tokens < 2:
        raise ValueError(
            "Use at least two generated tokens to cover prefill and decode"
        )

    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ["ZE_AFFINITY_MASK"] = args.xpu_affinity_mask
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")
    os.environ["SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE"] = "0"

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
    from inkling_xpu_offline_engine_tp8_smoke import (
        DENSE_INTERMEDIATE_SIZE,
        DENSE_MLP_IDX,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        LOCAL_LAYER_IDS,
        MTP_LOCAL_LAYER_IDS,
        NUM_EXPERTS_PER_TOK,
        NUM_HEADS,
        NUM_KV_HEADS,
        NUM_LAYERS,
        NUM_MTP_LAYERS,
        NUM_ROUTED_EXPERTS,
        NUM_SHARED_EXPERTS,
        SWA_NUM_KV_HEADS,
        VOCAB_SIZE,
    )

    spec = ReducedInklingSpec(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dense_intermediate_size=DENSE_INTERMEDIATE_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        swa_num_kv_heads=SWA_NUM_KV_HEADS,
        vocab_size=VOCAB_SIZE,
        unpadded_vocab_size=200058,
        local_layer_ids=LOCAL_LAYER_IDS,
        dense_mlp_idx=DENSE_MLP_IDX,
        n_routed_experts=NUM_ROUTED_EXPERTS,
        n_shared_experts=NUM_SHARED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        use_embed_norm=True,
        use_global_scale=True,
        num_mtp_layers=NUM_MTP_LAYERS,
        mtp_local_layer_ids=MTP_LOCAL_LAYER_IDS,
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
    with tempfile.TemporaryDirectory(prefix="inkling_tp8_compare_") as temp_dir:
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
