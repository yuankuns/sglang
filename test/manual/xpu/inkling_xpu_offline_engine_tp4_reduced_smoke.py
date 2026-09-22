#!/usr/bin/env python3
"""Four-card smoke test for one complete, production-width Inkling period.

The 66-layer target is [5 local + 1 global] repeated eleven times. This model
retains exactly one six-layer period, including the two dense layers followed
by four 256-expert MoE layers. Only routed expert storage is changed to OCP
MXFP4; attention, dense MLPs, gates, and shared experts remain BF16. Optional
MTP is omitted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TP_SIZE = 4
EP_SIZE = 1
DEFAULT_XPU_AFFINITY_MASK = "4,5,6,7"
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 3072
DENSE_INTERMEDIATE_SIZE = 24576
NUM_LAYERS = 6
NUM_HEADS = 64
NUM_KV_HEADS = 8
SWA_NUM_KV_HEADS = 16
VOCAB_SIZE = 201024
LOCAL_LAYER_IDS = (0, 1, 2, 3, 4)
DENSE_MLP_IDX = 2
NUM_ROUTED_EXPERTS = 256
NUM_SHARED_EXPERTS = 2
NUM_EXPERTS_PER_TOK = 6
TARGET_PARAMETER_COUNT = 62_598_208_006
BF16_TARGET_WEIGHT_BYTES = 125_196_418_068
MXFP4_TARGET_WEIGHT_BYTES = 40_035_269_652


def model_size_summary() -> dict[str, int | float]:
    return {
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "parameter_count_billions": TARGET_PARAMETER_COUNT / 1e9,
        "bf16_equivalent_weight_bytes": BF16_TARGET_WEIGHT_BYTES,
        "mxfp4_mixed_weight_bytes": MXFP4_TARGET_WEIGHT_BYTES,
        "mxfp4_mixed_weight_gib": MXFP4_TARGET_WEIGHT_BYTES / 2**30,
        "ideal_tp4_weight_gib_per_rank": (MXFP4_TARGET_WEIGHT_BYTES / TP_SIZE / 2**30),
    }


def _visible_xpu_count(affinity_mask: str) -> int:
    return len([device for device in affinity_mask.split(",") if device.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_tp4_6layer_mxfp4"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate checkpoint")
    parser.add_argument(
        "--reuse-existing-checkpoint",
        action="store_true",
        help="Use an existing runnable checkpoint without regenerating its manifest",
    )
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="Print the exact six-layer mixed BF16/MXFP4 weight size",
    )
    parser.add_argument(
        "--allow-large-checkpoint",
        action="store_true",
        help="Allow materializing the approximately 37.3 GiB fake checkpoint",
    )
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument(
        "--enable-scattered-sconv",
        action="store_true",
        help="Run the column-sharded SConv communication path",
    )
    parser.add_argument(
        "--enable-prefill-xpu-graph",
        action="store_true",
        help="Capture and replay the fixed prompt-length prefill graph",
    )
    parser.add_argument(
        "--enable-decode-xpu-graph",
        action="store_true",
        help="Capture and replay the batch-one decode graph",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Number of requests to run before the checked generation",
    )
    parser.add_argument(
        "--measure-ttft",
        action="store_true",
        help="Measure time from generate submission to the first streamed chunk",
    )
    parser.add_argument(
        "--xpu-affinity-mask",
        default=os.environ.get("ZE_AFFINITY_MASK", DEFAULT_XPU_AFFINITY_MASK),
        help="Four healthy physical XPU indices to expose",
    )
    args = parser.parse_args()

    if args.size_only:
        print(json.dumps(model_size_summary(), indent=2, sort_keys=True))
        return
    if not args.allow_large_checkpoint:
        raise RuntimeError(
            "The complete-period TP4 checkpoint is approximately 37.3 GiB. "
            "Use --size-only to inspect it, or pass "
            "--allow-large-checkpoint to materialize it."
        )
    if _visible_xpu_count(args.xpu_affinity_mask) != TP_SIZE:
        raise ValueError(
            f"TP{TP_SIZE} requires exactly {TP_SIZE} visible XPU devices; "
            f"got ZE_AFFINITY_MASK={args.xpu_affinity_mask!r}"
        )

    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ["ZE_AFFINITY_MASK"] = args.xpu_affinity_mask
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE", "1")

    import torch

    device_count = torch.xpu.device_count()
    if device_count != TP_SIZE:
        raise RuntimeError(
            f"TP{TP_SIZE} requires exactly {TP_SIZE} accessible XPU devices; "
            f"PyTorch found {device_count}"
        )

    from inkling_xpu_offline_engine_smoke import (
        ReducedInklingSpec,
        run_engine,
        write_fake_inkling_checkpoint,
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
        num_mtp_layers=0,
        routed_experts_mxfp4=True,
    )
    if args.reuse_existing_checkpoint:
        required = (
            args.model_dir / "config.json",
            args.model_dir / "model.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing checkpoint files: {missing}")
    else:
        write_fake_inkling_checkpoint(
            args.model_dir,
            force=args.force,
            spec=spec,
            tp_size=TP_SIZE,
        )
    from sglang.kernels.ops.moe.inkling_mxfp4_xpu_bridge import (
        ensure_inkling_mxfp4_xpu_op,
    )

    ensure_inkling_mxfp4_xpu_op()
    result = run_engine(
        args.model_dir,
        args.prompt_len,
        args.max_new_tokens,
        tp_size=TP_SIZE,
        ep_size=EP_SIZE,
        mem_fraction_static=0.80,
        enable_prefill_xpu_graph=args.enable_prefill_xpu_graph,
        enable_decode_xpu_graph=args.enable_decode_xpu_graph,
        warmup_requests=args.warmup_requests,
        measure_ttft=args.measure_ttft,
        enable_scattered_sconv=args.enable_scattered_sconv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
