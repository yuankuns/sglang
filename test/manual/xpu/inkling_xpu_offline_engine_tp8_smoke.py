#!/usr/bin/env python3
"""Eight-card tensor-parallel smoke test for the six-layer reduced Inkling model.

The retained decoder stack is one minimum Inkling attention period:
layers 0..4 use local/SWA attention and layer 5 uses global attention.

Run from the SGLang repo inside an XPU-enabled container, for example:

PYTHONPATH="$PWD/python" \
  SGLANG_KERNEL_XPU_REPO=/path/to/sgl-kernel-xpu-worktree \
  python \
  test/manual/xpu/inkling_xpu_offline_engine_tp8_smoke.py --force
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TP_SIZE = 8
DEFAULT_XPU_AFFINITY_MASK = "0,1,2,3,4,5,6,7"
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 3072
DENSE_INTERMEDIATE_SIZE = 24576
NUM_LAYERS = 6
NUM_HEADS = 48
NUM_KV_HEADS = 8
SWA_NUM_KV_HEADS = 16
VOCAB_SIZE = 201024
LOCAL_LAYER_IDS = (0, 1, 2, 3, 4)
DENSE_MLP_IDX = 2
NUM_ROUTED_EXPERTS = 256
NUM_SHARED_EXPERTS = 2
NUM_EXPERTS_PER_TOK = 6
NUM_MTP_LAYERS = 8
MTP_LOCAL_LAYER_IDS = (0, 2, 4, 5, 6, 7)
TARGET_PARAMETER_COUNT = 62_437_775_878
MTP_PARAMETER_COUNT = 5_046_437_896
TARGET_WEIGHT_BYTES = 124_875_551_756
MTP_WEIGHT_BYTES = 10_092_875_792


def model_size_summary() -> dict[str, int | float]:
    parameter_count = TARGET_PARAMETER_COUNT + MTP_PARAMETER_COUNT
    weight_bytes = TARGET_WEIGHT_BYTES + MTP_WEIGHT_BYTES
    return {
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "mtp_parameter_count": MTP_PARAMETER_COUNT,
        "parameter_count": parameter_count,
        "parameter_count_billions": parameter_count / 1e9,
        "bf16_weight_bytes": weight_bytes,
        "bf16_weight_gib": weight_bytes / 2**30,
        "ideal_tp8_weight_gib_per_rank": weight_bytes / TP_SIZE / 2**30,
    }


def _visible_xpu_count(affinity_mask: str) -> int:
    return len([device for device in affinity_mask.split(",") if device.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_tp8_6layer"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate checkpoint")
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="Print the exact reduced-model parameter and BF16 weight size",
    )
    parser.add_argument(
        "--allow-large-checkpoint",
        action="store_true",
        help="Allow materializing the approximately 126 GiB fake checkpoint",
    )
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Run this many unmeasured requests before the reported request",
    )
    parser.add_argument(
        "--enable-decode-xpu-graph",
        action="store_true",
        help="Capture and replay the TP8 decode step with the full XPU graph backend",
    )
    parser.add_argument(
        "--enable-prefill-xpu-graph",
        action="store_true",
        help="Capture and replay the TP8 prefill step with the full XPU graph backend",
    )
    parser.add_argument(
        "--decode-graph-batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Decode batch-size buckets to capture when decode XPU graph is enabled",
    )
    parser.add_argument(
        "--xpu-affinity-mask",
        default=os.environ.get("ZE_AFFINITY_MASK", DEFAULT_XPU_AFFINITY_MASK),
        help="ZE_AFFINITY_MASK value exposing at least eight XPU devices",
    )
    args = parser.parse_args()

    if args.size_only:
        print(json.dumps(model_size_summary(), indent=2, sort_keys=True))
        return
    if not args.allow_large_checkpoint:
        raise RuntimeError(
            "The official-width reduced model is approximately 126 GiB in BF16. "
            "Use --size-only to inspect its capacity, or explicitly pass "
            "--allow-large-checkpoint to materialize it."
        )

    if _visible_xpu_count(args.xpu_affinity_mask) < TP_SIZE:
        raise ValueError(
            f"TP{TP_SIZE} requires at least {TP_SIZE} visible XPU devices; "
            f"got ZE_AFFINITY_MASK={args.xpu_affinity_mask!r}"
        )

    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ["ZE_AFFINITY_MASK"] = args.xpu_affinity_mask
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE", "1")

    # Import after setting affinity so the first Level Zero initialization sees
    # all eight devices.
    import torch

    device_count = torch.xpu.device_count()
    if device_count < TP_SIZE:
        raise RuntimeError(
            f"TP{TP_SIZE} requires at least {TP_SIZE} accessible XPU devices; "
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
        num_mtp_layers=NUM_MTP_LAYERS,
        mtp_local_layer_ids=MTP_LOCAL_LAYER_IDS,
    )
    write_fake_inkling_checkpoint(
        args.model_dir,
        force=args.force,
        spec=spec,
        tp_size=TP_SIZE,
    )
    result = run_engine(
        args.model_dir,
        args.prompt_len,
        args.max_new_tokens,
        tp_size=TP_SIZE,
        mem_fraction_static=0.90,
        enable_decode_xpu_graph=args.enable_decode_xpu_graph,
        enable_prefill_xpu_graph=args.enable_prefill_xpu_graph,
        warmup_requests=args.warmup_requests,
        decode_graph_batch_sizes=args.decode_graph_batch_sizes,
        max_total_tokens=args.max_total_tokens,
        context_length=args.context_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
