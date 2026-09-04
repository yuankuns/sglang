#!/usr/bin/env python3
"""Four-card tensor-parallel smoke test for the six-layer reduced Inkling model.

The retained decoder stack is one minimum Inkling attention period:
layers 0..4 use local/SWA attention and layer 5 uses global attention.

Run from the SGLang repo inside an XPU-enabled container, for example:

PYTHONPATH="$PWD/python" \
  SGLANG_KERNEL_XPU_REPO=/path/to/sgl-kernel-xpu-worktree \
  python \
  test/manual/xpu/inkling_xpu_offline_engine_tp4_smoke.py --force
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

TP_SIZE = 4
DEFAULT_XPU_AFFINITY_MASK = "0,1,2,3"
TP4_HIDDEN_SIZE = 6144
TP4_INTERMEDIATE_SIZE = 3072
TP4_DENSE_INTERMEDIATE_SIZE = 24576
TP4_NUM_LAYERS = 6
TP4_NUM_HEADS = 64
TP4_NUM_KV_HEADS = 8
TP4_SWA_NUM_KV_HEADS = 16
TP4_VOCAB_SIZE = 201024
TP4_LOCAL_LAYER_IDS = (0, 1, 2, 3, 4)
TP4_DENSE_MLP_IDX = 2
TP4_NUM_ROUTED_EXPERTS = 256
TP4_NUM_SHARED_EXPERTS = 2
TP4_NUM_EXPERTS_PER_TOK = 6
TP4_NUM_MTP_LAYERS = 8
TP4_MTP_LOCAL_LAYER_IDS = (0, 2, 4, 5, 6, 7)
TP4_TARGET_PARAMETER_COUNT = 62_598_208_006
TP4_MTP_PARAMETER_COUNT = 5_260_445_704
TP4_TARGET_WEIGHT_BYTES = 125_196_418_068
TP4_MTP_WEIGHT_BYTES = 10_520_891_408


def model_size_summary() -> dict[str, int | float]:
    parameter_count = TP4_TARGET_PARAMETER_COUNT + TP4_MTP_PARAMETER_COUNT
    weight_bytes = TP4_TARGET_WEIGHT_BYTES + TP4_MTP_WEIGHT_BYTES
    return {
        "target_parameter_count": TP4_TARGET_PARAMETER_COUNT,
        "mtp_parameter_count": TP4_MTP_PARAMETER_COUNT,
        "parameter_count": parameter_count,
        "parameter_count_billions": parameter_count / 1e9,
        "bf16_weight_bytes": weight_bytes,
        "bf16_weight_gib": weight_bytes / 2**30,
        "ideal_tp4_weight_gib_per_rank": weight_bytes / TP_SIZE / 2**30,
    }


def _visible_xpu_count(affinity_mask: str) -> int:
    return len([device for device in affinity_mask.split(",") if device.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_tp4_6layer"),
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
    parser.add_argument(
        "--xpu-affinity-mask",
        default=os.environ.get("ZE_AFFINITY_MASK", DEFAULT_XPU_AFFINITY_MASK),
        help="ZE_AFFINITY_MASK value exposing at least four XPU devices",
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
    # all four devices.
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
        hidden_size=TP4_HIDDEN_SIZE,
        intermediate_size=TP4_INTERMEDIATE_SIZE,
        dense_intermediate_size=TP4_DENSE_INTERMEDIATE_SIZE,
        num_layers=TP4_NUM_LAYERS,
        num_heads=TP4_NUM_HEADS,
        num_kv_heads=TP4_NUM_KV_HEADS,
        swa_num_kv_heads=TP4_SWA_NUM_KV_HEADS,
        vocab_size=TP4_VOCAB_SIZE,
        unpadded_vocab_size=200058,
        local_layer_ids=TP4_LOCAL_LAYER_IDS,
        dense_mlp_idx=TP4_DENSE_MLP_IDX,
        n_routed_experts=TP4_NUM_ROUTED_EXPERTS,
        n_shared_experts=TP4_NUM_SHARED_EXPERTS,
        num_experts_per_tok=TP4_NUM_EXPERTS_PER_TOK,
        use_embed_norm=True,
        use_global_scale=True,
        num_mtp_layers=TP4_NUM_MTP_LAYERS,
        mtp_local_layer_ids=TP4_MTP_LOCAL_LAYER_IDS,
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
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
