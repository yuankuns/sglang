#!/usr/bin/env python3
"""Four-card tensor-parallel smoke test for the reduced Inkling XPU model.

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
TP4_HIDDEN_SIZE = 2048
TP4_INTERMEDIATE_SIZE = 1024
TP4_NUM_LAYERS = 4
TP4_NUM_HEADS = 16
TP4_NUM_KV_HEADS = 4
TP4_LOCAL_LAYER_IDS = (1, 3)


def _visible_xpu_count(affinity_mask: str) -> int:
    return len([device for device in affinity_mask.split(",") if device.strip()])


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
    parser.add_argument(
        "--xpu-affinity-mask",
        default=os.environ.get("ZE_AFFINITY_MASK", DEFAULT_XPU_AFFINITY_MASK),
        help="ZE_AFFINITY_MASK value exposing at least four XPU devices",
    )
    args = parser.parse_args()

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
        num_layers=TP4_NUM_LAYERS,
        num_heads=TP4_NUM_HEADS,
        num_kv_heads=TP4_NUM_KV_HEADS,
        local_layer_ids=TP4_LOCAL_LAYER_IDS,
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
