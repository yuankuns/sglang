from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch


def _kernel_repo() -> Path:
    configured = os.environ.get("SGLANG_KERNEL_XPU_REPO") or os.environ.get(
        "SGL_KERNEL_XPU_REPO"
    )
    candidates = [
        Path(configured) if configured else None,
        Path("/workspace/worktrees/sgl-kernel-xpu/inkling-xpu-e2e"),
        Path("/data2/syk/worktrees/sgl-kernel-xpu/inkling-xpu-e2e"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "include/sgl_kernel_ops.h").is_file():
            return candidate
    raise RuntimeError(
        "The Inkling MXFP4 XPU bridge needs a built sgl-kernel-xpu checkout; "
        "set SGLANG_KERNEL_XPU_REPO."
    )


def ensure_inkling_mxfp4_xpu_op() -> None:
    if hasattr(torch.ops.sgl_kernel, "moe_grouped_mm_nt_xe20_mxfp4_w4a16"):
        return

    from torch.utils.cpp_extension import load

    repo = _kernel_repo()
    build_root = Path(
        os.environ.get("SGLANG_KERNEL_XPU_BUILD_DIR", str(repo / "build"))
    )
    build_dir = build_root / "src"
    library = build_dir / "libsgl-ops-sycl-GroupGemmMxfp4W4A16Xe20.so"
    if not library.is_file():
        raise RuntimeError(f"Missing built MXFP4 XPU kernel: {library}")

    # The AOT tile instantiations are separate shared objects. The original
    # monolithic common_ops extension links them all; this narrow bridge loads
    # only the MXFP4 family, so publish those symbols before loading the
    # dispatcher library.
    for tile_library in sorted(
        build_dir.glob("libsgl-ops-sycl-GroupGemmMxfp4W4A16Xe20_inst_*.so")
    ):
        ctypes.CDLL(str(tile_library), mode=ctypes.RTLD_GLOBAL)

    load(
        name="inkling_mxfp4_xpu_bridge",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_ldflags=[
            f"-L{build_dir}",
            "-lsgl-ops-sycl-GroupGemmMxfp4W4A16Xe20",
            f"-Wl,-rpath,{build_dir}",
        ],
        verbose=False,
    )
    if not hasattr(torch.ops.sgl_kernel, "moe_grouped_mm_nt_xe20_mxfp4_w4a16"):
        raise RuntimeError("Inkling MXFP4 XPU bridge loaded without registering its op")


@lru_cache(maxsize=1)
def get_inkling_mxfp4_fused_experts():
    ensure_inkling_mxfp4_xpu_op()
    source = _kernel_repo() / "python/sgl_kernel/moe.py"
    spec = importlib.util.spec_from_file_location(
        "sgl_kernel._inkling_mxfp4_moe", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MXFP4 XPU MoE wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fused_experts
