from functools import lru_cache

import torch


def ensure_inkling_mxfp4_xpu_op() -> None:
    import sgl_kernel  # noqa: F401

    if not hasattr(torch.ops.sgl_kernel, "moe_grouped_mm_nt_xe20_w4a16"):
        raise RuntimeError(
            "The installed sgl-kernel-xpu package does not provide "
            "moe_grouped_mm_nt_xe20_w4a16; install a full wheel built from main."
        )


@lru_cache(maxsize=1)
def get_inkling_mxfp4_fused_experts():
    ensure_inkling_mxfp4_xpu_op()
    from sgl_kernel.moe import fused_experts

    return fused_experts
