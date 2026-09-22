"""TP4 benchmark for target-verify all-reduce plus SConv window saving."""

import os
import statistics

import torch
import torch.distributed as dist
from deep_symm import env
from deep_symm.collectives import allreduce_save_sconv_windows_verify
from sgl_kernel import fused_add_rmsnorm

from sglang.srt.models.inkling_common.kernels.sconv import (
    causal_conv1d,
    save_intermediate_conv_windows,
)


WARMUP = int(os.getenv("INKLING_VERIFY_BENCH_WARMUP", "20"))
REPETITIONS = int(os.getenv("INKLING_VERIFY_BENCH_REPETITIONS", "100"))
HIDDEN = int(os.getenv("INKLING_VERIFY_BENCH_HIDDEN", "6144"))
BATCH = int(os.getenv("INKLING_VERIFY_BENCH_BATCH", "2"))
DRAFT_TOKENS = int(os.getenv("INKLING_VERIFY_BENCH_DRAFT_TOKENS", "4"))


def measure(fn) -> list[float]:
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()
    dist.barrier()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(REPETITIONS)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(REPETITIONS)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.xpu.synchronize()
    dist.barrier()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def max_rank_mean(samples: list[float], device: torch.device) -> float:
    value = torch.tensor(statistics.mean(samples), dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def mismatch(actual: torch.Tensor, expected: torch.Tensor) -> tuple[int, float]:
    return (
        int(torch.count_nonzero(actual != expected).item()),
        float((actual.float() - expected.float()).abs().max().item()),
    )


def main() -> None:
    env.setup_distributed_env(master_addr="127.0.0.1", master_port="29549")
    dist.init_process_group("xccl")
    rank = dist.get_rank()
    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    tokens = BATCH * DRAFT_TOKENS

    torch.manual_seed(4100 + rank)
    input_tensor = torch.randn(
        (tokens, HIDDEN), dtype=torch.bfloat16, device=device
    )
    shared = torch.randn_like(input_tensor)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn((HIDDEN, 4), dtype=torch.bfloat16, device=device)
    cache = torch.randn(
        (BATCH + 1, 3, HIDDEN), dtype=torch.bfloat16, device=device
    )
    cache_indices = torch.arange(BATCH, dtype=torch.int32, device=device)
    cache_mask_1d = torch.ones(BATCH, dtype=torch.bool, device=device)
    cache_mask = cache_mask_1d[:, None, None]
    safe_idx = cache_indices.long()
    cu = torch.arange(
        0, tokens + 1, DRAFT_TOKENS, dtype=torch.int64, device=device
    )
    si = torch.arange(BATCH, device=device).repeat_interleave(DRAFT_TOKENS)
    inter_baseline = torch.empty(
        (BATCH, DRAFT_TOKENS, 3, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    inter_deepsymm = torch.empty_like(inter_baseline)
    reduced_baseline = torch.empty_like(input_tensor)
    baseline_residual = residual.clone()
    deepsymm_residual = residual.clone()

    def finish(reduced: torch.Tensor, residual_out: torch.Tensor) -> torch.Tensor:
        conv = causal_conv1d(
            reduced,
            weight,
            cache,
            cache_mask,
            safe_idx,
            cu,
            si,
            activation="silu",
            use_residual=True,
            is_decode=False,
        )
        fused_add_rmsnorm(conv, residual_out, gamma, 1e-6)
        return conv

    def oneccl_chain() -> torch.Tensor:
        torch.add(input_tensor, shared, out=reduced_baseline)
        dist.all_reduce(reduced_baseline)
        save_intermediate_conv_windows(
            cache,
            reduced_baseline.view(BATCH, DRAFT_TOKENS, HIDDEN),
            cache_indices,
            inter_baseline,
            BATCH,
            DRAFT_TOKENS,
        )
        return finish(reduced_baseline, baseline_residual)

    def deepsymm_chain() -> torch.Tensor:
        reduced = allreduce_save_sconv_windows_verify(
            input_tensor,
            residual,
            gamma,
            cache,
            cache_indices,
            cache_mask_1d,
            weight,
            inter_deepsymm,
            DRAFT_TOKENS,
            dist.group.WORLD,
            shared=shared,
        )
        return finish(reduced, deepsymm_residual)

    baseline_residual.copy_(residual)
    expected = oneccl_chain().clone()
    expected_residual = baseline_residual.clone()
    deepsymm_residual.copy_(residual)
    actual = deepsymm_chain().clone()
    torch.xpu.synchronize()
    accuracy = (
        mismatch(actual, expected),
        mismatch(deepsymm_residual, expected_residual),
        mismatch(inter_deepsymm, inter_baseline),
    )

    baseline = max_rank_mean(measure(oneccl_chain), device)
    deepsymm = max_rank_mean(measure(deepsymm_chain), device)
    if rank == 0:
        print(
            "accuracy "
            f"norm={accuracy[0]} residual={accuracy[1]} window={accuracy[2]}",
            flush=True,
        )
        print(
            f"B={BATCH} q={DRAFT_TOKENS} H={HIDDEN} "
            f"oneCCL={baseline:.4f}ms DeepSymm={deepsymm:.4f}ms "
            f"speedup={baseline / deepsymm:.3f}x",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
