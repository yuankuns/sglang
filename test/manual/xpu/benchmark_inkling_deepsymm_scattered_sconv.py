"""TP benchmark for Inkling's scattered-SConv collective.

Run through the BMG container wrapper with the DeepSymm worktree first on
PYTHONPATH, for example:

    ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 --bind-to none \
      python -B test/manual/xpu/benchmark_inkling_deepsymm_scattered_sconv.py
"""

import os
import statistics

import torch
import torch.distributed as dist
from deep_symm import env
from deep_symm.collectives import reduce_scatter_sconv_allgather
from sgl_kernel import fused_add_rmsnorm

from sglang.srt.models.inkling_common.kernels.sconv import (
    fused_causal_conv1d_update_decode,
)

WARMUP = int(os.getenv("INKLING_SCONV_BENCH_WARMUP", "20"))
REPETITIONS = int(os.getenv("INKLING_SCONV_BENCH_REPETITIONS", "100"))
TOKENS = tuple(
    int(value) for value in os.getenv("INKLING_SCONV_BENCH_TOKENS", "1,8").split(",")
)
HIDDEN = int(os.getenv("INKLING_SCONV_BENCH_HIDDEN", "6144"))


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


def max_rank(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def summarize(samples: list[float], device: torch.device) -> tuple[float, float]:
    return (
        max_rank(statistics.mean(samples), device),
        max_rank(statistics.median(samples), device),
    )


def main() -> None:
    env.setup_distributed_env(master_addr="127.0.0.1", master_port="29548")
    dist.init_process_group("xccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    if HIDDEN % world:
        raise ValueError("hidden size must divide world size")
    hidden_shard = HIDDEN // world

    for tokens in TOKENS:
        torch.manual_seed(3100 + rank)
        input_tensor = torch.randn(
            (tokens, HIDDEN), dtype=torch.bfloat16, device=device
        )
        weight = torch.randn((hidden_shard, 4), dtype=torch.bfloat16, device=device)
        cache = torch.randn(
            (max(tokens, 8), 3, hidden_shard),
            dtype=torch.bfloat16,
            device=device,
        )
        cache_indices = torch.arange(tokens, dtype=torch.int32, device=device)
        cache_mask = torch.ones(tokens, dtype=torch.bool, device=device)
        residual = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        norm_weight = torch.randn((HIDDEN,), dtype=torch.bfloat16, device=device)
        shard = torch.empty((tokens, hidden_shard), dtype=torch.bfloat16, device=device)
        gathered = torch.empty(
            (world, tokens, hidden_shard), dtype=torch.bfloat16, device=device
        )

        # Allocate and rendezvous DeepSymm's shape-specific resources before
        # timing. Cache mutation is intentional and identical on both paths.
        rank_major = (
            input_tensor.view(tokens, world, hidden_shard).movedim(1, 0).contiguous()
        )
        reduce_scatter_sconv_allgather(
            rank_major,
            cache,
            cache_indices,
            cache_mask,
            weight,
            dist.group.WORLD,
            activation="silu",
            use_residual=True,
        )

        def oneccl_production():
            rank_major_input = (
                input_tensor.view(tokens, world, hidden_shard)
                .movedim(1, 0)
                .contiguous()
            )
            dist.reduce_scatter_tensor(shard, rank_major_input)
            local = fused_causal_conv1d_update_decode(
                shard,
                weight,
                cache,
                cache_indices,
                cache_mask,
                activation="silu",
                use_residual=True,
            )
            dist.all_gather_into_tensor(gathered.view(-1, hidden_shard), local)
            return gathered.movedim(0, 1).reshape(tokens, HIDDEN)

        def deepsymm_fused():
            rank_major_input = (
                input_tensor.view(tokens, world, hidden_shard)
                .movedim(1, 0)
                .contiguous()
            )
            output = reduce_scatter_sconv_allgather(
                rank_major_input,
                cache,
                cache_indices,
                cache_mask,
                weight,
                dist.group.WORLD,
                activation="silu",
                use_residual=True,
            )
            return output.movedim(0, 1).reshape(tokens, HIDDEN)

        baseline = summarize(measure(oneccl_production), device)
        fused = summarize(measure(deepsymm_fused), device)
        baseline_norm = None
        fused_norm = None
        if tokens == 1:
            residual_work = residual.clone()

            def oneccl_production_norm():
                hidden = oneccl_production()
                fused_add_rmsnorm(hidden, residual_work, norm_weight, 1e-6)
                return hidden, residual_work

            def deepsymm_fused_norm():
                rank_major_input = (
                    input_tensor.view(tokens, world, hidden_shard)
                    .movedim(1, 0)
                    .contiguous()
                )
                return reduce_scatter_sconv_allgather(
                    rank_major_input,
                    cache,
                    cache_indices,
                    cache_mask,
                    weight,
                    dist.group.WORLD,
                    activation="silu",
                    use_residual=True,
                    residual=residual.view(-1),
                    norm_weight=norm_weight,
                    eps=1e-6,
                )

            baseline_norm = summarize(measure(oneccl_production_norm), device)
            fused_norm = summarize(measure(deepsymm_fused_norm), device)
        if rank == 0:
            line = (
                f"T={tokens} H={HIDDEN} TP={world} "
                f"oneCCL_mean={baseline[0]:.4f}ms "
                f"oneCCL_median={baseline[1]:.4f}ms "
                f"DeepSymm_mean={fused[0]:.4f}ms "
                f"DeepSymm_median={fused[1]:.4f}ms "
                f"speedup={baseline[0] / fused[0]:.3f}x"
            )
            if baseline_norm is not None and fused_norm is not None:
                line += (
                    f" oneCCL_norm_mean={baseline_norm[0]:.4f}ms"
                    f" DeepSymm_norm_mean={fused_norm[0]:.4f}ms"
                    f" norm_speedup={baseline_norm[0] / fused_norm[0]:.3f}x"
                )
            print(line, flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
