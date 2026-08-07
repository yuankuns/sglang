#!/usr/bin/env python3

import argparse
import json
import math
from types import SimpleNamespace

import torch
import triton

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
)
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend


DEVICE = torch.device("xpu")
DTYPE = torch.bfloat16


class ExtendMode:
    def is_extend_or_draft_extend_or_mixed(self):
        return True

    def is_target_verify(self):
        return False


class MockPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache

    def get_kv_buffer(self, layer_id):
        return self.k_cache, self.v_cache

    def set_kv_buffer(self, layer, loc_info, k, v, k_scale, v_scale):
        row_dim = layer.tp_k_head_num * layer.head_dim
        torch.ops.sgl_kernel.store_cache.default(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            self.k_cache.view(-1, row_dim),
            self.v_cache.view(-1, row_dim),
            loc_info.loc,
        )


def cumulative(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, device=DEVICE, dtype=torch.int32)


def make_case(family, tp, batch, new_len, old_len):
    if family == "sliding":
        head_dim = 256
        hkv = max(1, 16 // tp)
        sliding_window = 1023
    else:
        head_dim = 512
        hkv = max(1, 4 // tp)
        sliding_window = -1
    return {
        "family": family,
        "tp": tp,
        "head_dim": head_dim,
        "hq": 32 // tp,
        "hkv": hkv,
        "batch": batch,
        "new_len": new_len,
        "old_len": old_len,
        "page_size": 64,
        "sliding_window": sliding_window,
    }


def make_mixed_case(family, q_lens, old_len):
    case = make_case(family, 4, len(q_lens), max(q_lens), old_len)
    case["q_lens"] = q_lens
    case["pattern"] = (
        "decode_heavy"
        if sum(q_len == 1 for q_len in q_lens) > len(q_lens) // 2
        else "balanced"
    )
    return case


def make_large_cases():
    specs = (
        (1, 2048, 16384),
        (1, 4096, 32768),
        (1, 8192, 32768),
        (4, 1024, 16384),
        (4, 2048, 32768),
        (8, 1024, 16384),
    )
    return [
        make_case(family, 4, batch, new_len, old_len)
        for family in ("sliding", "full")
        for batch, new_len, old_len in specs
    ]


def make_large_mixed_cases():
    patterns = (
        ([1] * 8 + [512] * 4 + [1024] * 4, 16384),
        (
            [2048, 2048, 1024, 1024, 512, 512, 256, 256] + [1] * 8,
            32768,
        ),
    )
    return [
        make_mixed_case(family, q_lens, old_len)
        for family in ("sliding", "full")
        for q_lens, old_len in patterns
    ]


def make_harness(case):
    batch = case["batch"]
    q_lens = case.get("q_lens", [case["new_len"]] * batch)
    new_len = max(q_lens)
    old_len = case["old_len"]
    final_lens = [old_len + q_len for q_len in q_lens]
    page_size = case["page_size"]
    pages_per_row = math.ceil(max(final_lens) / page_size)
    total_new = sum(q_lens)
    torch.manual_seed(
        8803
        + case["head_dim"] * 7
        + case["hq"] * 13
        + case["hkv"] * 17
        + total_new
        + old_len
    )
    q = torch.randn(
        total_new, case["hq"], case["head_dim"], device=DEVICE, dtype=DTYPE
    )
    k = torch.randn(
        total_new, case["hkv"], case["head_dim"], device=DEVICE, dtype=DTYPE
    )
    v = torch.randn_like(k)
    k_cache = torch.randn(
        batch * pages_per_row,
        page_size,
        case["hkv"],
        case["head_dim"],
        device=DEVICE,
        dtype=DTYPE,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        batch * pages_per_row, device=DEVICE, dtype=torch.int32
    ).reshape(batch, pages_per_row)
    full_lens = torch.tensor(final_lens, device=DEVICE, dtype=torch.int32)
    extend_lens = torch.tensor(
        q_lens, device=DEVICE, dtype=torch.int32
    )
    cu_q = cumulative(q_lens)
    indices = []
    for batch_idx in range(batch):
        first_page = batch_idx * pages_per_row
        for token_idx in range(q_lens[batch_idx]):
            cache_pos = old_len + token_idx
            physical_page = first_page + cache_pos // page_size
            indices.append(physical_page * page_size + cache_pos % page_size)
    out_cache_loc = torch.tensor(indices, device=DEVICE, dtype=torch.int64)

    metadata = FlashAttentionMetadata()
    metadata.page_table = page_table
    metadata.swa_page_table = None
    metadata.swa_spec_metadata = None
    metadata.local_attn_metadata = None
    metadata.cache_seqlens_int32 = full_lens
    metadata.appendkv_cache_seqlens_int32 = full_lens - extend_lens
    metadata.cu_seqlens_q = cu_q
    metadata.cu_seqlens_k = cumulative(final_lens)
    metadata.max_seq_len_q = new_len

    backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
    backend.use_mla = False
    backend.forward_metadata = metadata
    backend.topk = 0
    backend.attention_chunk_size = None
    backend.kv_cache_dtype = DTYPE
    backend.page_size = page_size
    backend.use_sliding_window_kv_pool = False
    backend.token_to_kv_pool = MockPool(k_cache, v_cache)

    layer = SimpleNamespace(
        is_cross_attention=False,
        sliding_window_size=case["sliding_window"],
        head_dim=case["head_dim"],
        v_head_dim=case["head_dim"],
        tp_q_head_num=case["hq"],
        tp_k_head_num=case["hkv"],
        tp_v_head_num=case["hkv"],
        scaling=1.0 / math.sqrt(case["head_dim"]),
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
        layer_id=0,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ExtendMode(),
        out_cache_loc=out_cache_loc,
        encoder_out_cache_loc=None,
        extend_seq_lens=extend_lens,
        extend_seq_lens_cpu=q_lens,
        _attn_output=None,
        attn_attend_prefix_cache=None,
    )
    return backend, layer, forward_batch, q, k, v


def measure(fn, warmup, rep):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    value = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, return_mode="mean"
    )
    torch.xpu.synchronize()
    return float(value)


def benchmark(case, warmup, rep):
    backend, layer, forward_batch, q, k, v = make_harness(case)
    appendkv_selected = backend._appendkv_applicable(
        layer, forward_batch, q, k, v, True
    )

    def old_path():
        backend._appendkv_applicable = lambda *args: False
        return backend.forward_extend(q, k, v, layer, forward_batch)

    def append_path():
        backend._appendkv_applicable = lambda *args: True
        return backend.forward_extend(q, k, v, layer, forward_batch)

    old_ms = measure(old_path, warmup, rep)
    append_ms = measure(append_path, warmup, rep)
    return {
        **case,
        "old_forward_extend_ms": old_ms,
        "append_forward_extend_ms": append_ms,
        "appendkv_selected_by_policy": appendkv_selected,
        "speedup": old_ms / append_ms,
        "latency_reduction_pct": (old_ms - append_ms) / old_ms * 100,
    }


def benchmark_stack(case, warmup, rep, stack_mode):
    backend, layer, forward_batch, q, k, v = make_harness(case)
    appendkv_selected = (
        backend._appendkv_applicable(layer, forward_batch, q, k, v, True)
        if stack_mode == "candidate"
        else False
    )
    if stack_mode == "baseline":
        backend._appendkv_applicable = lambda *args: False

    def forward_extend():
        return backend.forward_extend(q, k, v, layer, forward_batch)

    return {
        **case,
        "stack_mode": stack_mode,
        "forward_extend_ms": measure(forward_extend, warmup, rep),
        "appendkv_selected_by_policy": appendkv_selected,
    }


def verify(case):
    old = make_harness(case)
    append = make_harness(case)
    old_backend, old_layer, old_batch, old_q, old_k, old_v = old
    append_backend, append_layer, append_batch, append_q, append_k, append_v = append
    old_backend._appendkv_applicable = lambda *args: False
    append_backend._appendkv_applicable = lambda *args: True
    old_out = old_backend.forward_extend(
        old_q, old_k, old_v, old_layer, old_batch
    )
    append_out = append_backend.forward_extend(
        append_q, append_k, append_v, append_layer, append_batch
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(
        append_backend.token_to_kv_pool.k_cache,
        old_backend.token_to_kv_pool.k_cache,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        append_backend.token_to_kv_pool.v_cache,
        old_backend.token_to_kv_pool.v_cache,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(append_out, old_out, rtol=0.02, atol=0.02)
    return {
        "event": "verification",
        **case,
        "cache_exact": True,
        "max_abs_diff": float((append_out.float() - old_out.float()).abs().max()),
        "mean_abs_diff": float(
            (append_out.float() - old_out.float()).abs().mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--large-only", action="store_true")
    parser.add_argument(
        "--stack-mode", choices=("baseline", "candidate"), default=None
    )
    args = parser.parse_args()
    if args.verify_only:
        verify_cases = [
            make_case("sliding", 4, 1, 16, 1024),
            make_case("sliding", 4, 8, 256, 1024),
            make_case("sliding", 4, 16, 512, 1024),
            make_case("full", 4, 1, 16, 4096),
            make_case("full", 4, 8, 256, 4096),
            make_case("full", 4, 16, 512, 8192),
        ]
        for case in verify_cases:
            print(json.dumps(verify(case), separators=(",", ":")), flush=True)
        return
    cases = []
    if args.large_only:
        cases.extend(make_large_cases())
        cases.extend(make_large_mixed_cases())
    elif args.mixed_only:
        patterns = (
            [1, 1, 1, 1, 1, 1, 1, 1, 16, 32, 64, 128, 1, 1, 8, 256],
            [1, 16, 1, 32, 1, 64, 1, 128, 1, 256, 8, 1, 64, 1, 32, 1],
            [64, 128, 256, 128, 64, 32, 16, 8, 1, 1, 64, 128, 256, 32, 1, 16],
        )
        for q_lens in patterns:
            cases.append(make_mixed_case("sliding", q_lens, 1024))
            cases.append(make_mixed_case("full", q_lens, 4096))
    else:
        for tp in (2, 4):
            for batch in (1, 8, 16):
                for new_len in (16, 64, 256, 512):
                    cases.append(make_case("sliding", tp, batch, new_len, 1024))
                    for old_len in (4096, 8192):
                        cases.append(make_case("full", tp, batch, new_len, old_len))
    print(
        json.dumps(
            {
                "event": "start",
                "repeat": args.repeat,
                "cases": len(cases),
                "stack_mode": args.stack_mode,
            }
        ),
        flush=True,
    )
    for index, case in enumerate(cases, 1):
        result = (
            benchmark_stack(
                case,
                args.warmup,
                args.rep,
                args.stack_mode,
            )
            if args.stack_mode is not None
            else benchmark(case, args.warmup, args.rep)
        )
        result["repeat"] = args.repeat
        result["case_index"] = index
        print(json.dumps(result, separators=(",", ":")), flush=True)
    print(json.dumps({"event": "done"}), flush=True)


if __name__ == "__main__":
    main()
