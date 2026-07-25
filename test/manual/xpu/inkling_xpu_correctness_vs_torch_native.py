#!/usr/bin/env python3
"""Compare Inkling's sgl-kernel XPU attention with torch-native SDPA.

Engine IPC exposes logprobs rather than raw logits. Requesting the full
vocabulary compares logits up to their shared log-normalization constant.

Run from the SGLang repo inside the `sglang-syk` container:

ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 \
PYTHONPATH=/workspace/worktrees/sglang-inkling-xpu/python:\
/workspace/python-targets/py312-xpu \
  /root/miniforge3/envs/py312/bin/python \
  test/manual/xpu/inkling_xpu_correctness_vs_torch_native.py --force
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from inkling_xpu_offline_engine_smoke import (
    VOCAB_SIZE,
    prepare_sgl_kernel_overlay,
    write_fake_inkling_checkpoint,
)


def _engine_kwargs(model_dir: Path, attention_backend: str) -> dict[str, Any]:
    return {
        "model_path": str(model_dir),
        "tokenizer_path": str(model_dir),
        "skip_tokenizer_init": True,
        "trust_remote_code": True,
        "load_format": "safetensors",
        "dtype": "bfloat16",
        "device": "xpu",
        "tp_size": 1,
        "attention_backend": attention_backend,
        "enable_multimodal": False,
        "max_running_requests": 1,
        "max_total_tokens": 1024,
        "context_length": 128,
        "swa_full_tokens_ratio": 1.0,
        "mem_fraction_static": 0.25,
        "disable_prefill_cuda_graph": True,
        "disable_decode_cuda_graph": True,
        "skip_server_warmup": True,
        "random_seed": 0,
        "log_level": "info",
    }


def _extract_output_ids(output: dict[str, Any]) -> list[int]:
    output_ids = output.get("output_ids")
    if not isinstance(output_ids, list):
        raise AssertionError(f"Engine response has no output_ids: {output}")
    return [int(token_id) for token_id in output_ids]


def _extract_full_logprobs(
    output: dict[str, Any], expected_steps: int
) -> list[torch.Tensor]:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        raise AssertionError(f"Engine response has no meta_info: {output}")
    per_step = meta_info.get("output_top_logprobs")
    if not isinstance(per_step, list) or len(per_step) != expected_steps:
        raise AssertionError(
            f"Expected {expected_steps} output logprob vectors, got {per_step}"
        )

    result = []
    for step, entries in enumerate(per_step):
        if not isinstance(entries, list) or len(entries) != VOCAB_SIZE:
            raise AssertionError(
                f"Step {step} returned {len(entries) if entries else 0} "
                f"logprobs, expected {VOCAB_SIZE}"
            )
        values = torch.full((VOCAB_SIZE,), float("nan"), dtype=torch.float32)
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise AssertionError(f"Malformed top-logprob entry: {entry}")
            logprob, token_id = float(entry[0]), int(entry[1])
            values[token_id] = logprob
        if not torch.isfinite(values).all():
            raise AssertionError(f"Step {step} contains missing or non-finite logprobs")
        result.append(values)
    return result


def run_engine(
    model_dir: Path,
    attention_backend: str,
    input_ids: list[int],
    max_new_tokens: int,
) -> tuple[list[int], list[torch.Tensor]]:
    import sglang as sgl

    print(f"Launching attention_backend={attention_backend}", flush=True)
    engine = sgl.Engine(**_engine_kwargs(model_dir, attention_backend))
    try:
        output = engine.generate(
            input_ids=input_ids,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            return_logprob=True,
            top_logprobs_num=VOCAB_SIZE,
        )
    finally:
        engine.shutdown()

    output_ids = _extract_output_ids(output)
    if len(output_ids) != max_new_tokens:
        raise AssertionError(
            f"{attention_backend} produced {len(output_ids)} tokens: {output_ids}"
        )
    return output_ids, _extract_full_logprobs(output, max_new_tokens)


def compare_outputs(
    xpu_logprobs: list[torch.Tensor],
    reference_logprobs: list[torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> list[dict[str, float | int]]:
    summaries = []
    for step, (actual, expected) in enumerate(
        zip(xpu_logprobs, reference_logprobs, strict=True)
    ):
        abs_error = (actual - expected).abs()
        summary = {
            "step": step,
            "max_abs_error": abs_error.max().item(),
            "mean_abs_error": abs_error.mean().item(),
            "p99_abs_error": torch.quantile(abs_error, 0.99).item(),
        }
        summaries.append(summary)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            msg=lambda msg, summary=summary: f"{msg}\nstep summary: {summary}",
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_correctness"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate checkpoint")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if args.max_new_tokens < 2:
        raise ValueError(
            "Use at least two generated tokens to cover prefill and decode"
        )

    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ.setdefault("ZE_AFFINITY_MASK", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE", "1")

    write_fake_inkling_checkpoint(args.model_dir, force=args.force)
    overlay = prepare_sgl_kernel_overlay()
    print(f"Using sgl_kernel overlay: {overlay}", flush=True)

    input_ids = list(range(3, 3 + args.prompt_len))
    xpu_ids, xpu_logprobs = run_engine(
        args.model_dir, "intel_xpu", input_ids, args.max_new_tokens
    )
    reference_ids, reference_logprobs = run_engine(
        args.model_dir, "torch_native", input_ids, args.max_new_tokens
    )

    if xpu_ids != reference_ids:
        raise AssertionError(
            f"Generated token IDs differ: intel_xpu={xpu_ids}, "
            f"torch_native={reference_ids}"
        )

    summaries = compare_outputs(
        xpu_logprobs,
        reference_logprobs,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not all(
        math.isfinite(metric)
        for summary in summaries
        for metric in summary.values()
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
