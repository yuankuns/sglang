#!/usr/bin/env python3
"""Inkling XPU offline-engine smoke test with a generated HF checkpoint.

Run from the SGLang repo inside the `sglang-syk` container, for example:

ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 \
PYTHONPATH=/workspace/worktrees/sglang-inkling-xpu/python:/workspace/python-targets/py312-xpu \
  /root/miniforge3/envs/py312/bin/python \
  test/manual/xpu/inkling_xpu_offline_engine_smoke.py --force
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 768
NUM_LAYERS = 2
NUM_HEADS = 12
NUM_KV_HEADS = 4
HEAD_DIM = 128
D_REL = 16
REL_EXTENT = 1024
SLIDING_WINDOW_SIZE = 512
VOCAB_SIZE = 4096
SCONV_KERNEL_SIZE = 4
LOCAL_LAYER_IDS = [1]


def prepare_sgl_kernel_overlay() -> Path:
    """Put the branch-built sgl_kernel package ahead of site-packages."""
    candidates = []
    if os.environ.get("SGL_KERNEL_XPU_REPO"):
        candidates.append(Path(os.environ["SGL_KERNEL_XPU_REPO"]))
    candidates.extend(
        [
            Path("/workspace/worktrees/sgl-kernel-xpu/port-inkling-kernel-to-sglang"),
            Path("/data2/syk/worktrees/sgl-kernel-xpu/port-inkling-kernel-to-sglang"),
        ]
    )
    repo = next((path for path in candidates if (path / "python/sgl_kernel").is_dir()), None)
    if repo is None:
        raise RuntimeError(
            "Could not find sgl-kernel-xpu checkout. Set SGL_KERNEL_XPU_REPO."
        )
    build_src = repo / "build" / "src"
    if not build_src.is_dir():
        raise RuntimeError(f"Missing built sgl-kernel XPU artifacts: {build_src}")

    ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if str(build_src) not in ld_paths and not os.environ.get(
        "SGLANG_KERNEL_XPU_LD_REEXEC"
    ):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            str(build_src)
            if not env.get("LD_LIBRARY_PATH")
            else f"{build_src}{os.pathsep}{env['LD_LIBRARY_PATH']}"
        )
        env["SGLANG_KERNEL_XPU_LD_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    overlay = Path(tempfile.mkdtemp(prefix="sgl_kernel_xpu_overlay_"))
    package_dst = overlay / "sgl_kernel"
    shutil.copytree(repo / "python" / "sgl_kernel", package_dst)
    for so_path in build_src.glob("*.abi3.so"):
        shutil.copy2(so_path, package_dst / so_path.name)

    sys.path.insert(0, str(overlay))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        str(overlay)
        if not existing_pythonpath
        else f"{overlay}{os.pathsep}{existing_pythonpath}"
    )
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        str(build_src) if not existing_ld else f"{build_src}{os.pathsep}{existing_ld}"
    )
    return overlay


def _randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    *,
    scale: float = 0.01,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(
        torch.bfloat16
    )


def _ones(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.bfloat16)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_fake_inkling_checkpoint(model_dir: Path, *, force: bool = False) -> None:
    if force and model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "model.safetensors"
    if ckpt_path.exists() and not force:
        return

    text_config = {
        "model_type": "inkling_model",
        "vocab_size": VOCAB_SIZE,
        "padded_vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "dense_intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "v_head_dim": HEAD_DIM,
        "d_rel": D_REL,
        "rel_extent": REL_EXTENT,
        "local_layer_ids": LOCAL_LAYER_IDS,
        "sliding_window_size": SLIDING_WINDOW_SIZE,
        "rms_norm_eps": 1e-6,
        "use_embed_norm": False,
        "use_sconv": True,
        "sconv_kernel_size": SCONV_KERNEL_SIZE,
        "dense_mlp_idx": NUM_LAYERS,
        "n_routed_experts": 0,
        "n_shared_experts": 0,
        "num_experts_per_tok": 1,
        "inference_moe_w13_interleaved": True,
        "tie_word_embeddings": False,
        "max_position_embeddings": 128,
    }
    config = {
        "architectures": ["InklingForConditionalGeneration"],
        "model_type": "inkling_mm_model",
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "text_config": text_config,
        "audio_config": {"model_type": "inkling_audio_model"},
        "vision_config": {"model_type": "inkling_vision_model"},
        "tie_word_embeddings": False,
    }
    _write_json(model_dir / "config.json", config)
    _write_json(
        model_dir / "generation_config.json",
        {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "do_sample": False,
            "temperature": 1.0,
        },
    )

    gen = torch.Generator(device="cpu").manual_seed(20260724)
    tensors: dict[str, torch.Tensor] = {
        "model.llm.embed_tokens.weight": _randn((VOCAB_SIZE, HIDDEN_SIZE), gen),
        "model.llm.lm_head.weight": _randn((VOCAB_SIZE, HIDDEN_SIZE), gen),
        "model.llm.norm.weight": _ones((HIDDEN_SIZE,)),
    }

    kv_width = NUM_KV_HEADS * HEAD_DIM
    rel_width = NUM_HEADS * D_REL
    for layer_id in range(NUM_LAYERS):
        prefix = f"model.llm.layers.{layer_id}"
        tensors[f"{prefix}.attn_norm.weight"] = _ones((HIDDEN_SIZE,))
        tensors[f"{prefix}.mlp_norm.weight"] = _ones((HIDDEN_SIZE,))
        tensors[f"{prefix}.attn.wq_du.weight"] = _randn(
            (NUM_HEADS * HEAD_DIM, HIDDEN_SIZE), gen
        )
        tensors[f"{prefix}.attn.wk_dv.weight"] = _randn((kv_width, HIDDEN_SIZE), gen)
        tensors[f"{prefix}.attn.wv_dv.weight"] = _randn((kv_width, HIDDEN_SIZE), gen)
        tensors[f"{prefix}.attn.wr_du.weight"] = _randn((rel_width, HIDDEN_SIZE), gen)
        tensors[f"{prefix}.attn.wo_ud.weight"] = _randn(
            (HIDDEN_SIZE, NUM_HEADS * HEAD_DIM), gen
        )
        layer_rel_extent = (
            SLIDING_WINDOW_SIZE if layer_id in LOCAL_LAYER_IDS else REL_EXTENT
        )
        tensors[f"{prefix}.attn.rel_logits_proj.proj"] = _randn(
            (D_REL, layer_rel_extent), gen, scale=0.001
        )
        tensors[f"{prefix}.attn.q_norm.weight"] = _ones((HEAD_DIM,))
        tensors[f"{prefix}.attn.k_norm.weight"] = _ones((HEAD_DIM,))
        tensors[f"{prefix}.attn.k_sconv.weight"] = _randn(
            (kv_width, 1, SCONV_KERNEL_SIZE), gen, scale=0.005
        )
        tensors[f"{prefix}.attn.v_sconv.weight"] = _randn(
            (kv_width, 1, SCONV_KERNEL_SIZE), gen, scale=0.005
        )
        tensors[f"{prefix}.attn_sconv.weight"] = _randn(
            (HIDDEN_SIZE, 1, SCONV_KERNEL_SIZE), gen, scale=0.005
        )
        tensors[f"{prefix}.mlp_sconv.weight"] = _randn(
            (HIDDEN_SIZE, 1, SCONV_KERNEL_SIZE), gen, scale=0.005
        )
        tensors[f"{prefix}.mlp.w13_dn.weight"] = _randn(
            (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), gen
        )
        tensors[f"{prefix}.mlp.w2_md.weight"] = _randn(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE), gen
        )

    save_file(tensors, ckpt_path, metadata={"format": "pt"})


def _collect_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        out: list[float] = []
        for item in value.values():
            out.extend(_collect_numbers(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_collect_numbers(item))
        return out
    return []


def _extract_output_ids(output: Any) -> list[int]:
    if isinstance(output, dict):
        ids = output.get("output_ids")
        if isinstance(ids, list):
            return [int(x) for x in ids]
        meta = output.get("meta_info")
        if isinstance(meta, dict) and isinstance(meta.get("output_ids"), list):
            return [int(x) for x in meta["output_ids"]]
    if isinstance(output, list) and output:
        return _extract_output_ids(output[0])
    return []


def run_engine(model_dir: Path, prompt_len: int, max_new_tokens: int) -> dict[str, Any]:
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    os.environ.setdefault("ZE_AFFINITY_MASK", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")
    os.environ.setdefault("SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE", "1")
    overlay = prepare_sgl_kernel_overlay()

    import sglang as sgl

    print(f"Using sgl_kernel overlay: {overlay}", flush=True)
    engine = sgl.Engine(
        model_path=str(model_dir),
        tokenizer_path=str(model_dir),
        skip_tokenizer_init=True,
        trust_remote_code=True,
        load_format="safetensors",
        dtype="bfloat16",
        device="xpu",
        tp_size=1,
        attention_backend="intel_xpu",
        enable_multimodal=False,
        max_running_requests=1,
        max_total_tokens=1024,
        context_length=128,
        swa_full_tokens_ratio=1.0,
        mem_fraction_static=0.25,
        disable_prefill_cuda_graph=True,
        disable_decode_cuda_graph=True,
        skip_server_warmup=True,
        random_seed=0,
        log_level="info",
    )
    try:
        input_ids = list(range(3, 3 + prompt_len))
        output = engine.generate(
            input_ids=input_ids,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=1,
        )
    finally:
        engine.shutdown()

    output_ids = _extract_output_ids(output)
    if len(output_ids) != max_new_tokens:
        raise AssertionError(
            f"expected {max_new_tokens} generated token ids, got {output_ids}"
        )
    numbers = _collect_numbers(output)
    bad = [x for x in numbers if not math.isfinite(x)]
    if bad:
        raise AssertionError(f"offline generation returned non-finite values: {bad[:5]}")
    return {"input_ids": input_ids, "output_ids": output_ids, "raw": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/workspace/tmp/sglang_fake_inkling_xpu_smoke"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate checkpoint")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()

    write_fake_inkling_checkpoint(args.model_dir, force=args.force)
    result = run_engine(args.model_dir, args.prompt_len, args.max_new_tokens)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
