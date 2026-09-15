from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("inkling_xpu_offline_engine_tp8_smoke.py")
SPEC = importlib.util.spec_from_file_location("inkling_tp8_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


def test_tp8_uses_the_tp4_mxfp4_grouped_gemm_layout() -> None:
    size = SMOKE.model_size_summary()

    assert SMOKE.TP_SIZE == 8
    assert SMOKE.EP_SIZE == 1
    assert SMOKE.MXFP4_TARGET_WEIGHT_BYTES < SMOKE.BF16_TARGET_WEIGHT_BYTES
    assert 5.0 < size["ideal_tp8_weight_gib_per_rank"] < 6.0
    assert size["mxfp4_mixed_weight_gib"] < 47.0


def test_tp8_retains_mtp_and_a_complete_attention_period() -> None:
    assert SMOKE.NUM_MTP_LAYERS == 8
    assert SMOKE.NUM_ROUTED_EXPERTS == 256
    assert SMOKE.LOCAL_LAYER_IDS == (0, 1, 2, 3, 4)
    assert set(range(SMOKE.NUM_LAYERS)) - set(SMOKE.LOCAL_LAYER_IDS) == {5}


def test_tp8_serving_shape_fits_generated_context() -> None:
    assert SMOKE.CONTEXT_LENGTH == 6144
    assert 4096 + 1024 <= SMOKE.CONTEXT_LENGTH
