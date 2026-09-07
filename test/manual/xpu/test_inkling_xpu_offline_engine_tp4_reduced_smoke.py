from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name(
    "inkling_xpu_offline_engine_tp4_reduced_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("inkling_tp4_reduced_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


def test_tp4_reduced_model_fits_four_b60s() -> None:
    size = SMOKE.model_size_summary()

    assert SMOKE.TP_SIZE == 4
    assert SMOKE.EP_SIZE == 1
    assert SMOKE.NUM_LAYERS == 6
    assert 9.0 < size["ideal_tp4_weight_gib_per_rank"] < 10.0
    assert size["mxfp4_mixed_weight_gib"] < 38.0


def test_tp4_reduced_model_retains_dense_moe_local_and_global_paths() -> None:
    assert SMOKE.DENSE_MLP_IDX == 2
    assert SMOKE.NUM_HEADS == 64
    assert SMOKE.NUM_ROUTED_EXPERTS == 256
    assert SMOKE.LOCAL_LAYER_IDS == (0, 1, 2, 3, 4)
    assert set(range(SMOKE.NUM_LAYERS)) - set(SMOKE.LOCAL_LAYER_IDS) == {5}
    assert SMOKE.DEFAULT_XPU_AFFINITY_MASK == "4,5,6,7"
