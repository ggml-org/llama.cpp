#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "tessera"
    / "semantic-repair-transfer.py"
)
SPEC = importlib.util.spec_from_file_location("semantic_repair_transfer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def checkpoint(scale: float, perturb: float = 0.0):
    base = np.linspace(1.0, 4.0, 128)
    values = scale * base
    if perturb:
        values = values.copy()
        values[:16] *= perturb
    return {
        "blk.0.ffn_up.weight": {
            "in_sum2": values,
            "in_sumabs": np.sqrt(values),
            "in_sum4": values**2,
            "counts": np.asarray([scale]),
        }
    }


prototype = checkpoint(64.0)
matching = checkpoint(128.0)
result = MODULE.evaluate_transfer(prototype, matching, 0.15, 0.70, 0.90, 0.05)
assert result["transferable"]
assert result["passed_tensors"] == 1

shifted = checkpoint(128.0, perturb=8.0)
result = MODULE.evaluate_transfer(prototype, shifted, 0.15, 0.70, 0.90, 0.05)
assert not result["transferable"]
assert result["passed_tensors"] == 0
