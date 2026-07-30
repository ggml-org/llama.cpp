import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "tessera" / "family_coverage.py"
SPEC = importlib.util.spec_from_file_location("tessera_family_coverage", MODULE)
assert SPEC and SPEC.loader
coverage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coverage
SPEC.loader.exec_module(coverage)


def base_tensors():
    return [
        ("token_embd.weight", (256, 32000)),
        ("output_norm.weight", (256,)),
        ("blk.0.attn_norm.weight", (256,)),
        ("blk.0.ffn_norm.weight", (256,)),
        ("blk.0.attn_q.weight", (256, 256)),
        ("blk.0.ffn_gate.weight", (256, 512)),
    ]


def test_gemma4_dense_contract():
    receipt = coverage.build_coverage_receipt(
        "gemma4", base_tensors(), 1, 1
    )
    assert receipt["family"] == "gemma4"
    assert receipt["features"] == {
        "text": True,
        "moe": False,
        "drafter": False,
        "multimodal": False,
    }
    assert receipt["complete"] is True


def test_gemma4_unified_requires_payload():
    with pytest.raises(ValueError, match="no --vision-from payload"):
        coverage.build_coverage_receipt(
            "gemma4",
            base_tensors(),
            1,
            1,
            has_multimodal_source=True,
        )


def test_qwen36_moe_and_drafter_contract():
    tensors = base_tensors() + [
        ("blk.0.ffn_gate_inp.weight", (256, 8)),
        ("blk.0.ffn_gate_up_exps.weight", (8, 1024, 256)),
        ("blk.0.ffn_down_exps.weight", (8, 256, 512)),
        ("blk.1.attn_norm.weight", (256,)),
    ]
    receipt = coverage.build_coverage_receipt(
        "qwen35moe", tensors, 1, 2
    )
    assert receipt["family"] == "qwen3.6"
    assert receipt["features"]["moe"] is True
    assert receipt["features"]["drafter"] is True
    assert receipt["treatments"]["tile640-expert-matmul"] == 2


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="no Tessera family coverage contract"):
        coverage.build_coverage_receipt("llama", base_tensors(), 1, 1)


def test_required_tensor_is_enforced():
    tensors = [
        tensor for tensor in base_tensors()
        if tensor[0] != "output_norm.weight"
    ]
    with pytest.raises(ValueError, match="missing required tensors"):
        coverage.build_coverage_receipt("gemma4", tensors, 1, 1)
