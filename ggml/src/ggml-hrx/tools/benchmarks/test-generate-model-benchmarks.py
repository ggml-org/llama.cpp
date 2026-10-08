#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate-model-benchmarks.py")
SPEC = importlib.util.spec_from_file_location("generate_model_benchmarks", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def gated_flash_attention_command(gate_bytes: int = 48128) -> dict:
    return {
        "ordinal": 59,
        "kernel": "loom_libs:ggml_flash_attention_f32_f16_wmma",
        "integer_parameters": {
            "query_token_count": 1,
            "key_value_token_count": 256,
        },
        "compile_parameters": {
            "ggml.flash_attention.apply_gate": "1",
            "ggml.flash_attention.query_head_count": "24",
            "ggml.flash_attention.key_value_head_count": "4",
            "ggml.flash_attention.qk_head_size": "256",
            "ggml.flash_attention.value_head_size": "256",
            "ggml.flash_attention.gate_stride_head": "512",
            "ggml.flash_attention.gate_stride_token": "12288",
        },
        "bindings": [
            {"name": "query", "length": 24576},
            {"name": "key", "length": 524288},
            {"name": "value", "length": 524288},
            {"name": "mask", "length": 512},
            {"name": "gate", "length": gate_bytes},
            {"name": "output", "length": 24576},
        ],
    }


class FlashAttentionCaseTest(unittest.TestCase):
    def test_materializes_distinct_gate_with_exact_binding_capacity(self) -> None:
        case = MODULE.case_flash_attention(
            "qwen38_tg8_flash_attention",
            gated_flash_attention_command(),
            "ggml_flash_attention_f32_f16_wmma",
            "ggml",
        )

        self.assertIn("%gate = check.generate.fill value(0.0) : tensor<12032xf32>", case)
        self.assertIn("%mask, %gate, %output)", case)
        self.assertIn("tensor<12032xf32>", case)
        self.assertNotIn("%mask, %query, %output)", case)

    def test_rejects_gate_binding_smaller_than_strided_access(self) -> None:
        with self.assertRaisesRegex(SystemExit, "strided access requires 12032"):
            MODULE.case_flash_attention(
                "qwen38_tg8_flash_attention",
                gated_flash_attention_command(gate_bytes=24576),
                "ggml_flash_attention_f32_f16_wmma",
                "ggml",
            )

    def test_rejects_gate_binding_not_aligned_to_f32(self) -> None:
        with self.assertRaisesRegex(SystemExit, "non-f32 gate binding length"):
            MODULE.case_flash_attention(
                "qwen38_tg8_flash_attention",
                gated_flash_attention_command(gate_bytes=48127),
                "ggml_flash_attention_f32_f16_wmma",
                "ggml",
            )


if __name__ == "__main__":
    unittest.main()
