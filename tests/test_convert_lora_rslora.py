"""Test the LoRA alpha that convert_lora_to_gguf.py stores for rsLoRA adapters. No downloads required.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_convert_lora_rslora.py' -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from conversion.base import gguf

ROOT = Path(__file__).resolve().parent.parent


class TestLoraConversion(unittest.TestCase):
    def convert(self, use_rslora: bool, r: int = 4, lora_alpha: float = 8.0) -> float:
        with tempfile.TemporaryDirectory() as directory:
            base, lora, out = Path(directory, "base"), Path(directory, "lora"), Path(directory, "lora.gguf")
            base.mkdir()
            lora.mkdir()
            (base / "config.json").write_text(json.dumps({
                "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 32,
                "max_position_embeddings": 64, "rms_norm_eps": 1e-5,
            }))
            (lora / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "r": r, "lora_alpha": lora_alpha, "use_rslora": use_rslora,
                "target_modules": ["q_proj"], "base_model_name_or_path": "unused",
            }))
            prefix = "base_model.model.model.layers.0.self_attn.q_proj"
            save_file({
                f"{prefix}.lora_A.weight": torch.randn(r, 8),
                f"{prefix}.lora_B.weight": torch.randn(8, r),
            }, str(lora / "adapter_model.safetensors"))
            subprocess.run(
                [sys.executable, "convert_lora_to_gguf.py", str(lora), "--base", str(base), "--outfile", str(out),
                 "--outtype", "f32"],
                check=True, cwd=ROOT, capture_output=True,
            )
            field = gguf.GGUFReader(out).get_field(gguf.Keys.Adapter.LORA_ALPHA)
            assert field is not None
            return float(field.contents())

    def test_lora_alpha(self):
        self.assertAlmostEqual(self.convert(use_rslora=False), 8.0, places=5)

    def test_rslora_alpha(self):
        # llama.cpp scale alpha / r must match PEFT scale lora_alpha / sqrt(r)
        self.assertAlmostEqual(self.convert(use_rslora=True) / 4, 8.0 / math.sqrt(4), places=5)


if __name__ == "__main__":
    unittest.main()
