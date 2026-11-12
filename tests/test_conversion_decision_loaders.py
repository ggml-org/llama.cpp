"""Test the pplx-decider hparams matcher with small synthetic directories; no model downloads required.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_conversion_decision_loaders.py' -v
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from conversion.pplx_decider import _is_pplx_decider_checkpoint

QWEN35_CONFIG = {"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5"}
QWEN35_TEXT_CONFIG = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text"}
GEMMA4_CONFIG = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"}


class TestPplxDeciderMatcher(unittest.TestCase):
    def matches(self, config, decision, readout: bool = True) -> bool:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "config.json").write_text(json.dumps(config))
            (path / "decision_config.json").write_text(json.dumps(decision))
            if readout:
                (path / "readout.safetensors").write_bytes(b"")
            return _is_pplx_decider_checkpoint(path)

    def test_pplx_decider(self):
        self.assertTrue(self.matches(QWEN35_CONFIG, {"format_version": 1, "temperature": 1.0}))

    def test_pplx_decider_text_only(self):
        self.assertTrue(self.matches(QWEN35_TEXT_CONFIG, {"format_version": 1, "temperature": 1.0}))

    def test_no_readout(self):
        self.assertFalse(self.matches(QWEN35_CONFIG, {"format_version": 1, "temperature": 1.0}, readout=False))

    def test_vllm_d3(self):
        self.assertFalse(self.matches(QWEN35_CONFIG, {"format_id": "d3-code-readout-v1", "format_version": 1}))

    def test_strands_decider(self):
        self.assertFalse(self.matches(GEMMA4_CONFIG, {"interface": "strands-decider", "format_version": 1}))

    def test_other_backbone(self):
        self.assertFalse(self.matches(GEMMA4_CONFIG, {"format_version": 1, "temperature": 1.0}))


if __name__ == "__main__":
    unittest.main()
