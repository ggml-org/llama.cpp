#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conversion.base import gguf
from conversion.qwen import _QwenMtpMixin


class _ModelBase:
    def __init__(self, config_count: int | None, tensor_count: int):
        self.hparams = {"num_hidden_layers": 32}
        if config_count is not None:
            self.hparams["mtp_num_hidden_layers"] = config_count
        self.opt_num_mtp_layers = tensor_count


class _Model(_QwenMtpMixin, _ModelBase):
    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = False
    mtp_only = False


class TestQwenMtpLayerCount(unittest.TestCase):
    def test_ignores_configured_mtp_layers_missing_from_checkpoint(self):
        with self.assertLogs("hf-to-gguf", level="WARNING"):
            model = _Model(config_count=1, tensor_count=0)

        self.assertEqual(model.block_count, 32)

    def test_keeps_mtp_layers_present_in_checkpoint(self):
        model = _Model(config_count=1, tensor_count=1)

        self.assertEqual(model.block_count, 33)

    def test_counts_mtp_layers_when_config_value_is_missing(self):
        model = _Model(config_count=None, tensor_count=1)

        self.assertEqual(model.block_count, 33)


if __name__ == "__main__":
    unittest.main()
