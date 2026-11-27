"""Keep expert repacking bounded when checkpoint files interleave layers."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import ModelBase, gguf
from conversion.qwen import Qwen2MoeModel


class TestNVFP4ExpertStreaming(unittest.TestCase):
    def test_interleaved_files_finish_one_layer_at_a_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for expert in range(2):
                tensors = {}
                for layer in range(2):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
                    tensors[name] = torch.full((16, 32), layer * 10 + expert, dtype=torch.uint8)
                    tensors[name + "_scale"] = torch.ones((16, 4), dtype=torch.float8_e4m3fn)
                save_file(tensors, str(path / f"model-{expert + 1:05d}-of-00002.safetensors"))
            hp = {"model_type": "qwen2_moe", "architectures": ["Qwen2MoeForCausalLM"], "num_hidden_layers": 2,
                  "hidden_size": 4, "num_attention_heads": 1, "num_key_value_heads": 1, "num_experts": 2,
                  "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
            model = Qwen2MoeModel(path, gguf.LlamaFileType.MOSTLY_F16, path / "out.gguf", hparams=hp)
            events = []
            pack = ModelBase._nvfp4_pack
            add = model.gguf_writer.add_tensor

            def track_pack(weight, scale):
                events.append(("pack", int(weight[0, 0])))
                return pack(weight, scale)

            def track_add(name, *args, **kwargs):
                if name.endswith("ffn_down_exps.weight"):
                    events.append(("write", name))
                return add(name, *args, **kwargs)

            with patch.object(ModelBase, "_nvfp4_pack", staticmethod(track_pack)), patch.object(model.gguf_writer, "add_tensor", track_add):
                model.prepare_tensors()
            self.assertEqual(events, [
                ("pack", 0), ("pack", 1), ("write", "blk.0.ffn_down_exps.weight"),
                ("pack", 10), ("pack", 11), ("write", "blk.1.ffn_down_exps.weight"),
            ])
            for layer in range(2):
                info = model.gguf_writer.tensors[0][f"blk.{layer}.ffn_down_exps.weight"]
                self.assertEqual(info.dtype, gguf.GGMLQuantizationType.NVFP4)
                expected = np.stack([pack(torch.full((16, 32), layer * 10 + expert, dtype=torch.uint8), torch.ones((16, 4), dtype=torch.float8_e4m3fn))[0] for expert in range(2)])
                np.testing.assert_array_equal(info.tensor, expected)
            del model, add


if __name__ == "__main__":
    unittest.main()
