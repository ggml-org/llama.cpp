import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("moe_pack", Path(__file__).resolve().parents[1] / "tools/moe_pack.py")
moe_pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_pack)
from gguf import GGUFWriter


class TestMoePack(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.gguf"
        self.output = self.root / "pack"

    def write_model(self, omit=None, wrong_shape=False):
        writer = GGUFWriter(self.model, "qwen4exp")
        for layer in range(2):
            dtype = np.float32 if layer == 0 else np.float16
            for role in ("gate", "up", "down"):
                if (layer, role) == omit:
                    continue
                shape = (4, 8, 4) if role == "down" else (4, 4, 8)
                if wrong_shape and role == "down":
                    shape = (4, 4, 8)
                values = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + layer
                writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_round_trip(self):
        self.write_model()
        metadata = moe_pack.create_pack(self.model, self.output)
        self.assertEqual(metadata["layers"], 2)
        self.assertEqual(metadata["experts_per_layer"], 4)
        self.assertEqual(moe_pack.verify_pack(self.model, self.output)["verified_experts"], 6)
        self.assertEqual(moe_pack.verify_pack(self.model, self.output, True)["verified_experts"], 8)
        self.assertEqual(json.loads((self.output / "expert-pack.json").read_text()), metadata)

    def test_no_overwrite(self):
        self.write_model()
        moe_pack.create_pack(self.model, self.output)
        original = (self.output / "experts.bin").read_bytes()
        with self.assertRaises(FileExistsError):
            moe_pack.create_pack(self.model, self.output)
        self.assertEqual((self.output / "experts.bin").read_bytes(), original)

    def test_corrupt_pack(self):
        self.write_model()
        moe_pack.create_pack(self.model, self.output)
        with (self.output / "experts.bin").open("r+b") as target:
            target.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "weight mismatch"):
            moe_pack.verify_pack(self.model, self.output)

    def test_truncated_pack(self):
        self.write_model()
        moe_pack.create_pack(self.model, self.output)
        (self.output / "experts.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "byte size"):
            moe_pack.verify_pack(self.model, self.output)

    def test_incomplete_layer(self):
        self.write_model(omit=(1, "up"))
        with self.assertRaisesRegex(ValueError, "requires separate"):
            moe_pack.create_pack(self.model, self.output)
        self.assertFalse(self.output.exists())

    def test_shape_mismatch(self):
        self.write_model(wrong_shape=True)
        with self.assertRaisesRegex(ValueError, "projection shape"):
            moe_pack.expert_layout(self.model)


if __name__ == "__main__":
    unittest.main()
