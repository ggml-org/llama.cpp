#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[1] / "tools" / "tessera" / "source-epoch.py"
SPEC = importlib.util.spec_from_file_location("tessera_source_epoch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourceEpochTests(unittest.TestCase):
    def component(self, root: Path, name: str, namespace: str, value: float) -> dict:
        directory = root / name
        directory.mkdir()
        save_file(
            {
                "model.weight": torch.full((2, 3), value, dtype=torch.bfloat16),
                "model.bias": torch.full((2,), value, dtype=torch.float32),
            },
            directory / "model.safetensors",
        )
        (directory / "config.json").write_text(
            json.dumps({"model_type": name}), encoding="utf-8"
        )
        return {
            "name": name,
            "namespace": namespace,
            "path": str(directory),
            "upstream_repo": f"owner/{name}",
            "upstream_revision": "revision",
            "license": "Apache-2.0",
            "redistribution": True,
        }

    def test_assembly_namespaces_and_validates_all_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema": MODULE.MANIFEST_SCHEMA,
                "epoch": 3,
                "components": [
                    self.component(root, "gemma", "gemma", 1),
                    self.component(root, "tts", "speech.qwen", 2),
                ],
            }
            output = root / "bundle"
            receipt = MODULE.assemble(manifest, output, max_shard_bytes=20)
            validated = MODULE.validate_bundle(output)
            self.assertEqual(receipt["artifact_digest"], validated["artifact_digest"])
            self.assertEqual(receipt["epoch"], 3)
            self.assertEqual(receipt["tensor_count"], 4)
            index = json.loads(
                (output / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(index["weight_map"]),
                {
                    "gemma.model.weight",
                    "gemma.model.bias",
                    "speech.qwen.model.weight",
                    "speech.qwen.model.bias",
                },
            )
            self.assertTrue((output / "components/gemma/config.json").is_file())
            self.assertTrue((output / "components/speech/qwen/config.json").is_file())

    def test_sealed_manifest_has_no_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema": MODULE.MANIFEST_SCHEMA,
                "epoch": 0,
                "components": [self.component(root, "gemma", "gemma", 1)],
            }
            sealed = MODULE.seal_manifest(manifest)
            encoded = json.dumps(sealed)
            self.assertNotIn(str(root), encoded)
            self.assertIn("source_digest", sealed)

    def test_validation_rejects_modified_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema": MODULE.MANIFEST_SCHEMA,
                "epoch": 1,
                "components": [self.component(root, "gemma", "gemma", 1)],
            }
            output = root / "bundle"
            receipt = MODULE.assemble(manifest, output, max_shard_bytes=1024)
            shard = output / receipt["shards"][0]["path"]
            shard.write_bytes(shard.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                MODULE.validate_bundle(output)

    def test_publication_rejects_unresolved_component_license(self):
        receipt = {
            "components": [
                {"name": "Gemma", "redistribution": True},
                {"name": "DFlash", "redistribution": False},
            ]
        }
        with self.assertRaisesRegex(ValueError, "DFlash"):
            MODULE.require_publishable(receipt)

    def test_fetch_rejects_mutable_upstream_revision(self):
        manifest = {
            "schema": MODULE.MANIFEST_SCHEMA,
            "epoch": 0,
            "components": [{
                "name": "Gemma",
                "namespace": "gemma",
                "upstream_repo": "google/gemma",
                "upstream_revision": "main",
                "license": "Apache-2.0",
                "redistribution": True,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "immutable commit"):
                MODULE.fetch_manifest(manifest, Path(directory))

    def test_training_lineage_is_bound_into_source_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = self.component(root, "gemma", "gemma", 1)
            first = {
                "schema": MODULE.MANIFEST_SCHEMA,
                "epoch": 1,
                "lineage": {"training_corpus_epoch": 1},
                "components": [component],
            }
            second = {
                "schema": MODULE.MANIFEST_SCHEMA,
                "epoch": 1,
                "lineage": {"training_corpus_epoch": 2},
                "components": [component],
            }
            self.assertNotEqual(
                MODULE.seal_manifest(first)["source_digest"],
                MODULE.seal_manifest(second)["source_digest"],
            )


if __name__ == "__main__":
    unittest.main()
