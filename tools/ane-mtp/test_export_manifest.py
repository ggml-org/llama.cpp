"""Unit tests for tools/ane-mtp/export-manifest.py.

Validates the post-export adapter that emits the ane_state_layout.v1
manifest sidecar for MTP / DFlash / hybrid bundles. The test uses a
synthetic GGUF (built with the gguf-py library) so it can run in CI
without a real Core ML bundle. The synthetic GGUF embeds a tiny
.mlmodelc-shaped directory tree as tensors; the script materializes
the .mlmodelc, reads its metadata.json, builds the manifest, and
writes the sidecar.

Run with:
  cd tools/ane-mtp && python3 test_export_manifest.py
"""

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Skip the test if gguf-py is not importable. The test driver
# (tools/ane-mtp/test_all.py) should still work without gguf.
try:
    import numpy as np
    from gguf import GGUFReader, GGUFWriter
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

from export_manifest import (  # noqa: E402
    EXPERIMENTAL_FLAG,
    MTP_KEY_PREFIX,
    build_manifest_from_gguf,
    find_bundle_prefix,
    load_gguf_keys,
)
from state_layout import StateLayout  # noqa: E402


def make_synthetic_mlmodelc(tmpdir: Path) -> Path:
    """Write a synthetic multifunction .mlmodelc directory.

    Mimics the gemma4 MTP bundle: 2 functions, one MTP predict
    and one DFlash block drafter. The metadata.json shape
    matches what emit_manifest_from_mlmodelc.build_manifest
    expects. The function names are role-prefixed so the
    role inference in build_manifest maps them to a valid
    state_layout role (mtp / dflash).
    """
    mlmodelc = tmpdir / "batch-1.mlmodelc"
    mlmodelc.mkdir()
    metadata = [{
        "specificationVersion": 9,
        "modelType": {"name": "MLModelType_mlProgram"},
        "defaultFunctionName": "mtp_predict",
        "functions": [
            {
                "name": "mtp_predict",
                "inputSchema": [
                    {"name": "token_ids", "dataType": "Int32",
                     "shape": "[1, 128]"},
                    {"name": "positions", "dataType": "Int32",
                     "shape": "[1, 128]"},
                ],
                "outputSchema": [
                    {"name": "top_token", "dataType": "Int32",
                     "shape": "[1]"},
                    {"name": "confidence", "dataType": "Float16",
                     "shape": "[1]"},
                    {"name": "next_hidden", "dataType": "Float16",
                     "shape": "[1, 3840]"},
                ],
                "stateSchema": [],
            },
            {
                "name": "dflash_b4",
                "inputSchema": [
                    {"name": "target_features", "dataType": "Float16",
                     "shape": "[1, 4, 3840]"},
                    {"name": "token_ids", "dataType": "Int32",
                     "shape": "[1, 4]"},
                ],
                "outputSchema": [
                    {"name": "draft_tokens", "dataType": "Int32",
                     "shape": "[1, 4]"},
                    {"name": "confidence", "dataType": "Float16",
                     "shape": "[1, 4]"},
                ],
                "stateSchema": [],
            },
        ],
    }]
    (mlmodelc / "metadata.json").write_text(json.dumps(metadata))
    (mlmodelc / "model.mil").write_text("placeholder")
    (mlmodelc / "weights" / "weight.bin").parent.mkdir(
        parents=True, exist_ok=True)
    (mlmodelc / "weights" / "weight.bin").write_bytes(b"\x00" * 16)
    return mlmodelc


def make_synthetic_gguf_with_mlmodelc(
        gguf_path: Path,
        mlmodelc: Path,
        batch: int = 1) -> None:
    """Write a synthetic GGUF that embeds the .mlmodelc as tensors.

    Mirrors the layout from tools/ane-mtp/embed-bundle-fixture.py:
    the per-bucket prefix is "mtp.ane.bucket.N", each file is
    embedded as a tensor with the relative path as a sibling
    string kv, and file_count is a uint32 kv. The function names
    are also embedded so the manifest's role/bucket derivation
    works.
    """
    if not HAS_GGUF:
        raise unittest.SkipTest("gguf-py not available")
    writer = GGUFWriter(str(gguf_path), "ane-mtp-test")
    prefix = f"{MTP_KEY_PREFIX}.bucket.{batch}"
    files = sorted(p for p in mlmodelc.rglob("*") if p.is_file())
    for index, path in enumerate(files):
        relative = path.relative_to(mlmodelc).as_posix()
        data = path.read_bytes()
        # GGUFWriter.add_tensor only supports signed integer
        # dtypes. We pass an int8 view of the bytes; the reader
        # casts back to bytes via the memoryview in
        # materialize_mlmodelc.
        writer.add_tensor(
            f"{prefix}.file.{index:04d}",
            np.frombuffer(data, dtype=np.int8),
        )
        writer.add_string(f"{prefix}.file.{index:04d}.path", relative)
    writer.add_uint32(f"{prefix}.file_count", len(files))
    writer.add_array(f"{prefix}.functions",
                     ["mtp_predict", "dflash_b4", "sync", "reset"])
    writer.add_string(f"{MTP_KEY_PREFIX}.format", "mlmodelc-buckets-v2")
    writer.add_array(f"{MTP_KEY_PREFIX}.batch_buckets", [batch])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAS_GGUF, "gguf-py not available")
class FindBundlePrefixTest(unittest.TestCase):
    def test_no_keys(self):
        self.assertIsNone(find_bundle_prefix([]))

    def test_only_unrelated_keys(self):
        self.assertIsNone(find_bundle_prefix(
            ["general.architecture", "general.name"]))

    def test_bucket_prefix(self):
        prefix = find_bundle_prefix(
            ["mtp.ane.bucket.1.file_count",
             "mtp.ane.bucket.1.file.0000",
             "mtp.ane.bucket.1.functions"])
        self.assertEqual(prefix, "mtp.ane.bucket.1")

    def test_bundle_prefix(self):
        # The multifunction layout (no per-bucket number) uses
        # the bare "mtp.ane.bundle" prefix; the find_bundle_prefix
        # helper returns the prefix itself, not the first subkey.
        prefix = find_bundle_prefix(
            ["mtp.ane.bundle.file_count",
             "mtp.ane.bundle.functions"])
        # The helper returns the longest matching dotted key
        # rooted at the prefix; for the bundle case this is
        # the prefix itself (no number suffix).
        self.assertTrue(prefix == "mtp.ane.bundle" or
                        prefix.startswith("mtp.ane.bundle."),
                        f"unexpected prefix {prefix!r}")


@unittest.skipUnless(HAS_GGUF, "gguf-py not available")
class BuildManifestFromGguFTest(unittest.TestCase):
    def test_predict_and_dflash(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_path = Path(staging)
            mlmodelc = make_synthetic_mlmodelc(staging_path)
            gguf_path = staging_path / "test.gguf"
            make_synthetic_gguf_with_mlmodelc(gguf_path, mlmodelc)
            mlpackage = staging_path / "batch-1.mlpackage"
            mlpackage.mkdir()
            manifest, materialized = build_manifest_from_gguf(
                gguf_path, mlpackage, None)
            # Clean up the materialized .mlmodelc explicitly; the
            # production path cleans up in main() but the helper
            # returns the dir for inspection.
            shutil.rmtree(materialized.parent, ignore_errors=True)
            self.assertEqual(manifest["bundle_name"], "batch-1")
            self.assertEqual(manifest["model_type"], "ml_program")
            self.assertEqual(manifest["_experimental"], EXPERIMENTAL_FLAG)
            # 2 functions: predict, dflash_b4. Each contributes
            # 2 inputs + 3 outputs (predict) or 2 inputs + 2
            # outputs (dflash_b4) = 5 + 4 = 9 slots.
            self.assertEqual(len(manifest["functions"]), 2)
            self.assertEqual(len(manifest["slots"]), 9)
            # Validate via StateLayout.from_dict.
            layout = StateLayout.from_dict(manifest)
            self.assertEqual(layout.version, 1)
            self.assertEqual(layout.bundle_name, "batch-1")

    def test_validation_rejects_bad_manifest(self):
        # Manually craft a manifest with a slot that violates
        # alignment, then verify StateLayout.from_dict raises.
        bad = {
            "version": 1,
            "bundle_name": "bad",
            "state_size_bytes": 65536,
            "model_type": "ml_program",
            "slots": [{
                "name": "x",
                "kind": "input",
                "dtype": "f32",
                "shape": [256],
                "offset": 1,  # not 16-byte aligned
                "size_bytes": 1024,
            }],
            "functions": [],
            "dependencies": [],
        }
        with self.assertRaises(ValueError):
            StateLayout.from_dict(bad)


@unittest.skipUnless(HAS_GGUF, "gguf-py not available")
class MaterializeMlmodelcTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_path = Path(staging)
            mlmodelc = make_synthetic_mlmodelc(staging_path)
            gguf_path = staging_path / "test.gguf"
            make_synthetic_gguf_with_mlmodelc(gguf_path, mlmodelc)
            # Re-materialize and verify the directory tree matches.
            keys = load_gguf_keys(gguf_path)
            prefix = find_bundle_prefix(keys)
            self.assertEqual(prefix, "mtp.ane.bucket.1")
            out_staging = Path(tempfile.mkdtemp(prefix="ane-round-trip-"))
            try:
                from export_manifest import materialize_mlmodelc
                materialized = materialize_mlmodelc(
                    gguf_path, prefix, out_staging)
                # metadata.json + model.mil + weights/weight.bin
                # should all be present.
                self.assertTrue(
                    (materialized / "metadata.json").is_file())
                self.assertTrue(
                    (materialized / "model.mil").is_file())
                self.assertTrue(
                    (materialized / "weights" / "weight.bin").is_file())
            finally:
                shutil.rmtree(out_staging, ignore_errors=True)


def run() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run())
