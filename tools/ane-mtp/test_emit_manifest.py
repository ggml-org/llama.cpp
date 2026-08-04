"""Unit tests for tools/ane-mtp/emit-manifest-from-mlmodelc.py.

Validates the bridge tool that introspects an existing multifunction
.mlmodelc and emits the ane_state_layout.v1 manifest sidecar. The
test uses a synthetic .mlmodelc (the metadata.json it expects) so it
can run in CI without a real Core ML bundle.

Run with:
  cd tools/ane-mtp && python3 test_emit_manifest.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from emit_manifest_from_mlmodelc import (  # noqa: E402
    ANE_MIN_ALLOC_BYTES,
    ANE_PAGE_BYTES,
    ANE_SIMD_ALIGN,
    build_manifest,
    parse_function_bucket,
    parse_function_role,
    parse_shape,
    slot_size,
)


def make_synthetic_mlmodelc(tmpdir: Path) -> Path:
    """Write a synthetic metadata.json that mimics a multifunction
    prefill bundle (3 functions, each with token_ids/positions
    inputs and hidden_states/key_states/value_states outputs)."""
    mlmodelc = tmpdir / "test-bundle.mlmodelc"
    mlmodelc.mkdir()
    metadata = [{
        "specificationVersion": 9,
        "modelType": {"name": "MLModelType_mlProgram"},
        "defaultFunctionName": "prefill_s128",
        "functions": [
            {
                "name": "prefill_s128",
                "inputSchema": [
                    {"name": "token_ids", "dataType": "Int32",
                     "shape": "[1, 128]"},
                    {"name": "positions", "dataType": "Int32",
                     "shape": "[1, 128]"},
                ],
                "outputSchema": [
                    {"name": "hidden_states", "dataType": "Float16",
                     "shape": "[1, 128, 3840]"},
                    {"name": "key_states", "dataType": "Float16",
                     "shape": "[1, 128, 2048]"},
                    {"name": "value_states", "dataType": "Float16",
                     "shape": "[1, 128, 2048]"},
                ],
                "stateSchema": [],
            },
            {
                "name": "prefill_s256",
                "inputSchema": [
                    {"name": "token_ids", "dataType": "Int32",
                     "shape": "[1, 256]"},
                    {"name": "positions", "dataType": "Int32",
                     "shape": "[1, 256]"},
                ],
                "outputSchema": [
                    {"name": "hidden_states", "dataType": "Float16",
                     "shape": "[1, 256, 3840]"},
                    {"name": "key_states", "dataType": "Float16",
                     "shape": "[1, 256, 2048]"},
                    {"name": "value_states", "dataType": "Float16",
                     "shape": "[1, 256, 2048]"},
                ],
                "stateSchema": [],
            },
        ],
    }]
    (mlmodelc / "metadata.json").write_text(json.dumps(metadata))
    return mlmodelc


class HelpersTest:
    def test_parse_shape_from_string(self):
        assert parse_shape("[1, 128, 3840]") == [1, 128, 3840]
        assert parse_shape("[]") == []
        assert parse_shape("[128]") == [128]

    def test_parse_shape_from_list(self):
        assert parse_shape([1, 128, 3840]) == [1, 128, 3840]
        assert parse_shape([]) == []

    def test_slot_size_alignment(self):
        # 1 fp16 element = 2 bytes, padded to 16 -> 16
        assert slot_size("Float16", [1]) == 16
        # 8 fp16 elements = 16 bytes, no padding needed
        assert slot_size("Float16", [8]) == 16
        # 9 fp16 elements = 18 bytes, padded to 32
        assert slot_size("Float16", [9]) == 32
        # 1 int32 element = 4 bytes, padded to 16
        assert slot_size("Int32", [1]) == 16
        # 4 int32 elements = 16 bytes, no padding
        assert slot_size("Int32", [4]) == 16

    def test_parse_function_role(self):
        assert parse_function_role("prefill_s128") == "prefill"
        assert parse_function_role("dflash_b8") == "dflash"
        assert parse_function_role("hybrid_b4") == "hybrid"
        assert parse_function_role("sync") == "sync"
        assert parse_function_role("reset") == "reset"
        assert parse_function_role("mtp_predict") == "mtp"

    def test_parse_function_bucket(self):
        assert parse_function_bucket("prefill_s128") == 128
        assert parse_function_bucket("prefill_s256") == 256
        assert parse_function_bucket("dflash_b8") == 8
        assert parse_function_bucket("hybrid_b4") == 4
        assert parse_function_bucket("sync") == 0
        assert parse_function_bucket("mtp_predict") == 0


class BuildManifestTest:
    def test_multifunction_manifest_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            mlmodelc = make_synthetic_mlmodelc(Path(tmp))
            m = build_manifest(mlmodelc, "test-bundle")
            assert m["version"] == 1
            assert m["bundle_name"] == "test-bundle"
            assert m["model_type"] == "ml_program"
            assert m["state_size_bytes"] >= ANE_MIN_ALLOC_BYTES
            assert m["state_size_bytes"] % ANE_PAGE_BYTES == 0
            # 2 functions x (2 inputs + 3 outputs) = 10 slots
            assert len(m["slots"]) == 10
            assert len(m["functions"]) == 2
            assert m["dependencies"] == []

    def test_slot_kinds_and_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            mlmodelc = make_synthetic_mlmodelc(Path(tmp))
            m = build_manifest(mlmodelc, "test-bundle")
            # Each function gets 2 input slots + 3 output slots,
            # in declaration order. The first function's slots are
            # 0-4, the second's are 5-9.
            for slot in m["slots"]:
                assert slot["offset"] % ANE_SIMD_ALIGN == 0, \
                    f"slot {slot['name']} offset not aligned"
                assert slot["size_bytes"] % ANE_SIMD_ALIGN == 0, \
                    f"slot {slot['name']} size not aligned"
            # KV outputs are STATE-kind; hidden_states is OUTPUT-kind
            for f in m["functions"]:
                outputs = f["output_slots"]
                # Find the per-function outputs by name pattern
                hs_slot = next(s for s in m["slots"]
                               if s["name"] == f"{f['name']}.hidden_states")
                ks_slot = next(s for s in m["slots"]
                               if s["name"] == f"{f['name']}.key_states")
                vs_slot = next(s for s in m["slots"]
                               if s["name"] == f"{f['name']}.value_states")
                assert hs_slot["kind"] == "output"
                assert ks_slot["kind"] == "state"
                assert vs_slot["kind"] == "state"
            # Inputs are INPUT-kind
            for f in m["functions"]:
                for sname in f["input_slots"]:
                    s = next(s for s in m["slots"] if s["name"] == sname)
                    assert s["kind"] == "input"

    def test_function_bucket_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mlmodelc = make_synthetic_mlmodelc(Path(tmp))
            m = build_manifest(mlmodelc, "test-bundle")
            assert m["functions"][0]["bucket"] == 128
            assert m["functions"][1]["bucket"] == 256
            assert m["functions"][0]["role"] == "prefill"
            assert m["functions"][1]["role"] == "prefill"

    def test_state_size_covers_all_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            mlmodelc = make_synthetic_mlmodelc(Path(tmp))
            m = build_manifest(mlmodelc, "test-bundle")
            # state_size_bytes is at least the sum of all slot sizes
            # (plus rounding); specifically the layout's last slot
            # plus its size must be <= state_size_bytes.
            last_offset = max(s["offset"] + s["size_bytes"]
                              for s in m["slots"])
            assert last_offset <= m["state_size_bytes"]
            # And state_size_bytes is a multiple of 16KB.
            assert m["state_size_bytes"] % ANE_PAGE_BYTES == 0


def run() -> int:
    """Minimal test runner so the file can be executed without pytest."""
    failures: list[str] = []
    test_classes = [HelpersTest, BuildManifestTest]
    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if not name.startswith("test_"):
                continue
            try:
                getattr(instance, name)()
                print(f"  ok   {cls.__name__}.{name}")
            except AssertionError as e:
                print(f"  FAIL {cls.__name__}.{name}: {e}")
                failures.append(f"{cls.__name__}.{name}")
            except Exception as e:
                print(f"  ERR  {cls.__name__}.{name}: {type(e).__name__}: {e}")
                failures.append(f"{cls.__name__}.{name}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nALL PASSED ({sum(1 for c in test_classes for n in dir(c()) if n.startswith('test_'))} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
