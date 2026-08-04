"""Unit tests for tools/ane-mtp/state_layout.py.

Validates the ane_state_layout_v1 manifest format: schema version,
slot alignment, slot/function name uniqueness, dead-state detection,
dependency references, and the W0 matmul builder's output.

Run with:
  cd tools/ane-mtp && python3 -m pytest test_state_layout.py -v

Or directly:
  cd tools/ane-mtp && python3 test_state_layout.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from state_layout import (  # noqa: E402
    ANE_MIN_ALLOC_BYTES,
    ANE_PAGE_BYTES,
    ANE_SIMD_ALIGN,
    DTYPE_BYTES,
    ROLE_MATMUL,
    ROLE_MTP,
    ROLE_PREFILL,
    ROLE_RESET,
    ROLE_SYNC,
    SCHEMA_VERSION,
    SLOT_KIND_INPUT,
    SLOT_KIND_OUTPUT,
    SLOT_KIND_SCRATCH,
    SLOT_KIND_STATE,
    Dependency,
    FunctionSpec,
    StateLayout,
    StateSlot,
    manifest_path_for,
)


class StateSlotTest(unittest.TestCase):
    def test_valid_slot_passes(self):
        slot = StateSlot(
            name="x", kind=SLOT_KIND_INPUT, dtype="f32",
            shape=[256], offset=0, size_bytes=1024,
        )
        slot.validate()

    def test_misaligned_offset_rejected(self):
        slot = StateSlot(
            name="x", kind=SLOT_KIND_INPUT, dtype="f32",
            shape=[256], offset=1, size_bytes=1024,
        )
        with self.assertRaises(ValueError):
            slot.validate()

    def test_misaligned_size_rejected(self):
        slot = StateSlot(
            name="x", kind=SLOT_KIND_INPUT, dtype="f32",
            shape=[256], offset=0, size_bytes=1023,  # not 16-aligned
        )
        with self.assertRaises(ValueError):
            slot.validate()

    def test_size_smaller_than_shape_rejected(self):
        slot = StateSlot(
            name="x", kind=SLOT_KIND_INPUT, dtype="f32",
            shape=[256], offset=0, size_bytes=512,  # need 1024 for 256 f32
        )
        with self.assertRaises(ValueError):
            slot.validate()

    def test_zero_dim_shape_rejected(self):
        slot = StateSlot(
            name="x", kind=SLOT_KIND_INPUT, dtype="f32",
            shape=[0], offset=0, size_bytes=16,
        )
        with self.assertRaises(ValueError):
            slot.validate()

    def test_bad_dtype_rejected(self):
        slot = StateSlot(
            name="x", kind=SLOT_KIND_INPUT, dtype="bf16",  # not supported
            shape=[256], offset=0, size_bytes=1024,
        )
        with self.assertRaises(ValueError):
            slot.validate()

    def test_bad_kind_rejected(self):
        slot = StateSlot(
            name="x", kind="temporary", dtype="f32",
            shape=[256], offset=0, size_bytes=1024,
        )
        with self.assertRaises(ValueError):
            slot.validate()


class FunctionSpecTest(unittest.TestCase):
    def test_valid_function_passes(self):
        func = FunctionSpec(
            name="main", role=ROLE_MATMUL, bucket=256,
            stateful=False, input_slots=["x"], output_slots=["y"],
        )
        func.validate()

    def test_sync_must_disable_ane(self):
        func = FunctionSpec(
            name="sync", role=ROLE_SYNC, use_ane=True,
        )
        with self.assertRaises(ValueError):
            func.validate()

    def test_sync_with_ane_disabled_passes(self):
        func = FunctionSpec(
            name="sync", role=ROLE_SYNC, use_ane=False,
        )
        func.validate()

    def test_too_many_inputs_rejected(self):
        func = FunctionSpec(
            name="too_many", role=ROLE_MATMUL,
            input_slots=[f"in{i}" for i in range(9)],
        )
        with self.assertRaises(ValueError):
            func.validate()


class StateLayoutTest(unittest.TestCase):
    def test_w0_matmul_layout_validates(self):
        layout = StateLayout.for_w0_matmul("w0-256x256", n=256)
        layout.validate()  # no exception
        self.assertEqual(layout.version, SCHEMA_VERSION)
        self.assertEqual(len(layout.slots), 2)
        self.assertEqual(layout.slots[0].name, "x")
        self.assertEqual(layout.slots[1].name, "y")
        self.assertEqual(len(layout.functions), 1)
        self.assertEqual(layout.functions[0].role, ROLE_MATMUL)
        self.assertEqual(layout.state_size_bytes, ANE_MIN_ALLOC_BYTES)

    def test_duplicate_slot_names_rejected(self):
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="dup",
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
            slots=[
                StateSlot(name="x", kind=SLOT_KIND_INPUT, dtype="f32",
                          shape=[256], offset=0, size_bytes=1024),
                StateSlot(name="x", kind=SLOT_KIND_OUTPUT, dtype="f32",
                          shape=[256], offset=1024, size_bytes=1024),
            ],
            functions=[FunctionSpec(name="main", role=ROLE_MATMUL,
                                    input_slots=["x"], output_slots=["x"])],
        )
        with self.assertRaises(ValueError):
            layout.validate()

    def test_duplicate_function_names_rejected(self):
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="dup",
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
            slots=[
                StateSlot(name="x", kind=SLOT_KIND_INPUT, dtype="f32",
                          shape=[256], offset=0, size_bytes=1024),
                StateSlot(name="y", kind=SLOT_KIND_OUTPUT, dtype="f32",
                          shape=[256], offset=1024, size_bytes=1024),
            ],
            functions=[
                FunctionSpec(name="main", role=ROLE_MATMUL,
                             input_slots=["x"], output_slots=["y"]),
                FunctionSpec(name="main", role=ROLE_MATMUL,
                             input_slots=["x"], output_slots=["y"]),
            ],
        )
        with self.assertRaises(ValueError):
            layout.validate()

    def test_function_referencing_unknown_slot_rejected(self):
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="bad",
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
            slots=[
                StateSlot(name="x", kind=SLOT_KIND_INPUT, dtype="f32",
                          shape=[256], offset=0, size_bytes=1024),
            ],
            functions=[FunctionSpec(name="main", role=ROLE_MATMUL,
                                    input_slots=["x"], output_slots=["ghost"])],
        )
        with self.assertRaises(ValueError):
            layout.validate()

    def test_dead_state_slot_rejected(self):
        # A STATE slot that no function references is dead code.
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="dead",
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
            slots=[
                StateSlot(name="x", kind=SLOT_KIND_INPUT, dtype="f32",
                          shape=[256], offset=0, size_bytes=1024),
                StateSlot(name="y", kind=SLOT_KIND_OUTPUT, dtype="f32",
                          shape=[256], offset=1024, size_bytes=1024),
                StateSlot(name="orphan", kind=SLOT_KIND_STATE, dtype="f32",
                          shape=[256], offset=2048, size_bytes=1024),
            ],
            functions=[FunctionSpec(name="main", role=ROLE_MATMUL,
                                    input_slots=["x"], output_slots=["y"])],
        )
        with self.assertRaises(ValueError):
            layout.validate()

    def test_state_size_below_minimum_rejected(self):
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="small",
            state_size_bytes=1024,  # below 64KB
            slots=[],
            functions=[],
        )
        with self.assertRaises(ValueError):
            layout.validate()

    def test_state_size_misaligned_rejected(self):
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="misaligned",
            state_size_bytes=ANE_MIN_ALLOC_BYTES + 1,  # not 16KB-aligned
            slots=[],
            functions=[],
        )
        with self.assertRaises(ValueError):
            layout.validate()

    def test_unsupported_version_rejected(self):
        layout = StateLayout(
            version=99, bundle_name="future",
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
        )
        with self.assertRaises(ValueError):
            layout.validate()


class SerializationTest(unittest.TestCase):
    def test_roundtrip_preserves_fields(self):
        original = StateLayout.for_w0_matmul("w0-256x256", n=256)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            original.write_json(Path(f.name))
            path = Path(f.name)
        try:
            restored = StateLayout.read_json(path)
            self.assertEqual(restored.bundle_name, original.bundle_name)
            self.assertEqual(restored.state_size_bytes, original.state_size_bytes)
            self.assertEqual(len(restored.slots), len(original.slots))
            self.assertEqual(len(restored.functions), len(original.functions))
            for orig_slot, restored_slot in zip(original.slots, restored.slots):
                self.assertEqual(orig_slot.name, restored_slot.name)
                self.assertEqual(orig_slot.kind, restored_slot.kind)
                self.assertEqual(orig_slot.dtype, restored_slot.dtype)
                self.assertEqual(orig_slot.shape, restored_slot.shape)
                self.assertEqual(orig_slot.offset, restored_slot.offset)
                self.assertEqual(orig_slot.size_bytes, restored_slot.size_bytes)
        finally:
            path.unlink()

    def test_json_is_human_readable(self):
        """The JSON should be readable (not binary) and self-describing."""
        layout = StateLayout.for_w0_matmul("w0-256x256", n=256)
        d = layout.to_dict()
        text = json.dumps(d, indent=2)
        self.assertIn('"version"', text)
        self.assertIn('"bundle_name"', text)
        self.assertIn('"slots"', text)
        self.assertIn('"functions"', text)
        self.assertIn('"dependencies"', text)


class MultifunctionLayoutTest(unittest.TestCase):
    """Smoke tests for the multifunction case (prefill + mtp).

    These don't run on real data; they validate that the manifest
    structure supports the cross-function dependency graph that the
    E-core pump needs to coordinate.
    """

    def test_prefill_then_mtp_dependency(self):
        # Simulate: prefill_s32 produces hidden_states, mtp_predict
        # reads it. The dependency is the cross-function edge.
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="gemma4-prefill-mtp",
            state_size_bytes=8 * 1024 * 1024,  # 8 MB
            slots=[
                StateSlot(name="token_ids", kind=SLOT_KIND_INPUT, dtype="i32",
                          shape=[1, 32], offset=0, size_bytes=2048),
                StateSlot(name="positions", kind=SLOT_KIND_INPUT, dtype="i32",
                          shape=[1, 32], offset=2048, size_bytes=2048),
                StateSlot(name="hidden_states", kind=SLOT_KIND_STATE,
                          dtype="f16", shape=[1, 32, 3072],
                          offset=16 * 1024, size_bytes=192 * 1024),
                StateSlot(name="k_cache", kind=SLOT_KIND_STATE, dtype="f16",
                          shape=[1, 8, 128, 32],
                          offset=256 * 1024, size_bytes=64 * 1024),
                StateSlot(name="v_cache", kind=SLOT_KIND_STATE, dtype="f16",
                          shape=[1, 8, 128, 32],
                          offset=320 * 1024, size_bytes=64 * 1024),
                StateSlot(name="h_nextn", kind=SLOT_KIND_INPUT, dtype="f32",
                          shape=[1, 3072], offset=384 * 1024, size_bytes=12 * 1024),
                StateSlot(name="top_token", kind=SLOT_KIND_OUTPUT, dtype="i32",
                          shape=[1], offset=396 * 1024, size_bytes=16),
                StateSlot(name="confidence", kind=SLOT_KIND_OUTPUT, dtype="f32",
                          shape=[1], offset=396 * 1024 + 16, size_bytes=16),
                StateSlot(name="next_hidden", kind=SLOT_KIND_OUTPUT, dtype="f32",
                          shape=[1, 3072], offset=400 * 1024, size_bytes=12 * 1024),
            ],
            functions=[
                FunctionSpec(name="prefill_s32", role=ROLE_PREFILL, bucket=32,
                             stateful=True,
                             input_slots=["token_ids", "positions"],
                             output_slots=["hidden_states", "k_cache", "v_cache"],
                             core_ml_function_name="prefill_s32",
                             use_ane=True),
                FunctionSpec(name="mtp_predict", role=ROLE_MTP, bucket=1,
                             stateful=True,
                             input_slots=["token_ids", "positions",
                                          "h_nextn", "k_cache", "v_cache"],
                             output_slots=["top_token", "confidence",
                                           "next_hidden"],
                             core_ml_function_name="mtp_predict",
                             use_ane=True),
            ],
            dependencies=[
                Dependency(producer="prefill_s32", slot="hidden_states",
                            consumers=["mtp_predict"]),
                Dependency(producer="prefill_s32", slot="k_cache",
                            consumers=["mtp_predict"]),
                Dependency(producer="prefill_s32", slot="v_cache",
                            consumers=["mtp_predict"]),
            ],
        )
        layout.validate()
        # The runtime's view: mtp_predict's input_slots include
        # k_cache and v_cache which are STATE slots produced by
        # prefill_s32. The pump builds the wait-on-prefill edge
        # from the dependencies list.
        self.assertEqual(len(layout.dependencies), 3)

    def test_sync_reset_are_cpu_only(self):
        # sync and reset are CPU-side mem{cpy,set} on the E-core
        # pump. sync reads K/V from input slots and writes to the
        # k_cache/v_cache STATE slots; reset clears the k_cache.
        # Both appear in the manifest as functions with
        # use_ane=False but the runtime still tracks them.
        layout = StateLayout(
            version=SCHEMA_VERSION, bundle_name="with-sync-reset",
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
            slots=[
                StateSlot(name="base_keys", kind=SLOT_KIND_INPUT, dtype="f16",
                          shape=[128], offset=0, size_bytes=256),
                StateSlot(name="k_cache", kind=SLOT_KIND_STATE, dtype="f16",
                          shape=[128], offset=16 * 1024, size_bytes=256),
            ],
            functions=[
                FunctionSpec(name="sync", role=ROLE_SYNC, use_ane=False,
                             input_slots=["base_keys"],
                             output_slots=["k_cache"]),
                FunctionSpec(name="reset", role=ROLE_RESET, use_ane=False,
                             output_slots=["k_cache"]),
            ],
            dependencies=[],
        )
        layout.validate()
        sync = next(f for f in layout.functions if f.role == ROLE_SYNC)
        reset = next(f for f in layout.functions if f.role == ROLE_RESET)
        self.assertFalse(sync.use_ane)
        self.assertFalse(reset.use_ane)


class ManifestPathTest(unittest.TestCase):
    def test_path_convention(self):
        d = Path("/tmp/ane-fixtures")
        self.assertEqual(
            manifest_path_for(d, "w0-256x256"),
            Path("/tmp/ane-fixtures/w0-256x256.ane_state.v1.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
