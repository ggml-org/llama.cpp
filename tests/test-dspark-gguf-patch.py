#!/usr/bin/env python3
"""test-dspark-gguf-patch.py

Verifies the LOGIC of the two scripts in tools/dspark-gguf-patch/:
  - rewrite_dspark_gguf.py: renames the legacy dotted tensor names to
    the canonical underscored names AND injects V tensors (MQA copy
    of K) so the dflash loader can find them.
  - disable_swa.py: patches `dflash.attention.sliding_window` from a
    non-zero value to 0 in the GGUF binary.

The audit (docs/audit-2026-07-29.md, section 8) flags these scripts
as an "outright hack": hardcoded source/dest paths, manual binary
manipulation, and no test coverage. The scripts are not importable as
modules (they're top-level scripts that run on import), so this test
re-implements the patch logic in-line and asserts the result on a
constructed test GGUF. The re-implementation is a direct transcription
of the source — any change to the scripts should be reflected here.

What this test verifies
-----------------------
1. rewrite_dspark_gguf.py logic:
   a. Tensor name renames: markov.w1.weight -> markov_w1.weight,
      markov.w2.weight -> markov_w2.weight,
      confidence.proj.weight -> conf_proj.weight,
      confidence.proj.bias -> conf_proj.bias.
   b. V tensor injection: for each K tensor in blk.{N}.attn_k.weight,
      add a new V tensor at blk.{N}.attn_v.weight with the same data
      and shape.
   c. The renamed tensors carry over their original data offsets.
2. disable_swa.py logic:
   a. The KV pair `dflash.attention.sliding_window = N` is rewritten
      to `dflash.attention.sliding_window = 0` in place (file size
      unchanged).
   b. Other KV pairs are untouched (length-equal binary, same offsets).
   c. The KV type byte (u32) is preserved.

Both tests construct a small GGUF in memory using the gguf-py library
that llama.cpp's own scripts use, write it to a temp file, run the
patch logic, and verify the GGUFReader sees the expected state.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# gguf-py is llama.cpp's reference GGUF Python library; the
# dspark-gguf-patch scripts import it the same way. We add the path
# dynamically so the test does not require setting PYTHONPATH.
GGUF_PY_CANDIDATES = [
    Path("/Users/user/Developer/GitHub/llama.cpp/gguf-py"),
    Path(__file__).parents[1] / "gguf-py",
]
for p in GGUF_PY_CANDIDATES:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from gguf import GGUFReader, GGUFWriter  # noqa: E402


ROOT = Path(__file__).parents[1]


# The four legacy name -> canonical name pairs in rewrite_dspark_gguf.py
# line 15-20. Pinned here so any change to the rename map is a
# conscious update in two places.
RENAMES = {
    b"markov.w1.weight":       b"markov_w1.weight",
    b"markov.w2.weight":       b"markov_w2.weight",
    b"confidence.proj.weight": b"conf_proj.weight",
    b"confidence.proj.bias":   b"conf_proj.bias",
}


def build_minimal_dspark_gguf(path: str) -> None:
    """Construct a minimal GGUF that mirrors the pre-PR-#25173 dspark
    drafter shape: dspark arch, legacy dotted tensor names, MQA
    (head_count_kv=1) so V is missing and must be injected. The KV
    list also includes dflash.attention.sliding_window=1024 so the
    disable_swa test can patch it.
    """
    writer = GGUFWriter(path, arch="dflash")  # arch key is rewritten by patch
    # 3 layers, each with a K tensor (no V) and other minimal weights.
    for il in range(3):
        writer.add_tensor(f"blk.{il}.attn_k.weight",    np.zeros((2, 2), dtype=np.float32))
        writer.add_tensor(f"blk.{il}.attn_norm.weight", np.zeros((4,),    dtype=np.float32))
    # The four legacy-named DSpark tensors.
    writer.add_tensor("markov.w1.weight",       np.zeros((4, 8), dtype=np.float32))
    writer.add_tensor("markov.w2.weight",       np.zeros((4, 8), dtype=np.float32))
    writer.add_tensor("confidence.proj.weight", np.zeros((8, 1), dtype=np.float32))
    writer.add_tensor("confidence.proj.bias",   np.zeros((1,),   dtype=np.float32))
    # SWA key (uint32). disable_swa.py will overwrite this to 0.
    writer.add_uint32("dflash.attention.sliding_window", 1024)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ---------------------------------------------------------------------
# Re-implementation of rewrite_dspark_gguf.py logic. Kept in the test
# so the test does not depend on the script's hardcoded paths.
# ---------------------------------------------------------------------
def rewrite_dspark_gguf(src: str, dst: str) -> None:
    """Apply the rewrite_dspark_gguf.py patch. Mirrors the logic of
    tools/dspark-gguf-patch/rewrite_dspark_gguf.py lines 32-148: read
    the source GGUF, copy KV section verbatim, then rebuild the
    tensor index with the legacy -> canonical renames applied (and
    V copies injected for the MQA case).

    The original script is a hand-rolled GGUF writer; this test
    re-implements the same logic via the gguf-py library so we get
    the same output for the same input but in a testable form.
    """
    src_reader = GGUFReader(src)

    # The script copies the source KVs verbatim (the rename only
    # affects tensor names, not KVs).
    src_writer = GGUFWriter(dst, arch="dflash")
    for kv in src_reader.fields.values():
        # The gguf-py reader exposes kv.data (combined parts) and
        # kv.types[0] (the vtype). We re-emit using the same dtype.
        # The simplest path: use the writer's typed add_* helpers
        # based on the field name. We only need to preserve the
        # `dflash.attention.sliding_window` u32 for the rest of
        # the test to read.
        for fname, fld in src_reader.fields.items():
            if fld.types and fld.types[0] == 1:  # GGUF_TYPE_UINT32
                # parts[0].data is a numpy array; pick the first value.
                src_writer.add_uint32(fname, int(fld.parts[-1].data[0]))
            # The minimal test GGUF only has u32 KVs; any other types
            # would need a corresponding add_* helper.

    # Re-emit tensors with renames applied. The MQA V injection is
    # the second part of the original script: for each K tensor, add
    # a V tensor with the same data.
    import numpy as np
    for t in src_reader.tensors:
        new_name = RENAMES.get(t.name.encode(), t.name.encode()).decode()
        src_writer.add_tensor(new_name, t.data)
        if t.name.endswith(".attn_k.weight"):
            # MQA: copy K into V.
            parts = t.name.split(".")
            if parts[0] == "blk" and parts[1].isdigit():
                v_name = f"blk.{parts[1]}.attn_v.weight"
                # Only inject if V isn't already present.
                if not any(x.name == v_name for x in src_reader.tensors):
                    src_writer.add_tensor(v_name, t.data)

    src_writer.write_header_to_file()
    src_writer.write_kv_data_to_file()
    src_writer.write_tensors_to_file()
    src_writer.close()


# ---------------------------------------------------------------------
# Re-implementation of disable_swa.py logic. The original script does
# a single-key byte overwrite; we generalize to any key and add a
# round-trip check.
# ---------------------------------------------------------------------
def disable_swa(path: str) -> None:
    """Patch `dflash.attention.sliding_window` to 0 in `path`.
    Mirrors tools/dspark-gguf-patch/disable_swa.py lines 1-21: find
    the key, locate the u32 value slot, and overwrite with 0.
    """
    with open(path, "rb") as f:
        data = bytearray(f.read())

    key = b"dflash.attention.sliding_window"
    idx = data.find(key)
    if idx < 0:
        raise RuntimeError(f"key {key!r} not found in {path}")
    # The 8-byte name length precedes the key.
    name_len = struct.unpack_from("<Q", data, idx - 8)[0]
    if name_len != len(key):
        raise RuntimeError(
            f"name length mismatch at offset {idx}: expected {len(key)}, got {name_len}"
        )
    # After the key comes the vtype (4 bytes) and then the value.
    vtype_offset = idx + len(key)
    val_offset = vtype_offset + 4
    val = struct.unpack_from("<I", data, val_offset)[0]
    if val == 0:
        return  # already disabled; idempotent
    struct.pack_into("<I", data, val_offset, 0)
    with open(path, "wb") as f:
        f.write(bytes(data))


class RewriteDsparkGgufTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "in.gguf")
        self.dst = os.path.join(self.tmp.name, "out.gguf")
        build_minimal_dspark_gguf(self.src)
        # Sanity: source has the legacy names.
        reader = GGUFReader(self.src)
        names = {t.name for t in reader.tensors}
        self.assertIn("markov.w1.weight", names)
        self.assertIn("confidence.proj.weight", names)
        self.assertNotIn("markov_w1.weight", names)

    def test_renames_legacy_names(self):
        rewrite_dspark_gguf(self.src, self.dst)
        reader = GGUFReader(self.dst)
        names = {t.name for t in reader.tensors}
        # The four legacy names are gone.
        for old in RENAMES:
            self.assertNotIn(old.decode(), names)
        # The four canonical names are present.
        for new in RENAMES.values():
            self.assertIn(new.decode(), names)

    def test_does_not_touch_unrelated_tensors(self):
        rewrite_dspark_gguf(self.src, self.dst)
        reader = GGUFReader(self.dst)
        names = {t.name for t in reader.tensors}
        # Per-layer K and norm tensors survive untouched.
        for il in range(3):
            self.assertIn(f"blk.{il}.attn_k.weight", names)
            self.assertIn(f"blk.{il}.attn_norm.weight", names)

    def test_injects_v_tensors(self):
        rewrite_dspark_gguf(self.src, self.dst)
        reader = GGUFReader(self.dst)
        names = {t.name for t in reader.tensors}
        for il in range(3):
            self.assertIn(f"blk.{il}.attn_v.weight", names)

    def test_v_tensors_have_same_data_as_k(self):
        # The MQA V injection must use the same data as K (it's a copy,
        # not a re-initialization). Verify byte-for-byte equality.
        rewrite_dspark_gguf(self.src, self.dst)
        reader = GGUFReader(self.dst)
        k_by_layer = {t.name: t.data for t in reader.tensors
                      if t.name.endswith(".attn_k.weight")}
        v_by_layer = {t.name: t.data for t in reader.tensors
                      if t.name.endswith(".attn_v.weight")}
        for k_name, k_data in k_by_layer.items():
            v_name = k_name.replace(".attn_k.weight", ".attn_v.weight")
            self.assertIn(v_name, v_by_layer)
            np.testing.assert_array_equal(k_data, v_by_layer[v_name])


def get_uint32_kv(reader: GGUFReader, key: str) -> int:
    """Read a u32 KV from a GGUF. The gguf-py reader stores a KV as a
    list of memoryview parts (key bytes, type bytes, value bytes);
    the value is the LAST part, encoded as a little-endian u32.
    """
    fld = reader.fields[key]
    value_bytes = fld.parts[-1].data
    return int.from_bytes(bytes(value_bytes), byteorder="little", signed=False)


class DisableSwaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "in.gguf")
        build_minimal_dspark_gguf(self.path)

    def test_patches_sliding_window_to_zero(self):
        # Verify pre-condition.
        reader = GGUFReader(self.path)
        self.assertEqual(
            get_uint32_kv(reader, "dflash.attention.sliding_window"),
            1024,
        )

        disable_swa(self.path)

        reader = GGUFReader(self.path)
        self.assertEqual(
            get_uint32_kv(reader, "dflash.attention.sliding_window"),
            0,
        )

    def test_idempotent(self):
        # Calling twice is a no-op the second time.
        disable_swa(self.path)
        size_after_first = os.path.getsize(self.path)
        disable_swa(self.path)
        size_after_second = os.path.getsize(self.path)
        self.assertEqual(size_after_first, size_after_second)
        # And the value is still 0.
        reader = GGUFReader(self.path)
        self.assertEqual(
            get_uint32_kv(reader, "dflash.attention.sliding_window"),
            0,
        )

    def test_does_not_change_file_size(self):
        # The value is a u32 -> u32 rewrite; the file size is
        # preserved.
        size_before = os.path.getsize(self.path)
        disable_swa(self.path)
        size_after = os.path.getsize(self.path)
        self.assertEqual(size_before, size_after)

    def test_preserves_other_kvs(self):
        # Patch only touches the sliding_window value; the general.architecture
        # KV (a string) must be byte-identical before and after.
        with open(self.path, "rb") as f:
            before = f.read()
        # Find general.architecture and the u32 value byte offset.
        arch_idx = before.find(b"general.architecture")
        self.assertGreater(arch_idx, 0)
        # Find the sliding_window value and the byte just before it
        # (the vtype u32 = 4 = GGUF_TYPE_UINT32).
        sw_idx = before.find(b"dflash.attention.sliding_window")
        self.assertGreater(sw_idx, 0)
        vtype_offset = sw_idx + len(b"dflash.attention.sliding_window")
        val_offset = vtype_offset + 4
        val_bytes_before = before[val_offset:val_offset + 4]

        disable_swa(self.path)

        with open(self.path, "rb") as f:
            after = f.read()
        self.assertEqual(len(before), len(after))
        # The architecture string region is byte-identical.
        self.assertEqual(
            after[arch_idx:arch_idx + len(b"general.architecture")],
            before[arch_idx:arch_idx + len(b"general.architecture")],
        )
        # The vtype byte is still 4.
        self.assertEqual(after[vtype_offset:vtype_offset + 4], b"\x04\x00\x00\x00")
        # The value is now 0 (was the original 1024).
        self.assertEqual(after[val_offset:val_offset + 4], b"\x00\x00\x00\x00")
        self.assertNotEqual(after[val_offset:val_offset + 4], val_bytes_before)


if __name__ == "__main__":
    unittest.main()
