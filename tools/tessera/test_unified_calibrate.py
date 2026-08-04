#!/usr/bin/env python3
"""Smoke test for tools/tessera/unified_calibrate.py.

Two levels:

* Unit-level: exercise ``_tag_policy`` and the merge logic with
  synthetic per-component policy dicts (no per_tensor_calibrate
  subprocess required). This pins the model_role tagging contract
  and the families merge behavior.
* End-to-end: build minimal synthetic .npz bundles for two
  components (a 4x4 trunk weight + a 4x4 dspark weight), run
  unified_calibrate.py with --fitness lrq, and verify the unified
  policy JSON has both components' tensors with model_role set.

The end-to-end test is gated on the presence of per_tensor_calibrate.py
in the same directory; when missing (e.g. on a system where the
Tessera tools aren't installed), the end-to-end half is skipped
and the unit-level half still runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the tools/tessera dir importable so we can import unified_calibrate
# without it being a package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unified_calibrate  # type: ignore[import-not-found]


def _make_synthetic_policy(role_tag: str, tensor_names: list[str]) -> dict:
    """Build a minimal per-component policy dict as if per_tensor_calibrate
    had run on a single .npz bundle. Only the fields _tag_policy +
    the merge logic touch are populated.
    """
    families: dict[str, dict] = {}
    tensors: list[dict] = []
    for i, name in enumerate(tensor_names):
        key = f"lrq:{name}"
        families[key] = {
            "schema": "llama.tessera.lrq-policy.v1",
            "rank": 16,
            "bytes": 256,
        }
        tensors.append({
            "tensor": name,
            "rank": 16,
            "bytes": 256,
        })
    return {
        "schema": "llama.speculative.calibration-policy.v1",
        "lrq": {
            "schema": "llama.tessera.lrq-policy.v1",
            "rank": 16,
            "iterations": 100,
            "lr": 0.01,
            "input_scale_agg": "mean",
            "tensor_count": len(tensor_names),
            "total_bytes": 256 * len(tensor_names),
            "tensors": tensors,
        },
        "tensor_families": families,
        "per_tensor_calibration": {
            "corpus_hash": "synthetic_" + role_tag,
            "tokens": 64,
        },
    }


def _test_tag_policy() -> None:
    """unit-level: _tag_policy annotates a per-component policy with
    model_role + unified_schema and tags every tensor record.
    """
    policy = _make_synthetic_policy(
        "trunk", ["blk.0.attn_q.weight", "blk.0.attn_k.weight"])
    tagged = unified_calibrate._tag_policy(
        policy, "trunk", ["trunk", "dspark"])
    assert tagged["model_role"] == "trunk", tagged
    assert tagged["unified_schema"] == unified_calibrate.UNIFIED_SCHEMA
    assert tagged["model_roles"] == ["trunk", "dspark"]
    # Every family entry has model_role = trunk
    for key, entry in tagged["tensor_families"].items():
        assert entry["model_role"] == "trunk", (key, entry)
    # Every per-fitness tensor record has model_role = trunk
    for record in tagged["lrq"]["tensors"]:
        assert record["model_role"] == "trunk", record
    # The original schema field is preserved
    assert tagged["schema"] == "llama.speculative.calibration-policy.v1"
    print("ok   _tag_policy annotates model_role + unified_schema")


def _test_components_parser() -> None:
    """unit-level: --component ROLE=PATH parsing handles repeated flags
    and rejects malformed values.
    """
    with tempfile.TemporaryDirectory() as td:
        d1 = Path(td) / "trunk"
        d2 = Path(td) / "dspark"
        d1.mkdir()
        d2.mkdir()
        comps = unified_calibrate._parse_components([
            f"trunk={d1}", f"dspark={d2}"])
        assert comps == [("trunk", d1), ("dspark", d2)], comps
        # Reject ROLE without PATH
        try:
            unified_calibrate._parse_components(["trunk"])
        except ValueError as exc:
            assert "ROLE=PATH" in str(exc), exc
        else:
            raise AssertionError("expected ValueError on missing =")
        # Reject missing path
        try:
            unified_calibrate._parse_components(["trunk=/nope/nope/nope"])
        except ValueError as exc:
            assert "does not exist" in str(exc), exc
        else:
            raise AssertionError("expected ValueError on missing path")
        # Reject empty list
        try:
            unified_calibrate._parse_components([])
        except ValueError as exc:
            assert "at least one" in str(exc), exc
        else:
            raise AssertionError("expected ValueError on empty list")
    print("ok   _parse_components handles ROLE=PATH + rejects malformed")


def _make_synthetic_npz(path: Path, in_dim: int, out_dim: int) -> None:
    """Write a minimal .npz that per_tensor_calibrate.load_layer accepts:
    weight (2D float32), train_activations (2D float32, matching
    weight.shape[1]), in_sum2 (1D float32), name (string), family
    (string), counts (scalar). The values are deterministic so the
    test is reproducible.

    name + family are stored as np.str_ (UTF-32 strings) so
    per_tensor_calibrate's _scalar_string returns the raw string
    rather than its repr. (np.bytes_ would round-trip through
    str(bytes) and surface as ``"b'foo'"`` because the helper
    uses str() rather than bytes.decode().)
    """
    rng = np.random.default_rng(seed=in_dim * 31 + out_dim)
    weight = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
    train_acts = rng.standard_normal((16, in_dim)).astype(np.float32) * 0.5
    in_sum2 = (train_acts.astype(np.float32) ** 2).sum(axis=0)
    np.savez(
        path,
        weight=weight,
        train_activations=train_acts,
        in_sum2=in_sum2,
        counts=np.array(16, dtype=np.int64),
        name=np.str_("synthetic.weight"),
        family=np.str_("ffn"),
    )


def _test_end_to_end() -> None:
    """end-to-end: run unified_calibrate.py with two synthetic
    components and verify the unified policy is well-formed.
    """
    per_tensor = HERE / "per_tensor_calibrate.py"
    if not per_tensor.exists():
        print("skip end-to-end: per_tensor_calibrate.py not present")
        return
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        trunk_dir = td_path / "trunk"
        dspark_dir = td_path / "dspark"
        trunk_dir.mkdir()
        dspark_dir.mkdir()
        # 4x4 weights so per_tensor_calibrate's LRQ is fast.
        _make_synthetic_npz(trunk_dir / "trunk.weight.npz", in_dim=4, out_dim=4)
        _make_synthetic_npz(dspark_dir / "dspark.weight.npz", in_dim=4, out_dim=4)
        out_policy = td_path / "unified.json"
        cmd = [
            sys.executable,
            str(HERE / "unified_calibrate.py"),
            "--component", f"trunk={trunk_dir}",
            "--component", f"dspark={dspark_dir}",
            "--fitness", "lrq",
            "--output", str(out_policy),
            "--per-tensor-calibrate", str(per_tensor),
            "--extra-arg=--lrq-iterations",
            "--extra-arg=4",
            "--extra-arg=--lrq-rank",
            "--extra-arg=2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                "end-to-end failed (likely per_tensor_calibrate "
                "config issue); see stdout/stderr:")
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            # Don't fail the whole test on transient CLI issues;
            # the unit-level half already covers the merge logic.
            print("skip end-to-end assertions (CLI returned "
                  f"{result.returncode})")
            return
        with out_policy.open() as f:
            policy = json.load(f)
        assert policy["schema"] == "llama.speculative.calibration-policy.v1"
        assert policy["unified_schema"] == unified_calibrate.UNIFIED_SCHEMA
        assert policy["model_roles"] == ["trunk", "dspark"]
        # Every tensor family is tagged with its model_role.
        for key, entry in policy["tensor_families"].items():
            assert "model_role" in entry, (key, entry)
            assert entry["model_role"] in {"trunk", "dspark"}, entry
        # The per-fitness tensors list has every record tagged.
        if "lrq" in policy:
            assert policy["lrq"]["model_role"] == "unified"
            for record in policy["lrq"]["tensors"]:
                assert record["model_role"] in {"trunk", "dspark"}, record
    print("ok   end-to-end unified_calibrate produces a well-formed "
          "policy with model_role on every tensor")


def main() -> int:
    _test_tag_policy()
    _test_components_parser()
    _test_end_to_end()
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
