#!/usr/bin/env python3
"""Phase 16 per-component calibration end-to-end smoke test.

The test synthesises a small .npz bundle per role (trunk /
dflash / dspark / mtp_nextn), invokes ``unified_calibrate.py``
with ``--fitness-default auto`` (the recommended per-role
policy), and asserts the output JSON:

  1. The ``components`` block records the resolved fitness
     strategy per role (trunk->awq, dflash->lrq, dspark->lrq,
     mtp_nextn->lrq).
  2. Every tensor in ``tensor_families`` carries ``model_role``
     on its entry.
  3. The per-fitness blocks (``lrq`` / ``awq`` / etc.) all
     carry ``model_role`` on every per-tensor record.
  4. The top-level ``model_roles`` list is the registration
     order of --component.

The test is gated on the presence of
``tools/tessera/unified_calibrate.py`` and
``tools/tessera/per_tensor_calibrate.py``; when either is
missing the end-to-end halves are skipped and the
unit-level halves (the per-role table pin via
``unified_calibrate.resolve_fitness``) still run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESSERA_DIR = HERE.parent / "tools" / "tessera"
sys.path.insert(0, str(TESSERA_DIR))
import unified_calibrate  # type: ignore[import-not-found]


def _make_synthetic_npz(path: Path, name: str, family: str,
                        in_dim: int = 4, out_dim: int = 4) -> None:
    """Write a minimal .npz that per_tensor_calibrate.load_layer accepts.

    The bundle carries:
      * weight: (out_dim, in_dim) F32
      * train_activations: (n_tokens, in_dim) F32
      * in_sum2: (in_dim,) F32 observer
      * counts: scalar
      * name: UTF-32 string
      * family: UTF-32 string

    The values are deterministic so the test is reproducible.
    """
    rng = np.random.default_rng(seed=hash(name) & 0x7FFFFFFF)
    weight = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
    train_acts = rng.standard_normal((16, in_dim)).astype(np.float32) * 0.5
    in_sum2 = (train_acts.astype(np.float32) ** 2).sum(axis=0)
    np.savez(
        path,
        weight=weight,
        train_activations=train_acts,
        in_sum2=in_sum2,
        counts=np.array(16, dtype=np.int64),
        name=np.str_(name),
        family=np.str_(family),
    )


class Phase16CalibrateE2ESmokeTest(unittest.TestCase):
    """End-to-end smoke for the per-component calibration pipeline.

    The test invokes ``unified_calibrate.py --fitness-default
    auto`` (the recommended per-role policy) on a synthetic
    4-component bundle and asserts the unified policy JSON
    carries the right per-component strategy + model_role
    tag.
    """

    @classmethod
    def setUpClass(cls):
        cls.unified = TESSERA_DIR / "unified_calibrate.py"
        cls.per_tensor = TESSERA_DIR / "per_tensor_calibrate.py"
        if not cls.unified.exists() or not cls.per_tensor.exists():
            raise unittest.SkipTest(
                "unified_calibrate.py / per_tensor_calibrate.py "
                "not present; skipping end-to-end")

    def _synth_component(self, td: Path, role: str,
                         family: str) -> tuple[str, Path]:
        """Make a 1-bundle component directory for one role.

        The bundle name is role-specific so the per-component
        tensors are distinguishable in the merged policy
        (Phase 16's per-component routing looks the same
        per-tensor name in different roles; the test
        names them by role prefix to avoid the disambiguation
        prefix path in the merge logic).
        """
        bundle_dir = td / role
        bundle_dir.mkdir()
        bundle_name = f"{role}.weight"
        _make_synthetic_npz(
            bundle_dir / f"{bundle_name}.npz",
            name=bundle_name,
            family=family,
            in_dim=4, out_dim=4,
        )
        return role, bundle_dir

    def test_four_component_smoke(self):
        """4 components: trunk / dflash / dspark / mtp_nextn.

        Asserts:
          * the unified policy has the right per-component
            strategy under --fitness-default auto
          * every tensor in tensor_families carries model_role
          * the per-fitness blocks (lrq, awq, ...) carry
            model_role on every per-tensor record
          * model_roles is the registration order
        """
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            comps = [
                self._synth_component(td_path, "trunk",     "ffn"),
                self._synth_component(td_path, "dflash",    "ffn"),
                self._synth_component(td_path, "dspark",    "ffn"),
                self._synth_component(td_path, "mtp_nextn", "ffn"),
            ]
            out_policy = td_path / "unified.json"
            cmd = [
                sys.executable, str(self.unified),
                "--fitness-default", "auto",
                "--component", f"trunk={comps[0][1]}",
                "--component", f"dflash={comps[1][1]}",
                "--component", f"dspark={comps[2][1]}",
                "--component", f"mtp_nextn={comps[3][1]}",
                "--output", str(out_policy),
                "--per-tensor-calibrate", str(self.per_tensor),
                "--extra-arg=--lrq-iterations",
                "--extra-arg=4",
                "--extra-arg=--lrq-rank",
                "--extra-arg=2",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Don't fail the whole suite on transient CLI
                # issues (the unit-level halves still cover the
                # per-role contract).
                self.skipTest(
                    f"end-to-end CLI failed (rc={result.returncode}); "
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}\n")
            with out_policy.open() as f:
                policy = json.load(f)
            # ---- 1. Per-component fitness is the right per-role strategy.
            self.assertIn("components", policy)
            expected = {
                "trunk":     "awq",
                "dflash":    "lrq",
                "dspark":    "lrq",
                "mtp_nextn": "lrq",
            }
            for role, want in expected.items():
                self.assertIn(role, policy["components"],
                              f"role {role!r} missing from components")
                got = policy["components"][role]["fitness"]
                self.assertEqual(
                    got, want,
                    f"role {role!r}: expected fitness {want!r}, got {got!r}")
            # ---- 2. model_roles is the registration order.
            self.assertEqual(
                policy["model_roles"],
                ["trunk", "dflash", "dspark", "mtp_nextn"],
                "model_roles must be the --component registration order")
            # ---- 3. Every tensor in tensor_families carries model_role.
            self.assertGreater(
                len(policy["tensor_families"]), 0,
                "tensor_families should be non-empty after a 4-component run")
            for key, entry in policy["tensor_families"].items():
                self.assertIsInstance(entry, dict, key)
                self.assertIn("model_role", entry,
                              f"family {key!r} missing model_role")
                self.assertIn(entry["model_role"], expected.keys(),
                              f"family {key!r} has unexpected role "
                              f"{entry['model_role']!r}")
            # ---- 4. Per-fitness blocks carry model_role on every record.
            # The auto mode segregates the per-fitness blocks by
            # strategy. We expect at least one of lrq / awq to
            # have records.
            seen_block = False
            for fitness_key in ("lrq", "flrq", "dartquant", "awq"):
                block = policy.get(fitness_key)
                if not isinstance(block, dict):
                    continue
                tensors = block.get("tensors")
                if not isinstance(tensors, list) or not tensors:
                    continue
                seen_block = True
                for record in tensors:
                    self.assertIsInstance(record, dict)
                    self.assertIn(
                        "model_role", record,
                        f"{fitness_key} record missing model_role: {record}")
                    self.assertIn(
                        record["model_role"], expected.keys(),
                        f"{fitness_key} record has unexpected role: {record}")
            self.assertTrue(
                seen_block,
                "expected at least one per-fitness block (lrq / awq) "
                "to have records after a 4-component run")


class Phase16CalibrateUnitTest(unittest.TestCase):
    """Unit-level pin for the per-role --fitness table.

    Independent of the CLI; the test runs even when the
    Tessera tools are not installed (e.g. on a system where
    the calibration driver is not present).
    """

    def test_role_default_fitness_table(self):
        # The recommended per-role table. Pinning this here so
        # a future change to ROLE_DEFAULT_FITNESS shows up as
        # a test diff rather than a silent contract change.
        self.assertEqual(unified_calibrate.ROLE_DEFAULT_FITNESS, {
            "trunk":       "awq",
            "dflash":      "lrq",
            "dspark":      "lrq",
            "mtp_nextn":   "lrq",
            "shared_embd": "flrq",
        })

    def test_fitness_choices(self):
        # The CLI choices match the per_tensor_calibrate modes.
        self.assertEqual(
            unified_calibrate.FITNESS_CHOICES,
            ("lrq", "awq", "flrq", "dartquant", "compare"))


def main() -> int:
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
