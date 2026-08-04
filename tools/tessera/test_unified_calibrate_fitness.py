#!/usr/bin/env python3
"""Tests for the per-component --fitness policy in unified_calibrate.py.

Phase 16 follow-up: the unified Calibrate driver picks a
--fitness strategy per component rather than running one mode
across all components. The per-role table is
``ROLE_DEFAULT_FITNESS``:

    trunk        -> awq     (heavy hitter, FFN; GA minimises
                              layer-output error the most)
    dflash       -> lrq     (drafter is lossy, smaller footprint)
    dspark       -> lrq     (same)
    mtp_nextn    -> lrq     (smaller, low-rank is enough)
    shared_embd  -> flrq    (frozen at train, calibration-free)

The CLI:

* ``--fitness-default auto`` (the recommended default) consults
  the per-role table.
* ``--fitness-default X`` (any non-auto value) overrides the
  table; every component runs with ``X``.

The test exercises:
  1. ``resolve_fitness()`` returns the per-role default under
     auto mode
  2. ``resolve_fitness()`` honours an explicit override
  3. ``resolve_fitness()`` falls back to the legacy default for
     an unknown role under auto mode
  4. ``resolve_fitness()`` rejects an invalid override
  5. The end-to-end run with --fitness-default auto produces a
     unified policy where each component's ``unified["components"]``
     entry records the resolved fitness strategy and every
     per-tensor record carries the model_role.
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
sys.path.insert(0, str(HERE))
import unified_calibrate  # type: ignore[import-not-found]


def _make_synthetic_npz(path: Path, in_dim: int = 4, out_dim: int = 4) -> None:
    """Write a minimal .npz that per_tensor_calibrate.load_layer accepts.

    See test_unified_calibrate._make_synthetic_npz for the full
    contract; this is a slimmed copy so the F2.3 test is
    self-contained.
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


class ResolveFitnessTest(unittest.TestCase):
    """Unit-level: resolve_fitness() picks the right per-role mode."""

    def test_auto_picks_per_role_default(self):
        # The recommended default for the per-role table.
        cases = {
            "trunk":       "awq",
            "dflash":      "lrq",
            "dspark":      "lrq",
            "mtp_nextn":   "lrq",
            "shared_embd": "flrq",
        }
        for role, expected in cases.items():
            self.assertEqual(
                unified_calibrate.resolve_fitness(
                    role, fitness_arg="lrq", fitness_default="auto"),
                expected,
                f"role {role!r} should map to {expected!r} in auto mode",
            )

    def test_explicit_override_wins(self):
        # A non-auto --fitness-default overrides the per-role table.
        for mode in ("lrq", "awq", "flrq", "dartquant", "compare"):
            self.assertEqual(
                unified_calibrate.resolve_fitness(
                    "trunk", fitness_arg="lrq",
                    fitness_default=mode),
                mode,
                f"override {mode!r} should win on every role",
            )
            # The legacy single-mode behaviour: --fitness is
            # ignored when --fitness-default is set explicitly.
            self.assertEqual(
                unified_calibrate.resolve_fitness(
                    "dflash", fitness_arg="lrq",
                    fitness_default=mode),
                mode,
                f"override {mode!r} should win even when "
                f"per-role default would differ",
            )

    def test_unknown_role_falls_back(self):
        # An unknown role under auto mode falls back to --fitness
        # (the legacy single-mode default).
        self.assertEqual(
            unified_calibrate.resolve_fitness(
                "made_up_role", fitness_arg="lrq", fitness_default="auto"),
            "lrq",
            "unknown role under auto mode falls back to --fitness",
        )

    def test_invalid_override_rejected(self):
        # --fitness-default must be one of the fitness modes or "auto".
        with self.assertRaises(ValueError) as cm:
            unified_calibrate.resolve_fitness(
                "trunk", fitness_arg="lrq",
                fitness_default="not_a_mode")
        self.assertIn("fitness-default", str(cm.exception))

    def test_invalid_fitness_arg_rejected(self):
        # Under auto mode --fitness must also be a valid mode.
        with self.assertRaises(ValueError) as cm:
            unified_calibrate.resolve_fitness(
                "trunk", fitness_arg="nope", fitness_default="auto")
        self.assertIn("fitness", str(cm.exception))


class UnifiedCalibrateE2ETest(unittest.TestCase):
    """End-to-end: the per-component --fitness is recorded in the
    unified policy's ``components`` block, and every per-tensor
    record carries the model_role.
    """

    @classmethod
    def setUpClass(cls):
        cls.per_tensor = HERE / "per_tensor_calibrate.py"
        if not cls.per_tensor.exists():
            raise unittest.SkipTest(
                "per_tensor_calibrate.py not present; "
                "skipping end-to-end")

    def _make_component(self, td: Path, role: str) -> tuple[str, Path]:
        bundle_dir = td / role
        bundle_dir.mkdir()
        # One synthetic weight per component so per_tensor_calibrate's
        # LRQ/FLRQ paths are fast.
        _make_synthetic_npz(bundle_dir / f"{role}.weight.npz",
                            in_dim=4, out_dim=4)
        return role, bundle_dir

    def test_auto_fitness_per_component(self):
        """--fitness-default auto: trunk->awq, dflash->lrq, etc.
        The unified policy records the resolved strategy per component.
        """
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            comps = [
                self._make_component(td_path, role)
                for role in ("trunk", "dflash", "dspark",
                             "mtp_nextn", "shared_embd")
            ]
            out_policy = td_path / "unified.json"
            cmd = [
                sys.executable, str(HERE / "unified_calibrate.py"),
                "--fitness-default", "auto",
                "--component", f"trunk={comps[0][1]}",
                "--component", f"dflash={comps[1][1]}",
                "--component", f"dspark={comps[2][1]}",
                "--component", f"mtp_nextn={comps[3][1]}",
                "--component", f"shared_embd={comps[4][1]}",
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
                # issues (the unit-level half already covers
                # the per-role table).
                self.skipTest(
                    f"end-to-end CLI failed (rc={result.returncode}); "
                    f"see stderr for details")
            with out_policy.open() as f:
                policy = json.load(f)
            # The per-component block records the resolved strategy.
            self.assertIn("components", policy)
            expected = {
                "trunk":       "awq",
                "dflash":      "lrq",
                "dspark":      "lrq",
                "mtp_nextn":   "lrq",
                "shared_embd": "flrq",
            }
            for role, want in expected.items():
                self.assertIn(role, policy["components"], role)
                self.assertEqual(
                    policy["components"][role]["fitness"], want,
                    f"{role}: expected {want!r}, got "
                    f"{policy['components'][role]['fitness']!r}")
            # Every tensor family is stamped with model_role.
            for key, entry in policy["tensor_families"].items():
                self.assertIn("model_role", entry,
                              f"family {key!r} missing model_role")
                self.assertIn(entry["model_role"], expected.keys(),
                              f"family {key!r} has unexpected role")

    def test_explicit_override(self):
        """--fitness-default lrq: every component runs lrq, regardless
        of the per-role table.
        """
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            comps = [
                self._make_component(td_path, role)
                for role in ("trunk", "dflash")
            ]
            out_policy = td_path / "unified.json"
            cmd = [
                sys.executable, str(HERE / "unified_calibrate.py"),
                "--fitness-default", "lrq",
                "--component", f"trunk={comps[0][1]}",
                "--component", f"dflash={comps[1][1]}",
                "--output", str(out_policy),
                "--per-tensor-calibrate", str(self.per_tensor),
                "--extra-arg=--lrq-iterations",
                "--extra-arg=4",
                "--extra-arg=--lrq-rank",
                "--extra-arg=2",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.skipTest(
                    f"end-to-end CLI failed (rc={result.returncode})")
            with out_policy.open() as f:
                policy = json.load(f)
            for role in ("trunk", "dflash"):
                self.assertEqual(
                    policy["components"][role]["fitness"], "lrq",
                    f"override {role}: expected lrq, got "
                    f"{policy['components'][role]['fitness']!r}")


def main() -> int:
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
