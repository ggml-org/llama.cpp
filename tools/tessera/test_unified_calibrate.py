#!/usr/bin/env python3
"""Tests for ``tools/tessera/unified_calibrate.py`` + the
``--model-role`` flag in ``tools/tessera/per_tensor_calibrate.py`` +
the ``model_role`` consumer routing in
``tools/tile640/quantize_v3.py``.

Coverage:

1. **per_tensor_calibrate --model-role** (single-component path).
   The default ``--model-role trunk`` is the no-op behaviour (the
   legacy single-model contract). Explicit ``--model-role
   {dflash, dspark, mtp_nextn, shared_embd}`` stamps the value on
   the top-level policy and on every per-tensor entry. Invalid
   roles are rejected by argparse.

2. **unified_calibrate driver** (multi-component path). The
   synthetic 4-component + 1-shared bundle (5 trunk + 2 dflash +
   1 dspark + 2 mtp_nextn + 1 shared_embd = 11 tensors) produces a
   unified policy with the right schema, the right top-level
   ``model_role`` (None), the right ``components`` metadata, and
   the right per-tensor ``model_role`` tags.

3. **Consumer routing** (``tile640_quantize_v3.tensor_policy`` +
   ``lrq_policy_for``). A unified policy routes per-tensor qtype
   per-role: trunk entry for trunk tensors, dflash entry for
   dflash tensors, etc. A legacy single-model policy (no
   ``model_role``) still works exactly as before. Mixed policies
   (some entries with role, some without) fall back gracefully.

Run as ``python3 tools/tessera/test_unified_calibrate.py``. Exit 0
on success, non-zero on any failure.
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

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))  # for top-level import

REPO_ROOT = THIS_DIR.parent.parent
PER_TENSOR_TOOL = THIS_DIR / "per_tensor_calibrate.py"
UNIFIED_TOOL = THIS_DIR / "unified_calibrate.py"
TILE640_DIR = REPO_ROOT / "tools" / "tile640"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_npz(path: Path, name: str, in_dim: int, out_dim: int,
              n_tokens: int = 8, seed: int = 0) -> None:
    """Write a synthetic .npz bundle matching the per_tensor_calibrate
    schema. We pick the smallest plausible shapes (in_dim >= 16 so
    the LRQ default rank of 16 does not error; the test forces
    --lrq-rank to 4 to keep the calibration cheap)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(out_dim, in_dim)).astype(np.float32) * 0.1
    X = rng.normal(size=(n_tokens, in_dim)).astype(np.float32) * 0.1
    np.savez(path, weight=W, train_activations=X, name=name, family="attn_q")


def _synthetic_component_dirs(root: Path) -> dict[str, Path]:
    """Build the 4-component + 1-shared synthetic corpus.

    The names follow the unified-archive convention so
    ``_infer_tensor_role`` in tile640_quantize_v3 routes them
    correctly: ``blk.*`` -> trunk, ``dflash.*`` -> dflash,
    ``markov_*`` -> dspark, ``blk.*.nextn.*`` -> mtp_nextn,
    ``token_embd.*`` -> shared_embd.
    """
    out: dict[str, Path] = {}
    # Trunk: 5 tensors, in=32, out=64.
    trunk = root / "trunk"
    trunk.mkdir()
    for i, name in enumerate((
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
    )):
        _make_npz(
            trunk / f"{name.replace('.', '_')}.npz",
            name, in_dim=32, out_dim=64, seed=i + 1,
        )
    out["trunk"] = trunk
    # DFlash: 2 tensors, in=24, out=48.
    dflash = root / "dflash"
    dflash.mkdir()
    for i, name in enumerate((
        "dflash.encoder.fc.weight",
        "dflash.decoder.fc.weight",
    )):
        _make_npz(
            dflash / f"{name.replace('.', '_')}.npz",
            name, in_dim=24, out_dim=48, seed=i + 10,
        )
    out["dflash"] = dflash
    # DSpark: 1 tensor.
    dspark = root / "dspark"
    dspark.mkdir()
    _make_npz(dspark / "markov_w1.npz", "markov_w1.weight",
              in_dim=16, out_dim=32, seed=42)
    out["dspark"] = dspark
    # MTP nextn: 2 tensors.
    mtp = root / "mtp"
    mtp.mkdir()
    for i, name in enumerate((
        "blk.0.nextn.eh_proj.weight",
        "blk.0.nextn.en_proj.weight",
    )):
        _make_npz(
            mtp / f"{name.replace('.', '_')}.npz",
            name, in_dim=32, out_dim=64, seed=i + 100,
        )
    out["mtp_nextn"] = mtp
    # Shared embedding: 1 tensor.
    shared = root / "shared"
    shared.mkdir()
    _make_npz(shared / "token_embd.npz", "token_embd.weight",
              in_dim=64, out_dim=64, seed=999)
    out["shared_embd"] = shared
    return out


def _run_per_tensor(
    layers_arg: Path, output: Path, *extra: str,
) -> int:
    """Run per_tensor_calibrate.py with the given extra CLI args.

    The synthetic 8x8 corpus is calibrated with --lrq-iterations 2
    and --lrq-rank 4 to keep the wall-clock cost of the test
    suite negligible. ``extra`` is forwarded verbatim, so the
    caller controls the --model-role / --lrq-agg / --seed knobs.
    """
    cmd = [
        sys.executable, str(PER_TENSOR_TOOL),
        "--fitness", "lrq",
        "--layers", str(layers_arg),
        "--output", str(output),
        "--lrq-iterations", "2",
        "--lrq-rank", "4",
        "--max-tokens", "4",
    ]
    cmd.extend(extra)
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def _run_unified(components: dict[str, Path], output: Path,
                 in_process: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(UNIFIED_TOOL),
           "--trunk-npz", str(components["trunk"]),
           "--dflash-npz", str(components["dflash"]),
           "--dspark-npz", str(components["dspark"]),
           "--mtp-npz", str(components["mtp_nextn"]),
           "--shared-embd-npz", str(components["shared_embd"]),
           "--fitness", "lrq",
           "--output", str(output),
           "--lrq-iterations", "2",
           "--lrq-rank", "4",
           "--max-tokens", "4"]
    if in_process:
        cmd.append("--in-process")
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 1. per_tensor_calibrate --model-role
# ---------------------------------------------------------------------------


class TestPerTensorModelRole(unittest.TestCase):
    """The per-component calibration sub-driver must stamp the
    model_role on the top-level policy and on every per-tensor
    entry. The default is ``trunk`` (legacy single-model)."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="per_tensor_role_")
        self._td = Path(self._tmp)
        self._npz = self._td / "tiny.npz"
        _make_npz(self._npz, "tiny.weight",
                  in_dim=32, out_dim=64, seed=0)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, *extra: str) -> dict:
        out = self._td / "out.json"
        rc = _run_per_tensor(self._npz, out, *extra)
        self.assertEqual(rc, 0, f"per_tensor_calibrate failed (extra={extra})")
        with out.open() as f:
            return json.load(f)

    def test_default_role_is_trunk(self) -> None:
        """No --model-role -> default 'trunk' (legacy behaviour)."""
        policy = self._run()
        self.assertEqual(policy["schema"],
                         "llama.speculative.calibration-policy.v1")
        self.assertEqual(policy["model_role"], "trunk")
        entries = policy["tensor_families"]
        self.assertEqual(len(entries), 1)
        for entry in entries.values():
            self.assertEqual(entry.get("model_role"), "trunk")

    def test_explicit_role_dflash(self) -> None:
        """--model-role dflash stamps 'dflash' on the policy and entries."""
        policy = self._run("--model-role", "dflash")
        self.assertEqual(policy["model_role"], "dflash")
        for entry in policy["tensor_families"].values():
            self.assertEqual(entry.get("model_role"), "dflash")

    def test_explicit_role_dspark(self) -> None:
        policy = self._run("--model-role", "dspark")
        self.assertEqual(policy["model_role"], "dspark")
        for entry in policy["tensor_families"].values():
            self.assertEqual(entry.get("model_role"), "dspark")

    def test_explicit_role_mtp_nextn(self) -> None:
        policy = self._run("--model-role", "mtp_nextn")
        self.assertEqual(policy["model_role"], "mtp_nextn")
        for entry in policy["tensor_families"].values():
            self.assertEqual(entry.get("model_role"), "mtp_nextn")

    def test_explicit_role_shared_embd(self) -> None:
        policy = self._run("--model-role", "shared_embd")
        self.assertEqual(policy["model_role"], "shared_embd")
        for entry in policy["tensor_families"].values():
            self.assertEqual(entry.get("model_role"), "shared_embd")

    def test_invalid_role_rejected(self) -> None:
        """--model-role bogus is rejected by argparse (exit 2)."""
        out = self._td / "out.json"
        rc = _run_per_tensor(self._npz, out, "--model-role", "bogus")
        self.assertEqual(rc, 2)

    def test_no_subkeys_unchanged(self) -> None:
        """The legacy per-tensor fields (match, exact, lrq_u, lrq_v,
        ...) are unaffected by the new role tag."""
        policy = self._run()
        entry = next(iter(policy["tensor_families"].values()))
        # The legacy fields are still there.
        self.assertIn("match", entry)
        self.assertIn("exact", entry)
        self.assertIn("lrq_u", entry)
        self.assertIn("lrq_v", entry)
        self.assertIn("lrq_rank", entry)
        # The new field is also there.
        self.assertIn("model_role", entry)


# ---------------------------------------------------------------------------
# 2. unified_calibrate driver
# ---------------------------------------------------------------------------


class TestUnifiedCalibrateDriver(unittest.TestCase):
    """The unified driver produces a single policy from N
    per-component bundles, tags every entry with model_role, and
    reports per-component metadata in the ``components`` section."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="unified_calibrate_")
        self._td = Path(self._tmp)
        self.components = _synthetic_component_dirs(self._td)
        self._out = self._td / "unified.json"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_in_process_full_synthetic(self) -> None:
        """4-component + 1-shared (11 tensors) -> unified policy."""
        r = _run_unified(self.components, self._out, in_process=True)
        self.assertEqual(r.returncode, 0,
                         f"unified_calibrate failed: stderr={r.stderr}")
        with self._out.open() as f:
            policy = json.load(f)
        # Top-level shape.
        self.assertEqual(policy["schema"],
                         "llama.speculative.calibration-policy.v1")
        self.assertIsNone(
            policy["model_role"],
            "unified policy's top-level model_role must be None",
        )
        # components section: one entry per role, with metadata.
        self.assertEqual(set(policy["components"]),
                         {"trunk", "dflash", "dspark", "mtp_nextn", "shared_embd"})
        # The per-component metadata: each role has a tensor_count
        # matching the synthetic corpus.
        expected_counts = {
            "trunk": 5, "dflash": 2, "dspark": 1, "mtp_nextn": 2,
            "shared_embd": 1,
        }
        for role, count in expected_counts.items():
            self.assertEqual(
                policy["components"][role]["tensor_count"], count,
                f"role {role!r} tensor_count mismatch",
            )
            self.assertEqual(
                policy["components"][role]["model_role"], role,
                f"role {role!r} model_role mismatch",
            )
            self.assertEqual(
                policy["components"][role]["sub_schema"], "lrq",
                f"role {role!r} sub_schema mismatch",
            )
        # tensor_families: 11 entries, each prefixed by role.
        families = policy["tensor_families"]
        self.assertEqual(len(families), sum(expected_counts.values()))
        # Per-role: every entry has the right model_role tag.
        role_to_names = {
            "trunk": [
                "blk.0.attn_q.weight", "blk.0.attn_k.weight",
                "blk.0.attn_v.weight", "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
            ],
            "dflash": [
                "dflash.encoder.fc.weight",
                "dflash.decoder.fc.weight",
            ],
            "dspark": ["markov_w1.weight"],
            "mtp_nextn": [
                "blk.0.nextn.eh_proj.weight",
                "blk.0.nextn.en_proj.weight",
            ],
            "shared_embd": ["token_embd.weight"],
        }
        seen_by_role: dict[str, list[str]] = {r: [] for r in role_to_names}
        for key, entry in families.items():
            self.assertTrue(
                key.startswith(tuple(f"{r}:" for r in role_to_names)),
                f"entry key {key!r} missing role prefix",
            )
            entry_role = entry.get("model_role")
            self.assertIn(entry_role, role_to_names)
            seen_by_role[entry_role].extend(entry.get("match", []))
        for role, names in role_to_names.items():
            self.assertEqual(
                sorted(seen_by_role[role]), sorted(names),
                f"role {role!r} tensor names mismatch: "
                f"got {seen_by_role[role]!r}, expected {names!r}",
            )

    def test_subprocess_mode(self) -> None:
        """Same shape via the subprocess execution path."""
        r = _run_unified(self.components, self._out, in_process=False)
        self.assertEqual(r.returncode, 0,
                         f"unified_calibrate subprocess failed: stderr={r.stderr}")
        with self._out.open() as f:
            policy = json.load(f)
        self.assertEqual(policy["schema"],
                         "llama.speculative.calibration-policy.v1")
        self.assertEqual(set(policy["components"]),
                         {"trunk", "dflash", "dspark", "mtp_nextn", "shared_embd"})

    def test_subset_of_components(self) -> None:
        """Supplying only trunk + dflash is allowed."""
        out = self._td / "unified_subset.json"
        cmd = [sys.executable, str(UNIFIED_TOOL),
               "--trunk-npz", str(self.components["trunk"]),
               "--dflash-npz", str(self.components["dflash"]),
               "--fitness", "lrq",
               "--output", str(out),
               "--in-process",
               "--lrq-iterations", "2", "--lrq-rank", "4", "--max-tokens", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0,
                         f"subset run failed: stderr={r.stderr}")
        with out.open() as f:
            policy = json.load(f)
        self.assertEqual(set(policy["components"]), {"trunk", "dflash"})
        self.assertEqual(policy["components"]["trunk"]["tensor_count"], 5)
        self.assertEqual(policy["components"]["dflash"]["tensor_count"], 2)

    def test_no_components_rejected(self) -> None:
        """No --{component}-npz -> argparse error (exit 2)."""
        out = self._td / "unified_none.json"
        r = subprocess.run(
            [sys.executable, str(UNIFIED_TOOL),
             "--fitness", "lrq",
             "--output", str(out),
             "--in-process"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("at least one", r.stderr)

    def test_missing_component_path_rejected(self) -> None:
        """A non-existent --trunk-npz path fails fast with FileNotFoundError."""
        out = self._td / "unified_missing.json"
        r = subprocess.run(
            [sys.executable, str(UNIFIED_TOOL),
             "--trunk-npz", str(self._td / "does_not_exist"),
             "--fitness", "lrq",
             "--output", str(out),
             "--in-process"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not found", r.stderr)


# ---------------------------------------------------------------------------
# 3. tile640_quantize_v3 consumer routing
# ---------------------------------------------------------------------------


class TestTensorPolicyRouting(unittest.TestCase):
    """The Python consumer routes per-tensor qtype per-role when the
    policy carries ``model_role`` metadata, and falls back to the
    legacy single-model behaviour otherwise."""

    def setUp(self) -> None:
        sys.path.insert(0, str(TILE640_DIR))
        # Imported lazily so the import only happens when the
        # tile640 module is available; in CI without numpy/scipy
        # the tile640 module may not import, and the corresponding
        # test class is skipped.
        try:
            import quantize_v3 as q  # type: ignore[import-not-found]
            self.q = q
        except Exception as exc:  # pragma: no cover - env-dependent
            self.skipTest(f"tile640_quantize_v3 import failed: {exc}")

    def test_infer_tensor_role(self) -> None:
        """Every name pattern returns the expected role."""
        q = self.q
        cases = {
            "blk.0.attn_q.weight": "trunk",
            "blk.0.ffn_gate.weight": "trunk",
            "dflash.encoder.fc.weight": "dflash",
            "dflash.decoder.fc.weight": "dflash",
            "markov_w1.weight": "dspark",
            "head_proj.weight": "dspark",
            "blk.0.nextn.eh_proj.weight": "mtp_nextn",
            "nextn.0.eh_proj.weight": "mtp_nextn",
            "token_embd.weight": "shared_embd",
            "output.weight": "shared_embd",
            "token_embd.norm.weight": "shared_embd",
            "unknown.weight": None,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(q._infer_tensor_role(name), expected)

    def test_unified_policy_routes_per_role(self) -> None:
        """Each tensor picks the entry whose model_role matches its
        inferred role. shared_embd is a cross-role fallback."""
        q = self.q
        policy = self._build_unified_policy()
        # Trunk tensor -> trunk entry.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "blk.0.attn_q.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.001, places=9)
        # DFlash tensor -> dflash entry.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "dflash.encoder.fc.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.01, places=9)
        # DSpark tensor -> dspark entry.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "markov_w1.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.02, places=9)
        # MTP tensor -> mtp_nextn entry.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "blk.0.nextn.eh_proj.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.015, places=9)
        # Shared tensor -> shared_embd entry.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "token_embd.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.05, places=9)

    def test_legacy_single_model_policy_unchanged(self) -> None:
        """A policy without model_role metadata takes the legacy
        single-model path. Any match is a candidate, highest rank
        wins (the same rule the pre-Phase-16 code used)."""
        q = self.q
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "lrq:attn_q": {
                    "match": ["attn_q"], "exact": False,
                    "outlier_fraction": 0.003, "ternary_threshold": 0.9,
                },
            },
        }
        # Trunk tensor still matches the legacy entry.
        frac, _, _, _, thresh = q.tensor_policy(
            policy, "blk.0.attn_q.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.003, places=9)
        self.assertAlmostEqual(thresh, 0.9, places=9)
        # Non-matching tensor falls back to the default.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "dflash.encoder.fc.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.005, places=9)

    def test_mixed_policy_role_specific_wins(self) -> None:
        """A policy with both role-specific and legacy entries:
        role-specific wins for the matching tensor, legacy is the
        fallback for everything else."""
        q = self.q
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "lrq:attn_q": {
                    "match": ["attn_q"], "exact": False,
                    "outlier_fraction": 0.003, "ternary_threshold": 0.9,
                },
                "trunk:lrq:blk.0.attn_q.weight": {
                    "match": ["blk.0.attn_q.weight"], "exact": False,
                    "model_role": "trunk",
                    "outlier_fraction": 0.001, "ternary_threshold": 1.0,
                },
            },
        }
        # Trunk tensor: trunk entry wins.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "blk.0.attn_q.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.001, places=9)
        # Non-trunk tensor (would match legacy): legacy entry wins.
        frac, _, _, _, _ = q.tensor_policy(
            policy, "dflash.attn_q.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.003, places=9)

    def test_shared_embd_fallback_for_shared_tensor(self) -> None:
        """A shared tensor (output.weight) with no role-specific
        entry falls back to a substring-matching shared_embd entry.
        """
        q = self.q
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "shared_embd:lrq:shared": {
                    "match": ["token_embd", "output"], "exact": False,
                    "model_role": "shared_embd",
                    "outlier_fraction": 0.07, "ternary_threshold": 1.1,
                },
            },
        }
        frac, _, _, _, thresh = q.tensor_policy(
            policy, "output.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.07, places=9)
        self.assertAlmostEqual(thresh, 1.1, places=9)
        frac, _, _, _, _ = q.tensor_policy(
            policy, "token_embd.weight", 0.005, 0.5
        )
        self.assertAlmostEqual(frac, 0.07, places=9)

    def _build_unified_policy(self) -> dict:
        """Build a unified policy covering all 5 roles + 5 sample
        tensors. Outlier fractions are unique so the routing test
        can detect which entry won."""
        return {
            "schema": "llama.speculative.calibration-policy.v1",
            "model_role": None,
            "tensor_families": {
                "trunk:lrq:blk.0.attn_q.weight": {
                    "match": ["blk.0.attn_q.weight"], "exact": False,
                    "model_role": "trunk",
                    "outlier_fraction": 0.001, "ternary_threshold": 1.0,
                },
                "dflash:lrq:dflash.encoder.fc.weight": {
                    "match": ["dflash.encoder.fc.weight"], "exact": False,
                    "model_role": "dflash",
                    "outlier_fraction": 0.01, "ternary_threshold": 1.5,
                },
                "dspark:lrq:markov_w1.weight": {
                    "match": ["markov_w1.weight"], "exact": False,
                    "model_role": "dspark",
                    "outlier_fraction": 0.02, "ternary_threshold": 2.0,
                },
                "mtp_nextn:lrq:blk.0.nextn.eh_proj.weight": {
                    "match": ["blk.0.nextn.eh_proj.weight"], "exact": False,
                    "model_role": "mtp_nextn",
                    "outlier_fraction": 0.015, "ternary_threshold": 1.8,
                },
                "shared_embd:lrq:token_embd.weight": {
                    "match": ["token_embd.weight"], "exact": False,
                    "model_role": "shared_embd",
                    "outlier_fraction": 0.05, "ternary_threshold": 1.0,
                },
            },
        }


class TestLrqPolicyForRouting(unittest.TestCase):
    """The LRQ path (the ``lrq_policy_for`` helper in
    tile640_quantize_v3) also routes by role when the policy
    carries ``model_role`` metadata."""

    def setUp(self) -> None:
        sys.path.insert(0, str(TILE640_DIR))
        try:
            import quantize_v3 as q  # type: ignore[import-not-found]
            self.q = q
        except Exception as exc:  # pragma: no cover - env-dependent
            self.skipTest(f"tile640_quantize_v3 import failed: {exc}")

    def test_role_specific_entry_wins(self) -> None:
        """A trunk tensor prefers the role=trunk entry over a
        substring-matching legacy entry."""
        q = self.q
        # Build a small synthetic U, V for each candidate.
        u_legacy = np.eye(4, 2, dtype=np.float32)
        v_legacy = np.eye(2, 4, dtype=np.float32)
        u_trunk = np.eye(4, 2, dtype=np.float32) * 2.0
        v_trunk = np.eye(2, 4, dtype=np.float32) * 3.0
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "lrq:attn_q": {
                    "match": ["attn_q"], "exact": False,
                    "lrq_u": u_legacy.tolist(),
                    "lrq_v": v_legacy.tolist(),
                    "lrq_rank": 2,
                },
                "trunk:lrq:blk.0.attn_q.weight": {
                    "match": ["blk.0.attn_q.weight"], "exact": False,
                    "model_role": "trunk",
                    "lrq_u": u_trunk.tolist(),
                    "lrq_v": v_trunk.tolist(),
                    "lrq_rank": 2,
                },
            },
        }
        result = q.lrq_policy_for(policy, "blk.0.attn_q.weight")
        self.assertIsNotNone(result)
        rank, u, v, agg = result
        self.assertEqual(rank, 2)
        # The trunk entry has u=2*I, the legacy has u=I. The trunk
        # entry should win because its role_score=2 vs legacy's 0.
        np.testing.assert_array_equal(u, u_trunk)
        np.testing.assert_array_equal(v, v_trunk)
        self.assertEqual(agg, "mean")

    def test_legacy_path_unchanged(self) -> None:
        """A policy with no model_role metadata takes the legacy
        path: the first valid match wins."""
        q = self.q
        u = np.eye(4, 2, dtype=np.float32)
        v = np.eye(2, 4, dtype=np.float32)
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "lrq:blk.0.attn_q.weight": {
                    "match": ["blk.0.attn_q.weight"], "exact": True,
                    "lrq_u": u.tolist(),
                    "lrq_v": v.tolist(),
                    "lrq_rank": 2,
                },
            },
        }
        result = q.lrq_policy_for(policy, "blk.0.attn_q.weight")
        self.assertIsNotNone(result)
        rank, u2, v2, agg = result
        self.assertEqual(rank, 2)
        np.testing.assert_array_equal(u2, u)
        np.testing.assert_array_equal(v2, v)


if __name__ == "__main__":
    unittest.main()
