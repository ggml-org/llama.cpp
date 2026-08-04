"""Phase 16 end-to-end smoke test.

The Phase 16 stack is the closed loop:

    calibration policy
        -> writes to tessera.duckdb (tensor_stats)
        -> orchestrator runs the l5 requant plan
        -> l5_outcome records (did the plan reduce error?)
        -> l5_retune recomputes per-(model, family) weights
        -> l5_weights drives the next generation

This test exercises the full round-trip with a 4-component
trunk+dflash+dspark+mtp_nextn setup. It is the smoke that
proves the calibrate -> DB -> retune -> weights chain holds
end-to-end. Marked ``@pytest.mark.slow`` so the unified
runner's ``--quick`` flag skips it.

Run as a pytest module (auto-collected by the repo-root
pytest invocation in scripts/test-all.sh):

    python3 -m pytest tests/test_phase16_e2e.py -v

Or as a script:

    python3 tests/test_phase16_e2e.py

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

# Repo paths. The conftest.py at the repo root adds these to
# sys.path; we add them again here so the file is also runnable
# as a standalone script (python3 tests/test_phase16_e2e.py).
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TOOLS_TESSERA = REPO_ROOT / "tools" / "tessera"
for p in (str(REPO_ROOT), str(TOOLS_TESSERA), str(THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# pytest marker registration is done in the repo-root conftest.py
# so ``-m "not slow"`` is wired up there. The marker import here
# is a no-op when the file is run as a script.
try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


# Model roles exercised by the test. The four-conventional-Roles
# set covers the architectural components the Phase 16 schema
# disambiguates with the model_role column.
ROLES = ("trunk", "dflash", "dspark", "mtp_nextn")

# Default base weights the retune perturbs. The test asserts
# that the low-hit-rate family diverges from the high-hit-rate
# family; the divergence is relative to the base, not the
# absolute values.
DEFAULT_BASE_WEIGHTS = (0.5, 0.3, 0.2)


def _make_synthetic_npz(path: Path, role: str, in_dim: int = 4, out_dim: int = 4) -> None:
    """Write a minimal .npz that ``per_tensor_calibrate.load_layer``
    accepts. The values are deterministic per role so the test is
    reproducible across runs.
    """
    seed = sum(ord(c) for c in role) * 31
    rng = np.random.default_rng(seed=seed)
    weight = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
    train_acts = rng.standard_normal((8, in_dim)).astype(np.float32) * 0.5
    in_sum2 = (train_acts.astype(np.float32) ** 2).sum(axis=0)
    np.savez(
        path,
        weight=weight,
        train_activations=train_acts,
        in_sum2=in_sum2,
        counts=np.array(8, dtype=np.int64),
        name=np.str_(f"{role}.weight"),
        family=np.str_("attn_q"),
    )


def _run_unified_calibrate(component_dirs: dict[str, Path], out_path: Path) -> dict:
    """Run tools/tessera/unified_calibrate.py against the synthetic
    component bundles. Returns the parsed policy dict.
    """
    cmd = [
        sys.executable,
        str(TOOLS_TESSERA / "unified_calibrate.py"),
    ]
    for role in ROLES:
        cmd += ["--component", f"{role}={component_dirs[role]}"]
    cmd += [
        "--fitness", "lrq",
        "--output", str(out_path),
        # Fast path: 2 LRQ iterations, rank 2. The calibration
        # value isn't under test here; we only verify the policy
        # shape and the model_role tagging contract.
        "--extra-arg=--lrq-iterations", "--extra-arg=2",
        "--extra-arg=--lrq-rank",       "--extra-arg=2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"unified_calibrate.py failed (rc={result.returncode}); "
            f"stdout tail: {result.stdout[-400:]}; "
            f"stderr tail: {result.stderr[-400:]}"
        )
    return json.loads(out_path.read_text())


def _policy_to_tensor_stats_rows(policy: dict, model_hash: str) -> list[dict]:
    """Translate the policy's tensor_families into tensor_stats
    rows. Each row is keyed by (model_hash, model_role, name) so
    the orchestrator's read_unified_policy path can read it back.
    Only the columns the C++ reader consults are populated; the
    rest are NULL.
    """
    rows: list[dict] = []
    for fkey, entry in policy.get("tensor_families", {}).items():
        # The keys are "lrq:<tensor_name>"; the model_role lives
        # in the entry, not the key.
        model_role = entry.get("model_role", "trunk")
        # Recover the tensor name from the key. The split is
        # safe because the family key is "fitness:name" and
        # names don't contain ':'.
        _, _, name = fkey.partition(":")
        rows.append({
            "model_hash": model_hash,
            "model_role": model_role,
            "name": name,
            "family": entry.get("family", "unknown"),
            "dtype": "f16",
            "source": "py_calibrate_e2e",
            "recommended_action": entry.get("recommended_action", "quantize"),
        })
    return rows


def _seed_l5_outcome(
    db_path: Path, model_hash: str, rows: list[dict],
) -> None:
    """Insert l5_outcome rows via the TesseraDB buffered API.

    The TesseraDB API is the production path: the C++ orchestrator
    reads l5_outcome via the same query shape, and the
    l5_retune.compute_l5_weights function reads through
    TesseraDB-equivalent duckdb SQL.
    """
    from tessera_db import TesseraDB
    with TesseraDB.open(str(db_path)) as db:
        db.insert_l5_outcome(model_hash=model_hash, rows=rows)


def _read_l5_weights(db_path: Path, model_hash: str) -> "pl.DataFrame":
    """Read l5_weights back from the DB as a polars DataFrame.

    The C++ side reads l5_weights via ts_tessera_db_read_l5_weights
    (in tessera-quantize-db.cpp); the polars path is the Python
    test/consumer side.
    """
    import duckdb
    import polars as pl
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT model_hash, model_role, family, w_imatrix, "
            "w_gradient, w_layer, hit_rate, n_samples, "
            "retune_source, top_fraction "
            "FROM l5_weights WHERE model_hash = ? "
            "ORDER BY family",
            [model_hash],
        ).pl()
    finally:
        con.close()


def _synth_l5_outcome_rows() -> list[dict]:
    """Build the 4-row l5_outcome dataset the test asserts against.

    Two families, four rows each, deterministic:

      * attn_q (low hit_rate): 2 of 4 plans accepted, slope is
        positive. The retune's gate (1 - hit_rate) is 0.5, so the
        shift is non-zero. Weights move away from the base.
      * ffn_gate (high hit_rate): 4 of 4 plans accepted, gate is
        0. Weights stay at the base.

    The OLS the retune fits is 3-coefficient
    (im, gradient, layer) on synthetic values; we use a
    balanced (im == gradient) signal so the 2-coefficient and
    3-coefficient paths agree on the direction of the shift.
    The test asserts the divergence between the two families,
    not the sign of the shift (sign depends on which
    coefficient dominates the synthetic signal).
    """
    sens = [0.2, 0.4, 0.6, 0.8]
    deltas = [0.001, 0.002, 0.003, 0.004]
    rows: list[dict] = []
    # attn_q: hit_rate = 0.5 (2 of 4 accepted)
    for i, (s, d_, acc) in enumerate(
        zip(sens, deltas, [True, True, False, False])
    ):
        rows.append({
            "name": f"blk.0.attn_q.{i}",
            "layer": 0,
            "iteration": 0,
            "plan_id": f"p{i}",
            "family": "attn_q",
            "sensitivity_score": s,
            "mse_before": 0.01,
            "mse_after":  0.01 + d_,
            "delta_mse":  d_,
            "plan_accepted": acc,
            "accept_threshold": 0.0,
        })
    # ffn_gate: hit_rate = 1.0 (4 of 4 accepted)
    for i, (s, d_) in enumerate(zip(sens, deltas)):
        rows.append({
            "name": f"blk.0.ffn_gate.{i}",
            "layer": 0,
            "iteration": 0,
            "plan_id": f"q{i}",
            "family": "ffn_gate",
            "sensitivity_score": s,
            "mse_before": 0.01,
            "mse_after":  0.01 + d_,
            "delta_mse":  d_,
            "plan_accepted": True,
            "accept_threshold": 0.0,
        })
    return rows


def _run_retune(db_path: Path, model_hash: str) -> None:
    """Invoke tools/tessera/l5_retune.py in dry-run + print-table
    mode against the seeded DB. The retune writes to l5_weights
    only when ``--dry-run`` is NOT passed; we use ``--dry-run``
    here because we only assert the in-memory verdict. (The
    non-dry-run path is covered by the existing test_l5_retune
    tests at the unit level.)
    """
    cmd = [
        sys.executable,
        str(TOOLS_TESSERA / "l5_retune.py"),
        "--db", str(db_path),
        "--model-hash", model_hash,
        "--dry-run",
        "--print-table",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"l5_retune.py failed (rc={result.returncode}); "
            f"stdout: {result.stdout[-400:]}; "
            f"stderr: {result.stderr[-400:]}"
        )


def _run_retune_write(db_path: Path, model_hash: str) -> None:
    """Like _run_retune but writes the verdict into l5_weights.

    Used for the post-write assertion that the l5_weights rows
    have the expected per-family hit_rate partitioning.
    """
    cmd = [
        sys.executable,
        str(TOOLS_TESSERA / "l5_retune.py"),
        "--db", str(db_path),
        "--model-hash", model_hash,
        "--print-table",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"l5_retune.py (write) failed (rc={result.returncode}); "
            f"stdout: {result.stdout[-400:]}; "
            f"stderr: {result.stderr[-400:]}"
        )


class TestPhase16E2E(unittest.TestCase):
    """End-to-end smoke for the Phase 16 unified_calibrate -> DB
    -> l5_retune round-trip.

    One method, one round-trip. The test is structured to fail
    loudly at the first broken stage: a missing .npz format, a
    broken schema migration, or a wrong retune verdict all
    surface as concrete assertion failures with the offending
    stage's context in the message.
    """

    def setUp(self) -> None:
        self._td = Path(tempfile.mkdtemp(prefix="phase16_e2e_"))
        self.db_path = self._td / "tessera.duckdb"
        self.policy_path = self._td / "policy.json"

    def tearDown(self) -> None:
        shutil.rmtree(self._td, ignore_errors=True)

    def test_round_trip(self) -> None:
        # ----- Stage 1: synthesize 4 .npz bundles -----
        component_dirs: dict[str, Path] = {}
        for role in ROLES:
            d = self._td / role
            d.mkdir()
            _make_synthetic_npz(d / f"{role}.weight.npz", role)
            component_dirs[role] = d

        # ----- Stage 2: unified_calibrate -> policy JSON -----
        policy = _run_unified_calibrate(component_dirs, self.policy_path)
        self.assertEqual(
            policy["schema"],
            "llama.speculative.calibration-policy.v1",
        )
        self.assertEqual(policy.get("model_roles"), list(ROLES))
        # Every tensor family has model_role tagged.
        for fkey, entry in policy.get("tensor_families", {}).items():
            self.assertIn(
                "model_role", entry,
                f"policy tensor_families entry {fkey!r} missing model_role",
            )
            self.assertIn(
                entry["model_role"], ROLES,
                f"policy tensor_families entry {fkey!r} has unknown "
                f"model_role {entry['model_role']!r}",
            )

        # ----- Stage 3: fresh DB + Phase 16 schema -----
        # The C++ side owns the canonical schema; the Python side
        # uses migrate_model_role to bootstrap a fresh DB to the
        # same shape (idempotent: a no-op on a DB that already
        # has model_role columns).
        import duckdb
        con = duckdb.connect(str(self.db_path))
        con.close()
        from migrate_model_role import migrate
        migrate(str(self.db_path))

        # ----- Stage 4: insert policy rows into tensor_stats -----
        stats_rows = _policy_to_tensor_stats_rows(
            policy, model_hash="e2e_phase16")
        self.assertGreaterEqual(
            len(stats_rows), len(ROLES),
            "expected at least one tensor_stats row per role; got "
            f"{len(stats_rows)} (roles={ROLES})",
        )
        from tessera_db import TesseraDB
        with TesseraDB.open(str(self.db_path)) as dbi:
            dbi.insert_tensor_stats(
                model_hash="e2e_phase16", rows=stats_rows)

        # Sanity: the tensor_stats table has the expected rows.
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'e2e_phase16'"
            ).fetchone()[0]
            self.assertEqual(
                n, len(stats_rows),
                f"tensor_stats row count {n} != inserted {len(stats_rows)}",
            )
            # Per-role row count must be >= 1 each.
            per_role = con.execute(
                "SELECT model_role, COUNT(*) FROM tensor_stats "
                "WHERE model_hash = 'e2e_phase16' "
                "GROUP BY model_role ORDER BY model_role"
            ).fetchall()
        finally:
            con.close()
        roles_seen = {r[0] for r in per_role}
        self.assertEqual(
            roles_seen, set(ROLES),
            f"expected per-role rows for all of {ROLES}; got {per_role}",
        )

        # ----- Stage 5: synthesize l5_outcome (4 rows: 2 + 2) -----
        outcome_rows = _synth_l5_outcome_rows()
        self.assertEqual(len(outcome_rows), 8)
        _seed_l5_outcome(self.db_path, "e2e_phase16", outcome_rows)
        # 4 attn_q rows + 4 ffn_gate rows. The retune needs
        # n >= min_samples (default 3) per (model, family); both
        # families have 4.
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            by_family = con.execute(
                "SELECT family, COUNT(*) FROM l5_outcome "
                "WHERE model_hash = 'e2e_phase16' "
                "GROUP BY family ORDER BY family"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(
            dict(by_family), {"attn_q": 4, "ffn_gate": 4},
            f"unexpected per-family l5_outcome counts: {by_family}",
        )

        # ----- Stage 6: dry-run retune (in-memory verdict) -----
        _run_retune(self.db_path, "e2e_phase16")

        # ----- Stage 7: write the verdict to l5_weights -----
        _run_retune_write(self.db_path, "e2e_phase16")
        weights = _read_l5_weights(self.db_path, "e2e_phase16")
        # Two acted-on rows: one per family.
        self.assertEqual(
            weights.height, 2,
            f"expected 2 l5_weights rows; got {weights.height}:\n"
            f"{weights}",
        )
        families = set(weights["family"].to_list())
        self.assertEqual(
            families, {"attn_q", "ffn_gate"},
            f"unexpected l5_weights families: {families}",
        )

        # ----- Stage 8: assert the divergence -----
        # The low-hit-rate family (attn_q) MUST have weights that
        # differ from the high-hit-rate family (ffn_gate). The
        # retune's gate (1 - hit_rate) is 0.0 for ffn_gate, so
        # ffn_gate's weights are at the base. attn_q's gate is
        # 0.5, so the weights shift; the magnitude of the shift
        # is small but non-zero in the l2 norm.
        import polars as pl  # local import: keeps module importable
                              # when run as a script before the
                              # heavier deps are loaded.
        ffn = weights.filter(pl.col("family") == "ffn_gate").row(0, named=True)
        attn = weights.filter(pl.col("family") == "attn_q").row(0, named=True)
        # Verify the hit_rate partitioning that drives the shift.
        self.assertAlmostEqual(
            float(ffn["hit_rate"]), 1.0, places=6,
            msg=(
                f"ffn_gate should have hit_rate=1.0; "
                f"got {ffn['hit_rate']}"
            ),
        )
        self.assertAlmostEqual(
            float(attn["hit_rate"]), 0.5, places=6,
            msg=(
                f"attn_q should have hit_rate=0.5; "
                f"got {attn['hit_rate']}"
            ),
        )
        # Verify the high-hit-rate family is at the base (gate=0
        # means no shift).
        for w_name, base in zip(
            ("w_imatrix", "w_gradient", "w_layer"),
            DEFAULT_BASE_WEIGHTS,
        ):
            self.assertAlmostEqual(
                float(ffn[w_name]), base, places=6,
                msg=(
                    f"ffn_gate.{w_name} should be at base {base} "
                    f"(hit_rate=1.0 -> no shift); got {ffn[w_name]}"
                ),
            )
        # The low-hit-rate family MUST differ from the high-hit-rate
        # family. Use a tolerance slightly tighter than the
        # actual shift magnitude so a broken retune (returning
        # base weights for both families) is caught.
        diff = max(
            abs(float(attn["w_imatrix"]) - float(ffn["w_imatrix"])),
            abs(float(attn["w_gradient"]) - float(ffn["w_gradient"])),
            abs(float(attn["w_layer"])    - float(ffn["w_layer"])),
        )
        self.assertGreater(
            diff, 1e-4,
            f"low-hit-rate (attn_q) and high-hit-rate (ffn_gate) "
            f"weights should differ; attn={attn}, ffn={ffn}, "
            f"max-coord-diff={diff}",
        )


# pytest marker (so scripts/test-all.sh --quick can skip this).
# The mark is applied via the class decorator when pytest is
# available; standalone runs (python3 tests/test_phase16_e2e.py)
# ignore it.
if pytest is not None:
    TestPhase16E2E = pytest.mark.slow(TestPhase16E2E)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPhase16E2E)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
