"""Tests for tools/tessera/embedding_budget.py (M2 producer).

Pins the residual-envelope formula, the role-priority table,
the confidence scaling, the mmproj term, and the writer-
loadable sidecar shape. The test fixture is in-memory (the
pure function takes pre-loaded data) so the tests run in
well under 5 seconds.

Run as a unittest module. Exit 0 on success, non-zero on
failure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import embedding_budget  # noqa: E402
from embedding_budget import (  # noqa: E402
    BUDGET_CLAMP_MAX,
    DEFAULT_N_TARGET,
    DEFAULT_ROLE_PRIORITIES,
    EnvelopeConfig,
    MMPROJ_NAME_PREFIXES,
    RoleBudget,
    SHARED_OWNING_ROLES,
    SHARED_TENSOR_NAMES,
    SIDECAR_SCHEMA,
    _dtype_bits,
    _find_n_elements,
    _mmproj_footprint_bits,
    _n_samples_for_role,
    _parse_role_priority_overrides,
    _priority_for_role,
    _source_footprint_bits,
    _verdict_dtype,
    compute_role_budgets,
    flatten_role_budgets_for_sidecar,
    write_sidecar_json,
)
from l5_retune import DTYPE_BITS  # noqa: E402


def _t(
    model_role: str,
    name: str,
    n_elements: int | None,
    dtype: str,
    family: str = "token_embd",
) -> dict:
    """Build one tensor_stats row for tests."""
    return {
        "model_hash": "m",
        "model_role": model_role,
        "name":       name,
        "family":     family,
        "n_elements": n_elements,
        "dtype":      dtype,
    }


def _v(
    model_role: str,
    name: str,
    dtype: str,
) -> dict:
    """Build one policy verdict (per-tensor calibration decision)."""
    return {
        "model_role": model_role,
        "name":       name,
        "dtype":      dtype,
    }


class TestPureProducer(unittest.TestCase):
    """Unit tests for the pure ``compute_role_budgets`` function."""

    # ---- 1. Trunk + dflash shared, no mmproj -----

    def test_trunk_dflash_shared_no_mmproj(self) -> None:
        """A simple trunk + dflash case: each role owns
        token_embd.weight, dflash has a larger budget
        because its non-shared footprint is smaller
        (fewer / smaller non-shared tensors). The dflash
        weight is the 2.0 default; trunk weight is 1.0.

        Fixture:
          trunk (F16, 1024-elems-per-tensor, 4 non-shared
                 tensors) -> 1024 * 16 * 4 = 65536 bits non-shared
                 + token_embd.weight at 1024 elems.
          dflash (F16, 1024 elems, 1 non-shared tensor) ->
                 1024 * 16 = 16384 bits non-shared
                 + token_embd.weight at 1024 elems.
          n_target = 8 (default); n_samples is the count of
                 non-shared verdicts per role:
                 trunk -> 4, dflash -> 1.
                 confidence = n_samples / 8:
                 trunk  = 4/8 = 0.5
                 dflash = 1/8 = 0.125
                 weight = priority * confidence:
                 trunk  = 1.0 * 0.5  = 0.5
                 dflash = 2.0 * 0.125 = 0.25
          E(r) = source_footprint_bits(r) * 1.0.
          E(trunk) = (4+1) * 1024 * 16 = 81920
          E(dflash) = (1+1) * 1024 * 16 = 32768
          S_t(trunk) = 4 * 1024 * 16 = 65536
          S_t(dflash) = 1 * 1024 * 16 = 16384
          M(r) = 0 (no mmproj)
          Residual:
            trunk  = 81920 - 65536 - 0 = 16384
            dflash = 32768 - 16384 - 0 = 16384
          Budget (clamp 0..16):
            trunk  = 16384 / 1024 = 16 -> clamp(16, 0, 16) = 16
            dflash = 16384 / 1024 = 16 -> clamp(16, 0, 16) = 16
        """
        rows = [
            # Trunk non-shared
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "blk.0.attn_k.weight", 1024, "f16", "attn_k"),
            _t("trunk", "blk.0.attn_v.weight", 1024, "f16", "attn_v"),
            _t("trunk", "blk.0.ffn_gate.weight", 1024, "f16", "ffn_gate"),
            # Trunk shared
            _t("trunk", "token_embd.weight", 1024, "f16"),
            # Dflash non-shared
            _t("dflash", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            # Dflash shared
            _t("dflash", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "blk.0.attn_k.weight", "f16"),
            _v("trunk", "blk.0.attn_v.weight", "f16"),
            _v("trunk", "blk.0.ffn_gate.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
            _v("dflash", "blk.0.attn_q.weight", "f16"),
            _v("dflash", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        by_role = {b.model_role: b for b in budgets}
        self.assertEqual(set(by_role), {"trunk", "dflash"})
        # Both end up with the same residual-per-element
        # (16 bits/elem), so the budgets match. dflash
        # has the higher weight (2.0 priority vs 1.0).
        self.assertEqual(by_role["trunk"].budget_bits_per_elem, 16)
        self.assertEqual(by_role["dflash"].budget_bits_per_elem, 16)
        self.assertAlmostEqual(
            by_role["dflash"].weight, 0.25, places=9,
            msg="dflash weight = 2.0 * 1/8 = 0.25",
        )
        self.assertAlmostEqual(
            by_role["trunk"].weight, 0.5, places=9,
            msg="trunk weight = 1.0 * 4/8 = 0.5",
        )
        # Writer-readable shape: flatten produces one entry
        # per role with the per-role MIN budget.
        flat = flatten_role_budgets_for_sidecar(budgets)
        self.assertEqual(len(flat), 2)
        for entry in flat:
            self.assertEqual(
                set(entry.keys()),
                {"model_role", "budget_bits", "weight"},
            )
        by_role_flat = {e["model_role"]: e for e in flat}
        self.assertEqual(by_role_flat["trunk"]["budget_bits"], 16)
        self.assertEqual(by_role_flat["dflash"]["budget_bits"], 16)
        self.assertAlmostEqual(
            by_role_flat["dflash"]["weight"], 0.25, places=9,
        )

    # ---- 2. With mmproj term -----

    def test_with_mmproj_term_reduces_residual(self) -> None:
        """When role r owns v./a./mm.* tensors, M(r) > 0
        reduces the residual and the budget goes down
        accordingly.

        Fixture: same as test 1 but trunk has 2 mmproj
        tensors (v.embed, mm.proj.0). M(trunk) = 2 * 1024
        * 16 * 1.0 = 32768 bits. Residual = 16384 - 32768
        = -16384 (negative). budget = 0 + warning.

        Dflash: no mmproj -> same as test 1.
        """
        rows = [
            # Trunk non-shared
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "blk.0.attn_k.weight", 1024, "f16", "attn_k"),
            _t("trunk", "blk.0.attn_v.weight", 1024, "f16", "attn_v"),
            _t("trunk", "blk.0.ffn_gate.weight", 1024, "f16", "ffn_gate"),
            # Trunk mmproj (v./a./mm. prefix)
            _t("trunk", "v.embed.weight", 1024, "f16", "vision_tower"),
            _t("trunk", "mm.proj.0.weight", 1024, "f16", "mm_projector"),
            # Trunk shared
            _t("trunk", "token_embd.weight", 1024, "f16"),
            # Dflash non-shared
            _t("dflash", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            # Dflash shared
            _t("dflash", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "blk.0.attn_k.weight", "f16"),
            _v("trunk", "blk.0.attn_v.weight", "f16"),
            _v("trunk", "blk.0.ffn_gate.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
            _v("dflash", "blk.0.attn_q.weight", "f16"),
            _v("dflash", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            budgets = compute_role_budgets(verdicts, rows, cfg)
        by_role = {b.model_role: b for b in budgets}
        # Trunk: negative residual -> budget=0 + warning.
        self.assertEqual(by_role["trunk"].budget_bits_per_elem, 0)
        self.assertTrue(
            any("negative residual" in str(w.message) for w in caught),
            "expected a 'negative residual' warning for trunk",
        )
        # Dflash: no mmproj -> same as test 1.
        self.assertEqual(by_role["dflash"].budget_bits_per_elem, 16)

    # ---- 3. Zero fraction -> empty -----

    def test_zero_fraction_returns_empty(self) -> None:
        """When the user opts out of the size envelope
        (base_budget_fraction <= 0), the producer returns
        ``[]`` — the writer's no-budget contract.
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        for fraction in (0.0, -0.1, -1.0):
            cfg = EnvelopeConfig(base_budget_fraction=fraction)
            self.assertEqual(
                compute_role_budgets(verdicts, rows, cfg),
                [],
                f"fraction={fraction} should return []",
            )

    # ---- 4. Negative residual -> budget = 0 + warning -----

    def test_negative_residual_emits_zero_with_warning(self) -> None:
        """When the residual is negative (non-shared +
        mmproj already exceed the envelope), the budget
        is 0 and a warning is logged. The writer's
        relaxation logic handles a 0 budget with the
        dynamic-weighting rule.

        To force a negative residual, the non-shared
        tensor's bits must exceed E(r). E(r) =
        source_footprint_bits * fraction (default 1.0).
        With source dtype = f32 (32 bits/elem) and a
        non-shared tensor that's larger than the role's
        total, we exceed the envelope. Easiest: split
        the role between a HUGE non-shared (32 bits)
        and a tiny shared (16 bits). E = huge + tiny;
        S_t = huge; M = 0. Residual = tiny - 0 which
        is still positive. To get NEGATIVE: upgrade the
        non-shared verdict to 64 bits (f64 isn't in
        DTYPE_BITS, so use a non-standard schema trick
        is wrong). Instead, set base_budget_fraction to
        0.5 so E shrinks below S_t.

        E = (huge + tiny) * 0.5; S_t = huge. Residual
        = (huge + tiny) * 0.5 - huge = -huge*0.5 + tiny
        = tiny - 0.5*huge. With huge = 10M, tiny = 1,
        0.5 fraction: residual = 1 - 5M < 0. Good.
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight",
               10_000_000, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig(base_budget_fraction=0.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            budgets = compute_role_budgets(verdicts, rows, cfg)
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0].model_role, "trunk")
        self.assertEqual(budgets[0].budget_bits_per_elem, 0)
        self.assertTrue(
            any("negative residual" in str(w.message) for w in caught),
        )

    # ---- 5. Dflash priority = 2.0 default; CLI override -----

    def test_dflash_priority_default_and_override(self) -> None:
        """Dflash weight = 2.0 * confidence by default; an
        explicit ``--role-priority dflash=3.0`` override
        bumps it to 3.0 * confidence.
        """
        # Minimal: 1 non-shared verdict per role, n_target=8,
        # confidence = 1/8.
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
            _t("dflash", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("dflash", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
            _v("dflash", "blk.0.attn_q.weight", "f16"),
            _v("dflash", "token_embd.weight", "f16"),
        ]
        # Default dflash priority = 2.0.
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        by_role = {b.model_role: b for b in budgets}
        self.assertAlmostEqual(
            by_role["dflash"].weight, 2.0 / DEFAULT_N_TARGET, places=9,
            msg=f"dflash weight = 2.0 / {DEFAULT_N_TARGET}",
        )
        self.assertAlmostEqual(
            by_role["trunk"].weight, 1.0 / DEFAULT_N_TARGET, places=9,
        )
        # Override dflash priority to 3.0.
        cfg2 = EnvelopeConfig(
            role_priorities={"dflash": 3.0},
        )
        budgets2 = compute_role_budgets(verdicts, rows, cfg2)
        by_role2 = {b.model_role: b for b in budgets2}
        self.assertAlmostEqual(
            by_role2["dflash"].weight, 3.0 / DEFAULT_N_TARGET, places=9,
        )
        # Trunk priority is unchanged.
        self.assertAlmostEqual(
            by_role2["trunk"].weight, 1.0 / DEFAULT_N_TARGET, places=9,
        )

    # ---- 6. Confidence scaling: n_samples = 0 and n_samples = 4 -----

    def test_confidence_scaling(self) -> None:
        """n_samples(r) = 0 -> confidence = 0 -> weight = 0
        (writer treats role as unconstrained).

        n_samples(r) = 4 with n_target=8 -> confidence = 0.5
        -> weight = 0.5 * priority.
        """
        # Case A: trunk has 0 non-shared verdicts (only the
        # shared tensor verdict); dflash has 4 non-shared verdicts.
        rows_a = [
            _t("trunk", "token_embd.weight", 1024, "f16"),
            _t("dflash", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("dflash", "blk.0.attn_k.weight", 1024, "f16", "attn_k"),
            _t("dflash", "blk.0.attn_v.weight", 1024, "f16", "attn_v"),
            _t("dflash", "blk.0.ffn_gate.weight", 1024, "f16", "ffn_gate"),
            _t("dflash", "token_embd.weight", 1024, "f16"),
        ]
        verdicts_a = [
            _v("trunk", "token_embd.weight", "f16"),
            _v("dflash", "blk.0.attn_q.weight", "f16"),
            _v("dflash", "blk.0.attn_k.weight", "f16"),
            _v("dflash", "blk.0.attn_v.weight", "f16"),
            _v("dflash", "blk.0.ffn_gate.weight", "f16"),
            _v("dflash", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets_a = compute_role_budgets(verdicts_a, rows_a, cfg)
        by_role_a = {b.model_role: b for b in budgets_a}
        # Trunk has 0 non-shared verdicts -> n_samples=0 -> weight=0.
        self.assertAlmostEqual(
            by_role_a["trunk"].weight, 0.0, places=9,
            msg="trunk weight = 1.0 * 0/8 = 0 (n_samples=0)",
        )
        # Dflash has 4 non-shared verdicts -> n_samples=4
        # -> confidence=0.5 -> weight = 2.0 * 0.5 = 1.0.
        self.assertAlmostEqual(
            by_role_a["dflash"].weight, 1.0, places=9,
            msg="dflash weight = 2.0 * 4/8 = 1.0",
        )

    def test_confidence_scaling_explicit(self) -> None:
        """Same as test 6 but with priority-1.0 roles so
        the weight directly reads as the confidence.
        n_samples=4 / n_target=8 -> weight = 0.5.
        """
        # Use only the trunk role (priority=1.0) so weight
        # = confidence.
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "blk.0.attn_k.weight", 1024, "f16", "attn_k"),
            _t("trunk", "blk.0.attn_v.weight", 1024, "f16", "attn_v"),
            _t("trunk", "blk.0.ffn_gate.weight", 1024, "f16", "ffn_gate"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "blk.0.attn_k.weight", "f16"),
            _v("trunk", "blk.0.attn_v.weight", "f16"),
            _v("trunk", "blk.0.ffn_gate.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        self.assertEqual(len(budgets), 1)
        self.assertAlmostEqual(
            budgets[0].weight, 0.5, places=9,
            msg="trunk weight = 1.0 * 4/8 = 0.5",
        )

    # ---- 7. Missing tensor_stats for a role: skipped -----

    def test_missing_tensor_stats_role_skipped(self) -> None:
        """If a role has policy verdicts but NO tensor_stats
        rows, the role is skipped. No spurious budget row
        is emitted.
        """
        verdicts = [
            # trunk has both policy + tensor_stats
            _v("trunk", "token_embd.weight", "f16"),
            # dflash has policy but no tensor_stats below
            _v("dflash", "token_embd.weight", "f16"),
        ]
        rows = [
            # Only trunk has tensor_stats rows.
            _t("trunk", "token_embd.weight", 1024, "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        by_role = {b.model_role: b for b in budgets}
        # Trunk: has tensor_stats -> emitted.
        self.assertIn("trunk", by_role)
        # Dflash: no tensor_stats -> skipped (E=0, S_t=0, M=0
        # would be NaN; the producer skips earlier).
        self.assertNotIn("dflash", by_role)

    # ---- 8. Empty policy entries: returns [] -----

    def test_empty_policy_entries(self) -> None:
        """No policy entries -> no verdicts -> no roles own
        shared tensors -> empty list. Should NOT error.
        """
        rows = [_t("trunk", "token_embd.weight", 1024, "f16")]
        self.assertEqual(compute_role_budgets([], rows, EnvelopeConfig()), [])
        # Also: empty tensor_stats with non-empty verdicts.
        verdicts = [_v("trunk", "token_embd.weight", "f16")]
        self.assertEqual(compute_role_budgets(verdicts, [], EnvelopeConfig()), [])
        # And: both empty.
        self.assertEqual(
            compute_role_budgets([], [], EnvelopeConfig()), [],
        )

    # ---- 9. JSON round-trip matches writer's load shape -----

    def test_json_roundtrip_writer_loadable(self) -> None:
        """The CLI's sidecar JSON parses back as the writer's
        load would parse it: schema + tensor_families (list)
        + role_budgets (list of {model_role, budget_bits,
        weight}). The writer's
        ``ts_unified_policy_load_json`` tolerates the
        absence of any key (it reads ``role_budgets`` as
        an additive field), so a minimal sidecar with
        only ``role_budgets`` is loadable.
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
            _t("dflash", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("dflash", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
            _v("dflash", "blk.0.attn_q.weight", "f16"),
            _v("dflash", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "sidecar.json"
            write_sidecar_json(out_path, budgets)
            with out_path.open() as f:
                loaded = json.load(f)
        # Writer-loadable shape.
        self.assertEqual(loaded["schema"], SIDECAR_SCHEMA)
        self.assertIsInstance(loaded["tensor_families"], list)
        self.assertIsInstance(loaded["role_budgets"], list)
        # Per-entry shape: {model_role, budget_bits, weight}.
        for entry in loaded["role_budgets"]:
            self.assertEqual(
                set(entry.keys()),
                {"model_role", "budget_bits", "weight"},
            )
            self.assertIsInstance(entry["model_role"], str)
            self.assertIsInstance(entry["budget_bits"], int)
            self.assertIsInstance(entry["weight"], (int, float))
        # Re-implement the writer's parse: tolerate missing
        # keys (the writer's load is additive; an older writer
        # without role_budgets support would just skip the
        # unknown key).
        def writer_load_compat(sidecar: dict) -> dict:
            """Mimic ts_unified_policy_load_json's contract.

            The C++ load reads tensor_families and
            role_budgets as additive fields. An absent
            role_budgets is fine (legacy pre-16.8 contract:
            plain worst-of). An absent tensor_families
            would error; we always write an empty list.
            """
            return {
                "entries":      sidecar.get("tensor_families", []),
                "role_budgets": sidecar.get("role_budgets", []),
            }
        parsed = writer_load_compat(loaded)
        self.assertEqual(len(parsed["role_budgets"]), 2)
        by_role = {e["model_role"]: e for e in parsed["role_budgets"]}
        self.assertIn("trunk", by_role)
        self.assertIn("dflash", by_role)

    # ---- 10. Source qtype when no verdict; resolved when present -----

    def test_verdict_dtype_overrides_source_for_non_shared(self) -> None:
        """When a non-shared tensor has a verdict, the
        verdict's qtype is used in S_t(r) (not the source
        qtype). A more-conservative verdict (more bits)
        increases S_t, which decreases the residual for
        the shared tensor. A more-aggressive verdict
        (fewer bits) does the opposite.
        """
        # Trunk: 1 non-shared tensor at f16 (16 bits/elem);
        # verdict downgrades it to q4_k (4 bits/elem).
        # Source-only: E = 2*1024*16 = 32768, S_t = 1024*16 = 16384,
        #              residual = 16384, budget = 16.
        # Verdict-downgraded: E = 32768, S_t = 1024*4 = 4096,
        #              residual = 28672, budget = clamp(28, 0, 16) = 16.
        #   (clamp caps at 16, the source's bit cost.)
        # Verdict-upgraded to f32 (32 bits): S_t = 1024*32 = 32768,
        #              residual = 0, budget = 0.
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
        ]
        cfg = EnvelopeConfig()

        # Verdict downgrades to q4_k: still clamped to 16.
        verdicts_down = [
            _v("trunk", "blk.0.attn_q.weight", "q4_k"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        budgets_down = compute_role_budgets(verdicts_down, rows, cfg)
        self.assertEqual(budgets_down[0].budget_bits_per_elem, 16)

        # Verdict upgrades to f32: residual = 0 -> budget = 0.
        verdicts_up = [
            _v("trunk", "blk.0.attn_q.weight", "f32"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            budgets_up = compute_role_budgets(verdicts_up, rows, cfg)
        # When the verdict-upgrade drives residual to 0, no
        # warning is emitted (residual is not negative). When
        # it drives residual below 0, a warning IS emitted.
        # Here residual = E - S_t = 32768 - (1024*32) = 32768 -
        # 32768 = 0, which is not < 0. So no warning, and
        # budget = clamp(0, 0, 16) = 0.
        self.assertEqual(budgets_up[0].budget_bits_per_elem, 0)
        self.assertFalse(
            any("negative residual" in str(w.message) for w in caught),
        )

    # ---- 11. Role that owns both token_embd and output -----

    def test_role_with_two_shared_tensors(self) -> None:
        """A role that owns BOTH token_embd.weight and
        output.weight emits two RoleBudgets. The CLI
        flatten picks the MIN per role (most
        conservative) for the writer's sidecar.
        """
        # Trunk: 1 non-shared tensor + token_embd + output.
        # All f16, 1024 elems.
        # E(trunk) = 3 * 1024 * 16 = 49152
        # S_t(trunk) = 1 * 1024 * 16 = 16384
        # M(trunk) = 0
        # Residual = 32768
        # token_embd budget = 32768 / 1024 = 32 -> clamp(16)
        # output budget = 32768 / 1024 = 32 -> clamp(16)
        # Both clamped to 16.
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
            _t("trunk", "output.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
            _v("trunk", "output.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        # Two per-(role, shared) entries.
        self.assertEqual(len(budgets), 2)
        for b in budgets:
            self.assertEqual(b.model_role, "trunk")
            self.assertEqual(b.budget_bits_per_elem, 16)
        # Sidecar flatten: one entry per role (min).
        flat = flatten_role_budgets_for_sidecar(budgets)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["model_role"], "trunk")
        self.assertEqual(flat[0]["budget_bits"], 16)

    def test_role_with_two_shared_tensors_asymmetric_budget(self) -> None:
        """When a role owns two shared tensors with
        different N(t) (e.g. token_embd is small,
        output is large), the budgets differ. The
        sidecar flatten picks the MIN (most
        conservative) for the role.
        """
        # Trunk: 0 non-shared + token_embd (small) + output (large).
        # token_embd: 1024 elems -> budget = E / 1024
        # output: 8192 elems -> budget = E / 8192
        # E = (1024 + 8192) * 16 = 147456
        # token_embd budget = 147456 / 1024 = 144 -> clamp(16)
        # output budget = 147456 / 8192 = 18 -> clamp(16)
        # Both clamp to 16 in this case. To get a meaningful
        # diff, use a tighter envelope.
        rows = [
            _t("trunk", "token_embd.weight", 1024, "f16"),
            _t("trunk", "output.weight", 8192, "f16"),
        ]
        verdicts = [
            _v("trunk", "token_embd.weight", "f16"),
            _v("trunk", "output.weight", "f16"),
        ]
        cfg = EnvelopeConfig(base_budget_fraction=0.5)
        budgets = compute_role_budgets(verdicts, rows, cfg)
        # E = 147456 * 0.5 = 73728
        # token_embd budget = 73728 / 1024 = 72 -> clamp(16)
        # output budget = 73728 / 8192 = 9 -> clamp(9)
        # Diff: token_embd=16, output=9.
        by_name = {b.model_role + ":" + str(idx): b
                   for idx, b in enumerate(budgets)}
        # Map by budget value.
        budgets_vals = sorted(
            b.budget_bits_per_elem for b in budgets
        )
        self.assertEqual(budgets_vals, [9, 16])
        # Flatten: min = 9.
        flat = flatten_role_budgets_for_sidecar(budgets)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["budget_bits"], 9)

    # ---- 12. Unknown dtype / NULL n_elements -----

    def test_unknown_dtype_skipped(self) -> None:
        """Rows with unknown dtypes (e.g. 'mxfp4' which
        is not in DTYPE_BITS) are skipped, not poisoned.
        The S_t / E sums exclude those rows entirely.
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "mxfp4", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "mxfp4"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        # E = 1 * 1024 * 16 = 16384 (unknown dtype skipped)
        # S_t = 0 (the unknown-dtype row is skipped)
        # Residual = 16384; budget = 16.
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0].budget_bits_per_elem, 16)

    def test_null_n_elements_skipped(self) -> None:
        """A tensor_stats row with n_elements=None is
        skipped, not poisoned. The S_t / E sums exclude
        those rows.
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight", None, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        self.assertEqual(len(budgets), 1)
        # n_elements=None for the non-shared row -> skipped.
        # E = 1 * 1024 * 16 = 16384
        # S_t = 0
        # Residual = 16384; budget = 16.
        self.assertEqual(budgets[0].budget_bits_per_elem, 16)

    # ---- 13. Shared tensor with missing n_elements -----

    def test_shared_tensor_missing_n_elements(self) -> None:
        """If the shared tensor itself has no n_elements
        in tensor_stats, that (role, shared) entry is
        skipped (cannot divide by zero).
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            # token_embd.weight has n_elements=None.
            _t("trunk", "token_embd.weight", None, "f16"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        # Skipped: cannot divide by 0.
        self.assertEqual(budgets, [])

    # ---- 14. Roles without shared-tensor ownership are skipped -----

    def test_role_without_shared_ownership_skipped(self) -> None:
        """A role that has tensor_stats + verdicts but no
        shared-tensor verdict is skipped entirely. The
        L5 family budget still governs its requant loop.
        """
        rows = [
            # dflash has token_embd (shared).
            _t("dflash", "token_embd.weight", 1024, "f16"),
            # vision_tower has only mmproj tensors (no shared).
            _t("vision_tower", "v.embed.weight", 1024, "f16", "vision_tower"),
            # mm_projector has only mmproj tensors (no shared).
            _t("mm_projector", "mm.proj.0.weight", 1024, "f16", "mm_projector"),
        ]
        verdicts = [
            _v("dflash", "token_embd.weight", "f16"),
            _v("vision_tower", "v.embed.weight", "f16"),
            _v("mm_projector", "mm.proj.0.weight", "f16"),
        ]
        cfg = EnvelopeConfig()
        budgets = compute_role_budgets(verdicts, rows, cfg)
        # Only dflash owns a shared tensor.
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0].model_role, "dflash")
        flat = flatten_role_budgets_for_sidecar(budgets)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["model_role"], "dflash")

    # ---- 15. mmproj prefix matching is exact (leading dot) -----

    def test_mmproj_prefix_matching_is_exact(self) -> None:
        """The mmproj prefix match is a leading-dot literal.
        ``v.attn.0.weight`` matches; ``vattn.0.weight`` does
        not. This mirrors the model side's v/a/mm namespacing.
        """
        rows = [
            _t("trunk", "blk.0.attn_q.weight", 1024, "f16", "attn_q"),
            _t("trunk", "token_embd.weight", 1024, "f16"),
            # Names that should NOT be counted as mmproj:
            _t("trunk", "vattn.0.weight", 1024, "f16", "attn_q"),
            _t("trunk", "aattn.0.weight", 1024, "f16", "attn_q"),
            _t("trunk", "mmproj.0.weight", 1024, "f16", "attn_q"),
            # Names that SHOULD be counted:
            _t("trunk", "v.embed.weight", 1024, "f16", "vision_tower"),
            _t("trunk", "a.audio.0.weight", 1024, "f16", "audio_tower"),
            _t("trunk", "mm.proj.0.weight", 1024, "f16", "mm_projector"),
        ]
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),
        ]
        # The M(r) sum should be 3 * 1024 * 16 = 49152 (the
        # three correct mmproj tensors). The three "false
        # positive" names (vattn, aattn, mmproj) are not
        # counted.
        cfg = EnvelopeConfig()
        m_raw = embedding_budget._mmproj_footprint_bits(
            rows, "trunk",
        )
        self.assertEqual(m_raw, 3 * 1024 * 16)
        # And the producer still emits a budget for trunk.
        budgets = compute_role_budgets(verdicts, rows, cfg)
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0].model_role, "trunk")

    # ---- 16. CLI: --role-priority parser -----

    def test_role_priority_parser(self) -> None:
        """--role-priority accepts multiple role=value
        entries; rejects malformed ones.
        """
        out = _parse_role_priority_overrides([
            "dflash=3.0",
            "trunk=0.5",
        ])
        self.assertEqual(out, {"dflash": 3.0, "trunk": 0.5})
        # Malformed: no '='
        with self.assertRaises(ValueError) as cm:
            _parse_role_priority_overrides(["dflash"])
        self.assertIn("role=priority", str(cm.exception))
        # Malformed: empty role
        with self.assertRaises(ValueError) as cm:
            _parse_role_priority_overrides(["=3.0"])
        self.assertIn("empty role", str(cm.exception))
        # Malformed: bad priority
        with self.assertRaises(ValueError) as cm:
            _parse_role_priority_overrides(["dflash=abc"])
        self.assertIn("bad priority", str(cm.exception))

    # ---- 17. CLI: --budget-fraction end-to-end via TesseraDB -----

    def test_cli_end_to_end_with_db(self) -> None:
        """End-to-end CLI: seed a fresh DuckDB with
        tensor_stats rows, invoke main() via
        subprocess, verify the sidecar is loadable.
        """
        import duckdb
        # Mirror the C++ schema's tensor_stats table.
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tessera.duckdb"
            con = duckdb.connect(str(db_path))
            try:
                con.execute(
                    """
                    CREATE TABLE tensor_stats (
                        model_hash    TEXT NOT NULL,
                        model_role    TEXT NOT NULL DEFAULT 'trunk',
                        name          TEXT NOT NULL,
                        family        TEXT,
                        layer_depth   INTEGER,
                        out_dim       BIGINT,
                        in_dim        BIGINT,
                        n_elements    BIGINT,
                        dtype         TEXT,
                        PRIMARY KEY (model_hash, model_role, name)
                    )
                    """
                )
                con.executemany(
                    """
                    INSERT INTO tensor_stats
                        (model_hash, model_role, name, n_elements, dtype)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        ("m", "trunk",  "blk.0.attn_q.weight", 1024, "f16"),
                        ("m", "trunk",  "token_embd.weight",   1024, "f16"),
                        ("m", "dflash", "blk.0.attn_q.weight", 1024, "f16"),
                        ("m", "dflash", "token_embd.weight",   1024, "f16"),
                    ],
                )
            finally:
                con.close()
            out_path = Path(td) / "sidecar.json"
            import subprocess
            cmd = [
                sys.executable,
                "-m", "tools.tessera.embedding_budget",
                "--db", str(db_path),
                "--model-hash", "m",
                "--output", str(out_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(THIS_DIR.parent.parent),
            )
            self.assertEqual(
                result.returncode, 0,
                f"CLI failed: stdout={result.stdout!r} "
                f"stderr={result.stderr!r}",
            )
            with out_path.open() as f:
                sidecar = json.load(f)
            self.assertEqual(sidecar["schema"], SIDECAR_SCHEMA)
            self.assertEqual(len(sidecar["role_budgets"]), 2)
            by_role = {
                e["model_role"]: e for e in sidecar["role_budgets"]
            }
            self.assertIn("trunk", by_role)
            self.assertIn("dflash", by_role)

    def test_cli_zero_fraction_emits_empty(self) -> None:
        """CLI: --budget-fraction 0 emits an empty
        role_budgets (the opt-out case).
        """
        import duckdb
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tessera.duckdb"
            con = duckdb.connect(str(db_path))
            try:
                con.execute(
                    """
                    CREATE TABLE tensor_stats (
                        model_hash    TEXT NOT NULL,
                        model_role    TEXT NOT NULL DEFAULT 'trunk',
                        name          TEXT NOT NULL,
                        n_elements    BIGINT,
                        dtype         TEXT,
                        PRIMARY KEY (model_hash, model_role, name)
                    )
                    """
                )
                con.executemany(
                    "INSERT INTO tensor_stats VALUES (?, ?, ?, ?, ?)",
                    [
                        ("m", "trunk",  "blk.0.attn_q.weight", 1024, "f16"),
                        ("m", "trunk",  "token_embd.weight",   1024, "f16"),
                    ],
                )
            finally:
                con.close()
            out_path = Path(td) / "sidecar.json"
            import subprocess
            cmd = [
                sys.executable,
                "-m", "tools.tessera.embedding_budget",
                "--db", str(db_path),
                "--model-hash", "m",
                "--output", str(out_path),
                "--budget-fraction", "0",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(THIS_DIR.parent.parent),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with out_path.open() as f:
                sidecar = json.load(f)
            self.assertEqual(sidecar["role_budgets"], [])


class TestUnifiedCalibrateIntegration(unittest.TestCase):
    """The unified_calibrate.py glue layer that calls
    ``compute_role_budgets`` and emits the role_budgets
    sidecar. The integration is intentionally thin: the M2
    producer is unit-tested in TestPureProducer; these
    tests pin the glue (DB read, dict extraction, sidecar
    write) so a future refactor of the glue cannot break
    the contract.

    The end-to-end unified_calibrate test
    (test_unified_calibrate._test_end_to_end) is slow
    (per_tensor_calibrate subprocess); this test exercises
    the glue directly so it runs in milliseconds.
    """

    def _fresh_db(self, td: Path) -> Path:
        """Create a fresh DuckDB with a minimal tensor_stats
        table seeded with one model's rows. Returns the
        DB path.
        """
        import duckdb
        db_path = td / "tessera.duckdb"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE tensor_stats (
                    model_hash    TEXT NOT NULL,
                    model_role    TEXT NOT NULL DEFAULT 'trunk',
                    name          TEXT NOT NULL,
                    n_elements    BIGINT,
                    dtype         TEXT,
                    PRIMARY KEY (model_hash, model_role, name)
                )
                """
            )
            con.executemany(
                "INSERT INTO tensor_stats VALUES (?, ?, ?, ?, ?)",
                [
                    ("m", "trunk",  "blk.0.attn_q.weight", 1024, "f16"),
                    ("t", "trunk",  "token_embd.weight",   1024, "f16"),
                    ("m", "dflash", "blk.0.attn_q.weight", 1024, "f16"),
                    ("m", "dflash", "token_embd.weight",   1024, "f16"),
                ],
            )
        finally:
            con.close()
        return db_path

    def test_build_role_budgets_with_db(self) -> None:
        """_build_role_budgets reads tensor_stats from the
        DB and emits the writer-loadable sidecar shape.
        """
        from unified_calibrate import _build_role_budgets
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = self._fresh_db(td_path)
            # Empty unified dict (the glue does not depend
            # on the unified content in the M2 scope; the
            # per-fitness verdict override is a follow-on).
            unified: dict = {}
            role_budgets = _build_role_budgets(
                db_path, unified, budget_fraction=1.0,
            )
            # The role_budgets is a list of {model_role,
            # budget_bits, weight} dicts.
            self.assertIsInstance(role_budgets, list)
            for entry in role_budgets:
                self.assertEqual(
                    set(entry.keys()),
                    {"model_role", "budget_bits", "weight"},
                )
            # Two roles own shared tensors: trunk and dflash.
            by_role = {e["model_role"]: e for e in role_budgets}
            self.assertIn("trunk", by_role)
            self.assertIn("dflash", by_role)

    def test_build_role_budgets_zero_fraction(self) -> None:
        """budget_fraction=0 opts out of the size envelope;
        the glue returns [].
        """
        from unified_calibrate import _build_role_budgets
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = self._fresh_db(td_path)
            self.assertEqual(
                _build_role_budgets(
                    db_path, {}, budget_fraction=0.0,
                ),
                [],
            )

    def test_build_role_budgets_missing_db(self) -> None:
        """When --db is not provided (or the DB is missing),
        the integration is a no-op. The CLI gates this
        with the args.no_embedding_budget / args.db
        check; the glue itself returns [] when tensor_stats
        is empty.
        """
        from unified_calibrate import _build_role_budgets
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # DB does not exist.
            self.assertEqual(
                _build_role_budgets(
                    td_path / "missing.duckdb", {}, 1.0,
                ),
                [],
            )
            # DB exists but has no tensor_stats table.
            import duckdb
            db_path = td_path / "no_ts.duckdb"
            con = duckdb.connect(str(db_path))
            try:
                con.execute(
                    "CREATE TABLE unrelated (x INTEGER)"
                )
            finally:
                con.close()
            self.assertEqual(
                _build_role_budgets(db_path, {}, 1.0),
                [],
            )

    def test_load_tensor_stats_from_db(self) -> None:
        """_load_tensor_stats_from_db returns the row list
        that drives the producer; missing DB / missing
        table returns [].
        """
        from unified_calibrate import _load_tensor_stats_from_db
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = self._fresh_db(td_path)
            rows = _load_tensor_stats_from_db(db_path)
            self.assertEqual(len(rows), 4)
            # Missing DB.
            self.assertEqual(
                _load_tensor_stats_from_db(td_path / "missing.duckdb"),
                [],
            )

    def test_extract_policy_entries_from_unified(self) -> None:
        """_extract_policy_entries_from_unified returns one
        policy_entry per tensor_stats row, each with the
        source dtype as the verdict dtype.
        """
        from unified_calibrate import _extract_policy_entries_from_unified
        rows = [
            {"model_role": "trunk",  "name": "blk.0.attn_q.weight",
             "n_elements": 1024, "dtype": "f16"},
            {"model_role": "trunk",  "name": "token_embd.weight",
             "n_elements": 1024, "dtype": "f16"},
            {"model_role": "dflash", "name": "token_embd.weight",
             "n_elements": 1024, "dtype": "f16"},
            # Incomplete row (missing dtype): skipped.
            {"model_role": "trunk",  "name": "broken",
             "n_elements": 1024},
        ]
        policy_entries = _extract_policy_entries_from_unified({}, rows)
        self.assertEqual(len(policy_entries), 3)
        for entry in policy_entries:
            self.assertIn("model_role", entry)
            self.assertIn("name", entry)
            self.assertIn("dtype", entry)
        by_role_name = {
            (e["model_role"], e["name"]): e for e in policy_entries
        }
        self.assertEqual(
            by_role_name[("trunk", "token_embd.weight")]["dtype"],
            "f16",
        )


class TestHelpers(unittest.TestCase):
    """Unit tests for the producer's helper functions."""

    def test_source_footprint_bits(self) -> None:
        rows = [
            _t("trunk", "a", 100, "f16"),
            _t("trunk", "b", 200, "q4_k"),
            _t("dflash", "a", 100, "f16"),  # different role, skipped
            _t("trunk", "c", None, "f16"),   # NULL n_elements, skipped
            _t("trunk", "d", 50, "mxfp4"),   # unknown dtype, skipped
        ]
        self.assertEqual(
            _source_footprint_bits(rows, "trunk"),
            100 * 16 + 200 * 4,
        )
        self.assertEqual(_source_footprint_bits(rows, "dflash"), 100 * 16)

    def test_mmproj_footprint_bits(self) -> None:
        rows = [
            _t("trunk", "v.embed.weight", 100, "f16"),
            _t("trunk", "a.audio.0.weight", 200, "f16"),
            _t("trunk", "mm.proj.0.weight", 300, "q4_k"),
            _t("trunk", "blk.0.attn_q.weight", 1000, "f16"),  # not mmproj
            _t("dflash", "v.embed.weight", 100, "f16"),  # wrong role
        ]
        self.assertEqual(
            _mmproj_footprint_bits(rows, "trunk"),
            100 * 16 + 200 * 16 + 300 * 4,
        )
        self.assertEqual(_mmproj_footprint_bits(rows, "dflash"), 100 * 16)

    def test_priority_for_role(self) -> None:
        # Default table.
        self.assertEqual(
            _priority_for_role("dflash", EnvelopeConfig()),
            DEFAULT_ROLE_PRIORITIES["dflash"],
        )
        self.assertEqual(
            _priority_for_role("trunk", EnvelopeConfig()),
            DEFAULT_ROLE_PRIORITIES["trunk"],
        )
        # Override wins.
        self.assertEqual(
            _priority_for_role(
                "dflash", EnvelopeConfig(role_priorities={"dflash": 5.0}),
            ),
            5.0,
        )
        # Unknown role: 1.0 fallback.
        self.assertEqual(
            _priority_for_role("totally_new_role", EnvelopeConfig()),
            1.0,
        )

    def test_n_samples_for_role_excludes_shared(self) -> None:
        """n_samples counts only NON-shared verdicts for the
        role. Shared-tensor verdicts do not count toward the
        role's confidence (they are the subject of the
        budget, not a calibration input).
        """
        verdicts = [
            _v("trunk", "blk.0.attn_q.weight", "f16"),
            _v("trunk", "blk.0.attn_k.weight", "f16"),
            _v("trunk", "token_embd.weight", "f16"),  # shared, excluded
            _v("dflash", "blk.0.attn_q.weight", "f16"),
        ]
        self.assertEqual(_n_samples_for_role(verdicts, "trunk"), 2)
        self.assertEqual(_n_samples_for_role(verdicts, "dflash"), 1)
        self.assertEqual(_n_samples_for_role(verdicts, "missing"), 0)

    def test_verdict_dtype_resolution(self) -> None:
        verdicts = {("trunk", "blk.0.attn_q.weight"): "q4_k"}
        # Verdict present -> use verdict.
        self.assertEqual(
            _verdict_dtype(verdicts, "trunk", "blk.0.attn_q.weight", "f16"),
            "q4_k",
        )
        # Verdict absent -> use fallback (source dtype).
        self.assertEqual(
            _verdict_dtype(verdicts, "trunk", "blk.0.attn_k.weight", "f16"),
            "f16",
        )
        # Verdict absent, no fallback -> None.
        self.assertIsNone(
            _verdict_dtype(verdicts, "trunk", "blk.0.attn_v.weight", None),
        )

    def test_find_n_elements(self) -> None:
        rows = [
            _t("trunk", "token_embd.weight", 12345, "f16"),
            _t("trunk", "output.weight", None, "f16"),
        ]
        self.assertEqual(
            _find_n_elements(rows, "trunk", "token_embd.weight"),
            12345,
        )
        self.assertIsNone(
            _find_n_elements(rows, "trunk", "output.weight"),
        )
        self.assertIsNone(
            _find_n_elements(rows, "trunk", "missing"),
        )


class TestDefaultTable(unittest.TestCase):
    """The default priority table is the architect's contract;
    pin its values so a future change is intentional.
    """

    def test_default_role_priorities(self) -> None:
        self.assertEqual(
            DEFAULT_ROLE_PRIORITIES,
            {
                "trunk":        1.0,
                "dflash":       2.0,
                "vision_tower": 1.0,
                "audio_tower":  1.0,
                "mm_projector": 1.0,
            },
        )

    def test_shared_owning_roles(self) -> None:
        # The set the M0a surface recognized (with the M2
        # addition of vision_tower / audio_tower / mm_projector).
        self.assertEqual(
            set(SHARED_OWNING_ROLES),
            {"trunk", "dflash", "vision_tower",
             "audio_tower", "mm_projector"},
        )

    def test_shared_tensor_names(self) -> None:
        self.assertEqual(
            set(SHARED_TENSOR_NAMES),
            {"token_embd.weight", "output.weight"},
        )

    def test_mmproj_name_prefixes(self) -> None:
        self.assertEqual(
            set(MMPROJ_NAME_PREFIXES),
            {"v.", "a.", "mm."},
        )

    def test_budget_clamp_max(self) -> None:
        # 16 = f16 bit cost; the upper bound of a reasonable
        # budget recommendation.
        self.assertEqual(BUDGET_CLAMP_MAX, 16)

    def test_dtype_bits_match_l5_retune(self) -> None:
        # The M2 producer reuses l5_retune's DTYPE_BITS map.
        # A change on one side without the other would be a
        # silent inconsistency; this test pins the reuse.
        from l5_retune import DTYPE_BITS as L5_DTYPE_BITS
        self.assertIs(DTYPE_BITS, L5_DTYPE_BITS)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
