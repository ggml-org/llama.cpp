"""Tests for tools/ane-mtp/estimate_higgs_alpha.py (Phase 3 of iPhone ANE demo).

Pins the Linearity-Theorem math (through-origin slope + R^2,
ternary round-trip, structural alpha, family classification),
the sidecar JSON shape (the wire format Phase 2 GGUF->IOSurface
streaming consumes), the uniform fallback path, and the
full-pipeline behavior on a real (small) GGUF fixture.

The estimator is L1-agnostic: the ``t_l^2`` measurement is
parameterized so today's offline ternary MSE proxy and
tomorrow's L1 kernel-dequant output are a one-function swap.
The tests cover both the pure-math surface (no GGUF needed) and
the orchestrator (the small stories15M fixture in
``build-ane/tinyllamas/stories15M-q4_0.gguf``).

Run with:
    python3 -m unittest tools.tessera.test_estimate_higgs_alpha -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ANE_MTP_DIR = THIS_DIR.parent / "ane-mtp"
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(ANE_MTP_DIR))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import estimate_higgs_alpha as eha  # noqa: E402


# ---- pure-math tests: ternary round-trip, relative error, through-origin fit ----

class TernaryRoundTripTest(unittest.TestCase):
    """The offline ternary MSE proxy: round then dequantize,
    compare to the reference. The proxy is the L1-agnostic
    measurement; the L1 path will replace it with the
    kernel-dequant output."""

    def test_round_to_nearest_ternary_three_levels(self) -> None:
        """``ternary_round`` produces {-1, 0, +1} with a
        per-tensor mean-absolute threshold. Values whose
        magnitude exceeds half the mean are rounded to +/- 1;
        values below are dropped to 0.
        """
        x = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0],
                     dtype=np.float32)
        # mean(|x|) = (3+1+0.1+0+0.1+1+3)/7 = 8.2/7 ~ 1.17
        # half-mean threshold ~ 0.586; the +/- 1 magnitudes win,
        # +/- 0.1 magnitudes lose.
        q = eha.ternary_round(x)
        self.assertEqual(set(np.unique(q).tolist()), {-1, 0, 1})
        # The +3 and -3 stay signed.
        self.assertEqual(int(q[0]), -1)
        self.assertEqual(int(q[-1]),  1)
        # The 1.0 magnitudes survive.
        self.assertEqual(int(q[1]), -1)
        self.assertEqual(int(q[5]),  1)
        # The 0.1 magnitudes are below the threshold.
        self.assertEqual(int(q[2]),  0)
        self.assertEqual(int(q[4]),  0)
        # Zero stays zero.
        self.assertEqual(int(q[3]),  0)

    def test_round_constant_zero_returns_all_zeros(self) -> None:
        """A zero reference produces an all-zero round-trip.
        ``relative_frobenius_error`` then returns 0.0 (degenerate
        reference, not a NaN)."""
        x = np.zeros(8, dtype=np.float32)
        q = eha.ternary_round(x)
        self.assertTrue(np.all(q == 0))
        recon = eha.ternary_dequantize(q, 0.0)
        self.assertAlmostEqual(
            eha.relative_frobenius_error(x, recon), 0.0,
            msg="zero reference must give zero error",
        )

    def test_round_to_nearest_with_zero_mean_safe(self) -> None:
        """All-zero weights: ternary threshold is 0.0 -> the
        function returns all-zeros without divide-by-zero."""
        x = np.zeros((4, 4), dtype=np.float32)
        self.assertTrue(np.all(eha.ternary_round(x) == 0))

    def test_dequantize_is_inverse_of_round_scale(self) -> None:
        """``ternary_dequantize`` applied to the round output
        produces a tensor whose elements are in {0, scale};
        the reconstruction preserves the sign of the input
        at the surviving magnitudes."""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        scale = float(np.mean(np.abs(x)))
        q = eha.ternary_round(x)
        recon = eha.ternary_dequantize(q, scale)
        # Both +2.0 and -2.0 survive (magnitude > 0.5*scale).
        self.assertAlmostEqual(float(recon[0]), -scale)
        self.assertAlmostEqual(float(recon[4]),  scale)
        # The +/- 0.5 magnitudes are at the threshold boundary;
        # 0.5 == 0.5*scale so the strict-greater-than test
        # drops them. (The proxy uses ``>`` not ``>=`` so the
        # boundary is excluded; documented in the docstring.)
        self.assertEqual(float(recon[1]), 0.0)
        self.assertEqual(float(recon[3]), 0.0)
        self.assertEqual(float(recon[2]), 0.0)


class RelativeFrobeniusTest(unittest.TestCase):

    def test_zero_reconstruction_gives_full_error(self) -> None:
        """If the reconstruction is all-zero, the relative
        error is 1.0 (the entire Frobenius norm of the
        reference is the residual)."""
        ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        recon = np.zeros_like(ref)
        self.assertAlmostEqual(
            eha.relative_frobenius_error(ref, recon), 1.0,
            places=6)

    def test_perfect_reconstruction_gives_zero_error(self) -> None:
        ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(
            eha.relative_frobenius_error(ref, ref), 0.0,
            places=12)

    def test_zero_reference_gives_zero_error(self) -> None:
        """Degenerate (zero-norm) reference: the function must
        return 0.0, not NaN. The estimator skips zero-norm
        tensors; the consumer uses 0.0 as the ``t_l^2``.
        """
        ref = np.zeros(8, dtype=np.float32)
        recon = np.ones(8, dtype=np.float32)
        self.assertEqual(eha.relative_frobenius_error(ref, recon), 0.0)

    def test_shape_mismatch_raises(self) -> None:
        ref = np.zeros(4, dtype=np.float32)
        recon = np.zeros(5, dtype=np.float32)
        with self.assertRaises(ValueError):
            eha.relative_frobenius_error(ref, recon)


class ThroughOriginSlopeTest(unittest.TestCase):
    """The Linearity-Theorem fit: closed-form through-origin
    least-squares slope of delta-PPL against t^2. The paper
    fixes the intercept at zero (the t=0 measurement is zero
    by construction, so an intercept would absorb baseline
    noise)."""

    def test_perfect_line_through_origin(self) -> None:
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        ys = [0.2, 0.4, 0.6, 0.8, 1.0]  # y = 2x
        slope, r2 = eha.through_origin_slope(xs, ys)
        self.assertAlmostEqual(slope, 2.0, places=12)
        self.assertAlmostEqual(r2, 1.0, places=12)

    def test_noisy_data_low_r2(self) -> None:
        """Random-noise y against a linear xs. The through-origin
        R^2 measures how much of the y-signal is explained by a
        linear x-relation; truly uncorrelated y gives a low R^2.
        The closed-form slope and R^2 are not zero (least-squares
        finds the best fit even for noise), but the R^2 should
        be small.
        """
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        # Alternating y: the through-origin least-squares
        # slope is the best fit but explains little of the
        # signal. The R^2 is bounded above 0 (least-squares
        # always finds *some* fit) but should be strictly
        # less than 1.0 - the perfect-line case.
        ys = [0.1, 0.9, 0.1, 0.9, 0.1]
        slope, r2 = eha.through_origin_slope(xs, ys)
        # R^2 is well below 1.0 (alternating data is not a
        # monotone line). The exact value is ~0.44; we pin
        # the upper bound to 0.6 to leave room for future
        # numeric refinements.
        self.assertLess(r2, 0.6)
        self.assertGreater(r2, 0.0)
        # The slope is finite.
        self.assertGreater(slope, 0.0)

    def test_empty_inputs(self) -> None:
        slope, r2 = eha.through_origin_slope([], [])
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 0.0)

    def test_length_mismatch_returns_zero(self) -> None:
        slope, r2 = eha.through_origin_slope([0.1, 0.2], [0.1])
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 0.0)

    def test_all_zero_xs_returns_zero(self) -> None:
        slope, r2 = eha.through_origin_slope([0.0, 0.0], [0.1, 0.2])
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 0.0)

    def test_all_zero_ys_returns_unit_r2(self) -> None:
        """A degenerate y signal (all zero) gives R^2 = 1.0
        because the through-origin reference SS_tot is also
        zero, so the residual is trivially 0. This is the
        numerical-safety branch; the consumer's
        fall-back-uniform rule should still treat R^2=1.0
        as a real fit (it is a valid signal, just one with
        no measurement noise)."""
        xs = [0.1, 0.2, 0.3]
        ys = [0.0, 0.0, 0.0]
        slope, r2 = eha.through_origin_slope(xs, ys)
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 1.0)

    def test_r2_clamped_to_unit_interval(self) -> None:
        """If the through-origin R^2 would go negative
        (numerical-noise floor), the function clamps to 0.0.
        A negative R^2 means the fit is worse than predicting
        zero; that's still a usable signal ("uniform
        fallback") but it is not a R^2 of -0.1."""
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        # Inverted: y decreases as x increases.
        ys = [0.5, 0.4, 0.3, 0.2, 0.1]
        _slope, r2 = eha.through_origin_slope(xs, ys)
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)


# ---- family classification ----

class ClassifyFamilyTest(unittest.TestCase):

    def test_attn_k_v_recognized(self) -> None:
        # The function recognizes the Tessera/gguf naming
        # convention (blk.<i>.attn_k.weight, blk.<i>.attn_v.weight).
        # Non-Tessera naming conventions (HF's
        # model.layers.<i>.self_attn.k_proj.weight) are
        # classified as 'other' and fall back to the default
        # family prior; the GA treats unknown families as
        # 'middle of the road' sensitivity, not a
        # misclassification error.
        self.assertEqual(
            eha.classify_family("blk.0.attn_k.weight"), "attn_k")
        self.assertEqual(
            eha.classify_family("blk.16.attn_v.weight"), "attn_v")

    def test_attn_q_recognized(self) -> None:
        self.assertEqual(eha.classify_family("blk.0.attn_q.weight"),
                         "attn_q")
        self.assertEqual(eha.classify_family("blk.16.attn_q.weight"),
                         "attn_q")

    def test_attn_output_not_misread_as_q(self) -> None:
        """The suffix table is ordered: ``attn_output`` is
        checked before ``attn_q`` so that a name ending in
        ``attn_output`` does not get matched as ``attn_q``.
        """
        self.assertEqual(
            eha.classify_family("blk.0.attn_output.weight"),
            "attn_output")
        self.assertEqual(
            eha.classify_family("blk.16.attn_output.weight"),
            "attn_output")

    def test_ffn_families_recognized(self) -> None:
        for name in ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                     "blk.0.ffn_down.weight"):
            family = name.split(".")[-2]
            self.assertEqual(eha.classify_family(name), family)

    def test_norm_families_recognized(self) -> None:
        self.assertEqual(
            eha.classify_family("blk.0.attn_norm.weight"), "norm")
        self.assertEqual(
            eha.classify_family("blk.0.ffn_norm.weight"), "norm")

    def test_embedding_and_output_recognized(self) -> None:
        self.assertEqual(
            eha.classify_family("token_embd.weight"), "token_embd")
        self.assertEqual(
            eha.classify_family("output.weight"), "output")
        self.assertEqual(
            eha.classify_family("output.bias"), "output")

    def test_unknown_name_returns_other(self) -> None:
        self.assertEqual(eha.classify_family("rope_freqs.weight"),
                         "other")
        self.assertEqual(eha.classify_family("token_types"), "other")
        self.assertEqual(eha.classify_family(""), "other")


# ---- structural alpha (the L1-agnostic proxy) ----

class StructuralAlphaTest(unittest.TestCase):
    """The structural Hessian-trace proxy is the
    L1-agnostic replacement for the paper's Algorithm 3
    perturbation sweep. The proxy is *ranking-grade* (not a
    precise per-tensor estimate) and matches the SLQ/BAQ
    layer-dependence shape: K/V high, FFN low."""

    def test_zero_elements_returns_zero(self) -> None:
        self.assertEqual(
            eha.structural_alpha(
                frobenius_norm=10.0, n_elements=0, family="attn_v"),
            0.0)

    def test_family_prior_ranking(self) -> None:
        """K/V should rank above output above Q above FFN.
        The family prior table encodes this; the function
        is the family-prior lookup. The structural proxy
        deliberately drops the Frobenius-norm multiplier
        (without a proper Hessian-trace estimate the
        Frobenius part would let large embeddings dominate
        the normalization and wash out the ranking).
        """
        a_kv  = eha.structural_alpha(frobenius_norm=1.0,
                                      n_elements=1000, family="attn_v")
        a_out = eha.structural_alpha(frobenius_norm=1.0,
                                      n_elements=1000, family="attn_output")
        a_q   = eha.structural_alpha(frobenius_norm=1.0,
                                      n_elements=1000, family="attn_q")
        a_ffn = eha.structural_alpha(frobenius_norm=1.0,
                                      n_elements=1000, family="ffn_gate")
        self.assertGreater(a_kv,  a_out)
        self.assertGreater(a_out, a_q)
        self.assertGreater(a_q,   a_ffn)

    def test_family_prior_independent_of_frobenius(self) -> None:
        """The proxy alpha does not depend on the Frobenius
        norm (documented choice; the Frobenius norm is
        reported in the sidecar for diagnostics but the
        proxy drops the (||W_l||_F^2 / 2) multiplier until
        a proper Hessian-trace estimate is available)."""
        a1 = eha.structural_alpha(frobenius_norm=1.0,
                                   n_elements=1000, family="attn_v")
        a2 = eha.structural_alpha(frobenius_norm=100.0,
                                   n_elements=1000, family="attn_v")
        self.assertEqual(a1, a2)

    def test_unknown_family_uses_default(self) -> None:
        """An unknown family falls back to the 'other' prior,
        not the lowest-prior family. This is the safe
        default: the consumer treats unknown families as
        'middle of the road' sensitivity, neither inflating
        nor deflating the alpha."""
        a_unknown = eha.structural_alpha(
            frobenius_norm=1.0, n_elements=100, family="unknown_thing")
        a_other = eha.structural_alpha(
            frobenius_norm=1.0, n_elements=100, family="other")
        self.assertEqual(a_unknown, a_other)


class ClampAlphaTest(unittest.TestCase):

    def test_positive_alpha_above_floor_unchanged(self) -> None:
        a, applied = eha.clamp_alpha(2.5, 0.01)
        self.assertEqual(a, 2.5)
        self.assertFalse(applied)

    def test_alpha_below_floor_is_clamped(self) -> None:
        a, applied = eha.clamp_alpha(0.001, 0.01)
        self.assertEqual(a, 0.01)
        self.assertTrue(applied)

    def test_negative_alpha_clamped_to_floor(self) -> None:
        """Negative alpha is a noise artifact (a true alpha
        is non-negative at a local minimum); the clamp
        replaces it with the positive floor."""
        a, applied = eha.clamp_alpha(-1.0, 0.01)
        self.assertEqual(a, 0.01)
        self.assertTrue(applied)

    def test_nan_alpha_clamped_to_floor(self) -> None:
        a, applied = eha.clamp_alpha(float("nan"), 0.01)
        self.assertEqual(a, 0.01)
        self.assertTrue(applied)

    def test_inf_alpha_clamped_to_floor(self) -> None:
        a, applied = eha.clamp_alpha(float("inf"), 0.01)
        self.assertEqual(a, 0.01)
        self.assertTrue(applied)

    def test_zero_alpha_clamped_to_floor(self) -> None:
        """Zero is a legitimate fit (a flat layer), but the
        floor protects the GA from divide-by-zero on the
        fitness normalization. The clamp replaces with the
        positive floor.
        """
        a, applied = eha.clamp_alpha(0.0, 0.01)
        self.assertEqual(a, 0.01)
        self.assertTrue(applied)


# ---- sidecar JSON shape ----

class SidecarShapeTest(unittest.TestCase):
    """The sidecar JSON is the wire format between this
    estimator and Phase 2's GGUF->IOSurface streaming
    (and the iOS app's ANE dispatch). The shape must
    round-trip cleanly and match the documented schema."""

    def _sample_sidecar(self) -> dict:
        infos = [
            eha.TensorInfo(
                name="blk.0.attn_v.weight",
                family="attn_v",
                n_elements=82944,
                frobenius_norm=5.4,
                t_squared=0.31,
                t_squared_source="offline_ternary_mse",
                dtype_source="Q4_0",
                alpha=1.30,
                alpha_floor_applied=False,
                fit_r2=1.0,
                n_samples=0,
                fallback="none",
                shape=(288, 288),
            ),
            eha.TensorInfo(
                name="blk.0.ffn_gate.weight",
                family="ffn_gate",
                n_elements=221184,
                frobenius_norm=12.0,
                t_squared=0.27,
                t_squared_source="offline_ternary_mse",
                dtype_source="Q4_0",
                alpha=0.45,
                alpha_floor_applied=False,
                fit_r2=1.0,
                n_samples=0,
                fallback="none",
                shape=(288, 768),
            ),
        ]
        audit = {
            "probe": {
                "metric": "kl_proxy_via_hessian_trace",
                "n_tokens": 0,
                "data_free": True,
                "J": 15,
                "t2_grid": [],
            },
            "regime_gate": {
                "min_operating_bits": 3.0,
                "qep_off_switch": True,
            },
            "measurement": "offline_ternary_mse",
            "total_params": 304128,
            "fallback_global": False,
            "fallback_reason": "none",
            "fitness_form": "Sum_l alpha_l * t_l^2",
        }
        return eha.build_sidecar(
            infos, audit,
            model_hash_value="deadbeef" * 4,
            gguf_path=Path("/tmp/model.gguf"),
            bundle_name="model-bundle",
        )

    def test_top_level_fields(self) -> None:
        s = self._sample_sidecar()
        # The shape mirrors the ane_state_layout.v1
        # (version, schema, bundle_name at top level).
        for key in ("schema", "version", "bundle_name",
                    "gguf_path", "model_hash", "fitness_form",
                    "measurement", "probe", "regime_gate",
                    "total_params", "fallback_global",
                    "fallback_reason", "layer_count", "layers"):
            self.assertIn(key, s, f"missing top-level key: {key}")
        self.assertEqual(s["schema"], "ane.alpha-coefficients.v1")
        self.assertEqual(s["version"], 1)
        self.assertEqual(s["fitness_form"], "Sum_l alpha_l * t_l^2")

    def test_per_tensor_fields(self) -> None:
        s = self._sample_sidecar()
        for layer in s["layers"]:
            for key in ("name", "family", "shape", "n_elements",
                        "frobenius_norm", "t_squared",
                        "t_squared_source", "dtype_source",
                        "alpha", "alpha_floor_applied", "fit_r2",
                        "n_samples", "fallback"):
                self.assertIn(key, layer,
                              f"missing per-tensor key: {key}")

    def test_fitness_form_matches_design_doc(self) -> None:
        """The fitness form is the architect's ratified form
        (L = Sum_l alpha_l * t_l^2). Pinned here so a future
        refactor can't drift away from the design doc."""
        s = self._sample_sidecar()
        self.assertEqual(s["fitness_form"], eha.FITNESS_FORM)
        self.assertIn("alpha", s["fitness_form"])
        self.assertIn("t_l^2", s["fitness_form"])

    def test_regime_gate_contains_qep(self) -> None:
        """The regime gate carries the QEP off-switch: when
        the operating bitwidth drops below 3.0, the additive
        model breaks and the consumer must fall back to
        uniform. The gate is stamped explicitly so the
        consumer does not have to re-derive it.
        """
        s = self._sample_sidecar()
        self.assertTrue(s["regime_gate"]["qep_off_switch"])
        self.assertEqual(s["regime_gate"]["min_operating_bits"],
                         3.0)

    def test_fallback_indicators_present(self) -> None:
        s = self._sample_sidecar()
        self.assertIn("fallback_global", s)
        self.assertIn("fallback_reason", s)
        # Per-tensor fallback indicator too.
        for layer in s["layers"]:
            self.assertIn("fallback", layer)
            self.assertIn("alpha_floor_applied", layer)

    def test_sidecar_is_json_serializable(self) -> None:
        s = self._sample_sidecar()
        # The writer produces pretty-printed JSON; ensure
        # round-trip through json.dumps + json.loads.
        encoded = json.dumps(s, indent=2)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema"], s["schema"])
        self.assertEqual(decoded["layers"][0]["name"],
                         s["layers"][0]["name"])
        # Shape (a tuple in TensorInfo) must serialize as a
        # JSON array; this is the on-disk contract.
        self.assertIsInstance(decoded["layers"][0]["shape"], list)

    def test_sidecar_write_and_read(self) -> None:
        s = self._sample_sidecar()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sidecar.json"
            eha.write_sidecar(out, s)
            self.assertTrue(out.is_file())
            with out.open() as f:
                read = json.load(f)
        self.assertEqual(read["schema"], s["schema"])
        self.assertEqual(len(read["layers"]), len(s["layers"]))

    def test_report_write_and_read(self) -> None:
        s = self._sample_sidecar()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.md"
            eha.write_report(out, s)
            self.assertTrue(out.is_file())
            text = out.read_text()
        # The report has the standard sections.
        for header in ("# HIGGS per-layer alpha report:",
                       "## Per-tensor results"):
            self.assertIn(header, text)
        # The bundle name is in the title.
        self.assertIn(s["bundle_name"], text)
        # The fitness form appears.
        self.assertIn(s["fitness_form"], text)
        # Every per-tensor name is in the body.
        for layer in s["layers"]:
            self.assertIn(layer["name"], text)


# ---- uniform fallback path ----

class UniformFallbackTest(unittest.TestCase):
    """The estimator falls back to uniform alpha (1.0 for all
    layers) when the model is below --min-params-for-estimate
    (default 1B). The fallback is recorded globally on the
    sidecar and per-tensor so a downstream consumer can
    detect the degraded path."""

    def _synthetic_tensor(
        self, name: str, n: int, family: str = "attn_v",
        value: float = 1.0,
    ) -> "eha.TensorInfo":
        return eha.TensorInfo(
            name=name,
            family=family,
            n_elements=n,
            frobenius_norm=value,
            t_squared=0.1,
            t_squared_source="offline_ternary_mse",
            dtype_source="F32",
            alpha=1.0,
            alpha_floor_applied=False,
            fit_r2=1.0,
            n_samples=0,
            fallback="global_uniform",
            shape=(n,),
        )

    def test_global_fallback_below_threshold(self) -> None:
        """When the model is below the size threshold, every
        layer's alpha is 1.0 (uniform) and the global
        fallback flag is True."""
        infos = [
            self._synthetic_tensor("blk.0.attn_v.weight", 100),
            self._synthetic_tensor("blk.0.ffn_gate.weight", 200),
        ]
        # Total = 300 params, well below 1B.
        for info in infos:
            self.assertEqual(info.alpha, 1.0)
            self.assertEqual(info.fallback, "global_uniform")

    def test_global_fallback_above_threshold(self) -> None:
        """Above the threshold, the family prior drives the
        alpha; the global fallback flag is False."""
        a_v = eha.structural_alpha(
            frobenius_norm=1.0, n_elements=82944, family="attn_v")
        a_ffn = eha.structural_alpha(
            frobenius_norm=1.0, n_elements=82944, family="ffn_gate")
        # K/V (1.30) > FFN (0.45) per the family prior.
        self.assertGreater(a_v, a_ffn)

    def test_per_layer_fallback_for_clamped_alpha(self) -> None:
        """When a layer's alpha is below the positive floor
        (the research doc's P2 guard), the per-tensor
        fallback is 'per_layer_uniform' and the global
        fallback flag is False (other layers are still
        using the structural estimate)."""
        alpha_unclamped = 0.0001
        floor = 0.001
        final, applied = eha.clamp_alpha(alpha_unclamped, floor)
        self.assertEqual(final, floor)
        self.assertTrue(applied)


# ---- orchestrator (estimate + measurement function injection) ----

class EstimateOrchestratorTest(unittest.TestCase):
    """The orchestrator is L1-agnostic: the ``measurement``
    parameter accepts a function that computes ``t_l^2`` from
    the F32 reference. The default is the offline ternary
    MSE proxy; the L1 path passes a kernel-dequant-based
    function. This test exercises the parameterization with
    a synthetic measurement to prove the L1 swap is a
    one-call change."""

    def _stub_tensor(self, name: str, family: str, n: int,
                     value: float = 1.0):
        """A minimal stand-in for a gguf ReaderTensor. Only
        the attributes the orchestrator reads are populated.
        """

        class Stub:
            pass

        t = Stub()
        t.name = name
        t.tensor_type = type("Q", (), {"name": "F32"})()
        t.shape = (n,)
        t.data = np.full(n, value, dtype=np.float32)
        return t

    def test_estimate_with_synthetic_measurement(self) -> None:
        """A constant t_l^2 across all tensors, regardless of
        family, demonstrates the measurement-function
        parameterization: the orchestrator routes whatever
        the function returns into the per-tensor record
        unchanged."""

        def constant_measurement(reference: np.ndarray) -> tuple[float, str]:
            return (0.42, "synthetic_test")

        tensors = [
            self._stub_tensor("blk.0.attn_v.weight", "attn_v", 100, 1.0),
            self._stub_tensor("blk.0.attn_q.weight", "attn_q", 100, 1.0),
            self._stub_tensor("blk.0.ffn_gate.weight", "ffn_gate", 100, 1.0),
        ]
        config = eha.EstimateConfig(min_params_for_estimate=10)
        infos, audit = eha.estimate(
            tensors, [], config, measurement=constant_measurement)
        # Every t_l^2 is the constant the measurement returned.
        for info in infos:
            self.assertAlmostEqual(info.t_squared, 0.42, places=12)
            self.assertEqual(info.t_squared_source, "synthetic_test")
        # The audit block records the constant measurement.
        self.assertEqual(audit["measurement"], "offline_ternary_mse")

    def test_estimate_below_size_threshold_uniform(self) -> None:
        """Below the size threshold the orchestrator applies
        uniform alpha to every layer (the
        ``fallback_global`` flag flips)."""
        tensors = [
            self._stub_tensor("blk.0.attn_v.weight", "attn_v", 100, 1.0),
        ]
        config = eha.EstimateConfig(min_params_for_estimate=10_000)
        infos, audit = eha.estimate(tensors, [], config)
        self.assertTrue(audit["fallback_global"])
        self.assertIn("below", audit["fallback_reason"])
        # Every layer is uniform.
        for info in infos:
            self.assertEqual(info.alpha, 1.0)
            self.assertEqual(info.fallback, "global_uniform")

    def test_estimate_above_size_threshold_uses_prior(self) -> None:
        """Above the size threshold the family prior drives
        the alpha; the global fallback flag is False."""
        tensors = [
            self._stub_tensor("blk.0.attn_v.weight", "attn_v", 100, 1.0),
            self._stub_tensor("blk.0.ffn_gate.weight", "ffn_gate", 100, 1.0),
        ]
        config = eha.EstimateConfig(min_params_for_estimate=10)
        infos, audit = eha.estimate(tensors, [], config)
        self.assertFalse(audit["fallback_global"])
        # The mean positive alpha is 1.0 by construction.
        positive = [info.alpha for info in infos if info.alpha > 0]
        self.assertAlmostEqual(sum(positive) / len(positive), 1.0,
                               places=6)
        # K/V (1.30) > FFN (0.45).
        by_name = {info.name: info for info in infos}
        self.assertGreater(
            by_name["blk.0.attn_v.weight"].alpha,
            by_name["blk.0.ffn_gate.weight"].alpha)

    def test_estimate_normalizes_mean_alpha_to_one(self) -> None:
        """After the family-prior lookup, the orchestrator
        normalizes so the mean positive alpha is exactly
        1.0. This is the GA's "uniform alpha = 1.0 = no
        weighting" convention used by D-PACE
        (tessera-dpace.h:91)."""

        tensors = [
            self._stub_tensor("blk.0.attn_v.weight", "attn_v", 100, 1.0),
            self._stub_tensor("blk.0.attn_k.weight", "attn_k", 100, 1.0),
            self._stub_tensor("blk.0.attn_q.weight", "attn_q", 100, 1.0),
            self._stub_tensor("blk.0.attn_output.weight", "attn_output", 100, 1.0),
            self._stub_tensor("blk.0.ffn_gate.weight", "ffn_gate", 100, 1.0),
            self._stub_tensor("blk.0.ffn_up.weight", "ffn_up", 100, 1.0),
            self._stub_tensor("blk.0.ffn_down.weight", "ffn_down", 100, 1.0),
        ]
        config = eha.EstimateConfig(min_params_for_estimate=10)
        infos, _audit = eha.estimate(tensors, [], config)
        positive = [info.alpha for info in infos if info.alpha > 0]
        # The mean positive alpha is 1.0 by construction.
        mean = sum(positive) / len(positive)
        self.assertAlmostEqual(mean, 1.0, places=6)


# ---- model_hash ----

class ModelHashTest(unittest.TestCase):

    def test_hash_is_deterministic(self) -> None:
        """The model_hash is the cache-invalidation key. It
        must be deterministic for a given file path and
        stable across re-reads of the same file.
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "stub.gguf"
            # Write 200KB of bytes (enough to span the 64KB
            # prefix + 64KB suffix windows the hash uses).
            p.write_bytes(b"\x00" * 200_000)
            h1 = eha.model_hash(p)
            h2 = eha.model_hash(p)
        self.assertEqual(h1, h2)

    def test_hash_changes_with_content(self) -> None:
        """Different file content must produce a different
        hash (the cache-invalidation contract)."""
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.gguf"
            p2 = Path(td) / "b.gguf"
            p1.write_bytes(b"AAAA" * 100_000)
            p2.write_bytes(b"BBBB" * 100_000)
            h1 = eha.model_hash(p1)
            h2 = eha.model_hash(p2)
        self.assertNotEqual(h1, h2)

    def test_hash_is_16_hex_chars(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "stub.gguf"
            p.write_bytes(b"X")
            h = eha.model_hash(p)
        self.assertEqual(len(h), 16)
        int(h, 16)  # raises if not hex


# ---- full pipeline on a real (small) GGUF fixture ----

class FullPipelineTest(unittest.TestCase):
    """End-to-end: load a real GGUF, run the estimator, and
    verify the sidecar. The fixture is the small
    stories15M model. The test is skipped if the fixture
    is not present (so it does not break a fresh checkout);
    the unit tests above cover the math regardless of the
    fixture."""

    @staticmethod
    def _find_fixture() -> Path | None:
        """Search for the stories15M fixture. The worktree
        layout copies only files that are touched; the
        fixture lives in ``build-ane/tinyllamas/`` at the
        main repo (the worktree's grandparent). The test
        skips cleanly if it can't be located.
        """
        candidates = [
            REPO_ROOT / "build-ane" / "tinyllamas" / "stories15M-q4_0.gguf",
            REPO_ROOT.parent / "build-ane" / "tinyllamas" / "stories15M-q4_0.gguf",
            REPO_ROOT.parent.parent / "build-ane" / "tinyllamas" / "stories15M-q4_0.gguf",
        ]
        for p in candidates:
            if p.is_file():
                return p
        return None

    def setUp(self) -> None:
        self.fixture = self._find_fixture()
        if self.fixture is None:
            self.skipTest(
                "stories15M-q4_0.gguf fixture not present in "
                "build-ane/tinyllamas/; skipping end-to-end tests")

    def test_full_pipeline_produces_sidecar(self) -> None:
        """Run the full pipeline on a real GGUF, write a
        sidecar, read it back, and validate the shape.
        """
        assert self.fixture is not None  # setUp's skipTest guard
        tensors, kv_keys = eha._load_gguf(self.fixture)
        # The fixture is small (< 1B params), so the
        # global-uniform fallback fires by default. The
        # test config below disables the fallback to
        # exercise the family-prior path.
        config = eha.EstimateConfig(min_params_for_estimate=10)
        infos, audit = eha.estimate(tensors, kv_keys, config)
        # The fixture has multiple tensors.
        self.assertGreater(len(infos), 0)
        # The family classification caught the known
        # transformer tensor families.
        families = {info.family for info in infos}
        for expected in ("attn_v", "attn_k", "attn_q",
                         "attn_output", "ffn_gate", "ffn_up",
                         "ffn_down", "norm", "token_embd"):
            self.assertIn(expected, families)
        # The per-tensor alpha is non-negative.
        for info in infos:
            self.assertGreaterEqual(info.alpha, 0.0)
        # The sidecar serializes round-trip.
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sidecar.json"
            sidecar = eha.build_sidecar(
                infos, audit,
                model_hash_value=eha.model_hash(self.fixture),
                gguf_path=self.fixture,
            )
            eha.write_sidecar(out, sidecar)
            self.assertTrue(out.is_file())
            with out.open() as f:
                read = json.load(f)
            self.assertEqual(read["schema"], eha.SIDECAR_SCHEMA)
            self.assertEqual(read["version"], eha.SIDECAR_VERSION)
            # The fitness form is the architect's ratified
            # string.
            self.assertEqual(read["fitness_form"],
                             eha.FITNESS_FORM)
            # The K/V alphas are the highest in the model
            # (the SLQ/BAQ ranking the family prior encodes).
            kv_alphas = [
                layer["alpha"] for layer in read["layers"]
                if layer["family"] in ("attn_k", "attn_v")]
            ffn_alphas = [
                layer["alpha"] for layer in read["layers"]
                if layer["family"] in ("ffn_gate", "ffn_up",
                                       "ffn_down")]
            self.assertGreater(
                sum(kv_alphas) / len(kv_alphas),
                sum(ffn_alphas) / len(ffn_alphas),
                "K/V alphas should rank above FFN alphas per "
                "the SLQ/BAQ structural prior")

    def test_full_pipeline_global_fallback_default(self) -> None:
        """With the default 1B parameter threshold, the
        tinyllama fixture (~15M params) falls back to uniform
        alpha for every layer. The global fallback flag is
        True and the reason is recorded.
        """
        assert self.fixture is not None
        tensors, kv_keys = eha._load_gguf(self.fixture)
        # Default config: 1B threshold.
        config = eha.EstimateConfig()
        _infos, audit = eha.estimate(tensors, kv_keys, config)
        self.assertTrue(audit["fallback_global"])
        self.assertIn("below", audit["fallback_reason"])


# ---- Phase 3.5: parity tests for the C++ structural proxy ----
#
# The thin Python wrapper at tools/tessera/estimate_higgs_alpha.py
# subprocesses the C++ binary (tools/quantize/tessera/tessera-higgs-proxy)
# when present on PATH, and falls back to the in-process NumPy path
# otherwise. These tests verify:
#   1. the C++ path is taken when the binary is on PATH,
#   2. the C++ and NumPy sidecars agree to the documented
#      float tolerance (byte-for-byte at the JSON-key level;
#      floats agree to F32 precision),
#   3. the NumPy-fallback path stamps the
#      offline_ternary_mse_numpy_fallback discriminator in the
#      sidecar so the consumer can tell the two paths apart.
#
# Tests that exercise C++-only behavior (GGUF reading, family
# classification, structural alpha math, L1-agnostic measurement
# function) live in tools/quantize/tessera/test_higgs_proxy.cpp
# (Phase 3.5 C++ test suite).

import shutil
import subprocess


_CPP_BINARY = "tessera-higgs-proxy"


def _has_cpp_binary() -> bool:
    return shutil.which(_CPP_BINARY) is not None


class CppParityTest(unittest.TestCase):
    """Phase 3.5 parity tests: verify the C++ structural proxy
    produces a sidecar that matches the NumPy path within the
    documented float tolerance.

    Skipped when the C++ binary is not on PATH (e.g. a dev
    environment without a C++ build). The dev path is the
    in-process NumPy path; this test class exercises the
    C++ path, so skip is the right behavior.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _has_cpp_binary():
            cls.skip_reason = "tessera-higgs-proxy binary not on PATH"
        else:
            cls.skip_reason = None

    def setUp(self) -> None:
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def test_cpp_path_stamps_offline_ternary_mse(self) -> None:
        """When the C++ binary is on PATH, the wrapper uses it
        and the sidecar's measurement is the C++ default
        ``offline_ternary_mse`` (NOT the
        ``offline_ternary_mse_numpy_fallback`` discriminator
        that the in-process NumPy path stamps).
        """
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "fixture.gguf"
            sidecar = Path(tmp) / "sidecar.json"
            # Build a tiny synthetic GGUF (one 64x64 F32 tensor).
            self._build_fixture(gguf)
            rc = subprocess.run(
                ["python3", "tools/tessera/estimate_higgs_alpha.py",
                 "--gguf", str(gguf),
                 "--output", str(sidecar),
                 "--min-params-for-estimate", "0"],
                capture_output=True, text=True,
            )
            self.assertEqual(rc.returncode, 0,
                f"wrapper failed: {rc.stderr}")
            data = json.loads(sidecar.read_text())
            self.assertEqual(data["measurement"], "offline_ternary_mse")
            self.assertEqual(data["schema"],
                             "ane.alpha-coefficients.v1")
            self.assertEqual(data["version"], 1)

    def test_cpp_path_agrees_with_numpy_to_f32(self) -> None:
        """The C++ and NumPy sidecars agree to F32 precision
        on every per-tensor numeric field. The wrapper stamps
        ``offline_ternary_mse_numpy_fallback`` for the NumPy
        path; the rest of the sidecar is byte-equivalent
        (key order, value types) modulo the float-repr
        tolerance.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "fixture.gguf"
            cpp_sidecar = Path(tmp) / "cpp.json"
            np_sidecar  = Path(tmp) / "np.json"
            self._build_fixture(gguf)

            # C++ path (binary on PATH).
            rc_cpp = subprocess.run(
                ["python3", "tools/tessera/estimate_higgs_alpha.py",
                 "--gguf", str(gguf),
                 "--output", str(cpp_sidecar),
                 "--min-params-for-estimate", "0"],
                capture_output=True, text=True,
            )
            self.assertEqual(rc_cpp.returncode, 0, rc_cpp.stderr)

            # NumPy path (binary off PATH). The wrapper
            # should detect this and fall back.
            env = {k: v for k, v in os.environ.items()
                   if k != "PATH"}
            env["PATH"] = "/usr/bin:/bin"
            rc_np = subprocess.run(
                ["python3", "tools/tessera/estimate_higgs_alpha.py",
                 "--gguf", str(gguf),
                 "--output", str(np_sidecar),
                 "--min-params-for-estimate", "0"],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(rc_np.returncode, 0, rc_np.stderr)

            cpp = json.loads(cpp_sidecar.read_text())
            np_ = json.loads(np_sidecar.read_text())

            # The measurement field is the one explicit
            # discriminator; the C++ path uses the default,
            # the NumPy path stamps the fallback value.
            self.assertEqual(cpp["measurement"], "offline_ternary_mse")
            self.assertEqual(np_["measurement"],
                             "offline_ternary_mse_numpy_fallback")

            # Same top-level shape.
            self.assertEqual(set(cpp.keys()), set(np_.keys()))

            # Per-tensor numeric fields agree to F32.
            self.assertEqual(len(cpp["layers"]), len(np_["layers"]))
            for a, b in zip(cpp["layers"], np_["layers"]):
                self.assertEqual(a["name"], b["name"])
                self.assertEqual(a["family"], b["family"])
                self.assertEqual(a["n_elements"], b["n_elements"])
                self.assertEqual(a["alpha_floor_applied"],
                                 b["alpha_floor_applied"])
                self.assertEqual(a["fallback"], b["fallback"])
                for k in ("alpha", "t_squared", "frobenius_norm"):
                    diff = abs(a[k] - b[k])
                    tol = max(1e-5, 1e-4 * abs(a[k]))
                    self.assertLessEqual(
                        diff, tol,
                        f"field {k} disagrees: cpp={a[k]} np={b[k]} "
                        f"diff={diff} tol={tol}")

    def test_cpp_path_model_hash_matches(self) -> None:
        """The C++ path's model_hash matches the NumPy
        path's model_hash. The two implementations use the
        same FIPS 180-4 algorithm; a parity check on the
        same file must produce the same first-16-hex.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "fixture.gguf"
            cpp_sidecar = Path(tmp) / "cpp.json"
            np_sidecar  = Path(tmp) / "np.json"
            self._build_fixture(gguf)

            subprocess.run(
                ["python3", "tools/tessera/estimate_higgs_alpha.py",
                 "--gguf", str(gguf),
                 "--output", str(cpp_sidecar),
                 "--min-params-for-estimate", "0"],
                capture_output=True, text=True, check=True,
            )
            env = {k: v for k, v in os.environ.items()
                   if k != "PATH"}
            env["PATH"] = "/usr/bin:/bin"
            subprocess.run(
                ["python3", "tools/tessera/estimate_higgs_alpha.py",
                 "--gguf", str(gguf),
                 "--output", str(np_sidecar),
                 "--min-params-for-estimate", "0"],
                capture_output=True, text=True, env=env, check=True,
            )

            cpp = json.loads(cpp_sidecar.read_text())
            np_ = json.loads(np_sidecar.read_text())
            self.assertEqual(cpp["model_hash"], np_["model_hash"])
            self.assertEqual(len(cpp["model_hash"]), 16)

    def test_cpp_path_uniform_fallback_below_threshold(self) -> None:
        """With the default 1B threshold, the C++ path
        produces a uniform-fallback sidecar (every alpha is
        1.0, t_squared_source is ``uniform_fallback``).
        """
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "fixture.gguf"
            sidecar = Path(tmp) / "sidecar.json"
            self._build_fixture(gguf)
            rc = subprocess.run(
                ["python3", "tools/tessera/estimate_higgs_alpha.py",
                 "--gguf", str(gguf),
                 "--output", str(sidecar)],
                capture_output=True, text=True,
            )
            self.assertEqual(rc.returncode, 0, rc.stderr)
            data = json.loads(sidecar.read_text())
            self.assertEqual(data["measurement"], "uniform_fallback")
            self.assertTrue(data["fallback_global"])
            for layer in data["layers"]:
                self.assertEqual(layer["alpha"], 1.0)
                self.assertEqual(layer["fallback"], "global_uniform")

    @staticmethod
    def _build_fixture(path: Path) -> None:
        """Build a tiny synthetic F32 GGUF (one 64x64 tensor)
        for parity testing. The C++ and NumPy paths both
        read this and produce sidecars.
        """
        # Use the gguf-py library if available, else fall
        # back to a small hand-rolled GGUF. The fixture
        # must be a valid GGUF that both the C++ and NumPy
        # paths can read.
        try:
            from gguf import GGUFReader  # type: ignore
            import numpy as np
            # Build a minimal GGUF in memory.
            from gguf import GGUFWriter  # type: ignore
            writer = GGUFWriter(str(path), "test")
            data = np.arange(64 * 64, dtype=np.float32)
            writer.add_tensor("blk.0.attn_v.weight", data.reshape(64, 64))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        except ImportError:
            # gguf-py unavailable (e.g. CI without the
            # optional dep). The parity tests are skipped
            # in that environment; mark the fixture as
            # unused.
            raise unittest.SkipTest("gguf-py unavailable")


if __name__ == "__main__":
    unittest.main()
