"""Tests for tools/tessera/l5_action.py (recommended_action rules).

The rules table is a small priority list; each test exercises one
branch and the no-l5-weights default. The thresholds are
documented at the top of l5_action.py; bumping them there is the
only way the rules change.

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from l5_action import (
    MONITOR_MISCALIBRATION_MAX,
    PROTECT_HIT_RATE_MAX,
    PROTECT_MISCALIBRATION_THRESHOLD,
    REQUANT_DOWN_HIT_RATE_MIN,
    REQUANT_UP_DELTA_MSE_MIN,
    derive_recommended_action,
)


class TestDeriveRecommendedAction(unittest.TestCase):
    # ---- 1. no l5_weights -> noop (the default branch) ----

    def test_no_l5_weights_returns_noop(self) -> None:
        # miscalibration_score is None -> noop regardless of other inputs.
        self.assertEqual(
            derive_recommended_action(None, 0.5, 0.0, True), "noop")
        self.assertEqual(
            derive_recommended_action(None, 0.5, 0.0, None), "noop")
        # hit_rate is None -> noop (l5_weights row is incomplete).
        self.assertEqual(
            derive_recommended_action(0.0, None, 0.0, True), "noop")

    # ---- 2. protect: high miscal + low hit rate ----

    def test_protect_high_miscalibration_low_hit_rate(self) -> None:
        # Strictly above the protect threshold + strictly below the
        # hit_rate max.
        self.assertEqual(
            derive_recommended_action(
                PROTECT_MISCALIBRATION_THRESHOLD + 0.1, 0.3, 0.0, True),
            "protect")
        # Boundary: at the threshold, not above -> falls through.
        self.assertNotEqual(
            derive_recommended_action(
                PROTECT_MISCALIBRATION_THRESHOLD, 0.3, 0.0, True),
            "protect")
        # Boundary: at the hit_rate max, not below -> falls through.
        self.assertNotEqual(
            derive_recommended_action(
                PROTECT_MISCALIBRATION_THRESHOLD + 0.1,
                PROTECT_HIT_RATE_MAX, 0.0, True),
            "protect")

    # ---- 3. monitor: low miscal + no negative delta_mse ----

    def test_monitor_low_miscalibration(self) -> None:
        # Strictly below the monitor max, delta_mse absent or >= 0.
        self.assertEqual(
            derive_recommended_action(
                MONITOR_MISCALIBRATION_MAX - 0.1, 0.7, None, True),
            "monitor")
        # A negative delta_mse disqualifies monitor (the plan
        # already reduced error, so the calibration is fine and
        # we don't need a watch flag).
        self.assertNotEqual(
            derive_recommended_action(
                MONITOR_MISCALIBRATION_MAX - 0.1, 0.7, -0.001, True),
            "monitor")
        # At the threshold, not below -> not monitor.
        self.assertNotEqual(
            derive_recommended_action(
                MONITOR_MISCALIBRATION_MAX, 0.7, None, True),
            "monitor")

    # ---- 4. requant_up: rejected plan that hurt ----

    def test_requant_up_rejected_plan_hurt(self) -> None:
        self.assertEqual(
            derive_recommended_action(
                0.0, 0.6, REQUANT_UP_DELTA_MSE_MIN + 0.01, False),
            "requant_up")
        # Boundary: at the threshold, not above -> not requant_up.
        self.assertNotEqual(
            derive_recommended_action(
                0.0, 0.6, REQUANT_UP_DELTA_MSE_MIN, False),
            "requant_up")
        # plan_accepted is None or True -> not requant_up.
        self.assertNotEqual(
            derive_recommended_action(
                0.0, 0.6, REQUANT_UP_DELTA_MSE_MIN + 0.01, None),
            "requant_up")
        self.assertNotEqual(
            derive_recommended_action(
                0.0, 0.6, REQUANT_UP_DELTA_MSE_MIN + 0.01, True),
            "requant_up")
        # delta_mse is None -> not requant_up.
        self.assertNotEqual(
            derive_recommended_action(0.0, 0.6, None, False),
            "requant_up")

    # ---- 5. requant_down: accepted plan + high hit rate ----

    def test_requant_down_accepted_high_hit_rate(self) -> None:
        self.assertEqual(
            derive_recommended_action(
                0.0, REQUANT_DOWN_HIT_RATE_MIN + 0.05, None, True),
            "requant_down")
        # Boundary: at the threshold, not above -> not requant_down.
        self.assertNotEqual(
            derive_recommended_action(
                0.0, REQUANT_DOWN_HIT_RATE_MIN, None, True),
            "requant_down")
        # plan_accepted is False or None -> not requant_down.
        self.assertNotEqual(
            derive_recommended_action(
                0.0, REQUANT_DOWN_HIT_RATE_MIN + 0.05, None, False),
            "requant_down")
        self.assertNotEqual(
            derive_recommended_action(
                0.0, REQUANT_DOWN_HIT_RATE_MIN + 0.05, None, None),
            "requant_down")

    # ---- 6. priority: protect wins over requant_up ----

    def test_protect_wins_over_requant_up(self) -> None:
        # Both rules would match; protect (declared first) wins.
        self.assertEqual(
            derive_recommended_action(
                PROTECT_MISCALIBRATION_THRESHOLD + 0.1, 0.3,
                REQUANT_UP_DELTA_MSE_MIN + 0.01, False),
            "protect")

    # ---- 7. priority: monitor wins over requant_down ----

    def test_monitor_wins_over_requant_down(self) -> None:
        # monitor is declared before requant_down; low miscal +
        # accepted plan with high hit_rate.
        self.assertEqual(
            derive_recommended_action(
                MONITOR_MISCALIBRATION_MAX - 0.1,
                REQUANT_DOWN_HIT_RATE_MIN + 0.05, None, True),
            "monitor")

    # ---- 8. default: no rule matches -> noop ----

    def test_default_noop(self) -> None:
        # Mild negative slope (between monitor max and protect
        # threshold), middling hit rate, no delta_mse signal,
        # plan_accepted None.
        self.assertEqual(
            derive_recommended_action(0.1, 0.7, None, None),
            "noop")
        # All-zero (the l5_weights row exists but the family is
        # perfectly calibrated): no rule fires.
        self.assertEqual(
            derive_recommended_action(0.0, 0.7, 0.0, None),
            "noop")
        self.assertEqual(
            derive_recommended_action(0.0, 0.7, -0.001, True),
            "noop")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestDeriveRecommendedAction
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
