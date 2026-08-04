"""Per-tensor ``recommended_action`` derivation rules.

The calibration side surfaces the orchestrator's per-(model, family)
feedback (``l5_outcome`` + ``l5_weights``) onto ``tensor_stats`` as
a single string per tensor. The string is the calibration pipeline's
suggestion for what to do with this tensor on the next pass:

  * ``protect``         - the orchestrator is hurting this family
                          (miscalibration_score > 0.5 and
                          hit_rate < 0.5); the calibration side
                          should bias this family toward
                          higher precision and skip aggressive
                          requant.
  * ``requant_up``      - a non-accepted plan increased error
                          materially (delta_mse > 0.001); the
                          family is being over-aggressively
                          quantized and a re-quant at a higher
                          qtype is the right next step.
  * ``requant_down``    - the family is over-quantized; a
                          re-quant at a lower qtype is safe
                          (plan was accepted and hit_rate
                          > 0.9).
  * ``monitor``         - the orchestrator is calibrated for this
                          family; just keep watching.
  * ``noop``            - the feedback loop has not produced a
                          verdict for this (model, family) yet
                          (no ``l5_weights`` row, or
                          miscalibration_score is NULL).

The rules are a small table; this module is the single source of
truth. The thresholds (0.5, -0.2, 0.001, 0.9) are KNOBs and are
documented below; the rules evaluate in declaration order and
return on the first match.

CONTRACT (for C++-side mirroring, if ever needed):

  inputs:
    miscalibration_score  float | None   - l5_retune slope;
                                            None = no l5_weights row
    hit_rate              float | None   - l5_weights.hit_rate
    delta_mse             float | None   - l5_outcome.delta_mse
                                            (most recent or aggregate)
    plan_accepted         bool  | None   - l5_outcome.plan_accepted
                                            (most recent or aggregate)

  output:
    one of "protect" / "requant_up" / "requant_down" /
          "monitor" / "noop"

C++ mirroring is intentionally not implemented in this commit;
the calibration pipeline is the only writer of
``recommended_action``. The schema change is additive (the C++
side just carries the column through the upsert with no logic).
"""

from __future__ import annotations

from typing import Optional


# Threshold knobs. These are the only places the per-action
# thresholds live; bump them here and the rules update everywhere.
PROTECT_MISCALIBRATION_THRESHOLD: float = 0.5
PROTECT_HIT_RATE_MAX: float = 0.5
MONITOR_MISCALIBRATION_MAX: float = -0.2
REQUANT_UP_DELTA_MSE_MIN: float = 0.001
REQUANT_DOWN_HIT_RATE_MIN: float = 0.9


def derive_recommended_action(
    miscalibration_score: Optional[float],
    hit_rate: Optional[float],
    delta_mse: Optional[float],
    plan_accepted: Optional[bool],
) -> str:
    """Apply the rules in declaration order; return the first match.

    Order matters: ``protect`` is the most specific (requires both a
    positive slope and a low hit rate); ``monitor`` is the negation
    (negative slope, no sign the plan hurt); ``requant_up`` is the
    "the last plan hurt" case; ``requant_down`` is the "the family
    is over-quantized" case. ``noop`` is the default.

    The order is also the priority: a tensor that matches
    ``protect`` rules wins over ``requant_up`` even if both apply.
    """
    # No l5_weights row yet -> the feedback loop has no verdict.
    if miscalibration_score is None or hit_rate is None:
        return "noop"
    if (miscalibration_score > PROTECT_MISCALIBRATION_THRESHOLD
            and hit_rate < PROTECT_HIT_RATE_MAX):
        return "protect"
    if (miscalibration_score < MONITOR_MISCALIBRATION_MAX
            and (delta_mse is None or delta_mse > 0.0)):
        return "monitor"
    if (plan_accepted is False
            and delta_mse is not None
            and delta_mse > REQUANT_UP_DELTA_MSE_MIN):
        return "requant_up"
    if (plan_accepted is True
            and hit_rate > REQUANT_DOWN_HIT_RATE_MIN):
        return "requant_down"
    return "noop"


__all__ = [
    "PROTECT_MISCALIBRATION_THRESHOLD",
    "PROTECT_HIT_RATE_MAX",
    "MONITOR_MISCALIBRATION_MAX",
    "REQUANT_UP_DELTA_MSE_MIN",
    "REQUANT_DOWN_HIT_RATE_MIN",
    "derive_recommended_action",
]
