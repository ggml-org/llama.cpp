"""L5 retune: per-(model, family) recompute of the orchestrator's
sensitivity scoring weights from the feedback loop's residual.

The ``l5_outcome`` table is the "did this requant plan reduce
error?" verdict. The next consumer is this script: it reads
``l5_outcome.delta_mse`` plus the per-tensor sensitivity
components (``imatrix_magnitude``, ``gradient_proxy``,
``layer_position_prior``), fits a per-(model, family)
weighted closed-form OLS, and projects the result onto the
(w_imatrix, w_gradient, w_layer) simplex. The projection lands
in ``l5_weights`` with PRIMARY KEY (model_hash, family).

The orchestrator's next generation reads ``l5_weights`` back
(via ``--retune-from-db``) and uses the per-family recommendation
as the starting point for ``SensitivityScorer``, closing the
loop.

Phase 15: the retune model is now a 3-coefficient OLS:

    delta_mse = a + b_im * im + b_grad * grad + b_layer * layer

This decomposes which component is miscalibrated per (model,
family) instead of the 2-coefficient ``a + b * sensitivity_score``
model that lumps the three components into one combined signal.
The 3-coefficient OLS uses sample weights derived from each
row's ``n_samples`` and ``in_sample_loss`` so high-confidence
rows count more. When the per-tensor components are NULL
(pre-Phase-15 rows or a DB that the C++ side has not yet
migrated) the retune falls back to the 2-coefficient OLS on
the combined ``sensitivity_score``.

The 3-coefficient model uses numpy's ``lstsq`` for stability
(the closed-form 4x4 normal-equation inversion would be
numerically brittle; lstsq is the standard tool for small
weighted least squares and handles rank-deficient cases by
returning the minimum-norm solution).

The retune also writes a per-family ``top_fraction`` recommendation:

    top_fraction[family] = base_top_fraction * (1 + tanh(2*slope) * (1 - hit_rate))

High slope + low hit rate -> increase top_fraction (more
aggressive requantization of the miscalibrated family); low
slope or high hit rate -> keep at base. The orchestrator's
``RequantPlanner`` reads this via ``--per-family-top-fraction``
and overrides the uniform ``--top-fraction`` for the families
the retune has flagged.

The cross-model retune (``--retune-cross-model``) writes a
single per-family aggregate row with ``model_hash = "*"`` (the
n_samples-weighted mean across all models for that family).
The orchestrator's ``--retune-from-db`` falls back to the
cross-model row when the per-model row is missing (warm-start
new models from the cross-model mean). The cross-model row is
a generalization of the per-model row, not a replacement; the
per-model rows still drive the model-specific recommendation.

EMA-aware retune: the orchestrator's ``SensitivityScorer``
tracks an EMA per tensor. The ``l5_plan_ema`` table records
the EMA value at the iteration of the plan. The retune can
fit on EMA scores (which are stable across iterations) by
joining ``l5_outcome`` with ``l5_plan_ema`` on
``(model_hash, name, iteration, plan_id)``. The join is
optional: when the table is missing the retune falls back to
the per-iteration ``sensitivity_score`` (the original
behaviour). The EMA value is preferred because per-iteration
scores are noisy; the OLS on EMA values is the production
path, the per-iteration scores are a fallback.

Writes:
  * ``l5_weights``: PRIMARY KEY (model_hash, family), one row
    per group with a non-empty fit. The cross-model retune
    writes additional rows with ``model_hash = "*"``.
  * No-op when a (model, family) group has fewer than
    ``min_samples`` rows (default 3): the OLS estimate is too
    noisy to act on. The decision is recorded in the
    ``retune_source`` field (NULL when the group was skipped).

Companion to:
  * docs/tessera-unified-db.md (the unified-DB design)
  * docs/tessera-polars-integration-scout.md §5.4 (the
    feedback loop's retune step)
  * tools/tessera/l5_outcome.py (the residual source)
  * tools/tessera/l5_orchestrator.py
    (--retune-from-db consumes the result)
  * tools/quantize/tessera/tessera-quantize-db.cpp
    (the l5_weights CREATE TABLE statement)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from tessera_db import L5_WEIGHTS_COLS, TesseraDB, sql_escape


# Default base weights, mirroring l5_metrics.DEFAULT_WEIGHTS. The
# retune starts from these and shifts them per (model, family).
DEFAULT_BASE_WEIGHTS: tuple[float, float, float] = (0.5, 0.3, 0.2)

# How aggressively the OLS slope perturbs the base weights. The
# shift magnitude is alpha * |b| * (1 - hit_rate); alpha = 1.0
# means a unit slope + zero hit rate would shift the imatrix
# weight to 0 (full inversion of the im vs gradient balance).
# In practice b is on the order of 0.01-0.1 (delta_mse per unit
# sensitivity), so 0.5 is a reasonable default.
DEFAULT_ALPHA: float = 0.5

# Minimum per-(model, family) sample count for the OLS to be
# acted on. Below this the residual is too noisy; the row is
# left un-written and the orchestrator falls back to the
# base weights.
DEFAULT_MIN_SAMPLES: int = 3

# The retune algorithm tag. Written into retune_source so the
# consumer can tell which algorithm produced the row.
#
# Phase 15: the 3-coefficient OLS path tags with
# ``ols_3coef_v1``; the 2-coefficient fallback (combined
# sensitivity_score) keeps the original ``ols_slope_v1`` tag
# so existing consumer log analysis remains valid.
RETUNE_SOURCE_TAG_3COEF: str = "ols_3coef_v1"
RETUNE_SOURCE_TAG_2COEF: str = "ols_slope_v1"
RETUNE_SOURCE_TAG_CROSSMODEL: str = "ols_3coef_crossmodel_v1"

# Backward-compat alias for the Phase 12 callers that imported
# ``RETUNE_SOURCE_TAG`` (the singular tag). The 3-coefficient
# path uses ``RETUNE_SOURCE_TAG_3COEF``; the 2-coefficient
# fallback uses ``RETUNE_SOURCE_TAG_2COEF``. The alias resolves
# to the 2-coefficient tag, which is what the Phase 12 code
# was implicitly writing.
RETUNE_SOURCE_TAG: str = RETUNE_SOURCE_TAG_2COEF

# Default base top_fraction used by the per-family
# recommendation. The retune produces a per-family value via
# top_fraction = base * (1 + tanh(2*slope) * (1 - hit_rate));
# the base is the orchestrator's --top-fraction flag value
# (default 0.10).
DEFAULT_BASE_TOP_FRACTION: float = 0.10

# Default base budget fraction for the per-family
# requant_budget_bits recommendation:
#   budget_bits = family_storage_bits * (1 - hit_rate) * fraction
# family_storage_bits is the family's reference footprint
# (sum of n_elements * dtype_bits over the family's
# tensor_stats rows), so the budget is expressed relative to
# the source-dtype size: fraction=1.0 lets a fully
# miscalibrated family (hit_rate=0) spend up to its full
# reference footprint, while a converged family (hit_rate->1)
# gets budget->0 (no new bits). The deployment knob for a
# memory-bound target (e.g. an M1 laptop) is --budget-fraction:
# 0.25 means no family may exceed 25% of its reference
# footprint scaled by its remaining miscalibration.
DEFAULT_BASE_BUDGET_FRACTION: float = 1.0

# Nominal bits per element for the dtype strings that appear in
# tensor_stats.dtype. Mirrors the integer ordering of the C++
# writer's ts_unified_writer_qtype_bits (the single source of
# truth on the C++ side): no block-overhead accounting, integer
# bits only, so the producer's budget and the consumer's
# bit-cost arithmetic agree. Unknown dtypes map to None and are
# skipped by the storage aggregation (they contribute no bits
# rather than poison the sum).
DTYPE_BITS: dict[str, int] = {
    "f32": 32, "f16": 16, "bf16": 16,
    "q8_0": 8, "q8_k": 8,
    "q6_k": 6,
    "q5_0": 5, "q5_1": 5, "q5_k": 5,
    "q4_0": 4, "q4_1": 4, "q4_k": 4,
    "q3_k": 3, "q2_k": 2,
}


def _dtype_bits(dtype: str | None) -> int | None:
    """Nominal bits per element for a tensor_stats.dtype string.

    Returns None for unknown/NULL dtypes; the caller skips
    those rows.
    """
    if not dtype:
        return None
    return DTYPE_BITS.get(str(dtype).strip().lower())

# The retune's sample-weight formula (Phase 15):
#   weight = 1 / (1 + in_sample_loss * 100) * sqrt(n_samples / max_n_samples)
# The in_sample_loss term damps rows whose post-fit loss is
# high (the retune doesn't trust them); the n_samples term
# rewards rows with more data. The 100x scale on in_sample_loss
# is a rough conversion: a 0.01 loss halves the weight, a
# 0.1 loss cuts it to ~10%.
DEFAULT_LOSS_SCALE: float = 100.0


@dataclass
class FamilyWeights:
    """The per-(model, model_role, family) retune verdict: the
    recommended (w_imatrix, w_gradient, w_layer) and the OLS
    fit diagnostics.

    Fields:
      model_hash:        the model the weights apply to (or
                         ``"*"`` for the cross-model aggregate).
      model_role:        the architectural role (Phase 16) the
                         weights apply to. One of
                         ``"trunk"``, ``"dflash"``, ``"dspark"``,
                         ``"mtp_nextn"``, ``"shared_embd"``. The
                         ``"*"`` model_hash / specific
                         model_role combination is the
                         cross-model, per-role aggregate. The
                         trunk's ``attn_q`` and the dflash
                         encoder's ``attn_q`` get independent
                         retune verdicts (different rows in
                         ``l5_weights``).
      family:            the tensor family the weights apply to.
      weights:           (w_imatrix, w_gradient, w_layer) on the
                         simplex. The combined shift is the
                         alpha-scaled per-component shift
                         projected to non-negative / sum-to-1.
      bias:              the OLS intercept.
      slopes:            (b_im, b_grad, b_layer) -- the
                         3-coefficient OLS coefficients. The
                         2-coefficient fallback packs the
                         single slope into b_grad and leaves
                         b_im / b_layer at 0.0.
      n_samples:         the count of l5_outcome rows that fed
                         the fit.
      in_sample_loss:    post-fit mean abs residual (the
                         average ``|delta_mse - (a + b_im*im +
                         b_grad*grad + b_layer*layer)|``).
      hit_rate:          fraction of plans in the group with
                         delta_mse < accept_threshold.
      top_fraction:      the per-family top_fraction
                         recommendation (Phase 15). NULL when
                         the retune was skipped (too few
                         samples, |slope| below threshold, or
                         the cross-model path). The
                         orchestrator's --per-family-top-fraction
                         reads this.
      coupling_score:    the per-(model, family) cross-component
                         coupling score. Pearson correlation
                         of the per-layer hit_rate between the
                         trunk's role and the dflash encoder's
                         role for the same family. A high score
                         means the two roles' miscalibration
                         moves together across layers; a low
                         score means they are independent.
                         ``None`` when the family has only one
                         role's rows (e.g. a single-role
                         retune) or when the per-role per-layer
                         hit rates have zero variance (the
                         correlation is undefined). NULL is
                         written to the l5_weights row.
      requant_budget_bits: the per-(model, model_role, family)
                         dispatch-side bit budget for the next
                         requant pass. Computed as
                         ``family_storage_bits * (1 - hit_rate)
                         * base_budget_fraction`` where
                         family_storage_bits is the family's
                         reference footprint from tensor_stats
                         (sum of n_elements * dtype_bits). NULL
                         when the retune was skipped (too few
                         samples) or the family has no
                         tensor_stats storage rows; the consumer
                         treats NULL as "no budget constraint".
                         The dispatch's L5 loop applies it as a
                         Lagrangian penalty on the per-family
                         A/B fitness.
      retune_algorithm:  one of ``RETUNE_SOURCE_TAG_3COEF`` /
                         ``RETUNE_SOURCE_TAG_2COEF`` /
                         ``RETUNE_SOURCE_TAG_CROSSMODEL``.
      was_acted_on:      True if the retune wrote weights; False
                         if the group was skipped.
    """

    model_hash: str
    model_role: str
    family: str
    weights: tuple[float, float, float]
    bias: float
    slopes: tuple[float, float, float]
    n_samples: int
    in_sample_loss: float
    hit_rate: float
    top_fraction: float | None
    coupling_score: float | None
    requant_budget_bits: int | None
    retune_algorithm: str
    was_acted_on: bool

    def to_dict(self) -> dict:
        return {
            "model_hash":       self.model_hash,
            "model_role":       self.model_role,
            "family":           self.family,
            "w_imatrix":        float(self.weights[0]),
            "w_gradient":       float(self.weights[1]),
            "w_layer":          float(self.weights[2]),
            "bias":             float(self.bias),
            "slope":            float(self.slopes[1]),  # b_grad is the
            # "primary" slope for the 2-coefficient fallback's
            # bookkeeping (the orchestrator's diagnostics surface
            # the "slope" column on l5_weights as the dominant
            # coefficient).
            "n_samples":        int(self.n_samples),
            "in_sample_loss":   float(self.in_sample_loss),
            "hit_rate":         float(self.hit_rate),
            "top_fraction":     (
                float(self.top_fraction) if self.top_fraction is not None
                else None
            ),
            "coupling_score":   (
                float(self.coupling_score) if self.coupling_score is not None
                else None
            ),
            "requant_budget_bits": (
                int(self.requant_budget_bits)
                if self.requant_budget_bits is not None
                else None
            ),
            "retune_source":    (self.retune_algorithm
                                 if self.was_acted_on else ""),
        }

    # Backward-compat property. Phase 12 code read
    # ``v.slope`` (the single 2-coefficient slope). Phase 15
    # uses ``v.slopes`` (a 3-tuple). The alias keeps both
    # APIs working; new code should prefer ``v.slopes``.
    @property
    def slope(self) -> float:
        return float(self.slopes[1])


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _ols_3coef_weighted(
    im: list[float],
    grad: list[float],
    layer: list[float],
    y: list[float],
    sample_weights: list[float] | None = None,
) -> tuple[float, float, float, float, float]:
    """3-coefficient weighted OLS: ``y = a + b_im*im + b_grad*grad + b_layer*layer``.

    Returns ``(a, b_im, b_grad, b_layer, in_sample_loss)`` where
    ``in_sample_loss`` is the mean absolute residual of the
    weighted fit.

    Implementation: numpy ``lstsq`` on the design matrix
    ``[1, im, grad, layer]`` with the sample weights folded
    into the rows (sqrt-weight the rows so the unweighted
    least-squares recovers the weighted least-squares
    solution; this is the standard trick). The lstsq is
    numerically stable for the 4x4 normal-equation
    inversion; the closed-form inverse via
    ``(X^T W X)^-1 X^T W y`` would be brittle on near-singular
    inputs (e.g. one of the components is constant across the
    group).

    On a degenerate input (all rows have the same (im, grad,
    layer) tuple) lstsq returns the minimum-norm solution;
    the residual is exactly the mean abs deviation of y from
    its mean.
    """
    import numpy as np
    n = len(y)
    if n < 1:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    X = np.column_stack([
        np.ones(n, dtype=np.float64),
        np.asarray(im, dtype=np.float64),
        np.asarray(grad, dtype=np.float64),
        np.asarray(layer, dtype=np.float64),
    ])
    y_arr = np.asarray(y, dtype=np.float64)
    if sample_weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weights, dtype=np.float64)
    # sqrt-weight: minimise sum(w_i * r_i^2) by least-squares
    # on (sqrt(w) * X) and (sqrt(w) * y). The lstsq solution
    # is the weighted least-squares estimator.
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y_arr * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    a, b_im, b_grad, b_layer = (float(v) for v in beta)
    pred = a + b_im * X[:, 1] + b_grad * X[:, 2] + b_layer * X[:, 3]
    resid = y_arr - pred
    in_sample_loss = float(np.mean(np.abs(resid)))
    return (a, b_im, b_grad, b_layer, in_sample_loss)


def _ols_slope_intercept(
    x: list[float], y: list[float],
) -> tuple[float, float, float]:
    """Closed-form 2-coefficient OLS: ``y = a + b*x``.

    Returns ``(a, b, mean_abs_residual)``. When the input has
    fewer than 2 distinct x values, the slope is 0 and the
    intercept is the mean of y. This is the Phase-12
    fallback for pre-Phase-15 l5_outcome rows (no per-tensor
    component columns); the 3-coefficient OLS in
    :py:func:`_ols_3coef_weighted` is the production path.
    """
    import numpy as np
    if len(x) < 2:
        return (float(np.mean(y)) if y else 0.0, 0.0, 0.0)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    xm = x_arr.mean()
    ym = y_arr.mean()
    dx = x_arr - xm
    if np.dot(dx, dx) == 0.0:
        # x is constant -> slope is 0, intercept is mean(y).
        return (float(ym), 0.0, float(np.abs(y_arr - ym).mean()))
    b = float(np.dot(dx, y_arr - ym) / np.dot(dx, dx))
    a = float(ym - b * xm)
    residual = y_arr - (a + b * x_arr)
    return (a, b, float(np.abs(residual).mean()))


def _project_simplex(
    w: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Project a 3-vector onto the (w >= 0, sum = 1) simplex.

    The classical Duchi et al. (2008) sort-and-cumsum algorithm
    is overkill for n=3; a 3-loop is fine and more readable.
    Negative entries are clipped to 0; if the sum collapses to
    0 (every weight is negative), the result is the uniform
    distribution (1/3, 1/3, 1/3).
    """
    w_clipped = [max(0.0, float(x)) for x in w]
    s = sum(w_clipped)
    if s <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w_clipped[0] / s, w_clipped[1] / s, w_clipped[2] / s)


def _table_has_column(
    db: "TesseraDB", table: str, column: str,
) -> bool:
    """Cheap column-existence probe against information_schema.

    Used by the retune's SELECT to skip Phase 15 columns
    that are not yet present on the table (DBs created before
    Phase 15). DuckDB's information_schema.columns has a row
    per (table, column) pair; a row count > 0 means the
    column exists. The query is cheap (one row read) and
    cached at the connection level.
    """
    try:
        rows = db._conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? "
            "AND column_name = ? LIMIT 1",
            [table, column],
        ).fetchall()
        return len(rows) > 0
    except Exception:
        return False


def _compute_top_fraction(
    base: float, slope: float, hit_rate: float,
) -> float:
    """Per-family top_fraction recommendation (Phase 15).

    The rule: ``top_fraction = base * (1 + tanh(2*slope) * (1 - hit_rate))``.

    The tanh squashes the slope to ``(-1, 1)`` so a
    pathological slope doesn't push the top_fraction off the
    chart; the ``1 - hit_rate`` gate is the same "don't fix
    what isn't broken" guard as the per-component shift
    (high hit rate -> keep the base). The output is clipped
    to ``[0, 1]`` to defend against the tanh tail + gate
    combination overshooting.

    A negative slope (sensitivity_score predicts *negative*
    delta_mse; the ranking is correct) yields a *lower*
    top_fraction (we're being too aggressive, ease off). A
    positive slope with low hit rate yields a *higher*
    top_fraction (the miscalibrated family needs more
    coverage).
    """
    if base <= 0.0:
        return 0.0
    gate = max(0.0, min(1.0, 1.0 - hit_rate))
    delta = math.tanh(2.0 * float(slope)) * gate
    val = float(base) * (1.0 + delta)
    return max(0.0, min(1.0, val))


def _compute_requant_budget(
    family_storage_bits: int | None,
    hit_rate: float,
    base_budget_fraction: float,
) -> int | None:
    """Per-(model, model_role, family) dispatch-side bit budget.

    The rule:
        budget = family_storage_bits * (1 - hit_rate) * fraction

    ``family_storage_bits`` is the family's reference footprint
    from tensor_stats (sum of n_elements * dtype_bits). The
    ``1 - hit_rate`` gate is the same "don't fix what isn't
    broken" guard as the shift and top_fraction rules: a
    converged family (hit_rate -> 1) gets budget -> 0 (the next
    requant pass should not grow it); a fully miscalibrated
    family (hit_rate = 0) may spend up to ``fraction`` of its
    reference footprint. Returns None (the consumer treats it as
    "no budget constraint") when the family has no tensor_stats
    storage rows (family_storage_bits is None) or the fraction
    is non-positive. The result is floored at 0 and truncated to
    an integer bit count.
    """
    if family_storage_bits is None:
        return None
    if base_budget_fraction <= 0.0:
        return None
    gate = max(0.0, min(1.0, 1.0 - hit_rate))
    budget = float(family_storage_bits) * gate * float(base_budget_fraction)
    return max(0, int(budget))


def _family_storage_bits_from_stats(
    db: "TesseraDB",
    model_hash: str,
    model_role: str,
    family: str,
) -> int | None:
    """Sum n_elements * dtype_bits over a family's tensor_stats rows.

    Returns None when the family has no tensor_stats rows (or no
    rows with a recognized dtype), so the caller emits a NULL
    budget rather than a spurious 0. dtype strings that are not
    in DTYPE_BITS are skipped rather than poisoning the sum.
    """
    try:
        con = db._conn
        res = con.execute(
            "SELECT n_elements, dtype FROM tensor_stats "
            "WHERE model_hash = ? AND model_role = ? AND family = ?",
            [model_hash, model_role, family],
        ).fetchall()
    except Exception:
        # tensor_stats may be missing or pre-date the family
        # column on older DBs; a NULL budget is the safe
        # fallback (the consumer treats it as unconstrained).
        return None
    total = 0
    saw_any = False
    for n_elements, dtype in res:
        bpe = _dtype_bits(dtype)
        if bpe is None or n_elements is None:
            continue
        total += int(n_elements) * int(bpe)
        saw_any = True
    return total if saw_any else None


# Retune follow-ups: the per-(model, family) cross-component
# coupling score. A high score means the trunk's per-layer
# hit_rate and the dflash encoder's per-layer hit_rate move
# together (a tensor that is miscalibrated in the trunk is
# also miscalibrated in the dflash, by the same per-layer
# factor). A low (or negative) score means the two roles'
# miscalibration is independent.
#
# The retune surfaces the score on the l5_weights row so the
# consumer can see whether a single retune covers both roles.
# The retune's shift rule does not depend on the score (the
# OLS already accounts for per-(model, model_role, family)
# effects), so the score is informational, not action-bound.
COUPLING_DEFAULT_ROLE_A: str = "trunk"
COUPLING_DEFAULT_ROLE_B: str = "dflash"
# Minimum per-role per-layer point count for the score to be
# well-defined. Below this the correlation is too noisy.
COUPLING_MIN_LAYERS: int = 2


def _compute_coupling_score(
    df: pl.DataFrame,
    *,
    model_hash: str,
    family: str,
    role_a: str = COUPLING_DEFAULT_ROLE_A,
    role_b: str = COUPLING_DEFAULT_ROLE_B,
    min_layers: int = COUPLING_MIN_LAYERS,
) -> float | None:
    """Cross-component coupling score for one (model, family).

    For the (model_hash, family), gather the l5_outcome rows
    for both ``role_a`` and ``role_b`` (default trunk /
    dflash). For each role, group by ``layer`` and compute
    the per-layer hit_rate (the mean of ``plan_accepted``).
    The score is the Pearson correlation between the two
    roles' per-layer hit rates, computed on the inner join
    of the per-layer tables (a layer present in only one
    role is dropped; we don't synthesise a hit rate from a
    missing role).

    Returns ``None`` when:
      - the family has rows for only one of the two roles
        (correlation is undefined for a single series).
      - either role has fewer than ``min_layers`` distinct
        layers (the correlation is too noisy to act on; the
        rule of thumb is at least 2-3 distinct points).
      - either role's per-layer hit rates have zero variance
        (the correlation is mathematically undefined; the
        result is NaN which is surfaced as ``None`` to keep
        the SQL column clean).

    The helper is called at the (model, family) level, not
    per role: the same score is shared by the
    (model, role_a, family) and (model, role_b, family)
    verdicts in a multi-role retune (a high score means
    "a single retune covers both roles"; a low score means
    "the two roles need independent retune").

    Args:
      df: the polars DataFrame the retune is reading from
        (must carry ``model_hash``, ``model_role``,
        ``layer``, ``family``, ``plan_accepted``).
      model_hash: the model the score is for.
      family: the family the score is for.
      role_a: the first role (default ``"trunk"``).
      role_b: the second role (default ``"dflash"``).
      min_layers: minimum per-role distinct layer count.

    Returns:
      The Pearson correlation in ``[-1.0, 1.0]``, or
      ``None`` when undefined.
    """
    sub = df.filter(
        (pl.col("model_hash") == model_hash)
        & (pl.col("family") == family)
        & pl.col("model_role").is_in([role_a, role_b])
        & pl.col("layer").is_not_null()
        & pl.col("plan_accepted").is_not_null()
    )
    if sub.height == 0:
        return None
    # The retune only knows about the per-(model, family)
    # group when the partition has rows for both roles. A
    # single-role partition returns None.
    roles_present = set(sub["model_role"].unique().to_list())
    if role_a not in roles_present or role_b not in roles_present:
        return None
    per_layer = sub.group_by(["model_role", "layer"]).agg(
        pl.col("plan_accepted").cast(pl.Float64).mean().alias("hit_rate")
    )
    a = per_layer.filter(pl.col("model_role") == role_a).sort("layer")
    b = per_layer.filter(pl.col("model_role") == role_b).sort("layer")
    # Inner-join on layer: only the layers both roles saw.
    aligned = a.join(
        b.select([pl.col("layer"), pl.col("hit_rate").alias("hit_rate_b")]),
        on="layer", how="inner",
    )
    if aligned.height < min_layers:
        return None
    import numpy as np
    a_vals = aligned["hit_rate"].to_numpy()
    b_vals = aligned["hit_rate_b"].to_numpy()
    if a_vals.std() == 0.0 or b_vals.std() == 0.0:
        return None
    # np.corrcoef returns the 2x2 matrix; the [0, 1] entry
    # is the correlation between the two series.
    corr = float(np.corrcoef(a_vals, b_vals)[0, 1])
    if corr != corr:  # NaN guard (defensive)
        return None
    return corr


# Retune follow-ups: cross-model dedup. The retune's
# --retune-from-db path looks up l5_weights by
# (model_hash, model_role, family). When the requested
# model_hash is not in the table, a fallback to a different
# model with a very similar per-tensor stat distribution
# is sometimes reasonable: a fine-tune of the same base
# model that re-uses the same retune weights is fine, a
# different architecture is not. The fingerprint is a
# short hash of the first N moments of the per-tensor
# (kurtosis, eff_rank, rms, mean_abs, tail_ratio)
# distributions.
#
# The fingerprint is intentionally coarse (5-10 moments
# rounded to 4 sig figs): it discriminates architectures
# (which have very different per-tensor stats) but
# accepts fine-tunes of the same base (which have nearly
# identical per-tensor stats). The dedup is opt-in via
# --cross-model-dedup; the default is "no dedup, fall
# back to the --w-* flag values".
FINGERPRINT_STAT_COLS: tuple[str, ...] = (
    "kurtosis", "eff_rank", "rms", "mean_abs", "tail_ratio",
)
# Number of significant figures used to round each moment
# before hashing. 4 sig figs tolerates small numerical
# drift (e.g. quantisation noise on rms) but rejects
# different architectures (which differ in the first
# significant figure on at least one moment).
FINGERPRINT_SIG_FIGS: int = 4
# Maximum number of distinct models we read when looking
# for a fingerprint match. Bounds the read cost on a DB
# with many models.
FINGERPRINT_MAX_MODELS: int = 256


def _model_hash_fingerprint(
    db: "TesseraDB",
    model_hash: str,
    *,
    model_role: str = "trunk",
    n_moments: int = 2,
) -> str | None:
    """Compute a stable fingerprint of a model's per-tensor
    stat distribution.

    The fingerprint is a SHA-1 (truncated to 16 hex chars)
    of the first ``n_moments`` central moments of each
    column in :py:data:`FINGERPRINT_STAT_COLS`. With
    ``n_moments=2`` (the default), the fingerprint is the
    hash of (mean, std) for each of the 5 stat columns ->
    10 numbers. The numbers are rounded to
    :py:data:`FINGERPRINT_SIG_FIGS` sig figs before
    hashing so small numerical drift does not break the
    match.

    Two models with the same fingerprint have very
    similar per-tensor stat distributions (a fine-tune of
    the same base, or two random seeds of the same
    architecture with the same training data). Two
    different architectures have different fingerprints.

    Returns ``None`` when:
      - the tensor_stats table is missing (the DB was
        created before the unified schema).
      - the model has no tensor_stats rows (the
        calibration side never ran).
      - all the stat columns are NULL (the C++ side
        hasn't written kurtosis / eff_rank yet and the
        Python side hasn't written rms / mean_abs /
        tail_ratio yet).

    Args:
      db: an open TesseraDB instance (the function does
        not close it).
      model_hash: the model to fingerprint.
      model_role: the architectural role to fingerprint
        (Phase 16; the per-role tensor_stats are
        independent). Default "trunk".
      n_moments: the number of central moments per stat
        column (1 = mean only, 2 = mean + std, 3 = mean
        + std + skew, 4 = mean + std + skew + kurt).
        Default 2 (mean + std).

    Returns:
      A 16-char hex string, or ``None`` when the
      fingerprint is undefined.
    """
    names = set(db.table_names())
    if "tensor_stats" not in names:
        return None
    has_model_role = _table_has_column(db, "tensor_stats", "model_role")
    cols = ", ".join(FINGERPRINT_STAT_COLS)
    role_filter = ""
    if has_model_role:
        cols = "model_role, " + cols
        role_filter = f" AND model_role = '{sql_escape(model_role)}'"
    df = db.query(
        f"SELECT {cols} FROM tensor_stats "
        f"WHERE model_hash = '{sql_escape(model_hash)}'" + role_filter
    )
    if df.height == 0:
        return None
    # Compute the first n_moments of each stat column.
    # numpy mean / std / skew / kurtosis are well-defined
    # for n_moments in {1, 2, 3, 4}. The skew/kurt helpers
    # require at least 3 non-null values; we filter NULLs
    # per column.
    import hashlib
    import math
    digest_input: list[str] = []
    has_any = False
    for col in FINGERPRINT_STAT_COLS:
        if col not in df.columns:
            digest_input.append("nan")
            continue
        vals = [v for v in df[col].to_list() if v is not None]
        if not vals:
            digest_input.append("nan")
            continue
        has_any = True
        arr = vals
        mean_v = sum(arr) / len(arr)
        digest_input.append(f"{col}:mean={_round_sig(mean_v, FINGERPRINT_SIG_FIGS)}")
        if n_moments >= 2:
            if len(arr) > 1:
                var_v = sum((x - mean_v) ** 2 for x in arr) / (len(arr) - 1)
                std_v = math.sqrt(max(0.0, var_v))
                digest_input.append(
                    f"{col}:std={_round_sig(std_v, FINGERPRINT_SIG_FIGS)}"
                )
            else:
                digest_input.append(f"{col}:std=nan")
        if n_moments >= 3:
            # skew: E[(X - mean)^3] / std^3. Skip when std=0.
            if len(arr) > 2 and std_v > 0.0:
                m3 = sum((x - mean_v) ** 3 for x in arr) / len(arr)
                skew_v = m3 / (std_v ** 3)
                digest_input.append(
                    f"{col}:skew={_round_sig(skew_v, FINGERPRINT_SIG_FIGS)}"
                )
            else:
                digest_input.append(f"{col}:skew=nan")
        if n_moments >= 4:
            # kurt: E[(X - mean)^4] / std^4 - 3 (excess kurt).
            if len(arr) > 3 and std_v > 0.0:
                m4 = sum((x - mean_v) ** 4 for x in arr) / len(arr)
                kurt_v = m4 / (std_v ** 4) - 3.0
                digest_input.append(
                    f"{col}:kurt={_round_sig(kurt_v, FINGERPRINT_SIG_FIGS)}"
                )
            else:
                digest_input.append(f"{col}:kurt=nan")
    if not has_any:
        return None
    raw = "|".join(digest_input).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _round_sig(x: float, sig_figs: int) -> str:
    """Round ``x`` to ``sig_figs`` significant figures and
    format as a decimal string. Used by the fingerprint
    hash so 0.001234 and 0.001233 round to the same
    string.

    Implementation: explicit sig-figs rounding rather
    than the ``g`` format's fixed-decimals rounding (the
    latter rounds to ``decimals`` digits after the
    point, which is a different operation for tiny
    values; e.g. 0.0977765 with ``.4g`` gives 0.09778
    while a hand-rolled decimals-of-5 gives 0.097777).
    The hand-rolled path here rounds to the nearest
    ``10^(mag - sig_figs + 1)`` and formats with
    ``%g`` to drop trailing zeros.

    Returns "0" for x == 0.
    """
    if x == 0.0:
        return "0"
    import math
    digits = max(1, int(sig_figs))
    magnitude = math.floor(math.log10(abs(x)))
    # The sig-figs rounding step is 10^(magnitude - digits + 1).
    # round() to the nearest multiple of that step.
    step = 10.0 ** (magnitude - digits + 1)
    rounded = round(x / step) * step
    # Format with %g for the right number of sig figs.
    # %g uses the number of significant digits from the
    # precision specifier; precision = digits.
    return f"{rounded:.{digits}g}"


def find_fingerprint_match(
    db_path: str | Path,
    model_hash: str,
    *,
    model_role: str = "trunk",
    n_moments: int = 2,
) -> str | None:
    """Find a different model whose tensor_stats
    distribution matches ``model_hash``'s.

    Used by the orchestrator's --retune-from-db path
    when the requested model_hash is not in the l5_weights
    table. If a different model with the same fingerprint
    is found, that model's l5_weights can be reused (the
    caller is responsible for verifying the l5_weights
    actually exist for the matched model_hash).

    Returns the matched model_hash, or ``None`` when no
    match is found. The function reads every model's
    tensor_stats; the read is bounded by
    :py:data:`FINGERPRINT_MAX_MODELS` (a soft cap that
    prevents the dedup from being a full table scan on
    a huge multi-model DB).
    """
    if not Path(db_path).is_file():
        return None
    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "tensor_stats" not in names:
            return None
        target_fp = _model_hash_fingerprint(
            db, model_hash, model_role=model_role, n_moments=n_moments,
        )
        if target_fp is None:
            return None
        has_model_role = _table_has_column(db, "tensor_stats", "model_role")
        role_filter = ""
        if has_model_role:
            role_filter = f" WHERE model_role = '{sql_escape(model_role)}'"
        # Read every distinct model_hash (limit by
        # FINGERPRINT_MAX_MODELS to bound the read on
        # huge DBs).
        models_df = db.query(
            "SELECT DISTINCT model_hash FROM tensor_stats"
            + role_filter
            + f" LIMIT {FINGERPRINT_MAX_MODELS}"
        )
        for cand in models_df["model_hash"].to_list():
            cand_str = str(cand)
            if cand_str == model_hash:
                continue
            cand_fp = _model_hash_fingerprint(
                db, cand_str, model_role=model_role, n_moments=n_moments,
            )
            if cand_fp is not None and cand_fp == target_fp:
                return cand_str
    return None


def _confidence_weight(
    n_samples: int, in_sample_loss: float, max_n_samples: int,
    loss_scale: float = DEFAULT_LOSS_SCALE,
) -> float:
    """Sample weight for the weighted 3-coefficient OLS (Phase 15).

    The rule:
      weight = 1 / (1 + in_sample_loss * loss_scale)
               * sqrt(n_samples / max_n_samples)

    The in_sample_loss term damps rows whose post-fit loss is
    high (the retune doesn't trust them). The sqrt(n_samples)
    term rewards rows with more data (a sub-linear reward to
    avoid letting a single model with 100x the data of all
    others dominate the fit). Both terms are bounded in
    ``(0, 1]`` so the weight is bounded in ``(0, 1]``.

    Args:
      n_samples: the per-row sample count. The retune passes
        ``n_samples`` of the family the row was scored on
        (i.e. the number of plans the family had at the
        iteration of the row). The retune's grouping by
        (model, family) does not collapse this further: the
        retune uses the per-row ``n_samples`` field that the
        upstream pipeline recorded.
      in_sample_loss: the per-row in-sample loss of the
        3-coefficient fit. ``0.0`` means a perfect fit; the
        loss_scale x 100 conversion makes a 0.01 loss halve
        the weight and a 0.1 loss cut it to ~10%.
      max_n_samples: the maximum n_samples across the rows in
        the current fit. Used to normalise the n_samples
        term to ``[0, 1]``.
      loss_scale: the per-loss unit weight. Default 100.0
        (matches the task spec's example).

    Returns:
      The weight, in ``(0, 1]``.
    """
    n = max(0, int(n_samples))
    l = max(0.0, float(in_sample_loss))
    m = max(1, int(max_n_samples))
    loss_term = 1.0 / (1.0 + l * float(loss_scale))
    n_term = math.sqrt(n / float(m)) if m > 0 and n > 0 else 0.0
    return float(loss_term * n_term)


def _retune_family(
    *,
    model_hash: str,
    family: str,
    sensitivity: list[float],
    delta_mse: list[float],
    plan_accepted: list[bool],
    components: list[tuple[float, float, float] | None] | None = None,
    in_sample_losses: list[float] | None = None,
    n_samples_per_row: list[int] | None = None,
    base_weights: tuple[float, float, float] = DEFAULT_BASE_WEIGHTS,
    base_top_fraction: float = DEFAULT_BASE_TOP_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    model_role: str = "trunk",
    coupling_score: float | None = None,
    family_storage_bits: int | None = None,
    base_budget_fraction: float = DEFAULT_BASE_BUDGET_FRACTION,
) -> FamilyWeights:
    """Retune one (model, model_role, family) group.

    Phase 15: when the per-tensor components are available, fit
    a 3-coefficient OLS with sample weights derived from
    in_sample_loss and n_samples. When the components are
    missing, fall back to the 2-coefficient OLS on the
    combined sensitivity_score.

    Phase 16: the ``model_role`` is the architectural role
    (``"trunk"``, ``"dflash"``, ``"dspark"``, ``"mtp_nextn"``,
    ``"shared_embd"``). The same ``family`` in different
    roles gets independent retune verdicts: the trunk's
    ``attn_q`` and the dflash encoder's ``attn_q`` may have
    very different (w_imatrix, w_gradient, w_layer) optimums
    because the data they fit is different (the dflash
    encoder consumes trunk hidden states; its calibration
    is independent). Defaults to ``"trunk"`` for backward
    compat with Phase 15 callers.

    Retune follow-ups: the ``coupling_score`` is the
    Pearson correlation of the per-layer hit_rate between
    the trunk's role and the dflash encoder's role for the
    same family. The caller computes the score at the
    (model_hash, family) level (so the same score is
    shared by the trunk/attn_q and dflash/attn_q verdicts
    in a multi-role retune) and passes the value through.
    The verdict stores the score; the l5_weights row
    surfaces it for the consumer.

    The shift rule (3-coefficient path):
        shift = alpha * (b_im, b_grad, b_layer) * (1 - hit_rate)
        new_w = base + shift
        project onto simplex

    The shift rule (2-coefficient path, fallback):
        shift = alpha * b * (1 - hit_rate)
        new_w = (w_im - shift, w_grad + shift, w_layer)
        project onto simplex

    The per-family top_fraction recommendation uses the
    dominant slope. For the 3-coefficient path the "dominant
    slope" is the sum of the squared coefficients (the L2
    norm of (b_im, b_grad, b_layer)); for the 2-coefficient
    path it is the single slope ``b``.
    """
    role = str(model_role) if model_role else "trunk"
    n = len(sensitivity)
    if n < 1:
        return FamilyWeights(
            model_hash=model_hash, model_role=role, family=family,
            weights=base_weights,
            bias=0.0, slopes=(0.0, 0.0, 0.0),
            n_samples=0, in_sample_loss=0.0, hit_rate=0.0,
            top_fraction=None, coupling_score=coupling_score,
            requant_budget_bits=None,
            retune_algorithm=RETUNE_SOURCE_TAG_2COEF,
            was_acted_on=False,
        )
    n_accepted = sum(1 for v in plan_accepted if v)
    hit_rate = n_accepted / n if n else 0.0
    if n < min_samples:
        return FamilyWeights(
            model_hash=model_hash, model_role=role, family=family,
            weights=base_weights,
            bias=0.0, slopes=(0.0, 0.0, 0.0),
            n_samples=n, in_sample_loss=0.0, hit_rate=hit_rate,
            top_fraction=None, coupling_score=coupling_score,
            requant_budget_bits=None,
            retune_algorithm=RETUNE_SOURCE_TAG_2COEF,
            was_acted_on=False,
        )

    # Compute the sample weights. ``in_sample_losses`` is
    # optional (Phase-12 l5_outcome rows do not have it; the
    # retune passes 0.0 in that case so the loss term is 1.0
    # and only the n_samples term contributes).
    if in_sample_losses is None:
        in_sample_losses = [0.0] * n
    if n_samples_per_row is None:
        n_samples_per_row = [1] * n
    max_n_samples = max(n_samples_per_row) if n_samples_per_row else 1
    sample_weights = [
        _confidence_weight(
            n_samples=n_samples_per_row[i],
            in_sample_loss=in_sample_losses[i],
            max_n_samples=max_n_samples,
        )
        for i in range(n)
    ]
    # Avoid zero-weight rows: the lstsq tolerates them
    # (sqrt(0) is 0; the row is dropped from the design
    # matrix effectively) but the residual is computed
    # against the unweighted y, which is the correct
    # in_sample_loss definition. Floor the weights at a tiny
    # positive value to keep the lstsq well-conditioned.
    eps = 1e-12
    sample_weights = [max(eps, w) for w in sample_weights]

    # 3-coefficient path when components are available for
    # every row. A mixed cohort (some rows have components,
    # some don't) is a corner case: the per-row components
    # are nullable, and the 3-coefficient OLS needs all
    # three. We fall back to the 2-coefficient OLS for mixed
    # cohorts (the more conservative choice; the 3-coefficient
    # path's value comes from having a clean per-component
    # signal).
    use_3coef = (
        components is not None
        and len(components) == n
        and all(c is not None for c in components)
    )
    gate = 1.0 - hit_rate
    if use_3coef:
        im = [c[0] for c in components]   # type: ignore[index]
        grad = [c[1] for c in components]  # type: ignore[index]
        layer = [c[2] for c in components]  # type: ignore[index]
        a, b_im, b_grad, b_layer, in_sample_loss = (
            _ols_3coef_weighted(
                im, grad, layer, delta_mse,
                sample_weights=sample_weights,
            )
        )
        # Shift: per-component, gated by hit_rate.
        shift_im = alpha * b_im * gate
        shift_grad = alpha * b_grad * gate
        shift_layer = alpha * b_layer * gate
        new_w = (
            base_weights[0] + shift_im,
            base_weights[1] + shift_grad,
            base_weights[2] + shift_layer,
        )
        projected = _project_simplex(new_w)
        # Dominant slope: the L2 norm of the coefficient
        # vector. The sign of the dominant slope matters
        # for the top_fraction formula; we use the L2 norm
        # as a positive magnitude and let tanh squash the
        # dynamics. The orchestrator's diagnostics read the
        # individual slopes from l5_weights["slope"] (b_grad
        # for compatibility with the 2-coefficient path);
        # the per-component shifts are encoded in the
        # recommended (w_im, w_grad, w_layer) tuple.
        dominant_slope = math.sqrt(
            b_im * b_im + b_grad * b_grad + b_layer * b_layer
        )
        return FamilyWeights(
            model_hash=model_hash, model_role=role, family=family,
            weights=projected,
            bias=a,
            slopes=(b_im, b_grad, b_layer),
            n_samples=n,
            in_sample_loss=in_sample_loss,
            hit_rate=hit_rate,
            top_fraction=_compute_top_fraction(
                base_top_fraction, dominant_slope, hit_rate,
            ),
            coupling_score=coupling_score,
            requant_budget_bits=_compute_requant_budget(
                family_storage_bits, hit_rate, base_budget_fraction,
            ),
            retune_algorithm=RETUNE_SOURCE_TAG_3COEF,
            was_acted_on=True,
        )

    # 2-coefficient fallback.
    a, b, in_sample_loss = _ols_slope_intercept(
        sensitivity, delta_mse,
    )
    shift = alpha * b * gate
    new_w = (
        base_weights[0] - shift,
        base_weights[1] + shift,
        base_weights[2],
    )
    projected = _project_simplex(new_w)
    return FamilyWeights(
        model_hash=model_hash, model_role=role, family=family,
        weights=projected,
        bias=a,
        slopes=(0.0, b, 0.0),
        n_samples=n,
        in_sample_loss=in_sample_loss,
        hit_rate=hit_rate,
        top_fraction=_compute_top_fraction(
            base_top_fraction, abs(b), hit_rate,
        ),
        coupling_score=coupling_score,
        requant_budget_bits=_compute_requant_budget(
            family_storage_bits, hit_rate, base_budget_fraction,
        ),
        retune_algorithm=RETUNE_SOURCE_TAG_2COEF,
        was_acted_on=True,
    )


def compute_l5_weights(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    model_role: str | None = None,
    base_weights: tuple[float, float, float] = DEFAULT_BASE_WEIGHTS,
    base_top_fraction: float = DEFAULT_BASE_TOP_FRACTION,
    base_budget_fraction: float = DEFAULT_BASE_BUDGET_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    use_ema: bool = True,
    write_back: bool = True,
) -> list[FamilyWeights]:
    """Read ``l5_outcome``, run the per-(model, model_role,
    family) retune, and (optionally) write the result to
    ``l5_weights``.

    Args:
      db_path: path to the unified tessera.duckdb file.
      model_hash: if non-None, restrict to this model. Default
        None = all models in the DB.
      model_role: if non-None, restrict to this architectural
        role. Default None = all roles in the DB. Phase 16.
        When set, the partition is
        ``(model_hash, model_role, family)`` and the write-back
        deletes only rows with the same
        ``(model_hash, model_role)`` tuple. A ``None`` value
        means "all roles" (the legacy pre-Phase-16 path); the
        write-back then DELETEs all rows for the model
        regardless of role. The ``model_role`` filter on the
        SELECT is only applied when both ``model_hash`` AND
        ``model_role`` are given (a bare ``model_role`` filter
        is ambiguous without a model; the conservative
        behaviour is to read all roles for the model).
      base_weights: the (w_imatrix, w_gradient, w_layer) the
        retune perturbs. Default = l5_metrics.DEFAULT_WEIGHTS.
      base_top_fraction: the per-family top_fraction base; see
        :py:func:`_compute_top_fraction`. Default 0.10.
      alpha: shift aggressiveness; see the module docstring.
      min_samples: minimum per-(model, family) sample count
        for the OLS to be acted on.
      use_ema: when True, join with ``l5_plan_ema`` (when
        present) and use the EMA-tracked sensitivity for the
        OLS instead of the per-iteration sensitivity_score.
        Default True (the production path); the per-iteration
        fallback is for DBs that pre-date the EMA table.
      write_back: if True, write the result to l5_weights in
        a single transaction. If False, return the verdicts
        without writing.

    Returns:
      A list of FamilyWeights, one per (model, model_role,
      family) group seen in the join. The list is sorted by
      (model_hash, model_role, family) for stable output.
    """
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"tessera.duckdb not found: {db_path}")

    with TesseraDB.open(db_path) as db:
        names = set(db.table_names())
        required = {"l5_outcome"}
        missing = required - names
        if missing:
            raise RuntimeError(
                f"unified schema is missing tables: {sorted(missing)}. "
                f"Run l5_outcome.py first (it produces l5_outcome)."
            )
        where_clauses: list[str] = []
        if model_hash:
            where_clauses.append(
                f"model_hash = '{sql_escape(model_hash)}'"
            )
        # Phase 16: only apply a model_role filter on the
        # SELECT when the caller supplied one. The legacy
        # pre-Phase-16 retune path (model_role=None) does
        # not filter; the l5_outcome rows are read for
        # every role, and the partition handles the role
        # dimension. (Pre-Phase-16 l5_outcome rows all have
        # model_role='trunk' or NULL via the column default.)
        if model_role is not None:
            where_clauses.append(
                f"model_role = '{sql_escape(model_role)}'"
            )
        where = (
            (" WHERE " + " AND ".join(where_clauses))
            if where_clauses else ""
        )
        # Phase 15: select the per-tensor component columns
        # alongside sensitivity_score. When they are NULL
        # (pre-Phase-15 rows or a DB the C++ side has not yet
        # migrated), the helper inside _retune_family falls
        # back to the 2-coefficient OLS on the combined
        # sensitivity_score.
        #
        # The SELECT is built defensively: each Phase 15
        # column is gated by a column-existence probe so DBs
        # that pre-date the column addition do not error at
        # SELECT time. The result has the column present
        # (always NULL on pre-Phase-15 DBs) when the column
        # exists; the helper's NULL handling then routes to
        # the 2-coefficient fallback.
        has_im = _table_has_column(db, "l5_outcome", "imatrix_magnitude")
        has_grad = _table_has_column(db, "l5_outcome", "gradient_proxy")
        has_layer = _table_has_column(
            db, "l5_outcome", "layer_position_prior"
        )
        has_in_sample_loss = _table_has_column(
            db, "l5_outcome", "in_sample_loss",
        )
        # Phase 16: model_role is the architectural role
        # (trunk / dflash / dspark / mtp_nextn / shared_embd).
        # Read it from l5_outcome so the partition can use it
        # as a groupby key. The column-existence probe is
        # defensive: pre-Phase-16 DBs do not have the
        # column; the backfill below substitutes a uniform
        # "trunk" string for those rows (the pre-Phase-16
        # behavior was a single (model, family) group, not
        # role-aware, so the partition still produces a
        # single group per (model, family) with the
        # synthetic role).
        has_model_role = _table_has_column(
            db, "l5_outcome", "model_role",
        )
        col_list = (
            "model_hash, family, sensitivity_score, "
            "delta_mse, plan_accepted, layer"
            + (", imatrix_magnitude" if has_im else "")
            + (", gradient_proxy" if has_grad else "")
            + (", layer_position_prior" if has_layer else "")
            + (", in_sample_loss" if has_in_sample_loss else "")
            + (", model_role" if has_model_role else "")
        )
        df = db.query(
            f"SELECT {col_list} FROM l5_outcome" + where
        )
        # Backfill any missing component / in_sample_loss /
        # model_role columns so the rest of the retune can
        # use a uniform schema.
        for col in ("imatrix_magnitude", "gradient_proxy",
                    "layer_position_prior", "in_sample_loss"):
            if col not in df.columns:
                df = df.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(col)
                )
        if "model_role" not in df.columns:
            # Pre-Phase-16 DB: every l5_outcome row is the
            # legacy trunk. Substitute a uniform "trunk"
            # string so the partition's role key is a
            # constant within a model. The retune still
            # produces a (model, family) verdict tagged with
            # model_role='trunk' (the legacy PK).
            df = df.with_columns(
                pl.lit("trunk", dtype=pl.Utf8).alias("model_role")
            )

        # EMA-aware path: optional join with l5_plan_ema. The
        # EMA table is additive; older DBs do not have it and
        # the retune falls back to the per-iteration
        # sensitivity_score.
        ema_join_applied = False
        if use_ema and "l5_plan_ema" in names and df.height > 0:
            try:
                ema_df = db.query(
                    "SELECT model_hash, name, iteration, plan_id, "
                    "ema_score FROM l5_plan_ema"
                    + (f" WHERE model_hash = '{sql_escape(model_hash)}'"
                       if model_hash else "")
                )
                if ema_df.height > 0:
                    # The l5_outcome / l5_plan_ema join key is
                    # the (model_hash, name, iteration, plan_id)
                    # tuple. The l5_plan_ema table doesn't carry
                    # a family column (the EMA is per-tensor);
                    # the join preserves the l5_outcome family's
                    # grouping.
                    df = df.join(
                        ema_df,
                        on=["model_hash", "name", "iteration", "plan_id"],
                        how="left",
                    )
                    # Replace the per-iteration sensitivity_score
                    # with the EMA value when the join hit; leave
                    # the per-iteration value when it missed (the
                    # l5_plan_ema table was not populated for that
                    # plan).
                    df = df.with_columns(
                        pl.coalesce([
                            pl.col("ema_score"),
                            pl.col("sensitivity_score"),
                        ]).alias("sensitivity_score")
                    ).drop("ema_score")
                    ema_join_applied = True
            except Exception as e:
                sys.stderr.write(
                    f"l5_retune: l5_plan_ema join failed; "
                    f"falling back to per-iteration "
                    f"sensitivity_score ({e.__class__.__name__}: "
                    f"{str(e)[:200]})\n"
                )

        # requant_budget_bits producer: the per-(model, role,
        # family) reference storage footprint from
        # tensor_stats (sum of n_elements * dtype_bits). The
        # retune multiplies it by (1 - hit_rate) *
        # base_budget_fraction per group in _retune_family.
        # Missing table / columns / rows degrade to an empty
        # map -> NULL budgets (the consumer treats NULL as
        # "no budget constraint").
        storage_map: dict[tuple[str, str, str], int] = {}
        if ("tensor_stats" in names
                and _table_has_column(db, "tensor_stats", "n_elements")
                and _table_has_column(db, "tensor_stats", "dtype")
                and _table_has_column(db, "tensor_stats", "family")):
            try:
                stats_df = db.query(
                    "SELECT model_hash, model_role, family, "
                    "n_elements, dtype FROM tensor_stats"
                    + (f" WHERE model_hash = '{sql_escape(model_hash)}'"
                       if model_hash else "")
                )
                for row in stats_df.iter_rows(named=True):
                    bpe = _dtype_bits(row.get("dtype"))
                    n_el = row.get("n_elements")
                    fam_val = row.get("family")
                    if bpe is None or n_el is None or fam_val is None:
                        continue
                    key = (
                        str(row["model_hash"]),
                        str(row.get("model_role") or "trunk"),
                        str(fam_val),
                    )
                    storage_map[key] = (
                        storage_map.get(key, 0) + int(n_el) * int(bpe)
                    )
            except Exception as e:
                sys.stderr.write(
                    f"l5_retune: tensor_stats storage aggregation "
                    f"failed; requant budgets will be NULL "
                    f"({e.__class__.__name__}: {str(e)[:200]})\n"
                )
                storage_map = {}

    if df.height == 0:
        return []

    # Retune follow-ups: per-(model_hash, family) cross-component
    # coupling score. The score is the Pearson correlation of
    # the per-layer hit_rate between the trunk's role and the
    # dflash encoder's role for the same family. The same
    # score is shared by both roles' verdicts in a multi-role
    # retune; a single-role retune has score = None (no
    # correlation possible). The score is computed at the
    # (model_hash, family) level, NOT at the
    # (model_hash, model_role, family) level.
    coupling_scores: dict[tuple[str, str], float | None] = {}
    if df.height > 0:
        for mh_val in df["model_hash"].unique().to_list():
            mh_str = str(mh_val)
            for fam_val in df.filter(pl.col("model_hash") == mh_str)[
                "family"
            ].unique().to_list():
                fam_str = str(fam_val)
                coupling_scores[(mh_str, fam_str)] = _compute_coupling_score(
                    df, model_hash=mh_str, family=fam_str,
                )

    # Per-(model, model_role, family) retune. partition_by is
    # stable in polars 0.20+, so the output is in
    # (model_hash, model_role, family) order. Phase 16: the
    # role is part of the groupby key so the trunk's
    # ``attn_q`` and the dflash encoder's ``attn_q`` get
    # independent retune verdicts.
    groups = df.partition_by(
        ["model_hash", "model_role", "family"],
        maintain_order=True,
    )
    verdicts: list[FamilyWeights] = []
    for g in groups:
        mh = str(g["model_hash"][0])
        # Phase 16: the role is part of the partition key.
        # NULL model_role (which can happen on pre-Phase-16
        # rows that we backfilled with "trunk") is normalised
        # to "trunk" so the verdict carries a real string.
        role_raw = g["model_role"][0]
        role = str(role_raw) if role_raw is not None else "trunk"
        if not role:
            role = "trunk"
        fam = str(g["family"][0])
        sens = [float(v) if v is not None else 0.0
                for v in g["sensitivity_score"].to_list()]
        deltas = [float(v) if v is not None else 0.0
                  for v in g["delta_mse"].to_list()]
        accepted = [bool(v) if v is not None else False
                    for v in g["plan_accepted"].to_list()]
        # The component columns: collect as a list of
        # (im, grad, layer) tuples, one per row. NULL becomes
        # None (the _retune_family helper treats None as
        # "fall back to 2-coefficient OLS").
        im_col = g["imatrix_magnitude"].to_list()
        grad_col = g["gradient_proxy"].to_list()
        layer_col = g["layer_position_prior"].to_list()
        components: list[tuple[float, float, float] | None] = []
        for i, (im, gr, la) in enumerate(zip(im_col, grad_col, layer_col)):
            if im is None and gr is None and la is None:
                # All three are NULL: pre-Phase-15 row or
                # older C++ writer. The 2-coefficient
                # fallback is the right path.
                components.append(None)
            elif im is None or gr is None or la is None:
                # A partial component (one or two NULLs)
                # is also treated as "fall back": the
                # 3-coefficient OLS needs all three. We
                # could impute the missing component with
                # the uniform-spread assumption
                # (l5_metrics.decompose) but the
                # conservative choice is the 2-coefficient
                # fallback. In practice the producer side
                # always populates all three together (the
                # SensitivityScorer.score() emits them as a
                # single set); a partial NULL would be a
                # bug upstream.
                components.append(None)
            else:
                components.append((float(im), float(gr), float(la)))
        # The per-row in_sample_loss is the upstream's loss
        # for that row (the post-fit L1 residual of the
        # sensitivity calibration). Phase 15: the confidence
        # weight uses this to damp rows whose residual is
        # high. Older DBs do not have the column; we use
        # 0.0 (no damping) in that case.
        if "in_sample_loss" in g.columns:
            in_sample_losses = [
                float(v) if v is not None else 0.0
                for v in g["in_sample_loss"].to_list()
            ]
        else:
            in_sample_losses = [0.0] * len(sens)
        # n_samples: per-row. Phase 12 does not write a
        # per-row n_samples (the per-(model, family) n_samples
        # is the count of rows); for the confidence weight we
        # use the family's n_samples as a constant. The
        # weight is then uniform within a family; the
        # in_sample_loss term still varies per row.
        n_samples_per_row = [len(sens)] * len(sens)
        # Retune follow-ups: lookup the coupling score for
        # this (model, family) group. The same score is
        # shared by the trunk's and dflash's verdicts (the
        # score is per (model, family), not per
        # (model, model_role, family)). A missing key
        # (which can happen on a single-role retune) maps
        # to None; the verdict stores None and the
        # l5_weights column is NULL.
        coupling = coupling_scores.get((mh, fam))
        # requant_budget_bits producer input: the family's
        # reference storage footprint (from the tensor_stats
        # aggregation above). A missing key (no tensor_stats
        # rows for the family) maps to None -> NULL budget.
        storage_bits = storage_map.get((mh, role, fam))
        verdicts.append(_retune_family(
            model_hash=mh, family=fam,
            model_role=role,
            sensitivity=sens, delta_mse=deltas,
            plan_accepted=accepted,
            components=components,
            in_sample_losses=in_sample_losses,
            n_samples_per_row=n_samples_per_row,
            base_weights=base_weights,
            base_top_fraction=base_top_fraction,
            alpha=alpha, min_samples=min_samples,
            coupling_score=coupling,
            family_storage_bits=storage_bits,
            base_budget_fraction=base_budget_fraction,
        ))

    if write_back and verdicts:
        rows_to_write = [v for v in verdicts if v.was_acted_on]
        if rows_to_write:
            with TesseraDB.open(db_path) as db:
                con = db._conn
                con.execute("BEGIN")
                try:
                    # Phase 16: the write-back DELETEs on
                    # (model_hash, model_role) so other roles
                    # for the same model are not clobbered.
                    # When model_role is None (the legacy
                    # pre-Phase-16 path), the DELETE is
                    # only keyed on model_hash (the whole
                    # model's rows are replaced); this is
                    # the conservative behaviour because a
                    # bare model without a role filter is
                    # ambiguous (the retune may have read
                    # multiple roles' rows).
                    if model_hash is not None and model_role is not None:
                        con.execute(
                            "DELETE FROM l5_weights "
                            f"WHERE model_hash = '{sql_escape(model_hash)}' "
                            f"AND model_role = '{sql_escape(model_role)}'"
                        )
                    elif model_hash is not None:
                        con.execute(
                            "DELETE FROM l5_weights "
                            f"WHERE model_hash = '{sql_escape(model_hash)}'"
                        )
                    else:
                        con.execute("DELETE FROM l5_weights")
                    db.insert_l5_weights(
                        rows=[v.to_dict() for v in rows_to_write],
                    )
                    con.execute("COMMIT")
                except Exception:
                    con.execute("ROLLBACK")
                    raise

    if ema_join_applied:
        sys.stderr.write(
            "l5_retune: used l5_plan_ema for EMA-aware OLS; "
            "set --no-ema to fall back to per-iteration "
            "sensitivity_score\n"
        )
    return verdicts


def write_cross_model_aggregate(
    db_path: str | Path,
    *,
    base_weights: tuple[float, float, float] = DEFAULT_BASE_WEIGHTS,
    base_top_fraction: float = DEFAULT_BASE_TOP_FRACTION,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[FamilyWeights]:
    """Read l5_weights grouped by (model_role, family) (across
    all models), write one row per (model_role, family) with
    ``model_hash = "*"``.

    The aggregate is the n_samples-weighted mean of the
    per-model rows, with the top_fraction and hit_rate
    aggregated similarly. The cross-model row is a
    generalization of the per-model row, not a replacement;
    the orchestrator's --retune-from-db falls back to the
    cross-model row when the per-model row is missing
    (warm-start new models from the cross-model mean).

    Phase 16: the cross-model aggregate is per-(model_role,
    family), not per-family. The trunk's ``attn_q`` and the
    dflash encoder's ``attn_q`` get independent
    cross-model rows; the orchestrator's
    ``--retune-from-db --model-role dflash`` looks up the
    dflash row independently of the trunk row. The
    cross-model ``model_role`` is the same string as the
    per-model rows it aggregates (e.g. ``"trunk"`` for
    trunk aggregates, ``"dflash"`` for dflash encoder
    aggregates).

    The aggregate is tagged with
    ``RETUNE_SOURCE_TAG_CROSSMODEL`` so the consumer can
    tell which path produced the row. The bias and
    in_sample_loss are aggregated as the n_samples-weighted
    mean of the per-model values; the slopes are aggregated
    as the n_samples-weighted mean of the b_grad component
    (the 2-coefficient path's primary slope; the 3-coefficient
    path's b_grad is the gradient coefficient which is the
    closest analog).

    Args:
      db_path: path to the unified tessera.duckdb file.
      base_weights: the (w_imatrix, w_gradient, w_layer) the
        aggregate falls back to when the table is empty.
      base_top_fraction: the per-family top_fraction base
        used for the aggregate's top_fraction (the
        formula is the same as the per-family path).
      min_samples: minimum total n_samples across models
        for a (model_role, family) pair to be aggregated.
        Below this the cross-model row is not written (the
        per-role n is too thin to warm-start from).

    Returns:
      A list of FamilyWeights, one per (model_role,
      family) the aggregate produced. The list is sorted
      by (model_role, family).
    """
    if not Path(db_path).is_file():
        return []
    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "l5_weights" not in names:
            return []
        # Read every row except the cross-model itself (we
        # don't want to fold the previous aggregate into the
        # new one). The model_role column is read for the
        # per-(model_role, family) grouping; pre-Phase-16
        # DBs do not have the column, so the backfill
        # below substitutes a uniform "trunk" string.
        has_model_role_col = _table_has_column(
            db, "l5_weights", "model_role",
        )
        # requant_budget_bits is a Phase-14 column; probe it so
        # pre-Phase-14 DBs (which pre-date the column) still
        # aggregate. When absent the aggregate's budget is NULL.
        has_budget_col = _table_has_column(
            db, "l5_weights", "requant_budget_bits",
        )
        cols = (
            "model_hash, family, w_imatrix, w_gradient, "
            "w_layer, bias, n_samples, in_sample_loss, hit_rate, "
            "top_fraction"
            + (", model_role" if has_model_role_col else "")
            + (", requant_budget_bits" if has_budget_col else "")
        )
        df = db.query(
            f"SELECT {cols} FROM l5_weights "
            f"WHERE model_hash != '*'"
        )
        if "model_role" not in df.columns:
            df = df.with_columns(
                pl.lit("trunk", dtype=pl.Utf8).alias("model_role")
            )
        if "requant_budget_bits" not in df.columns:
            # Pre-Phase-14 DB: no budget column. Backfill NULL
            # so the aggregate's budget is NULL (no constraint).
            df = df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("requant_budget_bits")
            )
    if df.height == 0:
        return []

    # Group by (model_role, family), aggregate. Polars'
    # group_by with multiple aggregations; the
    # n_samples-weighted mean is the standard sum(w*x) /
    # sum(w) expression. For the top_fraction we
    # aggregate only the rows that have a non-NULL
    # top_fraction (the per-family recommendation is
    # optional); when none of the rows have a top_fraction
    # the aggregate's top_fraction is NULL.
    import numpy as np
    agg_exprs = []
    for col in ("w_imatrix", "w_gradient", "w_layer",
                "bias", "in_sample_loss", "hit_rate"):
        # (col * n_samples).sum() / n_samples.sum().
        agg_exprs.append(
            ((pl.col(col) * pl.col("n_samples")).sum()
             / pl.col("n_samples").sum()).alias(col)
        )
    # top_fraction: n_samples-weighted mean over the rows
    # that have a non-NULL value. The expression
    # ``(top_fraction * n_samples).sum()`` is NULL when
    # top_fraction is NULL (any null * anything = null in
    # polars). We sum the weights separately to avoid
    # double-counting the rows with NULL top_fraction.
    agg_exprs.append(pl.col("n_samples").sum().alias("total_n"))
    aggregated = df.group_by(["model_role", "family"]).agg(agg_exprs)
    # top_fraction needs a separate pass because polars'
    # ``sum()`` over (top_fraction * n_samples) drops rows
    # where top_fraction is NULL. Compute it on the same
    # group_by with a different filter.
    top_frac_df = (
        df.filter(pl.col("top_fraction").is_not_null())
          .group_by(["model_role", "family"])
          .agg(
              ((pl.col("top_fraction") * pl.col("n_samples")).sum()
               / pl.col("n_samples").sum()).alias("top_fraction")
          )
    )
    aggregated = aggregated.join(
        top_frac_df, on=["model_role", "family"], how="left",
    )
    # requant_budget_bits: n_samples-weighted mean over the rows
    # that have a non-NULL budget (same NULL-skipping treatment
    # as top_fraction). When none of the per-model rows carry a
    # budget the aggregate's budget is NULL (unconstrained).
    budget_df = (
        df.filter(pl.col("requant_budget_bits").is_not_null())
          .group_by(["model_role", "family"])
          .agg(
              ((pl.col("requant_budget_bits") * pl.col("n_samples")).sum()
               / pl.col("n_samples").sum()).alias("requant_budget_bits")
          )
    )
    aggregated = aggregated.join(
        budget_df, on=["model_role", "family"], how="left",
    )
    # Filter by min_samples (total n across models).
    aggregated = aggregated.filter(pl.col("total_n") >= min_samples)

    verdicts: list[FamilyWeights] = []
    now_str = None  # filled in by TesseraDB
    rows_to_write: list[dict] = []
    for row in aggregated.iter_rows(named=True):
        role_raw = row.get("model_role")
        role = str(role_raw) if role_raw is not None else "trunk"
        if not role:
            role = "trunk"
        fam = str(row["family"])
        n_total = int(row["total_n"])
        if n_total < min_samples:
            continue
        projected = _project_simplex(
            (float(row["w_imatrix"]),
             float(row["w_gradient"]),
             float(row["w_layer"]))
        )
        top_frac_val = row.get("top_fraction")
        # Aggregate hit_rate (n_samples-weighted). Used by
        # the top_fraction formula when the aggregated
        # top_fraction is NULL.
        agg_hit_rate = float(row["hit_rate"])
        if top_frac_val is None:
            # The aggregate's top_fraction is the per-family
            # formula applied to the aggregated hit_rate and
            # the family's mean |b_grad| (we don't track
            # per-row slopes in l5_weights; the cross-model
            # aggregate is a coarse summary). Default to
            # base when we don't have a slope.
            top_frac_out = base_top_fraction
        else:
            top_frac_out = float(top_frac_val)
        # requant_budget_bits: the n_samples-weighted mean of
        # the per-model budgets (None when no per-model row
        # carried a budget). Truncated to an integer bit count.
        budget_val = row.get("requant_budget_bits")
        budget_out = int(budget_val) if budget_val is not None else None
        verdicts.append(FamilyWeights(
            model_hash="*", model_role=role, family=fam,
            weights=projected,
            bias=float(row["bias"]),
            slopes=(0.0, 0.0, 0.0),  # aggregated, not stored
            n_samples=n_total,
            in_sample_loss=float(row["in_sample_loss"]),
            hit_rate=agg_hit_rate,
            top_fraction=top_frac_out,
            # Retune follow-ups: cross-model aggregates do
            # not carry a per-model coupling score (the
            # score is per-(model_hash, family), and the
            # cross-model row has model_hash = "*"). The
            # column is left NULL.
            coupling_score=None,
            requant_budget_bits=budget_out,
            retune_algorithm=RETUNE_SOURCE_TAG_CROSSMODEL,
            was_acted_on=True,
        ))
        rows_to_write.append({
            "model_hash":      "*",
            "model_role":      role,
            "family":          fam,
            "w_imatrix":       projected[0],
            "w_gradient":      projected[1],
            "w_layer":         projected[2],
            "bias":            float(row["bias"]),
            "n_samples":       n_total,
            "in_sample_loss":  float(row["in_sample_loss"]),
            "hit_rate":        agg_hit_rate,
            "top_fraction":    top_frac_out,
            "requant_budget_bits": budget_out,
            "retune_source":   RETUNE_SOURCE_TAG_CROSSMODEL,
        })

    if rows_to_write:
        with TesseraDB.open(db_path) as db:
            con = db._conn
            con.execute("BEGIN")
            try:
                con.execute(
                    "DELETE FROM l5_weights WHERE model_hash = '*'"
                )
                db.insert_l5_weights(rows=rows_to_write)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
    return verdicts


def read_l5_weights(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    model_role: str | None = None,
    cross_model_fallback: bool = False,
) -> pl.DataFrame:
    """Read the l5_weights table for the consumer (the
    orchestrator's ``--retune-from-db`` path).

    Returns an empty DataFrame with the l5_weights schema when
    the table is missing or empty. When ``model_hash`` is given,
    the result is filtered to that model. When ``model_role``
    is given, the result is filtered to that role. When
    ``cross_model_fallback`` is True, rows with
    ``model_hash = "*"`` are appended for any family the
    per-model lookup missed; this is the orchestrator's
    "warm-start new model from cross-model mean" path.

    Phase 16: the ``model_role`` filter is part of the
    PRIMARY KEY lookup. The orchestrator's
    ``--retune-from-db --model-role dflash`` looks up the
    dflash-specific rows; the lookup is independent of
    the trunk rows (different ``model_role`` values get
    different (w_imatrix, w_gradient, w_layer) tuples).
    When ``model_role`` is None, no role filter is
    applied (the legacy pre-Phase-16 path). The
    cross-model fallback preserves the role dimension
    too: the ``("*", "dflash", "attn_q")`` row is the
    dflash-encoder cross-model aggregate, not the trunk
    cross-model aggregate.

    The orchestrator-side lookup is a 3-tier resolution
    (Phase 16):
      1. (model_hash, model_role, family) per-model, per-role, per-family
      2. ("*", model_role, family) cross-model, per-role, per-family
         (when cross_model_fallback=True)
      3. base weights
    The cross-model fallback only triggers for families the
    per-model lookup missed. A family with neither per-model
    nor cross-model data is not returned at all (the
    orchestrator uses the base weights for it).
    """
    if not Path(db_path).is_file():
        return pl.DataFrame(schema=L5_WEIGHTS_COLS)
    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "l5_weights" not in names:
            return pl.DataFrame(schema=L5_WEIGHTS_COLS)
        if not cross_model_fallback or not model_hash:
            where_clauses: list[str] = []
            if model_hash:
                where_clauses.append(
                    f"model_hash = '{sql_escape(model_hash)}'"
                )
            if model_role is not None:
                where_clauses.append(
                    f"model_role = '{sql_escape(model_role)}'"
                )
            where = (
                (" WHERE " + " AND ".join(where_clauses))
                if where_clauses else ""
            )
            return db.query("SELECT * FROM l5_weights" + where)
        # Cross-model fallback: union the per-model rows with
        # the cross-model rows for any family the per-model
        # lookup missed. Both the per-model SELECT and the
        # cross-model SELECT carry the model_role filter so
        # the fallback stays role-aware.
        per_where_clauses: list[str] = [
            f"model_hash = '{sql_escape(model_hash)}'"
        ]
        if model_role is not None:
            per_where_clauses.append(
                f"model_role = '{sql_escape(model_role)}'"
            )
        per_model = db.query(
            "SELECT * FROM l5_weights WHERE "
            + " AND ".join(per_where_clauses)
        )
        per_model_families = set(per_model["family"].to_list())
        cross_where_clauses: list[str] = ["model_hash = '*'"]
        if model_role is not None:
            cross_where_clauses.append(
                f"model_role = '{sql_escape(model_role)}'"
            )
        cross = db.query(
            "SELECT * FROM l5_weights WHERE "
            + " AND ".join(cross_where_clauses)
        )
        # The cross-model rows are an aggregate across
        # models; the orchestrator's --retune-from-db wants
        # them to look like per-family rows for the
        # missing-model case. We keep the model_hash = "*"
        # marker so the orchestrator can tell them apart
        # from genuine per-model rows; the
        # aggregate_weights helper handles the merge.
        if cross.height == 0:
            return per_model
        return pl.concat([per_model, cross], how="vertical_relaxed")


def aggregate_weights(
    df: pl.DataFrame,
    *,
    base_weights: tuple[float, float, float] = DEFAULT_BASE_WEIGHTS,
) -> tuple[float, float, float]:
    """Aggregate per-family weights into a single tuple for the
    orchestrator.

    The orchestrator has one (w_im, w_grad, w_layer) tuple, not
    per-family. We average across families with a per-family
    weight = n_samples (families with more data count more).
    Falls back to the base weights when the DataFrame is empty.

    Cross-model rows (``model_hash = "*"``) are weighted the
    same as per-model rows; the cross-model aggregation
    already folded the across-model n_samples into one row
    per family.

    Args:
      df: the l5_weights table (or a model-filtered subset).
      base_weights: the base weights used when df is empty.

    Returns:
      (w_imatrix, w_gradient, w_layer), projected to the
      simplex.
    """
    if df.height == 0:
        return base_weights
    n_total = int(df["n_samples"].sum())
    if n_total <= 0:
        return base_weights
    w_im = float((df["w_imatrix"] * df["n_samples"]).sum() / n_total)
    w_grad = float((df["w_gradient"] * df["n_samples"]).sum() / n_total)
    w_layer = float((df["w_layer"] * df["n_samples"]).sum() / n_total)
    return _project_simplex((w_im, w_grad, w_layer))


# Retune follow-ups: --retune-from-db cache layer. The
# orchestrator's 3-tier lookup chain (per-model+per-role
# -> cross-model+per-role -> per-model without role) is
# called per orchestrator iteration; on a hot loop
# (e.g. an online service that re-reads the DB on every
# generation) the redundant re-reads are wasteful. The
# cache memoises the result of the chain for a given
# (db_path, model_hash, model_role, cross_model_fallback)
# tuple. The second call returns the cached DataFrame
# without re-querying DuckDB.
#
# The cache is a per-process dict; the orchestrator's
# long-running mode (or a test that calls the lookup
# multiple times in the same process) benefits. The
# cache key includes the db_path so a path change
# produces a different entry (no manual invalidation
# required: a different path -> a different key).
#
# The cache is intentionally NOT a free function
# (lru_cache) because the result is a polars DataFrame,
# which is not hashable. The dict is process-local; no
# thread safety is enforced (the orchestrator's main
# loop is single-threaded, and a long-running service
# is expected to wrap the lookup in its own lock if
# multi-threading is needed). The cache is exposed via
# :py:func:`_l5_weights_lookup_cache` so tests can
# inspect / clear it.
_L5_WEIGHTS_LOOKUP_CACHE: dict[
    tuple[str, str | None, str | None, bool], pl.DataFrame
] = {}


def _l5_weights_lookup_cache() -> dict[
    tuple[str, str | None, str | None, bool], pl.DataFrame
]:
    """Return the process-local 3-tier lookup cache.

    Tests use this to inspect the cache (e.g. assert the
    second call populated the cache) and to clear it
    (e.g. when a DB is deleted and the cache should
    not return a stale DataFrame). Production code
    should not call this; the lookup helper manages
    the cache internally.
    """
    return _L5_WEIGHTS_LOOKUP_CACHE


def clear_l5_weights_lookup_cache() -> None:
    """Clear the process-local 3-tier lookup cache.

    Used by tests that mutate the DB between calls
    (the cache would otherwise return stale
    DataFrames). Production code may call this when
    a long-running service knows the DB has been
    replaced (e.g. on a schema migration).
    """
    _L5_WEIGHTS_LOOKUP_CACHE.clear()


def resolve_l5_weights_for_orchestrator(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    model_role: str | None = None,
    cross_model_fallback: bool = True,
) -> pl.DataFrame:
    """The orchestrator's --retune-from-db 3-tier lookup,
    with a process-local cache.

    Implements the 3-tier resolution that ``l5_orchestrator
    .main`` runs after parsing the CLI args. The tiers
    are:

      1. (model_hash, model_role, family) per-model,
         per-role, per-family (the production path).
      2. ("*", model_role, family) cross-model, per-role,
         per-family (when ``cross_model_fallback`` is
         True).
      3. (model_hash, *, family) per-model, no role
         (the legacy pre-Phase-16 path; the orchestrator
         uses this as the final fallback when the
         role-aware tiers miss).

    The first tier with a non-empty result wins; the
    remaining tiers are not consulted. The result is
    cached in :py:data:`_L5_WEIGHTS_LOOKUP_CACHE` keyed
    by ``(db_path, model_hash, model_role,
    cross_model_fallback)``; the second call with the
    same args returns the cached DataFrame without
    re-querying DuckDB.

    Args:
      db_path: the unified tessera.duckdb file.
      model_hash: the model to look up. ``None`` means
        the lookup returns only the cross-model row
        (legacy pre-Phase-16 path).
      model_role: when set, the lookup is per-role
        (Phase 16). When ``None``, the lookup is
        role-agnostic.
      cross_model_fallback: when True (default), the
        cross-model tier is consulted for families the
        per-model lookup missed.

    Returns:
      A polars DataFrame with the l5_weights schema
      (the union of the per-model and cross-model
      rows). Empty when no row is found at any tier.
    """
    db_str = str(db_path)
    cache_key = (
        db_str, model_hash, model_role, cross_model_fallback,
    )
    cached = _L5_WEIGHTS_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Phase 16: 3-tier lookup chain. The result is
    # an empty DataFrame with the l5_weights schema
    # so the empty-result placeholder has the right
    # columns for downstream code.
    weights_df = pl.DataFrame(schema=L5_WEIGHTS_COLS)
    if model_role is not None and model_hash is not None:
        # Tier 1: per-model + per-role.
        weights_df = read_l5_weights(
            db_path,
            model_hash=model_hash,
            model_role=model_role,
            cross_model_fallback=False,
        )
        # Tier 2: cross-model + per-role (only when
        # cross_model_fallback is on).
        if weights_df.height == 0 and cross_model_fallback:
            weights_df = read_l5_weights(
                db_path,
                model_hash="*",
                model_role=model_role,
                cross_model_fallback=False,
            )
    if weights_df.height == 0 and model_hash is not None:
        # Tier 3: per-model, no role (the legacy
        # pre-Phase-16 fallback; the orchestrator uses
        # this when the role-aware tiers miss and the
        # caller asked for a specific role).
        weights_df = read_l5_weights(
            db_path,
            model_hash=model_hash,
            model_role=None,
            cross_model_fallback=cross_model_fallback,
        )
    # Cache the result (even an empty DataFrame, so the
    # second call short-circuits the lookup chain).
    _L5_WEIGHTS_LOOKUP_CACHE[cache_key] = weights_df
    return weights_df


def resolve_per_family_top_fraction_for_orchestrator(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    model_role: str | None = None,
    cross_model_fallback: bool = True,
) -> dict[str, float]:
    """Cached variant of :py:func:`read_per_family_top_fraction`.

    The orchestrator's per-family top_fraction consumer
    is also re-queried per iteration. The cache key is
    ``(db_path, model_hash, model_role,
    cross_model_fallback)`` (the same shape as
    :py:func:`resolve_l5_weights_for_orchestrator`'s
    cache key for the (db_path, model_hash, model_role,
    cross_model_fallback) dimension; the value is
    stored in a separate dict because the type is
    different from the l5_weights DataFrame).
    """
    db_str = str(db_path)
    cache_key = (
        db_str, model_hash, model_role, cross_model_fallback,
    )
    cached = _L5_WEIGHTS_TOP_FRACTION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    out = read_per_family_top_fraction(
        db_path,
        model_hash=model_hash,
        model_role=model_role,
        cross_model_fallback=cross_model_fallback,
    )
    _L5_WEIGHTS_TOP_FRACTION_CACHE[cache_key] = out
    return out


# Retune follow-ups: separate dict for the per-family
# top_fraction cache (the value type is dict[str, float],
# not a polars DataFrame, so the two caches are not
# shareable). The cache shape is the same as the
# l5_weights cache: keyed by
# (db_path, model_hash, model_role, cross_model_fallback).
_L5_WEIGHTS_TOP_FRACTION_CACHE: dict[
    tuple[str, str | None, str | None, bool], dict[str, float]
] = {}


def _l5_weights_top_fraction_cache() -> dict[
    tuple[str, str | None, str | None, bool], dict[str, float]
]:
    """Return the per-family top_fraction cache (tests)."""
    return _L5_WEIGHTS_TOP_FRACTION_CACHE


def read_per_family_top_fraction(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    model_role: str | None = None,
    cross_model_fallback: bool = True,
) -> dict[str, float]:
    """Read the per-family top_fraction recommendations for
    the orchestrator's ``RequantPlanner``.

    The return value is a ``{family: top_fraction}`` dict. A
    family missing from the dict means "no recommendation"
    (use the --top-fraction flag value). Cross-model rows
    (model_hash = "*") fill in for families the per-model
    lookup missed; per-model rows take priority.

    Phase 16: the ``model_role`` filter is applied on both
    the per-model SELECT and the cross-model SELECT. The
    orchestrator's ``--retune-from-db --model-role dflash``
    looks up dflash-specific top_fraction recommendations;
    the dflash encoder's ``attn_q`` may have a very
    different top_fraction than the trunk's ``attn_q``
    (the dflash encoder's residual surface is different
    from the trunk's). The role filter is preserved on
    the cross-model fallback path so the dflash
    cross-model aggregate fills in for dflash
    per-model families the per-model lookup missed
    (not the trunk cross-model aggregate, which would
    be a category error).

    Args:
      db_path: path to the unified tessera.duckdb file.
      model_hash: if non-None, prefer the per-model rows
        for this model and fall back to the cross-model
        rows for families the per-model lookup missed.
        Default None = read the cross-model rows only.
      model_role: if non-None (Phase 16), restrict both
        the per-model and cross-model lookups to this
        role. Default None = no role filter (the legacy
        pre-Phase-16 path).
      cross_model_fallback: when True (default), use the
        cross-model rows for any family the per-model
        lookup missed. The orchestrator enables this on
        warm-start (the consumer side of the cross-model
        retune).
    """
    out: dict[str, float] = {}
    if not Path(db_path).is_file():
        return out
    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "l5_weights" not in names:
            return out
        if model_hash is not None:
            per_where = [
                f"model_hash = '{sql_escape(model_hash)}'",
                "top_fraction IS NOT NULL",
            ]
            if model_role is not None:
                per_where.append(
                    f"model_role = '{sql_escape(model_role)}'"
                )
            per_model = db.query(
                "SELECT family, top_fraction FROM l5_weights WHERE "
                + " AND ".join(per_where)
            )
            for row in per_model.iter_rows(named=True):
                if row.get("top_fraction") is not None:
                    out[str(row["family"])] = float(row["top_fraction"])
        if cross_model_fallback or model_hash is None:
            cross_where = [
                "model_hash = '*'",
                "top_fraction IS NOT NULL",
            ]
            if model_role is not None:
                cross_where.append(
                    f"model_role = '{sql_escape(model_role)}'"
                )
            cross = db.query(
                "SELECT family, top_fraction FROM l5_weights WHERE "
                + " AND ".join(cross_where)
            )
            for row in cross.iter_rows(named=True):
                fam = str(row["family"])
                if fam not in out and row.get("top_fraction") is not None:
                    out[fam] = float(row["top_fraction"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "L5 retune: per-(model, family) recompute of the "
            "orchestrator's sensitivity scoring weights from "
            "l5_outcome. The orchestrator's next generation reads "
            "the result via --retune-from-db."
        ),
    )
    p.add_argument(
        "--db", required=True, type=Path,
        help="Path to the unified tessera.duckdb file",
    )
    p.add_argument(
        "--model-hash", default=None,
        help="Restrict to this model_hash (default: all models)",
    )
    p.add_argument(
        "--model-role", default=None,
        choices=[
            "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
        ],
        help=(
            "Restrict to this model_role (default: all roles). "
            "Phase 16: the retune's per-(model, model_role, family) "
            "partition lets the same family in different "
            "architectural roles (e.g. the trunk's attn_q vs the "
            "dflash encoder's attn_q) get independent retune "
            "verdicts. The default 'all roles' path is the legacy "
            "pre-Phase-16 behaviour: one verdict per (model, "
            "family), the model_role column on the read rows is "
            "ignored."
        ),
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Shift aggressiveness; the shift is alpha * slope * "
             "(1 - hit_rate) (default 0.5)",
    )
    p.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
        help="Minimum per-(model, family) sample count for the OLS "
             "to be acted on (default 3)",
    )
    p.add_argument(
        "--base-top-fraction", type=float,
        default=DEFAULT_BASE_TOP_FRACTION,
        help="Base value the per-family top_fraction "
             "recommendation is anchored to (default 0.10; the "
             "orchestrator's --top-fraction default)",
    )
    p.add_argument(
        "--budget-fraction", type=float,
        default=DEFAULT_BASE_BUDGET_FRACTION,
        help="Base fraction for the per-family requant_budget_bits "
             "recommendation: budget = family_storage_bits * "
             "(1 - hit_rate) * fraction, where family_storage_bits "
             "is the family's reference footprint from tensor_stats "
             "(default 1.0; 0 disables budgets -> NULL). The "
             "deployment knob for a memory-bound target.",
    )
    p.add_argument(
        "--retune-cross-model", action="store_true",
        help="After the per-model retune, write a per-family "
             "aggregate row with model_hash='*' (n_samples-weighted "
             "mean across all models). The orchestrator's "
             "--retune-from-db falls back to this row when a "
             "model has no specific row.",
    )
    p.add_argument(
        "--no-ema", action="store_true",
        help="Skip the l5_plan_ema join; fit the OLS on the "
             "per-iteration sensitivity_score instead. The "
             "default is the EMA-aware path (stable across "
             "iterations).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute the retune and print the verdict table "
             "without writing to l5_weights",
    )
    p.add_argument(
        "--print-table", action="store_true",
        help="After writing, print the per-(model, family) weights "
             "to stdout (CSV with header)",
    )
    return p


def _format_table(verdicts: list[FamilyWeights]) -> str:
    rows = ["model_hash,model_role,family,w_imatrix,w_gradient,w_layer,"
            "b_im,b_grad,b_layer,slope,hit_rate,n_samples,"
            "top_fraction,coupling_score,requant_budget_bits,"
            "retune_source,was_acted_on"]
    for v in verdicts:
        rows.append(
            f"{v.model_hash},{v.model_role},{v.family},"
            f"{v.weights[0]:.4f},{v.weights[1]:.4f},{v.weights[2]:.4f},"
            f"{v.slopes[0]:+.6f},{v.slopes[1]:+.6f},{v.slopes[2]:+.6f},"
            f"{v.slopes[1]:+.6f},{v.hit_rate:.3f},"
            f"{v.n_samples},"
            f"{v.top_fraction if v.top_fraction is not None else ''},"
            f"{v.coupling_score if v.coupling_score is not None else ''},"
            f"{v.requant_budget_bits if v.requant_budget_bits is not None else ''},"
            f"{v.retune_algorithm},"
            f"{int(v.was_acted_on)}"
        )
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    verdicts = compute_l5_weights(
        args.db,
        model_hash=args.model_hash,
        model_role=args.model_role,
        base_weights=DEFAULT_BASE_WEIGHTS,
        base_top_fraction=args.base_top_fraction,
        base_budget_fraction=args.budget_fraction,
        alpha=args.alpha,
        min_samples=args.min_samples,
        use_ema=not args.no_ema,
        write_back=not args.dry_run,
    )
    n_acted = sum(1 for v in verdicts if v.was_acted_on)
    n_total = len(verdicts)
    n_skipped = n_total - n_acted

    cross_verdicts: list[FamilyWeights] = []
    if args.retune_cross_model and not args.dry_run:
        cross_verdicts = write_cross_model_aggregate(
            args.db,
            base_weights=DEFAULT_BASE_WEIGHTS,
            base_top_fraction=args.base_top_fraction,
            min_samples=args.min_samples,
        )
    n_cross = len(cross_verdicts)

    if args.dry_run or args.print_table:
        all_verdicts = verdicts + cross_verdicts
        print(_format_table(all_verdicts))
    if args.dry_run:
        return 0
    print(
        f"l5_weights: wrote {n_acted} row(s), "
        f"skipped {n_skipped} (insufficient samples), "
        f"of {n_total} (model, family) group(s)"
    )
    if args.retune_cross_model:
        print(
            f"l5_weights: wrote {n_cross} cross-model row(s) "
            f"(model_hash='*')"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
