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
    """The per-(model, family) retune verdict: the recommended
    (w_imatrix, w_gradient, w_layer) and the OLS fit diagnostics.

    Fields:
      model_hash:        the model the weights apply to (or
                         ``"*"`` for the cross-model aggregate).
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
      retune_algorithm:  one of ``RETUNE_SOURCE_TAG_3COEF`` /
                         ``RETUNE_SOURCE_TAG_2COEF`` /
                         ``RETUNE_SOURCE_TAG_CROSSMODEL``.
      was_acted_on:      True if the retune wrote weights; False
                         if the group was skipped.
    """

    model_hash: str
    family: str
    weights: tuple[float, float, float]
    bias: float
    slopes: tuple[float, float, float]
    n_samples: int
    in_sample_loss: float
    hit_rate: float
    top_fraction: float | None
    retune_algorithm: str
    was_acted_on: bool

    def to_dict(self) -> dict:
        return {
            "model_hash":       self.model_hash,
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
) -> FamilyWeights:
    """Retune one (model, family) group.

    Phase 15: when the per-tensor components are available, fit
    a 3-coefficient OLS with sample weights derived from
    in_sample_loss and n_samples. When the components are
    missing, fall back to the 2-coefficient OLS on the
    combined sensitivity_score.

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
    n = len(sensitivity)
    if n < 1:
        return FamilyWeights(
            model_hash=model_hash, family=family,
            weights=base_weights,
            bias=0.0, slopes=(0.0, 0.0, 0.0),
            n_samples=0, in_sample_loss=0.0, hit_rate=0.0,
            top_fraction=None,
            retune_algorithm=RETUNE_SOURCE_TAG_2COEF,
            was_acted_on=False,
        )
    n_accepted = sum(1 for v in plan_accepted if v)
    hit_rate = n_accepted / n if n else 0.0
    if n < min_samples:
        return FamilyWeights(
            model_hash=model_hash, family=family,
            weights=base_weights,
            bias=0.0, slopes=(0.0, 0.0, 0.0),
            n_samples=n, in_sample_loss=0.0, hit_rate=hit_rate,
            top_fraction=None,
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
            model_hash=model_hash, family=family,
            weights=projected,
            bias=a,
            slopes=(b_im, b_grad, b_layer),
            n_samples=n,
            in_sample_loss=in_sample_loss,
            hit_rate=hit_rate,
            top_fraction=_compute_top_fraction(
                base_top_fraction, dominant_slope, hit_rate,
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
        model_hash=model_hash, family=family,
        weights=projected,
        bias=a,
        slopes=(0.0, b, 0.0),
        n_samples=n,
        in_sample_loss=in_sample_loss,
        hit_rate=hit_rate,
        top_fraction=_compute_top_fraction(
            base_top_fraction, abs(b), hit_rate,
        ),
        retune_algorithm=RETUNE_SOURCE_TAG_2COEF,
        was_acted_on=True,
    )


def compute_l5_weights(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    base_weights: tuple[float, float, float] = DEFAULT_BASE_WEIGHTS,
    base_top_fraction: float = DEFAULT_BASE_TOP_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    use_ema: bool = True,
    write_back: bool = True,
) -> list[FamilyWeights]:
    """Read ``l5_outcome``, run the per-(model, family) retune,
    and (optionally) write the result to ``l5_weights``.

    Args:
      db_path: path to the unified tessera.duckdb file.
      model_hash: if non-None, restrict to this model. Default
        None = all models in the DB.
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
      A list of FamilyWeights, one per (model, family) group
      seen in the join. The list is sorted by (model_hash,
      family) for stable output.
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
        where = ""
        if model_hash:
            where = f" WHERE model_hash = '{sql_escape(model_hash)}'"
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
        col_list = (
            "model_hash, family, sensitivity_score, "
            "delta_mse, plan_accepted"
            + (", imatrix_magnitude" if has_im else "")
            + (", gradient_proxy" if has_grad else "")
            + (", layer_position_prior" if has_layer else "")
            + (", in_sample_loss" if has_in_sample_loss else "")
        )
        df = db.query(
            f"SELECT {col_list} FROM l5_outcome" + where
        )
        # Backfill any missing component / in_sample_loss
        # columns as NULL so the rest of the retune can
        # use a uniform schema.
        for col in ("imatrix_magnitude", "gradient_proxy",
                    "layer_position_prior", "in_sample_loss"):
            if col not in df.columns:
                df = df.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(col)
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

    if df.height == 0:
        return []

    # Per-(model, family) retune. partition_by is stable in
    # polars 0.20+, so the output is in (model_hash, family)
    # order.
    groups = df.partition_by(["model_hash", "family"], maintain_order=True)
    verdicts: list[FamilyWeights] = []
    for g in groups:
        mh = str(g["model_hash"][0])
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
        verdicts.append(_retune_family(
            model_hash=mh, family=fam,
            sensitivity=sens, delta_mse=deltas,
            plan_accepted=accepted,
            components=components,
            in_sample_losses=in_sample_losses,
            n_samples_per_row=n_samples_per_row,
            base_weights=base_weights,
            base_top_fraction=base_top_fraction,
            alpha=alpha, min_samples=min_samples,
        ))

    if write_back and verdicts:
        rows_to_write = [v for v in verdicts if v.was_acted_on]
        if rows_to_write:
            with TesseraDB.open(db_path) as db:
                con = db._conn
                con.execute("BEGIN")
                try:
                    if model_hash is not None:
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
    """Read l5_weights grouped by family (across all models),
    write one row per family with ``model_hash = "*"``.

    The aggregate is the n_samples-weighted mean of the
    per-model rows, with the top_fraction and hit_rate
    aggregated similarly. The cross-model row is a
    generalization of the per-model row, not a replacement;
    the orchestrator's --retune-from-db falls back to the
    cross-model row when the per-model row is missing
    (warm-start new models from the cross-model mean).

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
        for a family to be aggregated. Below this the
        cross-model row is not written (the per-family
        n is too thin to warm-start from).

    Returns:
      A list of FamilyWeights, one per family the
      aggregate produced. The list is sorted by family.
    """
    if not Path(db_path).is_file():
        return []
    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "l5_weights" not in names:
            return []
        # Read every row except the cross-model itself (we
        # don't want to fold the previous aggregate into the
        # new one).
        df = db.query(
            "SELECT model_hash, family, w_imatrix, w_gradient, "
            "w_layer, bias, n_samples, in_sample_loss, hit_rate, "
            "top_fraction FROM l5_weights "
            f"WHERE model_hash != '*'"
        )
    if df.height == 0:
        return []

    # Group by family, aggregate. Polars' group_by with
    # multiple aggregations; the n_samples-weighted mean is
    # the standard sum(w*x) / sum(w) expression. For the
    # top_fraction we aggregate only the rows that have a
    # non-NULL top_fraction (the per-family recommendation
    # is optional); when none of the rows have a top_fraction
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
    aggregated = df.group_by("family").agg(agg_exprs)
    # top_fraction needs a separate pass because polars'
    # ``sum()`` over (top_fraction * n_samples) drops rows
    # where top_fraction is NULL. Compute it on the same
    # group_by with a different filter.
    top_frac_df = (
        df.filter(pl.col("top_fraction").is_not_null())
          .group_by("family")
          .agg(
              ((pl.col("top_fraction") * pl.col("n_samples")).sum()
               / pl.col("n_samples").sum()).alias("top_fraction")
          )
    )
    aggregated = aggregated.join(top_frac_df, on="family", how="left")
    # Filter by min_samples (total n across models).
    aggregated = aggregated.filter(pl.col("total_n") >= min_samples)

    verdicts: list[FamilyWeights] = []
    now_str = None  # filled in by TesseraDB
    rows_to_write: list[dict] = []
    for row in aggregated.iter_rows(named=True):
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
        verdicts.append(FamilyWeights(
            model_hash="*", family=fam,
            weights=projected,
            bias=float(row["bias"]),
            slopes=(0.0, 0.0, 0.0),  # aggregated, not stored
            n_samples=n_total,
            in_sample_loss=float(row["in_sample_loss"]),
            hit_rate=agg_hit_rate,
            top_fraction=top_frac_out,
            retune_algorithm=RETUNE_SOURCE_TAG_CROSSMODEL,
            was_acted_on=True,
        ))
        rows_to_write.append({
            "model_hash":      "*",
            "family":          fam,
            "w_imatrix":       projected[0],
            "w_gradient":      projected[1],
            "w_layer":         projected[2],
            "bias":            float(row["bias"]),
            "n_samples":       n_total,
            "in_sample_loss":  float(row["in_sample_loss"]),
            "hit_rate":        agg_hit_rate,
            "top_fraction":    top_frac_out,
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
    cross_model_fallback: bool = False,
) -> pl.DataFrame:
    """Read the l5_weights table for the consumer (the
    orchestrator's ``--retune-from-db`` path).

    Returns an empty DataFrame with the l5_weights schema when
    the table is missing or empty. When ``model_hash`` is given,
    the result is filtered to that model. When
    ``cross_model_fallback`` is True, rows with
    ``model_hash = "*"`` are appended for any family the
    per-model lookup missed; this is the orchestrator's
    "warm-start new model from cross-model mean" path.

    The orchestrator-side lookup is a 3-tier resolution:
      1. (model_hash, family) per-model, per-family
      2. ("*", family) cross-model, per-family
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
            where = ""
            if model_hash:
                where = f" WHERE model_hash = '{sql_escape(model_hash)}'"
            return db.query("SELECT * FROM l5_weights" + where)
        # Cross-model fallback: union the per-model rows with
        # the cross-model rows for any family the per-model
        # lookup missed.
        per_model = db.query(
            "SELECT * FROM l5_weights "
            f"WHERE model_hash = '{sql_escape(model_hash)}'"
        )
        per_model_families = set(per_model["family"].to_list())
        cross = db.query(
            "SELECT * FROM l5_weights WHERE model_hash = '*'"
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


def read_per_family_top_fraction(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    cross_model_fallback: bool = True,
) -> dict[str, float]:
    """Read the per-family top_fraction recommendations for
    the orchestrator's ``RequantPlanner``.

    The return value is a ``{family: top_fraction}`` dict. A
    family missing from the dict means "no recommendation"
    (use the --top-fraction flag value). Cross-model rows
    (model_hash = "*") fill in for families the per-model
    lookup missed; per-model rows take priority.

    Args:
      db_path: path to the unified tessera.duckdb file.
      model_hash: if non-None, prefer the per-model rows
        for this model and fall back to the cross-model
        rows for families the per-model lookup missed.
        Default None = read the cross-model rows only.
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
            per_model = db.query(
                "SELECT family, top_fraction FROM l5_weights "
                f"WHERE model_hash = '{sql_escape(model_hash)}' "
                "AND top_fraction IS NOT NULL"
            )
            for row in per_model.iter_rows(named=True):
                if row.get("top_fraction") is not None:
                    out[str(row["family"])] = float(row["top_fraction"])
        if cross_model_fallback or model_hash is None:
            cross = db.query(
                "SELECT family, top_fraction FROM l5_weights "
                "WHERE model_hash = '*' AND top_fraction IS NOT NULL"
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
    rows = ["model_hash,family,w_imatrix,w_gradient,w_layer,"
            "b_im,b_grad,b_layer,slope,hit_rate,n_samples,"
            "top_fraction,retune_source,was_acted_on"]
    for v in verdicts:
        rows.append(
            f"{v.model_hash},{v.family},"
            f"{v.weights[0]:.4f},{v.weights[1]:.4f},{v.weights[2]:.4f},"
            f"{v.slopes[0]:+.6f},{v.slopes[1]:+.6f},{v.slopes[2]:+.6f},"
            f"{v.slopes[1]:+.6f},{v.hit_rate:.3f},"
            f"{v.n_samples},"
            f"{v.top_fraction if v.top_fraction is not None else ''},"
            f"{v.retune_algorithm},"
            f"{int(v.was_acted_on)}"
        )
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    verdicts = compute_l5_weights(
        args.db,
        model_hash=args.model_hash,
        base_weights=DEFAULT_BASE_WEIGHTS,
        base_top_fraction=args.base_top_fraction,
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
