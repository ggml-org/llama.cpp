"""L5 retune: per-(model, family) recompute of the orchestrator's
sensitivity scoring weights from the feedback loop's residual.

The ``l5_outcome`` table is the "did this requant plan reduce
error?" verdict. The next consumer is this script: it reads
``l5_outcome.delta_mse`` and ``l5_outcome.sensitivity_score``,
fits a per-(model, family) closed-form OLS, and projects the
result onto the (w_imatrix, w_gradient, w_layer) simplex. The
projection lands in ``l5_weights`` with PRIMARY KEY
(model_hash, family).

The orchestrator's next generation reads ``l5_weights`` back
(via ``--retune-from-db``) and uses the per-family recommendation
as the starting point for ``SensitivityScorer``, closing the
loop.

The OLS model is intentionally simple: the l5_outcome rows give
us ``(sensitivity_score, delta_mse)`` per (model, family). A
2-coefficient closed-form fit (``delta_mse = a + b *
sensitivity_score``) is enough to extract a calibration slope
``b``. The retune then decides which of the three component
weights to shift:

  b > 0  (sensitivity_score predicts *positive* delta_mse)
       -> the orchestrator is up-weighting the wrong component;
          shift weight AWAY from the imatrix signal (the
          imatrix magnitudes are the cheapest signal and the
          most likely to be miscalibrated) and TOWARD the
          gradient signal (the L4 differential is the
          second-derivative approximation and the most
          accurate).
  b < 0  (sensitivity_score predicts *negative* delta_mse)
       -> the orchestrator is correctly identifying the
          sensitive tensors, and those tensors are being
          protected (negative delta_mse = error reduced). Boost
          the imatrix signal that drove the ranking.
  |b| ≈ 0 (no signal)
       -> keep the base weights (the prior is doing the work).

The shift magnitude is ``alpha * |b| * (1 - hit_rate)``; the
(1 - hit_rate) gate stops a family with high hit rate from
being perturbed, and the |b| term scales the shift by how
much the slope says the calibration is off. After the shift
the weights are projected onto the simplex
(non-negative, sum to 1.0).

The l5_outcome ``residual`` column is the same OLS residual
that ``l5_outcome.py`` computes; this script re-uses the same
arithmetic so the two tools agree on what "calibrated" means.

Writes:
  * ``l5_weights``: PRIMARY KEY (model_hash, family), one row
    per group with a non-empty fit.
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
RETUNE_SOURCE_TAG: str = "ols_slope_v1"


@dataclass
class FamilyWeights:
    """The per-(model, family) retune verdict: the recommended
    (w_imatrix, w_gradient, w_layer) and the OLS fit diagnostics.

    Fields:
      model_hash:        the model the weights apply to
      family:            the tensor family the weights apply to
      weights:           (w_imatrix, w_gradient, w_layer) on the simplex
      bias:              the OLS intercept
      slope:             the OLS slope (delta_mse per unit sensitivity)
      n_samples:         the count of l5_outcome rows that fed the fit
      in_sample_loss:    post-fit mean abs residual
      hit_rate:          fraction of plans in the group with
                         delta_mse < accept_threshold
      was_acted_on:      True if the retune wrote weights; False if the
                         group was skipped (too few samples, or
                         |slope| * (1 - hit_rate) below threshold)
    """

    model_hash: str
    family: str
    weights: tuple[float, float, float]
    bias: float
    slope: float
    n_samples: int
    in_sample_loss: float
    hit_rate: float
    was_acted_on: bool

    def to_dict(self) -> dict:
        return {
            "model_hash":       self.model_hash,
            "family":           self.family,
            "w_imatrix":        float(self.weights[0]),
            "w_gradient":       float(self.weights[1]),
            "w_layer":          float(self.weights[2]),
            "bias":             float(self.bias),
            "slope":            float(self.slope),
            "n_samples":        int(self.n_samples),
            "in_sample_loss":   float(self.in_sample_loss),
            "hit_rate":         float(self.hit_rate),
            "retune_source":    (RETUNE_SOURCE_TAG if self.was_acted_on
                                 else ""),
        }


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _ols_slope_intercept(
    x: list[float], y: list[float],
) -> tuple[float, float, float]:
    """Closed-form OLS: y = a + b*x.

    Returns (a, b, mean_abs_residual). When the input has fewer
    than 2 distinct x values, the slope is 0 and the intercept
    is the mean of y. Mean abs residual is the L1 loss of the
    fit; zero is perfect.
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


def _retune_family(
    *,
    model_hash: str,
    family: str,
    sensitivity: list[float],
    delta_mse: list[float],
    plan_accepted: list[bool],
    base_weights: tuple[float, float, float],
    alpha: float,
    min_samples: int,
) -> FamilyWeights:
    """Retune one (model, family) group.

    The shift rule:
        w_im_new  = w_im_base - alpha * slope * (1 - hit_rate)
        w_grad_new = w_grad_base + alpha * slope * (1 - hit_rate)
        w_layer_new = w_layer_base
        project onto simplex

    Then project to non-negative, sum=1. The reasoning:
      b > 0: sensitivity_score is positively correlated with
        delta_mse -> the orchestrator's ranking is upside-down
        for this family; pulling mass from im to gradient
        corrects it.
      b < 0: sensitivity_score is negatively correlated with
        delta_mse -> the ranking is already correct; we boost
        the im that was driving it.
      b ≈ 0: keep the base.
    The (1 - hit_rate) gate stops high-performing families from
    being perturbed; the orchestrator is working for them, so
    don't fix what isn't broken.
    """
    n = len(sensitivity)
    if n < 1:
        return FamilyWeights(
            model_hash=model_hash, family=family,
            weights=base_weights,
            bias=0.0, slope=0.0, n_samples=0,
            in_sample_loss=0.0, hit_rate=0.0,
            was_acted_on=False,
        )
    a, b, in_sample_loss = _ols_slope_intercept(
        sensitivity, delta_mse,
    )
    n_accepted = sum(1 for v in plan_accepted if v)
    hit_rate = n_accepted / n if n else 0.0
    if n < min_samples:
        # Too few samples; don't shift. Return base weights.
        return FamilyWeights(
            model_hash=model_hash, family=family,
            weights=base_weights,
            bias=a, slope=b, n_samples=n,
            in_sample_loss=in_sample_loss, hit_rate=hit_rate,
            was_acted_on=False,
        )
    gate = 1.0 - hit_rate
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
        bias=a, slope=b, n_samples=n,
        in_sample_loss=in_sample_loss, hit_rate=hit_rate,
        was_acted_on=True,
    )


def compute_l5_weights(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    base_weights: tuple[float, float, float] = DEFAULT_BASE_WEIGHTS,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
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
        alpha: shift aggressiveness; see the module docstring.
        min_samples: minimum per-(model, family) sample count
            for the OLS to be acted on.
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
        df = db.query(
            "SELECT model_hash, family, sensitivity_score, "
            "delta_mse, plan_accepted FROM l5_outcome" + where
        )

    if df.height == 0:
        return []

    # Per-(model, family) retune. group_by is stable in polars 0.20+
    # so the output is in (model_hash, family) order.
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
        verdicts.append(_retune_family(
            model_hash=mh, family=fam,
            sensitivity=sens, delta_mse=deltas,
            plan_accepted=accepted,
            base_weights=base_weights,
            alpha=alpha, min_samples=min_samples,
        ))

    if write_back and verdicts:
        # Only write the rows that were acted on. Skipped rows
        # are not a "retune recommendation" - they're a
        # "no recommendation" - so the orchestrator's
        # --retune-from-db falls back to the base weights for
        # those (model, family) groups.
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

    return verdicts


def read_l5_weights(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
) -> pl.DataFrame:
    """Read the l5_weights table for the consumer (the
    orchestrator's ``--retune-from-db`` path).

    Returns an empty DataFrame with the l5_weights schema when
    the table is missing or empty. When ``model_hash`` is given,
    the result is filtered to that model.
    """
    if not Path(db_path).is_file():
        return pl.DataFrame(schema=L5_WEIGHTS_COLS)
    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "l5_weights" not in names:
            return pl.DataFrame(schema=L5_WEIGHTS_COLS)
        where = ""
        if model_hash:
            where = f" WHERE model_hash = '{sql_escape(model_hash)}'"
        return db.query("SELECT * FROM l5_weights" + where)


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
            "slope,hit_rate,n_samples,was_acted_on"]
    for v in verdicts:
        rows.append(
            f"{v.model_hash},{v.family},"
            f"{v.weights[0]:.4f},{v.weights[1]:.4f},{v.weights[2]:.4f},"
            f"{v.slope:+.6f},{v.hit_rate:.3f},"
            f"{v.n_samples},{int(v.was_acted_on)}"
        )
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    verdicts = compute_l5_weights(
        args.db,
        model_hash=args.model_hash,
        alpha=args.alpha,
        min_samples=args.min_samples,
        write_back=not args.dry_run,
    )
    n_acted = sum(1 for v in verdicts if v.was_acted_on)
    n_total = len(verdicts)
    n_skipped = n_total - n_acted
    if args.dry_run or args.print_table:
        print(_format_table(verdicts))
    if args.dry_run:
        return 0
    # Default: one-line summary.
    print(
        f"l5_weights: wrote {n_acted} row(s), "
        f"skipped {n_skipped} (insufficient samples), "
        f"of {n_total} (model, family) group(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
