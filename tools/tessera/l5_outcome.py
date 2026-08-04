"""L5 outcome: the \"did this requant plan reduce error?\" verdict.

Reads the four-way join

    l5_plan_summary  JOIN  l4_plan_outcome  ON (model_hash, name, iteration, plan_id)

computes per-tensor

    delta_mse   = mse_after  - mse_before
    delta_frob  = frob_after - frob_before
    plan_accepted = (delta_mse < accept_threshold)

and the per-(model, family) residual

    residual = delta_mse - (a + b * sensitivity_score)

(linear fit of delta_mse on sensitivity_score; the residual
measures how well the orchestrator's sensitivity scoring
predicts the actual error delta. The next generation of the
orchestrator can use this residual to retune the scoring
weights, closing the loop.)

Writes the result to the ``l5_outcome`` table in the same
``tessera.duckdb`` file.

Companion to:
  * docs/tessera-polars-integration-scout.md §5.4
  * docs/tessera-unified-db.md (the unified DB design)
  * tools/quantize/tessera/tessera-quantize-db.cpp
    (the l4_plan_outcome + l5_outcome CREATE TABLE statements)

The C++ dispatch's adaptive_requantize loop writes
``l4_plan_outcome`` rows as it runs (the audit trail). The
Python orchestrator writes ``l5_plan_summary`` rows when it
emits a requant plan. This script joins the two and writes
``l5_outcome`` — the consumer-side verdict. Run after both
the C++ dispatch and the Python orchestrator have produced
their rows.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from tessera_db import (
    L5_PLAN_COLS,
    L4_PLAN_OUTCOME_COLS,
    L5_OUTCOME_COLS,
    TesseraDB,
)


# Default threshold below which a plan is considered to have
# "reduced error". mse and frob are the relative Frobenius
# norms already (rel_frob = ||w - w_hat||^2 / ||w||^2), so a
# delta < 0 means the requant reduced the error. The threshold
# is a small positive number to absorb noise; the exact value
# is a knob the user can tighten or loosen.
DEFAULT_ACCEPT_THRESHOLD: float = 0.0


@dataclass
class OutcomeSummary:
    """Per-run summary of the l5_outcome join. The head-line
    numbers for \"did the requant loop actually help?\":

      n_plans            — total plans evaluated (joined rows)
      n_accepted         — plans with delta_mse < accept_threshold
      hit_rate           — n_accepted / n_plans (or 0 if no plans)
      mean_delta_mse     — average error delta across all plans
      mean_residual      — average absolute residual of the
                           (model, family) linear fit of
                           delta_mse on sensitivity_score
    """

    n_plans: int = 0
    n_accepted: int = 0
    hit_rate: float = 0.0
    mean_delta_mse: float = 0.0
    mean_residual: float = 0.0
    per_family_hit_rate: dict[str, float] | None = None

    def __str__(self) -> str:
        out = [
            f"plans:        {self.n_plans}",
            f"accepted:     {self.n_accepted}",
            f"hit_rate:     {self.hit_rate:.3f}",
            f"mean_delta:   {self.mean_delta_mse:+.4f}",
            f"mean_residual:{self.mean_residual:+.4f}",
        ]
        if self.per_family_hit_rate:
            out.append("per-family:")
            for fam, hr in sorted(self.per_family_hit_rate.items()):
                out.append(f"  {fam:>20s}  {hr:.3f}")
        return "\n".join(out)


# Phase 15: the per-tensor component columns were added to
# l5_plan_summary on the Python side via ALTER TABLE IF NOT EXISTS.
# DBs created before Phase 15 (or by a C++ binary that has not
# picked up the column addition yet) lack the columns. ``SELECT
# col, col, col FROM l5_plan_summary`` fails on those DBs. The
# helper below issues the full SELECT and falls back to a
# column-free projection on the failure path, so the rest of the
# pipeline can keep operating. The fallback projection is the
# subset of L5_PLAN_COLS that existed before Phase 15; the
# ``with_columns`` in compute_l5_outcome backfills the missing
# component columns as NULL.

# Pre-Phase-15 l5_plan_summary schema (no per-tensor component
# columns). Used by the read fallback when the column projection
# fails against a DB that was created before this commit. ``updated_at``
# is preserved here even though it is not strictly required for
# the verdict: the join's drop/rename logic in
# :py:func:`compute_l5_outcome` assumes both sides carry
# ``updated_at`` (so the join's ``_outcome`` suffix is in effect
# and the post-drop rename actually finds a ``updated_at_outcome``
# to promote). Without ``updated_at`` on the plan side the
# outcome's ``updated_at`` is left as ``updated_at`` (no conflict
# to suffix) and the post-drop step removes it instead of the
# plan's, breaking the final select.
L5_PLAN_PRE_PHASE15_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer", "iteration", "plan_id",
    "sensitivity_score", "recommended_qtype", "recommended_alpha",
    "recommended_clip", "updated_at",
)


def _read_l5_plan_safe(
    db: "TesseraDB",
    model_hash: str | None,
    full_cols: tuple[str, ...],
    model_role: str | None = None,
) -> pl.DataFrame:
    """Read l5_plan_summary with the Phase 15 column set; fall
    back to the pre-Phase-15 column set when the DB has not been
    migrated.

    The ``full_cols`` argument is the new column set (with the
    per-tensor component columns); on a successful read the
    returned DataFrame has all those columns. On a failure (the
    SELECT references a column that does not exist on the actual
    table), the helper retries with ``L5_PLAN_PRE_PHASE15_COLS``
    and the caller is responsible for backfilling the missing
    columns as NULL.

    Phase 16: the ``model_role`` filter is applied on the
    SELECT when the caller supplies one. A bare ``model_role``
    filter is not allowed (it would match across models and
    silently mix roles); the helper raises if ``model_role`` is
    given without ``model_hash``. The legacy pre-Phase-16
    callers pass ``model_role=None``; their SELECT is not
    role-filtered.
    """
    if model_role is not None and model_hash is None:
        raise ValueError(
            "l5_outcome: a model_role filter requires model_hash; "
            "a bare model_role filter would silently mix roles "
            "across models. Pass model_hash=... when model_role is set."
        )
    where_clauses: list[str] = []
    if model_hash:
        where_clauses.append(f"model_hash = '{sql_escape(model_hash)}'")
    if model_role is not None:
        where_clauses.append(
            f"model_role = '{sql_escape(model_role)}'"
        )
    where = (
        (" WHERE " + " AND ".join(where_clauses))
        if where_clauses else ""
    )
    try:
        return db.query(
            f"SELECT {', '.join(full_cols)} FROM l5_plan_summary"
            + where
        )
    except Exception as e:
        # DuckDB error messages for a missing column start with
        # "Binder Error: Referenced column ... not found in FROM
        # clause". We fall back on any exception here; the
        # narrower check would be brittle across DuckDB versions.
        sys.stderr.write(
            f"l5_outcome: l5_plan_summary lacks Phase 15 "
            f"component columns; falling back to pre-Phase-15 "
            f"read ({e.__class__.__name__}: {str(e)[:200]})\n"
        )
        return db.query(
            "SELECT " + ", ".join(L5_PLAN_PRE_PHASE15_COLS)
            + " FROM l5_plan_summary" + where
        )


def compute_l5_outcome(
    db_path: str | Path,
    *,
    model_hash: str | None = None,
    model_role: str | None = None,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    write_back: bool = True,
) -> pl.DataFrame:
    """Compute the L5 outcome verdict and (optionally) write it back.

    Args:
        db_path: path to the unified tessera.duckdb file.
        model_hash: if non-None, restrict the join to this model.
            Default None = all models in the DB.
        model_role: if non-None (Phase 16), restrict the join
            to this architectural role. Default None = no role
            filter (the legacy pre-Phase-16 path; reads every
            role for the model). The role propagates to the
            l5_outcome rows; the per-(model, model_role, family)
            residual fit uses the role as a groupby key.
        accept_threshold: delta_mse below this is "accepted".
            Default 0.0 (any reduction counts as accept).
        write_back: if True, write the result to the l5_outcome
            table. If False, return the DataFrame without writing.

    Returns:
        The l5_outcome polars DataFrame. The same DataFrame is
        written to the DB when write_back=True.
    """
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"tessera.duckdb not found: {db_path}")

    with TesseraDB.open(db_path) as db:
        # Verify the unified schema is in place. If l5_plan_summary
        # or l4_plan_outcome is missing, the producer side hasn't
        # run yet; raise with a clear message.
        names = set(db.table_names())
        required = {"l5_plan_summary", "l4_plan_outcome", "l5_outcome"}
        missing = required - names
        if missing:
            raise RuntimeError(
                f"unified schema is missing tables: {sorted(missing)}. "
                f"Run a C++ dispatch (writes l4_plan_outcome) and the "
                f"Python l5_orchestrator (writes l5_plan_summary) first."
            )

        # Read the two source tables.
        #
        # Phase 15: l5_plan_summary gained the per-tensor component
        # columns (imatrix_magnitude, gradient_proxy,
        # layer_position_prior) on the Python side via ALTER TABLE
        # IF NOT EXISTS. On DBs created before Phase 15 the columns
        # are absent and the SELECT would fail; fall back to a
        # column-free projection that returns the same row set with
        # the components set to NULL downstream (see the
        # with_columns below).
        plan = _read_l5_plan_safe(
            db, model_hash, L5_PLAN_COLS, model_role=model_role,
        )
        # Phase 16: l4_plan_outcome does NOT carry a
        # model_role column on the C++ side. The role lives
        # on the l5_plan_summary / l5_outcome side. The
        # l4_plan_outcome SELECT is filtered by model_hash
        # only; the join with l5_plan_summary on
        # (model_hash, name, iteration, plan_id) preserves
        # the role dimension (the role comes from the plan
        # side after the join). A role-agnostic l4_plan_outcome
        # read is correct because the per-tensor outcome
        # doesn't carry a role tag.
        outcome_where_clauses: list[str] = []
        if model_hash:
            outcome_where_clauses.append(
                f"model_hash = '{sql_escape(model_hash)}'"
            )
        outcome_where = (
            (" WHERE " + " AND ".join(outcome_where_clauses))
            if outcome_where_clauses else ""
        )
        # Read l4_plan_outcome without the model_role
        # column. The role comes from the l5_plan_summary
        # side after the join; including model_role in the
        # SELECT would require the column to exist on the
        # l4_plan_outcome side (Phase 16 does not migrate
        # this column on the C++ side; legacy pre-Phase-16
        # l4_plan_outcome rows would fail).
        outcome_cols = [c for c in L4_PLAN_OUTCOME_COLS if c != "model_role"]
        try:
            outcome = db.query(
                f"SELECT {', '.join(outcome_cols)} FROM l4_plan_outcome"
                + outcome_where
            )
        except Exception as e:
            # Fallback for legacy DBs where the column list
            # is the strict pre-Phase-16 subset.
            sys.stderr.write(
                f"l5_outcome: l4_plan_outcome read failed; "
                f"falling back to pre-Phase-16 column list "
                f"({e.__class__.__name__}: {str(e)[:200]})\n"
            )
            raise

    if plan.height == 0 and outcome.height == 0:
        # Nothing to do; return an empty table with the right schema.
        return pl.DataFrame(schema=L5_OUTCOME_COLS)

    # The join. The (model_hash, name, iteration, plan_id) is the
    # PRIMARY KEY on both sides. Use left=plan, right=outcome so
    # plans that haven't been applied yet are visible (with null
    # outcome fields and plan_accepted=False). For the moment
    # we keep the inner join (only plans that have an outcome);
    # plans without an outcome are an error state the C++ side
    # should not produce.
    joined = plan.join(
        outcome,
        on=["model_hash", "name", "iteration", "plan_id"],
        how="inner",
        suffix="_outcome",
    )
    if joined.height == 0:
        return pl.DataFrame(schema=L5_OUTCOME_COLS)

    # Compute the verdict columns.
    def _delta(col_a: str, col_b: str) -> pl.Expr:
        return pl.col(col_a) - pl.col(col_b)

    verdict = joined.with_columns([
        _delta("mse_after", "mse_before").alias("delta_mse"),
        _delta("frob_after", "frob_before").alias("delta_frob"),
        (pl.col("mse_after") - pl.col("mse_before") < accept_threshold).alias("plan_accepted"),
        pl.lit(accept_threshold).alias("accept_threshold"),
    ])

    # Per-(model, model_role, family) linear fit of delta_mse on
    # sensitivity_score. residual_i = delta_mse_i - (a + b *
    # sensitivity_score_i). Uses polars'
    # .group_by().map_groups() with a numpy linear fit (2 points,
    # so a closed-form is overkill; closed-form OLS for
    # 2-coefficient regression is fine). If a (model, model_role,
    # family) group has fewer than 2 points, the residual is left
    # as 0 (a constant fit with no slope).
    #
    # Phase 16: the groupby key is now (model_hash, model_role,
    # family) so the trunk's residual fit and the dflash
    # encoder's residual fit are independent (the data they
    # fit is different; the OLS slope for one is not the
    # OLS slope for the other).
    import numpy as np

    def _fit_residual(group: pl.DataFrame) -> pl.DataFrame:
        x = group["sensitivity_score"].to_numpy()
        y = group["delta_mse"].to_numpy()
        if len(x) < 2 or np.allclose(x, x[0]):
            return group.with_columns(pl.lit(0.0).alias("residual"))
        # OLS: y = a + b*x; b = cov(x,y)/var(x); a = mean(y) - b*mean(x)
        xm = x.mean()
        ym = y.mean()
        dx = x - xm
        if np.dot(dx, dx) == 0.0:
            return group.with_columns(pl.lit(0.0).alias("residual"))
        b = float(np.dot(dx, y - ym) / np.dot(dx, dx))
        a = float(ym - b * xm)
        return group.with_columns(
            (pl.col("delta_mse") - (a + b * pl.col("sensitivity_score")))
            .alias("residual")
        )

    # Phase 16: model_role is part of the groupby key. When the
    # pre-Phase-16 path is taken (model_role=None or the column
    # is missing on the joined frame), the groupby falls back to
    # (model_hash, family) so the residual is computed on the
    # union of all roles for the model (the legacy behavior).
    if "model_role" in verdict.columns:
        group_keys = ["model_hash", "model_role", "family"]
    else:
        group_keys = ["model_hash", "family"]
    verdict = verdict.group_by(
        group_keys, maintain_order=True,
    ).map_groups(_fit_residual)

    # Project down to the l5_outcome column set. The plan and
    # outcome tables share the join key columns (model_hash, name,
    # iteration, plan_id) and `layer` and `updated_at` (which the
    # polars join suffixes as layer_outcome / updated_at_outcome).
    # The plan side's `layer` and `updated_at` are dropped; the
    # outcome side wins (the post-apply state is the one we want
    # to keep in the audit trail).
    #
    # Phase 15: the three per-tensor sensitivity component columns
    # live on l5_plan_summary (the plan side wins on the join).
    # The retune reads them to fit a 3-coefficient OLS that
    # decomposes which component is miscalibrated per (model,
    # family). When the columns are NULL (older rows), the retune
    # falls back to the 2-coefficient OLS on the combined
    # sensitivity_score. The columns are surfaced on the
    # l5_outcome projection so the consumer (l5_retune.py) does
    # not need a second join.
    #
    # Phase 16: the model_role column lives on l5_plan_summary
    # (the plan side wins on the join, like the per-tensor
    # component columns). It propagates through to the
    # l5_outcome projection; the retune uses it as a groupby
    # key for the 3-coefficient OLS. The legacy path
    # (pre-Phase-16 DBs without the column) backfills a
    # uniform "trunk" string so the projection has a
    # consistent column set.
    plan_drop = [c for c in ("layer", "updated_at") if c in verdict.columns]
    if plan_drop:
        verdict = verdict.drop(plan_drop)
    rename_map = {}
    if "layer_outcome" in verdict.columns:
        rename_map["layer_outcome"] = "layer"
    if "updated_at_outcome" in verdict.columns:
        rename_map["updated_at_outcome"] = "updated_at"
    # Pull the per-tensor component columns from the plan side.
    # They live on l5_plan_summary in Phase 15; the join's
    # suffixing leaves them at the top level (no _plan suffix
    # because the outcome side never has these column names).
    for comp_col in ("imatrix_magnitude", "gradient_proxy",
                     "layer_position_prior"):
        if comp_col not in verdict.columns:
            verdict = verdict.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(comp_col)
            )
    # Phase 16: model_role column backfill. The column lives
    # on the plan side; the join's suffixing leaves it at
    # the top level (no _plan suffix). Pre-Phase-16 rows
    # have no model_role column; substitute a uniform
    # "trunk" string so the projection has the column
    # populated and the retune's partition is uniform
    # within a model (the legacy behavior).
    if "model_role" not in verdict.columns:
        verdict = verdict.with_columns(
            pl.lit("trunk", dtype=pl.Utf8).alias("model_role")
        )
    if rename_map:
        verdict = verdict.rename(rename_map)
    out_cols = [
        "model_hash", "model_role", "name", "layer", "iteration", "plan_id",
        "family", "sensitivity_score",
        "imatrix_magnitude", "gradient_proxy", "layer_position_prior",
        "recommended_alpha", "recommended_clip",
        "mse_before", "mse_after", "delta_mse", "delta_frob",
        "plan_accepted", "accept_threshold", "residual", "updated_at",
    ]
    verdict = verdict.select(out_cols)

    if write_back:
        # The l5_outcome table is an upsert target (PRIMARY KEY on
        # model_hash, name, iteration, plan_id). For the moment
        # the typed helper appends; the unique key conflict is
        # resolved by DELETE-then-INSERT in a single transaction.
        # (The append is OK for first writes; reruns would
        # duplicate. The DELETE-then-INSERT is the production
        # path; see l5_outcome.py:replace_l5_outcome below.)
        #
        # Phase 16: the DELETE-then-INSERT pass uses
        # replace_l5_outcome which now keys the delete on
        # (model_hash, model_role) so other roles for the
        # same model are preserved.
        replace_l5_outcome(
            db_path,
            model_hash=model_hash,
            model_role=model_role,
            new_rows=verdict,
        )

    return verdict


def replace_l5_outcome(
    db_path: str | Path,
    *,
    model_hash: str | None,
    new_rows: pl.DataFrame,
    model_role: str | None = None,
) -> int:
    """Delete the existing l5_outcome rows for
    ``(model_hash, model_role)`` (or all roles when
    ``model_role`` is None, or all models when ``model_hash``
    is also None) and re-insert the supplied rows. Wrapped
    in a single transaction so a concurrent reader sees
    either the old or the new state, not a partial one.
    Returns the number of rows written.

    Phase 16: the DELETE key is extended to
    ``(model_hash, model_role)`` so other roles for the
    same model are preserved. When both ``model_hash`` and
    ``model_role`` are None, the entire table is replaced
    (the legacy pre-Phase-16 path).
    """
    if new_rows.height == 0:
        return 0
    with TesseraDB.open(db_path) as db:
        con = db._conn
        con.execute("BEGIN")
        try:
            where_clauses: list[str] = []
            if model_hash is not None:
                where_clauses.append(
                    f"model_hash = '{sql_escape(model_hash)}'"
                )
            if model_role is not None:
                where_clauses.append(
                    f"model_role = '{sql_escape(model_role)}'"
                )
            if where_clauses:
                con.execute(
                    "DELETE FROM l5_outcome WHERE "
                    + " AND ".join(where_clauses)
                )
            else:
                con.execute("DELETE FROM l5_outcome")
            db.insert_l5_outcome(
                model_hash=model_hash or "all_models",
                model_role=model_role or "trunk",
                rows=[row for row in new_rows.to_dicts()],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    return new_rows.height


def summarize(verdict: pl.DataFrame) -> OutcomeSummary:
    """Compute a per-run summary of the l5_outcome verdict. The
    per-family hit_rate is the fraction of plans in that family
    with delta_mse < accept_threshold. Empty input -> empty
    summary.
    """
    if verdict.height == 0:
        return OutcomeSummary()
    n_plans = verdict.height
    n_accepted = int(verdict["plan_accepted"].sum())
    hit_rate = n_accepted / n_plans if n_plans else 0.0
    mean_delta = float(verdict["delta_mse"].mean())
    mean_residual = float(verdict["residual"].abs().mean())
    per_family = (
        verdict.group_by("family")
        .agg(
            n=pl.len(),
            n_acc=pl.col("plan_accepted").sum(),
        )
        .with_columns(
            (pl.col("n_acc") / pl.col("n")).alias("hit_rate")
        )
        .select("family", "hit_rate")
        .to_dicts()
    )
    return OutcomeSummary(
        n_plans=n_plans,
        n_accepted=n_accepted,
        hit_rate=hit_rate,
        mean_delta_mse=mean_delta,
        mean_residual=mean_residual,
        per_family_hit_rate={r["family"]: r["hit_rate"] for r in per_family},
    )


def sql_escape(s: str) -> str:
    return s.replace("'", "''")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "L5 outcome: join l5_plan_summary + l4_plan_outcome, compute "
            "delta_mse, write l5_outcome. The 'did this requant plan "
            "reduce error?' verdict for the cross-pipeline feedback loop."
        ),
    )
    p.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to the unified tessera.duckdb file",
    )
    p.add_argument(
        "--model-hash",
        default=None,
        help=(
            "Restrict to this model_hash. Default: all models in the "
            "DB. Useful when re-running for one model without touching "
            "the rest."
        ),
    )
    p.add_argument(
        "--model-role",
        default=None,
        choices=[
            "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
        ],
        help=(
            "Restrict to this model_role (Phase 16). Default: no "
            "role filter (the legacy pre-Phase-16 path; reads "
            "every role for the model). The role is part of the "
            "l5_outcome PRIMARY KEY: the same (model, name, "
            "iteration, plan_id) can have one l5_outcome row per "
            "role (e.g. one row for the trunk's attn_q plan and "
            "another for the dflash encoder's attn_q plan). "
            "Requires --model-hash (a bare model_role filter "
            "would silently mix roles across models)."
        ),
    )
    p.add_argument(
        "--accept-threshold",
        type=float,
        default=DEFAULT_ACCEPT_THRESHOLD,
        help=(
            "delta_mse below this is 'plan accepted' (i.e. the plan "
            "actually reduced the error). Default 0.0 (any reduction "
            "counts). Tighter values (e.g. 0.005) require a more "
            "meaningful reduction."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Compute the verdict and print the summary without "
            "writing back to l5_outcome. Useful for sanity checks."
        ),
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the per-run summary after writing.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.model_role is not None and args.model_hash is None:
        print(
            "ERROR: --model-role requires --model-hash; a bare "
            "model_role filter would silently mix roles across "
            "models.",
            file=sys.stderr,
        )
        return 2
    verdict = compute_l5_outcome(
        args.db,
        model_hash=args.model_hash,
        model_role=args.model_role,
        accept_threshold=args.accept_threshold,
        write_back=not args.dry_run,
    )
    summary = summarize(verdict)
    if args.print_summary or args.dry_run:
        print(summary)
    if args.dry_run:
        return 0
    # Default behavior: write + print a one-liner.
    print(
        f"l5_outcome: wrote {verdict.height} rows "
        f"(hit_rate={summary.hit_rate:.3f}, "
        f"mean_delta_mse={summary.mean_delta_mse:+.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
