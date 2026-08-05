"""tools/tessera/embedding_budget.py

M2 producer: per-role size budgets for the SHARED tensors
(token_embd / output) in the now-singular-GGUF's mmproj pipeline.

The M0a surface (commits 234333cec + c64e9a85a) made the mmproj
components (vision_tower / audio_tower / mm_projector) first-class
in the unified writer. M2 gives them a real budget: the producer
reads tensor_stats + the per-tensor calibration verdicts, computes
a per-role size envelope for the shared embeddings, and emits a
list of ``RoleBudget`` records that the Phase 16.8 writer consumes
via its ``role_budgets`` sidecar key.

The shape is the residual-envelope formula::

    budget_bits(r, t) = clamp( (E(r) - S_t(r) - M(r)) / N(t),  0,  16 )

    E(r)   = source_footprint_bits(r) * base_budget_fraction
    S_t(r) = sum over role r's NON-shared tensor_stats rows of
             n_elements * dtype_bits (verdict dtype when present,
             else source dtype)
    M(r)   = sum over role r's v./a./mm.* tensor_stats rows of
             n_elements * dtype_bits * base_budget_fraction
    N(t)   = the shared tensor's n_elements

The L5 producer (``l5_retune.py``) remains the source of truth
for the attn/ffn family budgets; its
``family_storage_bits * (1 - hit_rate) * fraction`` rule does not
transfer to embeddings (they are never requantized). The M2
producer is a NEW producer; it does not modify l5_retune.

The producer is pure: it takes pre-loaded ``policy_entries`` and
``tensor_stats_rows`` plus an ``EnvelopeConfig``, and returns a
list of ``RoleBudget`` dataclasses. A CLI entry point reads
tensor_stats from the unified ``tessera.duckdb`` and writes the
sidecar JSON shape the writer expects. ``unified_calibrate.py``
wires the producer into the calibration flow so the per-component
verdicts flow directly into the budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


# Reuse l5_retune's DTYPE_BITS / _dtype_bits. The C++ writer's
# ts_unified_writer_qtype_bits is the single source of truth on
# the bit-cost ordering; l5_retune's DTYPE_BITS map mirrors it
# (no block overhead, integer bits only). Importing the helpers
# from l5_retune keeps the producer and the L5 consumer in lock-
# step: a new dtype added to the writer shows up here on the
# next import. l5_retune is in the same directory, so the
# sys.path manipulation below makes it importable when this
# module is invoked as ``python3 -m tools.tessera.embedding_budget``.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from l5_retune import DTYPE_BITS, _dtype_bits  # noqa: E402


# Tensors the writer reconciles across multiple roles (Phase 16
# + M2). These are the SHARED tensors the producer emits budgets
# for. Non-shared tensors are governed by the L5 requant loop's
# family budget, not by this producer.
SHARED_TENSOR_NAMES: tuple[str, ...] = (
    "token_embd.weight",
    "output.weight",
)

# Roles that may own a shared tensor. The M0a surface added
# vision_tower / audio_tower / mm_projector (c64e9a85a) to the
# role set the unified writer recognizes; M2 treats them
# uniformly with the legacy trunk / dflash pair.
SHARED_OWNING_ROLES: tuple[str, ...] = (
    "trunk",
    "dflash",
    "vision_tower",
    "audio_tower",
    "mm_projector",
)

# Default per-role priority. Dflash is boosted to 2.0 because
# speculative decoding can absorb its errors; everything else
# is 1.0 unless the CLI / EnvelopeConfig overrides via
# ``--role-priority role=value`` (the CLI override is a list
# of ``role=priority`` strings, applied after the default).
DEFAULT_ROLE_PRIORITIES: dict[str, float] = {
    "trunk":        1.0,
    "dflash":       2.0,
    "vision_tower": 1.0,
    "audio_tower":  1.0,
    "mm_projector": 1.0,
}

# Confidence-scaling n_target. n_samples(r) is the count of
# NON-shared calibration verdicts for role r; confidence is
# ``min(1, n_samples(r) / n_target)``. With n_target=8, a
# well-calibrated role (8+ verdicts) gets full confidence; an
# under-calibrated role is damped. n_samples = 0 ->
# confidence = 0 -> weight = 0, which the writer's
# weight-relaxation logic treats as "unconstrained" (any
# conflicting role's weight >= 0 dominates, so the constraint
# is always relaxed).
DEFAULT_N_TARGET: int = 8

# Upper clamp on budget_bits/element. 16 is the bit cost of
# f16, the highest-precision source dtype; we never recommend
# a budget above the source (the verdict is already at the
# source when no calibration has downgraded the tensor).
BUDGET_CLAMP_MAX: int = 16

# The mmproj name prefixes the producer scans for when summing
# the role's mmproj footprint M(r). The prefix is a leading
# dot to match the model side's naming convention (e.g.
# ``v.attn.0.weight``); a ``vattn.0.weight`` (no dot) does not
# match. Mirrors tools/mtmd/clip.cpp's v/a/mm namespacing
# referenced in the M0a commit.
MMPROJ_NAME_PREFIXES: tuple[str, ...] = ("v.", "a.", "mm.")


@dataclass
class EnvelopeConfig:
    """The deployer-facing knobs for the M2 producer.

    Attributes:
      base_budget_fraction: the role's source_footprint_bits is
        scaled by this to form the envelope E(r). Mirrors
        l5_retune's --budget-fraction flag. Set to 0 (or any
        non-positive value) to opt out of the size envelope;
        the producer returns ``[]`` (the writer's no-budget
        contract).
      role_priorities: per-role priority override. The default
        table (``DEFAULT_ROLE_PRIORITIES``) is consulted
        first; any role present in this dict overrides the
        default. ``weight = priority(r) * confidence(r)``.
      n_target: the n_samples target for full confidence
        (1.0). A role with n_samples >= n_target gets the full
        priority. A role with n_samples < n_target is damped
        by n_samples / n_target. n_samples = 0 -> confidence
        = 0 -> weight = 0 (the writer treats this role as
        unconstrained: any conflicting role's weight >= 0
        dominates, so the constraint is always relaxed).
    """

    base_budget_fraction: float = 1.0
    role_priorities: dict[str, float] = field(default_factory=dict)
    n_target: int = DEFAULT_N_TARGET


@dataclass
class RoleBudget:
    """One per-(role, shared_tensor_name) verdict for the writer.

    Attributes:
      model_role: the role that owns the shared tensor.
      budget_bits_per_elem: the per-element bit budget,
        clamped to [0, 16]. 0 means the role has no residual
        envelope for the shared tensor (the writer's
        relaxation logic still emits a no-op event, but the
        verdict is preserved). -1 would mean unconstrained,
        but the M2 producer does not emit -1: the architect's
        design decision is that the producer is for the
        embedded path, not the requant path; "unconstrained"
        is the pre-M2 contract (the writer's no-budget
        default).
      weight: priority(r) * confidence(r); drives the
        writer's dynamic-weighting relaxation logic.

    The dataclass intentionally does NOT carry the
    shared_tensor_name. The producer's CLI flattens to one
    entry per role for the writer's role_budgets sidecar
    (the writer's lookup is first-match-wins per model_role;
    multiple per-role entries would shadow each other in
    undefined order). The flatten is the MIN budget across
    the role's shared tensors (most conservative) and the
    MAX weight (the strongest dynamic-weighting signal).
    Weight is the priority * confidence signal, which is the
    same per role regardless of shared tensor; "max" is
    defensive against a future spec change that weights per
    shared tensor. The per-(role, shared_tensor_name) list
    is exposed via ``compute_role_budgets`` for tests and
    downstream consumers that want the full resolution.
    """

    model_role: str
    budget_bits_per_elem: int
    weight: float


def _source_footprint_bits(
    tensor_stats_rows: Iterable[dict],
    model_role: str,
) -> int:
    """Sum n_elements * dtype_bits over a role's tensor_stats rows.

    Unknown dtypes and NULL n_elements are skipped (no
    poisoning of the sum). The function is the role-wide
    envelope helper; E(r) is
    ``source_footprint_bits(r) * base_budget_fraction``.
    """
    total = 0
    for r in tensor_stats_rows:
        if r.get("model_role") != model_role:
            continue
        n = r.get("n_elements")
        bpe = _dtype_bits(r.get("dtype"))
        if n is None or bpe is None:
            continue
        total += int(n) * int(bpe)
    return total


def _mmproj_footprint_bits(
    tensor_stats_rows: Iterable[dict],
    model_role: str,
) -> int:
    """Sum n_elements * dtype_bits over a role's v./a./mm.* rows.

    Returns the raw ``mmproj_total_bits(r)`` (NOT scaled by
    ``base_budget_fraction``); the main function applies the
    fraction. The prefix match is a leading-dot literal so
    ``v.attn.0.weight`` matches but ``vattn.0.weight`` does
    not.
    """
    total = 0
    for r in tensor_stats_rows:
        if r.get("model_role") != model_role:
            continue
        name = r.get("name") or ""
        if not any(name.startswith(p) for p in MMPROJ_NAME_PREFIXES):
            continue
        n = r.get("n_elements")
        bpe = _dtype_bits(r.get("dtype"))
        if n is None or bpe is None:
            continue
        total += int(n) * int(bpe)
    return total


def _priority_for_role(role: str, cfg: EnvelopeConfig) -> float:
    """Resolve the effective priority for a role.

    Lookup order:
      1. ``cfg.role_priorities[role]`` (CLI / EnvelopeConfig override)
      2. ``DEFAULT_ROLE_PRIORITIES[role]`` (architect's default)
      3. 1.0 (fallback; should rarely fire — fires only when
         the role is in ``SHARED_OWNING_ROLES``-adjacent but
         not in the default table, e.g. a future "mtp_nextn"
         owner of the shared tensor).
    """
    if role in cfg.role_priorities:
        return float(cfg.role_priorities[role])
    if role in DEFAULT_ROLE_PRIORITIES:
        return float(DEFAULT_ROLE_PRIORITIES[role])
    return 1.0


def _n_samples_for_role(
    policy_entries: Iterable[dict],
    model_role: str,
) -> int:
    """Count the NON-shared calibration verdicts for one role.

    n_samples is the calibration confidence proxy: a role with
    more non-shared verdicts is more confidently calibrated.
    Counting only non-shared verdicts makes the
    ``n_samples=0`` case reachable for a role that owns a
    shared tensor with no other calibration data; the
    writer's weight-relaxation logic treats a 0-weight role
    as unconstrained, which is the architect's "no
    calibration = unconstrained" rule. Shared-tensor verdicts
    are not counted toward n_samples because they are the
    subject of the budget, not a calibration input.
    """
    return sum(
        1 for e in policy_entries
        if e.get("model_role") == model_role
        and e.get("name") not in SHARED_TENSOR_NAMES
    )


def _verdict_dtype(
    verdicts: dict[tuple[str, str], str],
    role: str,
    name: str,
    fallback: str | None,
) -> str | None:
    """Resolve a tensor's effective dtype for the size sum.

    The verdict map is keyed by (model_role, name); the
    fallback is the tensor_stats dtype (the source qtype).
    Returns None when both are absent (the row is skipped,
    not poisoned).
    """
    v = verdicts.get((role, name))
    if v is not None:
        return v
    return fallback


def _find_n_elements(
    tensor_stats_rows: Iterable[dict],
    role: str,
    name: str,
) -> int | None:
    """Look up the n_elements for a (role, name) in tensor_stats.

    Returns None when no row matches or the matching row's
    n_elements is NULL; the caller treats None as "no budget"
    (we cannot divide by zero).
    """
    for r in tensor_stats_rows:
        if r.get("model_role") == role and r.get("name") == name:
            n = r.get("n_elements")
            if n is None:
                return None
            return int(n)
    return None


def compute_role_budgets(
    policy_entries: Sequence[dict],
    tensor_stats_rows: Sequence[dict],
    envelope_cfg: EnvelopeConfig,
) -> list[RoleBudget]:
    """The pure M2 producer.

    Returns a list of ``RoleBudget`` records, one per
    (role, shared_tensor_name) where the role owns the shared
    tensor. The CLI flattens this to a per-role sidecar via
    :py:func:`flatten_role_budgets_for_sidecar`; tests and
    downstream consumers that want the per-(role,
    shared_tensor) resolution use this function directly.

    Edge cases:
      * ``base_budget_fraction <= 0``: returns ``[]`` (mirrors
        l5_retune's NULL semantics; the user has opted out
        of the size envelope).
      * Empty policy_entries: returns ``[]``.
      * Role with policy verdicts but no shared-tensor
        ownership: skipped (the L5 family budget still
        governs the requant loop for that role).
      * Role with no tensor_stats rows: skipped (no spurious
        budget row when the calibration data is absent).
      * Shared tensor with no n_elements: that (role, name)
        is skipped (cannot divide).
      * Negative residual: budget = 0 (NOT NULL); the
        writer's relaxation logic handles a 0 budget with
        the dynamic-weighting rule. A warning is logged via
        ``warnings.warn`` so the operator can see when the
        role's envelope is over-subscribed.

    Args:
      policy_entries: the per-tensor verdicts. Each entry is
        a dict with at least ``model_role`` and ``name``;
        optional ``dtype`` is the resolved qtype (when
        absent, the entry is ignored — the tensor_stats
        dtype is the implicit verdict for non-shared rows).
      tensor_stats_rows: the per-tensor calibration stats.
        Each row is a dict with at least ``model_role``,
        ``name``, ``n_elements``, and ``dtype``. M0a
        v./a./mm.* rows are included (the M(r) term sums
        them).
      envelope_cfg: the deployer-facing knobs.

    Returns:
      A list of RoleBudget records, sorted by (model_role,
      shared_tensor_name) for stable output.
    """
    if envelope_cfg.base_budget_fraction <= 0.0:
        return []

    # 1. Build the verdict map. Verdicts without a dtype are
    #    skipped (they would be no-ops anyway; the
    #    tensor_stats dtype is the implicit fallback for
    #    non-shared rows, and a shared-tensor verdict
    #    without a dtype does not establish role ownership
    #    of the shared tensor).
    verdicts: dict[tuple[str, str], str] = {}
    for e in policy_entries:
        role = e.get("model_role")
        name = e.get("name")
        dtype = e.get("dtype")
        if not role or not name or not dtype:
            continue
        verdicts[(role, str(name))] = str(dtype)

    if not verdicts:
        return []

    # 2. Find the roles that own at least one shared tensor.
    roles_with_shared: set[str] = set()
    for (role, name) in verdicts:
        if name in SHARED_TENSOR_NAMES:
            roles_with_shared.add(role)

    out: list[RoleBudget] = []
    for role in sorted(roles_with_shared):
        # E(r) = source_footprint_bits(r) * base_budget_fraction.
        e_bits = _source_footprint_bits(tensor_stats_rows, role) \
            * envelope_cfg.base_budget_fraction
        # S_t(r) = sum of non-shared rows using verdict dtype
        #          when present, else tensor_stats dtype.
        s_minus = 0
        for r in tensor_stats_rows:
            if r.get("model_role") != role:
                continue
            rname = r.get("name")
            if rname in SHARED_TENSOR_NAMES:
                continue
            n = r.get("n_elements")
            effective_dtype = _verdict_dtype(
                verdicts, role, rname, r.get("dtype"),
            )
            bpe = _dtype_bits(effective_dtype)
            if n is None or bpe is None:
                continue
            s_minus += int(n) * int(bpe)
        # M(r) = mmproj_total_bits(r) * base_budget_fraction.
        m_bits = _mmproj_footprint_bits(tensor_stats_rows, role) \
            * envelope_cfg.base_budget_fraction

        # Per-(role, shared_tensor_name) budgets.
        n_samples = _n_samples_for_role(policy_entries, role)
        confidence = (
            min(1.0, n_samples / envelope_cfg.n_target)
            if envelope_cfg.n_target > 0
            else 0.0
        )
        priority = _priority_for_role(role, envelope_cfg)
        weight = priority * confidence

        for shared_name in SHARED_TENSOR_NAMES:
            if (role, shared_name) not in verdicts:
                continue
            n_t = _find_n_elements(tensor_stats_rows, role, shared_name)
            if n_t is None or n_t <= 0:
                # No n_elements for the shared tensor; we
                # cannot divide. Skip this (role, name) entry
                # rather than emitting a spurious 0/0 or
                # NaN.
                continue
            residual = e_bits - s_minus - m_bits
            if residual < 0:
                import warnings
                warnings.warn(
                    f"embedding_budget: role {role!r} "
                    f"shared_tensor {shared_name!r} has "
                    f"negative residual "
                    f"(E={e_bits:.0f}, S_t={s_minus:.0f}, "
                    f"M={m_bits:.0f}); budget=0 (writer will "
                    f"apply dynamic-weighting relaxation).",
                    stacklevel=2,
                )
                budget = 0
            else:
                budget = int(
                    max(0, min(BUDGET_CLAMP_MAX, residual / n_t))
                )
            out.append(RoleBudget(
                model_role=role,
                budget_bits_per_elem=budget,
                weight=weight,
            ))

    return out


def flatten_role_budgets_for_sidecar(
    role_budgets: Sequence[RoleBudget],
) -> list[dict]:
    """Flatten the per-(role, shared_tensor) list to the writer's
    sidecar shape.

    The writer's ``ts_unified_policy_load_json`` parses the
    sidecar's ``role_budgets`` as a list of
    ``{model_role, budget_bits, weight}`` entries. The
    writer's per-tensor lookup is first-match-wins per
    model_role; if the producer emitted multiple entries for
    the same role (the role owns both token_embd and
    output), the writer would pick whichever entry happens
    to be first in the array. To make the budget
    deterministic, this flattens per role to the MIN budget
    across the role's shared tensors (the most conservative
    constraint) and the MAX weight (the strongest
    dynamic-weighting signal). Weight is the priority *
    confidence signal, which is the same per role
    regardless of shared tensor; "max" is defensive against
    a future spec change that weights per shared tensor.

    The output is sorted by model_role for stable ordering
    across runs.
    """
    by_role: dict[str, list[RoleBudget]] = {}
    for rb in role_budgets:
        by_role.setdefault(rb.model_role, []).append(rb)
    flat: list[dict] = []
    for role in sorted(by_role):
        entries = by_role[role]
        budget = min(e.budget_bits_per_elem for e in entries)
        weight = max(e.weight for e in entries)
        flat.append({
            "model_role":  role,
            "budget_bits": int(budget),
            "weight":      float(weight),
        })
    return flat


# Schema marker for the CLI's standalone sidecar. Mirrors the
# writer's ``ts_unified_policy_save_json`` schema value so the
# sidecar is loadable as a minimal policy. The writer's load
# is tolerant of missing keys, so an empty ``tensor_families``
# list is fine; only ``role_budgets`` is non-empty.
SIDECAR_SCHEMA: str = "llama.speculative.calibration-policy.v1"


def write_sidecar_json(
    path: Path,
    role_budgets: Sequence[RoleBudget],
) -> None:
    """Write the writer-loadable sidecar JSON to ``path``.

    The schema is the same one ``unified_calibrate.py`` emits
    (the writer treats both as the same policy sidecar). The
    ``tensor_families`` field is an empty list (this sidecar
    carries no per-tensor verdicts; the producer that needs
    those would be ``unified_calibrate.py``). ``role_budgets``
    is the per-role flattened list.
    """
    sidecar: dict = {
        "schema":          SIDECAR_SCHEMA,
        "tensor_families": [],
        "role_budgets":    flatten_role_budgets_for_sidecar(role_budgets),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_role_priority_overrides(
    specs: Sequence[str],
) -> dict[str, float]:
    """Parse ``--role-priority role=value`` entries.

    Empty values and malformed values raise ValueError. The
    caller (the CLI) catches and reports.
    """
    out: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"--role-priority expects role=priority, got {spec!r}"
            )
        role, val_str = spec.split("=", 1)
        role = role.strip()
        if not role:
            raise ValueError(
                f"--role-priority: empty role in {spec!r}"
            )
        try:
            val = float(val_str)
        except ValueError as exc:
            raise ValueError(
                f"--role-priority: bad priority {val_str!r} "
                f"in {spec!r}: {exc}"
            ) from exc
        out[role] = val
    return out


def _sql_escape(s: str) -> str:
    """Crude SQL string escape for a value. Doubles single
    quotes so a model_hash with a quote cannot break the
    query.

    The function is intentionally minimal: model_hash values
    are SHA-256 / sha-1 hex strings in production, so the
    escape is defense-in-depth against a bad test fixture.
    """
    return str(s).replace("'", "''")


def _load_tensor_stats_from_db(
    db_path: Path,
    model_hash: str,
) -> list[dict]:
    """Read tensor_stats rows for a model as a list of dicts.

    The DB is the production path; the C++ side has already
    written the calibration stats. We read via polars +
    duckdb, then convert to plain dicts so the pure function
    can be tested without a live DB.

    The CLI does not read a separate ``policy_verdicts``
    table: that table does not exist on the unified-DB
    schema (the policy lives in the sidecar JSON produced
    by ``unified_calibrate.py``). The CLI's "verdict" is
    the source qtype from tensor_stats, which is what the
    role "owns" by default. The unified_calibrate path is
    the rich path with real verdicts.
    """
    from tessera_db import TesseraDB  # local import; the
    # tessera_db module is heavy (duckdb) and we do not
    # want the test path to pay for it.

    with TesseraDB.open(db_path, read_only=True) as db:
        names = set(db.table_names())
        if "tensor_stats" not in names:
            raise RuntimeError(
                f"tessera.duckdb is missing tensor_stats "
                f"table: {db_path}"
            )
        df = db.query(
            "SELECT model_hash, model_role, name, n_elements, "
            "dtype FROM tensor_stats WHERE model_hash = "
            f"'{_sql_escape(model_hash)}'"
        )
    return df.to_dicts()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "M2 producer: per-role size budgets for the "
            "SHARED tensors (token_embd / output) in the "
            "unified GGUF. Reads tensor_stats from the "
            "unified tessera.duckdb, calls "
            "compute_role_budgets, writes the "
            "writer-loadable role_budgets sidecar JSON."
        ),
    )
    p.add_argument(
        "--db", required=True, type=Path,
        help="Path to the unified tessera.duckdb file",
    )
    p.add_argument(
        "--model-hash", required=True,
        help="Restrict to this model_hash",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Sidecar JSON output path",
    )
    p.add_argument(
        "--budget-fraction", type=float, default=1.0,
        help=(
            "Base fraction for the role size envelope: "
            "E(r) = source_footprint_bits(r) * fraction. "
            "0 (or any non-positive value) returns an "
            "empty role_budgets (the user has opted out of "
            "the size envelope). Default 1.0."
        ),
    )
    p.add_argument(
        "--role-priority", action="append", default=[],
        metavar="ROLE=PRIORITY",
        help=(
            "Override the default per-role priority. May "
            "be specified multiple times. Example: "
            "--role-priority dflash=3.0. Default: "
            "dflash=2.0, others=1.0."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    priority_overrides = _parse_role_priority_overrides(
        args.role_priority
    )
    cfg = EnvelopeConfig(
        base_budget_fraction=args.budget_fraction,
        role_priorities=priority_overrides,
    )
    rows = _load_tensor_stats_from_db(args.db, args.model_hash)
    # No separate verdict table; tensor_stats is the verdict
    # at the source qtype. If a future commit adds a
    # ``policy_verdicts`` table, the CLI can be extended to
    # join on it. Today the standalone CLI produces budgets
    # from source qtypes only — the rich path is
    # unified_calibrate.py, which builds verdicts from the
    # per_tensor_calibrate output and passes them to
    # ``compute_role_budgets`` directly.
    budgets = compute_role_budgets(
        policy_entries=rows,
        tensor_stats_rows=rows,
        envelope_cfg=cfg,
    )
    write_sidecar_json(args.output, budgets)
    sidecar = flatten_role_budgets_for_sidecar(budgets)
    print(
        f"wrote embedding-budget sidecar: {args.output} "
        f"({len(budgets)} (role, shared_tensor) verdicts, "
        f"{len(sidecar)} per-role entries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
