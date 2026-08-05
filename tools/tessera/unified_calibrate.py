#!/usr/bin/env python3
"""Unified per-component Tessera calibration driver.

The Tessera quant pipeline has separate calibration paths for the
trunk, the drafter heads (DFlash / DSpark), and the assistant (MTP).
A speculative-decoding runtime needs the per-tensor quantization
parameters to be CONSISTENT across components: if the drafter
quantises blk.0.attn_q.weight to 3 bits with one scale and the
verifier quantises the same tensor to 3 bits with a different scale
(because the calibrations were independent), the acceptance rate
collapses. This driver runs ``per_tensor_calibrate.py`` on each
component in a single pass against a shared calibration corpus, then
combines the per-component policies into a single
``llama.speculative.calibration-policy.v1`` JSON keyed by
``model_role`` so the downstream quantizer
(``tile640_quantize_v3.py --calibration-policy``) and the unified
GGUF writer (``llama-quantize --write-unified-gguf``) can route
per-tensor parameters back to the right component.

Layout:

* ``--component role=path`` (one or more) declares each component.
  ``role`` is the ``model_role`` string the unified policy emits;
  conventional values are ``trunk`` / ``dspark`` / ``dflash`` /
  ``assistant`` / ``mtp``, but any non-empty string is accepted.
  ``path`` is a directory of ``.npz`` bundles or a single ``.npz``
  file consumable by ``per_tensor_calibrate.py --layers``.
* ``--fitness`` is forwarded to ``per_tensor_calibrate.py`` (the
  same ``lrq`` / ``awq`` / ``flrq`` / ``dartquant`` modes) when
  ``--fitness-default`` is not ``auto``.
* ``--fitness-default {auto,lrq,awq,flrq,dartquant,compare}``
  (default ``auto``) controls how the per-component ``--fitness``
  is chosen. ``auto`` consults ``ROLE_DEFAULT_FITNESS``:
  ``trunk->awq``, ``dflash/dspark/mtp_nextn->lrq``,
  ``shared_embd->flrq``. Any other value overrides ``--fitness``
  on every component.
* ``--per-tensor-calibrate`` overrides the path to the inner driver
  (default: ``tools/tessera/per_tensor_calibrate.py`` alongside
  this file).
* ``--output`` is the unified policy JSON.

The unified policy is structurally identical to the single-component
policy ``per_tensor_calibrate.py`` emits, with two additions:

* A top-level ``model_roles`` list enumerating the components in
  registration order.
* A ``model_role`` field on every tensor in
  ``tensor_families`` and every record in the per-fitness
  ``tensors`` list. The per-tensor name is preserved verbatim
  (the same name the single-component policy would emit), so the
  downstream quantizer's tensor-name resolution is unchanged; the
  ``model_role`` field is the disambiguator.

This is the "per-component calibration driver" the calibration
roadmap calls out as the missing step between the existing
single-component ``per_tensor_calibrate.py`` and the unified
quantize / retune flow. The ``model_role`` field is the
forward-compatible contract the roadmap says the DB schema
(``tensor_stats`` / ``l5_weights`` / etc.) will need; we surface
it at the policy layer first so the writer side can adopt it
without a DB migration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


SCHEMA = "llama.speculative.calibration-policy.v1"
UNIFIED_SCHEMA = "llama.tessera.unified-calibration-policy.v1"

# Per-component fitness policy (Phase 16 calibrate-model-role
# follow-up). Each model_role has a default --fitness strategy
# because the calibration cost / accuracy tradeoff differs by
# role:
#
#   trunk        -> awq   The heavy hitter is the FFN; the
#                          GA-driven AWQ search with the family
#                          warm-start minimises the trunk's
#                          layer-output error the most. AWQ is
#                          the legacy default for the trunk.
#   dflash       -> lrq   The drafter is already lossy
#                          (speculative decoding can absorb the
#                          error); LRQ's smaller-footprint
#                          low-rank scale fits the drafter's
#                          tight memory budget. AWQ is
#                          overkill here.
#   dspark       -> lrq   Same rationale as dflash.
#   mtp_nextn    -> lrq   The MTP nextn heads are smaller than
#                          the trunk and benefit from the
#                          low-rank compression. AWQ's GA
#                          search budget is wasted on a small
#                          tensor.
#   shared_embd  -> flrq  The token_embd / output layers are
#                          frozen at train; perturbing the
#                          weight is wasted compute. FLRQ is
#                          calibration-free (uses only W, not
#                          the activations) so it's the
#                          right tool for a frozen tensor.
#
# The --fitness flag (explicit) overrides this table on every
# component. The --fitness-default flag controls whether the
# table is consulted at all: "auto" uses the per-role default
# (this table), anything else overrides on every component
# the same way --fitness does. "auto" is the recommended
# default for unified Calibrate because the per-role policy
# is what the calibration roadmap says is correct.
FITNESS_CHOICES = ("lrq", "awq", "flrq", "dartquant", "compare")
ROLE_DEFAULT_FITNESS: dict[str, str] = {
    "trunk":       "awq",
    "dflash":      "lrq",
    "dspark":      "lrq",
    "mtp_nextn":   "lrq",
    "shared_embd": "flrq",
}


def _parse_components(values: Iterable[str]) -> list[tuple[str, Path]]:
    """Parse --component ROLE=PATH entries. PATH must exist."""
    out: list[tuple[str, Path]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"--component expects ROLE=PATH, got {raw!r}")
        role, path_str = raw.split("=", 1)
        role = role.strip()
        path = Path(path_str.strip())
        if not role:
            raise ValueError(f"--component: empty role in {raw!r}")
        if not path.exists():
            raise ValueError(f"--component: {path} does not exist")
        out.append((role, path))
    if not out:
        raise ValueError("--component: at least one component is required")
    return out


def resolve_fitness(
    role: str,
    fitness_arg: str,
    fitness_default: str,
) -> str:
    """Resolve the effective --fitness for one component.

    The semantics:

    * ``--fitness X`` (explicit) wins on every component
      regardless of role. This is the legacy single-mode
      behaviour: one fitness strategy drives every
      component.
    * ``--fitness-default auto`` (the recommended default)
      picks the per-role fitness from ``ROLE_DEFAULT_FITNESS``.
      Unknown roles fall back to ``awq`` (the legacy default).
    * ``--fitness-default X`` (any non-auto value) treats
      ``X`` as the override and ignores ``--fitness``.

    The function is the single source of truth for "what
    fitness should this component run?" so the test can
    pin the per-role table independently of the CLI.
    """
    # --fitness wins when set. The CLI default is "lrq" for
    # backward compat with the Phase 16 pre-followup default;
    # the unified Calibrate wrapper rewires that to
    # "auto" via --fitness-default so the per-role table is
    # consulted by default.
    if fitness_default != "auto":
        if fitness_default not in FITNESS_CHOICES:
            raise ValueError(
                f"--fitness-default {fitness_default!r} not in "
                f"{FITNESS_CHOICES!r}"
            )
        return fitness_default
    # fitness_default == "auto": consult the per-role table.
    if fitness_arg not in FITNESS_CHOICES:
        raise ValueError(
            f"--fitness {fitness_arg!r} not in {FITNESS_CHOICES!r}"
        )
    return ROLE_DEFAULT_FITNESS.get(role, fitness_arg)


def _run_per_tensor_calibrate(
    per_tensor_script: Path,
    layers_path: Path,
    fitness: str,
    work_dir: Path,
    extra_args: list[str],
    model_role: str = "trunk",
) -> Path:
    """Invoke per_tensor_calibrate.py on one component's layers and
    return the per-component policy JSON path. The work_dir holds the
    intermediate policy file (caller owns cleanup).

    ``model_role`` (Phase 16) is forwarded to per_tensor_calibrate.py
    via --model-role so the per-component policy is tagged with the
    role at the policy layer; the calibration_to_tensor_stats.py
    consumer then stamps the same role on the tensor_stats row.
    """
    out_path = work_dir / f"policy.{fitness}.json"
    cmd = [
        sys.executable,
        str(per_tensor_script),
        "--fitness", fitness,
        "--layers", str(layers_path),
        "--output", str(out_path),
        "--model-role", model_role,
        *extra_args,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _tag_policy(
    policy: dict,
    model_role: str,
    roles_in_order: list[str],
) -> dict:
    """Annotate a per-component policy with model_role metadata so
    the downstream quantizer can route per-tensor parameters to the
    right component. Returns a new dict; the input is not mutated.
    """
    tagged = dict(policy)
    tagged["model_role"] = model_role
    # Carry the role list at the top level for the unified writer to
    # enforce the same component set on the read side.
    tagged["model_roles"] = list(roles_in_order)
    # Upgrade the schema marker so the downstream knows this is a
    # unified policy, not a single-component one. The original
    # schema field is preserved (it's the per-component schema).
    tagged.setdefault("schema", SCHEMA)
    tagged["unified_schema"] = UNIFIED_SCHEMA
    families = tagged.get("tensor_families", {})
    if not isinstance(families, dict):
        families = {}
    new_families: dict[str, dict] = {}
    for key, entry in families.items():
        new_entry = dict(entry) if isinstance(entry, dict) else {"value": entry}
        new_entry["model_role"] = model_role
        new_families[key] = new_entry
    tagged["tensor_families"] = new_families
    # Per-fitness ``tensors`` list (LRQ, DartQuant, FLRQ all carry
    # one). Same model_role annotation per record.
    for fitness_key in ("lrq", "dartquant", "flrq"):
        block = tagged.get(fitness_key)
        if isinstance(block, dict):
            tensors = block.get("tensors")
            if isinstance(tensors, list):
                for record in tensors:
                    if isinstance(record, dict):
                        record["model_role"] = model_role
    return tagged


# ---------------------------------------------------------------------------
# M2 follow-up: the embedding-budget producer integration
# ---------------------------------------------------------------------------


def _extract_policy_entries_from_unified(
    unified: dict, tensor_stats_rows: list[dict],
) -> list[dict]:
    """Build the M2 producer's policy_entries from the merged
    unified policy.

    The per_tensor_calibrate output's tensor_families entries
    carry ``model_role`` and the per-fitness metadata, but they
    do NOT carry a direct ``dtype`` field (the qtype is
    implicit in the fitness mode). The M2 producer's
    policy_entries need a ``dtype`` to populate the verdicts
    map; we use the source qtype from ``tensor_stats`` as the
    verdict. This means the M2 producer's S_t(r) sum is
    source-qtype-based (no per-fitness override). The richer
    verdict-aware version requires the per_tensor_calibrate
    output to carry a ``dtype`` field; that is a follow-on
    outside the M2 scope.

    The role "owns" a shared tensor iff the unified policy
    has an entry for ``(role, shared_name)`` in either
    tensor_families or a per-fitness tensors list. With the
    source-qtype-as-verdict approach, the role also owns the
    shared tensor iff tensor_stats has a row for it. We union
    both: every tensor_stats row for a role is a "verdict"
    at the source qtype, so the role owns the shared tensor
    iff the DB has tensor_stats for it.

    This is the "shape-only" integration: the M2 producer
    provides the size-envelope budget based on source
    dtypes. The per-fitness verdict override is a follow-on.

    Args:
      unified: the merged per-component policy dict (output
        of the per-component merge). Currently unused at
        the call site (we drive policy_entries purely from
        tensor_stats); the parameter is kept for the
        follow-on that DOES consume the per-fitness verdicts
        so the call site does not have to change.
      tensor_stats_rows: the tensor_stats rows read from
        the unified DB. Each row is a dict with at least
        ``model_role``, ``name``, ``n_elements``, ``dtype``.

    Returns:
      A list of policy_entries (one per tensor_stats row,
      each carrying the source qtype as the verdict dtype).
    """
    del unified  # currently unused; see the docstring.
    out: list[dict] = []
    for row in tensor_stats_rows:
        role = row.get("model_role")
        name = row.get("name")
        dtype = row.get("dtype")
        if not role or not name or not dtype:
            continue
        out.append({
            "model_role": role,
            "name":       name,
            "dtype":      str(dtype),
        })
    return out


def _load_tensor_stats_from_db(
    db_path: Path,
) -> list[dict]:
    """Read every tensor_stats row from the unified DB.

    The M2 producer needs the source n_elements and dtype
    per (model_role, name); both come from tensor_stats.
    The DB read is independent of any per-component
    calibration — it is the deployment-side measurement of
    what the model is.

    We do NOT filter by model_hash: the per_tensor_calibrate
    flow calibrates one model at a time, but the DB can
    carry multiple models. The producer's role_budgets is
    per-role (not per-model); the per-role shape is the same
    across models, so a multi-model DB just contributes
    extra rows. The per-model filter is a follow-on.

    Args:
      db_path: path to the unified tessera.duckdb.

    Returns:
      A list of dicts with at least ``model_role``, ``name``,
      ``n_elements``, ``dtype``. Empty when tensor_stats is
      missing or empty.
    """
    from tessera_db import TesseraDB  # local import; the
    # tessera_db module is heavy (duckdb) and we do not
    # want every import of unified_calibrate to pay for it.

    try:
        with TesseraDB.open(db_path, read_only=True) as db:
            names = set(db.table_names())
            if "tensor_stats" not in names:
                return []
            df = db.query(
                "SELECT model_role, name, n_elements, dtype "
                "FROM tensor_stats"
            )
    except Exception:
        # Missing DB / IO error / duckdb error -> the
        # integration is a no-op (the CLI gates this with
        # the args.no_embedding_budget / args.db check;
        # this is defense-in-depth so a bad --db path
        # does not crash the calibration driver).
        return []
    return df.to_dicts()


def _build_role_budgets(
    db_path: Path,
    unified: dict,
    budget_fraction: float,
) -> list[dict]:
    """The M2 integration: build the role_budgets sidecar.

    The producer (``embedding_budget.compute_role_budgets``)
    is the upstream of the Phase 16.8 writer's role_budgets
    sidecar key. We read tensor_stats from ``db_path``,
    build policy_entries from the unified policy's per-role
    annotations (with source dtypes as the verdicts), and
    call the producer. The result is flattened to the
    writer's sidecar shape via
    ``embedding_budget.flatten_role_budgets_for_sidecar``.

    The function is a thin glue layer: the M2 producer is
    pure and unit-tested in test_embedding_budget.py. The
    integration is intentionally minimal so the failure
    modes are easy to debug (a typo here breaks the
    integration; a typo in the producer breaks the unit
    tests).

    Args:
      db_path: path to the unified tessera.duckdb.
      unified: the merged per-component policy dict.
      budget_fraction: the role size envelope scaling.

    Returns:
      A list of ``{model_role, budget_bits, weight}`` dicts
      (the writer's sidecar shape). Empty when the producer
      returns no verdicts (e.g. base_budget_fraction <= 0,
      no shared-tensor ownership, or insufficient data).
    """
    # Local import: the producer is optional (the
    # pre-M2 contract is the no-embedding-budget path), and
    # we want the producer to be a leaf module the
    # test_embedding_budget.py tests can drive without
    # pulling in unified_calibrate.
    from embedding_budget import (
        EnvelopeConfig,
        compute_role_budgets,
        flatten_role_budgets_for_sidecar,
    )

    tensor_stats_rows = _load_tensor_stats_from_db(db_path)
    if not tensor_stats_rows:
        return []
    policy_entries = _extract_policy_entries_from_unified(
        unified, tensor_stats_rows,
    )
    cfg = EnvelopeConfig(base_budget_fraction=budget_fraction)
    verdicts = compute_role_budgets(
        policy_entries, tensor_stats_rows, cfg,
    )
    return flatten_role_budgets_for_sidecar(verdicts)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Unified per-component Tessera calibration. Runs "
            "per_tensor_calibrate.py on each --component and "
            "combines the per-component policies into a single "
            "llama.speculative.calibration-policy.v1 JSON keyed by "
            "model_role."
        ),
    )
    p.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "A component to calibrate. ROLE is the model_role "
            "string (e.g. trunk, dspark, dflash, assistant). PATH is "
            "a directory of .npz bundles or a single .npz file "
            "consumable by per_tensor_calibrate.py --layers. May be "
            "specified multiple times; order is preserved in the "
            "unified policy."
        ),
    )
    p.add_argument(
        "--fitness",
        choices=("lrq", "awq", "flrq", "dartquant", "compare"),
        default="lrq",
        help=(
            "Forwarded to per_tensor_calibrate.py when --fitness-default "
            "is not 'auto' (the recommended default). When "
            "--fitness-default is 'auto', this flag is ignored: the "
            "per-component fitness is picked from the ROLE_DEFAULT_FITNESS "
            "table by model_role."
        ),
    )
    p.add_argument(
        "--fitness-default",
        choices=("auto", "lrq", "awq", "flrq", "dartquant", "compare"),
        default="auto",
        help=(
            "How to pick the per-component --fitness. 'auto' (the "
            "recommended default) consults ROLE_DEFAULT_FITNESS: "
            "trunk->awq, dflash/dspark/mtp_nextn->lrq, shared_embd->flrq. "
            "Any other value (lrq, awq, ...) overrides --fitness on every "
            "component (one strategy drives all components)."
        ),
    )
    p.add_argument(
        "--per-tensor-calibrate",
        type=Path,
        default=Path(__file__).parent / "per_tensor_calibrate.py",
        help=(
            "Path to per_tensor_calibrate.py. Defaults to the "
            "sibling script in the same directory as this one."
        ),
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Unified calibration policy JSON output path.",
    )
    p.add_argument(
        "--keep-intermediate",
        action="store_true",
        help=(
            "Keep the per-component intermediate policy files in a "
            "subdirectory next to --output (useful for debugging "
            "which component's calibration contributed which "
            "tensor)."
        ),
    )
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Extra CLI arg forwarded to per_tensor_calibrate.py on "
            "every component. May be specified multiple times."
        ),
    )
    # M2 follow-up: the embedding-budget producer
    # (``tools/tessera/embedding_budget.py``) computes per-role size
    # budgets for the SHARED tensors (token_embd / output) in the
    # now-singular-GGUF's mmproj pipeline. The producer is the
    # upstream of the Phase 16.8 writer's ``role_budgets`` sidecar
    # key. The M0a surface made the mmproj components first-class;
    # M2 gives them a real budget. Three flags gate the integration:
    #
    #   --db PATH: optional path to the unified tessera.duckdb.
    #     When provided, the producer reads tensor_stats from the
    #     DB and uses it as the source for n_elements (the
    #     producer's N(t) denominator) and the per-tensor dtype
    #     (the source qtype for the role's size envelope). Without
    #     --db, the integration is a no-op (the sidecar has no
    #     role_budgets; --budget-fraction is ignored). The unified
    #     Calibrate CLI does not require --db so the existing
    #     per-component flow runs unchanged when the operator has
    #     not yet calibrated tensor_stats.
    #
    #   --budget-fraction F: the role size envelope scaling.
    #     Mirrors l5_retune's --budget-fraction flag (default 1.0).
    #     A non-positive value opts out of the size envelope (the
    #     sidecar has an empty role_budgets).
    #
    #   --no-embedding-budget: opt-out escape hatch. The sidecar
    #     is emitted without the role_budgets key, preserving the
    #     pre-M2 contract (plain worst-of, no per-role budget).
    p.add_argument(
        "--db", type=Path, default=None,
        help=(
            "Optional path to the unified tessera.duckdb. When "
            "provided, the embedding-budget producer (M2) reads "
            "tensor_stats from the DB and emits a role_budgets "
            "sidecar for the writer. Without --db, no role_budgets "
            "is emitted and --budget-fraction is a no-op."
        ),
    )
    p.add_argument(
        "--budget-fraction", type=float, default=1.0,
        help=(
            "Base fraction for the role size envelope: "
            "E(r) = source_footprint_bits(r) * fraction. Mirrors "
            "l5_retune's --budget-fraction flag. Default 1.0; a "
            "non-positive value opts out of the size envelope "
            "(emits an empty role_budgets)."
        ),
    )
    p.add_argument(
        "--no-embedding-budget", action="store_true",
        help=(
            "Opt out of the M2 embedding-budget integration. The "
            "sidecar is emitted without the role_budgets key, "
            "preserving the pre-M2 contract (plain worst-of, no "
            "per-role budget)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    components = _parse_components(args.component)
    roles_in_order = [r for r, _ in components]
    work_dir_path: Path
    cleanup_work: bool
    if args.keep_intermediate:
        work_dir_path = args.output.parent / (args.output.stem + ".intermediate")
        work_dir_path.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    else:
        work_dir_path = Path(tempfile.mkdtemp(prefix="unified-calibrate-"))
        cleanup_work = True
    unified: dict = {
        "schema": SCHEMA,
        "unified_schema": UNIFIED_SCHEMA,
        "model_roles": list(roles_in_order),
        "tensor_families": {},
    }
    # Per-fitness aggregation. We keep a single representative block
    # per fitness mode (lrq / flrq / dartquant); the components'
    # tensors all roll up under the unified tensor_families with
    # their model_role annotation. The per-fitness ``tensors`` list
    # in the unified policy is the concatenation of the per-
    # component tensors, each tagged with model_role.
    per_fitness_tensors: dict[str, list[dict]] = {
        "lrq": [],
        "flrq": [],
        "dartquant": [],
    }
    # Per-component fitness resolution. Each role picks its own
    # --fitness via resolve_fitness(); the resolved values are
    # carried under unified["components"][<role>]["fitness"] so
    # the consumer can audit which strategy drove which component.
    components_meta: dict[str, dict] = {}
    for role, layers_path in components:
        if not args.per_tensor_calibrate.exists():
            print(
                f"error: per_tensor_calibrate.py not found at "
                f"{args.per_tensor_calibrate}",
                file=sys.stderr,
            )
            return 2
        component_fitness = resolve_fitness(
            role, args.fitness, args.fitness_default)
        per_comp = _run_per_tensor_calibrate(
            args.per_tensor_calibrate,
            layers_path,
            component_fitness,
            work_dir_path,
            args.extra_arg,
            model_role=role,
        )
        with per_comp.open() as f:
            policy = json.load(f)
        tagged = _tag_policy(policy, role, roles_in_order)
        # Record the resolved fitness + intermediate path for
        # this component. ``components_meta[role]["fitness"]``
        # is the audit trail the test pins.
        components_meta[role] = {
            "fitness": component_fitness,
            "intermediate": str(per_comp),
        }
        # Merge tensor_families (per-tensor entries, keyed by
        # ``fitness:tensor_name``) into the unified map. Same
        # tensor name across components would collide; the
        # per-tensor names from the drafter heads don't overlap
        # with the trunk's (the drafter has its own ``blk.N.`` set
        # plus dspark-specific tensors), so a flat merge is safe
        # in the canonical case. Component key prefixing is a
        # follow-on if a tensor genuinely overlaps.
        for key, entry in tagged.pop("tensor_families", {}).items():
            if key in unified["tensor_families"]:
                # Disambiguate by role prefix. Unusual path; mostly
                # affects hand-crafted bundles.
                unified["tensor_families"][f"{role}:{key}"] = entry
            else:
                unified["tensor_families"][key] = entry
        # Roll up the per-fitness tensors list.
        for fitness_key in per_fitness_tensors:
            block = tagged.get(fitness_key)
            if isinstance(block, dict):
                tensors = block.get("tensors")
                if isinstance(tensors, list):
                    per_fitness_tensors[fitness_key].extend(tensors)
        # Carry schema and provenance from the first component
        # (they're identical across components; the per-tensor
        # calibration script is deterministic given the same
        # seed).
        if "per_tensor_calibration" in tagged and "per_tensor_calibration" not in unified:
            unified["per_tensor_calibration"] = tagged["per_tensor_calibration"]
        # Carry the wrapper provenance. A future commit can fan
        # this out per-component.
    # Carry the per-component fitness decisions (audit trail).
    unified["components"] = dict(components_meta)
    # Stitch the aggregated per-fitness blocks back into the
    # unified policy. The wrapper ``schema`` stays at the
    # single-component value (the quantizer reads it); the
    # ``unified_schema`` marker tells the writer side to handle
    # the per-tensor model_role.
    #
    # When --fitness-default=auto, components may have picked
    # different fitness strategies (trunk->awq, dflash->lrq,
    # etc.). The legacy single-fitness stitch (one block keyed
    # by args.fitness) only makes sense when every component
    # used the same strategy. In the auto case, the per-role
    # decision is captured under unified["components"]; the
    # per_fitness aggregation already segregated the tensors
    # by their block key (lrq / flrq / dartquant / awq).
    fitness_keys_with_data = [
        key for key, tensors in per_fitness_tensors.items() if tensors
    ]
    # In auto mode, also include the awq block if any component
    # used awq (the per_fitness_tensors dict above only tracks
    # lrq / flrq / dartquant; awq's tensors live in the
    # intermediate policy's top-level search_schema).
    if args.fitness_default == "auto" and any(
            meta["fitness"] == "awq" for meta in components_meta.values()):
        # Locate the awq intermediate by its filename (any
        # component with fitness=awq contributes here).
        awq_inters = [
            Path(meta["intermediate"])
            for meta in components_meta.values()
            if meta["fitness"] == "awq"
        ]
        awq_tensors: list[dict] = []
        for inter in awq_inters:
            if not inter.exists():
                continue
            with inter.open() as f:
                inter_policy = json.load(f)
            awq_tensors.extend(
                entry for entry in inter_policy.get("tensor_families", {}).values()
                if isinstance(entry, dict)
            )
        if awq_tensors:
            # Use the first awq intermediate as the template.
            with awq_inters[0].open() as f:
                template = json.load(f)
            block = {
                "schema": "llama.tessera.awq-policy.v1",
                "search_schema": "llama.tessera.awq-evolution.v1",
                "fitness": "awq",
                "tensor_count": len(awq_tensors),
                "tensors": awq_tensors,
                "model_role": "unified",
            }
            if "evolution" in template:
                block["evolution"] = template["evolution"]
            unified["awq"] = block
    # Single-fitness stitch (legacy + the case where the user
    # overrode --fitness-default to a specific mode). When
    # args.fitness_default is a specific value (e.g. "lrq"),
    # every component ran with that strategy, and we keep the
    # legacy one-block layout.
    if args.fitness_default != "auto" and per_fitness_tensors.get(args.fitness):
        per_comp_first = work_dir_path / f"policy.{args.fitness}.json"
        if per_comp_first.exists():
            with per_comp_first.open() as f:
                first_policy = json.load(f)
            block = dict(first_policy.get(args.fitness, {}))
            block["tensors"] = per_fitness_tensors[args.fitness]
            block["tensor_count"] = len(per_fitness_tensors[args.fitness])
            total_bytes = sum(
                int(record.get("bytes", 0))
                for record in per_fitness_tensors[args.fitness]
                if isinstance(record, dict)
            )
            block["total_bytes"] = total_bytes
            block["model_role"] = "unified"
            unified[args.fitness] = block
    elif args.fitness_default == "auto" and len(fitness_keys_with_data) == 1:
        # Edge case: --fitness-default=auto, but every component
        # happened to pick the same strategy (e.g. only the
        # trunk + dspark where dspark falls back to the legacy
        # default). Stitch the single block.
        only = fitness_keys_with_data[0]
        per_comp_first = work_dir_path / f"policy.{only}.json"
        if per_comp_first.exists():
            with per_comp_first.open() as f:
                first_policy = json.load(f)
            block = dict(first_policy.get(only, {}))
            block["tensors"] = per_fitness_tensors[only]
            block["tensor_count"] = len(per_fitness_tensors[only])
            total_bytes = sum(
                int(record.get("bytes", 0))
                for record in per_fitness_tensors[only]
                if isinstance(record, dict)
            )
            block["total_bytes"] = total_bytes
            block["model_role"] = "unified"
            unified[only] = block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # M2 follow-up: the embedding-budget producer. Wired in here
    # (additive; the existing merge is unchanged). The producer
    # reads tensor_stats from the unified DB (--db) and emits a
    # role_budgets sidecar for the Phase 16.8 writer. The
    # integration is gated by --no-embedding-budget (opt-out) and
    # --db (data source). Without --db, the sidecar has no
    # role_budgets (the pre-M2 contract; --budget-fraction is a
    # no-op in that case).
    if not args.no_embedding_budget and args.db is not None:
        unified["role_budgets"] = _build_role_budgets(
            args.db, unified, args.budget_fraction,
        )
    with args.output.open("w") as f:
        json.dump(unified, f, indent=2, sort_keys=True)
        f.write("\n")
    if cleanup_work:
        import shutil
        shutil.rmtree(work_dir_path, ignore_errors=True)
    fitness_summary = ", ".join(
        f"{role}={meta['fitness']}"
        for role, meta in components_meta.items()
    )
    role_budgets_n = len(unified.get("role_budgets", []))
    print(
        f"wrote unified policy: {args.output} "
        f"({len(unified['tensor_families'])} tensor families, "
        f"{len(roles_in_order)} components: {fitness_summary}, "
        f"{role_budgets_n} role_budgets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
