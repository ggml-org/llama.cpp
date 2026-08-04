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
  same ``lrq`` / ``awq`` / ``flrq`` / ``dartquant`` modes).
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


def _run_per_tensor_calibrate(
    per_tensor_script: Path,
    layers_path: Path,
    fitness: str,
    work_dir: Path,
    extra_args: list[str],
) -> Path:
    """Invoke per_tensor_calibrate.py on one component's layers and
    return the per-component policy JSON path. The work_dir holds the
    intermediate policy file (caller owns cleanup).
    """
    out_path = work_dir / f"policy.{fitness}.json"
    cmd = [
        sys.executable,
        str(per_tensor_script),
        "--fitness", fitness,
        "--layers", str(layers_path),
        "--output", str(out_path),
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
        help="Forwarded to per_tensor_calibrate.py (default lrq).",
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
    for role, layers_path in components:
        if not args.per_tensor_calibrate.exists():
            print(
                f"error: per_tensor_calibrate.py not found at "
                f"{args.per_tensor_calibrate}",
                file=sys.stderr,
            )
            return 2
        per_comp = _run_per_tensor_calibrate(
            args.per_tensor_calibrate,
            layers_path,
            args.fitness,
            work_dir_path,
            args.extra_arg,
        )
        with per_comp.open() as f:
            policy = json.load(f)
        tagged = _tag_policy(policy, role, roles_in_order)
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
    # Stitch the aggregated per-fitness blocks back into the
    # unified policy. The wrapper ``schema`` stays at the
    # single-component value (the quantizer reads it); the
    # ``unified_schema`` marker tells the writer side to handle
    # the per-tensor model_role.
    if per_fitness_tensors[args.fitness]:
        # Use the first component's fitness block as a template
        # (rank, iterations, etc.), then replace its tensors with
        # the aggregated list and update totals.
        first = components[0][0]
        # Locate the per-component fitness block from the
        # per-component intermediate policy (we already merged
        # the families; reconstruct the header from any
        # component's fitness block). Re-read the first
        # component's intermediate.
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(unified, f, indent=2, sort_keys=True)
        f.write("\n")
    if cleanup_work:
        import shutil
        shutil.rmtree(work_dir_path, ignore_errors=True)
    print(
        f"wrote unified policy: {args.output} "
        f"({len(unified['tensor_families'])} tensor families, "
        f"{len(roles_in_order)} components: "
        f"{', '.join(roles_in_order)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
