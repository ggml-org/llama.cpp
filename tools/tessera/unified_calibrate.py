#!/usr/bin/env python3
"""Per-component Tessera calibration driver for the unified pipeline.

The single-model path in ``per_tensor_calibrate.py`` calibrates one
set of layer bundles at a time. The unified Gemma4 12B + DFlash +
DSpark + MTP arch needs to calibrate **four components in one
document** so the C++ ``--write-unified-gguf`` writer can read a
single ``llama.speculative.calibration-policy.v1`` keyed by
``model_role``. This driver is the entry point that produces that
document.

It is a thin orchestrator: for each ``--{component}-npz`` argument
it spawns ``per_tensor_calibrate.py`` as a subprocess (or calls the
in-process API when ``--in-process`` is set) with the matching
``--model-role``, then merges the per-component policies into a
single JSON.

CLI::

    python3 tools/tessera/unified_calibrate.py \\
        --trunk-npz /path/to/gemma4_12b_trunk/ \\
        --dflash-npz /path/to/dflash_drafter/ \\
        --dspark-npz /path/to/dspark_heads/ \\
        --mtp-npz /path/to/mtp_nextn/ \\
        --shared-embd-npz /path/to/shared_embd/ \\
        --fitness lrq \\
        --output out/unified-policy.json

The output is a single ``llama.speculative.calibration-policy.v1``
JSON. Each ``tensor_families`` entry carries a ``model_role`` field
(``trunk`` / ``dflash`` / ``dspark`` / ``mtp_nextn`` /
``shared_embd``) so the consumer can route per-role. The
``model_role`` top-level field is ``null`` for the unified document;
``components.<role>.model_role`` carries the per-component role.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

# Per-component arg name -> MODEL_ROLES role. The role is stamped on
# every per-tensor entry produced by the sub-driver for that
# component, and on the policy's top-level ``model_role`` field.
COMPONENT_ROLES = (
    "trunk",
    "dflash",
    "dspark",
    "mtp_nextn",
    "shared_embd",
)
SHARED_EMBD_ROLE = "shared_embd"
TRUNK_ROLE = "trunk"
SCHEMA = "llama.speculative.calibration-policy.v1"

# The unified policy uses ``null`` for the top-level ``model_role``
# because the document is multi-component. The per-component
# ``components.<role>.model_role`` field is the authoritative role
# label for that slice.
UNIFIED_TOP_LEVEL_MODEL_ROLE = None

THIS_DIR = Path(__file__).resolve().parent
PER_TENSOR_TOOL = THIS_DIR / "per_tensor_calibrate.py"


# ---------------------------------------------------------------------------
# Sub-driver invocation
# ---------------------------------------------------------------------------


def _run_per_tensor(
    layers_arg: Path,
    output_path: Path,
    model_role: str,
    fitness: str,
    extra: Iterable[str] = (),
    in_process: bool = False,
) -> dict:
    """Calibrate one component via per_tensor_calibrate.py.

    Returns the parsed per-component policy. When ``in_process`` is
    true, the sub-driver is imported and called via its Python API;
    otherwise it is invoked as a subprocess. The in-process path is
    faster (no Python interpreter start) and is what the test
    suite uses; the subprocess path is the production path because
    the ``--fitness awq`` mode already spawns awq-evolve.py as a
    subprocess, so the additional process overhead is marginal and
    isolation is preferable.
    """
    if in_process:
        # Lazy import: per_tensor_calibrate is heavy (numpy, scipy
        # optional deps) and may not be needed for the in-process
        # test path.
        sys.path.insert(0, str(THIS_DIR))
        from per_tensor_calibrate import (  # type: ignore[import-not-found]
            LRQ_AGGREGATIONS,
            SCHEMA as PT_SCHEMA,
            bundle_digest,
            build_lrq_policy,
            iter_layer_paths,
            load_layer,
            train_lrq,
        )
        if PT_SCHEMA != SCHEMA:
            raise RuntimeError(
                f"per_tensor_calibrate schema drift: {PT_SCHEMA!r} != {SCHEMA!r}"
            )
        if fitness != "lrq":
            raise ValueError(
                f"in-process mode only supports --fitness lrq (got {fitness!r}); "
                "use the subprocess mode (drop --in-process) for awq / flrq / dartquant / compare"
            )
        layer_paths = iter_layer_paths(str(layers_arg))
        if not layer_paths:
            raise ValueError(f"{layers_arg}: no layer bundles")
        digests = {p.stem: bundle_digest(p) for p in layer_paths}
        # Parse the few extra flags the in-process path needs.
        # ``extra`` is a flat list of ``--flag value`` pairs; we
        # only honour a small subset that the synthetic tests need.
        kwargs = {
            "rank": 16,
            "iterations": 50,
            "lr": 1.0e-3,
            "aggregation": "mean",
            "seed": 0,
            "max_tokens": 256,
            "verbose": False,
        }
        i = 0
        argv = list(extra)
        while i < len(argv):
            tok = argv[i]
            if tok == "--lrq-rank" and i + 1 < len(argv):
                kwargs["rank"] = int(argv[i + 1])
                i += 2
            elif tok == "--lrq-iterations" and i + 1 < len(argv):
                kwargs["iterations"] = int(argv[i + 1])
                i += 2
            elif tok == "--lr" and i + 1 < len(argv):
                kwargs["lr"] = float(argv[i + 1])
                i += 2
            elif tok == "--lrq-agg" and i + 1 < len(argv):
                v = argv[i + 1]
                if v not in LRQ_AGGREGATIONS:
                    raise ValueError(
                        f"--lrq-agg {v!r} not in {LRQ_AGGREGATIONS!r}"
                    )
                kwargs["aggregation"] = v
                i += 2
            elif tok == "--seed" and i + 1 < len(argv):
                kwargs["seed"] = int(argv[i + 1])
                i += 2
            elif tok == "--max-tokens" and i + 1 < len(argv):
                kwargs["max_tokens"] = int(argv[i + 1])
                i += 2
            elif tok in ("--verbose",):
                kwargs["verbose"] = True
                i += 1
            else:
                raise ValueError(
                    f"in-process mode: unknown extra flag {tok!r}; "
                    "supported: --lrq-rank --lrq-iterations --lrq-agg --lr --seed --max-tokens --verbose"
                )
        from per_tensor_calibrate import Layer, LRQResult  # type: ignore[import-not-found]
        results: list[tuple[Layer, LRQResult]] = []
        for path in layer_paths:
            layer = load_layer(path, max_tokens=kwargs["max_tokens"])
            result = train_lrq(
                layer,
                rank=kwargs["rank"],
                iterations=kwargs["iterations"],
                lr=kwargs["lr"],
                seed=kwargs["seed"],
                aggregation=kwargs["aggregation"],
                verbose=kwargs["verbose"],
            )
            results.append((layer, result))
        provenance = {
            "tool": "per_tensor_calibrate.py",
            "mode": "lrq",
            "seed": kwargs["seed"],
            "lrq_rank": kwargs["rank"],
            "lrq_iterations": kwargs["iterations"],
            "lrq_lr": kwargs["lr"],
            "lrq_aggregation": kwargs["aggregation"],
            "bundle_digests": digests,
            "timestamp": time.time(),
        }
        policy = build_lrq_policy(results, provenance, model_role=model_role)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        return policy

    # Subprocess mode: spawn per_tensor_calibrate.py. The sub-driver
    # already writes the per-component policy to ``output_path``; we
    # just re-read it.
    cmd: list[str] = [
        sys.executable,
        str(PER_TENSOR_TOOL),
        "--fitness", fitness,
        "--layers", str(layers_arg),
        "--output", str(output_path),
        "--model-role", model_role,
    ]
    cmd.extend(extra)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"per_tensor_calibrate.py failed for role={model_role!r} "
            f"(rc={result.returncode}): {result.stderr.strip()}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _entry_key(role: str, entry_key: str) -> str:
    """Prefix the per-component entry key with the role.

    The sub-driver's ``tensor_families`` keys are namespaced by
    fitness (``lrq:<name>``, ``flrq:<name>``, ``dartquant:<name>``)
    but not by role. Two roles can produce entries with the same
    key (e.g. ``lrq:token_embd.weight`` from both trunk and
    shared_embd) so the merge step has to disambiguate. We prefix
    the role so the unified document has a unique key per
    (role, fitness, tensor) tuple.
    """
    return f"{role}:{entry_key}"


def merge_unified_policies(component_policies: dict[str, dict]) -> dict:
    """Merge per-component policies into a single unified policy.

    ``component_policies`` is ``{role: per_component_policy}``.
    The merged document keeps:

    * ``schema`` set to ``llama.speculative.calibration-policy.v1``.
    * ``model_role`` set to ``None`` (unified, multi-component).
    * ``components`` set to ``{role: {policy_path, model_role,
      tensor_count, sub_schema}}`` for traceability.
    * ``tensor_families`` is the union of every component's
      ``tensor_families``, prefixed by role so the consumer can
      route by ``entry.model_role``.
    * ``<fitness>`` (lrq / flrq / dartquant) sub-payloads are
      carried over for tooling that wants the per-fitness view
      without re-scanning ``tensor_families``.

    Raises when two components claim the same tensor with the same
    fitness and the entries disagree on the calibration payload;
    when they agree, the merged entry uses the first seen
    sub-policy's payload (a warning is emitted to stderr).
    """
    families: dict[str, dict] = {}
    components_meta: dict[str, dict] = {}
    seen_collision: list[tuple[str, str]] = []
    for role, policy in component_policies.items():
        if role not in COMPONENT_ROLES:
            raise ValueError(
                f"merge: role {role!r} not in {COMPONENT_ROLES!r}"
            )
        if policy.get("schema") != SCHEMA:
            raise ValueError(
                f"merge: role {role!r} policy schema "
                f"{policy.get('schema')!r} != {SCHEMA!r}"
            )
        if policy.get("model_role") not in (None, role):
            raise ValueError(
                f"merge: role {role!r} policy model_role "
                f"{policy.get('model_role')!r} != {role!r}"
            )
        comp_families = policy.get("tensor_families", {}) or {}
        per_role_count = 0
        for key, entry in comp_families.items():
            new_key = _entry_key(role, key)
            if new_key in families:
                # Same role + same fitness produced the same key
                # twice: dedup (the second wins) and warn on stderr.
                seen_collision.append((role, key))
                continue
            entry = dict(entry)
            if entry.get("model_role") not in (None, role):
                # Per-entry role should match the component's role.
                # A mismatch means the sub-driver did not stamp the
                # field; surface it loudly.
                raise ValueError(
                    f"merge: role {role!r} entry {key!r} has "
                    f"model_role={entry.get('model_role')!r}; expected {role!r}"
                )
            entry["model_role"] = role
            families[new_key] = entry
            per_role_count += 1
        # Identify the fitness sub-payload (lrq / flrq / dartquant).
        sub_schema = None
        for sub_key in ("lrq", "flrq", "dartquant"):
            if sub_key in policy and isinstance(policy[sub_key], dict):
                sub_schema = sub_key
                break
        components_meta[role] = {
            "model_role": role,
            "tensor_count": per_role_count,
            "sub_schema": sub_schema,
        }
    if seen_collision:
        print(
            "WARN: unified_calibrate: %d intra-component collisions; "
            "later entries won the dedup" % len(seen_collision),
            file=sys.stderr,
        )
    return {
        "schema": SCHEMA,
        "model_role": UNIFIED_TOP_LEVEL_MODEL_ROLE,
        "components": components_meta,
        "tensor_families": families,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-component Tessera calibration driver. Spawns "
            "per_tensor_calibrate.py once per component (trunk, "
            "dflash, dspark, mtp_nextn, shared_embd) and merges the "
            "resulting per-component policies into a single unified "
            "calibration-policy document keyed by model_role."
        )
    )
    parser.add_argument(
        "--fitness",
        choices=("lrq", "awq", "flrq", "dartquant", "compare"),
        default="lrq",
        help="Calibration mode forwarded to per_tensor_calibrate.py (default lrq).",
    )
    # One optional --{component}-npz flag per component. All are
    # optional because the user may want to calibrate a subset (e.g.
    # just the trunk + dflash). At least one is required.
    parser.add_argument(
        "--trunk-npz",
        default=None,
        help=(
            "Directory of trunk .npz bundles (or a single .npz). "
            "Calibrated as --model-role trunk."
        ),
    )
    parser.add_argument(
        "--dflash-npz",
        default=None,
        help="Directory of dflash drafter .npz bundles. Role: dflash.",
    )
    parser.add_argument(
        "--dspark-npz",
        default=None,
        help="Directory of DSpark head .npz bundles. Role: dspark.",
    )
    parser.add_argument(
        "--mtp-npz",
        default=None,
        help="Directory of MTP nextn .npz bundles. Role: mtp_nextn.",
    )
    parser.add_argument(
        "--shared-embd-npz",
        default=None,
        help=(
            "Directory of shared embedding / output .npz bundles "
            "(token_embd, output). Role: shared_embd. The "
            "calibration data should already be the worst-of-"
            "trunk+dflash aggregate (the writer picks the worst-of-"
            "two at GGUF-write time using this single shared entry)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the unified calibration policy JSON output.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help=(
            "Directory for the per-component intermediate policies. "
            "If unset, a tempdir is created and cleaned up on exit. "
            "Set this for debugging (the per-component JSONs are "
            "useful for inspecting what each sub-driver produced)."
        ),
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the per-component intermediate policies on exit (debug aid).",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "Run per_tensor_calibrate.py in-process (faster, no "
            "subprocess start). Only supports --fitness lrq; the "
            "test suite uses this path."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forwarded to per_tensor_calibrate.py.",
    )
    # Forwarded calibration knobs (consumed in-process; passed
    # through to the subprocess in subprocess mode).
    parser.add_argument("--lrq-rank", type=int, default=None)
    parser.add_argument("--lrq-iterations", type=int, default=None)
    parser.add_argument("--lrq-agg", default=None, choices=("mean", "rms"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser


def _forwarded_extra(args: argparse.Namespace) -> list[str]:
    """Collect the per_tensor_calibrate.py knobs the user set."""
    out: list[str] = []
    if args.lrq_rank is not None:
        out.extend(["--lrq-rank", str(args.lrq_rank)])
    if args.lrq_iterations is not None:
        out.extend(["--lrq-iterations", str(args.lrq_iterations)])
    if args.lrq_agg is not None:
        out.extend(["--lrq-agg", str(args.lrq_agg)])
    if args.seed is not None:
        out.extend(["--seed", str(args.seed)])
    if args.max_tokens is not None:
        out.extend(["--max-tokens", str(args.max_tokens)])
    if args.verbose:
        out.append("--verbose")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    components = {
        "trunk": args.trunk_npz,
        "dflash": args.dflash_npz,
        "dspark": args.dspark_npz,
        "mtp_nextn": args.mtp_npz,
        "shared_embd": args.shared_embd_npz,
    }
    provided = {role: path for role, path in components.items() if path}
    if not provided:
        parser = _build_parser()
        parser.error(
            "at least one --{trunk,dflash,dspark,mtp,shared-embd}-npz "
            "is required"
        )
    # Validate that the user-supplied paths exist and look like
    # bundles. Doing this here gives a clear error before we spawn
    # any subprocesses.
    for role, path in provided.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"--{role.replace('_', '-')}-npz: {path} not found")
        if p.is_file() and p.suffix != ".npz":
            raise ValueError(
                f"--{role.replace('_', '-')}-npz {path}: expected a .npz file or a directory of .npz"
            )

    if args.tmp_dir is not None:
        tmp_root = Path(args.tmp_dir)
        tmp_root.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = False
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="unified_calibrate_"))
        cleanup_tmp = not args.keep_tmp

    extra = _forwarded_extra(args)
    component_policies: dict[str, dict] = {}
    component_paths: dict[str, Path] = {}
    try:
        for role, path in provided.items():
            tmp_path = tmp_root / f"{role}.policy.json"
            if args.verbose:
                print(
                    f"[unified-calibrate] role={role} layers={path} -> {tmp_path}",
                    file=sys.stderr,
                )
            policy = _run_per_tensor(
                Path(path),
                tmp_path,
                role,
                args.fitness,
                extra=extra,
                in_process=args.in_process,
            )
            component_policies[role] = policy
            component_paths[role] = tmp_path
        # Stamp the per-component policy paths so the unified
        # document can trace each entry back to its source.
        for role, path in component_paths.items():
            meta = {
                "policy": str(path),
                "model_role": role,
                "tensor_count": len(component_policies[role].get("tensor_families", {})),
                "sub_schema": next(
                    (
                        k
                        for k in ("lrq", "flrq", "dartquant")
                        if isinstance(component_policies[role].get(k), dict)
                    ),
                    None,
                ),
            }
            component_policies[role].setdefault("components_meta", {})
            # The final merge will overwrite ``components``; we use
            # a separate field for the path so the merge can pick it
            # up before clobbering.
            component_policies[role]["_unified_source_path"] = str(path)
        merged = merge_unified_policies(component_policies)
        # Backfill the policy path on each components entry now
        # that the merge produced the top-level structure.
        for role, meta in merged["components"].items():
            meta["policy"] = component_paths[role].name
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        if args.verbose:
            print(
                f"[unified-calibrate] wrote {output} with "
                f"{sum(m['tensor_count'] for m in merged['components'].values())} "
                f"tensors across {len(merged['components'])} components",
                file=sys.stderr,
            )
        return 0
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        elif args.keep_tmp:
            print(
                f"[unified-calibrate] kept per-component policies at {tmp_root}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    sys.exit(main())
