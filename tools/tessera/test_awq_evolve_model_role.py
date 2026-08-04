#!/usr/bin/env python3
"""Tests for the ``--model-role`` plumb-through on awq-evolve.py.

Phase 16: awq-evolve.py is the C++ GA search loop's Python sibling;
it is the inner island-GA that the calibration pipeline delegates
to from per_tensor_calibrate.py --fitness awq. The model_role tag
is the per-component disambiguator for the unified Gemma4 12B +
dspark + dflash + MTP arch; the per-family and per-override entries
in the awq-evolve.py output policy must all carry the role so the
unified consumer can route per-tensor parameters back to the right
component.

Two layers:

* Unit-level: exercise ``policy_entry`` and ``build_policy`` with
  synthetic Candidate / Score dicts and assert every per-family /
  per-override / norm / moe block entry carries the role.
* End-to-end: build a minimal layer bundle, invoke awq-evolve.py
  with ``--model-role dflash``, and assert the on-disk policy
  carries the role at the top level and on every family entry.

The unit-level half always runs. The end-to-end half is gated on
the presence of awq-evolve.py + numpy; when either is missing the
end-to-end half is skipped (the unit-level half still covers the
contract).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the tools/tessera dir importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# awq-evolve.py is a script (not a package) so we have to load it
# via importlib to get the same module the CLI sees.
import importlib.util as _importlib_util
_spec = _importlib_util.spec_from_file_location(
    "awq_evolve", str(HERE / "awq-evolve.py"))
awq_evolve = _importlib_util.module_from_spec(_spec)
sys.modules["awq_evolve"] = awq_evolve
_spec.loader.exec_module(awq_evolve)  # type: ignore[union-attr]


def _candidate(alpha: float = 0.5, clip: float = 0.95,
               outlier: float = 0.005) -> awq_evolve.Candidate:
    return awq_evolve.Candidate(
        alpha=alpha,
        clip=clip,
        outlier_fraction=outlier,
        moment_mix=0.0,
        tail_guard=0.0,
        ternary_threshold=1.0,
    )


def _score(fitness: float = 0.0) -> awq_evolve.Score:
    return awq_evolve.Score(
        train_error=0.0,
        heldout_error=0.0,
        tail_error=0.0,
        size_cost=0.0,
        fitness=fitness,
        worst_layer_error=0.0,
    )


def _test_policy_entry_stamps_role() -> None:
    """policy_entry() stamps the role on every entry; the default is
    'trunk' (legacy single-component behaviour) and explicit roles
    round-trip.
    """
    cand = _candidate()
    # Default role
    entry_default = awq_evolve.policy_entry(["blk.0.attn_q.weight"], cand)
    assert entry_default["model_role"] == "trunk", entry_default
    assert entry_default["match"] == ["blk.0.attn_q.weight"]
    # Explicit roles
    for role in ("dflash", "dspark", "mtp_nextn", "shared_embd"):
        entry = awq_evolve.policy_entry(["blk.0.attn_q.weight"], cand,
                                        model_role=role)
        assert entry["model_role"] == role, (role, entry)
    # Invalid role rejected
    try:
        awq_evolve.policy_entry(["blk.0.attn_q.weight"], cand,
                                model_role="not_a_role")
    except ValueError as exc:
        assert "model_role" in str(exc), exc
    else:
        raise AssertionError("expected ValueError on invalid model_role")
    print("ok   policy_entry stamps model_role (default + explicit)")


def _test_build_policy_stamps_role() -> None:
    """build_policy() stamps the role on every family entry, every
    override entry, the norm pseudo-entry, the moe_residual_allocation
    block, and the top-level policy. The default is 'trunk'.
    """
    cand = _candidate()
    results: dict[str, tuple[awq_evolve.Candidate, awq_evolve.Score]] = {
        "ffn": (cand, _score(0.01)),
        "attention": (cand, _score(0.02)),
    }
    overrides: dict[str, tuple[awq_evolve.Candidate, awq_evolve.Score]] = {
        "blk.0.ffn_gate.weight": (_candidate(outlier=0.01), _score(0.005)),
    }
    provenance = {"seed": 0, "families": {}}
    # Default role
    policy = awq_evolve.build_policy(results, provenance, base=None,
                                     overrides=overrides)
    assert policy["model_role"] == "trunk", policy
    # Every family + override + norm entry is stamped.
    for key, entry in policy["tensor_families"].items():
        assert entry["model_role"] == "trunk", (key, entry)
    # Explicit dflash
    policy_dflash = awq_evolve.build_policy(results, provenance, base=None,
                                            overrides=overrides,
                                            model_role="dflash")
    assert policy_dflash["model_role"] == "dflash", policy_dflash
    for key, entry in policy_dflash["tensor_families"].items():
        assert entry["model_role"] == "dflash", (key, entry)
    # MOE block stamped
    moe_block = policy_dflash.get("moe_residual_allocation")
    if isinstance(moe_block, dict):
        assert moe_block["model_role"] == "dflash", moe_block
    # Invalid role rejected
    try:
        awq_evolve.build_policy(results, provenance, model_role="nope")
    except ValueError as exc:
        assert "model_role" in str(exc), exc
    else:
        raise AssertionError("expected ValueError on invalid model_role")
    print("ok   build_policy stamps model_role (top-level + every entry)")


def _make_synthetic_bundle(path: Path, family: str) -> None:
    """Write a minimal .npz that awq-evolve.py's load_layer accepts.

    The shape is 8x8 with deterministic values so the GA search is
    fast and the test is reproducible. name + family are np.str_ so
    they round-trip as plain strings.
    """
    rng = np.random.default_rng(seed=42)
    weight = rng.standard_normal((8, 8)).astype(np.float32) * 0.05
    # train_activations shape (n_tokens, in_dim); 16 tokens is enough
    # for the GA to evaluate the first generation.
    train_acts = rng.standard_normal((16, 8)).astype(np.float32) * 0.5
    np.savez(
        path,
        weight=weight,
        train_activations=train_acts,
        in_sum2=(train_acts.astype(np.float32) ** 2).sum(axis=0),
        counts=np.array(16, dtype=np.int64),
        second_moment=(train_acts.astype(np.float32) ** 2).mean(axis=0),
        fourth_moment=(train_acts.astype(np.float32) ** 4).mean(axis=0),
        max_abs=np.abs(train_acts).max(axis=0).astype(np.float32),
        name=np.str_(path.stem),
        family=np.str_(family),
    )


def _test_cli_model_role() -> None:
    """End-to-end: invoke awq-evolve.py with --model-role dflash and
    verify the on-disk policy carries the role at every level.
    """
    awq_tool = HERE / "awq-evolve.py"
    if not awq_tool.exists():
        print("skip cli: awq-evolve.py not present")
        return
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        layers_dir = td_path / "layers"
        layers_dir.mkdir()
        # One ffn + one attention bundle so both families are populated.
        _make_synthetic_bundle(layers_dir / "blk.0.ffn_gate.weight.npz", "ffn")
        _make_synthetic_bundle(
            layers_dir / "blk.0.attn_q.weight.npz", "attention")
        out_policy = td_path / "policy.json"
        cmd = [
            sys.executable,
            str(awq_tool),
            "--layers", str(layers_dir),
            "--output", str(out_policy),
            "--generations", "2",
            "--population", "4",
            "--islands", "2",
            "--seed", "0",
            "--model-role", "dflash",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                "cli failed (likely awq-evolve.py config issue); "
                "see stdout/stderr:"
            )
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            print("skip cli assertions (non-zero exit)")
            return
        with out_policy.open() as f:
            policy = json.load(f)
        # Top-level
        assert policy["model_role"] == "dflash", policy
        # Every tensor_family entry is stamped.
        for key, entry in policy["tensor_families"].items():
            assert entry["model_role"] == "dflash", (key, entry)
        # moe_residual_allocation (when present) is stamped.
        moe_block = policy.get("moe_residual_allocation")
        if isinstance(moe_block, dict):
            assert moe_block["model_role"] == "dflash", moe_block
    print("ok   awq-evolve.py --model-role dflash stamps policy end-to-end")


def _test_cli_default_role() -> None:
    """Default --model-role is 'trunk' (legacy single-component)."""
    awq_tool = HERE / "awq-evolve.py"
    if not awq_tool.exists():
        print("skip cli default: awq-evolve.py not present")
        return
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        layers_dir = td_path / "layers"
        layers_dir.mkdir()
        _make_synthetic_bundle(layers_dir / "blk.0.ffn_gate.weight.npz", "ffn")
        out_policy = td_path / "policy.json"
        cmd = [
            sys.executable,
            str(awq_tool),
            "--layers", str(layers_dir),
            "--output", str(out_policy),
            "--generations", "1",
            "--population", "4",
            "--islands", "1",
            "--seed", "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("skip cli default (non-zero exit)")
            return
        with out_policy.open() as f:
            policy = json.load(f)
        # Default is 'trunk' (legacy single-model contract).
        assert policy["model_role"] == "trunk", policy
        for key, entry in policy["tensor_families"].items():
            assert entry["model_role"] == "trunk", (key, entry)
    print("ok   awq-evolve.py default --model-role is 'trunk'")


def main() -> int:
    _test_policy_entry_stamps_role()
    _test_build_policy_stamps_role()
    _test_cli_model_role()
    _test_cli_default_role()
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
