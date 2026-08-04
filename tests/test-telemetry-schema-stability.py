#!/usr/bin/env python3
"""
Schema stability regression test.

Pins the contract that there is EXACTLY ONE canonical spec-decoding
telemetry schema emitted by this repository: `llama.tessera.spec.v1`.
The previous versioned schemas (llama.dflash.acceptance.v1,
llama.spec_calib.v2, llama.spec_calib.v3) are gone.

The test scans the source tree for any reference to the legacy
schema names and fails if it finds one in a C/C++/Python file outside
the documented allowlist.

Run from the build directory: `ctest -R test-telemetry-schema-stability`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Files we never scan.
EXCLUDE_DIRS = {
    ".git",
    "build",
    "build-release",
    "build-debug",
    "build-agent-g",
    "node_modules",
    "__pycache__",
    # .zcode holds alphaevolve worktrees + integration snapshots whose
    # source predates the spec-consolidate refactor and may still mention
    # the legacy schema names. They are not part of the main source tree
    # so the schema-stability contract doesn't apply to them.
    ".zcode",
    # worktrees/ holds transient agent worktrees (e.g. an active cherry-pick
    # worktree that hasn't been pruned yet). Same logic as .zcode above.
    "worktrees",
}

# The single canonical schema name. Changing this requires updating
# the C++ serializer, the consumers, and this test together.
CANONICAL_SCHEMA = "llama.tessera.spec.v1"

# Legacy schema names that must NOT be emitted by anything in this
# repository. Any reference to one of these in a source file is a
# regression.
LEGACY_SCHEMAS = (
    "llama.dflash.acceptance.v1",
    "llama.spec_calib.v2",
    "llama.spec_calib.v3",
)

# Files where it is OK to mention a legacy schema (e.g. the docs
# that document the historical change, the merge notes, or this test
# file itself).
ALLOWED_EXCEPTIONS = {
    # This test file mentions the legacy names in its own docstring and
    # in the LEGACY_SCHEMAS tuple that drives the source scan.
    "tests/test-telemetry-schema-stability.py",
    # The spec.v1 serializer's no-leakage test intentionally references
    # the legacy schema names so it can assert that the emitted record
    # does not contain them.
    "tests/test-telemetry-record.cpp",
}


def iter_source_files() -> list[Path]:
    """Yield candidate source files under the repo root."""
    out: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.suffix not in (".py", ".cpp", ".h", ".hpp", ".cc", ".cxx",
                                ".md", ".sh", ".txt", ".json", ".yml", ".yaml"):
            continue
        out.append(path)
    return out


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


class SchemaStabilityTest(unittest.TestCase):
    def test_canonical_schema_is_spec_v1(self):
        # The C++ source of truth must declare spec.v1 as the canonical
        # schema name. We allow the constant to live in either the
        # production emission site (common/speculative-calibration.h)
        # or the test serializer (tools/imatrix/telemetry-record.h).
        spec_calib_h = REPO_ROOT / "common" / "speculative-calibration.h"
        text = _read(spec_calib_h)
        self.assertIn(CANONICAL_SCHEMA, text,
                      f"{spec_calib_h} must declare the canonical {CANONICAL_SCHEMA} schema name")

    def test_no_legacy_schema_in_source(self):
        """
        No source file (.cpp/.h/.hpp/.cc/.cxx/.py) may reference a legacy
        schema name. We enforce this aggressively because the whole point
        of the consolidation is that there is exactly one schema.
        """
        offenders: list[tuple[str, str]] = []
        for path in iter_source_files():
            rel = str(path.relative_to(REPO_ROOT))
            if rel in ALLOWED_EXCEPTIONS:
                continue
            if path.suffix not in (".cpp", ".h", ".hpp", ".cc", ".cxx",
                                    ".py"):
                continue
            text = _read(path)
            for legacy in LEGACY_SCHEMAS:
                if legacy in text:
                    offenders.append((rel, legacy))
        if offenders:
            pretty = "\n  - ".join(f"{rel}  ({name})"
                                    for rel, name in sorted(offenders))
            self.fail(
                "The following files still reference a legacy spec-decoding "
                "schema name. The consolidation to "
                f"{CANONICAL_SCHEMA!r} requires removing all references to "
                "the old v1/v2/v3 schemas:\n  - " + pretty
            )

    def test_no_telemetry_v1_compat(self):
        """
        The --telemetry-v1-compat flag and the common_params::telemetry_v1_compat
        field are gone. Any reference to them in a C++ source file is a
        regression.
        """
        offenders: list[str] = []
        for path in iter_source_files():
            rel = str(path.relative_to(REPO_ROOT))
            if rel in ALLOWED_EXCEPTIONS:
                continue
            if path.suffix not in (".cpp", ".h", ".hpp", ".cc", ".cxx"):
                continue
            text = _read(path)
            if "telemetry_v1_compat" in text:
                offenders.append(rel)
        if offenders:
            self.fail(
                "The following files still reference "
                "'telemetry_v1_compat'. The --telemetry-v1-compat flag is "
                "removed; the field is gone:\n  - "
                + "\n  - ".join(sorted(offenders))
            )

    def test_documentation_mentions_canonical(self):
        """
        The main README and the speculative-decoding doc should mention
        the new single schema name.
        """
        for doc_rel in ("README.md", "docs/speculative.md"):
            text = _read(REPO_ROOT / doc_rel)
            self.assertIn(CANONICAL_SCHEMA, text,
                          f"{doc_rel} must mention the canonical {CANONICAL_SCHEMA} schema name")


if __name__ == "__main__":
    unittest.main()
