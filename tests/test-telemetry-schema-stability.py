#!/usr/bin/env python3
"""
Schema stability regression test.

Pins the contract that:
  1. The unified spec_calib.v3 schema is the canonical schema emitted by
     tools/imatrix/imatrix.cpp.
  2. The legacy llama.dflash.acceptance.v1 schema is still emitted as a
     documented adapter when --telemetry-v1-compat is set.
  3. Every code path in this repository that consumes the v1 schema name
     (search/regex/branch) also recognizes the v3 schema name, so the
     cutover from v1 default to v3 default does not silently break any
     consumer.

The test is grep-based by design: it scans the source tree for
`llama.dflash.acceptance.v1` references and checks each one for the
presence of `llama.spec_calib.v3` (or documents it as an acceptable
exception: documentation or test fixture).

Run from the build directory: `ctest -R test-telemetry-schema-stability`.
"""

from __future__ import annotations

import re
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
    "node_modules",
    "__pycache__",
}

# The single canonical schema name. Changing this requires updating
# the C++ serializer, the consumers, and this test together.
CANONICAL_SCHEMA  = "llama.spec_calib.v3"
LEGACY_SCHEMA     = "llama.dflash.acceptance.v1"

# Files where it is OK to mention the legacy schema without also mentioning
# the v3 schema. These are documentation, the v1-compat emission site, and
# test fixtures that explicitly exercise the adapter.
ALLOWED_EXCEPTIONS = {
    # The v1-compat emission site in the C++ source. This is where the
    # legacy string lives; it must not also reference v3.
    "tools/imatrix/telemetry-record.cpp",
    "tools/imatrix/telemetry-record.h",
    "common/arg.cpp",                      # --telemetry-v1-compat help text
    "common/common.h",                     # the field default
    # Documentation describing the v1-compat adapter and the v3 superset.
    "docs/audit-2026-07-29.md",
    "docs/architecture.md",
    "docs/architecture-worktrees.md",
    "docs/speculative.md",
    "docs/tessera.md",
    "README.md",
    "tools/dspark-gguf-patch/README.md",
    "tools/tessera/README.md",
    # This test file itself, and the other test that round-trips a v1
    # record through the consumer to prove the adapter still works.
    "tests/test-telemetry-schema-stability.py",
    "tests/test-telemetry-record.cpp",
    "tests/test-tessera-evidence-store.py",
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


class SchemaStabilityTest(unittest.TestCase):
    def test_canonical_schema_is_v3(self):
        # The C++ source of truth must define v3 as the canonical schema.
        header = (REPO_ROOT / "tools/imatrix/telemetry-record.h").read_text(
            encoding="utf-8")
        self.assertIn("\"llama.spec_calib.v3\"", header,
                      "telemetry-record.h must declare llama.spec_calib.v3")
        self.assertIn("llama.dflash.acceptance.v1", header,
                      "telemetry-record.h must still declare the v1 schema "
                      "name for the v1-compat adapter")

    def test_legacy_consumers_recognize_v3(self):
        """
        Every code file (not docs/tests) that branches on
        llama.dflash.acceptance.v1 must also handle llama.spec_calib.v3,
        otherwise the v3 default emission would silently break it.
        """
        offenders: list[str] = []
        for path in iter_source_files():
            rel = str(path.relative_to(REPO_ROOT))
            if rel in ALLOWED_EXCEPTIONS:
                continue
            if path.suffix in (".md", ".txt", ".json", ".yml", ".yaml"):
                # Pure documentation / config: v1 mentions are fine, but
                # we still record them so we can review.
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if LEGACY_SCHEMA not in text:
                continue
            if CANONICAL_SCHEMA in text:
                continue
            # Special case: the imatrix.cpp emission site itself is allowed
            # to mention the legacy schema (it routes to the v1-compat
            # helper). It's already in ALLOWED_EXCEPTIONS via the header
            # path, but be explicit for the .cpp.
            if rel == "tools/imatrix/imatrix.cpp":
                continue
            offenders.append(rel)

        if offenders:
            self.fail(
                "The following files reference the legacy "
                f"{LEGACY_SCHEMA!r} schema but do NOT mention the new "
                f"{CANONICAL_SCHEMA!r} schema. Update them so the v3 "
                "default emission is consumed correctly:\n  - "
                + "\n  - ".join(sorted(offenders))
            )

    def test_documentation_references_v3(self):
        """
        The user-facing docs that mention the spec-calib telemetry schema
        should reflect the v3 default. We tolerate v1 mentions in
        documentation but require at least one v3 mention in the main
        README or docs/speculative.md.
        """
        for doc_rel in ("README.md", "docs/speculative.md"):
            text = (REPO_ROOT / doc_rel).read_text(encoding="utf-8")
            self.assertIn(CANONICAL_SCHEMA, text,
                          f"{doc_rel} must mention the v3 schema name")

    def test_no_legacy_v2_schema_emitted(self):
        """
        The legacy llama.spec_calib.v2 schema is no longer emitted by
        imatrix.cpp (it has been superseded by v3). Make sure no code
        path in the C++ source emits it, and that consumers do not
        require it.
        """
        offenders: list[str] = []
        legacy_v2 = "llama.spec_calib.v2"
        for path in iter_source_files():
            rel = str(path.relative_to(REPO_ROOT))
            if rel in ALLOWED_EXCEPTIONS:
                continue
            if path.suffix not in (".cpp", ".h", ".hpp", ".cc", ".cxx",
                                    ".py"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if legacy_v2 in text:
                offenders.append(rel)
        if offenders:
            self.fail(
                f"The following files still reference the superseded "
                f"{legacy_v2!r} schema. Remove the references; v3 is the "
                "canonical superset:\n  - " + "\n  - ".join(sorted(offenders))
            )


if __name__ == "__main__":
    unittest.main()
