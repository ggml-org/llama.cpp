"""Conftest for the importer / exporter Python tests.

This file is intentionally empty: the test files use
``sys.path.insert(0, WORKTREE)`` to make the
``tools.tessera.importers`` / ``tools.tessera.exporters``
packages importable. The conftest is here as a pytest
discovery point; pytest's auto-collection treats this
directory as a package.
"""

from __future__ import annotations

# No fixtures yet. The per-file ``sys.path`` insertion is
# the only setup the v1 tests need.
