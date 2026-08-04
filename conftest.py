"""Root conftest for pytest.

Two responsibilities:

1. Register the ``slow`` marker so tests can be filtered with
   ``pytest -m "not slow"`` (and the unified runner's ``--quick``
   flag, which forwards the same filter). Without this registration
   pytest 8+ emits ``PytestUnknownMarkWarning`` and treats the
   marker as an unknown expression element, which fails the run
   under ``--strict-markers``. The runner does not enable
   --strict-markers today, but a downstream CI might.

2. Skip pytest auto-collection of the script-runner test files
   under ``tools/tessera/`` that pre-date the pytest
   auto-discovery convention. These files are runnable as
   standalone scripts (``python3 tools/tessera/test_foo.py``) but
   their ``def test_*(llama_cli: str)`` signatures collide with
   pytest's fixture injection. Excluding them from collection
   keeps both invocation paths working: pytest discovers every
   pytest-native test, and the script-runner tests are still
   available via direct invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make tests/test_*.py and the tessera tools dir importable from
# the repo root without each test file having to mutate sys.path.
REPO_ROOT = Path(__file__).resolve().parent
TOOLS_TESSERA = REPO_ROOT / "tools" / "tessera"
TESTS = REPO_ROOT / "tests"

for p in (REPO_ROOT, TOOLS_TESSERA, TESTS):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def pytest_configure(config):  # noqa: ARG001
    """Register the ``slow`` marker on the pytest config so
    ``-m "not slow"`` works without warnings.
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (round-trip calibration E2Es); "
        "skipped by the unified runner's --quick flag",
    )


# Script-runner test files that are NOT pytest-collectable because
# their test_*() functions take positional parameters that pytest
# would otherwise try to inject as fixtures. They remain runnable
# as standalone scripts: `python3 tools/tessera/test_l2_forward.py`.
# Also: test_test_all_sh.py is excluded because it spawns the
# runner via subprocess; pytest auto-discovering it would create
# a self-reference (the runner invokes pytest, which discovers
# the test, which invokes the runner, ...).
collect_ignore_glob = [
    "tools/tessera/test_l2_forward.py",
    "tests/test_test_all_sh.py",
]
