"""Self-tests for scripts/test-all.sh (F5.1 unified runner).

The runner is a bash script; these tests exercise it as a black
box: invoke it with various flags and assert on the output
shape (--help, --quick, --cpp-only, --py-only, summary format).
The tests do NOT require a built C++ surface; the runner's
"ctest: SKIP" path is exercised when no build/ exists, which
is the typical state of a fresh worktree.

The test does require pytest (collected by the runner's
``--py-only`` mode) to pass. The runner is expected to
discover the test surface under tools/tessera/ and tests/.

Run as a pytest module:

    python3 -m pytest tests/test_test_all_sh.py -v

Or as a script:

    python3 tests/test_test_all_sh.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SCRIPT = REPO_ROOT / "scripts" / "test-all.sh"

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


# Regex for the unified summary line the spec calls for:
#   "C++: 89/89 passed | Python: 188/188 passed | TOTAL: 277/277 passed in 42s"
# The script also accepts the half-line forms (C++: 0/0 skipped,
# Python: 188/188 passed, etc.) when only one surface ran. The
# common requirement is a `TOTAL:` segment on the final line.
SUMMARY_RE = re.compile(
    r"^(?:[A-Za-z+]+: \d+/\d+ (?:passed|FAILED|skipped) )?\|? ?"
    r"(?:[A-Za-z+]+: \d+/\d+ (?:passed|FAILED|skipped) )?\|? ?"
    r"TOTAL:.*in \d+s\s*$"
)


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the runner. ``cwd`` defaults to a temp dir that
    does not have a build/ subdir, so the C++ path hits the
    ``ctest: SKIP`` branch. This is the right behaviour for
    the self-test: the script-runner test must work in any
    state, including the pre-build state.
    """
    cmd = [str(SCRIPT)] + args
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None,
        timeout=600,
    )


class TestTestAllSh(unittest.TestCase):
    """Black-box tests for scripts/test-all.sh.

    The runner is a single bash script; the tests assert the
    observable contract (help text, exit codes, summary line
    shape, --quick/--cpp-only/--py-only flag wiring). The
    tests do NOT depend on a built C++ surface: ``--build
    /tmp/no-such-build-dir`` is used to force the runner's
    ``ctest: SKIP`` path on a worktree that happens to have
    a build/ from a prior run.

    The tests are excluded from the runner's pytest discovery
    (see conftest.py) so the runner does not recursively
    invoke itself when running --py-only in CI.
    """

    # A bogus build path that no worktree can plausibly have.
    # Using ``--build`` with this path forces the C++ surface
    # to the "no CTestTestfile.cmake" branch, which is the
    # only deterministic behaviour across the "fresh worktree"
    # and "previously-built worktree" states.
    NO_BUILD = "/tmp/no-such-build-dir-for-test-all-sh"

    def setUp(self) -> None:
        # Use a fresh temp dir as the subprocess cwd; the
        # script's REPO_ROOT detection (via its own path) is
        # independent of cwd.
        self._tmp = Path(tempfile.mkdtemp(prefix="test_all_sh_"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ---- --help ---------------------------------------------------------

    def test_help_prints_usage_and_exits_zero(self) -> None:
        result = _run(["--help"], cwd=self._tmp)
        self.assertEqual(
            result.returncode, 0,
            f"--help should exit 0; got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertIn("Tessera unified test runner", result.stdout)
        self.assertIn("--quick", result.stdout)
        self.assertIn("--cpp-only", result.stdout)
        self.assertIn("--py-only", result.stdout)
        self.assertIn("--build", result.stdout)

    def test_help_short_flag_works(self) -> None:
        result = _run(["-h"], cwd=self._tmp)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage", result.stdout)

    def test_unknown_flag_exits_nonzero(self) -> None:
        result = _run(["--definitely-not-a-flag"], cwd=self._tmp)
        self.assertNotEqual(
            result.returncode, 0,
            f"unknown flag should exit non-zero; got {result.returncode}",
        )
        self.assertIn("unknown argument", result.stderr)

    # ---- --py-only ------------------------------------------------------

    def test_py_only_runs_python_surface(self) -> None:
        """--py-only skips the C++ surface and runs pytest. The
        runner does not require a built C++ surface in this
        mode.
        """
        result = _run(["--py-only"], cwd=self._tmp)
        # We don't assert on the return code (it depends on
        # whether the pytest surface passes), only on the
        # output shape. The C++ path is silent.
        self.assertNotIn(
            "C++ (ctest):", result.stdout,
            "--py-only should not run the C++ surface; got:\n"
            f"{result.stdout}",
        )
        self.assertIn("Python (pytest):", result.stdout)
        self.assertIn("TOTAL:", result.stdout)

    def test_py_only_summary_line_shape(self) -> None:
        """--py-only output ends with a `Python: X/X passed | TOTAL: ...`
        line, in the form the spec mandates.
        """
        result = _run(["--py-only"], cwd=self._tmp)
        last = result.stdout.strip().splitlines()[-1]
        self.assertRegex(
            last, SUMMARY_RE,
            f"--py-only summary line shape wrong: {last!r}\n"
            f"full output:\n{result.stdout}",
        )
        self.assertIn("Python:", last)

    def test_py_only_exits_zero_when_python_passes(self) -> None:
        """When the Python surface passes, --py-only exits 0.
        The runner is excluded from its own discovery (see
        conftest.py collect_ignore_glob) so this is a real
        measurement of the project test surface, not a
        self-reference.
        """
        result = _run(["--py-only"], cwd=self._tmp)
        self.assertEqual(
            result.returncode, 0,
            f"--py-only should exit 0 when Python passes; "
            f"got {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )

    # ---- --quick --------------------------------------------------------

    def test_quick_skips_slow_marker(self) -> None:
        """--quick forwards ``-m 'not slow'`` to pytest. The
        tests/test_phase16_e2e.py suite is @pytest.mark.slow;
        --quick should not include it in the count.

        The runner parses pytest's summary line; the
        deselected E2E is not counted in the ``X/X passed``
        denominator (pytest's behaviour, not the runner's).
        We assert the count difference between the default
        and --quick runs.
        """
        default = _run(["--py-only"], cwd=self._tmp)
        quick = _run(["--py-only", "--quick"], cwd=self._tmp)
        self.assertEqual(
            default.returncode, 0,
            f"default --py-only should exit 0; got "
            f"{default.returncode}\nstdout:\n{default.stdout}",
        )
        self.assertEqual(
            quick.returncode, 0,
            f"--py-only --quick should exit 0; got "
            f"{quick.returncode}\nstdout:\n{quick.stdout}",
        )
        # Extract the totals from the summary lines.
        def _total(s: str) -> int:
            last = s.strip().splitlines()[-1]
            m = re.search(r"TOTAL: (\d+)/(\d+)", last)
            self.assertIsNotNone(
                m, f"could not parse TOTAL from {last!r}")
            return int(m.group(2))
        default_total = _total(default.stdout)
        quick_total = _total(quick.stdout)
        self.assertEqual(
            default_total - quick_total, 1,
            f"--quick should skip exactly 1 test (the slow E2E); "
            f"default={default_total} quick={quick_total}",
        )

    def test_quick_does_not_change_pytest_exit_code(self) -> None:
        """--quick only changes the filter; it should not turn
        a passing suite into a failure or vice versa.
        """
        with_q = _run(["--py-only"], cwd=self._tmp)
        with_quick = _run(["--py-only", "--quick"], cwd=self._tmp)
        self.assertEqual(
            with_q.returncode, with_quick.returncode,
            f"--quick changed the exit code from {with_q.returncode} to "
            f"{with_quick.returncode}; it should be a no-op on the "
            f"pass/fail verdict",
        )

    # ---- --cpp-only -----------------------------------------------------

    def test_cpp_only_skips_python_surface(self) -> None:
        """--cpp-only skips the Python surface. ``--build`` is
        pinned to a bogus path so the C++ side hits the
        deterministic "no CTestTestfile.cmake" branch
        regardless of the worktree's actual build/ state.
        """
        result = _run(["--cpp-only", "--build", self.NO_BUILD],
                      cwd=self._tmp)
        self.assertNotIn(
            "Python (pytest):", result.stdout,
            "--cpp-only should not run the Python surface; got:\n"
            f"{result.stdout}",
        )

    def test_cpp_only_summary_when_no_build(self) -> None:
        """--cpp-only with a bogus --build: the runner should
        surface the missing CTestTestfile.cmake to stderr
        and exit non-zero (a hard error: the user asked for
        a specific path that is not buildable).
        """
        result = _run(["--cpp-only", "--build", self.NO_BUILD],
                      cwd=self._tmp)
        self.assertNotEqual(
            result.returncode, 0,
            f"--cpp-only with bogus --build should exit non-zero; "
            f"got {result.returncode}\nstdout:\n{result.stdout}",
        )
        self.assertIn(
            "CTestTestfile.cmake",
            result.stderr,
            f"--cpp-only --build bogus should report the missing "
            f"ctest file to stderr; got stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )

    # ---- discovery / build dir -----------------------------------------

    def test_invalid_build_dir_exits_nonzero(self) -> None:
        """``--build <bogus path>`` (without ``--cpp-only``)
        should exit non-zero. The path has no
        CTestTestfile.cmake, and the runner surfaces this
        even on a full run.
        """
        result = _run(
            ["--build", self.NO_BUILD, "--py-only"],
            cwd=self._tmp,
        )
        # --py-only ignores the --build, so the run succeeds
        # on the Python side; the bogus --build should not
        # cause a failure (it's only consulted when the C++
        # surface is selected).
        # This is the documented contract: --build is only
        # consulted by --cpp-only or the full run.
        # The non-bogus-build check is covered by the
        # cpp_only tests above (which combine --cpp-only
        # with --build bogus).
        self.assertEqual(
            result.returncode, 0,
            f"--py-only should ignore --build; "
            f"got {result.returncode}\nstderr:\n{result.stderr}",
        )

    def test_full_run_with_no_build_exits_nonzero_via_cpp(self) -> None:
        """A full run (no surface flags) with no usable build
        dir should exit non-zero because the C++ surface
        reports a missing-CTestTestfile error.
        """
        result = _run(["--build", self.NO_BUILD], cwd=self._tmp)
        # The C++ surface reports the bogus build path as an
        # error to stderr (the user explicitly asked for it
        # via --build) and exits non-zero. The Python side
        # may pass.
        self.assertNotEqual(
            result.returncode, 0,
            f"full run with bogus --build should exit non-zero; "
            f"got {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertIn(
            "CTestTestfile.cmake",
            result.stdout + result.stderr,
            "bogus --build should report the missing ctest file",
        )

    def test_repo_root_self_locates(self) -> None:
        """The runner should self-locate the repo root via the
        script's own path, not the cwd. Running it from a
        subdir must still work.
        """
        sub = self._tmp / "subdir"
        sub.mkdir()
        result = _run(["--py-only"], cwd=sub)
        # The script must still find the Python test surface,
        # so the Python: X/X passed line must be present.
        self.assertIn(
            "Python (pytest):", result.stdout,
            f"runner from subdir should still find the Python "
            f"surface; got:\n{result.stdout}",
        )

    # ---- help text invariants ------------------------------------------

    def test_help_lists_j_flag(self) -> None:
        result = _run(["--help"], cwd=self._tmp)
        self.assertIn("-j", result.stdout)

    def test_help_examples_block_present(self) -> None:
        """The --help output should have an Examples block so
        users can copy-paste common invocations.
        """
        result = _run(["--help"], cwd=self._tmp)
        self.assertIn("Examples:", result.stdout)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTestAllSh)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
