#!/bin/bash
# Tessera unified test runner (F5.1).
#
# Runs the C++ ctest surface (one of the build/ build-ane/ build-g0/
# build-st/ directories, picked by recency of the ctest artifact) and
# the Python pytest surface (test_*.py under tools/tessera/ and
# tests/), then prints a single unified summary.
#
# Stdlib only: bash + standard unix tools (ctest, pytest, sysctl,
# awk, grep, date, etc.). No new deps.
#
# Usage:
#   scripts/test-all.sh                # full run
#   scripts/test-all.sh --quick        # skip @pytest.mark.slow
#   scripts/test-all.sh --cpp-only     # only the C++ ctest surface
#   scripts/test-all.sh --py-only      # only the Python pytest surface
#   scripts/test-all.sh --help
#
# Exit status:
#   0   all selected surfaces passed
#   1   at least one surface failed
#   2   usage / configuration error (no build dir, missing pytest, ...)
#
# Run from the repository root. The script is safe to invoke from
# anywhere as long as the cwd is a worktree of tessera; it
# self-locates the repo root by resolving its own symlink target.

set -u
set -o pipefail

# --- self-locate repo root ------------------------------------------------
SELF="${BASH_SOURCE[0]}"
# Follow the symlink (if any) so a worktree-invoked script still
# resolves to the worktree's scripts/test-all.sh.
while [ -L "$SELF" ]; do
    TARGET=$(readlink "$SELF")
    if [[ "$TARGET" = /* ]]; then
        SELF="$TARGET"
    else
        SELF="$(dirname "$SELF")/$TARGET"
    fi
done
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- arg parsing ----------------------------------------------------------
QUICK=0
CPP_ONLY=0
PY_ONLY=0
PYTEST_OPTS=("-x" "-q")
CTEST_OPTS=("--output-on-failure")
JOBS=""
BUILD_DIR_OVERRIDE=""

usage() {
    cat <<'EOF'
Tessera unified test runner (scripts/test-all.sh)

Usage:
  scripts/test-all.sh [OPTIONS]

Options:
  --quick           Skip tests marked @pytest.mark.slow (calibration
                    E2Es and the Phase 16 round-trip).
  --cpp-only        Only run the C++ ctest surface.
  --py-only         Only run the Python pytest surface.
  --build DIR       Use DIR as the C++ build directory (skip discovery).
  -j N              Pass -jN to ctest (default: sysctl -n hw.ncpu).
  --help, -h        Print this help and exit.

Examples:
  scripts/test-all.sh
  scripts/test-all.sh --quick
  scripts/test-all.sh --cpp-only -j 4
  scripts/test-all.sh --py-only
  scripts/test-all.sh --build build-ane

Exits non-zero if any selected surface has a failure.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)        QUICK=1 ;;
        --cpp-only)     CPP_ONLY=1 ;;
        --py-only)      PY_ONLY=1 ;;
        --build)
            [ $# -ge 2 ] || { echo "ERROR: --build requires a path" >&2; exit 2; }
            BUILD_DIR_OVERRIDE="$2"
            shift
            ;;
        -j)
            [ $# -ge 2 ] || { echo "ERROR: -j requires N" >&2; exit 2; }
            JOBS="$2"
            shift
            ;;
        -j*)
            JOBS="${1#-j}"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

# --- jobs -----------------------------------------------------------------
if [ -z "$JOBS" ]; then
    if JOBS=$(sysctl -n hw.ncpu 2>/dev/null); then
        :
    elif JOBS=$(nproc 2>/dev/null); then
        :
    else
        JOBS=2
    fi
fi
# Clamp to >=1.
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
    JOBS=1
fi

# --- build dir discovery --------------------------------------------------
# Priority order is the same as the task spec: build/ > build-ane/ >
# build-g0/ > build-st/. The active build is the one with the most
# recent ctest artifact (CTestTestfile.cmake + LastTest.log mtime).
discover_build_dir() {
    if [ -n "$BUILD_DIR_OVERRIDE" ]; then
        if [ -f "$BUILD_DIR_OVERRIDE/CTestTestfile.cmake" ]; then
            cd "$REPO_ROOT"
            echo "$BUILD_DIR_OVERRIDE"
            return 0
        fi
        # The user explicitly asked for a specific build dir
        # but it has no CTestTestfile.cmake. This is a hard
        # error: silently falling back to discovery would
        # hide a typo or a stale path. The Python path can
        # still run, but the C++ surface returns a non-zero
        # code so the overall exit status reflects the
        # problem.
        echo "ERROR: --build $BUILD_DIR_OVERRIDE has no CTestTestfile.cmake" >&2
        return 2
    fi
    local candidates=("build" "build-ane" "build-g0" "build-st")
    local best=""
    local best_mtime=0
    local d
    for d in "${candidates[@]}"; do
        local p="$REPO_ROOT/$d"
        if [ ! -f "$p/CTestTestfile.cmake" ]; then
            continue
        fi
        # Mtime of either the CTestTestfile or the last ctest log;
        # whichever is newer captures "the most recent run".
        # Use `date -r FILE +%s` for portability: BSD date (macOS)
        # and GNU date (Linux) both support -r; the stat -f/-c
        # dialects differ and the system `stat` is often GNU
        # coreutils on macOS (Homebrew shadowing the BSD one).
        local m=0
        m=$(date -r "$p/CTestTestfile.cmake" +%s 2>/dev/null \
            || echo 0)
        if [ -f "$p/Testing/Temporary/LastTest.log" ]; then
            local lm=0
            lm=$(date -r "$p/Testing/Temporary/LastTest.log" +%s 2>/dev/null \
                || echo 0)
            if [ "$lm" -gt "$m" ]; then
                m="$lm"
            fi
        fi
        if [ "$m" -gt "$best_mtime" ]; then
            best_mtime="$m"
            best="$d"
        fi
    done
    cd "$REPO_ROOT"
    if [ -z "$best" ]; then
        return 1
    fi
    echo "$best"
}

# --- ctest surface --------------------------------------------------------
run_cpp_tests() {
    local build_dir
    if ! build_dir=$(discover_build_dir); then
        # The user explicitly asked for a specific build dir
        # via --build but the path is not buildable. This is
        # a hard error: surface it as a non-zero exit from
        # this function so the overall runner exits non-zero.
        if [ -n "$BUILD_DIR_OVERRIDE" ]; then
            echo "  ctest: ERROR (--build $BUILD_DIR_OVERRIDE not usable)"
            cpp_rc=2
            return 2
        fi
        echo "  ctest: SKIP (no build dir with CTestTestfile.cmake; run cmake -B build first)"
        echo "  C++: skipped"
        return 0
    fi
    echo "  build: $build_dir"
    local start
    start=$(date +%s)
    local ctest_log="$REPO_ROOT/.test-all-ctest.log"
    # --no-tests=ignore so a build with no tests does not error.
    (cd "$build_dir" && ctest "${CTEST_OPTS[@]}" -j "$JOBS" --no-tests=ignore) \
        > "$ctest_log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ "$rc" -ne 0 ]; then
        echo "  ctest: FAIL (see $ctest_log)"
        tail -20 "$ctest_log" | sed 's/^/    /'
        # Try to extract the total anyway so the unified summary
        # can still report a denominator for the C++ side.
        local fail_total=0
        local fail_passed=0
        local fail_line
        fail_line=$(grep -E '^[0-9]+% tests passed' "$ctest_log" | tail -1)
        if [ -n "$fail_line" ]; then
            fail_passed=$(echo "$fail_line" | awk '{print $1}' | tr -d '%')
            fail_total=$(echo "$fail_line" | awk '{print $NF}')
        fi
        if [ "${fail_total:-0}" -gt 0 ]; then
            echo "  C++: $fail_passed/$fail_total FAILED in ${elapsed}s"
            cpp_passed="$fail_passed"
            cpp_total="$fail_total"
        else
            echo "  C++: failed (rc=$rc) in ${elapsed}s"
        fi
        return 1
    fi
    # Parse the "tests passed" line. Examples we accept:
    #   "100% tests passed, 0 tests failed out of 89"
    #   "Total Tests: 89"
    local total=0
    local passed=0
    local line
    line=$(grep -E '^[0-9]+% tests passed' "$ctest_log" | tail -1)
    if [ -n "$line" ]; then
        passed=$(echo "$line" | awk '{print $1}' | tr -d '%')
        total=$(echo "$line" | awk '{print $NF}')
    fi
    if [ "${passed:-0}" = "0" ] || [ "${total:-0}" = "0" ]; then
        line=$(grep -E 'Total Tests:' "$ctest_log" | tail -1)
        if [ -n "$line" ]; then
            total=$(echo "$line" | awk '{print $NF}')
            passed="$total"
        fi
    fi
    if [ "${total:-0}" -eq 0 ]; then
        echo "  ctest: WARN (could not parse summary; treating as 0/0)"
        echo "  C++: 0/0 passed in ${elapsed}s"
        cpp_passed=0
        cpp_total=0
        return 0
    fi
    echo "  C++: $passed/$total passed in ${elapsed}s"
    cpp_passed="$passed"
    cpp_total="$total"
    return 0
}

# --- pytest surface -------------------------------------------------------
run_python_tests() {
    local start
    start=$(date +%s)
    local pytest_args=("${PYTEST_OPTS[@]}")
    if [ "$QUICK" = "1" ]; then
        # -m "not slow" -- the slow marker is registered in conftest.py
        pytest_args+=("-m" "not slow")
    fi
    # Test roots: tools/tessera (Python unit + integration tests for
    # the calibration / retune / DB / etc. surface) and tests/ (C++
    # tests use the test-*.cpp naming and are skipped by pytest, but
    # the new tests/test_phase16_e2e.py lives here so it's collected
    # by this glob).
    local roots=("tools/tessera" "tests")
    # Use `python3 -m pytest` (not a `pytest` shim on PATH) so the
    # pytest process inherits the same Python env as the test
    # scripts themselves. A standalone pytest shim is often pinned
    # to a different interpreter (e.g. Homebrew's pytest is on
    # python 3.13 while python3 is on 3.14); that mismatch causes
    # ModuleNotFoundError on polars / duckdb for the tessera
    # tests.
    if ! command -v python3 >/dev/null 2>&1; then
        echo "  pytest: SKIP (python3 not on PATH)"
        echo "  Python: skipped"
        return 0
    fi
    local pytest_log="$REPO_ROOT/.test-all-pytest.log"
    local -a cmd=(python3 -m pytest "${pytest_args[@]}")
    local r
    for r in "${roots[@]}"; do
        cmd+=("$r")
    done
    "${cmd[@]}" > "$pytest_log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ "$rc" -ne 0 ]; then
        echo "  pytest: FAIL (see $pytest_log)"
        tail -30 "$pytest_log" | sed 's/^/    /'
        echo "  Python: failed (rc=$rc) in ${elapsed}s"
        return 1
    fi
    # Parse the summary line. Example: "178 passed in 4.20s"
    local total=0
    local passed=0
    local line
    line=$(grep -E '^[0-9]+ (passed|failed|error)' "$pytest_log" | tail -1)
    if [ -n "$line" ]; then
        # Examples:
        #   "178 passed in 4.20s"
        #   "178 passed, 1 skipped in 4.20s"
        #   "1 failed, 178 passed in 4.20s"
        local p f e
        p=$(echo "$line" | grep -oE '[0-9]+ passed' | awk '{print $1}')
        f=$(echo "$line" | grep -oE '[0-9]+ failed' | awk '{print $1}')
        e=$(echo "$line" | grep -oE '[0-9]+ error' | awk '{print $1}')
        passed="${p:-0}"
        local fail="${f:-0}"
        local err="${e:-0}"
        total=$(( passed + fail + err ))
    fi
    if [ "$total" -eq 0 ]; then
        # pytest's quiet mode might emit only "no tests ran" or
        # "= ERRORS =" with no leading summary line.
        echo "  pytest: WARN (no summary parsed; treating as 0/0)"
        echo "  Python: 0/0 passed in ${elapsed}s"
        py_passed=0
        py_total=0
        return 0
    fi
    echo "  Python: $passed/$total passed in ${elapsed}s"
    py_passed="$passed"
    py_total="$total"
    return 0
}

# --- main -----------------------------------------------------------------
echo "Tessera test runner"
echo "  repo:    $REPO_ROOT"
echo "  quick:   $QUICK"
echo "  cpp-only: $CPP_ONLY"
echo "  py-only:  $PY_ONLY"
echo "  jobs:    $JOBS"
echo ""

overall_start=$(date +%s)
cpp_rc=0
py_rc=0
cpp_passed=0
cpp_total=0
py_passed=0
py_total=0

if [ "$PY_ONLY" = "0" ]; then
    echo "C++ (ctest):"
    run_cpp_tests || cpp_rc=$?
fi

if [ "$CPP_ONLY" = "0" ]; then
    echo ""
    echo "Python (pytest):"
    run_python_tests || py_rc=$?
fi

overall_elapsed=$(( $(date +%s) - overall_start ))

echo ""
echo "=========================================="
# Unified single-line summary in the format the spec calls for:
#   C++: 89/89 passed | Python: 188/188 passed | TOTAL: 277/277 passed in 42s
# When only one surface ran, omit the missing side; the format
# still has a `TOTAL: ...` line so downstream tooling can grep.
if [ "$CPP_ONLY" = "1" ] && [ "$PY_ONLY" = "0" ]; then
    if [ "$cpp_rc" -eq 0 ]; then
        echo "C++: $cpp_passed/$cpp_total passed | TOTAL: $cpp_passed/$cpp_total passed in ${overall_elapsed}s"
    else
        echo "C++: $cpp_passed/$cpp_total FAILED | TOTAL: FAILED in ${overall_elapsed}s"
    fi
    exit "$cpp_rc"
fi
if [ "$PY_ONLY" = "1" ] && [ "$CPP_ONLY" = "0" ]; then
    if [ "$py_rc" -eq 0 ]; then
        echo "Python: $py_passed/$py_total passed | TOTAL: $py_passed/$py_total passed in ${overall_elapsed}s"
    else
        echo "Python: $py_passed/$py_total FAILED | TOTAL: FAILED in ${overall_elapsed}s"
    fi
    exit "$py_rc"
fi
# Both surfaces ran.
cpp_label=""
py_label=""
if [ -n "${cpp_total:-}" ] && [ "$cpp_total" -gt 0 ]; then
    if [ "$cpp_rc" -eq 0 ]; then
        cpp_label="C++: $cpp_passed/$cpp_total passed"
    else
        cpp_label="C++: $cpp_passed/$cpp_total FAILED"
    fi
fi
if [ -n "${py_total:-}" ] && [ "$py_total" -gt 0 ]; then
    if [ "$py_rc" -eq 0 ]; then
        py_label="Python: $py_passed/$py_total passed"
    else
        py_label="Python: $py_passed/$py_total FAILED"
    fi
fi
total_passed=$(( ${cpp_passed:-0} + ${py_passed:-0} ))
total_count=$(( ${cpp_total:-0} + ${py_total:-0} ))
if [ "$cpp_rc" -eq 0 ] && [ "$py_rc" -eq 0 ]; then
    if [ -n "$cpp_label" ] && [ -n "$py_label" ]; then
        echo "$cpp_label | $py_label | TOTAL: $total_passed/$total_count passed in ${overall_elapsed}s"
    elif [ -n "$cpp_label" ]; then
        echo "$cpp_label | TOTAL: $total_passed/$total_count passed in ${overall_elapsed}s"
    elif [ -n "$py_label" ]; then
        echo "$py_label | TOTAL: $total_passed/$total_count passed in ${overall_elapsed}s"
    else
        echo "TOTAL: 0/0 (no tests ran) in ${overall_elapsed}s"
    fi
    exit 0
fi
# At least one surface failed.
if [ -n "$cpp_label" ] && [ -n "$py_label" ]; then
    echo "$cpp_label | $py_label | TOTAL: FAILED in ${overall_elapsed}s"
elif [ -n "$cpp_label" ]; then
    echo "$cpp_label | TOTAL: FAILED in ${overall_elapsed}s"
elif [ -n "$py_label" ]; then
    echo "$py_label | TOTAL: FAILED in ${overall_elapsed}s"
else
    echo "TOTAL: FAILED in ${overall_elapsed}s"
fi
exit 1
