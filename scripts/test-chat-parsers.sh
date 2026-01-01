#!/bin/bash
# Test both legacy and new PEG chat parsers
#
# This script runs chat parsing tests with both parser implementations
# to ensure the PEG migration doesn't introduce regressions.
#
# Usage:
#   ./scripts/test-chat-parsers.sh [build_dir]
#
# Examples:
#   ./scripts/test-chat-parsers.sh           # uses ./build
#   ./scripts/test-chat-parsers.sh buildDebug

set -e

BUILD_DIR="${1:-build}"
TEST_BINARY="$BUILD_DIR/bin/test-chat"

if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: $TEST_BINARY not found"
    echo "Build it with: cmake -B $BUILD_DIR && cmake --build $BUILD_DIR --target test-chat"
    exit 1
fi

echo "=============================================="
echo "Testing chat parsers (legacy vs PEG)"
echo "=============================================="
echo ""

LEGACY_PASSED=0
PEG_PASSED=0
NEEDLE_PASSED=0

# Test 1: Legacy parsers (default)
echo "[1/3] Testing legacy parsers (use_new_parsers=false)..."
if CHAT_TEST=template_output_parsers "$TEST_BINARY" > /dev/null 2>&1; then
    echo "      PASSED"
    LEGACY_PASSED=1
else
    echo "      FAILED"
fi

# Test 2: New PEG parsers
echo "[2/3] Testing new PEG parsers (use_new_parsers=true)..."
if LLAMA_USE_NEW_PARSERS=1 CHAT_TEST=template_output_parsers "$TEST_BINARY" > /dev/null 2>&1; then
    echo "      PASSED"
    PEG_PASSED=1
else
    echo "      FAILED"
fi

# Test 3: Needle streaming tests (always uses PEG)
echo "[3/3] Testing needle streaming (PEG only)..."
if CHAT_TEST=systematic_needle_streaming "$TEST_BINARY" > /dev/null 2>&1; then
    echo "      PASSED"
    NEEDLE_PASSED=1
else
    echo "      FAILED"
fi

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo "  Legacy parsers:    $([ $LEGACY_PASSED -eq 1 ] && echo 'PASSED' || echo 'FAILED')"
echo "  New PEG parsers:   $([ $PEG_PASSED -eq 1 ] && echo 'PASSED' || echo 'FAILED')"
echo "  Needle streaming:  $([ $NEEDLE_PASSED -eq 1 ] && echo 'PASSED' || echo 'FAILED')"
echo ""

if [ $LEGACY_PASSED -eq 1 ] && [ $PEG_PASSED -eq 1 ] && [ $NEEDLE_PASSED -eq 1 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi
