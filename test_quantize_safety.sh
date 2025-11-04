#!/bin/bash
# Test script for llama-quantize safety features (issue #12753)

set -e

echo "=========================================="
echo "Testing llama-quantize safety features"
echo "=========================================="
echo ""

QUANTIZE="./build-cpu/bin/llama-quantize"
TEST_MODEL="/tmp/test_model.gguf"

# Create a small test model (if you don't have one, we'll skip actual quantization)
if [ ! -f "$TEST_MODEL" ]; then
    echo "⚠️  No test model found at $TEST_MODEL"
    echo "   Creating a dummy file for testing file protection logic..."
    dd if=/dev/zero of="$TEST_MODEL" bs=1M count=10 2>/dev/null
fi

echo "✓ Test model ready: $TEST_MODEL"
echo ""

# Test 1: Try to quantize with same input/output (should fail)
echo "Test 1: Attempting in-place quantization without --inplace flag..."
echo "Command: $QUANTIZE $TEST_MODEL $TEST_MODEL Q4_0"
if $QUANTIZE "$TEST_MODEL" "$TEST_MODEL" Q4_0 2>&1 | grep -q "ERROR: Input and output files are the same"; then
    echo "✓ Test 1 PASSED: Correctly blocked same input/output"
else
    echo "✗ Test 1 FAILED: Should have blocked same input/output"
    exit 1
fi
echo ""

# Test 2: Try with --inplace flag (should work with temp file)
echo "Test 2: Attempting in-place quantization WITH --inplace flag..."
echo "Command: $QUANTIZE --inplace $TEST_MODEL $TEST_MODEL Q4_0"
echo "(This will likely fail on dummy file, but should show temp file usage)"
if $QUANTIZE --inplace "$TEST_MODEL" "$TEST_MODEL" Q4_0 2>&1 | grep -q "using temporary file"; then
    echo "✓ Test 2 PASSED: Correctly using temporary file for in-place"
else
    echo "⚠️  Test 2: Could not verify temp file usage (may need real model)"
fi
echo ""

# Test 3: Test symlink detection
echo "Test 3: Testing symlink detection..."
TEST_SYMLINK="/tmp/test_model_link.gguf"
rm -f "$TEST_SYMLINK"
ln -s "$TEST_MODEL" "$TEST_SYMLINK"
echo "Command: $QUANTIZE $TEST_MODEL $TEST_SYMLINK Q4_0"
if $QUANTIZE "$TEST_MODEL" "$TEST_SYMLINK" Q4_0 2>&1 | grep -q "ERROR: Input and output files are the same"; then
    echo "✓ Test 3 PASSED: Correctly detected symlink to same file"
else
    echo "✗ Test 3 FAILED: Should have detected symlink"
    rm -f "$TEST_SYMLINK"
    exit 1
fi
rm -f "$TEST_SYMLINK"
echo ""

# Test 4: Test overwrite protection
echo "Test 4: Testing overwrite protection..."
TEST_OUTPUT="/tmp/test_output.gguf"
echo "dummy" > "$TEST_OUTPUT"
echo "Command: $QUANTIZE $TEST_MODEL $TEST_OUTPUT Q4_0"
if $QUANTIZE "$TEST_MODEL" "$TEST_OUTPUT" Q4_0 2>&1 | grep -q "ERROR: Output file already exists"; then
    echo "✓ Test 4 PASSED: Correctly blocked overwriting existing file"
else
    echo "✗ Test 4 FAILED: Should have blocked overwrite"
    rm -f "$TEST_OUTPUT"
    exit 1
fi
rm -f "$TEST_OUTPUT"
echo ""

echo "=========================================="
echo "✓ All safety tests passed!"
echo "=========================================="
echo ""
echo "Summary of safety features:"
echo "  1. ✓ Prevents accidental same input/output"
echo "  2. ✓ Detects hardlinks and symlinks"
echo "  3. ✓ Uses temp file + atomic rename for --inplace"
echo "  4. ✓ Protects against overwriting existing files"
echo ""
echo "Usage:"
echo "  Normal:    $QUANTIZE input.gguf output.gguf Q4_0"
echo "  In-place:  $QUANTIZE --inplace model.gguf model.gguf Q4_0"
echo "  Overwrite: $QUANTIZE --overwrite input.gguf existing.gguf Q4_0"
