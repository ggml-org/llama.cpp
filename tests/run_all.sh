#!/bin/bash
PASS=0; FAIL=0

run() {
    echo "▶ $1"
    if eval "$2"; then ((PASS++)); echo "  ✓ PASS"
    else           ((FAIL++)); echo "  ✗ FAIL"; fi
}

run "P1: routing rules"       "python3 tests/unit/test_routing.py"

run "P1: convert_split_gguf output matches split metadata" \
    "timeout 5 llama.cpp-PoC/build/bin/llama-cli -m models/attn.gguf -p 'hi' -n 1 2>&1 | grep -i 'split.type: attention'"

run "P1: convert_split_gguf output ffn metadata" \
    "timeout 5 llama.cpp-PoC/build/bin/llama-cli -m models/ffn.gguf -p 'hi' -n 1 2>&1 | grep -i 'split.type: ffn'"

run "P2: test_mmap_loader" "python3 tests/unit/test_mmap_loader.py"

echo ""
echo "══════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed"
echo "══════════════════════════════════"
[[ $FAIL -eq 0 ]]
