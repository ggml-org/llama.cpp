#!/bin/bash
# TurboQuant quality gate — run BEFORE pushing any changes
# Checks perplexity against q8_0 baseline. Fails if >5% degradation.
#
# Usage: bash scripts/turbo-quality-gate.sh

set -e

LLAMA=~/local_llms/llama.cpp/build-turbo/bin
MODEL=~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf
WIKI=~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw

if [ ! -f "$WIKI" ]; then
    echo "Downloading wikitext-2..."
    bash ~/local_llms/llama.cpp/scripts/get-wikitext-2.sh
fi

echo "=== TurboQuant Quality Gate ==="
echo ""

# Run turbo3 perplexity
echo "Running turbo3 perplexity..."
PPL_TURBO=$($LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 2>&1 | grep "Final" | grep -oE 'PPL = [0-9.]+' | grep -oE '[0-9.]+')

if [ -z "$PPL_TURBO" ]; then
    echo "FAIL: Could not get turbo3 perplexity (crash or timeout)"
    exit 1
fi

echo "turbo3 PPL = $PPL_TURBO"

# q8_0 baseline: 6.111 (hardcoded from our measurements)
BASELINE=6.111
MAX_ALLOWED=$(echo "$BASELINE * 1.05" | bc)  # 5% threshold

echo "q8_0 baseline = $BASELINE"
echo "Max allowed (5%) = $MAX_ALLOWED"

PASS=$(echo "$PPL_TURBO < $MAX_ALLOWED" | bc)
if [ "$PASS" -eq 1 ]; then
    echo ""
    echo "PASS: turbo3 PPL $PPL_TURBO < $MAX_ALLOWED (within 5% of q8_0)"
    exit 0
else
    echo ""
    echo "FAIL: turbo3 PPL $PPL_TURBO > $MAX_ALLOWED (exceeds 5% threshold)"
    echo "DO NOT PUSH. Fix quality before committing."
    exit 1
fi
