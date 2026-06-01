#!/bin/bash
# TurboQuant quality + speed gate — run BEFORE pushing any changes
# Checks: (1) perplexity within 5% of q8_0, (2) context scaling ratio > 0.95
#
# Usage: bash scripts/turbo-quality-gate.sh
# Exit 0 = PASS, Exit 1 = FAIL

set -e

LLAMA=${LLAMA:-~/local_llms/llama.cpp/build-turbo/bin}
MODEL=${MODEL:-~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf}
WIKI=${WIKI:-~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw}

if [ ! -f "$WIKI" ]; then
    echo "Downloading wikitext-2..."
    bash ~/local_llms/llama.cpp/scripts/get-wikitext-2.sh
fi

FAIL=0

echo "========================================"
echo "  TurboQuant Quality + Speed Gate"
echo "========================================"
echo ""

# --- Test 1: Perplexity ---
echo "[1/2] Running perplexity check (8 chunks)..."
PPL_TURBO=$($LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 2>&1 | grep "Final" | grep -oE 'PPL = [0-9.]+' | grep -oE '[0-9.]+')

if [ -z "$PPL_TURBO" ]; then
    echo "  FAIL: Could not get turbo3 perplexity (crash or timeout)"
    FAIL=1
else
    BASELINE_PPL=6.111
    MAX_PPL=$(echo "$BASELINE_PPL * 1.05" | bc)
    PPL_OK=$(echo "$PPL_TURBO < $MAX_PPL" | bc)
    if [ "$PPL_OK" -eq 1 ]; then
        echo "  PASS: turbo3 PPL = $PPL_TURBO (< $MAX_PPL, within 5% of q8_0 $BASELINE_PPL)"
    else
        echo "  FAIL: turbo3 PPL = $PPL_TURBO (> $MAX_PPL, exceeds 5% threshold)"
        FAIL=1
    fi
fi
echo ""

# --- Test 2: Context Scaling ---
echo "[2/2] Running context scaling check (4K prefill)..."
TURBO_TPS=$($LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 4096 -ctk turbo3 -ctv turbo3 -fa on --chunks 4 -ngl 99 2>&1 | grep "prompt eval" | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+')
Q8_TPS=$($LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 4096 -ctk q8_0 -ctv q8_0 -fa on --chunks 4 -ngl 99 2>&1 | grep "prompt eval" | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+')

if [ -z "$TURBO_TPS" ] || [ -z "$Q8_TPS" ]; then
    echo "  FAIL: Could not measure speed (crash or timeout)"
    echo "  turbo3=$TURBO_TPS q8_0=$Q8_TPS"
    FAIL=1
else
    RATIO=$(echo "scale=4; $TURBO_TPS / $Q8_TPS" | bc)
    RATIO_OK=$(echo "$RATIO > 0.95" | bc)
    if [ "$RATIO_OK" -eq 1 ]; then
        echo "  PASS: turbo3/q8_0 = ${RATIO}x at 4K context (> 0.95 threshold)"
        echo "  turbo3 = $TURBO_TPS tok/s, q8_0 = $Q8_TPS tok/s"
    else
        echo "  FAIL: turbo3/q8_0 = ${RATIO}x at 4K context (< 0.95 threshold)"
        echo "  turbo3 = $TURBO_TPS tok/s, q8_0 = $Q8_TPS tok/s"
        echo "  Context scaling regression detected!"
        FAIL=1
    fi
fi
echo ""

# --- Summary ---
echo "========================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
    echo "========================================"
    exit 0
else
    echo "  CHECKS FAILED — DO NOT PUSH"
    echo "========================================"
    exit 1
fi
