#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/model-q8_0.gguf" >&2
    exit 2
fi

MODEL=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)

MACA_PATH=${MACA_PATH:-/opt/maca}
MACA_CU_BRIDGE=${MACA_CU_BRIDGE:-$MACA_PATH/tools/cu-bridge}
MACA_BUILD_DIR=${MACA_BUILD_DIR:-$REPO_ROOT/build-maca}
MACA_DEVICE=${MACA_DEVICE:-MACA0}
MACA_TEST_OUTPUT=${MACA_TEST_OUTPUT:-${TMPDIR:-/tmp}/llama-maca-validation}

if [[ ! -f "$MODEL" ]]; then
    echo "Model does not exist: $MODEL" >&2
    exit 1
fi

mkdir -p "$MACA_TEST_OUTPUT"

export LD_LIBRARY_PATH="$MACA_BUILD_DIR/bin:$MACA_PATH/lib:$MACA_CU_BRIDGE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO_ROOT"

"$MACA_BUILD_DIR/bin/llama-cli" --list-devices \
    2>&1 | tee "$MACA_TEST_OUTPUT/devices.log"

"$MACA_BUILD_DIR/bin/test-backend-ops" \
    -b "$MACA_DEVICE" \
    -o MUL_MAT \
    -p 'type_a=q8_0,type_b=f32' \
    -j 1 \
    2>&1 | tee "$MACA_TEST_OUTPUT/q8-mul-mat.log"

"$MACA_BUILD_DIR/bin/llama-bench" \
    -m "$MODEL" \
    -ngl 99 \
    -p 512 \
    -n 512 \
    -b 512 \
    -ub 512 \
    -r 5 \
    -o jsonl \
    > "$MACA_TEST_OUTPUT/q8-benchmark.jsonl" \
    2> "$MACA_TEST_OUTPUT/q8-benchmark.log"

echo "Validation artifacts: $MACA_TEST_OUTPUT"
echo "Interactive smoke test:"
echo "  $MACA_BUILD_DIR/bin/llama-cli -m $MODEL -ngl 99 -cnv --simple-io -n 128"
