#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/model.gguf (Q8_0 operator coverage plus the supplied model benchmark)" >&2
    exit 2
fi

MODEL=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)

MACA_PATH=${MACA_PATH:-/opt/maca}
MACA_CU_BRIDGE=${MACA_CU_BRIDGE:-$MACA_PATH/tools/cu-bridge}
MACA_BUILD_DIR=${MACA_BUILD_DIR:-$REPO_ROOT/build-maca}
MACA_DEVICE=${MACA_DEVICE:-MACA0}
MACA_TEST_OUTPUT=${MACA_TEST_OUTPUT:-$MACA_BUILD_DIR/validation}

if [[ ! -f "$MODEL" ]]; then
    echo "Model does not exist: $MODEL" >&2
    exit 1
fi

MODEL=$(cd -- "$(dirname -- "$MODEL")" && printf '%s/%s' "$PWD" "$(basename -- "$MODEL")")
mkdir -p "$MACA_TEST_OUTPUT"
MACA_TEST_OUTPUT=$(mktemp -d "$MACA_TEST_OUTPUT/$(basename -- "$MODEL" .gguf)-XXXXXX")
echo "Validation artifacts: $MACA_TEST_OUTPUT"

export LD_LIBRARY_PATH="$MACA_BUILD_DIR/bin:$MACA_PATH/lib:$MACA_CU_BRIDGE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$MACA_BUILD_DIR/bin/llama-cli" --list-devices \
    2>&1 | tee "$MACA_TEST_OUTPUT/devices.log"

if ! awk -v device="$MACA_DEVICE:" '$1 == device { found = 1 } END { exit !found }' "$MACA_TEST_OUTPUT/devices.log"; then
    echo "Requested device was not found: $MACA_DEVICE" >&2
    exit 1
fi

echo "Operator coverage: Q8_0 x F32 MUL_MAT, independent of the supplied model format"
"$MACA_BUILD_DIR/bin/test-backend-ops" \
    -b "$MACA_DEVICE" \
    -o MUL_MAT \
    -p 'type_a=q8_0,type_b=f32' \
    -j 1 \
    2>&1 | tee "$MACA_TEST_OUTPUT/q8-mul-mat.log"

if ! awk '/^[[:space:]]*[0-9]+\/[0-9]+ tests passed/ { split($1, counts, "/"); if (counts[1] > 0 && counts[1] == counts[2]) passed = 1 } END { exit !passed }' "$MACA_TEST_OUTPUT/q8-mul-mat.log"; then
    echo "No non-empty passing operator test summary was found" >&2
    exit 1
fi

"$MACA_BUILD_DIR/bin/llama-bench" \
    -m "$MODEL" \
    --device "$MACA_DEVICE" \
    -ngl 99 \
    -p 512 \
    -n 512 \
    -b 512 \
    -ub 512 \
    -r 5 \
    -o jsonl \
    > "$MACA_TEST_OUTPUT/benchmark.jsonl" \
    2> "$MACA_TEST_OUTPUT/benchmark.log"

echo "Validation artifacts: $MACA_TEST_OUTPUT"
echo "Interactive smoke test:"
printf '  %q -m %q --device %q -ngl 99 --simple-io -n 128\n' "$MACA_BUILD_DIR/bin/llama-cli" "$MODEL" "$MACA_DEVICE"
