#!/usr/bin/env bash
# Greedy generation on OpenVINO: all-CPU vs CPU prefill + GPU decode (phase split).
# Requires a GGUF model (LLAMACPP_TEST_MODELFILE or first argument).
set -euo pipefail

MODEL="${1:-${LLAMACPP_TEST_MODELFILE:-}}"
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
    printf '\033[33mWARNING: No model file. Skipping test-openvino-phase-split.\n\033[0m' >&2
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${LLAMA_BIN_DIR:-$(cd "${SCRIPT_DIR}/../build/bin" 2>/dev/null && pwd || true)}"
if [[ -z "${BIN_DIR}" || ! -d "${BIN_DIR}" ]]; then
    BIN_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY:-.}"
fi

SIMPLE="${BIN_DIR}/llama-simple"
CLI="${BIN_DIR}/llama-cli"

if [[ ! -x "${SIMPLE}" ]]; then
    echo "error: llama-simple not found at ${SIMPLE}" >&2
    exit 1
fi

PROMPT="The capital of France is"
N_PRED=12
CACHE_ROOT="${TMPDIR:-/tmp}/llama_ov_phase_split_test_$$"
mkdir -p "${CACHE_ROOT}"

run_simple() {
    local tag="$1"
    shift
    export GGML_OPENVINO_CACHE_DIR="${CACHE_ROOT}/${tag}"
    env "$@" "${SIMPLE}" -m "${MODEL}" -n "${N_PRED}" "${PROMPT}" 2>/dev/null
}

if ! command -v "${SIMPLE}" >/dev/null 2>&1 && [[ ! -x "${SIMPLE}" ]]; then
    echo "error: llama-simple not executable at ${SIMPLE}" >&2
    exit 1
fi

OUT_CPU="$(run_simple cpu GGML_OPENVINO_DEVICE=CPU)" || {
    printf '\033[33mWARNING: OpenVINO CPU run failed. Skipping.\n\033[0m' >&2
    exit 0
}
OUT_SPLIT="$(run_simple split \
    GGML_OPENVINO_PHASE_SPLIT=1 \
    GGML_OPENVINO_PREFILL_DEVICE=CPU \
    GGML_OPENVINO_DECODE_DEVICE=GPU.0 \
    GGML_OPENVINO_STATEFUL_EXECUTION=1)"

if [[ -z "${OUT_CPU}" || -z "${OUT_SPLIT}" ]]; then
    echo "error: empty generation output" >&2
    exit 1
fi

if [[ "${OUT_CPU}" != "${OUT_SPLIT}" ]]; then
    echo "error: phase-split tokens differ from all-CPU reference" >&2
    echo "--- CPU ---" >&2
    echo "${OUT_CPU}" >&2
    echo "--- split ---" >&2
    echo "${OUT_SPLIT}" >&2
    exit 1
fi

if [[ -x "${CLI}" ]]; then
    export GGML_OPENVINO_CACHE_DIR="${CACHE_ROOT}/cli"
    export GGML_OPENVINO_PHASE_SPLIT=1
    export GGML_OPENVINO_PREFILL_DEVICE=CPU
    export GGML_OPENVINO_DECODE_DEVICE=GPU.0
    export GGML_OPENVINO_STATEFUL_EXECUTION=1
    if ! "${CLI}" -m "${MODEL}" -p "Say OK." -n 4 -st --simple-io --no-display-prompt -c 512 </dev/null >/dev/null 2>&1; then
        echo "error: llama-cli phase-split smoke failed" >&2
        exit 1
    fi
fi

rm -rf "${CACHE_ROOT}"
echo "test-openvino-phase-split: OK"
