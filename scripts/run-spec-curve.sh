#!/usr/bin/env bash
# Record request-total MTP acceptance traces. Python owns the GPU evidence gates.
# SERVER_BIN=./build-sycl/bin/llama-server MODEL=/path/model.gguf bash scripts/run-spec-curve.sh
# Analyze with test-spec-adaptive-curve --curve-file <adaptive.jsonl> --fixed-curve-file <fixed7.jsonl>.
set -euo pipefail

if [[ -z "${SERVER_BIN:-}" || -z "${MODEL:-}" ]]; then
    echo "SERVER_BIN and MODEL must be set" >&2
    exit 2
fi
export MODE="${MODE:-acceptance-curve}"
export CTX="${CTX:-2048}"
SERVICE=llama-sycl.cpp.service
restore_service=0
cleanup() {
    status=$?
    if (( restore_service )); then
        if ! sudo -n systemctl start "$SERVICE"; then
            echo "failed to restore $SERVICE" >&2
            status=1
        fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${SKIP_STOP_SERVICE:-0}" != "1" ]] && systemctl is-active --quiet "$SERVICE"; then
    restore_service=1
    sudo -n systemctl stop "$SERVICE"
fi

timeout --signal=INT --kill-after=30s "${CURVE_TIMEOUT:-2h}" \
    python3 "$(dirname "$0")/perf/bench_spec.py"
