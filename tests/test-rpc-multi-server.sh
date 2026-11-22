#!/usr/bin/env bash
set -euo pipefail

server=$1
client=$2
port_a=$((40000 + $$ % 10000))
port_b=$((port_a + 1))
endpoint_a="127.0.0.1:${port_a}"
endpoint_b="127.0.0.1:${port_b}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid_a:-}" "${pid_b:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    local port=$1
    for _ in {1..2400}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

# NOTE: endpoint A must use a backend whose get_alloc_size() for GGML_OP_TOP_K
# actually scales with nelements(src0) (e.g. Metal, CUDA) for the growing-src
# alloc-size-cache regression check below to be meaningful. On a CPU-only backend
# alloc_size defaults to ggml_nbytes(output) and does not depend on src at all, so
# that check would pass trivially either way.
# ref: https://github.com/ggml-org/llama.cpp/issues/30794
"$server" --device "${TEST_RPC_DEVICE:-CPU}" --host 127.0.0.1 --port "$port_a" >"$test_dir/server-a.log" 2>&1 &
pid_a=$!
"$server" --device CPU --host 127.0.0.1 --port "$port_b" >"$test_dir/server-b.log" 2>&1 &
pid_b=$!
wait_for_port "$port_a"
wait_for_port "$port_b"

"$client" "$endpoint_a" "$endpoint_b"

if grep -q "invalid data ptr" "$test_dir/server-b.log"; then
    cat "$test_dir/server-b.log"
    exit 1
fi
