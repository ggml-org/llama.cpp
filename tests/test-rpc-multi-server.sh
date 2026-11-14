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
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

"$server" --device CPU --host 127.0.0.1 --port "$port_a" >"$test_dir/server-a.log" 2>&1 &
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

# a graph with an op the server's backend cannot run is refused, not computed
# (the client aborts on purpose here, so skip the backtrace)
if GGML_NO_BACKTRACE=1 "$client" "$endpoint_a" "$endpoint_b" unsupported >"$test_dir/client-unsupported.log" 2>&1; then
    echo "server B computed a graph with an op its backend cannot run"
    exit 1
fi
if ! grep -q "does not support CPY" "$test_dir/server-b.log"; then
    cat "$test_dir/server-b.log" "$test_dir/client-unsupported.log"
    exit 1
fi
