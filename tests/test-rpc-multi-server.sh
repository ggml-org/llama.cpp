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

if /bin/bash -c 'ulimit -c 0; "$@"' bash "$client" --zero-element-outside-buffer "$endpoint_a" >"$test_dir/rejected-client.log" 2>&1; then
    echo "RPC server accepted a zero-element tensor with data outside its buffer"
    exit 1
fi
if ! grep -q "data 0x1 with size 0 out of buffer bounds" "$test_dir/server-a.log"; then
    cat "$test_dir/server-a.log"
    exit 1
fi
"$client" --zero-element-buffer-end "$endpoint_a"

if /bin/bash -c 'ulimit -c 0; "$@"' bash "$client" --get-rows-in-buffer-stride "$endpoint_a" >"$test_dir/get-rows-rejected-client.log" 2>&1; then
    echo "RPC server accepted GET_ROWS with a forged in-buffer row stride"
    exit 1
fi
if ! grep -q "malformed GET_ROWS graph" "$test_dir/server-a.log"; then
    cat "$test_dir/server-a.log"
    exit 1
fi
"$client" --get-rows-valid "$endpoint_a"

"$client" "$endpoint_a" "$endpoint_b"

if grep -q "invalid data ptr" "$test_dir/server-b.log"; then
    cat "$test_dir/server-b.log"
    exit 1
fi
