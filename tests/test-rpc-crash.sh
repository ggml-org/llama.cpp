#!/usr/bin/env bash
set -euo pipefail

server=$1
client=$2
# a different base than test-rpc-multi-server, so the two cannot pick the same port
port=$((30000 + $$ % 10000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid:-}" 2>/dev/null || true
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

"$server" --device CPU --host 127.0.0.1 --port "$port" >"$test_dir/server.log" 2>&1 &
pid=$!
wait_for_port "$port"

# the client kills the server mid-test, so it needs the pid
"$client" "$endpoint" "$pid"
