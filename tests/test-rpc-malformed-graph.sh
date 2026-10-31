#!/usr/bin/env bash
set -euo pipefail

server=$1
client=$2
# 52000..59999: does not overlap the 40000..50000 range used by test-rpc-multi-server.sh
port=$((52000 + $$ % 8000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    local p=$1
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
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

# a well-formed PAD_REFLECT_1D graph must still be accepted and must compute the reflected output
"$client" "$endpoint" good

# a node that would write past its destination must be rejected instead of corrupting the heap
"$client" "$endpoint" bad || true

# wait for the rejection to reach the server log instead of sleeping for a fixed amount of time
for _ in {1..100}; do
    if grep -q "malformed PAD_REFLECT_1D graph detected" "$test_dir/server.log"; then
        break
    fi
    sleep 0.05
done

if ! grep -q "malformed PAD_REFLECT_1D graph detected" "$test_dir/server.log"; then
    echo "server did not report the malformed graph:"
    cat "$test_dir/server.log"
    exit 1
fi

# ... and it must still be running
if ! kill -0 "$pid" 2>/dev/null; then
    echo "server died on the malformed graph:"
    cat "$test_dir/server.log"
    exit 1
fi
