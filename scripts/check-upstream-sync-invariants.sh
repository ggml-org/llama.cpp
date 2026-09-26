#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

fail() {
    echo "upstream-sync invariant gate: $*" >&2
    exit 1
}

require_fixed() {
    local needle=$1
    local path=$2

    grep -Fq -- "$needle" "$path" || fail "missing '$needle' in $path"
}

# A parsed allowlist is not a security control unless the server route consumes it.
require_fixed 'proxy_handler_get(params.ui_mcp_proxy_allow)' tools/server/server.cpp
require_fixed 'proxy_handler_post(params.ui_mcp_proxy_allow)' tools/server/server.cpp

# Keep the compiled policy, its global-address classifiers, and handler boundary.
require_fixed 'static proxy_target_policy_result proxy_target_policy(' tools/server/server-cors-proxy.h
require_fixed 'static bool proxy_ipv4_is_global(' tools/server/server-cors-proxy.h
require_fixed 'static bool proxy_ipv6_is_global(' tools/server/server-cors-proxy.h
require_fixed 'static server_http_context::handler_t proxy_handler_get(' tools/server/server-cors-proxy.h
require_fixed 'static server_http_context::handler_t proxy_handler_post(' tools/server/server-cors-proxy.h
require_fixed 'metadata.google.internal' tools/server/server-cors-proxy.h
require_fixed 'metadata.goog' tools/server/server-cors-proxy.h

# Keep both the compiled policy target and the end-to-end private/metadata cases.
require_fixed 'test-server-cors-proxy-policy.cpp' tools/server/CMakeLists.txt
require_fixed 'test-server-cors-proxy-policy' tools/server/CMakeLists.txt
require_fixed 'def test_mcp_proxy_rejects_disallowed_direct_targets(' tools/server/tests/unit/test_proxy.py
require_fixed 'def test_mcp_proxy_rejects_malformed_url_as_client_error(' tools/server/tests/unit/test_proxy.py
require_fixed 'def test_mcp_proxy_allow_is_exact_not_near_match(' tools/server/tests/unit/test_proxy.py
require_fixed 'def test_mcp_proxy_allow_cli_replaces_environment(' tools/server/tests/unit/test_proxy.py
require_fixed 'def test_mcp_proxy_allow_does_not_disable_default_policy(' tools/server/tests/unit/test_proxy.py
require_fixed 'http://127.0.0.1:1/' tools/server/tests/unit/test_proxy.py
require_fixed 'http://169.254.169.254/' tools/server/tests/unit/test_proxy.py
require_fixed 'http://[::1]:1/' tools/server/tests/unit/test_proxy.py
require_fixed 'http://[::ffff:127.0.0.1]:1/' tools/server/tests/unit/test_proxy.py
require_fixed 'http://metadata.google.internal/' tools/server/tests/unit/test_proxy.py
require_fixed 'http://metadata.goog' tools/server/tests/unit/test_proxy.py

removed_backends=(
    ggml-cann
    ggml-cuda
    ggml-et
    ggml-hexagon
    ggml-hip
    ggml-metal
    ggml-musa
    ggml-opencl
    ggml-rpc
    ggml-virtgpu
    ggml-webgpu
    ggml-zdnn
    ggml-zendnn
)

for backend in "${removed_backends[@]}"; do
    [[ ! -e "ggml/src/$backend" ]] || fail "removed backend directory returned: ggml/src/$backend"
done

# Search build and CI registration identifiers, not prose or opt-in dynamic plugin
# calls. Raw OpenCL references in the OpenVINO workflow/actions and OpenCL target
# strings in ggml-sycl are retained integrations, not a restored ggml-opencl backend.
removed_ref_re='ggml-(cann|cuda|et|hexagon|hip|metal|musa|opencl|rpc|virtgpu|webgpu|zdnn|zendnn)|GGML_(USE_)?(CANN|CUDA|ET|HEXAGON|HIP|METAL|MUSA|OPENCL|RPC|VIRTGPU|WEBGPU|ZDNN|ZENDNN)'
removed_matches=$(mktemp)
trap 'rm -f "$removed_matches"' EXIT

set +e
git grep -nE "$removed_ref_re" -- \
    .github ci scripts \
    ':(glob)**/CMakeLists.txt' \
    ':(glob)**/*.cmake' \
    ':!scripts/check-upstream-sync-invariants.sh' \
    ':!.github/workflows/build-openvino.yml' \
    ':!.github/actions/linux-setup-openvino/action.yml' \
    ':!.github/actions/windows-setup-openvino/action.yml' \
    ':!ggml/src/ggml-openvino/**' \
    ':!ggml/src/ggml-sycl/**' \
    > "$removed_matches"
grep_status=$?
set -e

if (( grep_status == 0 )); then
    cat "$removed_matches" >&2
    fail "removed backend build/CI reference returned"
fi
if (( grep_status != 1 )); then
    fail "git grep failed while checking removed backend references"
fi

printf 'upstream-sync invariant gate: passed\n'
