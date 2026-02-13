#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="${1:-ggml/src/ggml-sycl}"
if [[ "$SCAN_ROOT" = "$ROOT_DIR"* ]]; then
    SCAN_ROOT="${SCAN_ROOT#$ROOT_DIR/}"
fi

if ! command -v rg >/dev/null 2>&1; then
    echo "error: ripgrep (rg) is required" >&2
    exit 2
fi

PATTERN='sycl::malloc_(device|host|shared)\s*\('
ALLOW_RE='^ggml/src/ggml-sycl/(ggml-sycl\.cpp|unified-cache\.cpp|unified-kernel\.cpp|common\.cpp|layer-streaming\.cpp)$'

RG_ARGS=(-n --no-heading --glob '!**/dpct/**')
# The default scan root is already inside ggml/src/ggml-sycl, but keep this
# exclusion when callers pass that tree explicitly.
if [[ "$SCAN_ROOT" == "ggml/src/ggml-sycl" || "$SCAN_ROOT" == "ggml/src/ggml-sycl/"* ]]; then
    RG_ARGS+=(--glob '!**/tests/**')
fi

violations=0

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if [[ "$rel" =~ $ALLOW_RE ]]; then
        continue
    fi
    echo "forbidden raw SYCL alloc: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PATTERN" "$SCAN_ROOT" || true
)

if [[ $violations -ne 0 ]]; then
    echo "SYCL alloc policy check failed: $violations violation(s)" >&2
    exit 1
fi

echo "SYCL alloc policy check passed"
