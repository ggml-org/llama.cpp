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

PATTERN='->data([^_[:alnum:]]|$)'
ALLOW_RE='^ggml/src/ggml-sycl/(ggml-sycl\.cpp|common\.(hpp|cpp))$'

RG_ARGS=(-n --no-heading --glob '!**/dpct/**' --glob '!**/docs/**' --glob '!**/tests/**')

violations=0

while IFS= read -r match; do
    rel="${match%%:*}"
    rest="${match#*:}"
    line="${rest%%:*}"
    code="${rest#*:}"

    if [[ "$rel" =~ $ALLOW_RE ]]; then
        continue
    fi
    if [[ "$code" =~ raw-ok ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi

    echo "forbidden raw ggml tensor storage access: ${rel}:${line}:${code}" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" -- "$PATTERN" "$SCAN_ROOT" || true
)

if [[ $violations -ne 0 ]]; then
    echo "SYCL tensor-access policy check failed: $violations violation(s)" >&2
    exit 1
fi

echo "SYCL tensor-access policy check passed"
