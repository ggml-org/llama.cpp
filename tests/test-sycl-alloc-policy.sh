#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-alloc-usage.sh"

"$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/good"

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad" >/dev/null 2>&1; then
    echo "expected bad fixture to fail policy check" >&2
    exit 1
fi

echo "sycl alloc policy fixtures passed"
