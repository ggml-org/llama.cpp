#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 BUILD_DIR {cpu|sycl}" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage

build_dir=$1
profile=$2

[[ -d "$build_dir" ]] || {
    echo "required-target gate: build directory does not exist: $build_dir" >&2
    exit 1
}

case "$profile" in
    cpu)
        required_targets=(
            test-server-cors-proxy-policy
        )
        ;;
    sycl)
        required_targets=(
            test-sycl-turbo-correctness
            test-sycl-fusion-eligibility
            test-sycl-fattn-mkl-policy
            test-sycl-status-propagation
        )
        ;;
    *)
        usage
        ;;
esac

command -v ninja >/dev/null 2>&1 || {
    echo "required-target gate: ninja is not available" >&2
    exit 1
}
command -v ctest >/dev/null 2>&1 || {
    echo "required-target gate: ctest is not available" >&2
    exit 1
}

ninja_targets=$(mktemp)
ctest_tests=$(mktemp)
trap 'rm -f "$ninja_targets" "$ctest_tests"' EXIT

ninja -C "$build_dir" -t targets all > "$ninja_targets"
ctest --test-dir "$build_dir" -N > "$ctest_tests"

missing=0
for target in "${required_targets[@]}"; do
    if ! grep -Eq "(^|/)${target}:" "$ninja_targets"; then
        echo "required-target gate: ninja target is missing: $target" >&2
        missing=1
    fi
    if ! grep -Eq "Test +#[0-9]+: ${target}$" "$ctest_tests"; then
        echo "required-target gate: CTest registration is missing: $target" >&2
        missing=1
    fi
done

if (( missing != 0 )); then
    exit 1
fi

printf 'required-target gate: %s profile passed (%d targets)\n' "$profile" "${#required_targets[@]}"
