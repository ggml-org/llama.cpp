#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    printf 'usage: %s TEST_LLAMA_ARCHS TEST_EXPORT_GRAPH_OPS\n' "$0" >&2
    exit 2
fi

arch_fixture_bin=$1
graph_ops_bin=$2

for bin in "$arch_fixture_bin" "$graph_ops_bin"; do
    if [ ! -x "$bin" ]; then
        printf 'required test binary is not executable: %s\n' "$bin" >&2
        exit 2
    fi
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-turbo-attention-architectures.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

architectures=(deepseek32 glm-dsa dots3note dflash deepseek4)

extract_attention_counts() {
    sed -n 's/.*GRAPH_ATTENTION_COUNTS label=\([^ ]*\) attention=\([0-9][0-9]*\) k_turbo=[0-9][0-9]* v_turbo=[0-9][0-9]* forward_wht=[0-9][0-9]* inverse_wht=[0-9][0-9]*.*/\1=\2/p'
}

run_graph_check() {
    local model=$1
    local cache_type_k=$2
    local cache_type_v=$3
    local expectation=$4
    local case_dir=$5
    mkdir -p "$case_dir"
    (
        cd "$case_dir"
        TURBO_LAYER_ADAPTIVE=0 \
        LLAMA_TEST_GRAPH_EXPECTATION=$expectation \
        TURBO_AUTO_ASYMMETRIC=0 \
            "$graph_ops_bin" \
                -m "$model" \
                -c 32 -b 4 -ub 4 \
                -ctk "$cache_type_k" -ctv "$cache_type_v" 2>&1
    )
}

for arch in "${architectures[@]}"; do
    fixture_dir="$work_dir/$arch/fixtures"
    mkdir -p "$fixture_dir"
    "$arch_fixture_bin" --arch "$arch" --seed 1 --out "$fixture_dir"
    model="$fixture_dir/$arch-dense.gguf"
    if [ ! -f "$model" ]; then
        model="$fixture_dir/$arch-moe.gguf"
    fi
    if [ ! -f "$model" ]; then
        printf '%s: no generated GGUF fixture\n' "$arch" >&2
        exit 1
    fi

    if ! f16_output=$(run_graph_check "$model" f16 f16 non-turbo "$work_dir/$arch/f16"); then
        printf '%s\n' "$f16_output"
        exit 1
    fi
    printf '%s\n' "$f16_output"
    f16_counts=$(printf '%s\n' "$f16_output" | extract_attention_counts)

    if ! turbo_output=$(run_graph_check "$model" turbo3 turbo3 turbo "$work_dir/$arch/turbo3"); then
        printf '%s\n' "$turbo_output"
        exit 1
    fi
    printf '%s\n' "$turbo_output"
    turbo_counts=$(printf '%s\n' "$turbo_output" | extract_attention_counts)
    if [ -z "$f16_counts" ] || [ -z "$turbo_counts" ]; then
        printf '%s: graph checker did not emit attention counts\n' "$arch" >&2
        exit 1
    fi
    if [ "$f16_counts" != "$turbo_counts" ]; then
        printf '%s: f16/turbo3 attention counts differ\nf16:\n%s\nturbo3:\n%s\n' \
            "$arch" "$f16_counts" "$turbo_counts" >&2
        exit 1
    fi


done

# K-only DSA attention does not consume the requested V cache type. Exercise
# the V-only asymmetric path on a standard architecture with a distinct V cache.
arch=llama
fixture_dir="$work_dir/$arch/fixtures"
mkdir -p "$fixture_dir"
"$arch_fixture_bin" --arch "$arch" --seed 1 --out "$fixture_dir"
model="$fixture_dir/llama-dense.gguf"

if ! f16_output=$(run_graph_check "$model" f16 f16 non-turbo "$work_dir/$arch/f16"); then
    printf '%s\n' "$f16_output"
    exit 1
fi
printf '%s\n' "$f16_output"
f16_counts=$(printf '%s\n' "$f16_output" | extract_attention_counts)

if ! asymmetric_output=$(run_graph_check "$model" q8_0 turbo3 turbo "$work_dir/$arch/q8-turbo3"); then
    printf '%s\n' "$asymmetric_output"
    exit 1
fi
printf '%s\n' "$asymmetric_output"
asymmetric_counts=$(printf '%s\n' "$asymmetric_output" | extract_attention_counts)

if [ -z "$f16_counts" ] || [ -z "$asymmetric_counts" ]; then
    printf '%s: asymmetric graph checker did not emit attention counts\n' "$arch" >&2
    exit 1
fi
if [ "$f16_counts" != "$asymmetric_counts" ]; then
    printf '%s: f16/q8_0+turbo3 attention counts differ\nf16:\n%s\nq8_0+turbo3:\n%s\n' \
        "$arch" "$f16_counts" "$asymmetric_counts" >&2
    exit 1
fi
