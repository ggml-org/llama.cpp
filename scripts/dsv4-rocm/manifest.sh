#!/usr/bin/env bash
# Capture a compact, reproducible DSV4 benchmark environment manifest.
set -Eeuo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 OUT_DIR BENCH_BINARY MODEL_FIRST_SHARD" >&2
    exit 2
fi

OUT_DIR=$1
BENCH=$(readlink -f "$2")
MODEL=$(readlink -f "$3")
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/manifest.txt"

section() {
    printf '\n=== %s ===\n' "$1"
}

model_files=("$MODEL")
base=$(basename "$MODEL")
dir=$(dirname "$MODEL")
if [[ "$base" =~ ^(.*)-[0-9]{5}-of-[0-9]{5}\.gguf$ ]]; then
    prefix=${BASH_REMATCH[1]}
    mapfile -t model_files < <(find "$dir" -maxdepth 1 -type f -name "$prefix-?????-of-?????.gguf" -print | sort)
fi

if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$ROOT_DIR" diff --binary HEAD > "$OUT_DIR/source.patch"
    git -C "$ROOT_DIR" status --porcelain=v1 > "$OUT_DIR/source-status.txt"
    : > "$OUT_DIR/untracked-files.sha256"
    while IFS= read -r -d '' file; do
        sha256sum "$ROOT_DIR/$file" >> "$OUT_DIR/untracked-files.sha256"
    done < <(git -C "$ROOT_DIR" ls-files --others --exclude-standard -z)
fi

{
    section identity
    printf 'captured_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'user=%s\n' "$(id -un)"
    printf 'kernel=%s\n' "$(uname -srvmo)"
    hostnamectl 2>/dev/null || true

    section source
    printf 'root=%s\n' "$ROOT_DIR"
    git -C "$ROOT_DIR" status --short --branch 2>&1 || true
    git -C "$ROOT_DIR" log -1 --format='commit=%H%ncommit_date=%cI%nsubject=%s' 2>&1 || true
    git -C "$ROOT_DIR" remote -v 2>&1 || true
    if [ -f "$OUT_DIR/source.patch" ]; then
        sha256sum "$OUT_DIR/source.patch" "$OUT_DIR/source-status.txt" "$OUT_DIR/untracked-files.sha256"
    fi

    section binary
    printf 'bench=%s\n' "$BENCH"
    stat -c 'size=%s mtime=%y inode=%i path=%n' "$BENCH"
    sha256sum "$BENCH"
    readelf -n "$BENCH" 2>/dev/null | grep -E 'Build ID|Owner|NT_GNU' || true
    printf '%s\n' '-- resolved dependencies --'
    ldd "$BENCH" 2>&1 || true
    printf '%s\n' '-- all resolved dependency hashes (local + ROCm/system DSOs) --'
    while IFS= read -r library; do
        [ -f "$library" ] && sha256sum "$library"
    done < <(ldd "$BENCH" 2>/dev/null | awk '
        /=> \// { print $3 }
        /^\// { print $1 }
    ' | sort -u)
    cache=$(readlink -f "$(dirname "$BENCH")/../CMakeCache.txt" 2>/dev/null || true)
    if [[ -n "$cache" && -f "$cache" ]]; then
        printf '%s\n' '-- selected CMake cache --'
        sha256sum "$cache"
        grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER:|CMAKE_C_COMPILER:|GGML_HIP:|GGML_HIPBLAS:|AMDGPU_TARGETS:|CMAKE_HIP_ARCHITECTURES:|GGML_NATIVE:|GGML_LTO:)' "$cache" || true
    fi

    section model_files
    printf 'hash_mode=%s\n' "${DSV4_HASH_MODE:-metadata}"
    for file in "${model_files[@]}"; do
        stat -c 'size=%s mtime=%y inode=%i path=%n' "$file"
        if [[ ${DSV4_HASH_MODE:-metadata} == full ]]; then
            sha256sum "$file"
        fi
    done

    section model_metadata
    if [ -d "$ROOT_DIR/gguf-py/gguf" ]; then
        PYTHONPATH="$ROOT_DIR/gguf-py" python3 - "$MODEL" <<'PY' 2>&1 || true
import sys
from gguf import GGUFReader
reader = GGUFReader(sys.argv[1], mode="r")
for name in (
    "general.name", "general.architecture", "general.file_type",
    "split.count", "split.no", "split.tensors.count",
    "deepseek4.block_count", "deepseek4.context_length",
):
    field = reader.fields.get(name)
    if field is not None:
        print(f"{name}={field.contents()}")
print(f"header_tensors={len(reader.tensors)}")
PY
    fi

    section cpu_memory
    lscpu
    free -h
    df -hT "$ROOT_DIR" "$MODEL"

    section pci
    lspci -nn | grep -Ei 'VGA|Display|3D' || true

    section rocm
    hipcc --version 2>&1 | head -20 || true
    # This query does not execute the GPU-linked benchmark binary.
    rocm-smi --showproductname --showuniqueid --showmeminfo vram --showclocks --showpower --showmaxpower --showperflevel --showprofile --showoverdrive --showmemoverdrive --showtopo 2>&1 || true

    section environment
    env | grep -E '^(DSV4_|GGML_|HSA_|HIP_|ROCM_|OMP_|MALLOC_|LD_LIBRARY_PATH=)' | sort || true
} > "$OUT"

printf '%s\n' "$OUT"