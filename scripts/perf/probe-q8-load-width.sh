#!/usr/bin/env bash
# Compile-only probe: count LSC load message widths in the D=128 q8_0 flash-attention
# VEC kernels for Xe-HPG (acm-g10). No GPU execution, no benchmark.
#
# Purpose. Before this probe was written, both the canonical and the quants-first q8_0
# KV paths fetched their 4-byte quant words through ggml_sycl_memcpy_1<N, 2>. That
# alignment argument is the literal per-copy width, not a hint (see
# ggml/src/ggml-sycl/common.hpp, the nb_per_cpy dispatch): an argument of 2 emits two
# 16-bit loads per dword rather than one 32-bit load. That <N, 2> form describes the
# pre-change baseline, not the final state of both sites.
#
# The canonical AoS layout must keep the 2-byte copy: block_q8_0 is 34 bytes with qs at
# offset 2, so 4-byte alignment alternates across blocks. The quants-first layout has a 136-byte
# group stride and every payload offset is a multiple of 4, so its loads could be
# widened, and now are: that path is <N, 4> at HEAD.
#
# Usage:
#   scripts/perf/probe-q8-load-width.sh <build-dir> <label> [out-dir]
#
# Run it once at the unmodified tree and once with the alignment argument changed, then
# diff the two summaries. Absolute counts are meaningless in isolation because unroll
# factors dominate; only the paired delta carries information.
#
# Go/no-go: if the two runs emit the same send.ugm totals, IGC already coalesced the
# adjacent 16-bit loads, the hypothesis is dead at zero cost, and no A770 time should be
# spent on it.

set -euo pipefail

BUILD=${1:?usage: $0 <build-dir> <label> [out-dir]}
LABEL=${2:?usage: $0 <build-dir> <label> [out-dir]}
OUT=${3:-/tmp/q8-load-width}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CC_JSON="$BUILD/compile_commands.json"
WORK="$OUT/$LABEL"
DUMP="$WORK/igc"

[ -f "$CC_JSON" ] || { echo "error: $CC_JSON missing; configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq required" >&2; exit 1; }

rm -rf "$WORK"
mkdir -p "$DUMP"

# Reuse the real build's flags rather than hand-rolling them, so the probe cannot drift
# from the shipped configuration (GGML_SYCL_F16, warp size, device-code split, ...).
CANON_TU="ggml/src/ggml-sycl/template-instances/fattn-vec-instance-q8_0-q8_0.cpp"
BASE_CMD=$(jq -r --arg f "$CANON_TU" '.[] | select(.file | endswith($f)) | .command' "$CC_JSON" | head -1)
[ -n "$BASE_CMD" ] || { echo "error: no compile command for $CANON_TU" >&2; exit 1; }

# Strip everything up to and including the icpx token (not just a leading "icpx" prefix):
# compile_commands.json can wrap the compiler behind ccache or `cmake -E env`, and a plain
# leading-token strip leaves the wrapper/compiler in FLAGS, which then fails as bogus args
# to the bare `icpx $FLAGS` invocation below. Take the last token that is or ends in "/icpx",
# then keep everything after it.
COMPILER_FLAGS=$(printf '%s\n' "$BASE_CMD" \
    | awk '{
          idx = 0
          for (i = 1; i <= NF; i++) if ($i == "icpx" || $i ~ /\/icpx$/) idx = i
          if (idx == 0) next
          out = ""
          for (i = idx + 1; i <= NF; i++) out = out $i " "
          print out
      }')
[ -n "$COMPILER_FLAGS" ] || { echo "error: no icpx token found in compile command: $BASE_CMD" >&2; exit 1; }

# Strip the object-output and compile-only flags; keep includes, defines, -O, -std.
FLAGS=$(printf '%s\n' "$COMPILER_FLAGS" \
    | sed -e 's/ -o [^ ]*//g' -e 's/ -c [^ ]*//g' -e 's/ -MD//g' -e 's/ -MT [^ ]*//g' -e 's/ -MF [^ ]*//g')

# The quants-first kernels are instantiated implicitly from fattn.cpp, so no instance TU
# carries them. Emit a minimal one; it is the smallest carrier of the two quants-first
# load sites and, unlike the canonical instance, it is D=128 only.
QF_TU="$WORK/probe-qf-q8_0.cpp"
cat > "$QF_TU" <<EOF
#include "$REPO/ggml/src/ggml-sycl/fattn-vec.hpp"
template void ggml_sycl_flash_attn_ext_vec_case_q8_quants_first<128>(
    ggml_backend_sycl_context &, ggml_tensor *);
EOF

# icpx 2026.0 removed -fsycl-link=image, so AOT goes the long way round:
# device-only compile to LLVM IR, translate to SPIR-V, then let ocloc run IGC.
# SPV_INTEL_subgroups is required by the sub-group shuffles in the reductions; the
# default extension set omits it and llvm-spirv then emits an empty module.
LLVM_SPIRV=${LLVM_SPIRV:-/opt/sycl/bin/llvm-spirv}
SPIRV_EXTS=${SPIRV_EXTS:-+SPV_INTEL_subgroups}

compile_and_dump() {
    local tu=$1 tag=$2
    local d="$DUMP/$tag"
    mkdir -p "$d"
    echo "== $tag: $tu"
    # shellcheck disable=SC2086
    icpx $FLAGS -fsycl-device-only "$tu" -o "$WORK/$tag.bc" \
        > "$WORK/$tag.compile.log" 2>&1 \
        || { echo "  DEVICE COMPILE FAILED, see $WORK/$tag.compile.log"; tail -20 "$WORK/$tag.compile.log"; return 1; }
    "$LLVM_SPIRV" --spirv-ext="$SPIRV_EXTS" "$WORK/$tag.bc" -o "$WORK/$tag.spv" \
        >> "$WORK/$tag.compile.log" 2>&1
    [ -s "$WORK/$tag.spv" ] || { echo "  SPIRV TRANSLATION EMPTY, see $WORK/$tag.compile.log"; return 1; }
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$d" \
        ocloc compile -device acm-g10 -file "$WORK/$tag.spv" -spirv_input \
              -out_dir "$WORK/$tag.ocl" >> "$WORK/$tag.compile.log" 2>&1 \
        || { echo "  OCLOC FAILED, see $WORK/$tag.compile.log"; tail -10 "$WORK/$tag.compile.log"; return 1; }
    echo "  dumped $(find "$d" -name '*.asm' | wc -l) .asm files"
    # IGC reports spill per kernel in the ocloc warning stream; surface it, it is the
    # other thing these dumps are good for.
    grep -oE 'compiled SIMD[0-9]+ allocated [0-9]+ regs and spilled around [0-9]+' \
        "$WORK/$tag.compile.log" | sort | uniq -c | sed 's/^/    /' || true
}

summarize() {
    local tag=$1
    local d="$DUMP/$tag"
    echo "---- $tag ----"
    # SIMD16 entries are the VEC kernels (reqd_sub_group_size(16) in fattn-common.hpp).
    # Collect into an array via NUL-delimited find so paths containing whitespace survive.
    local -a asms=()
    while IFS= read -r -d '' f; do asms+=("$f"); done \
        < <(find "$d" -name '*simd16*entry*.asm' -print0 2>/dev/null)
    if [ ${#asms[@]} -eq 0 ]; then
        echo "  no simd16 entry dumps; falling back to all .asm"
        while IFS= read -r -d '' f; do asms+=("$f"); done \
            < <(find "$d" -name '*.asm' -print0 2>/dev/null)
    fi
    [ ${#asms[@]} -gt 0 ] || { echo "  no ISA dumps found"; return 0; }
    # Xe-HPG LSC mnemonics look like: load.ugm.d16u32.a64.ca.ca / load.ugm.d32x4.a64.ca.ca
    # d16u32 is the narrow form emitted by an alignment-2 copy: a 16-bit fetch
    # zero-extended into a 32-bit destination, two per dword.
    echo "  global (ugm) load data types:"
    grep -ohE 'load\.ugm\.d[0-9]+[a-z0-9]*' "${asms[@]}" 2>/dev/null \
        | sed 's/^load\.ugm\.//' | sort | uniq -c | sort -rn | sed 's/^/    /' || echo "    (none matched)"
    echo "  narrow (d16*) global loads: $(grep -ohE 'load\.ugm\.d16[a-z0-9]*' "${asms[@]}" 2>/dev/null | wc -l)"
    echo "  total ugm loads          : $(grep -ohE 'load\.ugm\.d[0-9]+[a-z0-9]*' "${asms[@]}" 2>/dev/null | wc -l)"
}

compile_and_dump "$REPO/$CANON_TU" canonical || true
compile_and_dump "$QF_TU" quants-first || true

{
    echo "probe label : $LABEL"
    echo "repo HEAD   : $(git -C "$REPO" rev-parse --short HEAD)"
    echo "dirty       : $(git -C "$REPO" status --porcelain -- ggml/src/ggml-sycl | wc -l) modified sycl files"
    echo "icpx        : $(icpx --version | head -1)"
    echo "ocloc       : $(ocloc --version 2>/dev/null | head -1 || echo n/a)"
    echo
    summarize canonical
    echo
    summarize quants-first
} | tee "$WORK/summary.txt"

echo
echo "summary written to $WORK/summary.txt"
echo "compare two labels with: diff $OUT/<labelA>/summary.txt $OUT/<labelB>/summary.txt"
