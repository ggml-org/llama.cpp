#!/bin/bash
# Build libppu_gdn.so = FLA recurrent (AOT per-shape) + FLA chunked prefill (JIT-cubin per-shape), torch/python/
# triton-free (links only libcuda/libcudart). See ppu-gdn-so.h for the ABI. Point llama.cpp at it via
#   export GGML_PPU_GDN_SO=$PWD/libppu_gdn.so           # recurrent (default; decode + short prefill, correct)
#   export GGML_PPU_GDN_CHUNKED=1                        # OPT-IN chunked prefill (real L2-normed models only)
#
#   ./build.sh ["H,HV,S" ...]        (FLA_ROOT defaults to the thirdparty/flash-linear-attention submodule)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERE_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FLA_ROOT:=$HERE_/../../thirdparty/flash-linear-attention}"   # submodule by default
[ -d "$FLA_ROOT/fla" ] || { echo "FLA not found at $FLA_ROOT -- run: git submodule update --init thirdparty/flash-linear-attention"; exit 1; }
: "${CUDA_HOME:=/usr/local/cuda}"
SHAPES=("$@"); [ ${#SHAPES[@]} -eq 0 ] && SHAPES=("32,32,128" "16,16,64" "4,4,64" "4,8,64" "32,32,64" "2,4,128")

echo "[gdn] AOT recurrent kernels: ${SHAPES[*]}"
rm -rf "$HERE/aot"; FLA_ROOT="$FLA_ROOT" python3 "$HERE/aot_recurrent.py" "${SHAPES[@]}"

echo "[gdn] JIT-cubin chunked chain (per shape)"
FLA_ROOT="$FLA_ROOT" python3 "$HERE/gen_chunk_so.py" "${SHAPES[@]}"

echo "[gdn] linking libppu_gdn.so"
gcc -shared -fPIC -O2 -o "$HERE/libppu_gdn.so" \
    "$HERE"/aot/ppu_gdn_dispatch.c "$HERE"/aot/gdn_rec_*.c "$HERE"/aot_chunk_so/ppu_gdn_chunk.c \
    -I"$HERE/aot" -I"$CUDA_HOME/include" -lcuda -lcudart -L"$CUDA_HOME/lib64"

ldd "$HERE/libppu_gdn.so" | grep -iqE "torch|python|triton" && { echo "ERROR: not torch/python-free"; exit 1; }
echo "[gdn] done -> $HERE/libppu_gdn.so (torch/python/triton-free):"
nm -D "$HERE/libppu_gdn.so" | grep -E "ppu_gdn_(recurrent|chunked)" | awk '{print "  "$3}'
