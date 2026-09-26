#!/usr/bin/env bash
# Quantize the laya F16 GGUF with precision protection.
#
# Precision protection rules (kept F16/F32, never quantized):
#   token_embd, all *_norm.* (LayerNorm gamma/beta), type_emb, scorer.*, act_head.*
# Everything else (encoder blk.* matrices + decision-head transformer matrices)
# is quantized with the standard llama.cpp k-quant mixture.
#
# Requires: laya-f16.gguf in the llama.cpp repo root (see README.md for the
# conversion command) and a compiler with std::filesystem.
#
# Environment:
#   LAYA_PY   Python interpreter with gguf-py on PYTHONPATH (default: python3)
#
# Usage:  ./tests/laya/quantize.sh
set -euo pipefail

cd "$(dirname "$0")/../.."   # llama.cpp repo root

if [[ ! -f laya-f16.gguf ]]; then
    echo "error: laya-f16.gguf not found (run the conversion first)" >&2
    exit 1
fi

# ---- build llama-quantize ------------------------------------------------
if [[ ! -x build/bin/llama-quantize ]]; then
    cmake -S . -B build \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
    # GCC < 9 ships std::filesystem as a separate static lib; whole-archive so
    # the executable link resolves the symbols regardless of link order.
    cmake -S . -B build \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--whole-archive -lstdc++fs -Wl,--no-whole-archive"
    cmake --build build --target llama-quantize -j "$(nproc)"
fi

Q="build/bin/llama-quantize"
PROTECT=(
    --token-embedding-type f16
    --tensor-type 'type_emb\.weight=f16'
    --tensor-type 'scorer\..*\.weight=f16'
    --tensor-type 'act_head\..*\.weight=f16'
)

for ftype in Q4_K_M Q5_K_M Q8_0; do
    out="laya-$(echo "$ftype" | tr '[:upper:]' '[:lower:]').gguf"
    "$Q" "${PROTECT[@]}" laya-f16.gguf "$out" "$ftype" 8
done

echo "done. outputs: laya-q4_k_m.gguf laya-q5_k_m.gguf laya-q8_0.gguf"

# ---- verify precision protection + deviation ------------------------------
PYTHONPATH=gguf-py "${LAYA_PY:-python3}" tests/laya/verify_quantize.py .
