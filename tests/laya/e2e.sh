#!/usr/bin/env bash
# End-to-end reproduction of the laya pipeline.
#
#   safetensors -> F16 GGUF -> k-quant GGUF -> tools/laya inference
#
# Steps:
#   1. convert_hf_to_gguf.py            (F16 GGUF, ~9 s)
#   2. cmake build llama-laya-cli / llama-quantize
#   3. tests/laya/quantize.sh           (Q4_K_M / Q5_K_M / Q8_0 + tensor check)
#   4. tests/laya/verify_precision.py   (quant vs F16 vs PyTorch golden)
#   5. tests/laya/bench.py              (latency / batching / stability)
#
# All commands are idempotent and byte-reproducible; pass --skip-convert or
# --skip-quantize to reuse existing artifacts.
#
# Environment:
#   LAYA_MODEL_DIR   path to the laya-multilingual checkpoint directory
#   LAYA_PY          Python interpreter with the `laya` package (default: python3)
#
# Usage:  ./tests/laya/e2e.sh [--skip-convert] [--skip-quantize]
set -euo pipefail

cd "$(dirname "$0")/../.."   # llama.cpp repo root

MODEL_DIR="${LAYA_MODEL_DIR:?set LAYA_MODEL_DIR to the laya-multilingual checkpoint dir}"
PY="${LAYA_PY:-python3}"

skip_convert=0
skip_quantize=0
for a in "$@"; do
    case "$a" in
        --skip-convert)  skip_convert=1 ;;
        --skip-quantize) skip_quantize=1 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

echo "== [1/5] F16 GGUF =="
if [[ "$skip_convert" == 1 && -f laya-f16.gguf ]]; then
    echo "  laya-f16.gguf exists, skipping conversion"
else
    PYTHONPATH=gguf-py "$PY" convert_hf_to_gguf.py "$MODEL_DIR" \
        --outfile laya-f16.gguf --outtype f16
fi

echo "== [2/5] build tools =="
cmake -S . -B build -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
      -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-laya-cli llama-quantize -j "$(nproc)"

echo "== [3/5] quantize =="
if [[ "$skip_quantize" == 1 && -f laya-q4_k_m.gguf ]]; then
    echo "  quant artifacts exist, skipping"
else
    ./tests/laya/quantize.sh
fi

echo "== [4/5] precision regression =="
python3 tests/laya/verify_precision.py ./build/bin/llama-laya-cli

echo "== [5/5] performance / stability =="
python3 tests/laya/bench.py ./build/bin/llama-laya-cli --runs "${LAYA_BENCH_RUNS:-30}"

echo "done. artifacts: laya-f16.gguf laya-q4_k_m.gguf laya-q5_k_m.gguf laya-q8_0.gguf"
