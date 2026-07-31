#!/bin/bash
# Tessera C++ port integration test suite.
# Usage: bash tools/quantize/tessera/test_all.sh
# Run from the repository root.

set -e
PASS=0
FAIL=0
ERRORS=""

T=tools/quantize/tessera
C=common/tessera-debug
BIN=/tmp/tessera_test_bin
CXX="clang++ -std=c++17 -O2"

mkdir -p "$BIN"

run_test() {
    local name="$1"
    shift
    printf "  %-30s" "$name"
    if "$@" > /tmp/tessera_test_$name.log 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  $name: see /tmp/tessera_test_$name.log"
    fi
}

# compile_and_run <name> <source...> [extra flags...]
compile_and_run() {
    local name="$1"
    shift
    if ! $CXX "$@" -o "$BIN/$name" > /tmp/tessera_build_$name.log 2>&1; then
        printf "  %-30s" "$name"
        echo "FAIL (compile)"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  $name: compile error, see /tmp/tessera_build_$name.log"
        return
    fi
    run_test "$name" "$BIN/$name"
}

echo "Tessera integration tests"
echo ""

# --- Standalone (test + own module) ---
compile_and_run linalg      $T/test_linalg.cpp      $T/tessera-linalg.cpp
compile_and_run lbfgs       $T/test_lbfgs.cpp       $T/tessera-lbfgs.cpp
compile_and_run awq         $T/test_awq.cpp         $T/tessera-awq.cpp
compile_and_run l5          $T/test_l5.cpp          $T/tessera-l5.cpp
compile_and_run imatrix     $T/test_imatrix.cpp     $T/tessera-imatrix.cpp
compile_and_run corpus      $T/test_corpus.cpp      $T/tessera-corpus.cpp
compile_and_run ppl         $T/test_ppl.cpp         $T/tessera-ppl.cpp
compile_and_run ab_harness  $T/test_ab_harness.cpp  $T/tessera-ab-harness.cpp
compile_and_run higgs       $T/test_higgs.cpp       $T/tessera-higgs.cpp
compile_and_run regime      $T/test_regime.cpp      $T/tessera-regime.cpp
compile_and_run peqat       $T/test_peqat.cpp       $T/tessera-peqat.cpp

# --- Needs vec (Accelerate) ---
compile_and_run vec         $T/test_vec.cpp         $T/tessera-vec.cpp -framework Accelerate
compile_and_run quant       $T/test_quant.cpp       $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate
compile_and_run moe_shapes  $T/test_moe_shapes.cpp  $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate
# dispatch requires libgguf/libggml (full CMake build); skip in standalone mode
printf "  %-30s" "dispatch"
if [ -f build/ggml/src/libgguf.a ] || [ -f build/ggml/src/libgguf.dylib ]; then
    compile_and_run dispatch $T/test_dispatch.cpp $T/tessera-dispatch.cpp $T/tessera-quant.cpp $T/tessera-vec.cpp $T/tessera-awq.cpp -I ggml/include -I ggml/src -L build/ggml/src -lgguf -lggml -framework Accelerate
else
    echo "SKIP (needs CMake build for libgguf)"
fi

# --- Needs linalg + lbfgs ---
compile_and_run search      $T/test_search.cpp      $T/tessera-search.cpp $T/tessera-linalg.cpp $T/tessera-lbfgs.cpp

# --- HIGGS integration (higgs + cache + search + quant + vec) ---
compile_and_run higgs_integration $T/test_higgs_integration.cpp $T/tessera-higgs.cpp $T/tessera-higgs-cache.cpp $T/tessera-search.cpp $T/tessera-linalg.cpp $T/tessera-lbfgs.cpp $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate

# --- Needs sidecar + vec ---
compile_and_run l15         $T/test_l15.cpp         $T/tessera-l15.cpp $C/tessera-sidecar-v3.cpp $T/tessera-vec.cpp -I $C -framework Accelerate

# --- Needs vendor (nlohmann/json) ---
compile_and_run policy      $T/test_policy.cpp      $T/tessera-policy.cpp -I vendor

# --- common/tessera-debug ---
compile_and_run sidecar_v3  $C/test_sidecar_v3.cpp  $C/tessera-sidecar-v3.cpp -I $C

# --- CoreML bridge ---
compile_and_run coreml_bridge $T/test_coreml_bridge.cpp $T/tessera-coreml.cpp $T/tessera-coreml-builder.cpp $T/tessera-coreml-metadata.cpp -I ggml/include

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    printf "$ERRORS\n"
    exit 1
fi
