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
compile_and_run awq         $T/test_awq.cpp         $T/tessera-awq.cpp $T/tessera-policy.cpp -I vendor
compile_and_run l5          $T/test_l5.cpp          $T/tessera-l5.cpp
compile_and_run imatrix     $T/test_imatrix.cpp     $T/tessera-imatrix.cpp
compile_and_run corpus      $T/test_corpus.cpp      $T/tessera-corpus.cpp
compile_and_run ppl         $T/test_ppl.cpp         $T/tessera-ppl.cpp
compile_and_run ab_harness  $T/test_ab_harness.cpp  $T/tessera-ab-harness.cpp
compile_and_run acceptance  $T/test_acceptance.cpp  $T/tessera-acceptance.cpp $T/tessera-ab-harness.cpp
compile_and_run higgs       $T/test_higgs.cpp       $T/tessera-higgs.cpp
compile_and_run regime      $T/test_regime.cpp      $T/tessera-regime.cpp
compile_and_run peqat       $T/test_peqat.cpp       $T/tessera-peqat.cpp

# --- Needs vec (Accelerate) ---
compile_and_run vec         $T/test_vec.cpp         $T/tessera-vec.cpp -framework Accelerate
compile_and_run quant       $T/test_quant.cpp       $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate
compile_and_run w4a4        $T/test_w4a4.cpp        $T/tessera-w4a4.cpp $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate
compile_and_run moe_shapes  $T/test_moe_shapes.cpp  $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate
compile_and_run operative_routing $T/test_operative_routing.cpp $T/tessera-regime.cpp $T/tessera-quant.cpp $T/tessera-vec.cpp -framework Accelerate
# dispatch requires libgguf/libggml (full CMake build); skip in standalone mode
printf "  %-30s" "dispatch"
if [ -f build/ggml/src/libgguf.a ] || [ -f build/ggml/src/libgguf.dylib ]; then
    compile_and_run dispatch $T/test_dispatch.cpp $T/tessera-dispatch.cpp $T/tessera-quant.cpp $T/tessera-vec.cpp $T/tessera-awq.cpp -I ggml/include -I ggml/src -L build/ggml/src -lgguf -lggml -framework Accelerate
else
    echo "SKIP (needs CMake build for libgguf)"
fi

# --- Needs linalg + lbfgs (search pulls in vendor/nlohmann for the archive JSON) ---
compile_and_run search      $T/test_search.cpp      $T/tessera-lrq.cpp $T/tessera-dartquant.cpp $T/tessera-flrq.cpp $T/tessera-champq.cpp $T/tessera-archive.cpp $T/tessera-linalg.cpp $T/tessera-lbfgs.cpp -I vendor

# --- MAP-Elites archive (search + linalg + lbfgs + vendor/nlohmann) ---
compile_and_run map_elites  $T/test_map_elites.cpp  $T/tessera-lrq.cpp $T/tessera-dartquant.cpp $T/tessera-flrq.cpp $T/tessera-champq.cpp $T/tessera-archive.cpp $T/tessera-linalg.cpp $T/tessera-lbfgs.cpp -I vendor

# --- Modality as operative regime axis (regime + search + linalg + lbfgs + vendor) ---
compile_and_run modality_routing $T/test_modality_routing.cpp $T/tessera-regime.cpp $T/tessera-lrq.cpp $T/tessera-dartquant.cpp $T/tessera-flrq.cpp $T/tessera-champq.cpp $T/tessera-archive.cpp $T/tessera-linalg.cpp $T/tessera-lbfgs.cpp -I vendor

# --- HIGGS integration (higgs + cache + search + quant + vec) ---
compile_and_run higgs_integration $T/test_higgs_integration.cpp $T/tessera-higgs.cpp $T/tessera-higgs-cache.cpp $T/tessera-lrq.cpp $T/tessera-dartquant.cpp $T/tessera-flrq.cpp $T/tessera-champq.cpp $T/tessera-archive.cpp $T/tessera-linalg.cpp $T/tessera-lbfgs.cpp $T/tessera-quant.cpp $T/tessera-vec.cpp -I vendor -framework Accelerate

# --- Needs sidecar + vec ---
compile_and_run l15         $T/test_l15.cpp         $T/tessera-l15.cpp $C/tessera-sidecar-v3.cpp $T/tessera-vec.cpp -I $C -framework Accelerate

# --- Needs sidecar (L1 kernel-direct fitness) ---
compile_and_run l1_fitness  $T/test_l1_fitness.cpp  $T/tessera-l1-fitness.cpp $C/tessera-sidecar-v3.cpp -I $C

# --- L2-L5 runtime-aware pipeline (L2 diff + L3 coherence + L5 adaptive) ---
# L2 needs vendor/nlohmann (JSON report); L3 needs the sidecar reader.
compile_and_run l2l5        $T/test_l2l5.cpp        $T/tessera-l2-diff.cpp $T/tessera-l3-coherence.cpp $T/tessera-l5.cpp $T/tessera-ppl.cpp $C/tessera-sidecar-v3.cpp -I vendor -I $C

# --- Needs vendor (nlohmann/json) ---
compile_and_run policy      $T/test_policy.cpp      $T/tessera-policy.cpp -I vendor

# --- Self-improving capability loop (capability-eval + adapt; needs vendor) ---
compile_and_run capability_loop $T/test_capability_loop.cpp $T/tessera-capability-eval.cpp $T/tessera-adapt.cpp -I vendor

# --- Tier-2 anonymizer (needs vendor/nlohmann for the de-anonymization map) ---
compile_and_run anonymizer    $T/test_anonymizer.cpp  $T/tessera-anonymizer.cpp -I vendor

# --- Text secret redactor (standalone; no vendor needed) ---
compile_and_run scrub         $T/test_scrub.cpp       $T/tessera-scrub.cpp

# --- North-star throughput harness (needs vendor/nlohmann for workload+receipt JSON) ---
compile_and_run throughput    $T/test_throughput.cpp  $T/tessera-throughput.cpp -I vendor

# --- Drafter training pipeline: LK loss (pure math, no vendor) ---
compile_and_run lk_loss       $T/test_lk_loss.cpp     $T/tessera-lk-loss.cpp

# --- Drafter training pipeline: dataset prep (needs vendor/nlohmann + dpace weights) ---
compile_and_run dataset       $T/test_dataset.cpp     $T/tessera-dataset.cpp $T/tessera-dpace.cpp -I vendor

# --- Drafter training pipeline: D-PACE loss (pure math, no vendor) ---
compile_and_run dpace         $T/test_dpace.cpp       $T/tessera-dpace.cpp

# --- Drafter training pipeline: offline feature-capture file format (needs vendor) ---
compile_and_run features      $T/test_features.cpp    $T/tessera-features.cpp -I vendor

# --- Drafter training pipeline: LK training-data builder (needs vendor + lk-loss densify) ---
compile_and_run lk_train_data $T/test_lk_train_data.cpp $T/tessera-lk-train-data.cpp $T/tessera-lk-loss.cpp -I vendor

# --- common/tessera-debug ---
compile_and_run sidecar_v3  $C/test_sidecar_v3.cpp  $C/tessera-sidecar-v3.cpp -I $C

# L1 sidecar writer end-to-end (needs a stub tessera-build-info.h)
printf '#pragma once\n#define TESSERA_KERNEL_VERSION "test"\n#define TESSERA_MAIN_TIP "test"\n' > "$BIN/tessera-build-info.h"
compile_and_run l1_sidecar  $T/test_l1_sidecar.cpp  $C/tessera-debug.cpp $C/tessera-sidecar-v3.cpp -I $C -I "$BIN"

# --- CoreML bridge (builder pulls in the MIL + telemetry modules) ---
compile_and_run coreml_bridge $T/test_coreml_bridge.cpp $T/tessera-coreml.cpp $T/tessera-coreml-builder.cpp $T/tessera-coreml-metadata.cpp $T/tessera-coreml-mil.cpp $T/tessera-coreml-telemetry.cpp -I ggml/include

# --- CoreML MIL builder + weight serialization + IOReport telemetry scaffold ---
compile_and_run coreml_mil $T/test_coreml_mil.cpp $T/tessera-coreml-mil.cpp $T/tessera-coreml-telemetry.cpp $T/tessera-coreml-builder.cpp $T/tessera-coreml.cpp -I ggml/include

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    printf "$ERRORS\n"
    exit 1
fi
