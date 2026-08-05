# llama-quantize-impl: quantize logic, reusable by app

set(TARGET llama-quantize-impl)

# ---- DuckDB amalgamation (persistent quantize store) ----
# Vendored at third-party/duckdb/{duckdb.cpp,duckdb.hpp}. The amalgamation is
# a single ~25 MB translation unit; it compiles slowly (~2 min) but produces a
# self-contained library with no external runtime deps. Compiled as a static
# OBJECT library so it links cleanly into llama-quantize-impl and is not
# rebuilt on every quantize-impl change.
#
# TESSERA_DISABLE_DUCKDB (env or cache variable) skips compiling the
# amalgamation; the dispatch still builds, with --quantize-db a no-op. Useful
# for fast iteration cycles where the DB store is not being exercised.
set(TS_DUCKDB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/duckdb)
if(EXISTS ${TS_DUCKDB_DIR}/duckdb.cpp AND EXISTS ${TS_DUCKDB_DIR}/duckdb.hpp)
    set(TS_DUCKDB_AVAILABLE ON)
else()
    set(TS_DUCKDB_AVAILABLE OFF)
endif()
if(DEFINED ENV{TESSERA_DISABLE_DUCKDB})
    set(TS_DUCKDB_AVAILABLE OFF)
endif()
if(TS_DUCKDB_AVAILABLE)
    add_library(duckdb-amalgamation OBJECT
        ${TS_DUCKDB_DIR}/duckdb.cpp)
    target_include_directories(duckdb-amalgamation PUBLIC ${TS_DUCKDB_DIR})
    # The amalgamation is internally a single TU; give it lots of memory and
    # disable some pedantic warnings it does not pass.
    target_compile_options(duckdb-amalgamation PRIVATE
        -w
        $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-deprecated-declarations>)
    # Use the biggest reasonable optimization for the amalgamation: it is a
    # runtime hot path (Appender flushes, query parsing) and a -O0 build is
    # measurably slower at insert time. Falls back to the parent CMAKE flags
    # when CMAKE_BUILD_TYPE is unset.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(duckdb-amalgamation PRIVATE -O2)
    endif()
    message(STATUS "Tessera: DuckDB amalgamation enabled "
                   "(${TS_DUCKDB_DIR}/duckdb.cpp)")
else()
    message(STATUS "Tessera: DuckDB amalgamation not found "
                   "(--quantize-db will be a no-op)")
endif()

add_library(${TARGET}
    quantize.cpp
    # core numeric
    tessera/tessera-vec.cpp
    tessera/tessera-linalg.cpp
    tessera/tessera-lbfgs.cpp
    tessera/tessera-quant.cpp
    tessera/tessera-w4a4.cpp
    # calibration + imatrix + corpus
    tessera/tessera-imatrix.cpp
    tessera/tessera-corpus.cpp
    tessera/tessera-awq.cpp
    tessera/tessera-awq-fitness.cpp
    # search / GA / regime / HIGGS
    tessera/tessera-lrq.cpp
    tessera/tessera-dartquant.cpp
    tessera/tessera-flrq.cpp
    tessera/tessera-champq.cpp
    # SEPTQ banded-Cholesky Hessian weight quantizer (B3)
    tessera/tessera-septq.cpp
    tessera/tessera-archive.cpp
    tessera/tessera-regime.cpp
    tessera/tessera-higgs.cpp
    tessera/tessera-higgs-cache.cpp
    # HIGGS structural proxy (C++ first-class port of the
    # tools/ane-mtp/estimate_higgs_alpha.py sidecar). Phase 3.5
    # of docs/tessera-ane-ios-demo-design.md: the proxy is the
    # default "L1-agnostic by design" t_l^2 -> alpha_l estimator;
    # the family-prior table mirrors the NumPy FAMILY_PRIOR dict
    # byte-for-byte so a sidecar produced by the C++ binary is
    # interchangeable with one produced by the Python path.
    tessera/tessera-higgs-proxy.cpp
    # policy I/O + dispatch + GGUF writer
    tessera/tessera-policy.cpp
    tessera/tessera-dispatch.cpp
    tessera/tessera-gguf-writer.cpp
    # structured progress reporting (terminal live line + NDJSON for UI)
    tessera/tessera-progress.cpp
    # probes + harnesses
    tessera/tessera-l5.cpp
    tessera/tessera-l15.cpp
    tessera/tessera-l1-fitness.cpp
    tessera/tessera-ppl.cpp
    tessera/tessera-l2-diff.cpp
    tessera/tessera-l3-coherence.cpp
    tessera/tessera-ab-harness.cpp
    tessera/tessera-acceptance.cpp
    tessera/tessera-peqat.cpp
    # self-improving capability loop (section 4.7 multi-axis eval + adaptation engine)
    tessera/tessera-capability-eval.cpp
    tessera/tessera-adapt.cpp
    # tier-2 escalation anonymizer (Phase 5)
    tessera/tessera-anonymizer.cpp
    # text secret redactor (tier-2 egress privacy, R4)
    tessera/tessera-scrub.cpp
    # north-star batched-throughput workload harness
    tessera/tessera-throughput.cpp
    # drafter training pipeline: LK loss + dataset preparation + D-PACE
    tessera/tessera-lk-loss.cpp
    tessera/tessera-dataset.cpp
    tessera/tessera-dpace.cpp
    # drafter training pipeline: LK training-data builder (traces -> dense labels)
    tessera/tessera-lk-train-data.cpp
    # drafter training pipeline: DFlash training-data builder (block.v1 -> weighted CE labels)
    tessera/tessera-dflash-train-data.cpp
    # drafter training pipeline: offline trunk-feature capture file format
    tessera/tessera-features.cpp
    # multimodal
    tessera/tessera-mm-imatrix.cpp
    tessera/tessera-mm-awq.cpp
    tessera/tessera-mm-fitness.cpp
    # CoreML conversion bridge
    tessera/tessera-coreml.cpp
    tessera/tessera-coreml-builder.cpp
    tessera/tessera-coreml-metadata.cpp
    tessera/tessera-coreml-mil.cpp
    tessera/tessera-coreml-telemetry.cpp
    # shared sidecar v3 (lives in common/, needed by tessera-l15)
    ../../common/tessera-debug/tessera-sidecar-v3.cpp
    # DuckDB-backed persistent store (no-op when --quantize-db is not passed)
    tessera/tessera-quantize-db.cpp
    # Phase 16 unified GGUF writer for the Gemma4 12B + dspark + dflash +
    # MTP arch (LLM_ARCH_GEMMA4_ASSISTANT). Reads 4+ per-component
    # GGUFs + a calibration policy JSON, writes a single gemma4-assistant
    # GGUF. See tessera-unified-writer.h.
    tessera/tessera-unified-writer.cpp
    # Per-table write buffer: parallel producers -> single flusher -> DuckDB
    # bulk insert. Used for the unified tessera.duckdb schema (ga_evaluations,
    # tensor_stats, l3/l4/l5/ple summary tables).
    tessera/tessera-db-buffer.cpp
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tessera)

# ggml internal headers (ggml-quants.h, ggml-quants-v2.h,
# ggml-quants-v2-dispatch.h): tessera-higgs-proxy.cpp drives the same
# TILE640 quant/dequant dispatch the GGML_OP_TILE640_MATMUL inference
# path uses. PRIVATE: the headers never leak into downstream consumers.
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/src)

# Pull the duckdb include path through the amalgamation target when present,
# so tessera-quantize-db.cpp can find duckdb.hpp. The OBJECT library's
# sources are linked in by the $<TARGET_OBJECTS> below.
if(TS_DUCKDB_AVAILABLE)
    target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:duckdb-amalgamation>)
    target_link_libraries(${TARGET} PUBLIC duckdb-amalgamation)
endif()

target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(APPLE)
    target_link_libraries(${TARGET} PUBLIC "-framework Accelerate")

    # ---- Metal acceleration for the Tessera quantize pipeline ----
    # Build wiring (only on Apple; non-Apple builds use the vDSP/scalar
    # fallback transparently via the ts_metal_available() runtime check):
    #   1. compile tessera-metal.metal -> tessera-metal.metallib
    #   2. regenerate tessera-metal-source.h (embedded source fallback)
    #   3. add the Objective-C++ host dispatch (tessera-metal.mm)
    #   4. link Metal.framework + Foundation.framework
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

    find_package(Python3 COMPONENTS Interpreter)

    set(TS_METAL_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/tessera)
    set(TS_METAL_SOURCE   ${TS_METAL_DIR}/tessera-metal.metal)
    set(TS_METAL_LIB      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tessera-metal.metallib)
    set(TS_METAL_SRC_H    ${CMAKE_CURRENT_BINARY_DIR}/tessera-metal-source.h)
    set(TS_METAL_EMBED_PY ${CMAKE_CURRENT_SOURCE_DIR}/tessera/embed_metal_source.py)

    # compile the .metal to a .metallib at build time. A successful build
    # means the kernels are valid Metal.
    add_custom_command(
        OUTPUT ${TS_METAL_LIB}
        COMMAND xcrun -sdk macosx metal -O3 -std=metal3.0
                -c ${TS_METAL_SOURCE} -o -
            | xcrun -sdk macosx metallib - -o ${TS_METAL_LIB}
        DEPENDS ${TS_METAL_SOURCE}
        COMMENT "Compiling Tessera Metal kernels"
        VERBATIM
    )
    add_custom_target(tessera-metal-lib ALL DEPENDS ${TS_METAL_LIB})
    add_dependencies(${TARGET} tessera-metal-lib)

    # regenerate the embedded-source fallback header whenever the .metal changes.
    if(Python3_Interpreter_FOUND)
        set(TS_PY ${Python3_EXECUTABLE})
    else()
        set(TS_PY python3)
    endif()
    add_custom_command(
        OUTPUT ${TS_METAL_SRC_H}
        COMMAND ${TS_PY} ${TS_METAL_EMBED_PY}
                ${TS_METAL_SOURCE} ${TS_METAL_SRC_H}
        DEPENDS ${TS_METAL_SOURCE} ${TS_METAL_EMBED_PY}
        COMMENT "Embedding Tessera Metal source string"
        VERBATIM
    )
    target_sources(${TARGET} PRIVATE ${TS_METAL_SRC_H})
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_sources(${TARGET} PRIVATE tessera/tessera-metal.mm)
    target_link_libraries(${TARGET} PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_LIBRARY})
    set_source_files_properties(${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tessera-metal.metallib
                                PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    # install the metallib next to the binary so ts_metal_init can find it
    if(LLAMA_TOOLS_INSTALL)
        install(FILES ${TS_METAL_LIB} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
else()
    # Non-Apple: link the no-op stub so ts_quantize_2d's Metal-gated calls
    # resolve and the build still works (ts_metal_available() returns 0).
    target_sources(${TARGET} PRIVATE tessera/tessera-metal-stub.cpp)
endif()

if(UNIX AND NOT APPLE)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(${TARGET} PUBLIC ${BLAS_LIBRARIES})
    else()
        message(STATUS "BLAS not found - tessera will use scalar fallback")
    endif()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-tessera executable (Tier 2: subcommand-aware single binary)
# Replaces the legacy llama-quantize binary. The hard break is:
#   - All --tessera-* / --calib-* flags are removed.
#   - Use the subcommand syntax: `llama-tessera <subcommand> [flags]`
#   - For the main quantize path, omit the subcommand: `llama-tessera
#     model.gguf TESSERA_T640` (legacy positional) or
#     `llama-tessera --model model.gguf TESSERA_T640` (new).
# See common/arg.cpp for the subcommand table and docs/tier2-subcommand-design.md
# for the per-flag migration map.

set(TARGET llama-tessera)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# tessera-train-lk: native LK (acceptance-rate) drafter training driver.
# Reads spec_calib.v2 traces and trains a drafter with GGML_OPT_LOSS_TYPE_LK.
# See docs/tessera-lk-training-design.md.

set(TARGET tessera-train-lk)

add_executable(${TARGET} tessera/tessera-train-lk.cpp)
target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# tessera-higgs-proxy: native HIGGS structural proxy. CLI entry point for
# the ane.alpha-coefficients.v1 sidecar producer. The Python
# tools/tessera/estimate_higgs_alpha.py wrapper subprocesses this binary
# when present on PATH and falls back to the in-process NumPy path when
# the binary is not built. See docs/tessera-ane-ios-demo-design.md
# Phase 3.5 for the design and docs/tessera-higgs-estimator.md for the
# sidecar shape.

set(TARGET tessera-higgs-proxy)

add_executable(${TARGET} tessera/tessera-higgs-proxy-main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# tessera-train-dflash: native DFlash / D-PACE block-drafter training driver.
# Reads llama.tessera.dflash-block.v1 records (from tessera-dataset
# --tessera-dataset-mode dflash) and trains a drafter with weighted
# cross-entropy. The D-PACE weight is the per-position CE scale written at
# the llama-layer label fill (the additive change to src/llama-context.cpp);
# the ggml CE graph is unchanged. See docs/tessera-dflash-training-design.md.

set(TARGET tessera-train-dflash)

add_executable(${TARGET} tessera/tessera-train-dflash.cpp)
target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# tessera-metal: standalone correctness test for the Metal acceleration path.
# Compares the three Metal kernels (scale_clip_ternarize, awq_grid_search,
# dequant_mse_recon) against the CPU reference. macOS-only; the test exits 0
# with a SKIP message when no Metal device is present.
if(APPLE)
    set(TARGET test-tessera-metal)
    add_executable(${TARGET} tessera/test_metal.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TARGET} COMMAND ${TARGET})
endif()

# tessera-ffi: static library for the Tessera Studio xcframework.
# Contains the real FFI entry points; links the full engine via
# llama-quantize-impl. Built with BUILD_SHARED_LIBS=OFF by the
# TesseraStudio/scripts/build-xcframework.sh script.

set(TARGET tessera-ffi)

add_library(${TARGET} STATIC
    ${CMAKE_SOURCE_DIR}/TesseraStudio/ffi/tessera_ffi.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/TesseraStudio/Sources/CTesseraFFI
)

target_link_libraries(${TARGET} PUBLIC llama-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Regression test: GGUF writer data-lifetime (see tessera-gguf-writer.h).
# The cluster descriptors reference the quant-result buffers by data pointer,
# so those buffers must outlive gguf_write_to_file. This writes a real GGUF,
# tears down the write side, reads it back, and byte-compares the payload.
if(LLAMA_BUILD_TESTS)
    set(TARGET test-tessera-gguf-write)
    add_executable(${TARGET} tessera/test_gguf_write_lifetime.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # DuckDB persistent-store test: schema, Appender bulk insert, warm-start
    # cross-run lookup, and resumability (list_converged_for_model). Standalone
    # binary so it can run without a GGUF; links duckdb-amalgamation directly
    # plus the DB module (does not need the rest of the dispatch).
    set(TARGET test-tessera-quantize-db)
    add_executable(${TARGET} tessera/test_quantize_db.cpp
                                  tessera/tessera-quantize-db.cpp
                                  tessera/tessera-db-buffer.cpp)
    target_link_libraries(${TARGET} PRIVATE
        llama-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if(TS_DUCKDB_AVAILABLE)
        target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:duckdb-amalgamation>)
        target_link_libraries(${TARGET} PRIVATE duckdb-amalgamation)
    endif()
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # Phase 16.7: per-component (model_role, name) covering
    # index + migration audit sidecar. Verifies the 7
    # CREATE INDEX IF NOT EXISTS lines land on a fresh open
    # and survive re-opens, runs a smoke benchmark on the
    # per-component query, and confirms the model_role
    # migration writes the sidecar next to the duckdb file.
    set(TARGET test-tessera-db-indexes)
    add_executable(${TARGET} tessera/test_tessera_db_indexes.cpp
                                  tessera/tessera-quantize-db.cpp
                                  tessera/tessera-db-buffer.cpp)
    target_link_libraries(${TARGET} PRIVATE
        llama-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if(TS_DUCKDB_AVAILABLE)
        target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:duckdb-amalgamation>)
        target_link_libraries(${TARGET} PRIVATE duckdb-amalgamation)
    endif()
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # Per-table write buffer: parallel producer threads, single flusher
    # thread, count + time + explicit flush triggers, sync-on-exit. Standalone
    # test that builds against the DuckDB amalgamation + the buffer module.
    set(TARGET test-tessera-db-buffer)
    add_executable(${TARGET} tessera/test_db_buffer.cpp
                                  tessera/tessera-quantize-db.cpp
                                  tessera/tessera-db-buffer.cpp)
    target_link_libraries(${TARGET} PRIVATE
        llama-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if(TS_DUCKDB_AVAILABLE)
        target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:duckdb-amalgamation>)
        target_link_libraries(${TARGET} PRIVATE duckdb-amalgamation)
    endif()
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # End-to-end DuckDB integration test through the dispatch: cold start,
    # warm-start (cross-run family seed), and resume (skip converged). Builds
    # a synthetic GGUF and runs the full pipeline against a real DuckDB file.
    set(TARGET test-tessera-quantize-db-e2e)
    add_executable(${TARGET} tessera/test_quantize_db_e2e.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # Phase 16 follow-up: C++ GA-walk model_role stamp on tensor_stats.
    # The dispatch's GA-prep walk (tessera-dispatch.cpp) now reads
    # ts_dispatch_params.model_role and stamps the value on every
    # ts_tessera_db_upsert_tensor_stat call. This test pins the
    # contract end-to-end: ts_dispatch_params.model_role defaults
    # to empty (legacy single-component contract), accepts the
    # Phase 16 roles (dflash / dspark / mtp_nextn / shared_embd),
    # and the upsert with each role stamps the corresponding row.
    # Same (model_hash, name) with two model_role values coexist
    # on the new (model_hash, model_role, name) PK. Standalone
    # (no libgguf / libggml needed) so it runs in the standalone
    # test_all.sh too.
    set(TARGET test-tessera-ga-model-role)
    add_executable(${TARGET} ../../tests/test-tessera-ga-model-role.cpp
                                  tessera/tessera-quantize-db.cpp
                                  tessera/tessera-db-buffer.cpp)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tessera)
    target_link_libraries(${TARGET} PRIVATE
        llama-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if(TS_DUCKDB_AVAILABLE)
        target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:duckdb-amalgamation>)
        target_link_libraries(${TARGET} PRIVATE duckdb-amalgamation)
    endif()
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # L5 adaptive requantize loop integration test: builds a tiny synthetic
    # GGUF with two 2D weights (attn + FFN), runs the full dispatch pipeline
    # with adaptive_requantize enabled, and asserts the L2->L5->re-quantize
    # loop ran and emitted the l5-loop.v1 report. Mirrors the lifetime test's
    # link against llama-quantize-impl (which transitively brings in the L2/L5
    # modules + libgguf/libggml).
    set(TARGET test-tessera-l5-dispatch)
    add_executable(${TARGET} tessera/test_l5_dispatch.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # Phase 3.5 HIGGS structural proxy C++ test suite: family
    # classification, structural alpha, uniform fallback,
    # measurement-function swap, byte-equivalence with the
    # NumPy sidecar, GGUF round-trip, model_hash, JSON I/O.
    # 30+ tests, runs in <1s on a tiny synthetic GGUF.
    set(TARGET test-tessera-higgs-proxy)
    add_executable(${TARGET} tessera/test_higgs_proxy.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TARGET} COMMAND ${TARGET})

    # Unified GGUF writer (Phase 16): round-trip + CLI flow for the
    # gemma4-assistant arch. Builds against llama-quantize-impl
    # (which transitively brings in the writer + the DuckDB-backed
    # per-component qtype reader).
    set(TARGET test-unified-writer)
    add_executable(${TARGET} tessera/test_unified_writer.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TARGET} COMMAND ${TARGET})
endif()
