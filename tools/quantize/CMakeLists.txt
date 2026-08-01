# llama-quantize-impl: quantize logic, reusable by app

set(TARGET llama-quantize-impl)

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
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tessera)
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

# llama-quantize executable

set(TARGET llama-quantize)

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
endif()
