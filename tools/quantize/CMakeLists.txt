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
    # search / GA / regime / HIGGS
    tessera/tessera-search.cpp
    tessera/tessera-regime.cpp
    tessera/tessera-higgs.cpp
    tessera/tessera-higgs-cache.cpp
    # policy I/O + dispatch + GGUF writer
    tessera/tessera-policy.cpp
    tessera/tessera-dispatch.cpp
    tessera/tessera-gguf-writer.cpp
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
endif()
