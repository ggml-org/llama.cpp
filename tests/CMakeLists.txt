llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# SYCL tests include internal ggml headers under ggml/src.
if (GGML_SYCL)
    include_directories(${PROJECT_SOURCE_DIR}/ggml/src)
endif()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW
endif()

llama_build_and_test(test-chat-parser.cpp)
llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-json-partial.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-unicode.cpp
    peg-parser/testing.h
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    llama_build_and_test(test-thread-safety.cpp ARGS -hf ggml-org/models -hff tinyllamas/stories15M-q4_0.gguf -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
else()
    llama_build_and_test(test-thread-safety.cpp ARGS -hf ggml-org/models -hff tinyllamas/stories15M-be.Q4_0.gguf -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
endif()

# this fails on windows (github hosted runner) due to curl DLL not found (exit code 0xc0000135)
if (NOT WIN32)
    llama_build_and_test(test-arg-parser.cpp)
endif()

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_build_and_test(test-autorelease.cpp        LABEL "model")
llama_build_and_test(test-sycl-model-repro.cpp   LABEL "model")
if (GGML_SYCL_ASAN_PARTIAL)
    target_link_options(test-sycl-model-repro PRIVATE "-fsanitize=address")
endif()

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -hf ggml-org/models -hff tinyllamas/stories15M-q4_0.gguf)
else()
    llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -hf ggml-org/models -hff tinyllamas/stories15M-be.Q4_0.gguf)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

llama_build_and_test(test-alloc.cpp)
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-tensor-data.cpp)

# test-kv-block: tests for PagedAttention block management (links against llama for internal access)
add_executable(test-kv-block test-kv-block.cpp ${PROJECT_SOURCE_DIR}/src/llama-kv-block.cpp)
target_include_directories(test-kv-block PRIVATE ${PROJECT_SOURCE_DIR}/src)
install(TARGETS test-kv-block RUNTIME)
add_test(NAME test-kv-block COMMAND $<TARGET_FILE:test-kv-block>)

# test-tuning-engine: unit tests for TuningCache and related data structures
# Pure C++ test (no SYCL required), but needs access to tuning-engine.hpp
add_executable(test-tuning-engine test-tuning-engine.cpp)
target_include_directories(test-tuning-engine PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-tuning-engine RUNTIME)
add_test(NAME test-tuning-engine COMMAND $<TARGET_FILE:test-tuning-engine>)

# test-tuning-engine-impl: unit tests for TuningEngine implementation (progressive tuning)
# Pure C++ test (no SYCL required), needs access to tuning-engine.hpp and tuning-engine-impl.hpp
add_executable(test-tuning-engine-impl
    test-tuning-engine-impl.cpp
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/tuning-engine.cpp)
target_include_directories(test-tuning-engine-impl PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-tuning-engine-impl RUNTIME)
add_test(NAME test-tuning-engine-impl COMMAND $<TARGET_FILE:test-tuning-engine-impl>)

# test-observation-recording: unit tests for observation recording and statistics
# Tests: TuningObservation, ObservationStats, ObservationQueue, confidence tracking
# Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.9)
add_executable(test-observation-recording
    test-observation-recording.cpp
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/tuning-engine.cpp)
target_include_directories(test-observation-recording PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-observation-recording RUNTIME)
add_test(NAME test-observation-recording COMMAND $<TARGET_FILE:test-observation-recording>)

# test-op-context: unit tests for OperationContext builder (extracts M,N,K from tensors)
# Pure C++ test (no SYCL required), uses ggml for tensor creation
add_executable(test-op-context test-op-context.cpp)
target_link_libraries(test-op-context PRIVATE ggml)
target_include_directories(test-op-context PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-op-context RUNTIME)
add_test(NAME test-op-context COMMAND $<TARGET_FILE:test-op-context>)

# test-dispatch: unit tests for simplified unified kernel dispatch
# Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.17)
# Pure C++ test (no SYCL runtime required), tests dispatch helper functions
add_executable(test-dispatch test-dispatch.cpp)
target_link_libraries(test-dispatch PRIVATE ggml)
target_include_directories(test-dispatch PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-dispatch RUNTIME)
add_test(NAME test-dispatch COMMAND $<TARGET_FILE:test-dispatch>)

# test-heat-classifier: unit tests for tensor heat classification (HOT/WARM/COLD/FROZEN)
# Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.11)
# Pure C++ test (no SYCL required), uses ggml for tensor structure
add_executable(test-heat-classifier test-heat-classifier.cpp)
target_link_libraries(test-heat-classifier PRIVATE ggml)
target_include_directories(test-heat-classifier PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-heat-classifier RUNTIME)
add_test(NAME test-heat-classifier COMMAND $<TARGET_FILE:test-heat-classifier>)

# test-kv-cache-coordinator: unit tests for KV cache memory coordination
# Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.15)
# Pure C++ test (no SYCL required), tests KV cache memory tracking and coordination
add_executable(test-kv-cache-coordinator test-kv-cache-coordinator.cpp)
target_include_directories(test-kv-cache-coordinator PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-kv-cache-coordinator RUNTIME)
add_test(NAME test-kv-cache-coordinator COMMAND $<TARGET_FILE:test-kv-cache-coordinator>)

# test-layout-scheduler: unit tests for layout conversion scheduling
# Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.16)
# Pure C++ test (no SYCL required), tests idle detection, batching, and priority
add_executable(test-layout-scheduler test-layout-scheduler.cpp)
target_link_libraries(test-layout-scheduler PRIVATE ggml)
target_include_directories(test-layout-scheduler PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-layout-scheduler RUNTIME)
add_test(NAME test-layout-scheduler COMMAND $<TARGET_FILE:test-layout-scheduler>)

# test-ab-validation: unit tests for A/B validation mode infrastructure
# Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.21)
# Pure C++ test (no SYCL required), tests validation utilities for comparing kernel outputs
add_executable(test-ab-validation test-ab-validation.cpp)
target_include_directories(test-ab-validation PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-ab-validation RUNTIME)
add_test(NAME test-ab-validation COMMAND $<TARGET_FILE:test-ab-validation>)

# test-crossover-discovery: unit tests for unified/oneDNN crossover discovery
# Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.20)
# Pure C++ test (no SYCL required), tests binary search for batch size crossover points
add_executable(test-crossover-discovery test-crossover-discovery.cpp)
target_include_directories(test-crossover-discovery PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-crossover-discovery RUNTIME)
add_test(NAME test-crossover-discovery COMMAND $<TARGET_FILE:test-crossover-discovery>)

# test-edge-cases: unit tests for EdgeCaseSelfTest dimension edge case coverage
# Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.22)
# Pure C++ test (no SYCL required), tests edge case configuration generation
add_executable(test-edge-cases test-edge-cases.cpp)
target_include_directories(test-edge-cases PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
install(TARGETS test-edge-cases RUNTIME)
add_test(NAME test-edge-cases COMMAND $<TARGET_FILE:test-edge-cases>)

# test-scalar-fallback: unit tests for scalar fallback path in unified kernel
# Tests should_use_scalar_fallback, is_partial_tile, reference matmul
# Requires SYCL headers for unified-kernel.hpp types (sycl::half, etc.)
if(GGML_SYCL)
    add_executable(test-scalar-fallback test-scalar-fallback.cpp)
    target_include_directories(test-scalar-fallback PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-scalar-fallback PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-scalar-fallback PRIVATE "-fsycl")
    install(TARGETS test-scalar-fallback RUNTIME)
    add_test(NAME test-scalar-fallback COMMAND $<TARGET_FILE:test-scalar-fallback>)
endif()

# test-fattn-mask: unit tests for Flash Attention mask slope calculation (SYCL unified KV fix)
llama_build_and_test(test-fattn-mask.cpp)

# test-fattn-determinism: unit tests for Flash Attention determinism (SYCL barrier fix)
llama_build_and_test(test-fattn-determinism.cpp)

# test-q8-0-soa: CPU reference unit tests for Q8_0 SoA (Structure of Arrays) support
llama_build_and_test(test-q8-0-soa.cpp)

# test-mmq-soa-q4-0: CPU reference unit tests for Q4_0 MMQ SoA indexing and layout
llama_build_and_test(test-mmq-soa-q4-0.cpp)

# test-mmq-soa-q4-0-gpu: GPU unit tests for Q4_0 MMQ SoA (tests actual SYCL kernels)
if(GGML_SYCL)
    add_executable(test-mmq-soa-q4-0-gpu test-mmq-soa-q4-0-gpu.cpp)
    target_link_libraries(test-mmq-soa-q4-0-gpu PRIVATE ggml)
    target_include_directories(test-mmq-soa-q4-0-gpu PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmq-soa-q4-0-gpu PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-soa-q4-0-gpu PRIVATE "-fsycl")
    install(TARGETS test-mmq-soa-q4-0-gpu RUNTIME)
    add_test(NAME test-mmq-soa-q4-0-gpu COMMAND $<TARGET_FILE:test-mmq-soa-q4-0-gpu>)
endif()

# test-tensor-inventory-api: Tests for SYCL tensor inventory API (tiered memory placement)
if(GGML_SYCL)
    add_executable(test-tensor-inventory-api test-tensor-inventory-api.cpp)
    target_link_libraries(test-tensor-inventory-api PRIVATE ggml)
    target_include_directories(test-tensor-inventory-api PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-tensor-inventory-api PRIVATE "-fsycl")
    target_link_options(test-tensor-inventory-api PRIVATE "-fsycl")
    install(TARGETS test-tensor-inventory-api RUNTIME)
    add_test(NAME test-tensor-inventory-api COMMAND $<TARGET_FILE:test-tensor-inventory-api>)
endif()

# test-mmq-soa-q4-0-pipeline: Tests for Q4_0 MMQ pipeline (load_tiles + vec_dot)
# Tests actual tile loading and vec_dot functions to find SoA vs AoS divergence
if(GGML_SYCL)
    add_executable(test-mmq-soa-q4-0-pipeline test-mmq-soa-q4-0-pipeline.cpp)
    target_link_libraries(test-mmq-soa-q4-0-pipeline PRIVATE ggml)
    target_include_directories(test-mmq-soa-q4-0-pipeline PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmq-soa-q4-0-pipeline PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-soa-q4-0-pipeline PRIVATE "-fsycl")
    install(TARGETS test-mmq-soa-q4-0-pipeline RUNTIME)
    add_test(NAME test-mmq-soa-q4-0-pipeline COMMAND $<TARGET_FILE:test-mmq-soa-q4-0-pipeline>)
endif()

# test-mmq-q6k-gpu: GPU unit tests for Q6_K MMQ (tests production kernel path for batch>1)
if(GGML_SYCL)
    add_executable(test-mmq-q6k-gpu test-mmq-q6k-gpu.cpp)
    target_link_libraries(test-mmq-q6k-gpu PRIVATE ggml)
    target_include_directories(test-mmq-q6k-gpu PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmq-q6k-gpu PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-gpu PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-gpu RUNTIME)
    add_test(NAME test-mmq-q6k-gpu COMMAND $<TARGET_FILE:test-mmq-q6k-gpu>)

    # test-mmq-q6k-coalesced: GPU unit tests for Q6_K MMQ coalesced path
    add_executable(test-mmq-q6k-coalesced test-mmq-q6k-coalesced.cpp)
    target_link_libraries(test-mmq-q6k-coalesced PRIVATE ggml)
    target_include_directories(test-mmq-q6k-coalesced PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmq-q6k-coalesced PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-coalesced PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-coalesced RUNTIME)
    add_test(NAME test-mmq-q6k-coalesced COMMAND $<TARGET_FILE:test-mmq-q6k-coalesced>)

    # test-mmq-q8-0-tile-compare: Compare AoS vs SoA tile loading for Q8_0 MMQ
    add_executable(test-mmq-q8-0-tile-compare test-mmq-q8-0-tile-compare.cpp)
    target_link_libraries(test-mmq-q8-0-tile-compare PRIVATE ggml)
    target_include_directories(test-mmq-q8-0-tile-compare PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmq-q8-0-tile-compare PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q8-0-tile-compare PRIVATE "-fsycl")
    install(TARGETS test-mmq-q8-0-tile-compare RUNTIME)
    add_test(NAME test-mmq-q8-0-tile-compare COMMAND $<TARGET_FILE:test-mmq-q8-0-tile-compare>)

    # test-mmq-y-tile-compare: Compare AoS vs SoA Y-tile loading for Q8_1 quantized Y matrix
    add_executable(test-mmq-y-tile-compare test-mmq-y-tile-compare.cpp)
    install(TARGETS test-mmq-y-tile-compare RUNTIME)
    add_test(NAME test-mmq-y-tile-compare COMMAND $<TARGET_FILE:test-mmq-y-tile-compare>)

    # test-mmq-q6k-soa-step: Step-by-step debug of Q6_K SoA kernel logic (CPU simulation)
    add_executable(test-mmq-q6k-soa-step test-mmq-q6k-soa-step.cpp)
    target_link_libraries(test-mmq-q6k-soa-step PRIVATE ggml)
    target_include_directories(test-mmq-q6k-soa-step PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    install(TARGETS test-mmq-q6k-soa-step RUNTIME)

    # test-mmq-q6k-soa-gpu-step: GPU step-by-step debug (runs actual GPU kernel with value capture)
    add_executable(test-mmq-q6k-soa-gpu-step test-mmq-q6k-soa-gpu-step.cpp)
    target_link_libraries(test-mmq-q6k-soa-gpu-step PRIVATE ggml)
    target_include_directories(test-mmq-q6k-soa-gpu-step PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmq-q6k-soa-gpu-step PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-soa-gpu-step PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-soa-gpu-step RUNTIME)
endif()

# Test 1: Multi-thread X-tile loading test
if(GGML_SYCL)
    add_executable(test-mmq-q6k-soa-multithread test-mmq-q6k-soa-multithread.cpp)
    target_link_libraries(test-mmq-q6k-soa-multithread PRIVATE ggml)
    target_include_directories(test-mmq-q6k-soa-multithread PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmq-q6k-soa-multithread PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-soa-multithread PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-soa-multithread RUNTIME)
endif()

# Test 2: Y-tile loading test
if(GGML_SYCL)
    add_executable(test-mmq-q6k-soa-ytile test-mmq-q6k-soa-ytile.cpp)
    target_link_libraries(test-mmq-q6k-soa-ytile PRIVATE ggml)
    target_include_directories(test-mmq-q6k-soa-ytile PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmq-q6k-soa-ytile PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-soa-ytile PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-soa-ytile RUNTIME)
endif()

# Test 3: vec_dot with known tiles test
if(GGML_SYCL)
    add_executable(test-mmq-q6k-soa-vecdot test-mmq-q6k-soa-vecdot.cpp)
    target_link_libraries(test-mmq-q6k-soa-vecdot PRIVATE ggml)
    target_include_directories(test-mmq-q6k-soa-vecdot PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmq-q6k-soa-vecdot PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-soa-vecdot PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-soa-vecdot RUNTIME)
endif()

# Test 4: Tile index verification test (CPU only - no SYCL needed)
add_executable(test-mmq-q6k-soa-indices test-mmq-q6k-soa-indices.cpp)
install(TARGETS test-mmq-q6k-soa-indices RUNTIME)

# Test 5: SLM barrier test
if(GGML_SYCL)
    add_executable(test-mmq-q6k-soa-barrier test-mmq-q6k-soa-barrier.cpp)
    target_link_libraries(test-mmq-q6k-soa-barrier PRIVATE ggml)
    target_include_directories(test-mmq-q6k-soa-barrier PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmq-q6k-soa-barrier PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-soa-barrier PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-soa-barrier RUNTIME)

    # test-mmq-q6k-tile-boundary: Tests multi-tile iteration and tile boundary handling
    add_executable(test-mmq-q6k-tile-boundary test-mmq-q6k-tile-boundary.cpp)
    target_link_libraries(test-mmq-q6k-tile-boundary PRIVATE ggml)
    target_include_directories(test-mmq-q6k-tile-boundary PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmq-q6k-tile-boundary PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmq-q6k-tile-boundary PRIVATE "-fsycl")
    install(TARGETS test-mmq-q6k-tile-boundary RUNTIME)
endif()

# MMVQ Q6_K tests (tests small-batch decode path, batch 2-32)
if(GGML_SYCL)
    # test-mmvq-q6k-gpu: GPU unit tests for Q6_K MMVQ
    add_executable(test-mmvq-q6k-gpu test-mmvq-q6k-gpu.cpp)
    target_link_libraries(test-mmvq-q6k-gpu PRIVATE ggml)
    target_include_directories(test-mmvq-q6k-gpu PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmvq-q6k-gpu PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q6k-gpu PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q6k-gpu RUNTIME)
    add_test(NAME test-mmvq-q6k-gpu COMMAND $<TARGET_FILE:test-mmvq-q6k-gpu>)

    # test-mmvq-q6k-soa-rowslice: Tests row slice offset calculations for SoA
    add_executable(test-mmvq-q6k-soa-rowslice test-mmvq-q6k-soa-rowslice.cpp)
    target_link_libraries(test-mmvq-q6k-soa-rowslice PRIVATE ggml)
    target_include_directories(test-mmvq-q6k-soa-rowslice PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmvq-q6k-soa-rowslice PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q6k-soa-rowslice PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q6k-soa-rowslice RUNTIME)
    add_test(NAME test-mmvq-q6k-soa-rowslice COMMAND $<TARGET_FILE:test-mmvq-q6k-soa-rowslice>)

    # test-mmvq-q6k-debug: Debug utilities for Q6_K MMVQ
    add_executable(test-mmvq-q6k-debug test-mmvq-q6k-debug.cpp)
    target_link_libraries(test-mmvq-q6k-debug PRIVATE ggml)
    target_include_directories(test-mmvq-q6k-debug PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmvq-q6k-debug PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q6k-debug PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q6k-debug RUNTIME)

    # test-mmvq-q8-0-streaming-bench: CPU vs DMA-streamed GPU benchmark for Q8_0 MMVQ
    add_executable(test-mmvq-q8-0-streaming-bench test-mmvq-q8-0-streaming-bench.cpp)
    target_link_libraries(test-mmvq-q8-0-streaming-bench PRIVATE ggml)
    target_include_directories(test-mmvq-q8-0-streaming-bench PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmvq-q8-0-streaming-bench PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q8-0-streaming-bench PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q8-0-streaming-bench RUNTIME)
    add_test(NAME test-mmvq-q8-0-streaming-bench COMMAND $<TARGET_FILE:test-mmvq-q8-0-streaming-bench>)
    add_test(NAME test-mmvq-q8-0-streaming-smoke COMMAND $<TARGET_FILE:test-mmvq-q8-0-streaming-bench>)
    set_tests_properties(test-mmvq-q8-0-streaming-smoke PROPERTIES
        ENVIRONMENT "GGML_SYCL_MMVQ_BENCH=1;GGML_SYCL_MMVQ_BENCH_MODE=cache;GGML_SYCL_MMVQ_BENCH_NROWS=256;GGML_SYCL_MMVQ_BENCH_NCOLS=512;GGML_SYCL_MMVQ_BENCH_ITERS=1;GGML_SYCL_MMVQ_BENCH_WARMUP=0;GGML_SYCL_MMVQ_BENCH_SLICE_MB=1;GGML_SYCL_MMVQ_BENCH_BUFFERS=1")
add_test(NAME test-mmq-q8-0-streaming-smoke COMMAND $<TARGET_FILE:test-mmvq-q8-0-streaming-bench>)
set_tests_properties(test-mmq-q8-0-streaming-smoke PROPERTIES
    ENVIRONMENT "GGML_SYCL_MMVQ_BENCH=1;GGML_SYCL_MMVQ_BENCH_MODE=mmq;GGML_SYCL_MMVQ_BENCH_MMQ_NROWS=512;GGML_SYCL_MMVQ_BENCH_MMQ_NCOLS=2048;GGML_SYCL_MMVQ_BENCH_MMQ_NTOKENS=4;GGML_SYCL_MMVQ_BENCH_MMQ_ITERS=1;GGML_SYCL_MMVQ_BENCH_MMQ_WARMUP=0;GGML_SYCL_MMVQ_BENCH_MMQ_BUDGET_MB=64")
add_test(NAME test-mmq-q8-0-streaming-forced COMMAND $<TARGET_FILE:test-mmvq-q8-0-streaming-bench>)
    set_tests_properties(test-mmq-q8-0-streaming-forced PROPERTIES
        ENVIRONMENT "GGML_SYCL_MMVQ_BENCH=1;GGML_SYCL_MMVQ_BENCH_MODE=mmq;GGML_SYCL_MMVQ_BENCH_MMQ_NROWS=512;GGML_SYCL_MMVQ_BENCH_MMQ_NCOLS=2048;GGML_SYCL_MMVQ_BENCH_MMQ_NTOKENS=4;GGML_SYCL_MMVQ_BENCH_MMQ_ITERS=1;GGML_SYCL_MMVQ_BENCH_MMQ_WARMUP=0;GGML_SYCL_MMVQ_BENCH_MMQ_BUDGET_MB=1")

    # test-sycl-host-buffer-routing: verify host buffer uses pinned allocation when enabled
    add_executable(test-sycl-host-buffer-routing test-sycl-host-buffer-routing.cpp)
    target_link_libraries(test-sycl-host-buffer-routing PRIVATE ggml)
    target_include_directories(test-sycl-host-buffer-routing PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-sycl-host-buffer-routing PRIVATE "-fsycl")
    target_link_options(test-sycl-host-buffer-routing PRIVATE "-fsycl")
    install(TARGETS test-sycl-host-buffer-routing RUNTIME)
    add_test(NAME test-sycl-host-buffer-routing COMMAND $<TARGET_FILE:test-sycl-host-buffer-routing>)
    set_tests_properties(test-sycl-host-buffer-routing PROPERTIES
        ENVIRONMENT "GGML_SYCL_HOST_BUFFER_TEST=1;GGML_SYCL_HOST_BUFFER_SIZE_MB=4")
    add_test(NAME test-sycl-host-buffer-routing-fallback COMMAND $<TARGET_FILE:test-sycl-host-buffer-routing>)
    set_tests_properties(test-sycl-host-buffer-routing-fallback PROPERTIES
        ENVIRONMENT "GGML_SYCL_HOST_BUFFER_TEST=1;GGML_SYCL_HOST_BUFFER_SIZE_MB=4;GGML_SYCL_NO_PINNED=1")
endif()

# test-mmvq-coalesced: GPU unit tests for coalesced MMVQ (Q4_0 and Q6_K)
if(GGML_SYCL)
    add_executable(test-mmvq-coalesced test-mmvq-coalesced.cpp)
    target_link_libraries(test-mmvq-coalesced PRIVATE ggml)
    target_include_directories(test-mmvq-coalesced PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mmvq-coalesced PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-coalesced PRIVATE "-fsycl")
    install(TARGETS test-mmvq-coalesced RUNTIME)
    add_test(NAME test-mmvq-coalesced COMMAND $<TARGET_FILE:test-mmvq-coalesced>)
endif()

# test-tiled-weight-loading: Functional verification for XMX MXFP4 tiled MoE weights
if(GGML_SYCL)
    add_executable(test-tiled-weight-loading test-tiled-weight-loading.cpp)
    target_link_libraries(test-tiled-weight-loading PRIVATE ggml)
    target_include_directories(test-tiled-weight-loading PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-tiled-weight-loading PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-tiled-weight-loading PRIVATE "-fsycl")
    install(TARGETS test-tiled-weight-loading RUNTIME)
    add_test(NAME test-tiled-weight-loading COMMAND $<TARGET_FILE:test-tiled-weight-loading>)
endif()

# test-weight-loading: Unit tests for layout-aware weight loading in unified kernel
if(GGML_SYCL)
    add_executable(test-weight-loading test-weight-loading.cpp)
    target_link_libraries(test-weight-loading PRIVATE ggml)
    target_include_directories(test-weight-loading PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-weight-loading PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-weight-loading PRIVATE "-fsycl")
    install(TARGETS test-weight-loading RUNTIME)
    add_test(NAME test-weight-loading COMMAND $<TARGET_FILE:test-weight-loading>)
endif()

# test-mxfp4-xmx-tiled: MXFP4 XMX tiled conversion unit test
if(GGML_SYCL)
    add_executable(test-mxfp4-xmx-tiled test-mxfp4-xmx-tiled.cpp)
    target_link_libraries(test-mxfp4-xmx-tiled PRIVATE ggml)
    target_include_directories(test-mxfp4-xmx-tiled PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-mxfp4-xmx-tiled PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-mxfp4-xmx-tiled PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mxfp4-xmx-tiled PRIVATE "-fsycl")
    install(TARGETS test-mxfp4-xmx-tiled RUNTIME)
    add_test(NAME test-mxfp4-xmx-tiled COMMAND $<TARGET_FILE:test-mxfp4-xmx-tiled>)
endif()

# test-moe-mul-mat-id: MoE MXFP4 MUL_MAT_ID CPU vs SYCL comparison
if(GGML_SYCL)
    add_executable(test-moe-mul-mat-id test-moe-mul-mat-id.cpp)
    target_link_libraries(test-moe-mul-mat-id PRIVATE ggml ggml-cpu)
    target_include_directories(test-moe-mul-mat-id PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-moe-mul-mat-id PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-moe-mul-mat-id PRIVATE "-fsycl")
    install(TARGETS test-moe-mul-mat-id RUNTIME)
    add_test(NAME test-moe-mul-mat-id COMMAND $<TARGET_FILE:test-moe-mul-mat-id>)
endif()

# test-mul-mat-host-streaming: host weight streaming for MUL_MAT (CPU vs SYCL)
if(GGML_SYCL)
    add_executable(test-mul-mat-host-streaming test-mul-mat-host-streaming.cpp)
    target_link_libraries(test-mul-mat-host-streaming PRIVATE ggml ggml-cpu)
    target_include_directories(test-mul-mat-host-streaming PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-mul-mat-host-streaming PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mul-mat-host-streaming PRIVATE "-fsycl")
    install(TARGETS test-mul-mat-host-streaming RUNTIME)
    add_test(NAME test-mul-mat-host-streaming COMMAND $<TARGET_FILE:test-mul-mat-host-streaming>)
endif()

# test-xmx-host-streaming: XMX GEMM host streaming for Q4_0 (CPU vs SYCL)
if(GGML_SYCL)
    add_executable(test-xmx-host-streaming test-xmx-host-streaming.cpp)
    target_link_libraries(test-xmx-host-streaming PRIVATE ggml ggml-cpu)
    target_include_directories(test-xmx-host-streaming PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-xmx-host-streaming PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-xmx-host-streaming PRIVATE "-fsycl")
    target_compile_definitions(test-xmx-host-streaming PRIVATE GGML_SYCL_WARP_SIZE=32)
    if (GGML_SYCL_MMQ_XMX)
        target_compile_definitions(test-xmx-host-streaming PRIVATE GGML_SYCL_MMQ_XMX)
    endif()
    if (GGML_SYCL_XMX_GEMM)
        target_compile_definitions(test-xmx-host-streaming PRIVATE GGML_SYCL_XMX_GEMM)
    endif()
    install(TARGETS test-xmx-host-streaming RUNTIME)
    add_test(NAME test-xmx-host-streaming COMMAND $<TARGET_FILE:test-xmx-host-streaming>)
endif()

# test-moe-mul-mat-id-q4q8: MoE Q4_0/Q8_0 MUL_MAT_ID CPU vs SYCL comparison
if(GGML_SYCL)
    add_executable(test-moe-mul-mat-id-q4q8 test-moe-mul-mat-id-q4q8.cpp)
    target_link_libraries(test-moe-mul-mat-id-q4q8 PRIVATE ggml ggml-cpu)
    target_include_directories(test-moe-mul-mat-id-q4q8 PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-moe-mul-mat-id-q4q8 PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-moe-mul-mat-id-q4q8 PRIVATE "-fsycl")
    install(TARGETS test-moe-mul-mat-id-q4q8 RUNTIME)
    add_test(NAME test-moe-mul-mat-id-q4q8 COMMAND $<TARGET_FILE:test-moe-mul-mat-id-q4q8>)
endif()

# test-routing-indices-cache: MoE routing indices extraction for pre-staging
if(GGML_SYCL)
    add_executable(test-routing-indices-cache test-routing-indices-cache.cpp)
    target_link_libraries(test-routing-indices-cache PRIVATE ggml ggml-cpu)
    target_include_directories(test-routing-indices-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-routing-indices-cache PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-routing-indices-cache PRIVATE "-fsycl")
    install(TARGETS test-routing-indices-cache RUNTIME)
    add_test(NAME test-routing-indices-cache COMMAND $<TARGET_FILE:test-routing-indices-cache>)
endif()

# test-moe-mini-graph: embedding -> RMS -> MoE MUL_MAT_ID (MXFP4)
if(GGML_SYCL)
    add_executable(test-moe-mini-graph test-moe-mini-graph.cpp)
    target_link_libraries(test-moe-mini-graph PRIVATE ggml ggml-cpu)
    target_include_directories(test-moe-mini-graph PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-moe-mini-graph PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-moe-mini-graph PRIVATE "-fsycl")
    install(TARGETS test-moe-mini-graph RUNTIME)
    add_test(NAME test-moe-mini-graph COMMAND $<TARGET_FILE:test-moe-mini-graph>)
endif()

# test-layout-cache: layout selection + unified cache AoS drop
if(GGML_SYCL)
    add_executable(test-layout-cache test-layout-cache.cpp)
    target_link_libraries(test-layout-cache PRIVATE ggml ggml-cpu)
    target_include_directories(test-layout-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-layout-cache PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-layout-cache PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-layout-cache PRIVATE "-fsycl")
    install(TARGETS test-layout-cache RUNTIME)
    add_test(NAME test-layout-cache COMMAND $<TARGET_FILE:test-layout-cache>)
endif()

# test-pinned-probe: probe pinned host allocation behavior (JSON output)
if(GGML_SYCL)
    add_executable(test-pinned-probe test-pinned-probe.cpp)
    target_compile_options(test-pinned-probe PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-pinned-probe PRIVATE "-fsycl")
    install(TARGETS test-pinned-probe RUNTIME)
endif()

# test-iommu-dma-stress: opt-in pinned-host DMA stress test (can crash GPU driver)
if(GGML_SYCL)
    add_executable(test-iommu-dma-stress test-iommu-dma-stress.cpp)
    target_link_libraries(test-iommu-dma-stress PRIVATE ggml ggml-cpu)
    target_include_directories(test-iommu-dma-stress PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-iommu-dma-stress PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-iommu-dma-stress PRIVATE "-fsycl")
    install(TARGETS test-iommu-dma-stress RUNTIME)
endif()

# test-unified-cache-bugs: regression tests for unified cache bug fixes
if(GGML_SYCL)
    add_executable(test-unified-cache-bugs test-unified-cache-bugs.cpp)
    target_link_libraries(test-unified-cache-bugs PRIVATE ggml ggml-cpu)
    target_include_directories(test-unified-cache-bugs PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-unified-cache-bugs PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-unified-cache-bugs PRIVATE "-fsycl")
    install(TARGETS test-unified-cache-bugs RUNTIME)
    add_test(NAME test-unified-cache-bugs COMMAND $<TARGET_FILE:test-unified-cache-bugs>)
endif()

# test-unified-cache-integrity: layout cache stress + map consistency
if(GGML_SYCL)
    add_executable(test-unified-cache-integrity test-unified-cache-integrity.cpp)
    target_link_libraries(test-unified-cache-integrity PRIVATE ggml ggml-cpu)
    target_include_directories(test-unified-cache-integrity PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-unified-cache-integrity PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-unified-cache-integrity PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-unified-cache-integrity PRIVATE "-fsycl")
    install(TARGETS test-unified-cache-integrity RUNTIME)
    add_test(NAME test-unified-cache-integrity COMMAND $<TARGET_FILE:test-unified-cache-integrity>)
endif()

# test-host-cache-stress: stress host_cache map + guard checks under heavy layout caching
if(GGML_SYCL)
    add_executable(test-host-cache-stress test-host-cache-stress.cpp)
    target_link_libraries(test-host-cache-stress PRIVATE ggml ggml-cpu)
    target_include_directories(test-host-cache-stress PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-host-cache-stress PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-host-cache-stress PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-host-cache-stress PRIVATE "-fsycl")
    install(TARGETS test-host-cache-stress RUNTIME)
    add_test(NAME test-host-cache-stress COMMAND $<TARGET_FILE:test-host-cache-stress>)
endif()

# test-host-cache-layout-rehash: stress ggml_sycl_get_weight_layout_ptr insertion path
if(GGML_SYCL)
    add_executable(test-host-cache-layout-rehash test-host-cache-layout-rehash.cpp)
    target_link_libraries(test-host-cache-layout-rehash PRIVATE ggml ggml-cpu)
    target_include_directories(test-host-cache-layout-rehash PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-host-cache-layout-rehash PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-host-cache-layout-rehash PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-host-cache-layout-rehash PRIVATE "-fsycl")
    install(TARGETS test-host-cache-layout-rehash RUNTIME)
    add_test(NAME test-host-cache-layout-rehash COMMAND $<TARGET_FILE:test-host-cache-layout-rehash>)
endif()

# test-layout-bytes: validate layout-specific byte sizes
if(GGML_SYCL)
    add_executable(test-layout-bytes test-layout-bytes.cpp)
    target_link_libraries(test-layout-bytes PRIVATE ggml ggml-cpu)
    target_include_directories(test-layout-bytes PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-layout-bytes PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-layout-bytes PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-layout-bytes PRIVATE "-fsycl")
    install(TARGETS test-layout-bytes RUNTIME)
    add_test(NAME test-layout-bytes COMMAND $<TARGET_FILE:test-layout-bytes>)
endif()

# test-unified-cache-unpin-event: compare event sources for unpin_on_event
if(GGML_SYCL)
    add_executable(test-unified-cache-unpin-event test-unified-cache-unpin-event.cpp)
    target_link_libraries(test-unified-cache-unpin-event PRIVATE ggml ggml-cpu)
    target_include_directories(test-unified-cache-unpin-event PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-unified-cache-unpin-event PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-unified-cache-unpin-event PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-unified-cache-unpin-event PRIVATE "-fsycl")
    install(TARGETS test-unified-cache-unpin-event RUNTIME)
    add_test(NAME test-unified-cache-unpin-event COMMAND $<TARGET_FILE:test-unified-cache-unpin-event>)
endif()

# test-host-cache-guard: verifies host cache guard detects overruns
if(GGML_SYCL)
    add_executable(test-host-cache-guard test-host-cache-guard.cpp)
    target_link_libraries(test-host-cache-guard PRIVATE ggml ggml-cpu)
    target_include_directories(test-host-cache-guard PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-host-cache-guard PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-host-cache-guard PRIVATE "-fsycl")
    install(TARGETS test-host-cache-guard RUNTIME)
    add_test(NAME test-host-cache-guard COMMAND $<TARGET_FILE:test-host-cache-guard>)
endif()

# test-unified-cache-concurrent: concurrency stress tests for unified cache
if(GGML_SYCL)
    add_executable(test-unified-cache-concurrent test-unified-cache-concurrent.cpp)
    target_link_libraries(test-unified-cache-concurrent PRIVATE ggml ggml-cpu)
    target_include_directories(test-unified-cache-concurrent PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-unified-cache-concurrent PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-unified-cache-concurrent PRIVATE "-fsycl")
    install(TARGETS test-unified-cache-concurrent RUNTIME)
    add_test(NAME test-unified-cache-concurrent COMMAND $<TARGET_FILE:test-unified-cache-concurrent>)
endif()

# test-fattn-thread-local: ensure seq-id thread-local buffers free on thread exit
if(GGML_SYCL)
    add_executable(test-fattn-thread-local test-fattn-thread-local.cpp)
    target_link_libraries(test-fattn-thread-local PRIVATE ggml ggml-cpu)
    target_include_directories(test-fattn-thread-local PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-fattn-thread-local PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-fattn-thread-local PRIVATE "-fsycl")
    install(TARGETS test-fattn-thread-local RUNTIME)
    add_test(NAME test-fattn-thread-local COMMAND $<TARGET_FILE:test-fattn-thread-local>)
endif()

# test-q8-0-soa-gpu: GPU unit tests for Q8_0 SoA (tests actual SYCL kernels)
if(GGML_SYCL)
    add_executable(test-q8-0-soa-gpu test-q8-0-soa-gpu.cpp)
    target_link_libraries(test-q8-0-soa-gpu PRIVATE ggml)
    target_include_directories(test-q8-0-soa-gpu PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-q8-0-soa-gpu PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-soa-gpu PRIVATE "-fsycl")
    install(TARGETS test-q8-0-soa-gpu RUNTIME)
    add_test(NAME test-q8-0-soa-gpu COMMAND $<TARGET_FILE:test-q8-0-soa-gpu>)
endif()

# test-q8-0-production-kernels: Tests Q8_0 DMMV and MMQ kernels via ggml backend API
# Compares GPU SoA results against CPU reference implementation
if(GGML_SYCL)
    add_executable(test-q8-0-production-kernels test-q8-0-production-kernels.cpp)
    target_link_libraries(test-q8-0-production-kernels PRIVATE ggml ggml-cpu)
    target_include_directories(test-q8-0-production-kernels PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-q8-0-production-kernels PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-production-kernels PRIVATE "-fsycl")
    install(TARGETS test-q8-0-production-kernels RUNTIME)
    add_test(NAME test-q8-0-production-kernels COMMAND $<TARGET_FILE:test-q8-0-production-kernels>)
endif()

# test-q8-0-dmmv-soa-bisect: Bisect SoA DMMV variants in production path
if(GGML_SYCL)
    add_executable(test-q8-0-dmmv-soa-bisect test-q8-0-dmmv-soa-bisect.cpp)
    target_link_libraries(test-q8-0-dmmv-soa-bisect PRIVATE ggml ggml-cpu)
    target_include_directories(test-q8-0-dmmv-soa-bisect PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-q8-0-dmmv-soa-bisect PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-dmmv-soa-bisect PRIVATE "-fsycl")
    install(TARGETS test-q8-0-dmmv-soa-bisect RUNTIME)
    add_test(NAME test-q8-0-dmmv-soa-bisect COMMAND $<TARGET_FILE:test-q8-0-dmmv-soa-bisect>)
endif()

# test-dmmv-q8-0: Comprehensive DMMV Q8_0 unit tests
# Tests GPU DMMV kernel against CPU reference across many matrix sizes
if(GGML_SYCL)
    add_executable(test-dmmv-q8-0 test-dmmv-q8-0.cpp)
    target_link_libraries(test-dmmv-q8-0 PRIVATE ggml ggml-cpu)
    target_include_directories(test-dmmv-q8-0 PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-dmmv-q8-0 PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-q8-0 PRIVATE "-fsycl")
    install(TARGETS test-dmmv-q8-0 RUNTIME)
    add_test(NAME test-dmmv-q8-0 COMMAND $<TARGET_FILE:test-dmmv-q8-0>)
endif()

# test-dmmv-q4-0-coalesced: TDD test for Q4_0 DMMV with coalesced memory layout
# Tests that coalesced layout produces same output as SoA layout
if(GGML_SYCL)
    add_executable(test-dmmv-q4-0-coalesced test-dmmv-q4-0-coalesced.cpp)
    target_link_libraries(test-dmmv-q4-0-coalesced PRIVATE ggml)
    target_include_directories(test-dmmv-q4-0-coalesced PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-q4-0-coalesced PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-q4-0-coalesced PRIVATE "-fsycl")
    install(TARGETS test-dmmv-q4-0-coalesced RUNTIME)
    add_test(NAME test-dmmv-q4-0-coalesced COMMAND $<TARGET_FILE:test-dmmv-q4-0-coalesced>)
endif()

# test-dmmv-q6k-coalesced: Q6_K DMMV with coalesced layout
if(GGML_SYCL)
    add_executable(test-dmmv-q6k-coalesced test-dmmv-q6k-coalesced.cpp)
    target_link_libraries(test-dmmv-q6k-coalesced PRIVATE ggml ggml-cpu)
    target_include_directories(test-dmmv-q6k-coalesced PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-q6k-coalesced PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-q6k-coalesced PRIVATE "-fsycl")
    install(TARGETS test-dmmv-q6k-coalesced RUNTIME)
    add_test(NAME test-dmmv-q6k-coalesced COMMAND $<TARGET_FILE:test-dmmv-q6k-coalesced>)
endif()

# test-mmvq-q6k-56block-sycl: Q6_K MMVQ with 56 blocks (variable tile) - tests Mistral FFN case
if(GGML_SYCL)
    add_executable(test-mmvq-q6k-56block-sycl test-mmvq-q6k-56block-sycl.cpp)
    target_link_libraries(test-mmvq-q6k-56block-sycl PRIVATE ggml ggml-cpu)
    target_include_directories(test-mmvq-q6k-56block-sycl PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmvq-q6k-56block-sycl PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q6k-56block-sycl PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q6k-56block-sycl RUNTIME)
    add_test(NAME test-mmvq-q6k-56block-sycl COMMAND $<TARGET_FILE:test-mmvq-q6k-56block-sycl>)
endif()

# test-mmvq-q6k-mistral-dims: Q6_K MMVQ with exact Mistral 7B dimensions
if(GGML_SYCL)
    add_executable(test-mmvq-q6k-mistral-dims test-mmvq-q6k-mistral-dims.cpp)
    target_link_libraries(test-mmvq-q6k-mistral-dims PRIVATE ggml ggml-cpu)
    target_include_directories(test-mmvq-q6k-mistral-dims PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmvq-q6k-mistral-dims PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q6k-mistral-dims PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q6k-mistral-dims RUNTIME)
    add_test(NAME test-mmvq-q6k-mistral-dims COMMAND $<TARGET_FILE:test-mmvq-q6k-mistral-dims>)
endif()

# test-mmvq-q6k-row-sweep: Q6_K 56-block row sweep to find failure threshold
if(GGML_SYCL)
    add_executable(test-mmvq-q6k-row-sweep test-mmvq-q6k-row-sweep.cpp)
    target_link_libraries(test-mmvq-q6k-row-sweep PRIVATE ggml ggml-cpu)
    target_include_directories(test-mmvq-q6k-row-sweep PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-mmvq-q6k-row-sweep PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mmvq-q6k-row-sweep PRIVATE "-fsycl")
    install(TARGETS test-mmvq-q6k-row-sweep RUNTIME)
    add_test(NAME test-mmvq-q6k-row-sweep COMMAND $<TARGET_FILE:test-mmvq-q6k-row-sweep>)
endif()

# test-q8-0-production-flow: Production-like Q8_0 test (MMQ -> DMMV flow)
# Tests the actual inference pattern: prompt processing then token generation
if(GGML_SYCL)
    add_executable(test-q8-0-production-flow test-q8-0-production-flow.cpp)
    target_link_libraries(test-q8-0-production-flow PRIVATE ggml ggml-cpu)
    target_include_directories(test-q8-0-production-flow PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-q8-0-production-flow PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-production-flow PRIVATE "-fsycl")
    install(TARGETS test-q8-0-production-flow RUNTIME)
    add_test(NAME test-q8-0-production-flow COMMAND $<TARGET_FILE:test-q8-0-production-flow>)
endif()

# test-q8-0-reuse-weights: Tests if SoA reordering persists when weight buffer is reused
if(GGML_SYCL)
    add_executable(test-q8-0-reuse-weights test-q8-0-reuse-weights.cpp)
    target_link_libraries(test-q8-0-reuse-weights PRIVATE ggml ggml-cpu)
    target_include_directories(test-q8-0-reuse-weights PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-q8-0-reuse-weights PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-reuse-weights PRIVATE "-fsycl")
    install(TARGETS test-q8-0-reuse-weights RUNTIME)
    add_test(NAME test-q8-0-reuse-weights COMMAND $<TARGET_FILE:test-q8-0-reuse-weights>)
endif()

# test-q4-0-soa: Unit tests for Q4_0 SoA (Structure of Arrays) MMQ implementation
# Tests vec_dot, tile loading, sum handling, and mul_mat with known inputs
if(GGML_SYCL)
    add_executable(test-q4-0-soa test-q4-0-soa.cpp)
    target_link_libraries(test-q4-0-soa PRIVATE ggml)
    target_include_directories(test-q4-0-soa PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-q4-0-soa PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q4-0-soa PRIVATE "-fsycl")
    install(TARGETS test-q4-0-soa RUNTIME)
    add_test(NAME test-q4-0-soa COMMAND $<TARGET_FILE:test-q4-0-soa>)
endif()

# test-dmmv-soa-q4-0: CPU unit tests for Q4_0 DMMV SoA dequantize functions
# Tests actual dequantize_q4_0 vs dequantize_q4_0_reorder functions
if(GGML_SYCL)
    add_executable(test-dmmv-soa-q4-0 test-dmmv-soa-q4-0.cpp)
    target_link_libraries(test-dmmv-soa-q4-0 PRIVATE ggml)
    target_include_directories(test-dmmv-soa-q4-0 PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-soa-q4-0 PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-soa-q4-0 PRIVATE "-fsycl")
    install(TARGETS test-dmmv-soa-q4-0 RUNTIME)
    add_test(NAME test-dmmv-soa-q4-0 COMMAND $<TARGET_FILE:test-dmmv-soa-q4-0>)
endif()

# test-q4-0-dmmv-soa-gpu: GPU unit tests for Q4_0 DMMV SoA kernel
# Tests dequantize_q4_0_reorder, GPU reorder, full DMMV vs reference
if(GGML_SYCL)
    add_executable(test-q4-0-dmmv-soa-gpu test-q4-0-dmmv-soa-gpu.cpp)
    target_link_libraries(test-q4-0-dmmv-soa-gpu PRIVATE ggml)
    target_include_directories(test-q4-0-dmmv-soa-gpu PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-q4-0-dmmv-soa-gpu PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q4-0-dmmv-soa-gpu PRIVATE "-fsycl")
    install(TARGETS test-q4-0-dmmv-soa-gpu RUNTIME)
    add_test(NAME test-q4-0-dmmv-soa-gpu COMMAND $<TARGET_FILE:test-q4-0-dmmv-soa-gpu>)
endif()

# test-dmmv-soa-q8-0: CPU unit tests for Q8_0 DMMV SoA dequantize functions
# Tests actual dequantize_q8_0 vs dequantize_q8_0_reorder functions
if(GGML_SYCL)
    add_executable(test-dmmv-soa-q8-0 test-dmmv-soa-q8-0.cpp)
    target_link_libraries(test-dmmv-soa-q8-0 PRIVATE ggml)
    target_include_directories(test-dmmv-soa-q8-0 PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-soa-q8-0 PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-soa-q8-0 PRIVATE "-fsycl")
    install(TARGETS test-dmmv-soa-q8-0 RUNTIME)
    add_test(NAME test-dmmv-soa-q8-0 COMMAND $<TARGET_FILE:test-dmmv-soa-q8-0>)
endif()

# test-q8-0-dmmv-soa-gpu: GPU unit tests for Q8_0 DMMV SoA kernel
# Tests production-like AoS/SoA DMMV kernel parity
if(GGML_SYCL)
    add_executable(test-q8-0-dmmv-soa-gpu test-q8-0-dmmv-soa-gpu.cpp)
    target_link_libraries(test-q8-0-dmmv-soa-gpu PRIVATE ggml)
    target_include_directories(test-q8-0-dmmv-soa-gpu PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-q8-0-dmmv-soa-gpu PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-dmmv-soa-gpu PRIVATE "-fsycl")
    install(TARGETS test-q8-0-dmmv-soa-gpu RUNTIME)
    add_test(NAME test-q8-0-dmmv-soa-gpu COMMAND $<TARGET_FILE:test-q8-0-dmmv-soa-gpu>)
endif()

# test-q8-0-layout-cache-path: verifies SoA layout pointer comes from unified cache for DMMV
if(GGML_SYCL)
    add_executable(test-q8-0-layout-cache-path test-q8-0-layout-cache-path.cpp)
    target_link_libraries(test-q8-0-layout-cache-path PRIVATE ggml)
    target_include_directories(test-q8-0-layout-cache-path PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_definitions(test-q8-0-layout-cache-path PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-q8-0-layout-cache-path PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-layout-cache-path PRIVATE "-fsycl")
    install(TARGETS test-q8-0-layout-cache-path RUNTIME)
    add_test(NAME test-q8-0-layout-cache-path COMMAND $<TARGET_FILE:test-q8-0-layout-cache-path>)
endif()

# test-q8-0-layout-cache-path-mmvq: verifies SoA layout pointer comes from unified cache for MMVQ
if(GGML_SYCL)
    add_executable(test-q8-0-layout-cache-path-mmvq test-q8-0-layout-cache-path-mmvq.cpp)
    target_link_libraries(test-q8-0-layout-cache-path-mmvq PRIVATE ggml)
    target_include_directories(test-q8-0-layout-cache-path-mmvq PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_definitions(test-q8-0-layout-cache-path-mmvq PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-q8-0-layout-cache-path-mmvq PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q8-0-layout-cache-path-mmvq PRIVATE "-fsycl")
    install(TARGETS test-q8-0-layout-cache-path-mmvq RUNTIME)
    add_test(NAME test-q8-0-layout-cache-path-mmvq COMMAND $<TARGET_FILE:test-q8-0-layout-cache-path-mmvq>)
endif()

# test-soa-buffer-interaction: Tests for SoA buffer interaction bug
# Tests if F32 tensors (like inp_embd) are correctly read/written when
# Q4_0 weight tensors have SoA extras tracked in the same buffer context
if(GGML_SYCL)
    add_executable(test-soa-buffer-interaction test-soa-buffer-interaction.cpp)
    target_link_libraries(test-soa-buffer-interaction PRIVATE ggml)
    target_include_directories(test-soa-buffer-interaction PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-soa-buffer-interaction PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-soa-buffer-interaction PRIVATE "-fsycl")
    install(TARGETS test-soa-buffer-interaction RUNTIME)
    add_test(NAME test-soa-buffer-interaction COMMAND $<TARGET_FILE:test-soa-buffer-interaction>)
endif()

# test-cpu-gpu-soa-interaction: Tests CPU→GPU data path with SoA reordering
# Tests the actual inp_embd bug scenario with real kernel functions
if(GGML_SYCL)
    add_executable(test-cpu-gpu-soa-interaction test-cpu-gpu-soa-interaction.cpp)
    target_link_libraries(test-cpu-gpu-soa-interaction PRIVATE ggml)
    target_include_directories(test-cpu-gpu-soa-interaction PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-cpu-gpu-soa-interaction PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-cpu-gpu-soa-interaction PRIVATE "-fsycl")
    install(TARGETS test-cpu-gpu-soa-interaction RUNTIME)
    add_test(NAME test-cpu-gpu-soa-interaction COMMAND $<TARGET_FILE:test-cpu-gpu-soa-interaction>)
endif()

# test-ggml-sycl-soa: Tests SoA Q4_0 integration using ACTUAL ggml backend API
# This tests the full integration path including buffer init, graph execution, and prompt→decode transition
if(GGML_SYCL)
    add_executable(test-ggml-sycl-soa test-ggml-sycl-soa.cpp)
    target_link_libraries(test-ggml-sycl-soa PRIVATE ggml)
    target_include_directories(test-ggml-sycl-soa PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_definitions(test-ggml-sycl-soa PRIVATE GGML_SYCL_WARP_SIZE=32)
    target_compile_options(test-ggml-sycl-soa PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-ggml-sycl-soa PRIVATE "-fsycl")
    install(TARGETS test-ggml-sycl-soa RUNTIME)
    add_test(NAME test-ggml-sycl-soa COMMAND $<TARGET_FILE:test-ggml-sycl-soa>)

    add_executable(test-sycl-add-strided-view test-sycl-add-strided-view.cpp)
    target_link_libraries(test-sycl-add-strided-view PRIVATE ggml)
    target_include_directories(test-sycl-add-strided-view PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-add-strided-view PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-add-strided-view PRIVATE "-fsycl")
    install(TARGETS test-sycl-add-strided-view RUNTIME)
    add_test(NAME test-sycl-add-strided-view COMMAND $<TARGET_FILE:test-sycl-add-strided-view>)


    # test-rms-norm-fused-sycl: Validates fused RMS_NORM + MUL kernel on SYCL
    add_executable(test-rms-norm-fused-sycl test-rms-norm-fused-sycl.cpp)
    target_link_libraries(test-rms-norm-fused-sycl PRIVATE ggml)
    target_include_directories(test-rms-norm-fused-sycl PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-rms-norm-fused-sycl PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-rms-norm-fused-sycl PRIVATE "-fsycl")
    install(TARGETS test-rms-norm-fused-sycl RUNTIME)
    add_test(NAME test-rms-norm-fused-sycl COMMAND $<TARGET_FILE:test-rms-norm-fused-sycl>)

    # test-soa-pool-aliasing: Tests for buffer pool aliasing with SoA weight reordering
    # Verifies that compute pool allocations return valid data when SoA is enabled
    add_executable(test-soa-pool-aliasing test-soa-pool-aliasing.cpp)
    target_link_libraries(test-soa-pool-aliasing PRIVATE ggml)
    target_include_directories(test-soa-pool-aliasing PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-soa-pool-aliasing PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-soa-pool-aliasing PRIVATE "-fsycl")
    install(TARGETS test-soa-pool-aliasing RUNTIME)
    add_test(NAME test-soa-pool-aliasing COMMAND $<TARGET_FILE:test-soa-pool-aliasing>)

    # test-soa-roundtrip: Tests SYCL SoA kernels (DMMV/MMQ) vs CPU reference for Q4_0, Q8_0, Q6_K
    add_executable(test-soa-roundtrip test-soa-roundtrip.cpp)
    target_link_libraries(test-soa-roundtrip PRIVATE ggml)
    target_include_directories(test-soa-roundtrip PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-soa-roundtrip PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-soa-roundtrip PRIVATE "-fsycl")
    install(TARGETS test-soa-roundtrip RUNTIME)
    add_test(NAME test-soa-roundtrip COMMAND $<TARGET_FILE:test-soa-roundtrip>)

    # test-sycl-host-kquant: Validates unified-cache host-weight layouts for Q4_K/Q5_K
    add_executable(test-sycl-host-kquant test-sycl-host-kquant.cpp)
    target_link_libraries(test-sycl-host-kquant PRIVATE ggml)
    target_include_directories(test-sycl-host-kquant PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-host-kquant PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-host-kquant PRIVATE "-fsycl")
    install(TARGETS test-sycl-host-kquant RUNTIME)
    add_test(NAME test-sycl-host-kquant COMMAND $<TARGET_FILE:test-sycl-host-kquant>)

    # test-sycl-layout-choice: Ensures single layout per tensor after finalize_layouts.
    add_executable(test-sycl-layout-choice test-sycl-layout-choice.cpp)
    target_link_libraries(test-sycl-layout-choice PRIVATE ggml)
    target_include_directories(test-sycl-layout-choice PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-layout-choice PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-layout-choice PRIVATE "-fsycl")
    install(TARGETS test-sycl-layout-choice RUNTIME)
    add_test(NAME test-sycl-layout-choice COMMAND $<TARGET_FILE:test-sycl-layout-choice>)

    # test-sycl-kernel-selection: Verifies unified kernel selection logic.
    add_executable(test-sycl-kernel-selection test-sycl-kernel-selection.cpp)
    target_link_libraries(test-sycl-kernel-selection PRIVATE ggml)
    target_include_directories(test-sycl-kernel-selection PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-kernel-selection PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-kernel-selection PRIVATE "-fsycl")
    install(TARGETS test-sycl-kernel-selection RUNTIME)
    add_test(NAME test-sycl-kernel-selection COMMAND $<TARGET_FILE:test-sycl-kernel-selection>)

    # test-sycl-weight-key-stability: key should remain stable across repeated lookups.
    add_executable(test-sycl-weight-key-stability test-sycl-weight-key-stability.cpp)
    target_link_libraries(test-sycl-weight-key-stability PRIVATE ggml)
    target_include_directories(test-sycl-weight-key-stability PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-weight-key-stability PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-weight-key-stability PRIVATE "-fsycl")
    install(TARGETS test-sycl-weight-key-stability RUNTIME)
    add_test(NAME test-sycl-weight-key-stability COMMAND $<TARGET_FILE:test-sycl-weight-key-stability>)

    # test-sycl-weight-key-uniqueness: detect key collisions across multiple weights.
    add_executable(test-sycl-weight-key-uniqueness test-sycl-weight-key-uniqueness.cpp)
    target_link_libraries(test-sycl-weight-key-uniqueness PRIVATE ggml)
    target_include_directories(test-sycl-weight-key-uniqueness PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-weight-key-uniqueness PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-weight-key-uniqueness PRIVATE "-fsycl")
    install(TARGETS test-sycl-weight-key-uniqueness RUNTIME)
    add_test(NAME test-sycl-weight-key-uniqueness COMMAND $<TARGET_FILE:test-sycl-weight-key-uniqueness>)

    # test-sycl-moe-identity-hash: MoE identity hash collision detection.
    add_executable(test-sycl-moe-identity-hash test-sycl-moe-identity-hash.cpp)
    target_link_libraries(test-sycl-moe-identity-hash PRIVATE ggml)
    target_include_directories(test-sycl-moe-identity-hash PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-moe-identity-hash PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-moe-identity-hash PRIVATE "-fsycl")
    install(TARGETS test-sycl-moe-identity-hash RUNTIME)
    add_test(NAME test-sycl-moe-identity-hash COMMAND $<TARGET_FILE:test-sycl-moe-identity-hash>)

    # test-sycl-weight-cache-guard: stress weight cache guard + host reorders.
    add_executable(test-sycl-weight-cache-guard test-sycl-weight-cache-guard.cpp)
    target_link_libraries(test-sycl-weight-cache-guard PRIVATE ggml)
    target_include_directories(test-sycl-weight-cache-guard PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-weight-cache-guard PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-weight-cache-guard PRIVATE "-fsycl")
    install(TARGETS test-sycl-weight-cache-guard RUNTIME)
    add_test(NAME test-sycl-weight-cache-guard COMMAND $<TARGET_FILE:test-sycl-weight-cache-guard>)

    # test-sycl-tensor-usage: validates tensor usage inference by name.
    add_executable(test-sycl-tensor-usage test-sycl-tensor-usage.cpp)
    target_link_libraries(test-sycl-tensor-usage PRIVATE ggml)
    target_include_directories(test-sycl-tensor-usage PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-tensor-usage PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-tensor-usage PRIVATE "-fsycl")
    install(TARGETS test-sycl-tensor-usage RUNTIME)
    add_test(NAME test-sycl-tensor-usage COMMAND $<TARGET_FILE:test-sycl-tensor-usage>)

    # test-sycl-pointer-types: validates sycl::get_pointer_type classification.
    add_executable(test-sycl-pointer-types test-sycl-pointer-types.cpp)
    target_link_libraries(test-sycl-pointer-types PRIVATE ggml)
    target_include_directories(test-sycl-pointer-types PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-sycl-pointer-types PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-pointer-types PRIVATE "-fsycl")
    install(TARGETS test-sycl-pointer-types RUNTIME)
    add_test(NAME test-sycl-pointer-types COMMAND $<TARGET_FILE:test-sycl-pointer-types>)

    # test-q6k-dispatch: Tests Q6_K MMVQ dispatch path using actual ggml backend API
    add_executable(test-q6k-dispatch test-q6k-dispatch.cpp)
    target_link_libraries(test-q6k-dispatch PRIVATE ggml)
    target_include_directories(test-q6k-dispatch PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_options(test-q6k-dispatch PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q6k-dispatch PRIVATE "-fsycl")
    install(TARGETS test-q6k-dispatch RUNTIME)
    add_test(NAME test-q6k-dispatch COMMAND $<TARGET_FILE:test-q6k-dispatch>)

    # test-q6k-reorder-dispatch: Comprehensive Q6_K reorder and dispatch unit tests
    # Tests: CPU reorder layout, SoA offset calculations, kernel data access, multi-row MMVQ
    add_executable(test-q6k-reorder-dispatch test-q6k-reorder-dispatch.cpp)
    target_link_libraries(test-q6k-reorder-dispatch PRIVATE ggml)
    target_include_directories(test-q6k-reorder-dispatch PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-q6k-reorder-dispatch PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q6k-reorder-dispatch PRIVATE "-fsycl")
    install(TARGETS test-q6k-reorder-dispatch RUNTIME)
    add_test(NAME test-q6k-reorder-dispatch COMMAND $<TARGET_FILE:test-q6k-reorder-dispatch>)

    # test-dmmv-q6k: Tests Q6_K DMMV kernel directly (decode path, not MMVQ)
    # This test calls the actual dequantize_mul_mat_vec_q6_k kernel on GPU
    # Critical: unit tests via MUL_MAT use MMVQ, but token generation uses DMMV
    add_executable(test-dmmv-q6k test-dmmv-q6k.cpp)
    target_link_libraries(test-dmmv-q6k PRIVATE ggml)
    target_include_directories(test-dmmv-q6k PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-q6k PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-q6k PRIVATE "-fsycl")
    install(TARGETS test-dmmv-q6k RUNTIME)
    add_test(NAME test-dmmv-q6k COMMAND $<TARGET_FILE:test-dmmv-q6k>)

    # test-dmmv-q6k-soa: Comprehensive tests for Q6_K DMMV SoA kernel
    # Tests both AoS and SoA layouts, verifies offset calculations, conversion, and kernel correctness
    add_executable(test-dmmv-q6k-soa test-dmmv-q6k-soa.cpp)
    target_link_libraries(test-dmmv-q6k-soa PRIVATE ggml)
    target_include_directories(test-dmmv-q6k-soa PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-q6k-soa PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-q6k-soa PRIVATE "-fsycl")
    install(TARGETS test-dmmv-q6k-soa RUNTIME)
    add_test(NAME test-dmmv-q6k-soa COMMAND $<TARGET_FILE:test-dmmv-q6k-soa>)

    # test-dmmv-q6k-soa-reorder: Verifies SoA reordering correctness by comparing kernel memory access patterns
    add_executable(test-dmmv-q6k-soa-reorder test-dmmv-q6k-soa-reorder.cpp)
    target_link_libraries(test-dmmv-q6k-soa-reorder PRIVATE ggml)
    target_include_directories(test-dmmv-q6k-soa-reorder PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dmmv-q6k-soa-reorder PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dmmv-q6k-soa-reorder PRIVATE "-fsycl")
    install(TARGETS test-dmmv-q6k-soa-reorder RUNTIME)
    add_test(NAME test-dmmv-q6k-soa-reorder COMMAND $<TARGET_FILE:test-dmmv-q6k-soa-reorder>)

    # test-q6k-soa-integration: Integration test using actual ggml_backend API
    # Tests full production path: tensor upload → CPU reorder → DMMV kernel
    add_executable(test-q6k-soa-integration test-q6k-soa-integration.cpp)
    target_link_libraries(test-q6k-soa-integration PRIVATE ggml)
    target_include_directories(test-q6k-soa-integration PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(test-q6k-soa-integration PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-q6k-soa-integration PRIVATE "-fsycl")
    install(TARGETS test-q6k-soa-integration RUNTIME)
    add_test(NAME test-q6k-soa-integration COMMAND $<TARGET_FILE:test-q6k-soa-integration>)

    # test-mxfp4-soa: Comprehensive MXFP4 SoA vs AoS unit tests
    # Tests CPU reorder, GPU kernels, and MoE expert layout
    add_executable(test-mxfp4-soa test-mxfp4-soa.cpp)
    target_include_directories(test-mxfp4-soa PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-mxfp4-soa PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-mxfp4-soa PRIVATE "-fsycl")
    install(TARGETS test-mxfp4-soa RUNTIME)
    add_test(NAME test-mxfp4-soa COMMAND $<TARGET_FILE:test-mxfp4-soa>)
endif()

# test-ggml-flash-attn-ext: unit tests for ggml_flash_attn_ext_set_seq_ids() and ggml_flash_attn_ext_set_paged()
add_executable(test-ggml-flash-attn-ext test-ggml-flash-attn-ext.cpp)
target_link_libraries(test-ggml-flash-attn-ext PRIVATE ggml)
target_include_directories(test-ggml-flash-attn-ext PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
install(TARGETS test-ggml-flash-attn-ext RUNTIME)
add_test(NAME test-ggml-flash-attn-ext COMMAND $<TARGET_FILE:test-ggml-flash-attn-ext>)

# test-gpu-sampler: unit tests for SYCL GPU sampling (temp, argmax, random sampling)
if(GGML_SYCL)
    add_executable(test-gpu-sampler test-gpu-sampler.cpp)
    target_link_libraries(test-gpu-sampler PRIVATE ggml)
    target_include_directories(test-gpu-sampler PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    # Enable SYCL compilation
    target_compile_options(test-gpu-sampler PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-gpu-sampler PRIVATE "-fsycl")
    install(TARGETS test-gpu-sampler RUNTIME)
    add_test(NAME test-gpu-sampler COMMAND $<TARGET_FILE:test-gpu-sampler>)

    # test-multi-seq-sampler: unit tests for multi-sequence GPU sampling API
    add_executable(test-multi-seq-sampler test-multi-seq-sampler.cpp)
    target_link_libraries(test-multi-seq-sampler PRIVATE ggml)
    target_include_directories(test-multi-seq-sampler PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-multi-seq-sampler PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-multi-seq-sampler PRIVATE "-fsycl")
    install(TARGETS test-multi-seq-sampler RUNTIME)
    add_test(NAME test-multi-seq-sampler COMMAND $<TARGET_FILE:test-multi-seq-sampler>)
endif()

# test-sycl-soa: Comprehensive unit tests for Q4_0, Q8_0, Q6_K SoA kernels
if(GGML_SYCL)
    add_executable(test-sycl-soa test-sycl-soa.cpp)
    target_link_libraries(test-sycl-soa PRIVATE ggml)
    target_include_directories(test-sycl-soa PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-soa PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-soa PRIVATE "-fsycl")
    install(TARGETS test-sycl-soa RUNTIME)
    add_test(NAME test-sycl-soa COMMAND $<TARGET_FILE:test-sycl-soa>)

    # test-pinned-chunk-pool: Tests for 8GB chunked pinned host memory allocator
    # Part of tiered memory architecture for MoE models
    add_executable(test-pinned-chunk-pool test-pinned-chunk-pool.cpp)
    target_link_libraries(test-pinned-chunk-pool PRIVATE ggml)
    target_include_directories(test-pinned-chunk-pool PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-pinned-chunk-pool PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-pinned-chunk-pool PRIVATE "-fsycl")
    install(TARGETS test-pinned-chunk-pool RUNTIME)
    add_test(NAME test-pinned-chunk-pool COMMAND $<TARGET_FILE:test-pinned-chunk-pool>)

    # test-sycl-chunk-manager: Unit tests for sub-tensor streaming chunk abstraction
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-4ye)
    add_executable(test-sycl-chunk-manager test-sycl-chunk-manager.cpp)
    target_link_libraries(test-sycl-chunk-manager PRIVATE ggml)
    target_include_directories(test-sycl-chunk-manager PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-chunk-manager PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-chunk-manager PRIVATE "-fsycl")
    install(TARGETS test-sycl-chunk-manager RUNTIME)
    add_test(NAME test-sycl-chunk-manager COMMAND $<TARGET_FILE:test-sycl-chunk-manager>)

    # test-tensor-placement: Tests for tensor classification and buffer type routing
    # Part of tiered memory architecture for MoE models
    add_executable(test-tensor-placement test-tensor-placement.cpp)
    target_link_libraries(test-tensor-placement PRIVATE ggml)
    target_include_directories(test-tensor-placement PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-tensor-placement PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-tensor-placement PRIVATE "-fsycl")
    install(TARGETS test-tensor-placement RUNTIME)
    add_test(NAME test-tensor-placement COMMAND $<TARGET_FILE:test-tensor-placement>)

    # test-expert-cache: Tests for MoE expert VRAM cache with LRU/frequency eviction
    # Part of tiered memory architecture for MoE models
    add_executable(test-expert-cache test-expert-cache.cpp)
    target_link_libraries(test-expert-cache PRIVATE ggml)
    target_include_directories(test-expert-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-expert-cache PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-expert-cache PRIVATE "-fsycl")
    install(TARGETS test-expert-cache RUNTIME)
    add_test(NAME test-expert-cache COMMAND $<TARGET_FILE:test-expert-cache>)

    # test-dense-scheduler: Tests for double-buffered dense layer scheduler
    # Part of tiered memory architecture for MoE models
    add_executable(test-dense-scheduler test-dense-scheduler.cpp)
    target_link_libraries(test-dense-scheduler PRIVATE ggml)
    target_include_directories(test-dense-scheduler PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-dense-scheduler PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-dense-scheduler PRIVATE "-fsycl")
    install(TARGETS test-dense-scheduler RUNTIME)
    add_test(NAME test-dense-scheduler COMMAND $<TARGET_FILE:test-dense-scheduler>)

    # test-vram-pool: Tests for VRAM pool device memory allocator
    # Part of Unified Tensor Cache for device memory management
    add_executable(test-vram-pool test-vram-pool.cpp)
    target_link_libraries(test-vram-pool PRIVATE ggml)
    target_include_directories(test-vram-pool PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-vram-pool PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-vram-pool PRIVATE "-fsycl")
    install(TARGETS test-vram-pool RUNTIME)
    add_test(NAME test-vram-pool COMMAND $<TARGET_FILE:test-vram-pool>)

    # Tensor classification test (header-only, no SYCL runtime needed)
    add_executable(test-tensor-classification test-tensor-classification.cpp)
    target_include_directories(test-tensor-classification PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
    add_test(NAME test-tensor-classification COMMAND $<TARGET_FILE:test-tensor-classification>)

    # test-unified-tensor-cache: Tests for unified tensor cache with tiered memory
    # Part of Unified Tensor Cache for managing weights across VRAM/Host/mmap
    add_executable(test-unified-tensor-cache test-unified-tensor-cache.cpp)
    target_link_libraries(test-unified-tensor-cache PRIVATE ggml)
    target_include_directories(test-unified-tensor-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-unified-tensor-cache PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-unified-tensor-cache PRIVATE "-fsycl")
    install(TARGETS test-unified-tensor-cache RUNTIME)
    add_test(NAME test-unified-tensor-cache COMMAND $<TARGET_FILE:test-unified-tensor-cache>)

    # test-sycl-eviction-policy: Tests for priority-based LRU eviction policy
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-2oy)
    add_executable(test-sycl-eviction-policy test-sycl-eviction-policy.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/eviction-policy.cpp)
    target_link_libraries(test-sycl-eviction-policy PRIVATE ggml)
    target_include_directories(test-sycl-eviction-policy PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-eviction-policy PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-eviction-policy PRIVATE "-fsycl")
    install(TARGETS test-sycl-eviction-policy RUNTIME)
    add_test(NAME test-sycl-eviction-policy COMMAND $<TARGET_FILE:test-sycl-eviction-policy>)

    # test-heat-aware-eviction: Tests for heat-aware eviction with priority scoring
    # Part of unified memory management system (epic llama.cpp-6hp, task llama.cpp-6hp.12)
    # Note: Header-only implementation - no separate .cpp file needed
    add_executable(test-heat-aware-eviction test-heat-aware-eviction.cpp)
    target_include_directories(test-heat-aware-eviction PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-heat-aware-eviction PRIVATE "-Wno-narrowing")
    install(TARGETS test-heat-aware-eviction RUNTIME)
    add_test(NAME test-heat-aware-eviction COMMAND $<TARGET_FILE:test-heat-aware-eviction>)

    # test-memory-budget: Unit tests for MemoryBudget class
    # Part of unified memory management system (epic llama.cpp-6hp, task llama.cpp-6hp.13)
    # Tests memory pressure detection, tier allocation, and allocation tracking
    add_executable(test-memory-budget test-memory-budget.cpp)
    target_include_directories(test-memory-budget PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-memory-budget PRIVATE "-Wno-narrowing")
    install(TARGETS test-memory-budget RUNTIME)
    add_test(NAME test-memory-budget COMMAND $<TARGET_FILE:test-memory-budget>)

    # test-sycl-expert-prefetch: Tests for hybrid expert prefetching with sorted caching
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-eqa)
    # Note: Standalone test - does not require SYCL runtime, tests pure C++ logic only
    add_executable(test-sycl-expert-prefetch test-sycl-expert-prefetch.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/expert-prefetch.cpp)
    target_include_directories(test-sycl-expert-prefetch PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-expert-prefetch PRIVATE "-Wno-narrowing")
    install(TARGETS test-sycl-expert-prefetch RUNTIME)
    add_test(NAME test-sycl-expert-prefetch COMMAND $<TARGET_FILE:test-sycl-expert-prefetch>)

    # test-sycl-kv-cache-manager: Unit tests for KV cache manager with per-head granularity
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-vr5)
    # Conditionally built only if implementation exists
    if(EXISTS "${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/kv-cache-manager.cpp")
        add_executable(test-sycl-kv-cache-manager test-sycl-kv-cache-manager.cpp
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/kv-cache-manager.cpp
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/eviction-policy.cpp)
        target_link_libraries(test-sycl-kv-cache-manager PRIVATE ggml)
        target_include_directories(test-sycl-kv-cache-manager PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
        target_compile_options(test-sycl-kv-cache-manager PRIVATE "-fsycl" "-Wno-narrowing")
        target_link_options(test-sycl-kv-cache-manager PRIVATE "-fsycl")
        install(TARGETS test-sycl-kv-cache-manager RUNTIME)
        add_test(NAME test-sycl-kv-cache-manager COMMAND $<TARGET_FILE:test-sycl-kv-cache-manager>)
    endif()

    # test-sycl-compute-buffer-manager: Unit tests for compute buffer management
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-6s5)
    # Conditionally built only if implementation exists
    if(EXISTS "${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/compute-buffer-manager.cpp")
        add_executable(test-sycl-compute-buffer-manager test-sycl-compute-buffer-manager.cpp
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/compute-buffer-manager.cpp
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/eviction-policy.cpp)
        target_link_libraries(test-sycl-compute-buffer-manager PRIVATE ggml)
        target_include_directories(test-sycl-compute-buffer-manager PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
        target_compile_options(test-sycl-compute-buffer-manager PRIVATE "-fsycl" "-Wno-narrowing")
        target_link_options(test-sycl-compute-buffer-manager PRIVATE "-fsycl")
        install(TARGETS test-sycl-compute-buffer-manager RUNTIME)
        add_test(NAME test-sycl-compute-buffer-manager COMMAND $<TARGET_FILE:test-sycl-compute-buffer-manager>)
    endif()

    # test-sycl-unified-cache: Integration tests for unified cache system
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-dkr)
    # Tests allocation routing, memory pressure, eviction, and budget enforcement
    add_executable(test-sycl-unified-cache test-sycl-unified-cache.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/eviction-policy.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/expert-prefetch.cpp)
    target_link_libraries(test-sycl-unified-cache PRIVATE ggml)
    target_include_directories(test-sycl-unified-cache PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-unified-cache PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-unified-cache PRIVATE "-fsycl")
    install(TARGETS test-sycl-unified-cache RUNTIME)
    add_test(NAME test-sycl-unified-cache COMMAND $<TARGET_FILE:test-sycl-unified-cache>)

    # test-tiered-dispatch: Tests for tiered memory dispatch integration
    # Validates that tensor inventory and tiered mode affect kernel dispatch decisions
    add_executable(test-tiered-dispatch test-tiered-dispatch.cpp)
    target_link_libraries(test-tiered-dispatch PRIVATE ggml)
    target_include_directories(test-tiered-dispatch PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_options(test-tiered-dispatch PRIVATE "-fsycl")
    target_link_options(test-tiered-dispatch PRIVATE "-fsycl")
    install(TARGETS test-tiered-dispatch RUNTIME)
    add_test(NAME test-tiered-dispatch COMMAND $<TARGET_FILE:test-tiered-dispatch>)

    # test-tiered-memory-integration: Shell script integration test for tiered memory
    # Tests real model loading and inference with tiered memory system
    # Requires models in /Storage/GenAI/models/ (use --skip-large for CI)
    if (NOT WIN32)
        add_test(
            NAME test-tiered-memory-integration
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test-tiered-memory-integration.sh --skip-large
        )
        set_property(TEST test-tiered-memory-integration PROPERTY LABELS "sycl;integration")
        set_property(TEST test-tiered-memory-integration PROPERTY ENVIRONMENT "LLAMA_BIN_DIR=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()

    # test-sycl-unified-memory-e2e: End-to-end validation tests for unified memory system
    # Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-13o)
    # Validates 120B+ MoE inference patterns on 13.5GB VRAM
    add_executable(test-sycl-unified-memory-e2e test-sycl-unified-memory-e2e.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/eviction-policy.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/expert-prefetch.cpp)
    target_link_libraries(test-sycl-unified-memory-e2e PRIVATE ggml)
    target_include_directories(test-sycl-unified-memory-e2e PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-unified-memory-e2e PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-unified-memory-e2e PRIVATE "-fsycl")
    install(TARGETS test-sycl-unified-memory-e2e RUNTIME)
    add_test(NAME test-sycl-unified-memory-e2e COMMAND $<TARGET_FILE:test-sycl-unified-memory-e2e>)
    set_property(TEST test-sycl-unified-memory-e2e PROPERTY LABELS "sycl;e2e")

    # test-sycl-compute-buffers: Tests P0 (CRITICAL priority) compute buffer management
    add_executable(test-sycl-compute-buffers test-sycl-compute-buffers.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/compute-buffer-manager.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/eviction-policy.cpp)
    target_link_libraries(test-sycl-compute-buffers PRIVATE ggml ggml-cpu)
    target_include_directories(test-sycl-compute-buffers PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-compute-buffers PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-compute-buffers PRIVATE "-fsycl")
    install(TARGETS test-sycl-compute-buffers RUNTIME)
    add_test(NAME test-sycl-compute-buffers COMMAND $<TARGET_FILE:test-sycl-compute-buffers>)

    # test-sycl-blind-preload-threshold: Tests for disabling blind graph_preload_moe_experts
    # for large expert counts (128+ experts causes cache thrashing on 13.5GB VRAM)
    # Part of routing-aware MoE expert pre-staging optimization
    add_executable(test-sycl-blind-preload-threshold test-sycl-blind-preload-threshold.cpp)
    target_link_libraries(test-sycl-blind-preload-threshold PRIVATE ggml)
    target_include_directories(test-sycl-blind-preload-threshold PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-blind-preload-threshold PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-sycl-blind-preload-threshold PRIVATE "-fsycl")
    install(TARGETS test-sycl-blind-preload-threshold RUNTIME)
    add_test(NAME test-sycl-blind-preload-threshold COMMAND $<TARGET_FILE:test-sycl-blind-preload-threshold>)
    # Test with custom threshold via env var
    add_test(NAME test-sycl-blind-preload-threshold-custom COMMAND $<TARGET_FILE:test-sycl-blind-preload-threshold>)
    set_tests_properties(test-sycl-blind-preload-threshold-custom PROPERTIES
        ENVIRONMENT "GGML_SYCL_MAX_BLIND_PRELOAD_EXPERTS=32")

    # test-sycl-prestage-routed-experts: Tests for routing-aware MoE expert pre-staging
    # Part of MoE optimization for 128+ expert models (epic llama.cpp-qz6)
    # Standalone test - does not require SYCL runtime, tests pure C++ deduplication logic
    add_executable(test-sycl-prestage-routed-experts test-sycl-prestage-routed-experts.cpp)
    target_include_directories(test-sycl-prestage-routed-experts PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-sycl-prestage-routed-experts PRIVATE "-Wno-narrowing")
    install(TARGETS test-sycl-prestage-routed-experts RUNTIME)
    add_test(NAME test-sycl-prestage-routed-experts COMMAND $<TARGET_FILE:test-sycl-prestage-routed-experts>)

    # test-cold-start: Hardware heuristics for cold-start kernel configuration
    # Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.2)
    # Tests GPU capability detection and initial config derivation for Arc A770/A750/B580
    add_executable(test-cold-start test-cold-start.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl/cold-start.cpp)
    target_include_directories(test-cold-start PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-cold-start PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-cold-start PRIVATE "-fsycl")
    install(TARGETS test-cold-start RUNTIME)
    add_test(NAME test-cold-start COMMAND $<TARGET_FILE:test-cold-start>)
    set_property(TEST test-cold-start PROPERTY LABELS "sycl;auto-tune")

    # test-tuning-cache-io: JSON serialization/deserialization for tuning cache
    # Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.3)
    # Pure C++ test - no SYCL runtime required
    add_executable(test-tuning-cache-io test-tuning-cache-io.cpp)
    target_include_directories(test-tuning-cache-io PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-tuning-cache-io PRIVATE "-Wno-narrowing")
    install(TARGETS test-tuning-cache-io RUNTIME)
    add_test(NAME test-tuning-cache-io COMMAND $<TARGET_FILE:test-tuning-cache-io>)
    set_property(TEST test-tuning-cache-io PROPERTY LABELS "sycl;auto-tune")

    # test-prefetch-scheduler: Predictive weight prefetching scheduler
    # Part of heat-aware memory management (epic llama.cpp-6hp, task llama.cpp-6hp.14)
    # Header-only implementation - no SYCL runtime needed
    add_executable(test-prefetch-scheduler test-prefetch-scheduler.cpp)
    target_include_directories(test-prefetch-scheduler PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-prefetch-scheduler PRIVATE "-Wno-narrowing")
    install(TARGETS test-prefetch-scheduler RUNTIME)
    add_test(NAME test-prefetch-scheduler COMMAND $<TARGET_FILE:test-prefetch-scheduler>)
    set_property(TEST test-prefetch-scheduler PROPERTY LABELS "sycl;memory")

    # test-onednn-fallback: Unit tests for OneDNN fallback wrapper (primitive caching, M bucketing)
    # Header-only test - no SYCL device code, just tests the C++ interface
    add_executable(test-onednn-fallback test-onednn-fallback.cpp)
    target_include_directories(test-onednn-fallback PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    install(TARGETS test-onednn-fallback RUNTIME)
    add_test(NAME test-onednn-fallback COMMAND $<TARGET_FILE:test-onednn-fallback>)
    set_property(TEST test-onednn-fallback PROPERTY LABELS "sycl;onednn")

    # test-transfer-learning: Transfer learning manager for bootstrapping tuning caches
    # Part of unified kernel auto-tuning (epic llama.cpp-a3t, task llama.cpp-6hp.10)
    # Pure C++ test - no SYCL runtime required
    add_executable(test-transfer-learning test-transfer-learning.cpp)
    target_include_directories(test-transfer-learning PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-transfer-learning PRIVATE "-Wno-narrowing")
    install(TARGETS test-transfer-learning RUNTIME)
    add_test(NAME test-transfer-learning COMMAND $<TARGET_FILE:test-transfer-learning>)
    set_property(TEST test-transfer-learning PROPERTY LABELS "sycl;auto-tune")
endif()
