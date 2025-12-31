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

# test-kv-block: tests for PagedAttention block management (links against llama for internal access)
add_executable(test-kv-block test-kv-block.cpp ${PROJECT_SOURCE_DIR}/src/llama-kv-block.cpp)
target_include_directories(test-kv-block PRIVATE ${PROJECT_SOURCE_DIR}/src)
install(TARGETS test-kv-block RUNTIME)
add_test(NAME test-kv-block COMMAND $<TARGET_FILE:test-kv-block>)

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
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-ggml-sycl-soa PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-ggml-sycl-soa PRIVATE "-fsycl")
    install(TARGETS test-ggml-sycl-soa RUNTIME)
    add_test(NAME test-ggml-sycl-soa COMMAND $<TARGET_FILE:test-ggml-sycl-soa>)

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
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-sycl)
    target_compile_options(test-soa-roundtrip PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(test-soa-roundtrip PRIVATE "-fsycl")
    install(TARGETS test-soa-roundtrip RUNTIME)
    add_test(NAME test-soa-roundtrip COMMAND $<TARGET_FILE:test-soa-roundtrip>)

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
endif()