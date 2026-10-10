llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common) # TODO: [TAG_TESTS_LLAMA_LINK]
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
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

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

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
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_build_chat_test)
    if (TARGET server-context AND NOT TARGET test-chat)
        llama_build_and_test(${PROJECT_SOURCE_DIR}/tests/test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_link_libraries(test-chat PRIVATE server-context)
    endif()
endfunction()

function(llama_build_mtmd_tests)
    if (TARGET mtmd)
        if (NOT TARGET test-mtmd-c-api)
            set(LLAMA_TEST_NAME test-mtmd-c-api)
            llama_build_and_test(${PROJECT_SOURCE_DIR}/tests/test-mtmd-c-api.c)
            target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
            unset(LLAMA_TEST_NAME)
        endif()

        if (NOT TARGET test-mtmd-impl)
            llama_build_and_test(${PROJECT_SOURCE_DIR}/tests/test-mtmd-impl.cpp)
            target_link_libraries(test-mtmd-impl PRIVATE mtmd)
        endif()
    endif()
endfunction()

llama_test_cmd(${CMAKE_COMMAND} NAME test-sycl-xmx-gather
               ARGS -P ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-xmx-gather.cmake)
llama_test_cmd(${CMAKE_COMMAND} NAME test-sycl-build-metadata
               ARGS -P ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-build-metadata.cmake)

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
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

llama_build(test-recurrent-state-rollback.cpp)
llama_build(test-save-load-state.cpp)

llama_build_and_test(test-gdn-ingredient-replay.cpp LABEL "turbo")
llama_build_and_test(test-upstream-api-shims.cpp LABEL "main")
target_include_directories(test-upstream-api-shims PRIVATE ${PROJECT_SOURCE_DIR}/src)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_chat_test()
    llama_build_and_test(test-common-json-const.cpp)
    llama_build_and_test(test-imatrix-loader.cpp)
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

    llama_build(test-llama-archs.cpp)
    llama_build_and_test(test-qwen4exp-mtp.cpp)
    llama_test(test-qwen4exp-mtp NAME test-qwen4exp-mtp-q8 ARGS --q8-kv)
    if (GGML_SYCL AND NOT GGML_SYCL_SEPARATE_BUILD)
        llama_test(test-qwen4exp-mtp NAME test-qwen4exp-mtp-sycl-f16 LABEL sycl ARGS --backend SYCL0)
        llama_test(test-qwen4exp-mtp NAME test-qwen4exp-mtp-sycl-q8 LABEL sycl ARGS --backend SYCL0 --q8-kv)
        set_tests_properties(test-qwen4exp-mtp-sycl-f16 test-qwen4exp-mtp-sycl-q8
            PROPERTIES TIMEOUT 180 RESOURCE_LOCK sycl0)
    endif()

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Test recurrent-state rollback across all architectures, using the generated dummy models
    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-gdn-replay
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-gdn-replay PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_TEST_GDN_REPLAY=1"
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-kimi-k3
        LABEL main
        ARGS -m "${MODEL_DIR}/kimi-k3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-kimi-k3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-lfm2
        LABEL main
        ARGS -m "${MODEL_DIR}/lfm2-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-lfm2 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-bailingmoe3
        LABEL main
        ARGS -m "${MODEL_DIR}/bailingmoe3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-bailingmoe3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # Test state save/load functionality across all architectures, using the generated dummy models
    llama_test(
        test-save-load-state
        LABEL main
        ARGS --models "${MODEL_DIR}" -t 4 -tb 4
    )
    set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
if (LLAMA_SUBPROCESS)
    llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
endif()
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
# the test serves its repos from an httplib server, so it needs cpp-httplib (vendor/CMakeLists.txt
# only adds that target for LLAMA_BUILD_SERVER OR LLAMA_DOWNLOAD -- absent under the pruned/
# dependency-free SDK profile).
if (TARGET cpp-httplib)
  llama_build_and_test(test-model-resolution.cpp)
  target_link_libraries(test-model-resolution PRIVATE cpp-httplib)
endif()

llama_build_and_test(test-speculative-adaptive.cpp)

# Synthetic checks always run; absent GPU traces are explicitly skipped.
llama_build_and_test(test-spec-adaptive-curve.cpp)
set_tests_properties(test-spec-adaptive-curve PROPERTIES SKIP_RETURN_CODE 77)
llama_test(test-spec-adaptive-curve NAME test-spec-adaptive-curve-self-test ARGS --self-test)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
if (NOT LLAMA_USE_SYSTEM_GGML)
    llama_build(test-moe-cache.cpp)
    llama_test(test-moe-cache LABEL "main;backend")
    # The test exits 77 when no supported cache provider is present. ctest only treats that
    # as a skip when SKIP_RETURN_CODE says so; without it, every CPU-only
    # runner records the skip as a failure.
    set_tests_properties(test-moe-cache PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

llama_build_and_test(test-moe-cache-fit.cpp)

# TODO: make this test (and others) not link `libllama` as it is not needed [TAG_TESTS_LLAMA_LINK]
llama_build(test-backend-ops.cpp)

if (TARGET ggml-cpu AND NOT GGML_BACKEND_DL AND NOT LLAMA_USE_SYSTEM_GGML)
    add_executable(test-sched-stream-ordered test-sched-stream-ordered.cpp)
    target_link_libraries(test-sched-stream-ordered PRIVATE ggml ggml-cpu)
    llama_test(test-sched-stream-ordered LABEL "main;backend")
    set_tests_properties(test-sched-stream-ordered PROPERTIES TIMEOUT 30 ENVIRONMENT "GGML_SCHED_COPY_SYNC=0")
    add_test(NAME test-sched-stream-ordered-default COMMAND ${CMAKE_COMMAND} -E env
        --unset=GGML_SCHED_COPY_SYNC $<TARGET_FILE:test-sched-stream-ordered> --expect-sync)
    add_test(NAME test-sched-stream-ordered-sync COMMAND ${CMAKE_COMMAND} -E env
        GGML_SCHED_COPY_SYNC=1 $<TARGET_FILE:test-sched-stream-ordered> --expect-sync)
    set_tests_properties(test-sched-stream-ordered-default test-sched-stream-ordered-sync
        PROPERTIES TIMEOUT 30 LABELS "main;backend")
endif()

llama_build_and_test(test-tiled-mulmat.cpp)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
# TODO: run on all dummy models
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
llama_build_mtmd_tests()

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

# TurboQuant WHT round-trip test (links ggml turbo codec symbols via llama)
llama_build_and_test(test-turbo-quant.c LABEL "turbo")

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)
    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

find_program(LLAMA_TEST_BASH NAMES bash)
if (LLAMA_TEST_BASH AND TARGET test-llama-archs AND TARGET test-export-graph-ops)
    add_custom_target(
        test-turbo-attention-architectures
        DEPENDS test-llama-archs test-export-graph-ops
    )
    llama_test_cmd(
        ${LLAMA_TEST_BASH}
        NAME test-turbo-attention-architectures
        ARGS
            ${CMAKE_CURRENT_SOURCE_DIR}/test-turbo-attention-architectures.sh
            $<TARGET_FILE:test-llama-archs>
            $<TARGET_FILE:test-export-graph-ops>
    )
endif()

# Fork-specific TurboQuant and SYCL tests.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(P45A_DEPS bash jq timeout sha256sum ldd fuser git realpath stat grep hostname
        find cut sed sort tail env dirname mktemp rm mkdir ln cat chmod mv cp)
    foreach (P45A_DEP ${P45A_DEPS})
        find_program(P45A_TEST_${P45A_DEP} NAMES ${P45A_DEP})
    endforeach()
    set(P45A_FOUND TRUE)
    foreach (P45A_DEP ${P45A_DEPS})
        if (NOT P45A_TEST_${P45A_DEP})
            set(P45A_FOUND FALSE)
        endif()
    endforeach()
    if (P45A_FOUND)
        execute_process(
            COMMAND ${P45A_TEST_git} rev-parse --show-toplevel
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE P45A_GIT_TOPLEVEL
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE P45A_GIT_TOPLEVEL_RESULT
        )
        get_filename_component(P45A_SOURCE_ROOT "${PROJECT_SOURCE_DIR}" REALPATH)
        if (P45A_GIT_TOPLEVEL_RESULT EQUAL 0)
            get_filename_component(P45A_GIT_ROOT "${P45A_GIT_TOPLEVEL}" REALPATH)
        endif()
        if (NOT P45A_GIT_TOPLEVEL_RESULT EQUAL 0 OR NOT P45A_GIT_ROOT STREQUAL P45A_SOURCE_ROOT)
            set(P45A_FOUND FALSE)
        endif()
    endif()
    if (P45A_FOUND)
        llama_test_cmd(
            ${P45A_TEST_bash}
            NAME test-validate-dense-turbo4-capacity
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-validate-dense-turbo4-capacity.sh
        )
    endif()
endif()


if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build(test-kv-cache-adaptive-mode.cpp)
    llama_test(test-kv-cache-adaptive-mode)
endif()

llama_build(test-turbo-innerq-runtime.cpp)
llama_test(test-turbo-innerq-runtime)

if (GGML_OPENVINO AND GGML_CPU AND NOT GGML_BACKEND_DL AND (NOT WIN32 OR NOT BUILD_SHARED_LIBS))
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    add_executable(test-openvino-repeat test-openvino-repeat.cpp)
    target_include_directories(test-openvino-repeat PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-openvino-repeat PRIVATE ggml ggml-cpu ggml-openvino openvino::runtime)
    llama_test(test-openvino-repeat LABEL openvino)
endif()

if (GGML_SYCL AND NOT GGML_SYCL_SEPARATE_BUILD)
    add_executable(test-sycl-turbo test-sycl-turbo.cpp)
    target_link_libraries(test-sycl-turbo PRIVATE llama-common ggml-sycl)
    add_test(NAME test-sycl-turbo COMMAND test-sycl-turbo)

    add_executable(test-stress-context test-stress-context.cpp)
    target_link_libraries(test-stress-context PRIVATE llama-common ggml-sycl)
    add_test(NAME test-stress-context COMMAND test-stress-context)

    add_executable(test-sycl-fuzz test-sycl-fuzz.cpp)
    target_link_libraries(test-sycl-fuzz PRIVATE llama-common ggml-sycl)
    add_test(NAME test-sycl-fuzz COMMAND test-sycl-fuzz)

    add_executable(test-sycl-stress-deep test-sycl-stress-deep.cpp)
    target_link_libraries(test-sycl-stress-deep PRIVATE llama-common ggml-sycl)
    add_test(NAME test-sycl-stress-deep COMMAND test-sycl-stress-deep)

    # xe KMD runtime defaults: pure sysfs/env logic, no GPU needed (see the note below
    # about linking `ggml` explicitly for the include path). Linux only: the probe is a
    # stub elsewhere and the test builds a fake sysfs tree with symlinks.
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(test-sycl-xe-defaults test-sycl-xe-defaults.cpp)
        target_link_libraries(test-sycl-xe-defaults PRIVATE ggml ggml-sycl)
        target_include_directories(test-sycl-xe-defaults PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
        add_test(NAME test-sycl-xe-defaults COMMAND test-sycl-xe-defaults)
        set_property(TEST test-sycl-xe-defaults PROPERTY LABELS sycl)
    endif()

    add_executable(test-sycl-turbo-correctness test-sycl-turbo-correctness.cpp)
    target_link_libraries(test-sycl-turbo-correctness PRIVATE llama llama-common ggml-cpu ggml-sycl)
    add_test(NAME test-sycl-turbo-correctness COMMAND test-sycl-turbo-correctness)
    set_property(TEST test-sycl-turbo-correctness PROPERTY LABELS sycl)
    set_property(TEST test-sycl-turbo-correctness PROPERTY TIMEOUT 180)
    # GGML_SYCL_FA_LARGE_GRF: the pure parse and launch decision, no GPU needed.
    add_executable(test-sycl-fa-large-grf test-sycl-fa-large-grf.cpp)
    target_link_libraries(test-sycl-fa-large-grf PRIVATE ggml ggml-sycl)
    target_include_directories(test-sycl-fa-large-grf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    add_test(NAME test-sycl-fa-large-grf COMMAND test-sycl-fa-large-grf)
    set_property(TEST test-sycl-fa-large-grf PROPERTY LABELS sycl)
    if (GGML_SYCL_FA_LARGE_GRF)
        # The oracle with the 256-GRF tile variants selected and the FA geometry profile on;
        # the wrapper requires the selection signal (a TILE prefill launch at grf=256, or the
        # per-device warning that the architecture has no 256-GRF mode), not just a green sweep.
        add_test(NAME test-sycl-turbo-correctness-large-grf
                 COMMAND ${CMAKE_COMMAND} -DORACLE=$<TARGET_FILE:test-sycl-turbo-correctness>
                         -P ${CMAKE_CURRENT_SOURCE_DIR}/check-sycl-fa-large-grf.cmake)
        set_property(TEST test-sycl-turbo-correctness-large-grf PROPERTY LABELS sycl)
        set_property(TEST test-sycl-turbo-correctness-large-grf PROPERTY TIMEOUT 660)
    endif()

    # NOTE: linking only ggml-sycl does NOT bring in ggml/include (that dir is exported PUBLIC
    # by the `ggml`/`ggml-base` targets only, see ggml/src/CMakeLists.txt); a test that links
    # ggml-sycl alone gets zero -I flags and silently falls through to the C compiler's system
    # include path. On this machine that resolves to an unrelated system-installed
    # llama.cpp-openvino package's stale /usr/include/ggml*.h instead of this tree's headers -
    # a silent ABI/declaration mismatch, not a build error, so every one of these targets must
    # also link `ggml` explicitly.
    add_executable(test-sycl-fusion-eligibility test-sycl-fusion-eligibility.cpp)
    # fusion.hpp pulls in <sycl/sycl.hpp> and <oneapi/mkl.hpp> via common.hpp/dpct/helper.hpp,
    # so this translation unit needs -fsycl and the MKL SYCL headers itself; linking against
    # ggml-sycl does not propagate its PRIVATE compile options/includes to consumers.
    target_compile_options(test-sycl-fusion-eligibility PRIVATE "-fsycl")
    target_link_options(test-sycl-fusion-eligibility PRIVATE "-fsycl")
    # common.hpp also reaches for the private ggml-impl.h (ggml/src, not the public ggml/include).
    target_include_directories(test-sycl-fusion-eligibility PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    # common.hpp needs the same directory-scoped defines ggml/src/ggml-sycl/CMakeLists.txt sets
    # via add_compile_definitions() for its own translation units, which (being directory-scoped,
    # not target-scoped) do not reach this file's separate directory/target at all.
    if (GGML_SYCL_TARGET STREQUAL "INTEL")
        target_compile_definitions(test-sycl-fusion-eligibility PRIVATE GGML_SYCL_WARP_SIZE=16)
    endif()
    if (GGML_SYCL_F16)
        target_compile_definitions(test-sycl-fusion-eligibility PRIVATE GGML_SYCL_F16)
    endif()
    target_link_libraries(test-sycl-fusion-eligibility PRIVATE ggml ggml-sycl MKL::MKL_SYCL::BLAS)
    add_test(NAME test-sycl-fusion-eligibility COMMAND test-sycl-fusion-eligibility)
    set_property(TEST test-sycl-fusion-eligibility PROPERTY LABELS sycl)

    add_executable(test-sycl-fattn-mkl-policy test-sycl-fattn-mkl-policy.cpp)
    target_include_directories(test-sycl-fattn-mkl-policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-sycl)
    target_link_libraries(test-sycl-fattn-mkl-policy PRIVATE ggml ggml-sycl)
    add_test(NAME test-sycl-fattn-mkl-policy COMMAND test-sycl-fattn-mkl-policy)
    set_property(TEST test-sycl-fattn-mkl-policy PROPERTY LABELS sycl)

    add_executable(test-sycl-status-propagation test-sycl-status-propagation.cpp)
    target_link_libraries(test-sycl-status-propagation PRIVATE ggml ggml-sycl)
    add_test(NAME test-sycl-status-propagation COMMAND test-sycl-status-propagation)
    set_property(TEST test-sycl-status-propagation PROPERTY LABELS sycl)

    # scheduler refuses an in-place op assigned to a backend that cannot reach its destination
    add_executable(test-sycl-sched-inplace-guard test-sycl-sched-inplace-guard.cpp)
    target_link_libraries(test-sycl-sched-inplace-guard PRIVATE ggml ggml-sycl)
    add_test(NAME test-sycl-sched-inplace-guard COMMAND test-sycl-sched-inplace-guard)
    set_property(TEST test-sycl-sched-inplace-guard PROPERTY LABELS sycl)

    # DSV4 rollback bookkeeping under a graph failure reported asynchronously:
    # needs a SYCL device for the injection and the generated DSV4 test model.
    if (DEFINED MODEL_DIR)
        add_executable(test-dsv4-async-failure test-dsv4-async-failure.cpp)
        target_link_libraries(test-dsv4-async-failure PRIVATE llama ggml ggml-sycl)
        add_test(NAME test-dsv4-async-failure COMMAND test-dsv4-async-failure -m "${MODEL_DIR}/deepseek4-moe.gguf")
        set_tests_properties(test-dsv4-async-failure PROPERTIES LABELS sycl FIXTURES_REQUIRED generate-models)
    endif()
endif()

if (TARGET test-sycl-xmx-gather-stubs)
    add_test(NAME test-sycl-xmx-gather-stubs COMMAND test-sycl-xmx-gather-stubs)
    set_tests_properties(test-sycl-xmx-gather-stubs PROPERTIES LABELS "sycl-build" TIMEOUT 30)
endif()

if (TARGET test-sycl-xmx-gather-images)
    add_test(NAME test-sycl-xmx-gather-images COMMAND test-sycl-xmx-gather-images)
    set_tests_properties(test-sycl-xmx-gather-images PROPERTIES LABELS "sycl-build" TIMEOUT 30)
endif()
