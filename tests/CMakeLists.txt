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

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-speculative.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-backend-memory.cpp)
    llama_build_and_test(test-kv-stream-positions.cpp)
    llama_build_and_test(test-kv-stream-tile-resume.cpp)
    llama_build_and_test(test-context-memory.cpp)
    llama_build_and_test(test-kv-stream-attention-plan.cpp)
    llama_build_and_test(test-kv-stream-binding.cpp)
    llama_build_and_test(test-kv-stream-block.cpp)
    llama_build_and_test(test-kv-stream-capture.cpp)
    llama_build_and_test(test-kv-stream-content.cpp)
    llama_build_and_test(test-kv-stream-context.cpp)
    if (GGML_CUDA)
        add_executable(test-cuda-graph-types test-cuda-graph-types.cpp)
        target_link_libraries(test-cuda-graph-types PRIVATE ggml)
        llama_test(test-cuda-graph-types)
        set_tests_properties(test-cuda-graph-types PROPERTIES SKIP_RETURN_CODE 77)
        llama_build_and_test(test-kv-stream-copy-init.cpp)
        target_link_libraries(test-kv-stream-copy-init PRIVATE CUDA::cudart)
        set_tests_properties(test-kv-stream-copy-init PROPERTIES SKIP_RETURN_CODE 77)
        target_link_libraries(test-kv-stream-copy-init PRIVATE ${CMAKE_DL_LIBS})
        target_compile_definitions(test-kv-stream-copy-init PRIVATE $<$<BOOL:${GGML_CUDA_FA}>:KV_COPY_INIT_MODEL_TEST>)
    endif()
    llama_build_and_test(test-kv-stream-copy.cpp)
    llama_build_and_test(test-kv-stream-geometry.cpp)
    llama_build_and_test(test-kv-stream-host.cpp)
    llama_build_and_test(test-kv-stream-layer-lease.cpp)
    llama_build_and_test(test-kv-stream-logical-cache.cpp)
    llama_build_and_test(test-kv-stream-model.cpp)
    llama_build_and_test(test-kv-stream-mtp-lease.cpp)
    llama_build_and_test(test-kv-stream-partial.cpp)
    llama_build_and_test(test-kv-stream-policy.cpp)
    llama_build_and_test(test-kv-stream-prefetch.cpp)
    llama_build_and_test(test-kv-stream-producer.cpp)
    llama_build_and_test(test-kv-stream-publication.cpp)
    llama_build_and_test(test-kv-stream-reference.cpp)
    llama_build_and_test(test-kv-stream-resident.cpp)
    llama_build_and_test(test-kv-stream-resume.cpp)
    llama_build_and_test(test-kv-stream-session.cpp)
    llama_build_and_test(test-kv-stream-spans.cpp)
    llama_build_and_test(test-kv-stream-tile-dispatch.cpp)
    llama_build_and_test(test-kv-stream-tile.cpp)
    llama_build_and_test(test-kv-stream-transition.cpp)
    llama_build_and_test(test-kv-stream-vector-spans.cpp)
    llama_build_and_test(test-kv-stream-writer.cpp)
    llama_build_and_test(test-memory-activation.cpp)
    llama_build_and_test(test-memory-budget.cpp)
    llama_build_and_test(test-memory-completion-backends.cpp)
    llama_build_and_test(test-memory-completion.cpp)
    llama_build_and_test(test-memory-executor-cuda.cpp)
    llama_build_and_test(test-memory-executor.cpp)
    llama_build_and_test(test-memory-layout.cpp)
    llama_build_and_test(test-memory-phase-owner.cpp)
    llama_build_and_test(test-memory-phase.cpp)
    llama_build_and_test(test-memory-plan.cpp)
    llama_build_and_test(test-memory-recovery.cpp)
    llama_build_and_test(test-memory-recurrent-spill.cpp)
    llama_build_and_test(test-memory-requirements.cpp)
    llama_build_and_test(test-memory-transition.cpp)
    llama_build_and_test(test-memory-workspace.cpp)
    set_tests_properties(test-kv-stream-tile test-kv-stream-tile-dispatch PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(test-kv-stream-policy PROPERTIES TIMEOUT 30)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    if (LLAMA_BUILD_SERVER)
        llama_build_and_test(test-server-vision.cpp)
        target_link_libraries(test-server-vision PRIVATE server-context)
    endif()
    llama_build_and_test(test-json-schema-to-grammar.cpp)

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

    # Test state save/load functionality across all architectures, using the generated dummy models
    llama_test(
        test-save-load-state
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
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
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
    target_link_libraries(test-metal-graph-optimize PRIVATE ggml-metal)
endif()

# TODO: make this test (and others) not link `libllama` as it is not needed [TAG_TESTS_LLAMA_LINK]
llama_build(test-backend-ops.cpp)

add_executable(test-cuda-fattn-launch-plan test-cuda-fattn-launch-plan.cpp)
target_include_directories(test-cuda-fattn-launch-plan PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
llama_test(test-cuda-fattn-launch-plan)

add_executable(test-cuda-fattn-resume-plan test-cuda-fattn-resume-plan.cpp)
target_include_directories(test-cuda-fattn-resume-plan PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
llama_test(test-cuda-fattn-resume-plan)

add_executable(test-cuda-fattn-native test-cuda-fattn-native.cpp)
target_link_libraries(test-cuda-fattn-native PRIVATE ggml)
llama_test(test-cuda-fattn-native ARGS --backend CPU)
llama_test(test-cuda-fattn-native NAME test-cuda-fattn-native-gpu ARGS --backend CUDA0)
set_tests_properties(test-cuda-fattn-native-gpu PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test-cuda-vmm-probe test-cuda-vmm-probe.cpp)
target_link_libraries(test-cuda-vmm-probe PRIVATE llama-common)
target_include_directories(test-cuda-vmm-probe PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_test(test-cuda-vmm-probe)
llama_test_cmd(${CMAKE_COMMAND} NAME test-cuda-architecture-config
    ARGS -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/test-cuda-architecture-config.cmake)
llama_test_cmd(${CMAKE_COMMAND} NAME test-cuda-kv-runtime-config
    ARGS -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/test-cuda-kv-runtime-config.cmake)

if (GGML_CUDA AND GGML_CUDA_FA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    get_target_property(FATTN_TEST_ARCHITECTURES ggml-cuda CUDA_ARCHITECTURES)
    # skipped on Windows shared builds where the kv-stream test targets are not created
    if (TARGET test-kv-stream-vector-spans)
        foreach(TEST IN ITEMS vector-spans context tile tile-dispatch)
            target_sources(test-kv-stream-${TEST} PRIVATE kv-stream-pascal-native.cu)
            target_include_directories(test-kv-stream-${TEST} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
            target_compile_definitions(test-kv-stream-${TEST} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:$<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>>)
            if ("${TEST}" STREQUAL "vector-spans")
                target_compile_definitions(test-kv-stream-${TEST} PRIVATE KV_STREAM_PASCAL_CUDA_TEST)
            else()
                target_compile_definitions(test-kv-stream-${TEST} PRIVATE KV_STREAM_TILE_NATIVE_TEST)
            endif()
            set_property(TARGET test-kv-stream-${TEST} PROPERTY CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
            target_link_libraries(test-kv-stream-${TEST} PRIVATE ggml-cuda CUDA::cudart CUDA::cublas CUDA::cuda_driver)
        endforeach()
        target_sources(test-kv-stream-capture PRIVATE kv-stream-capture-native.cu)
        target_include_directories(test-kv-stream-capture PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
        target_compile_definitions(test-kv-stream-capture PRIVATE KV_CAPTURE_CUDA_TEST
            $<$<COMPILE_LANGUAGE:CUDA>:$<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>>)
        set_property(TARGET test-kv-stream-capture PROPERTY CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
        target_link_libraries(test-kv-stream-capture PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver)
        target_link_libraries(test-kv-stream-copy PRIVATE CUDA::cudart)
        target_compile_definitions(test-kv-stream-copy PRIVATE KV_COPY_CUDA_TEST)
    endif()
    add_executable(test-cuda-compiled-features test-cuda-compiled-features.cu)
    target_include_directories(test-cuda-compiled-features PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-cuda-compiled-features PRIVATE $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
    target_compile_options(test-cuda-compiled-features PRIVATE $<TARGET_GENEX_EVAL:ggml-cuda,$<TARGET_PROPERTY:ggml-cuda,COMPILE_OPTIONS>>)
    set_target_properties(test-cuda-compiled-features PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
    target_link_libraries(test-cuda-compiled-features PRIVATE ggml-base CUDA::cudart CUDA::cublas CUDA::cuda_driver)
    llama_test(test-cuda-compiled-features)
    set_tests_properties(test-cuda-compiled-features PROPERTIES SKIP_RETURN_CODE 77)
    foreach(TEST IN ITEMS device-buffer kv-host)
        llama_build(test-cuda-${TEST}.cpp)
        target_link_libraries(test-cuda-${TEST} PRIVATE CUDA::cudart)
    endforeach()
    add_test(NAME test-cuda-kv-host COMMAND ${CMAKE_COMMAND} -E env --unset=GGML_CUDA_NO_PINNED $<TARGET_FILE:test-cuda-kv-host>)
    add_test(NAME test-cuda-kv-host-disabled COMMAND ${CMAKE_COMMAND} -E env GGML_CUDA_NO_PINNED=1 $<TARGET_FILE:test-cuda-kv-host>)
    add_test(NAME test-cuda-device-buffer COMMAND ${CMAKE_COMMAND} -E env --unset=GGML_CUDA_ENABLE_UNIFIED_MEMORY $<TARGET_FILE:test-cuda-device-buffer>)
    add_test(NAME test-cuda-device-buffer-uvm COMMAND ${CMAKE_COMMAND} -E env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 $<TARGET_FILE:test-cuda-device-buffer>)
    set_tests_properties(test-cuda-kv-host test-cuda-kv-host-disabled test-cuda-device-buffer test-cuda-device-buffer-uvm PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(test-cuda-fattn-spans test-cuda-fattn-spans.cu)
    target_include_directories(test-cuda-fattn-spans PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    target_link_libraries(test-cuda-fattn-spans PRIVATE ggml CUDA::cudart)
    target_compile_options(test-cuda-fattn-spans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math;-extended-lambda>)
    set_target_properties(test-cuda-fattn-spans PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
    llama_test(test-cuda-fattn-spans)
    set_tests_properties(test-cuda-fattn-spans PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-cuda-fattn-mma-resume test-cuda-fattn-spans.cu)
    target_compile_definitions(test-cuda-fattn-mma-resume PRIVATE FATTN_MMA_RESUME_ONLY)
    target_include_directories(test-cuda-fattn-mma-resume PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    target_link_libraries(test-cuda-fattn-mma-resume PRIVATE ggml CUDA::cudart)
    target_compile_options(test-cuda-fattn-mma-resume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math;-extended-lambda>)
    set_target_properties(test-cuda-fattn-mma-resume PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
    llama_test(test-cuda-fattn-mma-resume)
    set_tests_properties(test-cuda-fattn-mma-resume PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test-cuda-fattn-prefill COMMAND test-cuda-fattn-mma-resume --prefill-only)
    set_tests_properties(test-cuda-fattn-prefill PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

    add_executable(test-cuda-fattn-decode-chunk test-cuda-fattn-decode-chunk.cu)
    target_include_directories(test-cuda-fattn-decode-chunk PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    target_link_libraries(test-cuda-fattn-decode-chunk PRIVATE ggml CUDA::cudart)
    target_compile_options(test-cuda-fattn-decode-chunk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math;-extended-lambda>)
    set_target_properties(test-cuda-fattn-decode-chunk PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
    llama_test(test-cuda-fattn-decode-chunk)
    set_tests_properties(test-cuda-fattn-decode-chunk PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test-cuda-fattn-resume test-cuda-fattn-resume.cu)
    target_include_directories(test-cuda-fattn-resume PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    target_link_libraries(test-cuda-fattn-resume PRIVATE ggml CUDA::cudart)
    target_compile_options(test-cuda-fattn-resume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math;-extended-lambda>)
    set_target_properties(test-cuda-fattn-resume PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${FATTN_TEST_ARCHITECTURES}")
    llama_test(test-cuda-fattn-resume)
    set_tests_properties(test-cuda-fattn-resume PROPERTIES SKIP_RETURN_CODE 77)
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
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

foreach(TEST IN ITEMS workspace embeddings session projector-storage)
    llama_build_and_test(test-mtmd-${TEST}.cpp)
    target_link_libraries(test-mtmd-${TEST} PRIVATE mtmd)
endforeach()

llama_build_and_test(test-mtmd-impl.cpp)
target_link_libraries(test-mtmd-impl PRIVATE mtmd)

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

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
    llama_build_and_test(test-backend-buffer.cpp)
    llama_build_and_test(test-backend-meta.cpp)
    llama_build_and_test(test-backend-execution.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
