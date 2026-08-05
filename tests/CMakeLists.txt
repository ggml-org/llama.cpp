llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
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

llama_build(test-recurrent-state-rollback.cpp get-model.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-prompt-cache.cpp)
    target_include_directories(test-server-prompt-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-prompt-cache PRIVATE server-context)
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

    llama_build_and_test(test-llama-archs.cpp)

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

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
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
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)
# Project B (auto backend selector) + Project C (memory
# residency tracker) smoke test. Self-contained (no model
# fixture); only the CPU backend is registered, the auto-select
# heuristic's ACCEL-prefer branch is exercised as a no-op
# fallback in that configuration.
llama_build(test-backend-auto-and-residency.cpp)
llama_test(test-backend-auto-and-residency)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-tessera-kv-block-table.cpp)
target_include_directories(test-tessera-kv-block-table PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_build(test-component-load.cpp)
llama_build(test-server-auto-mtp-trigger.cpp)
llama_build(test-imatrix-observer-classification.cpp)
llama_test(test-imatrix-observer-classification)
llama_build(test-imatrix-observer-epoch.cpp)
llama_test(test-imatrix-observer-epoch)
llama_build(test-imatrix-observer-scope.cpp)
llama_test(test-imatrix-observer-scope)
# Phase 16.9: unit tests for the imatrix crash-safety / long-run
# safeguards (memory precheck, PID file, wall-time cap, dynamic
# save-frequency ladder). The header is self-contained (header-only
# inline functions); no llama / ggml link needed.
llama_build(test-imatrix-safeguards.cpp)
target_include_directories(test-imatrix-safeguards PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/imatrix)
llama_test(test-imatrix-safeguards)
llama_build_and_test(test-imatrix-ffn-down-coverage.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-imatrix-ffn-down-coverage PROPERTIES FIXTURES_REQUIRED test-download-model)
llama_build_and_test(test-dflash-gemma4-base.cpp LABEL "model")
llama_build_and_test(test-dflash-gemma4-gemma4.cpp LABEL "model")
llama_build(test-server-mtp-trigger.cpp)
llama_test(test-server-mtp-trigger)
target_include_directories(test-server-mtp-trigger PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(test-server-mtp-trigger PRIVATE llama-common)
llama_build(test-telemetry-jsonl.cpp)
llama_test(test-telemetry-jsonl)
llama_build(test-tessera-config.cpp)
llama_test(test-tessera-config)
llama_build(test-dflash-loader.cpp)
llama_test(test-dflash-loader)
target_include_directories(test-dflash-loader PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_build(test-dspark-markov-head.cpp)
llama_test(test-dspark-markov-head)
target_include_directories(test-dspark-markov-head PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_build(test-telemetry-record.cpp ${PROJECT_SOURCE_DIR}/tools/imatrix/telemetry-record.cpp)
target_include_directories(test-telemetry-record PRIVATE ${PROJECT_SOURCE_DIR}/tools/imatrix)
llama_test(test-telemetry-record)
llama_build(test-telemetry-golden.cpp)
llama_test(test-telemetry-golden)
llama_build_and_test(test-spec-adaptive-mux.cpp)
add_test(
    NAME test-telemetry-schema-stability
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-telemetry-schema-stability.py
)
add_executable(test-tessera-engram test-tessera-engram.cpp)
target_include_directories(test-tessera-engram PRIVATE ${PROJECT_SOURCE_DIR}/tools/tessera)
add_test(NAME test-tessera-engram COMMAND test-tessera-engram)
add_test(
    NAME test-tessera-semantic-repair-transfer
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-tessera-semantic-repair-transfer.py
)
add_test(
    NAME test-tessera-awq-evolve
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-tessera-awq-evolve.py
)
add_test(
    NAME test-tessera-shadow-calibration
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-tessera-shadow-calibration.py
)
add_test(
    NAME test-tessera-challenge-corpus
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-tessera-challenge-corpus.py
)
add_test(
    NAME test-tessera-policy-prior
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-tessera-policy-prior.py
)
add_test(
    NAME test-per-tensor-policy
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-per-tensor-policy.py
)
add_test(
    NAME test-dspark-gguf-patch
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-dspark-gguf-patch.py
)
llama_build(${PROJECT_SOURCE_DIR}/tools/tessera/kv-compress-reference.cpp)
add_executable(test-tessera-kv-compaction test-tessera-kv-compaction.cpp)
target_include_directories(test-tessera-kv-compaction PRIVATE ${PROJECT_SOURCE_DIR}/tools/tessera)
add_test(NAME test-tessera-kv-compaction COMMAND test-tessera-kv-compaction)
if (APPLE)
    llama_build(test-ane-mtp.cpp)
    llama_build(test-ane-prefill-manifest.cpp)
    llama_build(test-ane-prefill-slab.cpp)
    llama_build(test-ane-prefill-buckets.cpp)
    llama_build(test-ane-prefill-chunked.cpp)
    # W0 ANE matmul spike: standalone Objective-C++ test that uses Core ML
    # directly to validate the ANE fp16 matmul end-to-end path. The
    # ggml-ane backend is not currently wired into the default build
    # (ggml/src/CMakeLists.txt does not call `ggml_add_backend(ANE)`),
    # so this spike bypasses that backend and validates the ANE
    # integration through the canonical Core ML public API. We add the
    # Core ML / Foundation / IOSurface frameworks explicitly because
    # `llama_build` only links `llama` and `llama-common`, which do not
    # transitively expose the CoreML framework to a standalone .mm
    # binary that does not call into the existing C wrappers.
    llama_build(test-ane-matmul-w0-spike.mm)
    if (TARGET test-ane-matmul-w0-spike)
        target_link_libraries(test-ane-matmul-w0-spike PRIVATE
                              "-framework Foundation"
                              "-framework CoreML"
                              "-framework IOSurface")
    endif()
    # Data plane test for the cross-backend IOSurface buffer
    # (ggml_backend_ane_iosurface_buffer_alloc). Lock-free dispatch
    # foundation; the control plane (MTLSharedEvent) and dispatch API
    # build on top of this in follow-on commits.
    llama_build(test-ane-iosurface-buffer.mm)
    if (TARGET test-ane-iosurface-buffer)
        target_link_libraries(test-ane-iosurface-buffer PRIVATE
                              ggml ggml-base
                              "-framework Foundation"
                              "-framework IOSurface"
                              "-framework Metal")
    endif()
    # Control plane test for the cross-backend MTLSharedEvent
    # (ggml_mtl_shared_event_t). Lock-free dispatch foundation.
    llama_build(test-ane-mtl-shared-event.mm)
    if (TARGET test-ane-mtl-shared-event)
        target_link_libraries(test-ane-mtl-shared-event PRIVATE
                              ggml ggml-base
                              "-framework Foundation"
                              "-framework Metal")
    endif()
    # Dispatch API test: cross-timeline CPU <-> Metal coordination
    # through MTLSharedEvent + IOSurface (the lock-free dispatch
    # primitive end-to-end).
    llama_build(test-ane-metal-dispatch.mm)
    if (TARGET test-ane-metal-dispatch)
        target_link_libraries(test-ane-metal-dispatch PRIVATE
                              ggml ggml-base
                              "-framework Foundation"
                              "-framework Metal"
                              "-framework IOSurface")
    endif()
    # Shared ane_state_layout.v1 reader test. Exercises the
    # C++ side of the manifest contract from
    # common/ane-state-layout.h. The Python side has its own
    # 24-test suite in tools/ane-mtp/test_state_layout.py.
    # .mm because ane-state-layout.h imports Foundation.
    llama_build_and_test(test-ane-state-layout.mm)
    if (TARGET test-ane-state-layout)
        target_link_libraries(test-ane-state-layout PRIVATE
                              llama-common
                              "-framework Foundation")
    endif()
    # Pinned-slot dispatch end-to-end test. Validates the W3
    # architecture pivot: a real multifunction .mlmodelc (the
    # gemma4 prefill bundle) loads with the manifest sidecar,
    # the runtime allocates one state_iosurface, pins every
    # declared slot, and the dispatch goes through the pinned
    # IOSurface state. The test is gated on the gemma4 prefill
    # bundle (TESSERA_ANE_STATE_LAYOUT_MANIFEST env var +
    # a real GGUF arg). Skipped automatically if the env var
    # or GGUF is not present.
    llama_build(test-ane-pinned-slot-dispatch.cpp)
    if (TARGET test-ane-pinned-slot-dispatch)
        target_link_libraries(test-ane-pinned-slot-dispatch PRIVATE
                              llama-common)
        # The test reads TESSERA_ANE_STATE_LAYOUT_MANIFEST from
        # the environment at runtime. CI passes it via a fixture
        # pointing at the gemma4 prefill bundle; the test refuses
        # to run without it. The test ARG is the gemma4 prefill
        # GGUF; it is also expected to be present. We look for it
        # in the standard external-drive location used by the
        # prefill bundle. When the GGUF is missing, the test is
        # skipped (ctest -E pattern below).
        if (EXISTS "/Volumes/Julian T7/models/gemma4-ane-prefill-flat-slab.gguf")
            add_test(
                NAME test-ane-pinned-slot-dispatch
                COMMAND test-ane-pinned-slot-dispatch
                    /Volumes/Julian\ T7/models/gemma4-ane-prefill-flat-slab.gguf
            )
            set_tests_properties(test-ane-pinned-slot-dispatch PROPERTIES
                LABELS "ane"
                ENVIRONMENT "TESSERA_ANE_STATE_LAYOUT_MANIFEST=/Volumes/Julian T7/models/gemma4-ane-prefill-bundle/prefill-bundle.ane_state.v1.json"
            )
        endif()
    endif()
    # E-core pump state machine unit test. Lock-free SPSC across
    # 1000 single-thread cycles and 10000 cross-thread cycles.
    # No real Core ML required: the noop_submit callback verifies
    # the state transitions without invoking the model. The
    # submit_fn / common_ane_mtp_program / common_ane_compute_instance
    # types are C++ forward-declared so the test can compile
    # without the full program header (the test casts to the
    # forward-declared types; the noop_submit doesn't dereference).
    llama_build_and_test(test-ane-pump.cpp)
    if (TARGET test-ane-pump)
        target_link_libraries(test-ane-pump PRIVATE
                              llama-common)
    endif()
    # F4.1/F4.2/F4.3 follow-up tests: focused assertions on the
    # pump's routing (F4.1), monotonic counter (F4.2), and
    # QOS background affinity (F4.3). Additive to test-ane-pump;
    # no real .mlmodelc required (the no-op submit callback
    # doesn't dereference program/instance pointers).
    llama_build_and_test(test-ane-pump-followups.cpp)
    if (TARGET test-ane-pump-followups)
        target_link_libraries(test-ane-pump-followups PRIVATE
                              llama-common)
    endif()
    # W7 MTLSharedEvent handoff per state slot. Validates that
    # the per-slot events are signaled by the dispatch path on
    # OUTPUT_READY and that a consumer thread can wait on the
    # event with the signaled value (the cross-thread lock-free
    # contract). Gated on the gemma4 prefill bundle +
    # TESSERA_ANE_STATE_LAYOUT_MANIFEST.
    llama_build(test-ane-slot-event.cpp)
    if (TARGET test-ane-slot-event)
        target_link_libraries(test-ane-slot-event PRIVATE
                              llama-common
                              ggml ggml-base
                              "-framework Foundation"
                              "-framework Metal")
    endif()
    # Phase 0 profile: per-function phase table for a real ANE
    # dispatch. Validates the four-phase timing
    # (input_prep / ane_dispatch / output_read / signal) is
    # collected lock-free and exposed via
    # common_ane_mtp_program_phase_stats. Gated on the gemma4
    # prefill bundle + TESSERA_ANE_STATE_LAYOUT_MANIFEST.
    llama_build(test-ane-phase-profile.cpp)
    if (TARGET test-ane-phase-profile)
        target_link_libraries(test-ane-phase-profile PRIVATE
                              llama-common)
        if (EXISTS "/Volumes/Julian T7/models/gemma4-ane-prefill-flat-slab.gguf")
            add_test(
                NAME test-ane-phase-profile
                COMMAND test-ane-phase-profile
                    /Volumes/Julian\ T7/models/gemma4-ane-prefill-flat-slab.gguf
            )
            set_tests_properties(test-ane-phase-profile PROPERTIES
                LABELS "ane"
                ENVIRONMENT "TESSERA_ANE_STATE_LAYOUT_MANIFEST=/Volumes/Julian T7/models/gemma4-ane-prefill-bundle/prefill-bundle.ane_state.v1.json"
            )
        endif()
    endif()
    # F4.5 follow-up: Phase 0 profile emit test. Validates the
    # --tessera-ane-profile-out NDJSON writer end-to-end. The
    # core test (set_output / get_output / lazy open / disabled
    # flag) runs without a real bundle; the end-to-end emit
    # line-shape check is gated on the gemma4 prefill bundle +
    # TESSERA_ANE_STATE_LAYOUT_MANIFEST + TESSERA_ANE_PHASE_PROFILE_TEST_GGUF.
    llama_build_and_test(test-ane-phase-profile-emit.cpp)
    if (TARGET test-ane-phase-profile-emit)
        target_link_libraries(test-ane-phase-profile-emit PRIVATE
                              llama-common)
    endif()
    # Drafter shape contract: synthetic validation of dflash /
    # hybrid / mtp_predict slot shape contract without a real
    # drafter bundle. The test builds a synthetic
    # ane_state_layout.v1.json with the drafter functions and
    # their declared slots, then asserts the slot dtypes and
    # shapes match the dispatch path's contract. Also calls
    # the drafter public API on a prefill bundle to verify
    # graceful rejection when the bundle declares no drafter
    # functions. End-to-end validation against a real MTP /
    # DFlash bundle is deferred until a bundle with manifest
    # sidecar is available. .mm because ane-state-layout.h
    # imports Foundation.
    llama_build_and_test(test-ane-drafter-shape.mm)
    if (TARGET test-ane-drafter-shape)
        target_link_libraries(test-ane-drafter-shape PRIVATE
                              llama-common
                              "-framework Foundation")
        if (EXISTS "/Volumes/Julian T7/models/gemma4-ane-prefill-flat-slab.gguf")
            set_tests_properties(test-ane-drafter-shape PROPERTIES
                ENVIRONMENT "TESSERA_ANE_STATE_LAYOUT_MANIFEST=/Volumes/Julian T7/models/gemma4-ane-prefill-bundle/prefill-bundle.ane_state.v1.json"
                ARGS "/Volumes/Julian T7/models/gemma4-ane-prefill-flat-slab.gguf"
            )
        endif()
    endif()
    # W1 ANE MUL_MAT dispatch test: end-to-end ggml graph -> bundle.
    llama_build(test-ane-matmul-w1-dispatch.cpp)
    if (TARGET test-ane-matmul-w1-dispatch)
        target_link_libraries(test-ane-matmul-w1-dispatch PRIVATE
                              ggml ggml-base)
    endif()
    # Phase 1 body-op dispatch tests. Each op (RMS_NORM, SOFT_MAX,
    # ROPE, GLU, GET_ROWS) loads its own single-function .mlmodelc
    # fixture, runs the dispatch path through the ANE backend, and
    # verifies the output against a CPU reference. The fixtures
    # are committed under tools/ane-mtp/fixtures/<op>-*/. The
    # test is skipped when a fixture is missing; the build script
    # for each fixture is tools/ane-mtp/build-<op>-fixture.py.
    llama_build(test-ane-rmsnorm.cpp)
    if (TARGET test-ane-rmsnorm)
        target_link_libraries(test-ane-rmsnorm PRIVATE
                              ggml ggml-base)
    endif()
    llama_build(test-ane-softmax.cpp)
    if (TARGET test-ane-softmax)
        target_link_libraries(test-ane-softmax PRIVATE
                              ggml ggml-base)
    endif()
    llama_build(test-ane-rope.cpp)
    if (TARGET test-ane-rope)
        target_link_libraries(test-ane-rope PRIVATE
                              ggml ggml-base)
    endif()
    llama_build(test-ane-glu.cpp)
    if (TARGET test-ane-glu)
        target_link_libraries(test-ane-glu PRIVATE
                              ggml ggml-base)
    endif()
    llama_build(test-ane-get-rows.cpp)
    if (TARGET test-ane-get-rows)
        target_link_libraries(test-ane-get-rows PRIVATE
                              ggml ggml-base)
    endif()
    # Phase 0 L1 matmul on ANE: end-to-end parity test for
    # GGML_OP_TILE640_MATMUL dispatch. The test packs a TILE640
    # weight via the existing ts_quantize_2d (the
    # llama-quantize-impl library), dequants on the host via
    # dequantize_row_tessera_t640, and asserts the ANE fp16
    # matmul matches the L0.5 reference (CPU dequant + matmul)
    # within 1e-2 abs / 1e-1 rel. The fixture is
    # tools/ane-mtp/fixtures/tile640-matmul-256x256x1/.
    llama_build(test-ane-tile640-matmul.cpp)
    if (TARGET test-ane-tile640-matmul)
        target_link_libraries(test-ane-tile640-matmul PRIVATE
                              ggml ggml-base llama-quantize-impl ggml-ane)
        target_include_directories(test-ane-tile640-matmul PRIVATE
                                   ${PROJECT_SOURCE_DIR}/tools/quantize/tessera
                                   ${PROJECT_SOURCE_DIR}/ggml/src)
    endif()
endif()
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)
llama_build_and_test(test-spec-calibration.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-spec-calibration PROPERTIES FIXTURES_REQUIRED test-download-model)
llama_build_and_test(test-tessera-runtime.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-tessera-runtime PROPERTIES FIXTURES_REQUIRED test-download-model)
llama_build_and_test(test-tessera-replay-parity.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-tessera-replay-parity PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
    # TILE640 v2 (Accelerate + NEON) parity test for the 5 quant
    # helpers in ggml/src/ggml-quants-v2.c. Verifies bit-identical
    # output vs the C references in ggml/src/ggml-quants.c.
    llama_build_and_test(test-tessera-quants-v2.cpp)
    if (TARGET test-tessera-quants-v2)
        target_include_directories(test-tessera-quants-v2 PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/src)
    endif()
    # TILE640 v2 throughput benchmark. Times each of the 5
    # functions on the canonical Phase 0 shapes; reports
    # speedup vs the C reference. Not a ctest target (it's
    # a manual benchmark binary, not a pass/fail test).
    llama_build(bench-tessera-quants-v2.cpp)
    if (TARGET bench-tessera-quants-v2)
        target_include_directories(bench-tessera-quants-v2 PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/src)
    endif()
    # TILE640 v2 dynamic dispatch test. Asserts the cost model
    # helpers in ggml/src/ggml-quants-v2-dispatch.h return the
    # right bool per shape, and that the C ref fallback helpers
    # are self-consistent with the v2's scalar fallback.
    llama_build_and_test(test-ane-tile640-dispatch.cpp)
    if (TARGET test-ane-tile640-dispatch)
        target_include_directories(test-ane-tile640-dispatch PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/src)
    endif()
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# tessera: v2 clip-graph activation capture tests. The
# fixture is a real mmproj-tinygemma3 GGUF; if the file is
# missing the integration tests are skipped. The pure-function
# tests (bogus-path, empty-input, peak-RSS gate) run
# regardless.
set(LLAMA_TEST_NAME test-clip-capture)
llama_build_and_test(${PROJECT_SOURCE_DIR}/tools/mtmd/tests/test-clip-capture.cpp
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
target_include_directories(${LLAMA_TEST_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/mtmd
    ${PROJECT_SOURCE_DIR}/vendor)
unset(LLAMA_TEST_NAME)

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
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
