if(NOT DEFINED LLAMA_BENCH)
    message(FATAL_ERROR "LLAMA_BENCH is not set")
endif()
if(NOT DEFINED GOOD_MODEL)
    message(FATAL_ERROR "GOOD_MODEL is not set")
endif()
if(NOT DEFINED MISSING_MODEL)
    message(FATAL_ERROR "MISSING_MODEL is not set")
endif()

execute_process(
    COMMAND "${LLAMA_BENCH}"
        -m "${MISSING_MODEL}"
        -m "${GOOD_MODEL}"
        -p 1
        -n 0
        -r 1
        -o csv
        --no-warmup
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if(result EQUAL 0)
    message(FATAL_ERROR "llama-bench should report failure when one benchmark combination fails")
endif()

if(NOT error MATCHES "failed to load model")
    message(FATAL_ERROR "expected failed model load in stderr, got:\n${error}")
endif()

if(NOT output MATCHES "model_filename")
    message(FATAL_ERROR "expected CSV header in stdout, got:\n${output}")
endif()

if(NOT output MATCHES "${GOOD_MODEL}")
    message(FATAL_ERROR "llama-bench did not continue to the valid model after a failed combination.\nstdout:\n${output}\nstderr:\n${error}")
endif()
