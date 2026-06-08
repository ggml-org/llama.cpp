execute_process(
    COMMAND "${LLAMA_SERVER_BIN}" --embedding --batch-size 16 --ubatch-size 8
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 5
)

if ("${result}" STREQUAL "0")
    message(FATAL_ERROR "expected llama-server to reject mismatched embedding batch sizes")
endif()

if (NOT "${result}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR "llama-server did not exit after detecting mismatched embedding batch sizes: ${result}")
endif()

set(log "${output}${error}")
if (NOT log MATCHES "embeddings require n_batch \\(16\\) to equal n_ubatch \\(8\\)")
    message(FATAL_ERROR "expected embedding batch size error, got:\n${log}")
endif()
