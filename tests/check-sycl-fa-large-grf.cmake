# Runs the CPU-vs-SYCL correctness oracle with the 256-GRF tile variants selected and the FA
# geometry profile on, then requires an observable selection signal in the profile: a TILE
# prefill launch at grf=256, or the one-time warning that this device has no 256-GRF mode.
# A green sweep alone would also pass at 128 GRF and prove nothing about the gates.
if (NOT ORACLE)
    message(FATAL_ERROR "ORACLE (path to test-sycl-turbo-correctness) not set")
endif()
set(ENV{GGML_SYCL_FA_LARGE_GRF} 1)
set(ENV{GGML_SYCL_FA_PROFILE} 1)
execute_process(COMMAND ${ORACLE} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 600)
if (NOT rc EQUAL 0)
    message(FATAL_ERROR "oracle failed (${rc})\n${out}\n${err}")
endif()
string(REGEX MATCH "route=TILE phase=prefill[^\n]*grf=256" selected "${err}")
string(FIND "${err}" "has no 256-GRF mode" rejected)
if (selected STREQUAL "" AND rejected EQUAL -1)
    message(FATAL_ERROR "no 256-GRF selection signal in the FA geometry profile\n${err}")
endif()
if (NOT selected STREQUAL "")
    message(STATUS "256-GRF tile launch observed: ${selected}")
else()
    message(STATUS "device has no 256-GRF mode; request rejected as designed")
endif()
