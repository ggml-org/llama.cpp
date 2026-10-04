if (DEFINED CHILD_VERSION)
    include("${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda/architectures.cmake")
    ggml_cuda_check_toolkit_architectures("${CHILD_VERSION}" "${CHILD_ARCHS}")
    return()
endif()

# Test the early diagnostic without a CUDA compiler, GPU, or parent build configuration.
function(check_case version architectures should_fail)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DCHILD_VERSION=${version}" "-DCHILD_ARCHS=${architectures}"
            -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if (should_fail)
        if (result EQUAL 0 OR NOT "${output}${error}" MATCHES "CUDA 12.9 or earlier")
            message(FATAL_ERROR "Expected actionable rejection: ${version}, ${architectures}\n${output}${error}")
        endif()
    elseif (NOT result EQUAL 0)
        message(FATAL_ERROR "Valid configuration rejected: ${version}, ${architectures}\n${output}${error}")
    endif()
endfunction()

foreach(architecture 50 60 61 62 70 72)
    check_case("12.9.1" "${architecture}" FALSE)
    check_case("13.0.0" "${architecture}" TRUE)
    check_case("13.3.0" "${architecture}-real;120a-real" TRUE)
    check_case("13.0.0" "${architecture}-virtual" TRUE)
endforeach()
check_case("12.9.1" "61;75;86;89;120" FALSE)
foreach(architectures "75;86;89;120a" "75-virtual;86-real;120a-real" "native" "all" "all-major" "OFF" "")
    check_case("13.0.0" "${architectures}" FALSE)
endforeach()
message(STATUS "CUDA toolkit/architecture cases passed")
