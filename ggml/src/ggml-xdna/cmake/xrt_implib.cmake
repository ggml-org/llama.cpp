# Generate an MSVC import library for xrt_coreutil.dll (shipped with the AMD NPU
# driver, which provides no .lib). Mirrors the mlir-aie Windows host guide:
#   dumpbin /exports xrt_coreutil.dll  ->  .def  ->  lib /def:... /machine:x64
function(ggml_xdna_make_implib dll implib)
    if (EXISTS "${implib}")
        return()
    endif()

    get_filename_component(_cxx_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(GGML_XDNA_DUMPBIN NAMES dumpbin HINTS "${_cxx_bindir}")
    find_program(GGML_XDNA_LIB_EXE NAMES lib     HINTS "${_cxx_bindir}")
    if (NOT GGML_XDNA_DUMPBIN OR NOT GGML_XDNA_LIB_EXE)
        message(FATAL_ERROR "ggml-xdna: dumpbin/lib not found (needed to generate ${implib} from ${dll}). "
                            "Build from a Visual Studio developer shell, or pass -DGGML_XDNA_XRT_LIB=<path to xrt_coreutil.lib>.")
    endif()

    execute_process(COMMAND "${GGML_XDNA_DUMPBIN}" /nologo /exports "${dll}"
                    OUTPUT_VARIABLE _exports RESULT_VARIABLE _rc)
    if (NOT _rc EQUAL 0)
        message(FATAL_ERROR "ggml-xdna: dumpbin failed on ${dll}")
    endif()

    # export table rows: "<ordinal> <hint> <RVA> <decorated name>"
    string(REGEX MATCHALL "\n +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +[^ \r\n]+" _rows "${_exports}")
    set(_def "LIBRARY xrt_coreutil\nEXPORTS\n")
    set(_count 0)
    foreach(_row IN LISTS _rows)
        string(REGEX REPLACE "\n +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +([^ \r\n]+)" "\\1" _name "${_row}")
        string(APPEND _def "${_name}\n")
        math(EXPR _count "${_count} + 1")
    endforeach()
    if (_count EQUAL 0)
        message(FATAL_ERROR "ggml-xdna: no exports parsed from ${dll}")
    endif()

    file(WRITE "${implib}.def" "${_def}")
    execute_process(COMMAND "${GGML_XDNA_LIB_EXE}" /nologo "/def:${implib}.def" /machine:x64 "/out:${implib}"
                    RESULT_VARIABLE _rc OUTPUT_QUIET)
    if (NOT _rc EQUAL 0 OR NOT EXISTS "${implib}")
        message(FATAL_ERROR "ggml-xdna: lib.exe failed to create ${implib}")
    endif()
    message(STATUS "ggml-xdna: generated ${implib} (${_count} exports) from ${dll}")
endfunction()
