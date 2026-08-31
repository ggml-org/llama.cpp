if(NOT DEFINED UI_ASSETS_SCRIPT OR NOT DEFINED LLAMA_UI_EMBED OR NOT DEFINED TEST_WORK_DIR)
    message(FATAL_ERROR "missing test configuration")
endif()

function(make_complete_dist dist_dir)
    file(WRITE "${dist_dir}/index.html" "<html></html>")
    set(probe_dir "${dist_dir}/../probe")
    file(MAKE_DIRECTORY "${probe_dir}")
    set(probe_args "${probe_dir}/probe.cpp" "${probe_dir}/probe.h" "${dist_dir}")

    execute_process(
        COMMAND "${LLAMA_UI_EMBED}" ${probe_args}
        RESULT_VARIABLE probe_result
        OUTPUT_VARIABLE probe_output
        ERROR_VARIABLE probe_error
    )
    if(probe_result EQUAL 0)
        return()
    endif()

    set(probe_log "${probe_output}${probe_error}")
    string(REGEX MATCH "missing required asset\\(s\\):\n([^\n]+\n)+hint:" missing_block "${probe_log}")
    if("${missing_block}" STREQUAL "")
        message(FATAL_ERROR "could not determine required UI assets:\n${probe_log}")
    endif()

    string(REGEX REPLACE "^missing required asset\\(s\\):\n" "" missing_block "${missing_block}")
    string(REGEX REPLACE "\nhint:$" "" missing_block "${missing_block}")
    string(REPLACE "\n" ";" missing_assets "${missing_block}")
    foreach(asset IN LISTS missing_assets)
        string(STRIP "${asset}" asset)
        string(REPLACE "[hash]" "abc123" asset "${asset}")
        get_filename_component(asset_dir "${dist_dir}/${asset}" DIRECTORY)
        file(MAKE_DIRECTORY "${asset_dir}")
        file(WRITE "${dist_dir}/${asset}" "test")
    endforeach()

    execute_process(
        COMMAND "${LLAMA_UI_EMBED}" ${probe_args}
        RESULT_VARIABLE probe_result
        OUTPUT_VARIABLE probe_output
        ERROR_VARIABLE probe_error
    )
    if(NOT probe_result EQUAL 0)
        message(FATAL_ERROR "derived UI assets are still incomplete:\n${probe_output}${probe_error}")
    endif()
endfunction()

function(test_assets name build_ui hf_enabled expected_result)
    set(test_dir   "${TEST_WORK_DIR}/${name}")
    set(source_dir "${test_dir}/source")
    set(binary_dir "${test_dir}/binary")

    file(REMOVE_RECURSE "${test_dir}")
    file(MAKE_DIRECTORY "${source_dir}/dist" "${binary_dir}")
    if(expected_result STREQUAL "embedded")
        make_complete_dist("${source_dir}/dist")
    else()
        file(WRITE "${source_dir}/dist/index.html" "<html></html>")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DUI_SOURCE_DIR=${source_dir}"
            "-DUI_BINARY_DIR=${binary_dir}"
            "-DLLAMA_SOURCE_DIR=${source_dir}"
            "-DLLAMA_BUILD_NUMBER=1"
            "-DHF_BUCKET=ggml-org/llama-ui"
            "-DHF_VERSION=b1"
            "-DHF_ENABLED=${hf_enabled}"
            "-DBUILD_UI=${build_ui}"
            "-DLLAMA_UI_EMBED=${LLAMA_UI_EMBED}"
            "-DLLAMA_UI_GZIP=OFF"
            -P "${UI_ASSETS_SCRIPT}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    if(expected_result STREQUAL "fatal")
        if(result EQUAL 0)
            message(FATAL_ERROR "${name} unexpectedly accepted incomplete assets")
        endif()
        if(NOT "${output}${error}" MATCHES "llama-ui-embed failed")
            message(FATAL_ERROR "${name} failed for an unexpected reason:\n${output}${error}")
        endif()
    else()
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "${name} failed unexpectedly:\n${output}${error}")
        endif()
        if(NOT EXISTS "${binary_dir}/ui.cpp" OR NOT EXISTS "${binary_dir}/ui.h")
            message(FATAL_ERROR "${name} did not generate UI sources")
        endif()
        file(READ "${binary_dir}/ui.cpp" ui_source)
        if(expected_result STREQUAL "empty" AND ui_source MATCHES "index\\.html")
            message(FATAL_ERROR "${name} embedded incomplete assets")
        endif()
        if(expected_result STREQUAL "empty" AND NOT "${output}${error}" MATCHES "building without an embedded UI")
            message(FATAL_ERROR "${name} did not warn about the incomplete assets")
        endif()
        if(expected_result STREQUAL "embedded" AND NOT ui_source MATCHES "index\\.html")
            message(FATAL_ERROR "${name} did not embed the valid local assets")
        endif()
    endif()
endfunction()

test_assets(provisioning-disabled OFF OFF empty)
test_assets(build-enabled         ON  OFF fatal)
# Priority 1 fails before the prebuilt path can access the network.
test_assets(prebuilt-enabled      OFF ON  fatal)
test_assets(valid-local-assets    OFF OFF embedded)
