# Provision UI assets and generate ui.cpp/ui.h.
#
# Asset provisioning priority:
#   1. Pre-built assets in SRC_DIST_DIR (manually built by user)
#   2. If BUILD_UI=ON: npm build
#   3. If above did not produce assets and HF_ENABLED=ON: HF Bucket download

cmake_minimum_required(VERSION 3.16)

set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(LLAMA_SOURCE_DIR  "" CACHE STRING "Project source root (to resolve version from git)")
set(HF_BUCKET         "" CACHE STRING "Hugging Face bucket name")
set(HF_VERSION        "" CACHE STRING "Version to download (empty = resolve from git)")
set(HF_ENABLED        "" CACHE STRING "Whether to allow HF Bucket download (ON/OFF)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")

set(ASSETS
    bundle.css
    bundle.js
    index.html
    loading.html
)

set(DIST_DIR     "${UI_BINARY_DIR}/dist")
set(SRC_DIST_DIR "${UI_SOURCE_DIR}/dist")
set(STAMP_FILE   "${UI_BINARY_DIR}/.ui-stamp")
set(UI_CPP       "${UI_BINARY_DIR}/ui.cpp")
set(UI_H         "${UI_BINARY_DIR}/ui.h")

function(assets_present out_var)
    set(present TRUE)
    foreach(asset ${ASSETS})
        if(NOT EXISTS "${DIST_DIR}/${asset}")
            set(present FALSE)
            break()
        endif()
    endforeach()
    set(${out_var} ${present} PARENT_SCOPE)
endfunction()

function(copy_src_dist out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    foreach(asset ${ASSETS})
        if(NOT EXISTS "${SRC_DIST_DIR}/${asset}")
            return()
        endif()
    endforeach()

    file(MAKE_DIRECTORY "${DIST_DIR}")
    message(STATUS "UI: using pre-built assets from ${SRC_DIST_DIR}")
    foreach(asset ${ASSETS})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SRC_DIST_DIR}/${asset}" "${DIST_DIR}/${asset}"
        )
    endforeach()
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(npm_build_should_skip out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    assets_present(present)
    if(NOT present)
        return()
    endif()

    if(NOT EXISTS "${UI_SOURCE_DIR}/sources.cmake")
        return()
    endif()
    include("${UI_SOURCE_DIR}/sources.cmake")

    set(globs "")
    foreach(g ${UI_SOURCE_GLOBS})
        list(APPEND globs "${UI_SOURCE_DIR}/${g}")
    endforeach()
    file(GLOB_RECURSE sources ${globs})
    foreach(f ${UI_SOURCE_FILES})
        list(APPEND sources "${UI_SOURCE_DIR}/${f}")
    endforeach()

    file(TIMESTAMP "${DIST_DIR}/index.html" out_ts)

    foreach(s ${sources})
        if(NOT EXISTS "${s}")
            continue()
        endif()
        file(TIMESTAMP "${s}" s_ts)
        if(s_ts STRGREATER out_ts)
            return()
        endif()
    endforeach()

    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(npm_build out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    if(NOT EXISTS "${UI_SOURCE_DIR}/package.json")
        message(STATUS "UI: ${UI_SOURCE_DIR}/package.json not found, skipping npm")
        return()
    endif()

    npm_build_should_skip(skip)
    if(skip)
        message(STATUS "UI: npm output up-to-date, skipping build")
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()

    find_program(NPM_EXECUTABLE npm)
    if(NOT NPM_EXECUTABLE)
        message(STATUS "UI: npm not found, skipping npm build")
        return()
    endif()

    if(NOT EXISTS "${UI_SOURCE_DIR}/node_modules")
        message(STATUS "UI: running npm install (first time)")
        execute_process(
            COMMAND ${NPM_EXECUTABLE} install
            WORKING_DIRECTORY "${UI_SOURCE_DIR}"
            RESULT_VARIABLE rc
            ERROR_VARIABLE  err
        )
        if(NOT rc EQUAL 0)
            message(STATUS "UI: npm install failed (${rc})")
            message(STATUS "  stderr: ${err}")
            return()
        endif()
    endif()

    file(MAKE_DIRECTORY "${DIST_DIR}")

    message(STATUS "UI: running npm run build, output -> ${DIST_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "LLAMA_UI_OUT_DIR=${DIST_DIR}"
                ${NPM_EXECUTABLE} run build
        WORKING_DIRECTORY "${UI_SOURCE_DIR}"
        RESULT_VARIABLE rc
        ERROR_VARIABLE  err
    )
    if(NOT rc EQUAL 0)
        message(STATUS "UI: npm run build failed (${rc})")
        message(STATUS "  stderr: ${err}")
        return()
    endif()

    assets_present(present)
    if(NOT present)
        message(STATUS "UI: npm build finished but assets missing in ${DIST_DIR}")
        return()
    endif()

    message(STATUS "UI: npm build succeeded")
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(resolve_version out_var)
    if(NOT "${HF_VERSION}" STREQUAL "")
        set(${out_var} "${HF_VERSION}" PARENT_SCOPE)
        return()
    endif()

    if(EXISTS "${LLAMA_SOURCE_DIR}/cmake/build-info.cmake")
        include("${LLAMA_SOURCE_DIR}/cmake/build-info.cmake")
        if(NOT "${BUILD_NUMBER}" STREQUAL "" AND NOT BUILD_NUMBER EQUAL 0)
            set(${out_var} "b${BUILD_NUMBER}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(hf_download version out_var out_resolved)
    set(${out_var}      FALSE PARENT_SCOPE)
    set(${out_resolved} ""    PARENT_SCOPE)

    file(MAKE_DIRECTORY "${DIST_DIR}")

    set(candidates "")
    if(NOT "${version}" STREQUAL "")
        list(APPEND candidates "${version}")
    endif()
    list(APPEND candidates "latest")

    foreach(resolved ${candidates})
        set(base "https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve/${resolved}")

        message(STATUS "UI: downloading from ${resolved}: ${base}")

        set(ok TRUE)
        foreach(asset ${ASSETS})
            file(DOWNLOAD "${base}/${asset}?download=true" "${DIST_DIR}/${asset}"
                STATUS status TIMEOUT 60
            )
            list(GET status 0 rc)
            if(NOT rc EQUAL 0)
                list(GET status 1 errmsg)
                message(STATUS "UI: download ${asset} from ${resolved} failed: ${errmsg}")
                set(ok FALSE)
                break()
            endif()
            message(STATUS "UI: downloaded ${asset}")
        endforeach()

        if(NOT ok)
            continue()
        endif()

        # Best-effort checksum verification
        file(DOWNLOAD "${base}/checksums.txt?download=true" "${DIST_DIR}/checksums.txt"
            STATUS cs_status TIMEOUT 30
        )
        list(GET cs_status 0 cs_rc)
        if(cs_rc EQUAL 0)
            message(STATUS "UI: verifying checksums")
            file(STRINGS "${DIST_DIR}/checksums.txt" cs_lines)
            foreach(asset ${ASSETS})
                file(SHA256 "${DIST_DIR}/${asset}" h)
                string(TOLOWER "${h}" h)
                string(REGEX MATCH "${h}[ \t]+${asset}" m "${cs_lines}")
                if(NOT m)
                    message(WARNING "UI: checksum verification failed for ${asset}")
                    set(ok FALSE)
                    break()
                endif()
            endforeach()
            if(ok)
                message(STATUS "UI: all checksums verified")
            endif()
        endif()

        if(ok)
            set(${out_var}      TRUE         PARENT_SCOPE)
            set(${out_resolved} "${resolved}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

function(write_if_different path content)
    set(tmp "${path}.tmp")
    file(WRITE "${tmp}" "${content}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${tmp}" "${path}"
    )
    file(REMOVE "${tmp}")
endfunction()

function(emit_files)
    assets_present(present)

    set(h "#pragma once\n\n#include <stddef.h>\n\n")
    if(present)
        string(APPEND h "#define LLAMA_UI_HAS_ASSETS 1\n\n")
    endif()
    string(APPEND h "struct llama_ui_asset {\n")
    string(APPEND h "    const char *          name;\n")
    string(APPEND h "    const unsigned char * data;\n")
    string(APPEND h "    size_t                size;\n")
    string(APPEND h "};\n\n")
    string(APPEND h "const llama_ui_asset * llama_ui_find_asset(const char * name);\n")

    set(cpp "#include \"ui.h\"\n\n#include <cstring>\n\n")

    if(present)
        set(idx 0)
        foreach(asset ${ASSETS})
            file(READ "${DIST_DIR}/${asset}" hex HEX)
            string(LENGTH "${hex}" hex_len)
            math(EXPR len "${hex_len} / 2")
            string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
            string(APPEND cpp "static const unsigned char asset_${idx}_data[] = {${bytes}};\n")
            string(APPEND cpp "static const size_t        asset_${idx}_size = ${len};\n\n")
            math(EXPR idx "${idx} + 1")
        endforeach()

        string(APPEND cpp "static const llama_ui_asset g_assets[] = {\n")
        set(idx 0)
        foreach(asset ${ASSETS})
            string(APPEND cpp "    { \"${asset}\", asset_${idx}_data, asset_${idx}_size },\n")
            math(EXPR idx "${idx} + 1")
        endforeach()
        string(APPEND cpp "};\n\n")

        string(APPEND cpp "const llama_ui_asset * llama_ui_find_asset(const char * name) {\n")
        string(APPEND cpp "    for (const auto & a : g_assets) {\n")
        string(APPEND cpp "        if (std::strcmp(a.name, name) == 0) {\n")
        string(APPEND cpp "            return &a;\n")
        string(APPEND cpp "        }\n")
        string(APPEND cpp "    }\n")
        string(APPEND cpp "    return nullptr;\n")
        string(APPEND cpp "}\n")
    else()
        string(APPEND cpp "const llama_ui_asset * llama_ui_find_asset(const char *) {\n")
        string(APPEND cpp "    return nullptr;\n")
        string(APPEND cpp "}\n")
    endif()

    write_if_different("${UI_H}"   "${h}")
    write_if_different("${UI_CPP}" "${cpp}")
endfunction()

# ---------------------------------------------------------------------------
# 1. Priority 1: pre-built assets supplied in tools/ui/dist
# ---------------------------------------------------------------------------
copy_src_dist(SRC_OK)
if(SRC_OK)
    emit_files()
    return()
endif()

# ---------------------------------------------------------------------------
# 2. Priority 2: npm build (if BUILD_UI=ON)
# ---------------------------------------------------------------------------
set(provisioned FALSE)

if(BUILD_UI)
    npm_build(NPM_OK)
    if(NPM_OK)
        set(provisioned TRUE)
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. Priority 3: HF Bucket download (if npm did not produce assets and HF_ENABLED=ON)
# ---------------------------------------------------------------------------
if(NOT provisioned AND HF_ENABLED)
    resolve_version(VERSION)

    set(stamp_ok FALSE)
    if(EXISTS "${STAMP_FILE}" AND NOT "${VERSION}" STREQUAL "")
        file(READ "${STAMP_FILE}" stamped)
        string(STRIP "${stamped}" stamped)
        if("${stamped}" STREQUAL "${VERSION}")
            set(stamp_ok TRUE)
        endif()
    endif()

    assets_present(have_assets)
    if(stamp_ok AND have_assets)
        message(STATUS "UI: HF stamp '${stamped}' matches version, skipping HF fetch")
        set(provisioned TRUE)
    else()
        hf_download("${VERSION}" HF_OK HF_RESOLVED)
        if(HF_OK)
            file(WRITE "${STAMP_FILE}" "${HF_RESOLVED}")
            message(STATUS "UI: HF download succeeded, stamp updated (${HF_RESOLVED})")
            set(provisioned TRUE)
        else()
            message(STATUS "UI: HF download failed")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 4. Fallback: warn about stale or missing assets, then emit whatever we have
# ---------------------------------------------------------------------------
if(NOT provisioned)
    assets_present(have_assets)
    if(have_assets)
        message(WARNING "UI: provisioning failed; embedding stale assets from ${DIST_DIR}")
    else()
        message(WARNING "UI: no UI assets available — building without embedded UI")
    endif()
endif()

emit_files()
