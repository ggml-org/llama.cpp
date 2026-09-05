# Provision UI assets and generate ui.cpp/ui.h.
#
# Asset provisioning priority:
#   1. Pre-built assets in SRC_DIST_DIR (manually built by user)
#   2. If BUILD_UI=ON: npm build
#   3. If above did not produce assets and HF_ENABLED=ON: HF Bucket download
#      of dist.tar.gz (verified against dist.tar.gz.sha256)

cmake_minimum_required(VERSION 3.18)

set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(LLAMA_SOURCE_DIR  "" CACHE STRING "Project source root (to resolve version from git)")
set(HF_BUCKET         "" CACHE STRING "Hugging Face bucket name")
set(HF_VERSION        "" CACHE STRING "Version to download (empty = resolve from git)")
set(HF_ENABLED        "" CACHE STRING "Whether to allow HF Bucket download (ON/OFF)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")
set(LLAMA_UI_GZIP     "" CACHE STRING "Apply gzip compress to assets to save bandwidth")

set(DIST_DIR     "${UI_BINARY_DIR}/dist")
set(SRC_DIST_DIR "${UI_SOURCE_DIR}/dist")
set(WORK_DIR     "${UI_BINARY_DIR}/ui-src")
set(STAMP_FILE   "${UI_BINARY_DIR}/.ui-stamp")
set(UI_CPP       "${UI_BINARY_DIR}/ui.cpp")
set(UI_H         "${UI_BINARY_DIR}/ui.h")

function(mime_from_ext name out_var)
    string(FIND "${name}" "." ext REVERSE)
    if(ext GREATER -1)
        string(SUBSTRING "${name}" ${ext} -1 ext_full)
        string(SUBSTRING "${ext_full}" 1 -1 ext_str)
    else()
        set(ext_str "")
    endif()
    if(ext_str STREQUAL "html")
        set(m "text/html; charset=utf-8")
    elseif(ext_str STREQUAL "css")
        set(m "text/css")
    elseif(ext_str STREQUAL "js")
        set(m "application/javascript")
    elseif(ext_str STREQUAL "json")
        set(m "application/json")
    elseif(ext_str STREQUAL "webmanifest")
        set(m "application/manifest+json")
    elseif(ext_str STREQUAL "svg")
        set(m "image/svg+xml")
    elseif(ext_str STREQUAL "png")
        set(m "image/png")
    elseif(ext_str STREQUAL "jpg" OR ext_str STREQUAL "jpeg")
        set(m "image/jpeg")
    elseif(ext_str STREQUAL "ico")
        set(m "image/x-icon")
    elseif(ext_str STREQUAL "woff")
        set(m "font/woff")
    elseif(ext_str STREQUAL "woff2")
        set(m "font/woff2")
    else()
        set(m "application/octet-stream")
    endif()
    set(${out_var} "${m}" PARENT_SCOPE)
endfunction()

# Fail when a dist tree is present but is missing files the UI needs at
# runtime; catches truncated/stale asset trees early with a useful message.
function(ui_validate_assets files in_dir)
    list(LENGTH files n_assets)
    if(n_assets EQUAL 0)
        return()
    endif()

    set(found_index FALSE)
    set(found_manifest FALSE)
    set(found_sw FALSE)
    set(found_build_json FALSE)
    set(found_version_json FALSE)
    set(found_bundle_js FALSE)
    set(found_bundle_css FALSE)
    set(found_workbox_js FALSE)

    foreach(f ${files})
        get_filename_component(base "${f}" NAME)
        if(base STREQUAL "index.html")
            set(found_index TRUE)
        elseif(base STREQUAL "manifest.webmanifest")
            set(found_manifest TRUE)
        elseif(base STREQUAL "sw.js")
            set(found_sw TRUE)
        elseif(base STREQUAL "build.json")
            set(found_build_json TRUE)
        elseif(base STREQUAL "version.json")
            set(found_version_json TRUE)
        elseif(base MATCHES "^bundle.*\\.js$")
            set(found_bundle_js TRUE)
        elseif(base MATCHES "^bundle.*\\.css$")
            set(found_bundle_css TRUE)
        elseif(base MATCHES "^workbox.*\\.js$")
            set(found_workbox_js TRUE)
        endif()
    endforeach()

    set(missing "")
    if(NOT found_index)
        list(APPEND missing "index.html")
    endif()
    if(NOT found_manifest)
        list(APPEND missing "manifest.webmanifest")
    endif()
    if(NOT found_sw)
        list(APPEND missing "sw.js")
    endif()
    if(NOT found_build_json)
        list(APPEND missing "build.json")
    endif()
    if(NOT found_version_json)
        list(APPEND missing "version.json")
    endif()
    if(NOT found_bundle_js)
        list(APPEND missing "bundle[hash].js")
    endif()
    if(NOT found_bundle_css)
        list(APPEND missing "bundle[hash].css")
    endif()
    if(NOT found_workbox_js)
        list(APPEND missing "workbox[hash].js")
    endif()

    if(missing)
        set(listing "")
        foreach(f ${files})
            string(APPEND listing "    ${f}\n")
        endforeach()
        set(missing_list "")
        foreach(m ${missing})
            string(APPEND missing_list "    ${m}\n")
        endforeach()
        message(FATAL_ERROR
            "UI: current asset files:\n${listing}"
            "UI: missing required asset(s):\n${missing_list}"
            "UI: hint: try cleaning your build directory: ${in_dir}")
    endif()
endfunction()

# Generate ui.cpp/ui.h embedding every file of ${dist_dir} (empty table when
# it has no index.html). When LLAMA_UI_GZIP is enabled, assets are compressed
# first and served pre-gzipped (llama_ui_use_gzip()).
function(emit_files dist_dir)
    set(embed_dir "${dist_dir}")
    set(use_gzip FALSE)

    if(EXISTS "${dist_dir}/index.html")
        if(EXISTS "${dist_dir}/_gzip")
            # a _gzip tree inside dist_dir can only be a leftover from an
            # older version of this script that staged it there
            file(REMOVE_RECURSE "${dist_dir}/_gzip")
            message(STATUS "UI: removed stale gzip tree ${dist_dir}/_gzip")
        endif()
        if(LLAMA_UI_GZIP)
            # Compress every asset into a parallel _gzip/ tree under the build
            # directory (never write into the source or dist tree); the
            # structure stays the same: /abc/def --> /_gzip/abc/def.
            # FORMAT raw produces a bare gzip stream (no archive container)
            # that can be served with Content-Encoding: gzip. SOURCE_DATE_EPOCH
            # zeroes the header timestamp so identical inputs give identical
            # bytes (and therefore stable ETags) on every machine.
            if(NOT DEFINED ENV{SOURCE_DATE_EPOCH})
                set(ENV{SOURCE_DATE_EPOCH} 0)
            endif()
            set(gzip_root "${UI_BINARY_DIR}/ui-gzip")
            set(gzip_dir  "${gzip_root}/_gzip")
            file(REMOVE_RECURSE "${gzip_root}")
            file(GLOB_RECURSE all_files RELATIVE "${dist_dir}" "${dist_dir}/*")
            list(FILTER all_files EXCLUDE REGEX "^_gzip/")
            foreach(f ${all_files})
                get_filename_component(asset_path "${dist_dir}/${f}" REALPATH)
                get_filename_component(dst_dir "${gzip_dir}/${f}" DIRECTORY)
                file(MAKE_DIRECTORY "${dst_dir}")
                file(ARCHIVE_CREATE
                    OUTPUT "${gzip_dir}/${f}"
                    PATHS "${asset_path}"
                    FORMAT raw
                    COMPRESSION GZip
                )
            endforeach()
            message(STATUS "UI: gzip compression applied (${gzip_dir})")
            set(embed_dir "${gzip_dir}")
            set(use_gzip TRUE)
        endif()
    endif()

    set(assets "")
    if(EXISTS "${embed_dir}/index.html")
        file(GLOB_RECURSE assets RELATIVE "${embed_dir}" "${embed_dir}/*")
        list(FILTER assets EXCLUDE REGEX "^_gzip/")
        list(SORT assets)
        ui_validate_assets("${assets}" "${embed_dir}")
    endif()

    list(LENGTH assets n_assets)

    # Only the per-asset data arrays and table rows are built here; all
    # static C++ lives in the ui.h.in / ui.cpp.in templates. configure_file
    # rewrites an output only when its contents change, so the library is
    # not recompiled needlessly. @ONLY keeps ${...} in the content literal;
    # mime types come from a fixed list.
    set(ASSET_ARRAYS "")
    set(ASSET_TABLE "")
    set(idx 0)

    foreach(f IN LISTS assets)
        file(READ "${embed_dir}/${f}" hex HEX)
        if(hex STREQUAL "")
            message(FATAL_ERROR "UI: empty file: ${embed_dir}/${f}")
        endif()

        string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
        file(SHA256 "${embed_dir}/${f}" etag)
        mime_from_ext("${f}" mime)

        string(APPEND ASSET_ARRAYS
            "static const unsigned char asset_${idx}[] = {${bytes}};\n")

        string(APPEND ASSET_TABLE
            "    { \"${f}\", asset_${idx}, sizeof(asset_${idx}), \"\\\"${etag}\\\"\", \"${mime}\" },\n")

        math(EXPR idx "${idx} + 1")
    endforeach()

    set(LLAMA_UI_HAS_ASSETS 0)
    if(n_assets GREATER 0)
        set(LLAMA_UI_HAS_ASSETS 1)
    endif()
    set(N_ASSETS "${n_assets}")
    set(USE_GZIP false)
    if(use_gzip)
        set(USE_GZIP true)
    endif()

    set(UI_TEMPLATE_DIR "${LLAMA_SOURCE_DIR}/tools/ui")
    configure_file("${UI_TEMPLATE_DIR}/ui.h.in"   "${UI_H}"   @ONLY)
    configure_file("${UI_TEMPLATE_DIR}/ui.cpp.in" "${UI_CPP}" @ONLY)
    message(STATUS "UI: embedded ${n_assets} assets")
endfunction()

function(npm_build_should_skip out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    if(NOT EXISTS "${DIST_DIR}/index.html")
        return()
    endif()

    if(EXISTS "${STAMP_FILE}")
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

function(stage_sources)
    if(EXISTS "${WORK_DIR}")
        file(GLOB staged RELATIVE "${WORK_DIR}" "${WORK_DIR}/*")
        list(REMOVE_ITEM staged "node_modules")
        foreach(entry ${staged})
            file(REMOVE_RECURSE "${WORK_DIR}/${entry}")
        endforeach()
    endif()

    file(COPY "${UI_SOURCE_DIR}/"
        DESTINATION "${WORK_DIR}"
        NO_SOURCE_PERMISSIONS
        PATTERN "node_modules" EXCLUDE
    )
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

    if(CMAKE_HOST_WIN32)
        find_program(NPM_EXECUTABLE NAMES npm.cmd npm.bat npm)
    else()
        find_program(NPM_EXECUTABLE npm)
    endif()
    if(NOT NPM_EXECUTABLE)
        message(STATUS "UI: npm not found, skipping npm build")
        return()
    endif()

    stage_sources()

    # npm writes node_modules/.package-lock.json on every successful install,
    # so a package-lock.json newer than this marker means node_modules is stale
    set(NPM_MARKER "${WORK_DIR}/node_modules/.package-lock.json")
    set(need_install FALSE)
    if(NOT EXISTS "${NPM_MARKER}")
        set(need_install TRUE)
    else()
        file(TIMESTAMP "${WORK_DIR}/package-lock.json" lock_ts)
        file(TIMESTAMP "${NPM_MARKER}" marker_ts)
        if(lock_ts STRGREATER marker_ts)
            set(need_install TRUE)
        endif()
    endif()

    if(need_install)
        message(STATUS "UI: running npm ci")
        execute_process(
            COMMAND ${NPM_EXECUTABLE} ci
            WORKING_DIRECTORY "${WORK_DIR}"
            RESULT_VARIABLE rc
            ERROR_VARIABLE  err
        )
        if(NOT rc EQUAL 0)
            message(STATUS "UI: npm ci failed (${rc})")
            message(STATUS "  stderr: ${err}")
            return()
        endif()
    endif()

    file(MAKE_DIRECTORY "${DIST_DIR}")

    message(STATUS "UI: running npm run build, output -> ${DIST_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "LLAMA_UI_OUT_DIR=${DIST_DIR}" "LLAMA_UI_VERSION=${HF_VERSION}" "LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
                ${NPM_EXECUTABLE} run build
        WORKING_DIRECTORY "${WORK_DIR}"
        RESULT_VARIABLE rc
        ERROR_VARIABLE  err
    )
    if(NOT rc EQUAL 0)
        message(STATUS "UI: npm run build failed (${rc})")
        message(STATUS "  stderr: ${err}")
        return()
    endif()

    if(NOT EXISTS "${DIST_DIR}/index.html")
        message(STATUS "UI: npm build finished but assets missing in ${DIST_DIR}")
        return()
    endif()

    message(STATUS "UI: npm build succeeded")
    file(REMOVE "${STAMP_FILE}")
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

    set(archive "${UI_BINARY_DIR}/dist.tar.gz")

    # Use HF_TOKEN to benefit from higher rate limits
    set(auth_headers "")
    if(DEFINED ENV{HF_TOKEN} AND NOT "$ENV{HF_TOKEN}" STREQUAL "")
        list(APPEND auth_headers "HTTPHEADER" "Authorization: Bearer $ENV{HF_TOKEN}")
    endif()

    set(candidates "")
    if(NOT "${version}" STREQUAL "")
        list(APPEND candidates "${version}")
    endif()
    list(APPEND candidates "latest")

    foreach(resolved ${candidates})
        set(base "https://huggingface.co/buckets/${HF_BUCKET}/resolve/${resolved}")

        message(STATUS "UI: downloading from ${resolved}: ${base}/dist.tar.gz")

        file(DOWNLOAD "${base}/dist.tar.gz?download=true" "${archive}"
            STATUS status TIMEOUT 300 ${auth_headers}
        )
        list(GET status 0 rc)
        if(NOT rc EQUAL 0)
            list(GET status 1 errmsg)
            message(STATUS "UI: download dist.tar.gz from ${resolved} failed: ${errmsg}")
            continue()
        endif()

        file(DOWNLOAD "${base}/dist.tar.gz.sha256?download=true" "${archive}.sha256"
            STATUS status TIMEOUT 30 ${auth_headers}
        )
        list(GET status 0 rc)
        if(NOT rc EQUAL 0)
            list(GET status 1 errmsg)
            message(STATUS "UI: download dist.tar.gz.sha256 from ${resolved} failed: ${errmsg}")
            continue()
        endif()

        # Validate sha256 checkums
        file(READ "${archive}.sha256" expected)
        string(REGEX MATCH "^[0-9a-fA-F]+" expected "${expected}")
        string(TOLOWER "${expected}" expected)
        file(SHA256 "${archive}" actual)
        if("${expected}" STREQUAL "" OR NOT "${actual}" STREQUAL "${expected}")
            message(STATUS "UI: checksum mismatch for dist.tar.gz from ${resolved}")
            continue()
        endif()

        # Clear DIST_DIR to remove stale files first
        file(REMOVE_RECURSE "${DIST_DIR}")

        file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${DIST_DIR}")

        if(NOT EXISTS "${DIST_DIR}/index.html")
            message(STATUS "UI: archive from ${resolved} is missing required assets")
            continue()
        endif()

        message(STATUS "UI: archive verified and extracted")
        set(${out_var}      TRUE          PARENT_SCOPE)
        set(${out_resolved} "${resolved}" PARENT_SCOPE)
        return()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# 1. Priority 1: pre-built assets supplied in tools/ui/dist
# ---------------------------------------------------------------------------
if(EXISTS "${SRC_DIST_DIR}/index.html")
    message(STATUS "UI: using pre-built assets from ${SRC_DIST_DIR}")
    emit_files("${SRC_DIST_DIR}")
    return()
endif()

# ---------------------------------------------------------------------------
# 2. Priority 2: npm build (if BUILD_UI=ON)
# ---------------------------------------------------------------------------
set(provisioned FALSE)

if(BUILD_UI)
    # Resolve version from git build-info if not explicitly set
    resolve_version(HF_VERSION)
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

    set(have_assets FALSE)
    if(EXISTS "${DIST_DIR}/index.html")
        set(have_assets TRUE)
    endif()
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
    if(EXISTS "${DIST_DIR}/index.html")
        message(WARNING "UI: provisioning failed; embedding stale assets from ${DIST_DIR}")
    else()
        message(WARNING "UI: no assets available - building without an embedded UI. "
                        "In a disconnected environment, download the pre-built UI "
                        "from a llama.cpp release at "
                        "https://github.com/ggml-org/llama.cpp/releases and "
                        "extract to tools/ui/dist.")
    endif()
endif()

emit_files("${DIST_DIR}")
