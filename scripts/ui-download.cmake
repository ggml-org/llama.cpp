# Download UI assets from Hugging Face Bucket at build time
# Usage: cmake -DPUBLIC_DIR=... -DHF_BUCKET=... -DHF_VERSION=... -DASSETS="a;b;c" -P scripts/ui-download.cmake
#
# Asset provisioning priority:
#   1. Pre-built assets already in PUBLIC_DIR (cached from a previous run)
#   2. Local npm build (if NPM_DIR is provided and has package.json)
#   3. Hugging Face Bucket download (version-specific, then 'latest' fallback)

cmake_minimum_required(VERSION 3.16)

set(PUBLIC_DIR   "" CACHE STRING "Directory to store/download assets")
set(HF_BUCKET    "" CACHE STRING "Hugging Face bucket name")
set(HF_VERSION   "" CACHE STRING "Version to download (empty = resolve from git)")
set(ASSETS       "" CACHE STRING "Plus-separated list of asset filenames (+)")
set(STAMP_FILE   "" CACHE STRING "Stamp file to create on success (optional)")
set(SOURCE_DIR   "" CACHE STRING "Project source root (to resolve version from git)")
set(NPM_DIR      "" CACHE STRING "UI source directory (to run npm build)")
set(HF_ENABLED   "" CACHE STRING "Whether to allow HF Bucket download (ON/OFF)")

# ---------------------------------------------------------------------------
# 1. Resolve version from git if not provided at configure time
# ---------------------------------------------------------------------------
set(RESOLVED_VERSION "${HF_VERSION}")
if("${RESOLVED_VERSION}" STREQUAL "" AND NOT "${SOURCE_DIR}" STREQUAL "")
    if(EXISTS "${SOURCE_DIR}/cmake/build-info.cmake")
        include("${SOURCE_DIR}/cmake/build-info.cmake")
        if(NOT "${BUILD_NUMBER}" STREQUAL "" AND NOT BUILD_NUMBER EQUAL 0)
            set(RESOLVED_VERSION "b${BUILD_NUMBER}")
            message(STATUS "UI: resolved version from git: ${RESOLVED_VERSION}")
        endif()
    endif()
endif()

# Convert + back to CMake list (+ is used as separator instead of ; to
# avoid platform-specific escaping issues when passing via -D arguments)
string(REGEX REPLACE "\\+" ";" ASSETS "${ASSETS}")

# ---------------------------------------------------------------------------
# 2. Check stamp freshness — re-download if resolved version changed
# ---------------------------------------------------------------------------
set(FORCE_REBUILD FALSE)
if(NOT "${STAMP_FILE}" STREQUAL "" AND EXISTS "${STAMP_FILE}")
    file(READ "${STAMP_FILE}" STAMPED_VERSION)
    string(STRIP "${STAMPED_VERSION}" STAMPED_VERSION)
    if(NOT "${STAMPED_VERSION}" STREQUAL "${RESOLVED_VERSION}")
        message(STATUS "UI: version changed (${STAMPED_VERSION} -> ${RESOLVED_VERSION}), re-building")
        set(FORCE_REBUILD TRUE)
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. Check if assets already exist (cached from a previous run)
# ---------------------------------------------------------------------------
set(ALL_EXISTS TRUE)
foreach(asset ${ASSETS})
    if(NOT EXISTS "${PUBLIC_DIR}/${asset}")
        set(ALL_EXISTS FALSE)
        break()
    endif()
endforeach()

if(ALL_EXISTS AND NOT FORCE_REBUILD)
    message(STATUS "UI: all assets already exist in ${PUBLIC_DIR}, skipping")
    return()
endif()

file(MAKE_DIRECTORY "${PUBLIC_DIR}")

# ---------------------------------------------------------------------------
# 4. Priority 2: build from source via npm (fast path for developers)
# ---------------------------------------------------------------------------
set(PROVISION_SUCCESS FALSE)

if(NOT PROVISION_SUCCESS AND NOT "${NPM_DIR}" STREQUAL "")
    if(EXISTS "${NPM_DIR}/package.json")
        # Check if npm is available before attempting npm build
        find_program(NPM_EXECUTABLE npm)
        if(NPM_EXECUTABLE)
            message(STATUS "UI: building from source in ${NPM_DIR}")

            # Run npm install if node_modules is missing
            if(NOT EXISTS "${NPM_DIR}/node_modules")
                message(STATUS "UI: running npm install (first time)")
                execute_process(
                    COMMAND ${NPM_EXECUTABLE} install
                    WORKING_DIRECTORY "${NPM_DIR}"
                    RESULT_VARIABLE NPM_INSTALL_RESULT
                    OUTPUT_VARIABLE NPM_OUT
                    ERROR_VARIABLE  NPM_ERR
                )
                if(NOT NPM_INSTALL_RESULT EQUAL 0)
                    message(STATUS "UI: npm install failed (${NPM_INSTALL_RESULT}), falling back to download")
                    message(STATUS "  stderr: ${NPM_ERR}")
                endif()
            endif()

            # Run the build
            execute_process(
                COMMAND ${NPM_EXECUTABLE} run build
                WORKING_DIRECTORY "${NPM_DIR}"
                RESULT_VARIABLE NPM_BUILD_RESULT
                OUTPUT_VARIABLE NPM_OUT
                ERROR_VARIABLE  NPM_ERR
            )

            if(NPM_BUILD_RESULT EQUAL 0)
                # Verify that the expected assets were produced
                set(ALL_BUILT TRUE)
                foreach(asset ${ASSETS})
                    if(NOT EXISTS "${PUBLIC_DIR}/${asset}")
                        set(ALL_BUILT FALSE)
                        break()
                    endif()
                endforeach()

                if(ALL_BUILT)
                    message(STATUS "UI: local npm build succeeded")
                    set(PROVISION_SUCCESS TRUE)
                else()
                    message(STATUS "UI: npm build completed but assets missing from ${PUBLIC_DIR}, falling back to download")
                endif()
            else()
                message(STATUS "UI: npm build failed (${NPM_BUILD_RESULT}), falling back to download")
                message(STATUS "  stderr: ${NPM_ERR}")
            endif()
        else()
            message(STATUS "UI: npm not found, skipping npm build and trying HF Bucket download")
        endif()
    else()
        message(STATUS "UI: NPM_DIR (${NPM_DIR}) has no package.json, skipping npm build")
    endif()
endif()

# ---------------------------------------------------------------------------
# 5. Priority 3: download from Hugging Face Bucket (if enabled)
# ---------------------------------------------------------------------------
if(NOT PROVISION_SUCCESS AND HF_ENABLED)
    # Build list of URLs to try — version-specific first, then 'latest'
    set(URL_ENTRIES "")
    if(NOT "${RESOLVED_VERSION}" STREQUAL "")
        list(APPEND URL_ENTRIES
            "version:https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve/${RESOLVED_VERSION}")
    endif()
    list(APPEND URL_ENTRIES
        "latest:https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve/latest")

    foreach(entry ${URL_ENTRIES})
        string(REGEX REPLACE "^([^:]+):.*$" "\\1" url_label "${entry}")
        string(REGEX REPLACE "^[^:]+:(.*)$" "\\1" base_url "${entry}")

        message(STATUS "UI: downloading assets from ${url_label}: ${base_url}")

        # Download each asset
        set(ALL_OK TRUE)
        foreach(asset ${ASSETS})
            set(download_url "${base_url}/${asset}?download=true")
            set(download_path "${PUBLIC_DIR}/${asset}")
            file(DOWNLOAD "${download_url}" "${download_path}"
                STATUS download_status TIMEOUT 60
            )
            list(GET download_status 0 download_result)
            if(NOT download_result EQUAL 0)
                list(GET download_status 1 error_message)
                message(STATUS "UI: failed to download ${asset} from ${url_label}: ${error_message}")
                set(ALL_OK FALSE)
                break()
            endif()
            message(STATUS "UI: downloaded ${asset}")
        endforeach()

        if(NOT ALL_OK)
            continue()
        endif()

        # Verify checksums if the server provides them
        file(DOWNLOAD "${base_url}/checksums.txt?download=true"
            "${PUBLIC_DIR}/checksums.txt"
            STATUS checksum_status TIMEOUT 30
        )
        list(GET checksum_status 0 checksum_result)
        if(checksum_result EQUAL 0)
            message(STATUS "UI: verifying checksums...")
            file(STRINGS "${PUBLIC_DIR}/checksums.txt" CHECKSUMS_CONTENT)
            foreach(asset ${ASSETS})
                set(download_path "${PUBLIC_DIR}/${asset}")
                file(SHA256 "${download_path}" asset_hash)
                string(TOLOWER "${asset_hash}" EXPECTED_HASH_LOWER)
                string(REGEX MATCH "${EXPECTED_HASH_LOWER}[ \\t]+${asset}" CHECKSUM_LINE "${CHECKSUMS_CONTENT}")
                if(NOT CHECKSUM_LINE)
                    message(WARNING "UI: checksum verification failed for ${asset}")
                    set(ALL_OK FALSE)
                    break()
                endif()
            endforeach()
            if(ALL_OK)
                message(STATUS "UI: all checksums verified")
            endif()
        endif()

        if(ALL_OK)
            set(PROVISION_SUCCESS TRUE)
            break()
        endif()
    endforeach()

    if(PROVISION_SUCCESS)
        message(STATUS "UI: provisioning complete")
    else()
        message(WARNING "UI: failed to download assets from HF Bucket (${HF_BUCKET})")
    endif()
endif()

# ---------------------------------------------------------------------------
# 6. Write stamp file on success (stores resolved version for freshness check)
# ---------------------------------------------------------------------------
if(PROVISION_SUCCESS)
    if(NOT "${STAMP_FILE}" STREQUAL "")
        file(WRITE "${STAMP_FILE}" "${RESOLVED_VERSION}")
    endif()
else()
    message(WARNING "UI: no source available. Neither local build (${NPM_DIR}) nor HF Bucket download succeeded.")
    message(WARNING "UI: building server without embedded UI. Set LLAMA_BUILD_UI=OFF to suppress this warning.")
endif()
