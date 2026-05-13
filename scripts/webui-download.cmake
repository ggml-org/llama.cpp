# Download webui assets from Hugging Face Bucket at build time
# Usage: cmake -DPUBLIC_DIR=... -DHF_BUCKET=... -DHF_VERSION=... -DASSETS="a;b;c" -P scripts/webui-download.cmake
#
# Downloads from HF Bucket if assets don't already exist locally.
# Tries version-specific URL first, then falls back to 'latest'.

cmake_minimum_required(VERSION 3.16)

set(PUBLIC_DIR   "" CACHE STRING "Directory to store downloaded assets")
set(HF_BUCKET    "" CACHE STRING "Hugging Face bucket name")
set(HF_VERSION   "" CACHE STRING "Version to download (empty = resolve from git)")
set(ASSETS       "" CACHE STRING "Semicolon-separated list of asset filenames")
set(STAMP_FILE   "" CACHE STRING "Stamp file to create on success (optional)")
set(SOURCE_DIR   "" CACHE STRING "Project source root (to resolve version from git)")

# If no version was provided at configure time, try to resolve it from
# the git commit count at build time via cmake/build-info.cmake.
# This covers the case where LLAMA_BUILD_NUMBER was not set explicitly.
set(RESOLVED_VERSION "${HF_VERSION}")
if("${RESOLVED_VERSION}" STREQUAL "" AND NOT "${SOURCE_DIR}" STREQUAL "")
    if(EXISTS "${SOURCE_DIR}/cmake/build-info.cmake")
        include("${SOURCE_DIR}/cmake/build-info.cmake")
        if(NOT "${BUILD_NUMBER}" STREQUAL "" AND NOT BUILD_NUMBER EQUAL 0)
            set(RESOLVED_VERSION "${BUILD_NUMBER}")
            message(STATUS "WebUI: resolved version from git: ${RESOLVED_VERSION}")
        endif()
    endif()
endif()

# If a stamp file exists from a previous download, check whether the
# resolved version has changed (e.g. after git pull). If so, re-download.
set(FORCE_DOWNLOAD FALSE)
if(NOT "${STAMP_FILE}" STREQUAL "" AND EXISTS "${STAMP_FILE}")
    file(READ "${STAMP_FILE}" STAMPED_VERSION)
    string(STRIP "${STAMPED_VERSION}" STAMPED_VERSION)
    if(NOT "${STAMPED_VERSION}" STREQUAL "${RESOLVED_VERSION}")
        message(STATUS "WebUI: version changed (${STAMPED_VERSION} -> ${RESOLVED_VERSION}), re-downloading")
        set(FORCE_DOWNLOAD TRUE)
    endif()
endif()

# Check if all assets already exist locally
set(ALL_EXISTS TRUE)
foreach(asset ${ASSETS})
    if(NOT EXISTS "${PUBLIC_DIR}/${asset}")
        set(ALL_EXISTS FALSE)
        break()
    endif()
endforeach()

if(ALL_EXISTS AND NOT FORCE_DOWNLOAD)
    message(STATUS "WebUI: all assets already exist in ${PUBLIC_DIR}, skipping download")
    return()
endif()

# Ensure output directory exists
file(MAKE_DIRECTORY "${PUBLIC_DIR}")

# Build list of URLs to try — version-specific first, then 'latest'
set(URL_ENTRIES "")
if(NOT "${RESOLVED_VERSION}" STREQUAL "")
    list(APPEND URL_ENTRIES
        "version:https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve/${RESOLVED_VERSION}")
endif()
list(APPEND URL_ENTRIES
    "latest:https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve/latest")

set(DOWNLOAD_SUCCESS FALSE)

foreach(entry ${URL_ENTRIES})
    string(REGEX REPLACE "^([^:]+):.*$" "\\1" url_label "${entry}")
    string(REGEX REPLACE "^[^:]+:(.*)$" "\\1" base_url "${entry}")

    message(STATUS "WebUI: trying ${url_label}: ${base_url}")

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
            message(STATUS "WebUI: failed to download ${asset} from ${url_label}: ${error_message}")
            set(ALL_OK FALSE)
            break()
        endif()
        message(STATUS "WebUI: downloaded ${asset}")
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
        message(STATUS "WebUI: verifying checksums...")
        file(STRINGS "${PUBLIC_DIR}/checksums.txt" CHECKSUMS_CONTENT)
        foreach(asset ${ASSETS})
            set(download_path "${PUBLIC_DIR}/${asset}")
            file(SHA256 "${download_path}" asset_hash)
            string(TOUPPER "${asset_hash}" EXPECTED_HASH_UPPER)
            string(REGEX MATCH "${EXPECTED_HASH_UPPER}[ \\t]+${asset}" CHECKSUM_LINE "${CHECKSUMS_CONTENT}")
            if(NOT CHECKSUM_LINE)
                message(WARNING "WebUI: checksum verification failed for ${asset}")
                set(ALL_OK FALSE)
                break()
            endif()
        endforeach()
        if(ALL_OK)
            message(STATUS "WebUI: all checksums verified")
        endif()
    endif()

    if(ALL_OK)
        set(DOWNLOAD_SUCCESS TRUE)
        break()
    endif()
endforeach()

if(DOWNLOAD_SUCCESS)
    if(NOT "${STAMP_FILE}" STREQUAL "")
        file(WRITE "${STAMP_FILE}" "${RESOLVED_VERSION}")
    endif()
    message(STATUS "WebUI: download complete")
else()
    message(WARNING "WebUI: failed to download assets from HF Bucket (${HF_BUCKET})")
endif()
