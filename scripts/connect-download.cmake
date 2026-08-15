# download the prebuilt llama-connect binary for the given platform
# the release assets are made by: https://github.com/ggml-org/llama-connect/blob/master/.github/workflows/build.yml

cmake_minimum_required(VERSION 3.18)

set(REPO     "" CACHE STRING "GitHub repository to download from (owner/name)")
set(VERSION  "" CACHE STRING "Release tag to download, or 'latest'")
set(PLATFORM "" CACHE STRING "Platform suffix of the release asset (ex: linux-x64)")
set(OUT_DIR  "" CACHE STRING "Directory to place the binary into")
set(WORK_DIR "" CACHE STRING "Scratch directory used for the download")

if (PLATFORM MATCHES "^win-")
    set(ASSET  "llama-connect-${PLATFORM}.zip")
    set(BINARY "llama-connect.exe")
else()
    set(ASSET  "llama-connect-${PLATFORM}.tar.gz")
    set(BINARY "llama-connect")
endif()

if (VERSION STREQUAL "latest")
    set(URL "https://github.com/${REPO}/releases/latest/download/${ASSET}")
else()
    set(URL "https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}")
endif()

set(DST     "${OUT_DIR}/${BINARY}")
set(ARCHIVE "${WORK_DIR}/${ASSET}")
set(STAMP   "${WORK_DIR}/${BINARY}.url")

# skip if already downloaded from the same URL
if (EXISTS "${DST}" AND EXISTS "${STAMP}")
    file(READ "${STAMP}" prev_url)
    if (prev_url STREQUAL "${URL}")
        return()
    endif()
endif()

file(REMOVE "${STAMP}")
file(MAKE_DIRECTORY "${WORK_DIR}")

message(STATUS "llama-connect: downloading ${URL}")

file(DOWNLOAD "${URL}" "${ARCHIVE}" STATUS status TLS_VERIFY ON)

list(GET status 0 code)
list(GET status 1 msg)

if (NOT code EQUAL 0)
    file(REMOVE "${ARCHIVE}")
    message(FATAL_ERROR
        "llama-connect: failed to download ${URL} (${code}: ${msg})\n"
        "  use -DLLAMA_CONNECT_VERSION=<tag> to select another release, "
        "or -DLLAMA_CONNECT=OFF to skip this tool")
endif()

set(EXTRACT_DIR "${WORK_DIR}/extract")

file(REMOVE_RECURSE "${EXTRACT_DIR}")
file(MAKE_DIRECTORY "${EXTRACT_DIR}")
file(ARCHIVE_EXTRACT INPUT "${ARCHIVE}" DESTINATION "${EXTRACT_DIR}")

if (NOT EXISTS "${EXTRACT_DIR}/${BINARY}")
    message(FATAL_ERROR "llama-connect: ${ASSET} does not contain ${BINARY}")
endif()

file(MAKE_DIRECTORY "${OUT_DIR}")
file(COPY "${EXTRACT_DIR}/${BINARY}" DESTINATION "${OUT_DIR}"
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

file(WRITE "${STAMP}" "${URL}")

message(STATUS "llama-connect: ready at ${DST}")
