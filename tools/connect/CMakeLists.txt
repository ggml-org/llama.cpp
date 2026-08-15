# llama-connect is developed in a separate repo, nothing is compiled here
# we only download the prebuilt binary, see scripts/connect-download.cmake

set(TARGET llama-connect)

set(LLAMA_CONNECT_REPO    "ggml-org/llama-connect" CACHE STRING "llama-connect: GitHub repository to download from")
set(LLAMA_CONNECT_VERSION "latest"                 CACHE STRING "llama-connect: release tag to download (ex: v0.0.1), or 'latest'")

# map the target platform to the release asset name
set(LLAMA_CONNECT_PLATFORM "")

if (APPLE)
    set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
    if (CMAKE_OSX_ARCHITECTURES)
        list(LENGTH CMAKE_OSX_ARCHITECTURES _narch)
        if (_narch EQUAL 1)
            set(_arch "${CMAKE_OSX_ARCHITECTURES}")
        else()
            set(_arch "universal") # no such asset, so the tool is skipped below
        endif()
    endif()

    if (_arch MATCHES "^(arm64|aarch64)$")
        set(LLAMA_CONNECT_PLATFORM "macos-arm64")
    elseif (_arch MATCHES "^(x86_64|AMD64|amd64|x64)$")
        set(LLAMA_CONNECT_PLATFORM "macos-x64")
    endif()
elseif (WIN32)
    # with Visual Studio generators, the target arch comes from -A
    set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
    if (CMAKE_GENERATOR_PLATFORM)
        set(_arch "${CMAKE_GENERATOR_PLATFORM}")
    endif()

    if (_arch MATCHES "^(x86_64|AMD64|amd64|x64)$")
        set(LLAMA_CONNECT_PLATFORM "win-x64")
    endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
        set(LLAMA_CONNECT_PLATFORM "linux-x64")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(LLAMA_CONNECT_PLATFORM "linux-arm64")
    endif()
endif()

if (LLAMA_CONNECT_PLATFORM STREQUAL "")
    message(WARNING "llama-connect: no prebuilt binary for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}, skipping")
    return()
endif()

if (LLAMA_CONNECT_PLATFORM MATCHES "^win-")
    set(LLAMA_CONNECT_BINARY "llama-connect.exe")
else()
    set(LLAMA_CONNECT_BINARY "llama-connect")
endif()

# multi-config generators put the binaries in a per-config subdir
get_property(LLAMA_CONNECT_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (LLAMA_CONNECT_MULTI_CONFIG)
    set(LLAMA_CONNECT_OUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
else()
    set(LLAMA_CONNECT_OUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

# the script is a no-op if the binary is already downloaded
add_custom_target(${TARGET} ALL
    BYPRODUCTS "${LLAMA_CONNECT_OUT_DIR}/${LLAMA_CONNECT_BINARY}"
    COMMAND ${CMAKE_COMMAND}
        "-DREPO=${LLAMA_CONNECT_REPO}"
        "-DVERSION=${LLAMA_CONNECT_VERSION}"
        "-DPLATFORM=${LLAMA_CONNECT_PLATFORM}"
        "-DOUT_DIR=${LLAMA_CONNECT_OUT_DIR}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${PROJECT_SOURCE_DIR}/scripts/connect-download.cmake"
    COMMENT "Downloading llama-connect (${LLAMA_CONNECT_PLATFORM}, ${LLAMA_CONNECT_VERSION})"
    VERBATIM
)

if (LLAMA_TOOLS_INSTALL)
    install(PROGRAMS "${LLAMA_CONNECT_OUT_DIR}/${LLAMA_CONNECT_BINARY}" TYPE BIN)
endif()
