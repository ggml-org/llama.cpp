include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

# --------------------------------------------------------------------------
# Web UI / UI build configuration
# --------------------------------------------------------------------------

# Deprecated: use LLAMA_UI_HF_BUCKET instead
# Usage: cmake -B build -DLLAMA_UI_HF_BUCKET=llama-ui
set(LLAMA_WEBUI_HF_BUCKET "llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt webui assets (deprecated: use LLAMA_UI_HF_BUCKET)")
set(LLAMA_UI_HF_BUCKET "llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt UI assets")

# Backward compat: forward old var to new one
if(DEFINED LLAMA_WEBUI_HF_BUCKET AND NOT DEFINED LLAMA_UI_HF_BUCKET)
    set(LLAMA_UI_HF_BUCKET ${LLAMA_WEBUI_HF_BUCKET})
elseif(DEFINED LLAMA_WEBUI_HF_BUCKET AND NOT "${LLAMA_WEBUI_HF_BUCKET}" STREQUAL "${LLAMA_UI_HF_BUCKET}")
    message(DEPRECATION "LLAMA_WEBUI_HF_BUCKET is deprecated, use LLAMA_UI_HF_BUCKET instead")
endif()

# Support both old (LLAMA_BUILD_WEBUI) and new (LLAMA_BUILD_UI) option names
if(LLAMA_BUILD_WEBUI OR LLAMA_BUILD_UI)
    if(LLAMA_BUILD_WEBUI AND NOT LLAMA_BUILD_UI)
        message(DEPRECATION "LLAMA_BUILD_WEBUI is deprecated, use LLAMA_BUILD_UI instead")
    endif()

    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    # Determine source of UI assets (priority: local > HF Bucket)
    set(UI_SOURCE "")
    set(UI_SOURCE_DIR "")

    # Priority 1: Check for local build output
    set(LOCAL_UI_DIR "${PROJECT_SOURCE_DIR}/build/tools/ui")

    # Verify all required assets exist before declaring local source valid
    set(ALL_ASSETS_PRESENT TRUE)
    foreach(asset ${PUBLIC_ASSETS})
        if(NOT EXISTS "${LOCAL_UI_DIR}/${asset}")
            set(ALL_ASSETS_PRESENT FALSE)
            break()
        endif()
    endforeach()

    if(ALL_ASSETS_PRESENT)
        set(UI_SOURCE "local")
        set(UI_SOURCE_DIR "${LOCAL_UI_DIR}")
        message(STATUS "UI: using local build from ${UI_SOURCE_DIR}")
    endif()

    # Priority 2: Build-time asset provisioning (npm build → HF Bucket fallback)
    if(NOT UI_SOURCE_DIR)
        # Environment variable takes precedence (e.g., from CI workflows)
        # Deprecated: use HF_UI_VERSION instead
        if(DEFINED ENV{HF_WEBUI_VERSION})
            set(HF_UI_VERSION "$ENV{HF_WEBUI_VERSION}")
            message(DEPRECATION "HF_WEBUI_VERSION env var is deprecated, use HF_UI_VERSION instead")
            # Validate against allowed characters to prevent CMake list separator
            # or path-traversal issues in stamp filenames and download URLs
            if(NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
                message(FATAL_ERROR "UI: invalid HF_WEBUI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
            endif()
        elseif(DEFINED ENV{HF_UI_VERSION})
            set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")
            # Validate against allowed characters to prevent CMake list separator
            # or path-traversal issues in stamp filenames and download URLs
            if(NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
                message(FATAL_ERROR "UI: invalid HF_UI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
            endif()
        elseif(DEFINED LLAMA_BUILD_NUMBER)
            set(HF_UI_VERSION "b${LLAMA_BUILD_NUMBER}")
            message(STATUS "UI: derived HF_UI_VERSION=b${LLAMA_BUILD_NUMBER}")
        else()
            set(HF_UI_VERSION "")
            message(STATUS "UI: version not specified (will use HF 'latest')")
        endif()

        # Stamp file embeds the version tag so a changed build number triggers
        # a fresh provision run on the next `cmake --build` without reconfiguring.
        if("${HF_UI_VERSION}" STREQUAL "")
            set(UI_VERSION_TAG "provisioned")
        else()
            set(UI_VERSION_TAG "${HF_UI_VERSION}")
        endif()
        set(UI_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.ui-${UI_VERSION_TAG}.stamp")

        # Join assets with + separator (safe across all platforms, unlike ; and |)
        string(REPLACE ";" "+" PUBLIC_ASSETS_JOINED "${PUBLIC_ASSETS}")

        add_custom_command(
            OUTPUT ${UI_STAMP}
            COMMAND ${CMAKE_COMMAND}
                "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
                "-DPUBLIC_DIR=${PROJECT_SOURCE_DIR}/build/tools/ui"
                "-DHF_BUCKET=${LLAMA_UI_HF_BUCKET}"
                "-DHF_VERSION=${HF_UI_VERSION}"
                "-DHF_ENABLED=${LLAMA_USE_PREBUILT_UI}"
                "-DASSETS=${PUBLIC_ASSETS_JOINED}"
                "-DSTAMP_FILE=${UI_STAMP}"
                "-DNPM_DIR=${PROJECT_SOURCE_DIR}/tools/ui"
                -P ${PROJECT_SOURCE_DIR}/scripts/ui-download.cmake
            COMMENT "Building/provisioning UI assets (npm build -> HF Bucket fallback)"
        )

        set(UI_SOURCE "provisioned")
        set(UI_SOURCE_DIR "${PROJECT_SOURCE_DIR}/build/tools/ui")
    endif()

    # Process assets from the determined source
    if(UI_SOURCE_DIR)
        foreach(asset ${PUBLIC_ASSETS})
            set(input "${UI_SOURCE_DIR}/${asset}")
            set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
            list(APPEND TARGET_SRCS ${output})

            if(UI_SOURCE STREQUAL "local")
                # Local build: files exist at configure time
                if(NOT EXISTS "${input}")
                    message(FATAL_ERROR "UI asset not found: ${input}")
                endif()
                set(dependency "${input}")
            else()
                # HF Bucket: files are downloaded at build time
                set(dependency "${UI_STAMP}")
            endif()

            add_custom_command(
                DEPENDS ${dependency}
                OUTPUT "${output}"
                COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
            )
            set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
        endforeach()

        # Define both old and new preprocessor symbols for backward compat
        add_definitions(-DLLAMA_BUILD_WEBUI)   # Deprecated: use LLAMA_BUILD_UI
        add_definitions(-DLLAMA_BUILD_UI)
        add_definitions(-DLLAMA_WEBUI_DEFAULT_ENABLED=1)   # Deprecated: use LLAMA_UI_DEFAULT_ENABLED
        add_definitions(-DLLAMA_UI_DEFAULT_ENABLED=1)
        message(STATUS "UI: embedded with source: ${UI_SOURCE}")
    else()
        # UI source not found - issue warning but don't fail the build
        # The server will still build but without UI embedded
        message(WARNING "UI: no source available. Neither local build (build/tools/ui/) nor HF Bucket download succeeded.")
        message(WARNING "UI: building server without embedded UI. Set LLAMA_BUILD_UI=OFF to suppress this warning.")
        add_definitions(-DLLAMA_WEBUI_DEFAULT_ENABLED=0)   # Deprecated
        add_definitions(-DLLAMA_UI_DEFAULT_ENABLED=0)
    endif()
else()
    # UI is disabled at build time
    add_definitions(-DLLAMA_WEBUI_DEFAULT_ENABLED=0)   # Deprecated
    add_definitions(-DLLAMA_UI_DEFAULT_ENABLED=0)
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
