set(TARGET llama-ui)

set(LLAMA_UI_HF_BUCKET "llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt UI assets")

# Backward compat: forward old var to new one
if(DEFINED LLAMA_BUILD_WEBUI)
    set(LLAMA_BUILD_UI ${LLAMA_BUILD_WEBUI})
    message(DEPRECATION "LLAMA_BUILD_WEBUI is deprecated, use LLAMA_BUILD_UI instead")
endif()
if(DEFINED LLAMA_USE_PREBUILT_WEBUI)
    set(LLAMA_USE_PREBUILT_UI ${LLAMA_USE_PREBUILT_WEBUI})
    message(DEPRECATION "LLAMA_USE_PREBUILT_WEBUI is deprecated, use LLAMA_USE_PREBUILT_UI instead")
endif()
if(DEFINED LLAMA_WEBUI_HF_BUCKET)
    set(LLAMA_UI_HF_BUCKET ${LLAMA_WEBUI_HF_BUCKET})
    message(DEPRECATION "LLAMA_WEBUI_HF_BUCKET is deprecated, use LLAMA_UI_HF_BUCKET instead")
endif()

# Resolve HF asset version: explicit env var > derived from build number > unset
if(DEFINED ENV{HF_WEBUI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_WEBUI_VERSION}")
    message(DEPRECATION "HF_WEBUI_VERSION env var is deprecated, use HF_UI_VERSION instead")
elseif(DEFINED ENV{HF_UI_VERSION})
    set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")
elseif(DEFINED LLAMA_BUILD_NUMBER)
    set(HF_UI_VERSION "b${LLAMA_BUILD_NUMBER}")
else()
    set(HF_UI_VERSION "")
endif()

if(NOT "${HF_UI_VERSION}" STREQUAL "" AND NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "UI: invalid HF_UI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
endif()

set(UI_CPP "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(UI_H   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

if(CMAKE_CROSSCOMPILING)
    find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    if(NOT HOST_CXX_COMPILER)
        message(FATAL_ERROR "UI: no host C++ compiler (g++/clang++) found to build llama-ui-embed; set -DHOST_CXX_COMPILER=<path>")
    endif()
    message(STATUS "UI: building llama-ui-embed with host compiler ${HOST_CXX_COMPILER}")

    if(CMAKE_HOST_WIN32)
        set(LLAMA_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/llama-ui-embed-host.exe")
    else()
        set(LLAMA_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/llama-ui-embed-host")
    endif()

    add_custom_command(
        OUTPUT  "${LLAMA_UI_EMBED_EXE}"
        COMMAND "${HOST_CXX_COMPILER}" -O2 -std=c++17
                -o "${LLAMA_UI_EMBED_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
        COMMENT "Building llama-ui-embed (host)"
        VERBATIM
    )

    # phony target to tie it into the dependency graph
    add_custom_target(llama-ui-embed DEPENDS "${LLAMA_UI_EMBED_EXE}")
else()
    add_executable(llama-ui-embed embed.cpp)
    target_compile_features(llama-ui-embed PRIVATE cxx_std_17)
    set_target_properties(llama-ui-embed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set(LLAMA_UI_EMBED_EXE "$<TARGET_FILE:llama-ui-embed>")
if(LLAMA_BUILD_UI)
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
    set(LOCAL_UI_DIR "${PROJECT_SOURCE_DIR}/build/tools/ui/dist")

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
            if(NOT HF_UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
                message(FATAL_ERROR "UI: invalid HF_WEBUI_VERSION='${HF_UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
            endif()
        elseif(DEFINED ENV{HF_UI_VERSION})
            set(HF_UI_VERSION "$ENV{HF_UI_VERSION}")
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

        if("${HF_UI_VERSION}" STREQUAL "")
            set(UI_VERSION_TAG "provisioned")
        else()
            set(UI_VERSION_TAG "${HF_UI_VERSION}")
        endif()
        set(UI_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.ui-${UI_VERSION_TAG}.stamp")

        string(REPLACE ";" "+" PUBLIC_ASSETS_JOINED "${PUBLIC_ASSETS}")

        add_custom_command(
            OUTPUT ${UI_STAMP}
            COMMAND ${CMAKE_COMMAND}
                "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
                "-DPUBLIC_DIR=${PROJECT_SOURCE_DIR}/build/tools/ui/dist"
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
        set(UI_SOURCE_DIR "${PROJECT_SOURCE_DIR}/build/tools/ui/dist")
    endif()

    # Process assets from the determined source
    if(UI_SOURCE_DIR)
        foreach(asset ${PUBLIC_ASSETS})
            set(input "${UI_SOURCE_DIR}/${asset}")
            set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
            list(APPEND TARGET_SRCS ${output})

            if(UI_SOURCE STREQUAL "local")
                if(NOT EXISTS "${input}")
                    message(FATAL_ERROR "UI asset not found: ${input}")
                endif()
                set(dependency "${input}")
            else()
                set(dependency "${UI_STAMP}")
            endif()

            add_custom_command(
                DEPENDS ${dependency}
                OUTPUT "${output}"
                COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
            )
            set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
        endforeach()

        list(APPEND UI_COMPILE_DEFS
            LLAMA_BUILD_UI
            LLAMA_UI_DEFAULT_ENABLED=1
        )
        message(STATUS "UI: embedded with source: ${UI_SOURCE}")
    else()
        message(WARNING "UI: no source available. Neither local build (build/tools/ui/dist/) nor HF Bucket download succeeded.")
        message(WARNING "UI: building server without embedded UI. Set LLAMA_BUILD_UI=OFF to suppress this warning.")
        list(APPEND UI_COMPILE_DEFS LLAMA_UI_DEFAULT_ENABLED=0)
    endif()
else()
    list(APPEND UI_COMPILE_DEFS LLAMA_UI_DEFAULT_ENABLED=0)
endif()

# Run the provisioning script every build so source changes in tools/ui/ are
# always picked up. The script uses copy_if_different for ui.cpp/ui.h, so the
# library only recompiles when contents actually change.
add_custom_target(llama-ui-assets ALL
    BYPRODUCTS ${UI_CPP} ${UI_H}
    COMMAND ${CMAKE_COMMAND}
        "-DUI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DUI_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DLLAMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DHF_BUCKET=${LLAMA_UI_HF_BUCKET}"
        "-DHF_VERSION=${HF_UI_VERSION}"
        "-DHF_ENABLED=${LLAMA_USE_PREBUILT_UI}"
        "-DBUILD_UI=${LLAMA_BUILD_UI}"
        "-DLLAMA_UI_EMBED=${LLAMA_UI_EMBED_EXE}"
        -P "${PROJECT_SOURCE_DIR}/scripts/ui-assets.cmake"
    COMMENT "Provisioning UI assets"
    VERBATIM
)

add_dependencies(llama-ui-assets llama-ui-embed)

set_source_files_properties(${UI_CPP} ${UI_H} PROPERTIES GENERATED TRUE)

add_library(${TARGET} STATIC ${UI_CPP} ${UI_H})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_dependencies(${TARGET} llama-ui-assets)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
