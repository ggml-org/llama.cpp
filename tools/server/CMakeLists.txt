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

# Option to specify custom HF bucket for webui (defaults to llama-webui)
option(LLAMA_WEBUI_HF_BUCKET "Hugging Face bucket name for prebuilt webui assets" "llama-webui")

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
    )

    # Determine source of webui assets (priority: local > HF Bucket)
    set(WEBUI_SOURCE "")
    set(WEBUI_SOURCE_DIR "")

    # Priority 1: Check for local webui build output
    set(LOCAL_WEBUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/public")
    if(EXISTS "${LOCAL_WEBUI_DIR}/index.html")
        set(WEBUI_SOURCE "local")
        set(WEBUI_SOURCE_DIR "${LOCAL_WEBUI_DIR}")
        message(STATUS "WebUI: using local build from ${WEBUI_SOURCE_DIR}")
    endif()

    # Priority 2: Try to download from Hugging Face Bucket if local not available
    if(NOT WEBUI_SOURCE_DIR AND LLAMA_USE_PREBUILD_WEBUI)
        # Determine version tag for HF Bucket
        # Use LLAMA_BUILD_TAG env var if set, otherwise try to get from git, otherwise use "latest"
        if(DEFINED ENV{LLAMA_BUILD_TAG})
            set(HF_WEBUI_VERSION "$ENV{LLAMA_BUILD_TAG}")
        elseif(EXISTS "${CMAKE_SOURCE_DIR}/.git")
            execute_process(
                COMMAND git describe --tags --abbrev=0 2>/dev/null || echo "latest"
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE HF_WEBUI_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            set(HF_WEBUI_VERSION "latest")
        endif()

        # Hugging Face Buckets URL format
        set(HF_BUCKET_URL "https://huggingface.co/buckets/ggml-org/${LLAMA_WEBUI_HF_BUCKET}/resolve/${HF_WEBUI_VERSION}")

        # Create download directory
        set(WEBUI_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/webui-download")
        file(MAKE_DIRECTORY "${WEBUI_DOWNLOAD_DIR}")

        # Download each asset
        set(ALL_DOWNLOADS_SUCCESS TRUE)
        foreach(asset ${PUBLIC_ASSETS})
            set(download_url "${HF_BUCKET_URL}/${asset}")
            set(download_path "${WEBUI_DOWNLOAD_DIR}/${asset}")

            message(STATUS "WebUI: downloading ${asset} from ${HF_BUCKET_URL}")

            file(
                DOWNLOAD "${download_url}" "${download_path}"
                STATUS download_status
                EXPECTED_HASH SHA256=IGNORE
                TIMEOUT 60
            )

            list(GET download_status 0 download_result)

            if(download_result EQUAL 0)
                message(STATUS "WebUI: downloaded ${asset}")
            else()
                list(GET download_status 1 error_message)
                message(WARNING "WebUI: failed to download ${asset}: ${error_message}")
                set(ALL_DOWNLOADS_SUCCESS FALSE)
                break()
            endif()
        endforeach()

        if(ALL_DOWNLOADS_SUCCESS)
            set(WEBUI_SOURCE "hf-bucket")
            set(WEBUI_SOURCE_DIR "${WEBUI_DOWNLOAD_DIR}")
            message(STATUS "WebUI: using prebuilt assets from HF Bucket (version: ${HF_WEBUI_VERSION})")
        else()
            message(WARNING "WebUI: HF Bucket download failed, no local build available")
        endif()
    endif()

    # Process assets from the determined source
    if(WEBUI_SOURCE_DIR)
        foreach(asset ${PUBLIC_ASSETS})
            set(input "${WEBUI_SOURCE_DIR}/${asset}")
            if(NOT EXISTS "${input}")
                message(FATAL_ERROR "WebUI asset not found: ${input}")
            endif()

            set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
            list(APPEND TARGET_SRCS ${output})
            add_custom_command(
                DEPENDS "${input}"
                OUTPUT "${output}"
                COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
            )
            set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
        endforeach()

        add_definitions(-DLLAMA_BUILD_WEBUI)
        add_definitions(-DLLAMA_WEBUI_DEFAULT_ENABLED=1)
        message(STATUS "WebUI: embedded with source: ${WEBUI_SOURCE}")
    else()
        # WebUI source not found - issue warning but don't fail the build
        # The server will still build but without webui embedded
        message(WARNING "WebUI: no source available. Neither local build (tools/server/public/) nor HF Bucket download succeeded.")
        message(WARNING "WebUI: building server without embedded WebUI. Set LLAMA_BUILD_WEBUI=OFF to suppress this warning.")
        add_definitions(-DLLAMA_WEBUI_DEFAULT_ENABLED=0)
    endif()
else()
    # WebUI is disabled at build time
    add_definitions(-DLLAMA_WEBUI_DEFAULT_ENABLED=0)
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)