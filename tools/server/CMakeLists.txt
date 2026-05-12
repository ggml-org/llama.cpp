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

# Option to specify custom HF bucket for webui (defaults to llama-ui)
# Usage: cmake -B build -DLLAMA_WEBUI_HF_BUCKET=llama-ui
set(LLAMA_WEBUI_HF_BUCKET "llama-ui" CACHE STRING "Hugging Face bucket name for prebuilt webui assets")

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    # Determine source of webui assets (priority: local > HF Bucket)
    set(WEBUI_SOURCE "")
    set(WEBUI_SOURCE_DIR "")

    # Priority 1: Check for local webui build output
    set(LOCAL_WEBUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/public")

    # Verify all required assets exist before declaring local source valid
    set(ALL_ASSETS_PRESENT TRUE)
    foreach(asset ${PUBLIC_ASSETS})
        if(NOT EXISTS "${LOCAL_WEBUI_DIR}/${asset}")
            set(ALL_ASSETS_PRESENT FALSE)
            break()
        endif()
    endforeach()

    if(ALL_ASSETS_PRESENT)
        set(WEBUI_SOURCE "local")
        set(WEBUI_SOURCE_DIR "${LOCAL_WEBUI_DIR}")
        message(STATUS "WebUI: using local build from ${WEBUI_SOURCE_DIR}")
    endif()

    # Priority 2: Try to download from Hugging Face Bucket if local not available
    if(NOT WEBUI_SOURCE_DIR AND LLAMA_USE_PREBUILT_WEBUI)
        # Determine version for HF Bucket using LLAMA_BUILD_NUMBER
        if(DEFINED LLAMA_BUILD_NUMBER)
            set(HF_WEBUI_VERSION "${LLAMA_BUILD_NUMBER}")
            message(STATUS "WebUI: using LLAMA_BUILD_NUMBER=${HF_WEBUI_VERSION}")
        else()
            message(WARNING "WebUI: LLAMA_BUILD_NUMBER not defined, cannot download from HF Bucket")
        endif()

        # Skip HF Bucket if no version available
        if("${HF_WEBUI_VERSION}" STREQUAL "")
            message(WARNING "WebUI: no version available (set LLAMA_BUILD_NUMBER)")
        else()
        # Hugging Face Buckets URL format
        set(HF_BUCKET_URL "https://huggingface.co/buckets/ggml-org/${LLAMA_WEBUI_HF_BUCKET}/resolve/${HF_WEBUI_VERSION}")

        # Download to source tree's public directory for persistence
        # This way subsequent builds use the local files without re-downloading
        set(WEBUI_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/public")

        # Check if we can write to the source directory
        file(WRITE "${WEBUI_DOWNLOAD_DIR}/.write_test" "test")
        if(NOT EXISTS "${WEBUI_DOWNLOAD_DIR}/.write_test")
            message(WARNING "WebUI: Cannot write to source directory (${WEBUI_DOWNLOAD_DIR}). "
                          "Ensure it exists and is writable, or pre-populate with WebUI assets.")
        else()
            file(REMOVE "${WEBUI_DOWNLOAD_DIR}/.write_test")

            file(MAKE_DIRECTORY "${WEBUI_DOWNLOAD_DIR}")

            # Download each asset
            set(ALL_DOWNLOADS_SUCCESS TRUE)
            foreach(asset ${PUBLIC_ASSETS})
                set(download_url "${HF_BUCKET_URL}/${asset}?download=true")
                set(download_path "${WEBUI_DOWNLOAD_DIR}/${asset}")

                message(STATUS "WebUI: downloading ${asset} from ${HF_BUCKET_URL}")

                file(
                    DOWNLOAD "${download_url}" "${download_path}"
                    STATUS download_status
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

            # Verify checksums if available (supply-chain integrity)
            if(ALL_DOWNLOADS_SUCCESS)
                file(
                    DOWNLOAD "${HF_BUCKET_URL}/checksums.txt?download=true"
                    "${WEBUI_DOWNLOAD_DIR}/checksums.txt"
                    STATUS checksum_status
                    TIMEOUT 30
                )
                list(GET checksum_status 0 checksum_result)
                if(checksum_result EQUAL 0)
                    message(STATUS "WebUI: verifying checksums...")
                    file(STRINGS "${WEBUI_DOWNLOAD_DIR}/checksums.txt" CHECKSUMS_CONTENT)
                    foreach(asset ${PUBLIC_ASSETS})
                        set(download_path "${WEBUI_DOWNLOAD_DIR}/${asset}")
                        file(SHA256 "${asset}_hash" "${download_path}")
                        # Format: "hash  filename" (space-separated)
                        string(TOUPPER "${${asset}_hash}" EXPECTED_HASH_UPPER)
                        string(REGEX MATCH "${EXPECTED_HASH_UPPER}[ \\t]+${asset}" CHECKSUM_LINE "${CHECKSUMS_CONTENT}")
                        if(NOT CHECKSUM_LINE)
                            message(WARNING "WebUI: checksum verification failed for ${asset}")
                            set(ALL_DOWNLOADS_SUCCESS FALSE)
                        endif()
                    endforeach()
                    if(ALL_DOWNLOADS_SUCCESS)
                        message(STATUS "WebUI: all checksums verified")
                    endif()
                endif()
            endif()

            if(ALL_DOWNLOADS_SUCCESS)
                set(WEBUI_SOURCE "hf-bucket")
                set(WEBUI_SOURCE_DIR "${WEBUI_DOWNLOAD_DIR}")
                message(STATUS "WebUI: using prebuilt assets from HF Bucket (version: ${HF_WEBUI_VERSION})")
            else()
                # Fallback: try the 'latest' directory on HF Bucket
                set(HF_LATEST_URL "https://huggingface.co/buckets/ggml-org/${LLAMA_WEBUI_HF_BUCKET}/resolve/latest")
                message(STATUS "WebUI: exact version not found, trying latest release from ${HF_LATEST_URL}")

                set(ALL_DOWNLOADS_SUCCESS TRUE)
                foreach(asset ${PUBLIC_ASSETS})
                    set(download_url "${HF_LATEST_URL}/${asset}?download=true")
                    set(download_path "${WEBUI_DOWNLOAD_DIR}/${asset}")

                    message(STATUS "WebUI: downloading ${asset} from latest")

                    file(
                        DOWNLOAD "${download_url}" "${download_path}"
                        STATUS download_status
                        TIMEOUT 60
                    )

                    list(GET download_status 0 download_result)

                    if(download_result EQUAL 0)
                        message(STATUS "WebUI: downloaded ${asset} from latest")
                    else()
                        list(GET download_status 1 error_message)
                        message(WARNING "WebUI: failed to download ${asset} from latest: ${error_message}")
                        set(ALL_DOWNLOADS_SUCCESS FALSE)
                        break()
                    endif()
                endforeach()

                # Verify checksums if available
                if(ALL_DOWNLOADS_SUCCESS)
                    file(
                        DOWNLOAD "${HF_LATEST_URL}/checksums.txt?download=true"
                        "${WEBUI_DOWNLOAD_DIR}/checksums.txt"
                        STATUS checksum_status
                        TIMEOUT 30
                    )
                    list(GET checksum_status 0 checksum_result)
                    if(checksum_result EQUAL 0)
                        message(STATUS "WebUI: verifying checksums for latest release...")
                        file(STRINGS "${WEBUI_DOWNLOAD_DIR}/checksums.txt" CHECKSUMS_CONTENT)
                        foreach(asset ${PUBLIC_ASSETS})
                            set(download_path "${WEBUI_DOWNLOAD_DIR}/${asset}")
                            file(SHA256 "${asset}_hash" "${download_path}")
                            string(TOUPPER "${${asset}_hash}" EXPECTED_HASH_UPPER)
                            string(REGEX MATCH "${EXPECTED_HASH_UPPER}[ \\t]+${asset}" CHECKSUM_LINE "${CHECKSUMS_CONTENT}")
                            if(NOT CHECKSUM_LINE)
                                message(WARNING "WebUI: checksum verification failed for ${asset}")
                                set(ALL_DOWNLOADS_SUCCESS FALSE)
                            endif()
                        endforeach()
                        if(ALL_DOWNLOADS_SUCCESS)
                            message(STATUS "WebUI: all checksums verified for latest release")
                        endif()
                    endif()
                endif()

                if(ALL_DOWNLOADS_SUCCESS)
                    set(WEBUI_SOURCE "hf-bucket-latest")
                    set(WEBUI_SOURCE_DIR "${WEBUI_DOWNLOAD_DIR}")
                    message(STATUS "WebUI: using prebuilt assets from HF Bucket latest (version: ${HF_WEBUI_VERSION})")
                else()
                    message(WARNING "WebUI: HF Bucket download failed (both version and latest), no local build available")
                endif()
            endif()
            endif()  # Close: if write test passed
        endif()  # Close: if(HF_WEBUI_VERSION not empty)
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
