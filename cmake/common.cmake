include("ggml/cmake/common.cmake")

function(llama_add_compile_flags)
    if (LLAMA_FATAL_WARNINGS)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND C_FLAGS   -Werror)
            list(APPEND CXX_FLAGS -Werror)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_compile_options(/WX)
        endif()
    endif()

    if (LLAMA_ALL_WARNINGS)
        if (NOT MSVC)
            list(APPEND C_FLAGS -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                -Werror=implicit-int -Werror=implicit-function-declaration)

            list(APPEND CXX_FLAGS -Wmissing-declarations -Wmissing-noreturn)

            list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)

            list(APPEND C_FLAGS   ${WARNING_FLAGS})
            list(APPEND CXX_FLAGS ${WARNING_FLAGS})

            ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

            add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                                "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
        else()
            # todo : msvc
            set(C_FLAGS   "" PARENT_SCOPE)
            set(CXX_FLAGS "" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(llama_download_model NAME HASH)
    set(DEST "${CMAKE_BINARY_DIR}/${NAME}")
    get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
    file(MAKE_DIRECTORY "${DEST_DIR}")
    if(NOT EXISTS "${DEST}")
        message(STATUS "Downloading ${NAME} from ggml-org/models...")
    endif()
    file(DOWNLOAD
        "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
        "${DEST}"
        TLS_VERIFY ON
        EXPECTED_HASH ${HASH}
        STATUS status
    )
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        list(GET status 1 msg)
        message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
    endif()
    set(LLAMA_DOWNLOAD_MODEL "${DEST}" PARENT_SCOPE)
endfunction()
