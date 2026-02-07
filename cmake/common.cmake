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

    if (NOT MSVC)
        if (LLAMA_SANITIZE_THREAD)
            message(STATUS "Using -fsanitize=thread")

            add_compile_options(-fsanitize=thread)
            link_libraries     (-fsanitize=thread)
        endif()

        if (LLAMA_SANITIZE_ADDRESS)
            message(STATUS "Using -fsanitize=address")

            add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
            link_libraries     (-fsanitize=address)
        endif()

        if (LLAMA_SANITIZE_UNDEFINED)
            message(STATUS "Using -fsanitize=undefined")

            add_compile_options(-fsanitize=undefined)
            link_libraries     (-fsanitize=undefined)
        endif()
    endif()
endfunction()
