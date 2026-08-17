cmake_minimum_required(VERSION 3.18)

if (GGML_BACKEND_DL)
    # this would break llama_collect_shared_lib_files
    message(FATAL_ERROR "Desktop: GGML_BACKEND_DL=ON is not supported yet when building llama-desktop")
endif()

# returns the paths to all of our shared libs that the target depends on
function(llama_collect_shared_lib_files tgt out_var)
    set(_seen)
    set(_files)
    set(_stack ${tgt})
    while(_stack)
        list(POP_FRONT _stack _cur)
        if(NOT TARGET "${_cur}" OR _cur IN_LIST _seen)
            continue()
        endif()
        list(APPEND _seen ${_cur})

        get_target_property(_aliased ${_cur} ALIASED_TARGET)
        if(_aliased)
            set(_cur ${_aliased})
        endif()

        get_target_property(_type ${_cur} TYPE)
        if(_type STREQUAL "SHARED_LIBRARY")
            list(APPEND _files "$<TARGET_FILE:${_cur}>")
            if(UNIX)
                list(APPEND _files "$<TARGET_SONAME_FILE:${_cur}>")
                list(APPEND _files "$<TARGET_LINKER_FILE:${_cur}>")
            endif()
        endif()

        foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_libs ${_cur} ${_prop})
            if(_libs)
                list(APPEND _stack ${_libs})
            endif()
        endforeach()
    endwhile()
    set(${out_var} "${_files}" PARENT_SCOPE)
endfunction()

set(DESKTOP_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(CMAKE_HOST_WIN32)
    find_program(NPM_EXECUTABLE NAMES npm.cmd npm.bat npm)
else()
    find_program(NPM_EXECUTABLE npm)
endif()
if(NOT NPM_EXECUTABLE)
    message(FATAL_ERROR "Desktop: npm not found")
endif()

llama_collect_shared_lib_files(llama-server LLAMA_SERVER_SHARED_LIBS)

add_custom_target(llama-desktop ALL
    COMMAND ${NPM_EXECUTABLE} ci
    COMMAND ${CMAKE_COMMAND} -E env
            "LLAMA_SERVER_PATH=$<TARGET_FILE:llama-server>"
            "LLAMA_SHARED_LIBS=$<JOIN:${LLAMA_SERVER_SHARED_LIBS},$<SEMICOLON>>"
            "OUT_DIR=${DESKTOP_OUT_DIR}"
            ${NPM_EXECUTABLE} run make
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building llama-desktop"
    VERBATIM
)

add_dependencies(llama-desktop llama-server)
