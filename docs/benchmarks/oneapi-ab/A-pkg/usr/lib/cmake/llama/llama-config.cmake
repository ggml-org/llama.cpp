set(LLAMA_VERSION      0.5.0-dev)
set(LLAMA_BUILD_COMMIT bb6908513)
set(LLAMA_BUILD_NUMBER 12275)
set(LLAMA_SHARED_LIB   ON)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was llama-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# Use original install prefix when loaded through a "/usr move"
# cross-prefix symbolic link such as /lib -> /usr/lib.
get_filename_component(_realCurr "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_realOrig "/usr/lib/cmake/llama" REALPATH)
if(_realCurr STREQUAL _realOrig)
  set(PACKAGE_PREFIX_DIR "/usr")
endif()
unset(_realOrig)
unset(_realCurr)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check(LLAMA_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set_and_check(LLAMA_LIB_DIR     "${PACKAGE_PREFIX_DIR}/lib")
set(LLAMA_BIN_DIR "${PACKAGE_PREFIX_DIR}/bin")

find_package(ggml REQUIRED HINTS ${LLAMA_LIB_DIR}/cmake)

find_library(llama_LIBRARY llama
    REQUIRED
    HINTS ${LLAMA_LIB_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)

# Guarded so a second find_package(llama) (e.g. from another subdirectory)
# re-evaluating this file does not re-add the imported target and error out.
if (NOT TARGET llama)
    add_library(llama UNKNOWN IMPORTED GLOBAL)
    set_target_properties(llama
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "ggml::ggml;ggml::ggml-base"
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${llama_LIBRARY}"
            INTERFACE_COMPILE_FEATURES c_std_90
            POSITION_INDEPENDENT_CODE ON)

    add_library(llama::llama ALIAS llama)
endif()

# optional helper library, present when the SDK was built with LLAMA_BUILD_COMMON
find_library(llama_common_LIBRARY llama-common
    HINTS ${LLAMA_LIB_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)
if (llama_common_LIBRARY AND IS_DIRECTORY "${LLAMA_INCLUDE_DIR}/llama-common" AND NOT TARGET llama-common)
    find_package(Threads REQUIRED)
    add_library(llama-common UNKNOWN IMPORTED GLOBAL)
    set_target_properties(llama-common
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_INCLUDE_DIR}/llama-common;${LLAMA_INCLUDE_DIR}"
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${llama_common_LIBRARY}"
            INTERFACE_LINK_LIBRARIES "llama;Threads::Threads")
    add_library(llama::common ALIAS llama-common)
endif()

check_required_components(Llama)
