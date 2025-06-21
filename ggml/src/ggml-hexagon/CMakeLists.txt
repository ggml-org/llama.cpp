project(ggml-hexagon)
message(STATUS "Using HEXAGON backend")
message("CMAKE_SYSTEM_NAME : ${CMAKE_SYSTEM_NAME}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED QNN_SDK_PATH)
    message(FATAL_ERROR "QNN_SDK_PATH not defined")
endif()

if(NOT DEFINED HEXAGON_SDK_PATH)
    message(FATAL_ERROR "HEXAGON_SDK_PATH not defined")
endif()

message("QNN_SDK_PATH    : ${QNN_SDK_PATH}")
message("HEXAGON_SDK_PATH: ${HEXAGON_SDK_PATH}")
message("HTP_ARCH_VERSION: ${HTP_ARCH_VERSION}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_FLAG "-DDEBUG -Wall")
    message("Debug mode:${DEBUG_FLAG}")
else()
    set(DEBUG_FLAG "-DNDEBUG -Wall")
#manually disable all verbose logs in ggml-hexagon/CMakeLists.txt to
#make compare NPU performance through llama-bench more clear
#set(DEBUG_FLAG "-DNDEBUG -Wall -DDISABLE_ALL_LOG")
    message("Release mode:${DEBUG_FLAG}")
endif()

#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka Gen4)
if(NOT DEFINED HTP_ARCH_VERSION)
    message(FATAL_ERROR "HTP_ARCH_VERSION not defined, valid htp arch: v68,v69,v73,v75,v79")
endif()

#check whether user's specified htp arch is valid
set(CHECK_HTP_ARCH "WRONG")
foreach (feat v68 v69 v73 v75 v79)
    if (${feat} STREQUAL ${HTP_ARCH_VERSION})
        set(CHECK_HTP_ARCH "GOOD")
    endif()
endforeach()
if (${CHECK_HTP_ARCH} STREQUAL "WRONG")
    message(FATAL_ERROR "ggml-hexagon backend only support htp arch v68,v69,v73,v75,v79")
endif()

#check optimization flags
set(OPT_FLAG " ")
if (${HTP_ARCH_VERSION} STREQUAL "v75" OR ${HTP_ARCH_VERSION} STREQUAL "v79")
    #works fine on Snapdragon 8Gen3&8Elite with 1.5x - 3x performance gains with the default ggml backend
    set(OPT_FLAG " -O3 -march=armv8.7-a -mcpu=cortex-x1 -mtune=cortex-x1 -flto -D_GNU_SOURCE -fvectorize -ffp-model=fast -fno-finite-math-only")
endif()
message("OPT_FLAG:${OPT_FLAG}")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(LOG_LIB log)

    add_library(cdsprpc
        SHARED
        IMPORTED)
    set_target_properties(cdsprpc
        PROPERTIES
        IMPORTED_LOCATION
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so)

    set(QNN_LINK_LIBRARIES ${LOG_LIB} cdsprpc)
    set(QNN_DEFAULT_LIB_SEARCH_PATH "/data/local/tmp/" CACHE STRING "customized library search path for QNN backend")

    include_directories(${HEXAGON_SDK_PATH}/incs)
    include_directories(${HEXAGON_SDK_PATH}/incs/stddef)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/incs)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/remote/ship/android_Debug_aarch64)
    include_directories(${HEXAGON_SDK_PATH}/utils/examples)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/android_aarch64)
    include_directories(${HEXAGON_SDK_PATH}/libs/atomic/inc)
    include_directories(${HEXAGON_SDK_PATH}/libs/atomic/android_Debug_aarch64/ship)
    include_directories(${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/)
    include_directories(${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/kernels/)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "C:\\" CACHE STRING "customized library search path for QNN backend")
else()
    message(FATAL_ERROR "ggml-hexagon now only available on Android and Windows(Windows on ARM)")
endif()

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")

file(GLOB HEXAGON_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/kernels/stub.c")
ggml_add_backend_library(ggml-hexagon ${HEXAGON_SOURCES})

target_include_directories(ggml-hexagon PRIVATE ${QNN_SDK_PATH}/include/QNN ${HEXAGON_SDK_PATH} ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(ggml-hexagon PRIVATE ${QNN_LINK_LIBRARIES})

string(REGEX REPLACE "/$" "" QNN_DEFAULT_LIB_SEARCH_PATH "${QNN_DEFAULT_LIB_SEARCH_PATH}")
target_compile_definitions(ggml-hexagon PRIVATE QNN_DEFAULT_LIB_SEARCH_PATH="${QNN_DEFAULT_LIB_SEARCH_PATH}/")

#cross compiling source codes of hexagon kernels which running on cDSP side
function(ggml_hexagon_build_kernel KNAME)
    message(STATUS "ggml_hexagon: build hexagon-kernel ${KNAME}")

    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND echo "current working path:`pwd`\n"
        COMMAND echo "${CMAKE_CURRENT_LIST_DIR}/kernels"
        COMMAND make -C ${CMAKE_CURRENT_LIST_DIR}/kernels/ clean
        COMMAND make -C ${CMAKE_CURRENT_LIST_DIR}/kernels/ HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} HTP_ARCH_VERSION=${HTP_ARCH_VERSION} DEBUG_FLAG=${DEBUG_FLAG}
        COMMAND echo "current working path:`pwd`\n"
        COMMAND ls -l  ../../../bin/libggmldsp-skel.so
        COMMENT "build hexagon-kernel"
    )
endfunction()

function(ggml_hexagon_setup_cfg KNAME)
    message(STATUS "ggml_hexagon: setup runtime configuration file ${KNAME}")
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND echo "current working path:`pwd`\n"
        COMMAND /bin/cp -fv ../../../../../scripts/${KNAME}  ../../../bin/
        COMMENT "setup runtime configuration file"
    )
endfunction()

ggml_hexagon_build_kernel("cdsp")
ggml_hexagon_setup_cfg("ggml-hexagon.cfg")
