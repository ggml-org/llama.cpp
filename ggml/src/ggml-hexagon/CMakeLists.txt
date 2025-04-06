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
    set(DEBUG_FLAG "-Wall")
    message("Debug mode:${DEBUG_FLAG}")
else()
    set(DEBUG_FLAG "-DNDEBUG -Wall")
    message("Release mode:${DEBUG_FLAG}")
endif()


#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka Gen4)
if(NOT DEFINED HTP_ARCH_VERSION)
    #set default HTP_ARCH_VERSION to v75
    set(HTP_ARCH_VERSION v75)
endif()

#cross compiling for hexagon kernels on cDSP side
set(HEXAGON_CC              "${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang")
set(HEXAGON_CXX             "${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang")
set(HEXAGON_TARGET          libggmlop_skel${HTP_ARCH_VERSION}.so)
set(HEXAGON_KERNELS_PATH    "${CMAKE_CURRENT_LIST_DIR}/kernels")
set(HEXAGON_COMPUTE         "compute${HTP_ARCH_VERSION}")

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
    message(FATAL_ERROR "QNN now only available on Android and Windows(Windows on ARM)")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_USE_HEXAGON ${DEBUG_FLAG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

file(GLOB QNN_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/kernels/ggmlop_ap_skel.c")
ggml_add_backend_library(ggml-hexagon ${QNN_SOURCES})

target_include_directories(ggml-hexagon PRIVATE ${QNN_SDK_PATH}/include/QNN ${HEXAGON_SDK_PATH} ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(ggml-hexagon PRIVATE ${QNN_LINK_LIBRARIES})

string(REGEX REPLACE "/$" "" QNN_DEFAULT_LIB_SEARCH_PATH "${QNN_DEFAULT_LIB_SEARCH_PATH}")
target_compile_definitions(ggml-hexagon PRIVATE QNN_DEFAULT_LIB_SEARCH_PATH="${QNN_DEFAULT_LIB_SEARCH_PATH}/")

function(ggml_hexagon_build_kernel KNAME)
    message(STATUS "ggml_hexagon: build kernel ${KNAME}")

    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND echo "current working path:`pwd`\n"
        COMMAND ${HEXAGON_CC} -o ${HEXAGON_KERNELS_PATH}/ggml-dsp.o -c ${HEXAGON_KERNELS_PATH}/ggml-dsp.c -m${HTP_ARCH_VERSION} -c -Ofast -Wall -Wstrict-prototypes -fno-zero-initialized-in-bss -fdata-sections -fpic ${DEBUG_FLAG} -D__V_DYNAMIC__ -mhvx -mhvx-length=128B -fno-finite-math-only -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/libs/qprintf/inc -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc -I${HEXAGON_SDK_PATH}/utils/examples -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/inc -I${HEXAGON_SDK_PATH}/libs/atomic/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/posix -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/qurt/
        COMMAND ${HEXAGON_CC} -o ${HEXAGON_KERNELS_PATH}/ggmlop_cdsp_skel.o -c ${HEXAGON_KERNELS_PATH}/ggmlop_cdsp_skel.c -m${HTP_ARCH_VERSION} -c -Ofast -Wall -Wstrict-prototypes -fno-zero-initialized-in-bss -fdata-sections -fpic -D__V_DYNAMIC__ -mhvx -mhvx-length=128B -fno-finite-math-only -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/libs/qprintf/inc -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc -I${HEXAGON_SDK_PATH}/utils/examples -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/inc -I${HEXAGON_SDK_PATH}/libs/atomic/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc
        COMMAND ${HEXAGON_CC} -m${HTP_ARCH_VERSION} -Wl,--defsym=ISDB_TRUSTED_FLAG=2 -Wl,--defsym=ISDB_SECURE_FLAG=2 -Wl,--no-threads -fpic -shared -Wl,-Bsymbolic -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -lc -Wl,-soname=${HEXAGON_TARGET} -o ${HEXAGON_KERNELS_PATH}/${HEXAGON_TARGET} -Wl,--start-group ${HEXAGON_KERNELS_PATH}/ggmlop_cdsp_skel.o ${HEXAGON_KERNELS_PATH}/ggml-dsp.o -Wl,--end-group
        COMMAND ls -l ${HEXAGON_KERNELS_PATH}/${HEXAGON_TARGET}
        COMMENT "build hexagon-kernel"
    )
endfunction()

ggml_hexagon_build_kernel("cdsp")
