if (HEXAGON_TOOLCHAIN_INCLUDED)
  return()
endif()
set(HEXAGON_TOOLCHAIN_INCLUDED true)

#Cross Compiling for Hexagon
set(HEXAGON TRUE)
set(CMAKE_SYSTEM_NAME QURT)
set(CMAKE_SYSTEM_PROCESSOR Hexagon)
set(CMAKE_SYSTEM_VERSION "1") #${HEXAGON_PLATFORM_LEVEL})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CUSTOM_RUNELF_PATH "")

#To fix backward compatibility with EAI addon.
if (NOT HEXAGON_SDK_ROOT)
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
endif()

if (NOT HEXAGON_TOOLS_ROOT)
    if (DEFINED ENV{HEXAGON_TOOLS_ROOT})
        set(HEXAGON_TOOLS_ROOT $ENV{HEXAGON_TOOLS_ROOT})
    endif()
    if(NOT HEXAGON_TOOLS_ROOT)
        set(HEXAGON_TOOLS_ROOT $ENV{DEFAULT_HEXAGON_TOOLS_ROOT})
    endif()
endif()

file(TO_CMAKE_PATH "${HEXAGON_TOOLS_ROOT}" HEXAGON_TOOLS_ROOT)
file(TO_CMAKE_PATH "${HEXAGON_SDK_ROOT}"   HEXAGON_SDK_ROOT)

#Get the Binary extension of the Hexagon Toolchain
if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
    set(HEXAGON_TOOLCHAIN_SUFFIX .exe)
endif()
message(DEBUG "CMAKE_HOST_SYSTEM_NAME:${CMAKE_HOST_SYSTEM_NAME}")

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_arch.cmake)

set(HEXAGON_TOOLCHAIN ${HEXAGON_TOOLS_ROOT})
set(HEXAGON_LIB_DIR "${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib")
set(HEXAGON_ISS_DIR ${HEXAGON_TOOLCHAIN}/Tools/lib/iss)

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    HEXAGON_SDK_ROOT
    HEXAGON_TOOLS_ROOT
)

#QURT Related includes and linker flags
set(V_ARCH ${HEXAGON_ARCH})
set(_QURT_INSTALL_DIR "${HEXAGON_SDK_ROOT}/rtos/qurt/ADSP${V_ARCH}MP${V_ARCH_EXTN}")
set(_QURT_INSTALL_DIR "${HEXAGON_SDK_ROOT}/rtos/qurt/compute${V_ARCH}${V_ARCH_EXTN}")

if( ${TREE} MATCHES PAKMAN )
    set(_QURT_INSTALL_DIR "${QURT_IMAGE_DIR}/compute${V_ARCH}${V_ARCH_EXTN}")
endif()
message(DEBUG "_QURT_INSTALL_DIR:${_QURT_INSTALL_DIR}")
set(RTOS_DIR ${_QURT_INSTALL_DIR})
set(QCC_DIR "${HEXAGON_QCC_DIR}/${V_ARCH}/G0")
set(TARGET_DIR "${HEXAGON_LIB_DIR}/${V_ARCH}/G0")

include_directories(
    ${_QURT_INSTALL_DIR}/include
    ${_QURT_INSTALL_DIR}/include/qurt
    ${_QURT_INSTALL_DIR}/include/posix
    )

set(QURT_START_LINK_LIBS)
set(QURT_START_LINK_LIBS
    "${TARGET_DIR}/init.o"
    "${RTOS_DIR}/lib/crt1.o"
    "${RTOS_DIR}/lib/debugmon.o"
    "${RTOS_DIR}/lib/libqurt.a"
    "${TARGET_DIR}/libc.a"
    "${TARGET_DIR}/libqcc.a"
    "${TARGET_DIR}/libhexagon.a"
    "${RTOS_DIR}/lib/libqurtcfs.a"
    "${RTOS_DIR}/lib/libtimer_island.a"
    "${RTOS_DIR}/lib/libtimer_main.a"
    "${RTOS_DIR}/lib/libposix.a"
    )
STRING(REPLACE ";" " " QURT_START_LINK_LIBS "${QURT_START_LINK_LIBS}")

set(QURT_END_LINK_LIBS
    ${TARGET_DIR}/fini.o
    )

#Non QURT related includes and linker flags

set(TARGET_DIR_NOOS "${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib/${HEXAGON_ARCH}")

if (NOT NO_WRAP_MEM_API)
    set(WRAP_MALLOC   -Wl,--wrap=malloc)
    set(WRAP_CALLOC   -Wl,--wrap=calloc)
    set(WRAP_FREE     -Wl,--wrap=free)
    set(WRAP_REALLOC  -Wl,--wrap=realloc)
    set(WRAP_MEMALIGN -Wl,--wrap=memalign)
endif()

set(PIC_SHARED_LD_FLAGS
    -mcpu=${V_ARCH} -m${V_ARCH} -mhvx=${V_ARCH}
    -G0
    -fpic
    -Wl,-Bsymbolic
    -Wl,-L${TARGET_DIR_NOOS}/G0/pic
    -Wl,-L${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib/
    -Wl,--no-threads ${WRAP_MALLOC} ${WRAP_CALLOC} ${WRAP_FREE} ${WRAP_REALLOC} ${WRAP_MEMALIGN}
    -shared
    "-o <TARGET> <SONAME_FLAG><TARGET_SONAME>"
    "<LINK_FLAGS>"
    -Wl,--start-group
    "<OBJECTS>"
    "<LINK_LIBRARIES>"
    -Wl,--end-group
    -lc
    )
STRING(REPLACE ";" " " PIC_SHARED_LD_FLAGS "${PIC_SHARED_LD_FLAGS}")

set(HEXAGON_PIC_SHARED_LINK_OPTIONS "${PIC_SHARED_LD_FLAGS}")

#System include paths
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/incs)
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/incs/stddef)
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/ipc/fastrpc/incs)

#LLVM toolchain setup
#Compiler paths, options and architecture
set(CMAKE_C_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_CXX_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_AR ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-ar${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_ASM_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++${HEXAGON_TOOLCHAIN_SUFFIX})
set(HEXAGON_LINKER ${CMAKE_C_COMPILER})
set(CMAKE_PREFIX_PATH ${HEXAGON_TOOLCHAIN}/Tools/target/hexagon)

set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG   "-Wl,-soname,")
set(CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Wl,-soname,")

#Compiler Options
set(COMMON_FLAGS "-mcpu=hexagon${V_ARCH} -m${V_ARCH} -mhvx=${V_ARCH} -fvectorize -Wall -Werror -fno-zero-initialized-in-bss -G0 -fdata-sections -fpic ${XQF_ARGS}")

set(CMAKE_CXX_FLAGS_DEBUG          "${COMMON_FLAGS} -O0 -D_DEBUG -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g")
set(CMAKE_CXX_FLAGS_RELEASE        "${COMMON_FLAGS} -O3")

set(CMAKE_C_FLAGS_DEBUG            "${COMMON_FLAGS} -O0 -D_DEBUG -g")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${COMMON_FLAGS} -O3 -g")
set(CMAKE_C_FLAGS_RELEASE          "${COMMON_FLAGS} -O3")

set(CMAKE_ASM_FLAGS_DEBUG          "${COMMON_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_ASM_FLAGS_RELEASE        "${COMMON_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_ASM_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )

#Linker Options
set(CMAKE_C_CREATE_SHARED_LIBRARY   "${HEXAGON_LINKER} ${HEXAGON_PIC_SHARED_LINK_OPTIONS}")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${HEXAGON_LINKER} ${HEXAGON_PIC_SHARED_LINK_OPTIONS}")
