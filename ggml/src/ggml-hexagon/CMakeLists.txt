file(TO_CMAKE_PATH "${HEXAGON_SDK_ROOT}"   HEXAGON_SDK_ROOT)
file(TO_CMAKE_PATH "${HEXAGON_TOOLS_ROOT}" HEXAGON_TOOLS_ROOT)

if (NOT IS_DIRECTORY "${HEXAGON_SDK_ROOT}")
    message(FATAL_ERROR "Make sure HEXAGON_SDK_ROOT point to the correct Hexagon SDK installation.")
endif()

if (NOT IS_DIRECTORY "${HEXAGON_TOOLS_ROOT}")
    message("Try to read HEXAGON_TOOLS_ROOT from hexagon_sdk.json")
    file(READ "${HEXAGON_SDK_ROOT}/hexagon_sdk.json" HEXAGON_SDK_CONFIG_PATH)
    string(JSON HEXAGON_TOOLS_PATH GET ${HEXAGON_SDK_CONFIG_PATH} "root" "tools" "info" 0 "path")
    message("Found HEXAGON_TOOLS_PATH: ${HEXAGON_TOOLS_PATH}")
    set(HEXAGON_TOOLS_ROOT "${HEXAGON_SDK_ROOT}/${HEXAGON_TOOLS_PATH}")
    file(TO_CMAKE_PATH "${HEXAGON_TOOLS_ROOT}" HEXAGON_TOOLS_ROOT)
    if (NOT IS_DIRECTORY "${HEXAGON_TOOLS_ROOT}")
        message(FATAL_ERROR "Make sure HEXAGON_TOOLS_ROOT point to the correct Hexagon SDK installation.")
    endif()
endif()

option(GGML_HEXAGON_JZ "ggml-hexagon: use JZ's alternative AP implementation (builds DSP skel via kernels/Makefile)" OFF)

set(TARGET_NAME ggml-hexagon)

if (GGML_HEXAGON_JZ)
    # --- JZ's ggml-hexagon implementation ---
    # kernels/Makefile expects HEXAGON_SDK_PATH / HEXAGON_TOOLS_PATH variable names
    set(HEXAGON_SDK_PATH   ${HEXAGON_SDK_ROOT})
    set(HEXAGON_TOOLS_PATH ${HEXAGON_TOOLS_ROOT})

    if(NOT DEFINED HTP_ARCH_VERSION)
        message(FATAL_ERROR "GGML_HEXAGON_JZ=ON requires HTP_ARCH_VERSION (one of: v73, v75, v79, v81)")
    endif()

    set(CHECK_HTP_ARCH "WRONG")
    foreach (feat v73 v75 v79 v81)
        if (${feat} STREQUAL ${HTP_ARCH_VERSION})
            set(CHECK_HTP_ARCH "GOOD")
        endif()
    endforeach()
    if (${CHECK_HTP_ARCH} STREQUAL "WRONG")
        message(FATAL_ERROR "GGML_HEXAGON_JZ=ON requires HTP_ARCH_VERSION to be one of: v73, v75, v79, v81")
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(DEBUG_FLAG "-DDEBUG -Wall")
    else()
        set(DEBUG_FLAG "-DNDEBUG -Wall")
    endif()

    set(OPT_FLAG " -O3 -march=armv8.7-a+dotprod+fp16+i8mm -ffp-model=fast -fno-finite-math-only")

    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        find_library(LOG_LIB log)
        add_library(cdsprpc SHARED IMPORTED)
        set_target_properties(cdsprpc PROPERTIES
            IMPORTED_LOCATION ${HEXAGON_SDK_PATH}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so)
        set(HEXAGON_LINK_LIBRARIES ${LOG_LIB} cdsprpc)
        set(HEXAGON_DEFAULT_LIB_SEARCH_PATH "/data/local/tmp/" CACHE STRING "customized library search path for hexagon backend")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(HEXAGON_DEFAULT_LIB_SEARCH_PATH "C:\\" CACHE STRING "customized library search path for hexagon backend")
    else()
        message(FATAL_ERROR "ggml-hexagon (JZ) only available on Android and Windows(Windows on ARM)")
    endif()

    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGGML_USE_HEXAGON ${DEBUG_FLAG} ${OPT_FLAG}")

    include_directories(
        ${HEXAGON_SDK_PATH}/incs
        ${HEXAGON_SDK_PATH}/incs/stddef
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/incs
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/remote/ship/android_Debug_aarch64
        ${HEXAGON_SDK_PATH}/utils/examples
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/android_aarch64
        ${HEXAGON_SDK_PATH}/libs/atomic/inc
        ${HEXAGON_SDK_PATH}/libs/atomic/android_Debug_aarch64/ship
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/kernels/)

    # Generate ggml_dsp_{stub.c,skel.c,h} from ggml_dsp.idl via qaic
    set(GGML_DSP_STUB ${CMAKE_CURRENT_LIST_DIR}/kernels/ggml_dsp_stub.c)
    set(QAIC_BIN ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic)
    if (NOT EXISTS "${GGML_DSP_STUB}" AND EXISTS "${QAIC_BIN}")
        message(STATUS "hexagon-jz: generating ggml_dsp stubs from ggml_dsp.idl")
        execute_process(
            COMMAND ${QAIC_BIN} -mdll
                -o ${CMAKE_CURRENT_LIST_DIR}/kernels
                -I${HEXAGON_SDK_ROOT}/incs
                -I${HEXAGON_SDK_ROOT}/incs/stddef
                -I${HEXAGON_SDK_ROOT}/ipc/fastrpc/incs
                ${CMAKE_CURRENT_LIST_DIR}/kernels/ggml_dsp.idl
            RESULT_VARIABLE QAIC_RESULT)
        if (NOT QAIC_RESULT EQUAL 0)
            message(FATAL_ERROR "hexagon-jz: qaic failed to generate ggml_dsp stubs")
        endif()
    endif()

    ggml_add_backend_library(${TARGET_NAME}
        ggml-hexagon-jz.cpp
        htp-drv.cpp
        htp-drv.h
        libdl.h
        kernels/ggml_dsp_stub.c)

    target_include_directories(${TARGET_NAME} PRIVATE
        ${HEXAGON_SDK_PATH} ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/kernels)
    target_link_libraries(${TARGET_NAME} PRIVATE ${HEXAGON_LINK_LIBRARIES})

    string(REGEX REPLACE "/$" "" HEXAGON_DEFAULT_LIB_SEARCH_PATH "${HEXAGON_DEFAULT_LIB_SEARCH_PATH}")
    target_compile_definitions(${TARGET_NAME} PRIVATE
        HEXAGON_DEFAULT_LIB_SEARCH_PATH="${HEXAGON_DEFAULT_LIB_SEARCH_PATH}/")

    # Build DSP skels for all supported HTP arch versions via kernels/Makefile.
    # CMake owns the multi-version build; scripts no longer call build_extra_dsp_skels.
    # Note: kernels/Makefile deletes generated stubs (ggml_dsp_skel.c/.h/_stub.c) after
    # each link, so qaic must regenerate them before each arch build to avoid
    # "missing ggml_dsp_skel.c" failures on the 2nd/3rd/4th version.
    set(JZ_DSP_VERSIONS v73 v75 v79 v81)
    set(JZ_DSP_BUILD_CMDS "")
    foreach(V ${JZ_DSP_VERSIONS})
        list(APPEND JZ_DSP_BUILD_CMDS
            COMMAND ${QAIC_BIN} -mdll
                -o ${CMAKE_CURRENT_LIST_DIR}/kernels
                -I${HEXAGON_SDK_ROOT}/incs
                -I${HEXAGON_SDK_ROOT}/incs/stddef
                -I${HEXAGON_SDK_ROOT}/ipc/fastrpc/incs
                ${CMAKE_CURRENT_LIST_DIR}/kernels/ggml_dsp.idl
            COMMAND make -C ${CMAKE_CURRENT_LIST_DIR}/kernels/ clean
            COMMAND make -C ${CMAKE_CURRENT_LIST_DIR}/kernels/
                HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH}
                HEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH}
                HTP_ARCH_VERSION=${V}
                DEBUG_FLAG=${DEBUG_FLAG}
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_LIST_DIR}/kernels/libggmldsp-skel.so
                ${CMAKE_BINARY_DIR}/bin/libggmldsp-skel-${V}.so
        )
        if (${V} STREQUAL ${HTP_ARCH_VERSION})
            list(APPEND JZ_DSP_BUILD_CMDS
                COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_LIST_DIR}/kernels/libggmldsp-skel.so
                    ${CMAKE_BINARY_DIR}/bin/libggmldsp-skel.so
            )
        endif()
    endforeach()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        ${JZ_DSP_BUILD_CMDS}
        COMMENT "build JZ hexagon DSP skels (v73, v75, v79, v81)")

    # Copy runtime configuration file
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/scripts/ggml-hexagon.cfg ${CMAKE_BINARY_DIR}/bin/
        COMMENT "copy ggml-hexagon.cfg")

else()
    # --- Qualcomm's official implementation (default, upstream) ---
    message(STATUS "hexagon: using ${HEXAGON_SDK_ROOT} and ${HEXAGON_TOOLS_ROOT} for building libggml-htp skels")

    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
    include(ExternalProject)

    option(GGML_HEXAGON_HTP_DEBUG  "ggml-hexagon: enable HTP debug output" OFF)
    set(GGML_HEXAGON_HTP_CERT  "$ENV{HEXAGON_HTP_CERT}" CACHE PATH "ggml-hexagon: enable HTP library signing using certificate")

    add_library(htp_iface OBJECT
        ${CMAKE_CURRENT_BINARY_DIR}/htp_iface_stub.c)

    set_target_properties(htp_iface PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(htp_iface PUBLIC
        ${HEXAGON_SDK_ROOT}/incs
        ${HEXAGON_SDK_ROOT}/incs/stddef
        ${HEXAGON_SDK_ROOT}/utils/examples
        ${CMAKE_CURRENT_SOURCE_DIR}/htp
        ${CMAKE_CURRENT_BINARY_DIR})

    build_idl(htp/htp_iface.idl htp_iface)

    if (CMAKE_SYSTEM_NAME MATCHES Android)
        target_link_options(htp_iface PUBLIC -llog -ldl)
    elseif (CMAKE_SYSTEM_NAME MATCHES Windows)
        target_precompile_headers(htp_iface PUBLIC <sal.h>)
    else()
        target_link_options(htp_iface PUBLIC -ldl)
    endif()

    ggml_add_backend_library(${TARGET_NAME}
        ggml-hexagon.cpp
        htp-drv.cpp
        htp-drv.h
        libdl.h
        ../../include/ggml-hexagon.h)

    target_link_libraries(${TARGET_NAME} PRIVATE htp_iface)
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/htp ${CMAKE_CURRENT_BINARY_DIR})

    # Build HTP skels
    set(HTP_SKELS)
    function(build_htp_skel V)
        ExternalProject_Add(htp-${V}
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
            BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-${V}.so
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/htp/cmake-toolchain.cmake
                -DCMAKE_INSTALL_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}
                -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
                -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}
                -DHEXAGON_HTP_DEBUG=${GGML_HEXAGON_HTP_DEBUG}
                -DDSP_VERSION=${V}
                -DPREBUILT_LIB_DIR="toolv19_${V}")
        list(APPEND HTP_SKELS ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-${V}.so)
        set(HTP_SKELS ${HTP_SKELS} PARENT_SCOPE)
    endfunction()

    build_htp_skel(v73)
    build_htp_skel(v75)
    build_htp_skel(v79)
    build_htp_skel(v81)

    install(FILES ${HTP_SKELS} TYPE LIB)

    if (CMAKE_SYSTEM_NAME MATCHES Windows AND GGML_HEXAGON_HTP_CERT)
        file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/arm64"      WINSDK_BIN0_ARM64)
        file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/x86"        WINSDK_BIN0_X86)
        file(TO_CMAKE_PATH "$ENV{WindowsSdkVerBinPath}/arm64" WINSDK_BIN1_ARM64)
        file(TO_CMAKE_PATH "$ENV{WindowsSdkVerBinPath}/x86"   WINSDK_BIN1_X86)

        set(WINSDK_PATHS ${WINSDK_BIN0_ARM64} ${WINSDK_BIN0_X86} ${WINSDK_BIN1_ARM64} ${WINSDK_BIN1_X86})

        find_program(INF2CAT  NAMES inf2cat.exe  PATHS ${WINSDK_PATHS} REQUIRED)
        find_program(SIGNTOOL NAMES signtool.exe PATHS ${WINSDK_PATHS} REQUIRED)

        message(STATUS "hexagon: using ${GGML_HEXAGON_HTP_CERT} to sign libggml-htp skels")

        set(LIBGGML_HTP_CAT ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp.cat)
        add_custom_target(libggml-htp-cat
            BYPRODUCTS ${LIBGGML_HTP_CAT}
            DEPENDS libggml-htp.inf ${HTP_SKELS}
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libggml-htp.inf ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${INF2CAT} /driver:${CMAKE_CURRENT_BINARY_DIR} /os:10_25H2_ARM64
            COMMAND ${SIGNTOOL} sign /fd sha256 /f ${GGML_HEXAGON_HTP_CERT} ${LIBGGML_HTP_CAT}
            COMMENT "generating and signing libggml-htp.cat file"
            VERBATIM
        )

        add_dependencies(${TARGET_NAME} libggml-htp-cat)
        install(FILES ${LIBGGML_HTP_CAT} TYPE LIB)
    endif()
endif()
