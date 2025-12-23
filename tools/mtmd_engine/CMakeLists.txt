# mtmd

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd.h
            mtmd-helper.cpp
            mtmd-helper.h
            clip.cpp
            clip.h
            clip-impl.h
            clip-model.h
            clip-graph.h
            models/models.h
            models/cogvlm.cpp
            models/conformer.cpp
            models/glm4v.cpp
            models/internvl.cpp
            models/kimivl.cpp
            models/llama4.cpp
            models/llava.cpp
            models/minicpmv.cpp
            models/pixtral.cpp
            models/qwen2vl.cpp
            models/qwen3vl.cpp
            models/siglip.cpp
            models/whisper-enc.cpp
            )

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../vendor)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd        BUILD_INFO)
    add_dependencies(mtmd-helper BUILD_INFO)
endif()

# if mtmd is linked against common, we throw an error
if (TARGET mtmd)
    get_target_property(libs mtmd LINK_LIBRARIES)
    if (libs AND "common" IN_LIST libs)
        message(FATAL_ERROR "mtmd is designed to be a public library.\n"
                            "It must not link against common")
    endif()
endif()

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

set(TARGET llama-mtmd-cli)
add_executable         (${TARGET} mtmd-cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_library(mtmd_engine SHARED mtmd-engine.cpp)
target_compile_definitions(mtmd_engine PRIVATE LLAMA_BUILD)
target_link_libraries  (mtmd_engine PUBLIC common mtmd Threads::Threads)
target_compile_features(mtmd_engine PRIVATE cxx_std_17)
install(TARGETS mtmd_engine RUNTIME)

add_executable(test_mtmd_engine111 test_engine.cpp)
target_link_libraries(test_mtmd_engine111 PRIVATE mtmd_engine)

target_include_directories(mtmd_engine PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/srcs>"
    "$<INSTALL_INTERFACE:include>"
)

macro(export_lib target_name)
install(TARGETS ${target_name}
    EXPORT ${target_name}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
# install(EXPORT ${target_name}Targets
#     FILE ${target_name}-config.cmake
#     DESTINATION lib/cmake/
# )
endmacro()

export_lib(mtmd_engine)
# export_lib(mtmd)
# export_lib(common)
# export_lib(build_info)
# export_lib(cpp-httplib)
# export_lib(llama)
# export_lib(ggml)
# export_lib(ggml-base)
# export_lib(ggml-cpu)
# export_lib(ggml-cuda)
# install(TARGETS mtmd_engine
#     EXPORT mtmd_engineTargets
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
# install(EXPORT mtmd_engineTargets
#     FILE mtmd_engine-config.cmake
#     DESTINATION lib/cmake/
# )

# install(TARGETS common
#     EXPORT commonTargets
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
# install(EXPORT commonTargets
#     FILE common-config.cmake
#     DESTINATION lib/cmake/
# )

install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/mtmd-engine.hpp
    DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/
)
if(WIN32)
install(FILES $<TARGET_PDB_FILE:mtmd_engine>
    DESTINATION "pdb/$<CONFIG>"
    OPTIONAL
)
endif()