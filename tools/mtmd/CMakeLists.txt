# llava (legacy)

add_library(llava OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(llava PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)

target_compile_features(llava PRIVATE cxx_std_17)

add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)
if (BUILD_SHARED_LIBS)
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llava PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    target_link_libraries(llava_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS llava_shared LIBRARY)
endif()

# mtmd

add_library(mtmd OBJECT
            mtmd.cpp
            mtmd-helper.cpp
            mtmd.h
            clip.cpp
            clip.h
            clip-impl.h
            )

target_link_libraries(mtmd PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(mtmd PUBLIC .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../common) # for stb_image.h

target_compile_features(mtmd PRIVATE cxx_std_17)

add_library(mtmd_static STATIC $<TARGET_OBJECTS:mtmd>)
if (BUILD_SHARED_LIBS)
    set_target_properties(mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(mtmd_shared SHARED $<TARGET_OBJECTS:mtmd>)
    target_link_libraries(mtmd_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS mtmd_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(llava BUILD_INFO)
    add_dependencies(mtmd BUILD_INFO)
endif()

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

set(TARGET llama-mtmd-cli)
add_executable(${TARGET} mtmd-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-llava-clip-quantize-cli)
add_executable(${TARGET} clip-quantize-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-llava-clip-quantize-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
