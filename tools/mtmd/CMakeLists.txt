# mtmd

add_library(mtmd OBJECT
            mtmd.cpp
            mtmd-audio.cpp
            mtmd.h
            clip.cpp
            clip.h
            clip-impl.h
            )

target_link_libraries(mtmd PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mtmd PUBLIC .)
target_include_directories(mtmd PRIVATE ../..)
target_compile_features(mtmd PRIVATE cxx_std_17)

# compile the helper separately, to avoid long compile times with miniaudio.h and stb_image.h

add_library(mtmd_helper OBJECT
            mtmd-helper.cpp
            mtmd-helper.h
            )

target_link_libraries(mtmd_helper PRIVATE ggml llama mtmd ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(mtmd_helper PUBLIC .)
target_include_directories(mtmd_helper PRIVATE ./vendor)
target_include_directories(mtmd_helper PRIVATE ../..)
target_compile_features(mtmd_helper PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(mtmd_shared SHARED $<TARGET_OBJECTS:mtmd>)
    target_link_libraries(mtmd_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS mtmd_shared LIBRARY)

    set_target_properties(mtmd_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd_helper PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(mtmd_helper_shared SHARED $<TARGET_OBJECTS:mtmd>)
    target_link_libraries(mtmd_helper_shared PRIVATE ggml llama mtmd ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS mtmd_helper_shared LIBRARY)
endif()

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd_helper PRIVATE -Wno-cast-qual)
endif()

if(TARGET BUILD_INFO)
    add_dependencies(mtmd BUILD_INFO)
    add_dependencies(mtmd_helper BUILD_INFO)
endif()

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

set(TARGET llama-mtmd-cli)
add_executable(${TARGET} mtmd-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common mtmd mtmd_helper ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
