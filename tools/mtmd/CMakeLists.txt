# mtmd

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd.h
            clip.cpp
            clip.h
            clip-impl.h
            mtmd-helper.cpp
            mtmd-helper.h
            mtmd-ios.cpp
            mtmd-ios.h
            )

# 在Apple平台上添加ANE相关文件
if(APPLE)
    target_sources(mtmd PRIVATE
        ane.h
        ane.mm
        ane_minicpm4v3b_vision_f16_b1.h
        ane_minicpm4v3b_vision_f16_b1.m
    )
    
    # 为Objective-C文件启用ARC
    set_source_files_properties(ane.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(ane_minicpm4v3b_vision_f16_b1.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

target_link_libraries     (mtmd PUBLIC ggml llama common)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../common)
target_include_directories(mtmd PRIVATE ../../include)
target_include_directories(mtmd PRIVATE ../../ggml/include)
target_include_directories(mtmd PRIVATE ../../src)
target_include_directories(mtmd PRIVATE ../../vendor)
target_compile_features   (mtmd PRIVATE cxx_std_17)

# 在Apple平台上链接CoreML和Accelerate框架
if(APPLE)
    target_link_libraries(mtmd PRIVATE 
        "-framework Foundation" 
        "-framework CoreML" 
        "-framework Accelerate"
        "-ObjC"
    )
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-ios.h
    )

# 在Apple平台上添加ANE公共头文件
if(APPLE)
    list(APPEND MTMD_PUBLIC_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/ane.h
    )
endif()

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

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

set(TARGET llama-mtmd-cli)
add_executable         (${TARGET} mtmd-cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
install                (TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION .)
target_link_libraries  (${TARGET} PRIVATE common mtmd Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# iOS接口测试
set(TARGET_IOS llama-mtmd-ios-test)
add_executable         (${TARGET_IOS} mtmd-ios-example.cpp)
set_target_properties  (${TARGET_IOS} PROPERTIES OUTPUT_NAME llama-mtmd-ios-test)
target_include_directories(${TARGET_IOS} PRIVATE .)
target_include_directories(${TARGET_IOS} PRIVATE ../..)
target_include_directories(${TARGET_IOS} PRIVATE ../../common)
target_include_directories(${TARGET_IOS} PRIVATE ../../include)
target_include_directories(${TARGET_IOS} PRIVATE ../../ggml/include)
target_include_directories(${TARGET_IOS} PRIVATE ../../src)
target_include_directories(${TARGET_IOS} PRIVATE ../../vendor)
target_link_libraries  (${TARGET_IOS} PRIVATE mtmd common llama ggml Threads::Threads)
target_compile_features(${TARGET_IOS} PRIVATE cxx_std_17)

# 确保依赖库先构建
add_dependencies(${TARGET_IOS} mtmd)
if (TARGET common)
    add_dependencies(${TARGET_IOS} common)
endif()

# 在Apple平台上为iOS测试添加额外的链接设置
if(APPLE)
    target_link_libraries(${TARGET_IOS} PRIVATE 
        "-framework Foundation" 
        "-framework CoreML" 
        "-framework Accelerate"
        "-ObjC"
    )
endif()
