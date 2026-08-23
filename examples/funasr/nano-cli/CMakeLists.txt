set(TARGET llama-funasr-cli)
add_executable(${TARGET} funasr-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/miniaudio)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(MSVC)
    # 源文件含 UTF-8 中文(如 "语音转写" / "no(截断)"), 避免 MSVC 按 GBK 解析报错
    target_compile_options(${TARGET} PRIVATE /utf-8)
    # Windows min/max 宏会污染 std::min/std::max; M_PI 需要 _USE_MATH_DEFINES
    target_compile_definitions(${TARGET} PRIVATE NOMINMAX _USE_MATH_DEFINES)
endif()

