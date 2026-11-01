set(TARGET llama-funasr-cli)
add_executable(${TARGET} funasr-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/miniaudio)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(MSVC)
    # the source contains a UTF-8 Chinese prompt string; tell MSVC not to parse as GBK
    target_compile_options(${TARGET} PRIVATE /utf-8)
    # Windows min/max macros clash with std::min/std::max; M_PI needs _USE_MATH_DEFINES
    target_compile_definitions(${TARGET} PRIVATE NOMINMAX _USE_MATH_DEFINES)
endif()

