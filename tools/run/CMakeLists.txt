set(TARGET llama-run)
add_executable(${TARGET} run.cpp linenoise.cpp/linenoise.cpp)

# TODO: avoid copying this code block from common/CMakeLists.txt
set(LLAMA_RUN_EXTRA_LIBS "")
if (LLAMA_CURL)
    find_package(CURL REQUIRED)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    set(LLAMA_RUN_EXTRA_LIBS ${LLAMA_RUN_EXTRA_LIBS} ${CURL_LIBRARIES})
endif ()

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} ${LLAMA_RUN_EXTRA_LIBS})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
