set(TARGET llama-run)

if (MINGW)
    # fix: https://github.com/ggml-org/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

# Include server source files (except server.cpp which has its own main())
set(SERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../server)
set(READLINE_DIR ${PROJECT_SOURCE_DIR}/vendor/readline.cpp)
set(TARGET_SRCS
    run.cpp
    ${SERVER_DIR}/server-context.cpp
    ${SERVER_DIR}/server-context.h
    ${SERVER_DIR}/server-task.cpp
    ${SERVER_DIR}/server-task.h
    ${SERVER_DIR}/server-queue.cpp
    ${SERVER_DIR}/server-queue.h
    ${SERVER_DIR}/server-common.cpp
    ${SERVER_DIR}/server-common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/run-chat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/run-chat.h
    ${READLINE_DIR}/src/readline.cpp
    ${READLINE_DIR}/src/buffer.cpp
    ${READLINE_DIR}/src/history.cpp
    ${READLINE_DIR}/src/terminal.cpp
)

add_executable(${TARGET} ${TARGET_SRCS})

# TODO: avoid copying this code block from common/CMakeLists.txt
set(LLAMA_RUN_EXTRA_LIBS "")
if (LLAMA_CURL)
    find_package(CURL REQUIRED)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    set(LLAMA_RUN_EXTRA_LIBS ${LLAMA_RUN_EXTRA_LIBS} ${CURL_LIBRARIES})
endif ()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # AIX's flock() function comes from libbsd.a
    target_link_libraries(${TARGET} PRIVATE -lbsd)
endif()

# Include directories for server headers and readline
target_include_directories(${TARGET} PRIVATE ${SERVER_DIR})
target_include_directories(${TARGET} PRIVATE ${SERVER_DIR}/../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${READLINE_DIR}/include)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${TARGET} PRIVATE common mtmd llama ${CMAKE_THREAD_LIBS_INIT} ${LLAMA_RUN_EXTRA_LIBS})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
