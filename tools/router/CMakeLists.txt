set(TARGET llama-router)

add_executable(${TARGET}
    router.cpp
    router-app.cpp
    router-config.cpp
    router-endpoints.cpp
    logging.cpp
    router-process.cpp
    router-proxy.cpp
    router-scanner.cpp
)

target_link_libraries(${TARGET} PRIVATE common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
