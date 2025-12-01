set(TARGET llama-router)

set(TARGET_SRCS
    router.cpp
    router-app.cpp
    router-config.cpp
    router-admin.cpp
    router-endpoints.cpp
    logging.cpp
    router-process.cpp
    router-proxy.cpp
    router-scanner.cpp
)

set(PUBLIC_ASSETS
    index.html.gz
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${PROJECT_SOURCE_DIR}/tools/server/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})

target_link_libraries(${TARGET} PRIVATE common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
