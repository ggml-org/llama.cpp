set(TARGET llama-imatrix)
add_executable(${TARGET}
    imatrix.cpp
    # offline DFlash feature-capture file format (shared with the training driver)
    ${CMAKE_CURRENT_SOURCE_DIR}/../quantize/tessera/tessera-features.cpp
    # unified spec-calib telemetry writer (v3 schema, with v1 adapter)
    telemetry-record.cpp
)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../quantize/tessera)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # AIX's flock() function comes from libbsd.a
    target_link_libraries(${TARGET} PRIVATE -lbsd)
endif()
