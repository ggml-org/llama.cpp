set(TARGET frameforge-sidecar)

add_executable(${TARGET}
    frameforge-sidecar.cpp
    frameforge-schema.cpp
    frameforge-validator.cpp
    frameforge-json.cpp
    frameforge-ipc.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/whisper/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/ggml/include
)

target_link_libraries(${TARGET} PRIVATE
    common
    llama
)

# Link Whisper library
# We need to add whisper as a subdirectory or link to it
# For now, we'll create a target for whisper if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/external/whisper/CMakeLists.txt)
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/whisper ${CMAKE_BINARY_DIR}/whisper EXCLUDE_FROM_ALL)
    target_link_libraries(${TARGET} PRIVATE whisper)
else()
    message(WARNING "Whisper not found at ${CMAKE_SOURCE_DIR}/external/whisper, frameforge-sidecar will build without Whisper support")
endif()

# Platform-specific libraries
if(WIN32)
    # Windows-specific libraries
elseif(UNIX)
    # Unix-specific libraries
    target_link_libraries(${TARGET} PRIVATE pthread)
endif()

install(TARGETS ${TARGET} RUNTIME)
