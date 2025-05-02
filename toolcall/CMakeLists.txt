
set(TARGET toolcall)

set(SOURCES
    client.cpp
    mcp_messages.cpp
    mcp_stdio_transport.cpp
    params.cpp
)

set(HEADERS
    toolcall-params.h
    toolcall-client.h
    mcp_transport.h
    mcp_messages.h
    mcp_stdio_transport.h
)

add_library(${TARGET} STATIC ${SOURCES} ${HEADERS})

target_include_directories(${TARGET}  # Right now only for "json.hpp"
    PRIVATE $<TARGET_PROPERTY:common,INTERFACE_INCLUDE_DIRECTORIES>)

if (LLAMA_CURL)
    find_package(CURL REQUIRED)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    find_library(CURL_LIBRARY curl REQUIRED)

    target_link_libraries(${TARGET} PRIVATE ${CURL_LIBRARY})
    target_sources(${TARGET} PRIVATE mcp_sse_transport.cpp mcp_sse_transport.h)

endif()

target_compile_definitions(${TARGET} INTERFACE LLAMA_USE_TOOLCALL)
target_include_directories(${TARGET} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_features (${TARGET} PUBLIC cxx_std_17)
