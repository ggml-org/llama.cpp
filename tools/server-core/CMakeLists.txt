# serve

set(TARGET server-core)

add_library(${TARGET}
    server-core.cpp
    server-utils.hpp
)

target_include_directories(${TARGET} PUBLIC .)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${TARGET} PRIVATE common mtmd)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC  LLAMA_SHARED)
endif()

set(serve_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/server-core.h
    ${CMAKE_CURRENT_SOURCE_DIR}/server-core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/server-utils.hpp
    )

set_target_properties(${TARGET}
    PROPERTIES
    PUBLIC_HEADER "${serve_PUBLIC_HEADERS}")

install(TARGETS ${TARGET} LIBRARY PUBLIC_HEADER)

if (TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
endif()
