# mediagen

find_package(Threads REQUIRED)

add_library(mediagen
            mediagen.cpp
            mediagen.h
            mediagen-impl.h
            mediagen-common.cpp
            mediagen-ltx.h
            mediagen-ltx-graph.h
            mediagen-ltx-text.cpp
            mediagen-ltx-enhance.cpp
            mediagen-ltx-dit.cpp
            mediagen-ltx-vae.cpp
            mediagen-ltx-audio.cpp
            )

set_target_properties(mediagen PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0
    MACHO_COMPATIBILITY_VERSION 0
)

target_link_libraries     (mediagen PUBLIC  ggml llama)
target_link_libraries     (mediagen PRIVATE Threads::Threads vendor::nlohmann vendor::stb vendor::sheredom)
if (LLAMA_SUBPROCESS)
    target_compile_definitions(mediagen PRIVATE MEDIAGEN_FFMPEG) # mp4 output via ffmpeg
endif()
target_include_directories(mediagen PUBLIC  .)
target_include_directories(mediagen PRIVATE ../..)
target_compile_features   (mediagen PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mediagen PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mediagen PRIVATE LLAMA_BUILD)
    target_compile_definitions(mediagen PUBLIC  LLAMA_SHARED)
endif()

set(MEDIAGEN_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mediagen.h
    )

set_target_properties(mediagen
    PROPERTIES
        PUBLIC_HEADER "${MEDIAGEN_PUBLIC_HEADERS}")

install(TARGETS mediagen LIBRARY PUBLIC_HEADER)

add_executable(llama-mediagen mediagen-cli.cpp)
set_target_properties(llama-mediagen PROPERTIES OUTPUT_NAME llama-mediagen)
install(TARGETS llama-mediagen RUNTIME)
target_link_libraries(llama-mediagen PRIVATE llama-common mediagen ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-mediagen PRIVATE cxx_std_17)
