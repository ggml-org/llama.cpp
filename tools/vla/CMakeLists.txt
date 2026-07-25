# VLA runtime: standalone component GGUFs conditioned by libllama hidden states.

set(TARGET vla)

add_library(${TARGET}
    vla.cpp
    vla.h
    vla-impl.h
    models/models.h
    models/minicpm-robot.cpp
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE VLA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC VLA_SHARED)
endif()

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC ggml llama)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set_target_properties(${TARGET} PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/vla.h")
install(TARGETS ${TARGET} LIBRARY PUBLIC_HEADER)

add_executable(test-vla test-vla.cpp)
target_link_libraries(test-vla PRIVATE vla)
target_compile_features(test-vla PRIVATE cxx_std_17)

add_executable(llama-vla-cli vla-cli.cpp)
target_link_libraries(llama-vla-cli PRIVATE llama-common mtmd vla Threads::Threads)
target_compile_features(llama-vla-cli PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS llama-vla-cli RUNTIME)
endif()
