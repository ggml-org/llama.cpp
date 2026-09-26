# the readout library: a shared component, named like mtmd rather than like an executable
add_library(system-one STATIC system-one.cpp system-one.h)
target_link_libraries(system-one PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(system-one PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# it is linked into libllama-server-impl.so, so it has to be position independent
if (BUILD_SHARED_LIBS)
    set_target_properties(system-one PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_compile_features(system-one PRIVATE cxx_std_17)

set(TARGET llama-system-one)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE system-one llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
