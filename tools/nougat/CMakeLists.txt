set(TARGET nougat-cli)

# Add executable
add_executable(${TARGET} ../nougat-cli.cpp)

# Link with llama library
target_link_libraries(${TARGET} PRIVATE
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)

# Include directories
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tools
)

# Compile flags
llama_add_compile_flags()

# Set output name
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME "nougat-cli")

# Install target
if(LLAMA_INSTALL)
    install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
endif()