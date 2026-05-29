set(TARGET llama-app)

add_executable(${TARGET} llama.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama)

target_link_libraries(${TARGET} PRIVATE
    llama-server-impl
    llama-cli-impl
    llama-completion-impl
    llama-bench-impl
    llama-batched-bench-impl
    llama-fit-params-impl
    llama-quantize-impl
    llama-perplexity-impl
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Automatically add all files from the 'licenses' directory
file(GLOB EXTRA_LICENSES "${CMAKE_SOURCE_DIR}/licenses/LICENSE-*")

foreach(FILE_PATH ${EXTRA_LICENSES})
    get_filename_component(FILE_NAME "${FILE_PATH}" NAME)
    string(REGEX REPLACE "^LICENSE-" "" NAME "${FILE_NAME}")
    license_add_file("${NAME}" "${FILE_PATH}")
endforeach()

license_generate(${TARGET})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
