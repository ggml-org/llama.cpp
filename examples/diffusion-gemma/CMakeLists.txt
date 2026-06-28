set(TARGET llama-diffusion-gemma-cli)
add_executable(${TARGET} diffusion-gemma-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
# mtmd (tools/) is added after examples/, so add its include dir explicitly for the headers
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mtmd)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# OpenAI-compatible HTTP server for the block-diffusion models (llama-server analogue)
set(TARGET llama-diffusion-gemma-server)
add_executable(${TARGET} diffusion-gemma-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mtmd
    ${CMAKE_SOURCE_DIR}/vendor)   # cpp-httplib/httplib.h, nlohmann/json.hpp
target_compile_features(${TARGET} PRIVATE cxx_std_17)
