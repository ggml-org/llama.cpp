# llama-completion-impl: completion logic, reusable by app

set(TARGET llama-completion-impl)

add_library(${TARGET} STATIC completion.cpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

# llama-completion executable

set(TARGET llama-completion)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-completion-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
