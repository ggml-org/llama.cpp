# llama-fit-params-impl: fit-params logic, reusable by app

set(TARGET llama-fit-params-impl)

add_library(${TARGET} fit-params.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-fit-params executable

set(TARGET llama-fit-params)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-fit-params-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
