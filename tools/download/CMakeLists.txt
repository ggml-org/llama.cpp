# llama-download-impl: model download logic, reusable by app

set(TARGET llama-download-impl)

add_library(${TARGET} download.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-download executable

set(TARGET llama-download)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-download-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
