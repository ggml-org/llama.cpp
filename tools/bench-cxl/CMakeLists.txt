if (GGML_CXL)
    set(TARGET llama-bench-cxl)
    add_executable(${TARGET} bench-cxl.cpp
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cxl/cxl-device.cpp)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cxl
        ${CMAKE_SOURCE_DIR}/ggml/include)
    target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
endif()
