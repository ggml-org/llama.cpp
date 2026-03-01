set(TARGET onednn-woq-repro)
add_executable(${TARGET}
    onednn_woq_repro.cpp
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_SYCL)
    target_compile_options(${TARGET} PRIVATE "-fsycl" "-Wno-narrowing")
    target_link_options(${TARGET} PRIVATE "-fsycl")

    if (GGML_SYCL_TARGET STREQUAL "INTEL")
        find_package(DNNL REQUIRED)
        target_link_libraries(${TARGET} PRIVATE DNNL::dnnl)
        if (DEFINED DNNL_INCLUDE_DIRS)
            target_include_directories(${TARGET} PRIVATE ${DNNL_INCLUDE_DIRS})
        endif()
        if (EXISTS "/opt/intel/oneapi/dnnl/2025.3/include")
            target_include_directories(${TARGET} PRIVATE /opt/intel/oneapi/dnnl/2025.3/include)
        endif()
    endif()
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
