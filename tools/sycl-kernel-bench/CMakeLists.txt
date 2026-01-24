set(TARGET sycl-kernel-bench)
add_executable(${TARGET}
    main.cpp
    kernels/reference/onednn_fp16_gemm.cpp
    kernels/reference/onednn_int8_gemm.cpp
    kernels/reference/memory_bandwidth.cpp
    kernels/reference/roofline_compute.cpp
    kernels/tier1/mmvq_aos_baseline.cpp
    kernels/tier1/mmvq_soa_baseline.cpp
    kernels/tier1/mmvq_coalesced.cpp
    kernels/tier1/mmvq_slm_cached.cpp
    kernels/tier1/mmvq_prefetch.cpp
    kernels/tier1/mmvq_wide_load.cpp
    kernels/tier1/esimd_block_load.cpp
    kernels/tier1/esimd_slm.cpp
    kernels/tier2/mmvq_tier2_spec.cpp
    kernels/tier3/mmvq_tier3_spec.cpp
    kernels/tier4/hybrid_adaptive.cpp
    kernels/tier4/mmvq_xmx_fused.cpp
    kernels/tier4/coalesced_xmx_aligned.cpp
    kernels/tier4/esimd_hybrid.cpp
    kernels/tier4/esimd_cooperative.cpp
    kernels/tier4/q4_0_specialized.cpp
    kernels/tier4/q6_k_specialized.cpp
    kernels/tier4/mxfp4_native.cpp
    kernels/dpas_exploration/dpas_baseline.cpp
    kernels/dpas_exploration/dpas_sweep.cpp
    kernels/dpas_exploration/dpas_memory_patterns.cpp
)

target_link_libraries(${TARGET} PRIVATE common ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src
)

if (GGML_SYCL)
    target_compile_options(${TARGET} PRIVATE "-fsycl" "-Wno-narrowing" "-fsycl-device-code-split=per_kernel")
    target_link_options(${TARGET} PRIVATE "-fsycl" "-fsycl-device-code-split=per_kernel")
    target_compile_definitions(${TARGET} PRIVATE GGML_SYCL_WARP_SIZE=32)

    if (GGML_SYCL_TARGET STREQUAL "INTEL")
        find_package(DNNL REQUIRED)
        target_link_libraries(${TARGET} PRIVATE DNNL::dnnl)
        target_compile_definitions(${TARGET} PRIVATE GGML_SYCL_DNNL=1)
    endif()
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
