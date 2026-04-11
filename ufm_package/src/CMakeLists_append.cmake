# ── UFM Custom Kernels — append to ggml/src/ggml-vulkan/CMakeLists.txt ────────
#
# Requires: Vulkan SDK 1.4.341+ installed, run scripts/compile_shaders.bat first
# cmake_minimum_required version in root llama.cpp CMakeLists.txt is 3.21
# (as of b5000+) — no need to raise it here.
#
# Usage:
#   cat src/CMakeLists_append.cmake >> ggml/src/ggml-vulkan/CMakeLists.txt

cmake_minimum_required(VERSION 3.21)  # matches llama.cpp root minimum

set(UFM_SPV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spv_out")

if(EXISTS "${UFM_SPV_DIR}")
    target_compile_definitions(ggml-vulkan PRIVATE UFM_CUSTOM_KERNELS=1)
    # Tell MSVC to treat source files as UTF-8 — suppresses C4828 on SPV headers
    if(MSVC)
        target_compile_options(ggml-vulkan PRIVATE /utf-8)
    endif()
    target_include_directories(ggml-vulkan PRIVATE "${UFM_SPV_DIR}")
    message(STATUS "[UFM] Custom kernels ENABLED — ${UFM_SPV_DIR}")

    # Verify all expected SPV headers are present
    set(UFM_EXPECTED_HEADERS
        flash_attention_kv_quant_spv.h
        flash_attention_paged_spv.h
        kvcache_update_q8_spv.h
        linear_coop_32_spv.h
        linear_coop_fp8_spv.h
        linear_coop_q4k_spv.h
        linear_coop_q4k_silu_spv.h
        linear_coop_q4k_w32_spv.h
        linear_coop_q8_spv.h
        linear_coop_q8_w32_spv.h
    )
    foreach(HDR ${UFM_EXPECTED_HEADERS})
        if(NOT EXISTS "${UFM_SPV_DIR}/${HDR}")
            message(WARNING "[UFM] Missing: ${UFM_SPV_DIR}/${HDR}  — re-run compile_shaders.bat")
        endif()
    endforeach()
else()
    message(WARNING "[UFM] spv_out/ not found at ${UFM_SPV_DIR} — run scripts/compile_shaders.bat first")
endif()
