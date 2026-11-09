# quantize-raw: quantize a raw f32 tensor to a ggml quant type (stdin -> stdout)
# used by convert_hf_to_gguf.py to quantize K/I-quants directly from safetensors,
# avoiding an FP16 GGUF intermediate (gguf-py can only dequantize these types)

set(TARGET quantize-raw)

add_executable(${TARGET} quantize-raw.cpp)
# llama-common provides common_imatrix_load (common/imatrix-loader.h)
target_link_libraries(${TARGET} PRIVATE ggml llama-common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
