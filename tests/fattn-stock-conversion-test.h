#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

// Capture the actual unmodified stock FA conversion output, not a second decoder formula.
static std::vector<ggml_fp16_t> stock_f16_rows(ggml_backend_t backend, ggml_type type,
        const std::vector<char> & encoded, int tokens, int heads, bool strided, int dimension = 256) {
    // Source type changes must not reuse a graph cached for another fixture at the same address.
    ggml_backend_ptr reference_backend(ggml_backend_dev_init(ggml_backend_get_device(backend),nullptr));
    if (!reference_backend) throw std::runtime_error("stock conversion backend failed");
    backend = reference_backend.get();
    constexpr int queries = 3;
    const size_t row_bytes = ggml_row_size(type,dimension);
    ggml_context_ptr context(ggml_init({2*1024*1024,nullptr,true}));
    if (!context) throw std::runtime_error("stock conversion context failed");
    auto * base = ggml_new_tensor_3d(context.get(),type,dimension,tokens*(strided ? 2 : 1),heads);
    auto * k = strided ? ggml_view_3d(context.get(),base,dimension,tokens,heads,base->nb[1],base->nb[2],0) : base;
    auto * v = ggml_new_tensor_3d(context.get(),GGML_TYPE_F16,dimension,tokens,heads);
    auto * q = ggml_new_tensor_3d(context.get(),GGML_TYPE_F32,dimension,queries,heads*6);
    auto * mask = ggml_new_tensor_2d(context.get(),GGML_TYPE_F16,tokens,queries);
    auto * output = ggml_flash_attn_ext(context.get(),q,k,v,mask,1.0f/16.0f,0.0f,0.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph,output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(),backend));
    if (!buffer) throw std::runtime_error("stock conversion allocation failed");
    ggml_backend_buffer_clear(buffer.get(),0);
    std::vector<char> storage(ggml_nbytes(base),0);
    for (int head = 0; head < heads; ++head) std::memcpy(
        storage.data()+size_t(head)*base->nb[2],encoded.data()+size_t(head)*tokens*row_bytes,size_t(tokens)*row_bytes);
    ggml_backend_tensor_set(base,storage.data(),0,storage.size());
    if (ggml_backend_graph_compute(backend,graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("stock conversion dispatch failed");
    ggml_backend_synchronize(backend);
    std::vector<ggml_fp16_t> converted(size_t(tokens)*heads*dimension);
    if (type == GGML_TYPE_F16) {
        for (int head = 0; head < heads; ++head) std::memcpy(
            converted.data()+size_t(head)*tokens*dimension,encoded.data()+size_t(head)*tokens*row_bytes,size_t(tokens)*row_bytes);
    } else {
        const auto extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(output,true,true);
        if (!extra.K) throw std::runtime_error("stock FA did not allocate converted keys");
        check_cuda(cudaMemcpy(converted.data(),reinterpret_cast<const void *>(extra.K),converted.size()*2,cudaMemcpyDeviceToHost));
    }
    return converted;
}

    int tokens, capacity;
