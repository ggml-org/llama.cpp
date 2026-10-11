#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

// Keep source pointers, dimensions and strides unchanged while reusing a captured conversion graph.
int main() {
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name("CUDA0",nullptr));
    if (!backend) return 77;
    ggml_context_ptr context(ggml_init({1024*1024,nullptr,true}));
    auto * source = ggml_new_tensor_2d(context.get(),GGML_TYPE_F16,256,4);
    auto * output = ggml_cast(context.get(),source,GGML_TYPE_F32);
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph,output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(),backend.get()));
    if (!buffer) return 1;
    std::vector<uint16_t> encoded(1024,0x3c00);
    ggml_backend_tensor_set(source,encoded.data(),0,encoded.size()*sizeof(uint16_t));
    std::vector<float> actual(1024);
    for (auto type : {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_F16,GGML_TYPE_BF16}) {
        source->type = type;
        const float expected = type == GGML_TYPE_F16 ? 1.0f : 0.0078125f;
        for (int repeat = 0; repeat < 5; ++repeat) {
            if (ggml_backend_graph_compute(backend.get(),graph) != GGML_STATUS_SUCCESS) return 1;
            ggml_backend_synchronize(backend.get());
            ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
            for (float value : actual) if (value != expected) {
                std::fprintf(stderr,"graph source type %s: expected %g, got %g\n",ggml_type_name(type),expected,value);
                return 1;
            }
        }
    }
    std::puts("Type-changing graph capture reuse passed");
    return 0;
}
