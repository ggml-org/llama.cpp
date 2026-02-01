#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace sycl_bench {

struct ModelMatmulShape {
    std::string name;
    ggml_type   type = GGML_TYPE_F32;
    int64_t     dim_m = 0;
    int64_t     dim_n = 0;
    int64_t     dim_k = 0;
    int64_t     instances = 1;
    int         n_dims = 0;
    int64_t     dims[GGML_MAX_DIMS] = {0, 0, 0, 0};
};

inline bool load_model_matmul_shapes(const std::string & path,
                                     std::vector<ModelMatmulShape> & out,
                                     std::string & error) {
    out.clear();

    ggml_context * ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf || !ctx) {
        error = "Failed to open GGUF model or create ggml context.";
        if (gguf) {
            gguf_free(gguf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        return false;
    }

    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx);
         tensor != nullptr;
         tensor = ggml_get_next_tensor(ctx, tensor)) {
        const int n_dims = ggml_n_dims(tensor);
        if (n_dims < 2) {
            continue;
        }
        if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
            continue;
        }

        ModelMatmulShape shape{};
        shape.name = ggml_get_name(tensor);
        shape.type = tensor->type;
        shape.dim_k = tensor->ne[0];
        shape.dim_m = tensor->ne[1];
        shape.n_dims = n_dims;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            shape.dims[d] = tensor->ne[d];
        }
        int64_t instances = 1;
        for (int d = 2; d < n_dims; ++d) {
            if (tensor->ne[d] <= 0) {
                instances = 0;
                break;
            }
            instances *= tensor->ne[d];
        }
        shape.instances = std::max<int64_t>(instances, 1);
        out.push_back(std::move(shape));
    }

    gguf_free(gguf);
    ggml_free(ctx);

    if (out.empty()) {
        error = "No matrix-shaped tensors found in model.";
        return false;
    }
    return true;
}

}  // namespace sycl_bench
