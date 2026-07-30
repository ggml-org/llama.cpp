#include "ggml.h"

#include <cassert>
#include <cstddef>

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/16 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * activations     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2048, 128, 2, 1);
    ggml_tensor * weight_anchor   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 2048);
    ggml_tensor * stats           = nullptr;
    ggml_tensor * activation_view = ggml_imatrix_observer_cast(ctx, activations, weight_anchor, &stats);
    ggml_tensor * storage         = activation_view->view_src;

    assert(storage != nullptr);
    assert(storage == stats->view_src);
    assert(storage->op == GGML_OP_IMATRIX_OBSERVER);
    assert(storage->type == GGML_TYPE_F16);
    assert(activation_view->op == GGML_OP_VIEW);
    assert(activation_view->type == GGML_TYPE_F16);
    assert(stats->op == GGML_OP_VIEW);
    assert(stats->type == GGML_TYPE_F32);

    assert(!ggml_imatrix_observer_is_stats(storage));
    assert(!ggml_imatrix_observer_is_stats(activation_view));
    assert(ggml_imatrix_observer_is_stats(stats));

    const size_t original_offset = stats->view_offs;
    stats->view_offs             = 0;
    assert(!ggml_imatrix_observer_is_stats(stats));
    stats->view_offs = original_offset;

    const int64_t original_elements = stats->ne[0];
    stats->ne[0]                    = original_elements - 1;
    assert(!ggml_imatrix_observer_is_stats(stats));
    stats->ne[0] = original_elements;
    assert(ggml_imatrix_observer_is_stats(stats));

    ggml_free(ctx);
    return 0;
}
