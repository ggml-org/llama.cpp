#include "ggml.h"
#include <stdio.h>
#include <math.h>

int main() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(params);

    // יצירת טנזורים קטנים לבדיקה
    int n = 2;
    float q_data[4] = {1.0, 2.0, 3.0, 4.0};
    float k_data[4] = {1.0, 0.0, 0.0, 1.0};
    float v_data[4] = {5.0, 6.0, 7.0, 8.0};

    struct ggml_tensor *Q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    struct ggml_tensor *K = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    struct ggml_tensor *V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);

    memcpy(Q->data, q_data, sizeof(q_data));
    memcpy(K->data, k_data, sizeof(k_data));
    memcpy(V->data, v_data, sizeof(v_data));

    printf("Running ggml_sparsek_attn CPU test...\n");
    struct ggml_tensor *Y = ggml_sparsek_attn(ctx, Q, K, V, 1, 0, 0);

    ggml_build_forward_expand(NULL, Y);
    ggml_graph_compute_with_ctx(ctx, NULL, 1);

    printf("Output tensor:\n");
    for (int i = 0; i < n*n; ++i)
        printf("%.6f ", ((float*)Y->data)[i]);
    printf("\n");

    ggml_free(ctx);
    return 0;
}
