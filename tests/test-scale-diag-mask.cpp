#include "ggml/ggml.h"
#include "common/common.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

bool test_scale_diag_mask_inf_softmax() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    // initialize context
    struct ggml_context * ctx = ggml_init(params);

    // create test tensor (2x2 matrix)
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    
    // fill with test values
    float * data = (float *) x->data;
    data[0] = 1.0f;  // [1.0  2.0]
    data[1] = 2.0f;  // [3.0  4.0]
    data[2] = 3.0f;
    data[3] = 4.0f;

    // apply operation
    float scale = 2.0f;
    int n_past = 0;
    struct ggml_tensor * y = ggml_scale_diag_mask_inf_softmax_inplace(ctx, scale, n_past, x);

    // compute
    struct ggml_cgraph gf = ggml_build_forward(y);
    ggml_graph_compute(ctx, &gf);

    // check results
    float * result = (float *) y->data;
    
    // Expected values after scale=2.0, masking, and softmax:
    // For first row: [exp(2)/sum, exp(4)/sum] where sum = exp(2) + exp(4)
    // For second row: [exp(6)/sum, exp(8)/sum] where sum = exp(6) + exp(8)
    const float eps = 1e-5f;
    bool success = true;
    
    float sum1 = expf(2.0f) + expf(4.0f);
    float sum2 = expf(6.0f) + expf(8.0f);
    
    success &= fabsf(result[0] - expf(2.0f)/sum1) < eps;
    success &= fabsf(result[1] - expf(4.0f)/sum1) < eps;
    success &= fabsf(result[2] - expf(6.0f)/sum2) < eps;
    success &= fabsf(result[3] - expf(8.0f)/sum2) < eps;

    // cleanup
    ggml_free(ctx);

    return success;
}

int main(int argc, char ** argv) {
    bool success = test_scale_diag_mask_inf_softmax();
    printf("%s: %s\n", __func__, success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}
