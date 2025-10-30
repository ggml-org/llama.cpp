#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-axelera.h"
#include <cstdio>
#include <cstdlib>

int main() {
    printf("=== Testing Axelera Backend Graph Computation ===\n\n");

    // Initialize GGML context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,  // Let GGML allocate memory
    };

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }

    // Create simple tensors for matrix multiplication: C = A * B
    // A: [4, 3] - 3x4 matrix
    // B: [4, 2] - 2x4 matrix
    // C: [3, 2] - Result 3x2 matrix

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);

    ggml_set_name(a, "matrix_A");
    ggml_set_name(b, "matrix_B");

    // Initialize data
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    for (int i = 0; i < 12; i++) a_data[i] = (float)(i + 1);
    for (int i = 0; i < 8; i++) b_data[i] = (float)(i + 1) * 0.1f;

    // Create multiplication operation
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_set_name(c, "result_C");

    // Build computation graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    printf("Created computation graph\n");
    printf("Input A shape: [%lld, %lld]\n", a->ne[0], a->ne[1]);
    printf("Input B shape: [%lld, %lld]\n", b->ne[0], b->ne[1]);
    printf("Result C shape: [%lld, %lld]\n\n", c->ne[0], c->ne[1]);

    // Initialize Axelera backend
    printf("Initializing Axelera backend...\n");
    ggml_backend_t backend = ggml_backend_axelera_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize Axelera backend\n");
        ggml_free(ctx);
        return 1;
    }
    printf("Backend initialized: %s\n\n", ggml_backend_name(backend));

    // Compute the graph - this will trigger our debug output
    printf("Computing graph on Axelera backend...\n");
    printf("==========================================\n");
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    printf("==========================================\n");
    printf("Compute status: %s\n\n", status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

    if (status == GGML_STATUS_FAILED) {
        printf("Note: Backend returned FAILED (expected for debugging)\n");
        printf("The graph info was printed above.\n");
    }

    // Cleanup
    ggml_backend_free(backend);
    ggml_free(ctx);

    printf("\nTest completed\n");
    return 0;
}
