#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#define MAX_NARGS 3

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static struct ggml_tensor * get_random_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return result;
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

// helper to print a tensor
static void print_tensor(const struct ggml_tensor * tensor, const char * name) {
    printf("%s: shape(%lld, %lld, %lld, %lld), type %s\n",
        name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], ggml_type_name(tensor->type));
    const float * data = (const float *)tensor->data;
    // Print first few elements for brevity
    int n_to_print = MIN(10, ggml_nelements(tensor));
    for (int i = 0; i < n_to_print; ++i) {
        printf("%.4f ", data[i]);
    }
    if (ggml_nelements(tensor) > n_to_print) {
        printf("...");
    }
    printf("\n");
}


int main(int /*argc*/, const char ** /*argv*/) {
    srand(0); // for reproducibility

    struct ggml_init_params params = {
        /* .mem_size   = */ 16*1024*1024, // 16 MB
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<uint8_t> work_buffer;

    struct ggml_context * ctx0 = ggml_init(params);

    // Define matrix A and vector x
    // A: shape (rows_A, cols_A)
    // x: shape (cols_A, 1) to be treated as a vector by ggml_mul_mat
    // Result y = A*x will have shape (rows_A, 1)

    const int64_t rows_A = 3;
    const int64_t cols_A = 4;

    const int64_t ne_A[2] = {cols_A, rows_A}; // GGML tensors are typically row-major in memory but dimensions are (cols, rows)
    const int64_t ne_x[2] = {cols_A, 1};    // Vector x (effectively a column vector for mul_mat)

    struct ggml_tensor * a = get_random_tensor_f32(ctx0, 2, ne_A, -1.0f, 1.0f);
    struct ggml_tensor * x = get_random_tensor_f32(ctx0, 2, ne_x, -1.0f, 1.0f);

    // ggml_mul_mat expects the second tensor (x) to be contiguous.
    // If x was created differently, a ggml_cont(ctx0, x) might be needed.
    // Our get_random_tensor_f32 creates contiguous tensors.

    // Compute y = A*x
    struct ggml_tensor * y = ggml_mul_mat(ctx0, a, x);

    // Build and compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_helper(work_buffer, gf, 1); // Using 1 thread for simplicity

    // Print tensors for verification
    print_tensor(a, "Matrix A");
    print_tensor(x, "Vector x");
    print_tensor(y, "Result y = A*x");

    // Manual check for a small example (optional)
    // For A = [[a11, a12, a13, a14],
    //          [a21, a22, a23, a24],
    //          [a31, a32, a33, a34]]
    // and x = [[x1], [x2], [x3], [x4]]
    // y1 = a11*x1 + a12*x2 + a13*x3 + a14*x4
    // y2 = a21*x1 + a22*x2 + a23*x3 + a24*x4
    // y3 = a31*x1 + a32*x2 + a33*x3 + a34*x4

    const float * a_data = (const float *)a->data;
    const float * x_data = (const float *)x->data;
    const float * y_data = (const float *)y->data;

    printf("Manual verification of first element of y:\n");
    float y0_manual = 0.0f;
    for (int i = 0; i < cols_A; ++i) {
        y0_manual += a_data[i] * x_data[i]; // a_data[0*cols_A + i] for first row of A
    }
    printf("y_data[0] = %.4f, y0_manual = %.4f\n", y_data[0], y0_manual);
    GGML_ASSERT(fabs(y_data[0] - y0_manual) < 1e-5);


    printf("Manual verification of second element of y (if rows_A > 1):\n");
    if (rows_A > 1) {
        float y1_manual = 0.0f;
        for (int i = 0; i < cols_A; ++i) {
            y1_manual += a_data[cols_A + i] * x_data[i]; // a_data[1*cols_A + i] for second row of A
        }
        printf("y_data[1] = %.4f, y1_manual = %.4f\n", y_data[1], y1_manual);
        GGML_ASSERT(fabs(y_data[1] - y1_manual) < 1e-5);
    }


    printf("Test ggml_mul_mat completed successfully.\n");

    ggml_free(ctx0);

    return 0;
} 