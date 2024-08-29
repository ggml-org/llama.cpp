#include "ggml.h"

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

#define GGML_SILU_FP16

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

static int irand(int n) {
    if (n == 0) return 0;
    return rand()%n;
}

static void get_random_dims(int64_t * dims, int ndims) {
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = 1 + irand(4);
    }
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

int main(int /*argc*/, const char ** /*argv*/) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 128*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<uint8_t> work_buffer;

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * x;

    // rope f32
    for (int m = 0; m < 3; ++m) {
        const int ndims = 4;

        const int64_t n_rot = 128;
        const int64_t ne[4] = { 2*n_rot, 32, 73, 1 };

        const int n_past_0 = 100;
        const int n_past_2 = 33;

        struct ggml_tensor * p0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);
        struct ggml_tensor * p1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);
        struct ggml_tensor * p2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);

        for (int i = 0; i < ne[2]; ++i) {
            ((int32_t *) p0->data)[i] = n_past_0 + i;
            ((int32_t *) p1->data)[i] = n_past_2 - n_past_0;
            ((int32_t *) p2->data)[i] = n_past_2 + i;
        }

        // test mode 0, 2, 4 (standard, GPT-NeoX, GLM)
        const int mode = m == 0 ? 0 : m == 1 ? 2 : 4;

        x = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);

        // 100, 101, 102, ..., 172
        struct ggml_tensor * r0 = ggml_rope(ctx0, x,  p0, n_rot, mode);
        // -67, -67, -67, ..., -67
        struct ggml_tensor * r1 = ggml_rope(ctx0, r0, p1, n_rot, mode); // "context swap", i.e. forget n_past_0 - n_past_2 tokens

        //  33,  34,  35, ..., 105
        struct ggml_tensor * r2 = ggml_rope(ctx0, x,  p2, n_rot, mode);

        ggml_cgraph * gf = ggml_new_graph(ctx0);

        ggml_build_forward_expand(gf, r0);
        ggml_build_forward_expand(gf, r1);
        ggml_build_forward_expand(gf, r2);

        ggml_graph_compute_helper(work_buffer, gf, 4);

        // check that r1 and r2 are the same
        {
            double sum0 = 0.0f;
            double sum1 = 0.0f;
            double diff = 0.0f;

            const float * r1_data = (float *) r1->data;
            const float * r2_data = (float *) r2->data;

            const int n_elements = ggml_nelements(r1);

            for (int i = 0; i < n_elements; ++i) {
                sum0 += fabs(r1_data[i]);
                sum1 += fabs(r2_data[i]);
                diff += fabs(r1_data[i] - r2_data[i]);
                //if (fabs(r1_data[i] - r2_data[i]) > 0.0001f) {
                //    printf("%d: %f %f\n", i, r1_data[i], r2_data[i]);
                //    printf("diff: %f\n", fabs(r1_data[i] - r2_data[i]));
                //}
            }

            //for (int i = 4096; i < 4096 + 128; ++i) {
            //    printf("%f %f\n", r1_data[i], r2_data[i]);
            //}

            printf("mode: %d\n", mode);
            printf("sum0: %f\n", sum0);
            printf("sum1: %f\n", sum1);
            printf("diff: %f\n", diff);
            printf("rel err: %f\n", diff / sum0);
            printf("rel err: %f\n", diff / sum1);

            GGML_ASSERT(diff / sum0 < 0.0001f);
            GGML_ASSERT(diff / sum1 < 0.0001f);
        }
    }

    ggml_free(ctx0);

    return 0;
}
