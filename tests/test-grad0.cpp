#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnings on Windows
#include "ggml.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <initializer_list>
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
        int64_t ne[],
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
    }

    return result;
}

static struct ggml_tensor * get_random_tensor_f16(
        struct ggml_context * ctx0,
        int ndims,
        int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F16, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((ggml_fp16_t *)result->data)[i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((ggml_fp16_t *)result->data)[i1*ne[0] + i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((ggml_fp16_t *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((ggml_fp16_t *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return result;
}

static struct ggml_tensor * get_random_tensor_i32(
        struct ggml_context * ctx0,
        int ndims,
        int64_t ne[],
        int32_t imin,
        int32_t imax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_I32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((int32_t *)result->data)[i0] = irand(imax - imin) + imin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((int32_t *)result->data)[i1*ne[0] + i0] = irand(imax - imin) + imin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((int32_t *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = irand(imax - imin) + imin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((int32_t *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = irand(imax - imin) + imin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return result;
}

static bool check_gradient(
        const char * op_name,
        struct ggml_context * ctx0,
        struct ggml_tensor * x[],
        struct ggml_tensor * f,
        int ndims,
        int nargs,
        float eps,
        float max_error_abs,
        float max_error_rel,
        std::vector<double> expected_vals) {

    static int n_threads = -1;
    if (n_threads < 0) {
        n_threads = GGML_DEFAULT_N_THREADS;

        const char *env = getenv("GGML_N_THREADS");
        if (env) {
            n_threads = atoi(env);
        }

        printf("GGML_N_THREADS = %d\n", n_threads);
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
    struct ggml_cgraph * gb = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, f);
    ggml_graph_cpy(gf, gb);
    ggml_build_backward_expand(ctx0, gf, gb, false);

    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    ggml_graph_reset(gb);
    if (f->grad) {
        ggml_set_f32(f->grad, 1.0f);
    }

    ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

    // ggml_graph_dump_dot(gf, NULL, "test-grad0-forward.dot");
    // ggml_graph_dump_dot(gb, gf,  "test-grad0-backward.dot");

    for (int i = 0; i < nargs; ++i) {
        bool all_g0_bad = true;
        const int nelements = ggml_nelements(x[i]);
        for (int k = 0; k < nelements; ++k) {
            // Calculate gradient numerically:
            const float x0 = ggml_get_f32_1d(x[i], k);
            const float xm = x0 - eps;
            const float xp = x0 + eps;
            ggml_set_f32_1d(x[i], k, xp);

            ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

            const double f0 = ggml_get_f32_1d(f, 0);

            ggml_set_f32_1d(x[i], k, xm);

            ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

            const double f1 = ggml_get_f32_1d(f, 0);
            const double g0 = (f0 - f1)/(2.0*(double) eps);

            // The numerical calculation of the gradient fails around noncontinuities (e.g. 0 for ReLU).
            // In such cases, provide a vector of expected values and skip the comparison for failed calculations.
            if (!expected_vals.empty()) {
                bool matches_any = false;
                for (const double & ev : expected_vals) {
                    const double error_abs = std::fabs(g0 - ev);
                    if (error_abs > max_error_abs) {
                        continue;
                    }
                    const double error_rel = g0 != 0.0 ? fabs(g0 - ev)/fabs(g0) : 0.0;
                    if (error_rel > max_error_rel) {
                        continue;
                    }
                    matches_any = true;
                    break;
                }
                if (!matches_any) {
                    continue;
                }
            }
            all_g0_bad = false;

            ggml_set_f32_1d(x[i], k, x0);

            // compute gradient using backward graph
            ggml_graph_reset(gb);
            if (f->grad) {
                ggml_set_f32(f->grad, 1.0f);
            }

            ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

            const double g1 = ggml_get_f32_1d(x[i]->grad, k);

            const double error_abs = fabs(g0 - g1);
            const double error_rel = g0 != 0.0 ? fabs(g0 - g1)/fabs(g0) : 0.0;

            if (error_abs > max_error_abs || error_rel > max_error_rel) {
                printf("%s: ndims=%d, i=%d, k=%d, x0=%f, xm=%f, xp=%f, f0=%f, f1=%f, g0=%f, g1=%f, eps=%f, error_abs=%f, error_rel=%f\n",
                            op_name, ndims, i, k, x0, xm, xp, f0, f1, g0, g1, eps, error_abs, error_rel);
                //assert(false);
                return false;
            }
        }
        if (all_g0_bad) {
            printf("%s: numerical calculation of the gradient failed for all values\n", op_name);
            return false;
        }
    }

    return true;
}

// TODO: clean-up this ..
static bool check_mat_mul(
        const struct ggml_tensor * y,
        const struct ggml_tensor * x0,
        const struct ggml_tensor * x1) {
    float * dst  = (float *) y->data;
    float * src0 = (float *) x0->data;
    float * src1 = (float *) x1->data;

    const int nc = x0->ne[1];
    const int nr = x1->ne[1];
    const int nk = x0->ne[0];

    GGML_PRINT_DEBUG("check_mat_mul: nc=%d, nr=%d, nk=%d\n", nc, nr, nk);

    GGML_PRINT_DEBUG("x0:\n");
    for (int j = 0; j < x0->ne[1]; ++j) {
        for (int i = 0; i < x0->ne[0]; ++i) {
            GGML_PRINT_DEBUG("%6.3f ", src0[j*nk + i]);
        }
        GGML_PRINT_DEBUG("\n");
    }
    GGML_PRINT_DEBUG("\n");

    GGML_PRINT_DEBUG("x1:\n");
    for (int j = 0; j < x1->ne[1]; ++j) {
        for (int i = 0; i < x1->ne[0]; ++i) {
            GGML_PRINT_DEBUG("%6.3f ", src1[j*nk + i]);
        }
        GGML_PRINT_DEBUG("\n");
    }
    GGML_PRINT_DEBUG("\n");

    GGML_PRINT_DEBUG("y: n_dims = %d, (%lld, %lld)\n", y->n_dims, y->ne[0], y->ne[1]);
    for (int j = 0; j < y->ne[1]; ++j) {
        for (int i = 0; i < y->ne[0]; ++i) {
            GGML_PRINT_DEBUG("%6.3f ", dst[j*nr + i]);
        }
        GGML_PRINT_DEBUG("\n");
    }

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nc; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < nk; ++k) {
                sum += src0[j*nk + k]*src1[i*nk + k];
            }

            if (fabsf(dst[i*nc + j] - sum) > 1e-5f) {
                fprintf(stderr, "check_mat_mul: dst[%d] = %f, sum = %f\n", i*nc + j, dst[i*nc + j], sum);
                assert(false);
                return false;
            }
        }
    }

    return true;
}

#define NUM_PERMUTATIONS (4*3*2*1)

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 256*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    int64_t ne[4];

    int all_permutations[4 * NUM_PERMUTATIONS];
    {
        int count = 0;
        for (int ax0=0; ax0<4; ++ax0) {
            for (int ax1=0; ax1<4; ++ax1) {
                if (ax1 == ax0) continue;
                for (int ax2=0; ax2<4; ++ax2) {
                    if (ax2 == ax0) continue;
                    if (ax2 == ax1) continue;
                    for (int ax3=0; ax3<4; ++ax3) {
                        if (ax3 == ax0) continue;
                        if (ax3 == ax1) continue;
                        if (ax3 == ax2) continue;
                        assert(count < NUM_PERMUTATIONS);
                        all_permutations[count*4+0] = ax0;
                        all_permutations[count*4+1] = ax1;
                        all_permutations[count*4+2] = ax2;
                        all_permutations[count*4+3] = ax3;
                        ++count;
                    }
                }
            }
        }
    }

    unsigned seed_iter = 1;

    // original loop: 1000
    int niter = 4;
    const char *env = getenv("GGML_NLOOP");
    if (env != NULL) {
        niter = atoi(env);
    }
    if (argc > 1) {
        niter = atoi(argv[1]);
    }
    for (int iter = 0; iter < niter; ++iter) {
        srand(seed_iter);
        seed_iter = rand();
        unsigned seed = rand();

        printf("test-grad0: iter:%d/%d\n", (iter+1), niter);
        struct ggml_context * ctx0 = ggml_init(params);

        get_random_dims(ne, 4);

        struct ggml_tensor * x[MAX_NARGS];

        // add f32
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_add(ctx0, x[0], x[1]));

                check_gradient("add f32", ctx0, x, f, ndims, nargs, 1e-3f, 2e-3f, 2e-3f, {});
            }
        }

        // add f16
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f16(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_add(ctx0, x[0], x[1]));

                check_gradient("add f16", ctx0, x, f, ndims, nargs, 1e-1f, 2e-1f, 2e-1f, {});
            }
        }

        // sub
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sub(ctx0, x[0], x[1]));

                check_gradient("sub", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // mul
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_mul(ctx0, x[0], x[1]));

                check_gradient("mul", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // div
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, 0.5f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_div(ctx0, x[0], x[1]));

                check_gradient("div", ctx0, x, f, ndims, nargs, 1e-3f, 1e-1f, 1e-1f, {});
            }
        }

        // sqr
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqr(ctx0, x[0]));

                check_gradient("sqr", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // sqrt
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, 2.0f*1e-3f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqrt(ctx0, x[0]));

                check_gradient("sqrt", ctx0, x, f, ndims, nargs, 1e-3f, 2e-2f, 1e-1f, {});
            }
        }

        // log
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, 2.0f*1e-3f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_log(ctx0, x[0]));

                check_gradient("log", ctx0, x, f, ndims, nargs, 1e-3f, INFINITY, 1e-1f, {});
            }
        }

        // sum
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, x[0]);

                check_gradient("sum", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }


        // sum_rows
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqr(ctx0, ggml_sum_rows(ctx0, x[0])));

                check_gradient("sum_rows", ctx0, x, f, ndims, nargs, 1e-3f, 1e-2f, INFINITY, {});
            }
        }

        // mean, not yet fully implemented
        if(0)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_mean(ctx0, x[0]));

                check_gradient("mean", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // argmax
        if (0)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_argmax(ctx0, x[0]));

                check_gradient("argmax", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // repeat
        {
            srand(seed);
            int64_t ne2[4];
            get_random_dims(ne2, 4);

            ne2[0] = ne[0] * ne2[0];
            ne2[1] = ne[1] * ne2[1];
            ne2[2] = 1;
            ne2[3] = 1;

            const int nargs = 1;
            for (int ndims = 1; ndims <= 2; ++ndims) {
                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                x[1] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqr(ctx0, ggml_sub(ctx0, x[1], ggml_repeat(ctx0, x[0], x[1]))));

                check_gradient("repeat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-2f, INFINITY, {});
            }
        }

        // repeat back
        {
            srand(seed);
            int64_t ne2[4];
            get_random_dims(ne2, 4);

            ne2[0] = ne[0] * ne2[0];
            ne2[1] = ne[1] * ne2[1];
            ne2[2] = 1;
            ne2[3] = 1;

            const int nargs = 1;
            for (int ndims = 1; ndims <= 2; ++ndims) {
                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                x[1] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqr(ctx0, ggml_sub(ctx0, x[0], ggml_repeat_back(ctx0, x[1], x[0]))));

                check_gradient("repeat back", ctx0, x, f, ndims, nargs, 1e-3f, 1e-2f, INFINITY, {});
            }
        }

        // abs
        {
           const int nargs = 1;

           for (int ndims = 1; ndims <= 4; ++ndims) {
               for (int i = 0; i < nargs; ++i) {
                   x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                   ggml_set_param(ctx0, x[i]);
               }

               struct ggml_tensor * f = ggml_sum(ctx0, ggml_abs(ctx0, x[0]));

               check_gradient("abs", ctx0, x, f, ndims, nargs, 1e-3f, INFINITY, 1e-3f, {-1.0, 1.0});
           }
        }

        // sgn
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_sgn(ctx0, x[0]));

                check_gradient("sgn", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {0.0});
            }
        }

        // neg
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_neg(ctx0, x[0]));

                check_gradient("neg", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // step
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_step(ctx0, x[0]));

                check_gradient("step", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {0.0});
            }
        }

        // tanh, not yet fully implemented
        if(0)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_tanh(ctx0, x[0]));

                check_gradient("tanh", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // mul_mat
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                int max_nrep = (ndims >= 3) ? 2 : 1;
                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                for (int nrep2 = 1; nrep2 < max_nrep; ++nrep2) {
                    for (int nrep3 = 1; nrep3 < max_nrep; ++nrep3) {
                        {
                            int64_t ne2[4];
                            get_random_dims(ne2, 4);
                            ne2[0] = ne[0];
                            ne2[2] = nrep2 * ne[2];
                            ne2[3] = nrep3 * ne[3];
                            x[1] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);
                        }

                        ggml_set_param(ctx0, x[0]);
                        ggml_set_param(ctx0, x[1]);

                        struct ggml_tensor * m = ggml_mul_mat(ctx0, x[1], x[0]);
                        struct ggml_tensor * f = ggml_sum(ctx0, m);

                        GGML_PRINT_DEBUG("testing: mul_mat, [%lld, %lld] (%d) * [%lld, %lld] (%d)\n", x[1]->ne[0], x[1]->ne[1], x[1]->n_dims, x[0]->ne[0], x[0]->ne[1], x[0]->n_dims);

                        check_gradient("mul_mat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
                        if (ndims == 2) {
                            // check_mat_mul does not support ndims > 2
                            check_mat_mul(m, x[1], x[0]);
                        }
                    }
                }
            }
        }

        // elu, not yet fully implemented
        if(0)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_elu(ctx0, x[0]));

                check_gradient("elu", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // relu
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_relu(ctx0, x[0]));

                check_gradient("relu", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {0.0, 1.0});
            }
        }

        // gelu, not yet fully implemented
        if(0)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 4; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor* f = ggml_sum(ctx0, ggml_gelu(ctx0, x[0]));

                check_gradient("gelu", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f, {});
            }
        }

        // silu
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_silu(ctx0, x[0]));

#ifdef GGML_SILU_FP16
                // due to GGML_SILU_FP16 the finite difference method will be slightly wrong -> increase error bounds.
                check_gradient("silu", ctx0, x, f, ndims, nargs, 1e-3f, 0.5, INFINITY, {});
#else
                check_gradient("silu", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
#endif
            }
        }

        // rms_norm
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_rms_norm(ctx0, x[0], 1e-6f));

                check_gradient("rms_norm", ctx0, x, f, ndims, nargs, 1e-4f, 1.0f, INFINITY, {});
            }
        }

        // scale
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);

                const float s = -1.0f + 2.0f*frand();

                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_scale(ctx0, x[0], s));

                check_gradient("scale", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // cpy f32
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }
                // x[1] is overwritten by x[0], so the gradients don't propagate to x[1]

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_cpy(ctx0, x[0], x[1]));

                check_gradient("cpy f32", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // cpy f16
        {
            srand(seed);
            const int nargs = 2;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor_f16(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }
                // x[1] is overwritten by x[0], so the gradients don't propagate to x[1]

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_cpy(ctx0, x[0], x[1]));

                check_gradient("cpy f16", ctx0, x, f, ndims, nargs, 1e-1f, 1e-1f, INFINITY, {});
            }
        }

        // reshape (1d->nd)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                int64_t ne2[4];
                ne2[0] = 1;
                ne2[1] = 1;
                ne2[2] = 1;
                ne2[3] = 1;
                for (int i = 0; i < ndims; ++i) {
                    ne2[0] *= ne[i];
                }
                x[0] = get_random_tensor_f32(ctx0, 1, ne2, -1.0f, 1.0f);
                x[1] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);


                struct ggml_tensor * f = ggml_sum(ctx0, ggml_reshape(ctx0, x[0], x[1]));
                check_gradient("reshape", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // reshape (nd->1d)
        {
            srand(seed);
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                int64_t ne2[4];
                ne2[0] = 1;
                ne2[1] = 1;
                ne2[2] = 1;
                ne2[3] = 1;
                for (int i = 0; i < ndims; ++i) {
                    ne2[0] *= ne[i];
                }
                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                x[1] = get_random_tensor_f32(ctx0, 1, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);


                struct ggml_tensor * f = ggml_sum(ctx0, ggml_reshape(ctx0, x[0], x[1]));
                check_gradient("reshape", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // acc 1d
        {
            srand(seed);
            int64_t ne2[4] = { 1, 1, 1, 1 };

            const int nargs = 2;
            for (int ndims = 1; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                get_random_dims(ne2, 1);
                while ((ne2[0] > ne[0]) || (ne2[0] > ggml_nelements(x[0]))) {
                    get_random_dims(ne2, 1);
                }

                x[1] = get_random_tensor_f32(ctx0, 1, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[1]);

                const int max_offset = MAX(0, ggml_nelements(x[0]) - ggml_nelements(x[1]));
                const int offset = irand(max_offset) * ggml_element_size(x[0]);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_acc(ctx0, x[0], x[1], x[0]->nb[1], x[0]->nb[2], x[0]->nb[3], offset));

                check_gradient("acc 1d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // acc 2d
        {
            srand(seed);
            int64_t ne2[4]         = { 1, 1, 1, 1 };
            int64_t max_offsets[4] = { 0, 0, 0, 0 };
            int64_t offsets[4]     = { 0, 0, 0, 0 };

            const int nargs = 2;
            for (int ndims = 2; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                get_random_dims(ne2, 2);
                while ((ne2[0] > ne[0]) || (ne2[1] > ne[1]) || (ne2[0]*ne2[1] > ggml_nelements(x[0]))) {
                    get_random_dims(ne2, 2);
                }

                x[1] = get_random_tensor_f32(ctx0, 2, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[1]);

                max_offsets[0] = MAX(0, x[0]->ne[0] - x[1]->ne[0]);
                max_offsets[1] = MAX(0, x[0]->ne[1] - x[1]->ne[1]);
                offsets[0] = irand(max_offsets[0]) * x[0]->nb[0];
                offsets[1] = irand(max_offsets[1]) * x[0]->nb[1];
                const int offset = offsets[0] + offsets[1];

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_acc(ctx0, x[0], x[1], x[0]->nb[1], x[0]->nb[2], x[0]->nb[3], offset));

                check_gradient("acc 2d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // acc 3d
        {
            srand(seed);
            int64_t ne2[4]         = { 1, 1, 1, 1 };
            int64_t max_offsets[4] = { 0, 0, 0, 0 };
            int64_t offsets[4]     = { 0, 0, 0, 0 };

            const int nargs = 2;
            for (int ndims = 3; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                get_random_dims(ne2, 3);
                while ((ne2[0] > ne[0]) || (ne2[1] > ne[1]) || (ne2[2] > ne[2]) || (ne2[0]*ne2[1]*ne2[2] > ggml_nelements(x[0]))) {
                    get_random_dims(ne2, 3);
                }

                x[1] = get_random_tensor_f32(ctx0, 3, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[1]);

                max_offsets[0] = MAX(0, x[0]->ne[0] - x[1]->ne[0]);
                max_offsets[1] = MAX(0, x[0]->ne[1] - x[1]->ne[1]);
                max_offsets[2] = MAX(0, x[0]->ne[2] - x[1]->ne[2]);
                offsets[0] = irand(max_offsets[0]) * x[0]->nb[0];
                offsets[1] = irand(max_offsets[1]) * x[0]->nb[1];
                offsets[2] = irand(max_offsets[2]) * x[0]->nb[2];
                const int offset = offsets[0] + offsets[1] + offsets[2];

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_acc(ctx0, x[0], x[1], x[0]->nb[1], x[0]->nb[2], x[0]->nb[3], offset));

                check_gradient("acc 3d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // acc 4d
        {
            srand(seed);
            int64_t ne2[4]         = { 1, 1, 1, 1 };
            int64_t max_offsets[4] = { 0, 0, 0, 0 };
            int64_t offsets[4]     = { 0, 0, 0, 0 };

            const int nargs = 2;
            for (int ndims = 4; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                get_random_dims(ne2, 4);
                while ((ne2[0] > ne[0]) || (ne2[1] > ne[1]) || (ne2[2] > ne[2]) || (ne2[3] > ne[3]) || (ne2[0]*ne2[1]*ne2[2]*ne2[3] > ggml_nelements(x[0]))) {
                    get_random_dims(ne2, 4);
                }

                x[1] = get_random_tensor_f32(ctx0, 4, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[1]);

                max_offsets[0] = MAX(0, x[0]->ne[0] - x[1]->ne[0]);
                max_offsets[1] = MAX(0, x[0]->ne[1] - x[1]->ne[1]);
                max_offsets[2] = MAX(0, x[0]->ne[2] - x[1]->ne[2]);
                max_offsets[3] = MAX(0, x[0]->ne[3] - x[1]->ne[3]);
                offsets[0] = irand(max_offsets[0]) * x[0]->nb[0];
                offsets[1] = irand(max_offsets[1]) * x[0]->nb[1];
                offsets[2] = irand(max_offsets[2]) * x[0]->nb[2];
                offsets[3] = irand(max_offsets[3]) * x[0]->nb[3];
                const int offset = offsets[0] + offsets[1] + offsets[2] + offsets[3];

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_acc(ctx0, x[0], x[1], x[0]->nb[1], x[0]->nb[2], x[0]->nb[3], offset));

                check_gradient("acc 4d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // set_1d
        {
            srand(seed);
            int64_t ne2[4];

            const int nargs = 2;
            for (int ndims = 1; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                get_random_dims(ne2, 1);
                while ((ne2[0] > ne[0]) || (ne2[0] > ggml_nelements(x[0]))) {
                    get_random_dims(ne2, 1);
                }

                x[1] = get_random_tensor_f32(ctx0, 1, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[1]);

                const int max_offset = MAX(0, ggml_nelements(x[0]) - ggml_nelements(x[1]));
                const int offset = irand(max_offset) * ggml_element_size(x[0]);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_set_1d(ctx0, x[0], x[1], offset));

                check_gradient("set_1d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // set_2d
        {
            srand(seed);
            int64_t ne2[4];
            int64_t max_offsets[4] = { 0, 0, 0, 0 };
            int64_t offsets[4]     = { 0, 0, 0, 0 };

            const int nargs = 1;
            for (int ndims = 2; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                get_random_dims(ne2, 2);
                while ((ne2[0] > ne[0]) || (ne2[1] > ne[1]) || (ne2[0]*ne2[1] > ggml_nelements(x[0]))) {
                    get_random_dims(ne2, 2);
                }

                x[1] = get_random_tensor_f32(ctx0, 2, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[1]);

                max_offsets[0] = MAX(0, x[0]->ne[0] - x[1]->ne[0]);
                max_offsets[1] = MAX(0, x[0]->ne[1] - x[1]->ne[1]);
                offsets[0] = irand(max_offsets[0]) * x[0]->nb[0];
                offsets[1] = irand(max_offsets[1]) * x[0]->nb[1];
                const int offset = offsets[0] + offsets[1];

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_set_2d(ctx0, x[0], x[1], x[1]->nb[1], offset));

                check_gradient("set_2d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // view_1d
        {
            srand(seed);
            const int nargs = 1;
            for (int ndims = 1; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                const int k0 = irand(ggml_nelements(x[0]));
                const int k1 = irand(ggml_nelements(x[0]));
                const int i0 = MIN(k0, k1);
                const int i1 = MAX(k0, k1);

                const int offset = i0 * sizeof(float);
                const int nelem  = i1 - i0;

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_view_1d(ctx0, x[0], nelem, offset));

                check_gradient("view_1d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // view_2d
        {
            srand(seed);
            int64_t ne2[4];
            int64_t nb2[4];

            const int nargs = 1;
            for (int ndims = 1; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);

                get_random_dims(ne2, 2);
                while (ne2[0]*ne2[1] > ggml_nelements(x[0])) {
                    get_random_dims(ne2, 2);
                }
                const int count = ne2[0]*ne2[1];

                nb2[0] = sizeof(float);
                nb2[1] = nb2[0]*ne2[0];

                ggml_set_param(ctx0, x[0]);

                const int max_offset = ggml_nelements(x[0]) - count;
                const int offset = irand(max_offset+1) * sizeof(float);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_view_2d(ctx0, x[0], ne2[0], ne2[1], nb2[1], offset));

                check_gradient("view_2d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // view_3d
        {
            srand(seed);
            int64_t ne2[4] = {1,1,1,1};
            int64_t nb2[4] = {0,0,0,0};

            const int nargs = 1;
            for (int ndims = 1; ndims <= 4; ++ndims) {

                x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);

                get_random_dims(ne2, 3);
                while (ne2[0]*ne2[1]*ne2[2] > ggml_nelements(x[0])) {
                    get_random_dims(ne2, 3);
                }
                const int count = ne2[0]*ne2[1]*ne2[2];

                nb2[0] = sizeof(float);
                nb2[1] = nb2[0]*ne2[0];
                nb2[2] = nb2[1]*ne2[1];

                ggml_set_param(ctx0, x[0]);

                const int max_offset = ggml_nelements(x[0]) - count;
                const int offset = irand(max_offset+1) * sizeof(float);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_view_3d(ctx0, x[0], ne2[0], ne2[1], ne2[2], nb2[1], nb2[2], offset));

                check_gradient("view_3d", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // permute
        {
            srand(seed);
            int64_t ne2[4];

            const int nargs = 1;
            for (int ndims = 1; ndims <= 4; ++ndims)
            {
                // ggml_permute will set axes of dimensions below n_dims to 1.
                // to make ggml_permute work correctly on all axes,
                // the input tensor needs maximal n_dim of 4.
                for (int i=0; i<ndims; ++i) {
                    ne2[i] = ne[i];
                }
                for (int i=ndims; i<4; ++i) {
                    ne2[i] = 1;
                }
                x[0] = get_random_tensor_f32(ctx0, 4, ne2, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                const int p = irand(NUM_PERMUTATIONS);
                const int ax0 = all_permutations[p*4+0];
                const int ax1 = all_permutations[p*4+1];
                const int ax2 = all_permutations[p*4+2];
                const int ax3 = all_permutations[p*4+3];

                // sum requires contiguous tensor rows
                struct ggml_tensor * f = ggml_sum(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, x[0], ax0, ax1, ax2, ax3)));

                check_gradient("permute", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // transpose
        {
            srand(seed);
            int64_t ne2[4];

            const int nargs = 1;
            for (int ndims = 1; ndims <= 4; ++ndims)
            {
                // ggml_transpose will set axes of dimensions below n_dims to 1.
                // to make ggml_transpose work correctly on all axes,
                // the input tensor needs maximal n_dim of 4.
                for (int i=0; i<ndims; ++i) {
                    ne2[i] = ne[i];
                }
                for (int i=ndims; i<4; ++i) {
                    ne2[i] = 1;
                }
                x[0] = get_random_tensor_f32(ctx0, 4, ne2, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                // sum requires contiguous tensor rows
                struct ggml_tensor * f = ggml_sum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, x[0])));

                check_gradient("transpose", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // get_rows
        {
            srand(seed);
            int64_t ne2[4] = {ne[0], ne[1], 1, 1};
            int64_t ne3[4] = {1+irand(ne[1]), 1, 1, 1};
            const int nargs = 1;
            const int ndims = 2;
            x[0] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);
            x[1] = get_random_tensor_i32(ctx0, 1, ne3, 0, ne2[1]);

            ggml_set_param(ctx0, x[0]);

            struct ggml_tensor * f = ggml_sum(ctx0, ggml_get_rows(ctx0, x[0], x[1]));

            check_gradient("get_rows", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
        }

        // diag_mask_inf
        {
            srand(seed);
            const int nargs = 1;
            const int ndims = 2;

            x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
            ggml_set_param(ctx0, x[0]);

            int n_past = irand(ne[0]);

            struct ggml_tensor * f = ggml_sum(ctx0, ggml_diag_mask_inf(ctx0, x[0], n_past));

            check_gradient("diag_mask_inf", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
        }

        // diag_mask_zero
        {
            srand(seed);
            const int nargs = 1;
            const int ndims = 2;

            x[0] = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
            ggml_set_param(ctx0, x[0]);

            int n_past = irand(ne[0]);

            struct ggml_tensor * f = ggml_sum(ctx0, ggml_diag_mask_zero(ctx0, x[0], n_past));

            check_gradient("diag_mask_zero", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
        }

        // softmax
        {
            srand(seed);
            const int nargs = 1;

            int64_t ne2[4];
            get_random_dims(ne2, 4);

            for (int ndims = 1; ndims <= 3; ++ndims) {
                x[0] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);
                ggml_set_param(ctx0, x[0]);

                float eps = 1e-6f;
                // dont use only sum as aggregation, because sum of softmax is always 1 -> finite differences should not work
                // instead use sum(log(soft_max()*(1-eps)+eps)); use eps to avoid log(0)
                struct ggml_tensor * f = ggml_sum(ctx0,
                                            ggml_log(ctx0,
                                                ggml_add1(ctx0,
                                                    ggml_scale(ctx0,
                                                        ggml_soft_max(ctx0, x[0]),
                                                        1.0f - eps),
                                                    ggml_new_f32(ctx0, eps))));

                check_gradient("softmax", ctx0, x, f, ndims, nargs, 1e-3f, 2e-1f, INFINITY, {});
                // NOTE: softmax forward is computed using f16 table lookup instead of using actual expf, but backward assumes actual expf.
                // this may result in different gradients too finite differences.
                // when this test reports errors, first try to replace the table lookup with actual expf and test again to see if just that was the cause.
                // if only the table lookup causes gradients to differ this is acceptable.
            }
        }

        // cross_entropy_loss
        {
            srand(seed);
            const int nargs = 1;

            int64_t ne2[4];
            get_random_dims(ne2, 4);

            for (int ndims = 1; ndims <= 4; ++ndims) {
                x[0] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);
                x[1] = get_random_tensor_f32(ctx0, ndims, ne2, 0.0f, 1.0f);
                // the second argument to cross_entropy_loss must sum up to 1 for each row
                int nr = ggml_nrows(x[1]);
                int nc = ggml_nelements(x[1]) / nr;
                for (int ir = 0; ir < nr; ++ir) {
                    float sum = 0;
                    for (int ic = 0; ic < nc; ++ic) {
                        sum += ((float *) x[1]->data)[ic + ir*nc];
                    }
                    for (int ic = 0; ic < nc; ++ic) {
                        ((float *) x[1]->data)[ic + ir*nc] /= sum;
                    }
                }
                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * f = ggml_cross_entropy_loss(ctx0, x[0], x[1]);

                check_gradient("cross_entropy_loss", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, {});
            }
        }

        // rope f32
        {
            srand(seed);
            const int nargs = 1;

            int64_t ne2[4];
            get_random_dims(ne2, 4);
            ne2[0] += ne2[0] % 2;
            int n_rot = ne2[0];

            for (int ndims = 3; ndims <= 4; ++ndims) {
                for (int mode = 0; mode < 4; ++mode) {
                    for (int n_past = 1; n_past < ne2[2]; ++n_past) {
                        x[0] = get_random_tensor_f32(ctx0, ndims, ne2, -1.0f, 1.0f);

                        struct ggml_tensor * p = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne2[2]);
                        for (int i = 0; i < ne2[2]; ++i) {
                            ((int32_t *) p->data)[i] = n_past + i;
                        }

                        ggml_set_param(ctx0, x[0]);

                        const bool skip_past = (mode & 1);
                        if (skip_past) {
                            // we have no past, so this would have to work on uninitialized memory.
                            // we only test the gradients here;
                            // skip_past should have no influence on gradient computation.
                            // so when other modes work, we assume that this does as well.
                            continue;
                        }

                        struct ggml_tensor * f = ggml_sum(ctx0, ggml_rope(ctx0, x[0], p, n_rot, mode));

                        GGML_PRINT_DEBUG("rope f32: n_past: %d n_rot: %d mode: %d\n", n_past, n_rot, mode);
                        check_gradient("rope f32", ctx0, x, f, ndims, nargs, 1e-2f, 1e-3f, INFINITY, {});
                    }
                }
            }
        }

        // rope f16
        {
            srand(seed);
            const int nargs = 1;

            int64_t ne2[4];
            get_random_dims(ne2, 4);
            ne2[0] += ne2[0] % 2;
            int n_rot = ne2[0];

            for (int ndims = 3; ndims <= 4; ++ndims) {
                for (int mode = 0; mode < 4; ++mode) {
                    for (int n_past = 1; n_past < ne2[2]; ++n_past) {
                        x[0] = get_random_tensor_f16(ctx0, ndims, ne2, -1.0f, 1.0f);

                        struct ggml_tensor * p = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne2[2]);
                        for (int i = 0; i < ne2[2]; ++i) {
                            ((int32_t *) p->data)[i] = n_past + i;
                        }

                        ggml_set_param(ctx0, x[0]);

                        const bool skip_past = (mode & 1);
                        if (skip_past) {
                            // we have no past, so this would have to work on uninitialized memory.
                            // we only test the gradients here;
                            // skip_past should have no influence on gradient computation.
                            // so when other modes work, we assume that this does as well.
                            continue;
                        }

                        struct ggml_tensor * f = ggml_sum(ctx0, ggml_rope(ctx0, x[0], p, n_rot, mode));

                        GGML_PRINT_DEBUG("rope f16: n_past: %d n_rot: %d mode: %d\n", n_past, n_rot, mode);
                        check_gradient("rope f16", ctx0, x, f, ndims, nargs, 1e-1f, 1e-1f, INFINITY, {});
                    }
                }
            }
        }

        // im2col f32
        {
            srand(seed);
            const int nargs = 1;
            const int ndims = 4;

            for (const bool is_2D : {false, true}) {
                int64_t ne0[ndims];
                int64_t ne1[ndims];
                get_random_dims(ne0, ndims);
                get_random_dims(ne1, ndims);

                // // Ensure that the output is not zero-sized:
                ne1[0] += 8;
                ne1[1] += 8;

                if (is_2D) {
                    ne1[2] = ne0[2];
                } else {
                    ne1[1] = ne0[1];
                    ne0[3] = 1;
                    ne1[3] = 1;
                }

                // The order of arguments is swapped because the first tensor is only used for its shape.
                x[1] = get_random_tensor_f16(ctx0, ndims, ne0, -1.0f, 1.0f);
                x[0] = get_random_tensor_f32(ctx0, ndims, ne1, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                const int s0 =         1 + irand(2);
                const int s1 = is_2D ? 1 + irand(2) : 0;
                const int p0 =         0 + irand(2);
                const int p1 = is_2D ? 0 + irand(2) : 0;
                const int d0 =         1 + irand(2);
                const int d1 = is_2D ? 1 + irand(2) : 0;

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_im2col(ctx0, x[1], x[0], s0, s1, p0, p1, d0, d1, is_2D, GGML_TYPE_F32));

                GGML_PRINT_DEBUG("im2col f32: is_2D=%s, s0=%d, s1=%d, p0=%d, p1=%d, d0=%d, d1=%d\n", is_2D ? "yes" : "no", s0, s1, p0, p1, d0, d1);
                check_gradient("im2col f32", ctx0, x, f, ndims, nargs, 1e-2f, 1e-3f, INFINITY, {});
            }
        }

        // pool_2d f32
        {
            srand(seed);
            const int nargs = 1;
            const int ndims = 4;

            for (const enum ggml_op_pool op : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
                int64_t ne0[ndims];
                get_random_dims(ne0, ndims);

                ne0[0] += 8;
                ne0[1] += 8;

                x[0] = get_random_tensor_f32(ctx0, ndims, ne0, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                const int k0 = 2 + irand(2);
                const int k1 = 2 + irand(2);
                const int s0 = 2 + irand(2);
                const int s1 = 2 + irand(2);
                const int p0 = 0 + irand(2);
                const int p1 = 0 + irand(2);

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_pool_2d(ctx0, x[0], op, k0, k1, s0, s1, p0, p1));

                GGML_PRINT_DEBUG("ggml_pool_2d f32: op=%s k0=%d, k1=%d, s0=%d, s1=%d, p0=%d, p1=%d\n",
                                 op == GGML_OP_POOL_MAX ? "max" : "avg", k0, k1, s0, s1, p0, p1);
                std::vector<double> expected_vals;
                if (op == GGML_OP_POOL_MAX) {
                    expected_vals.push_back(0.0);
                    expected_vals.push_back(1.0);
                }
                check_gradient("ggml_pool_2d f32", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY, expected_vals);
            }
        }

        // flash_attn f32
        // TODO: adapt to ggml_flash_attn_ext() changes
        //{
        //    srand(seed);
        //    const int nargs = 3;

        //    int64_t ne2[4];

        //    get_random_dims(ne2, 4);
        //    int64_t D = ne2[0];
        //    int64_t N = ne2[1];
        //    int64_t M = ne2[2] + N;
        //    int64_t B = ne2[3];

        //    for (int masked = 0; masked <= 1; ++masked) {
        //        for (int ndims = 2; ndims <= 4; ++ndims) {
        //            int max_nrep = (ndims >= 3) ? 2 : 1;
        //            for (int nrep = 1; nrep < max_nrep; ++nrep) {
        //                int64_t neq[4] = { D, N, B*nrep, ne[3] };
        //                int64_t nek[4] = { D, M, B, ne[3] };
        //                int64_t nev[4] = { M, D, B, ne[3] };
        //                if (ndims == 2) {
        //                    neq[2] = 1; neq[3] = 1;
        //                    nek[2] = 1; nek[3] = 1;
        //                    nev[2] = 1; nev[3] = 1;
        //                } else if (ndims == 3) {
        //                    neq[3] = 1;
        //                    nek[3] = 1;
        //                    nev[3] = 1;
        //                }
        //                x[0] = get_random_tensor_f32(ctx0, ndims, neq, -0.1250f, 0.1250f);
        //                x[1] = get_random_tensor_f32(ctx0, ndims, nek, -0.1250f, 0.1250f);
        //                x[2] = get_random_tensor_f32(ctx0, ndims, nev, -0.1250f, 0.1250f);
        //                ggml_set_param(ctx0, x[0]);
        //                ggml_set_param(ctx0, x[1]);
        //                ggml_set_param(ctx0, x[2]);

        //                struct ggml_tensor * f = ggml_sum(ctx0, ggml_flash_attn(ctx0, x[0], x[1], x[2], (masked == 0)));

        //                check_gradient("flash_attn f32", ctx0, x, f, ndims, nargs, 1.5e-4f, 1e-3f, INFINITY, {});
        //            }
        //        }
        //    }
        //}

        ggml_free(ctx0);
    }

    return 0;
}
