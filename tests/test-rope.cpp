#include "ggml.h"
#include "ggml-cpu.h"

#include <array>
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

//
// test comparing rope and rope_comp
//

struct test_rope {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx; // used to generate positions
    float fs; // freq_scale
    float ef; // ext_factor
    float af; // attn_factor
    bool ff;
    int v; // view (1 : non-contiguous a)
    bool forward; // unused for now
    bool inplace;

    bool use_comp = false;

    std::string vars() {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "type=%d ne=(%lld,%lld,%lld,%lld) n_dims=%d mode=%d fs=%f ef=%f af=%f ff=%d v=%d inplace=%d",
            type, ne_a[0], ne_a[1], ne_a[2], ne_a[3], n_dims, mode, fs, ef, af, ff ? 1 : 0, v, inplace ? 1 : 0);
        return std::string(buf);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 3, 1},
            int n_dims = 10, int mode = GGML_ROPE_TYPE_NORMAL, int n_ctx = 512, float fs = 1.0f,
            float ef = 0.0f, float af = 0.0f, bool ff = false, int v = 0, bool forward = true, bool inplace = false)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx), fs(fs), ef(ef), af(af), ff(ff), v(v), forward(forward), inplace(inplace) {}

    ggml_tensor * _ggml_rope_multi(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c,
                int                   n_dims,
                int                   sections[GGML_MROPE_SECTIONS],
                int                   mode,
                int                   n_ctx_orig,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow) {
        if (use_comp) {
            int legacy_type = GGML_ROPE_TYPE_NEOX; // the only mode supported by legacy m-rope kernel
            int n_dims_corrected = n_dims;
            float freq_base_corrected = freq_base;
            ggml_tensor * pos = ggml_cast(ctx, b, GGML_TYPE_F32); // pos must be F32

            // pos must be 4D
            pos = ggml_reshape_2d(ctx, pos, b->ne[0]/4, 4);

            if (mode == GGML_ROPE_TYPE_VISION) {
                // correct n_dims for vision
                // instead of rotating n_dims/2, we actually rotate all dims
                // in other words, there are no dimensions that are not rotated
                n_dims_corrected = n_dims * 2;
                // correct theta_scale for vision
                // only n_dims/2 are rotated for each dimension (we have 2 dimensions: x, y)
                // to adjust theta_scale, we need to adjust freq_base accordingly
                freq_base_corrected = powf(freq_base, 2.0f);
            }

            ggml_tensor * x = _ggml_rope_ext(
                ctx, a, pos, c, n_dims_corrected, legacy_type, n_ctx_orig,
                freq_base_corrected, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
            GGML_ASSERT(x->op == GGML_OP_ROPE_COMP);
            x = ggml_rope_comp_set_multi(ctx, x, mode, sections);
            return x;
        } else {
            return ggml_rope_multi(
                ctx, a, b, c, n_dims, sections, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        }
    }

    struct ggml_tensor * _ggml_rope_ext(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c,
                int                   n_dims,
                int                   mode,
                int                   n_ctx_orig,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow) {
        if (use_comp) {
            b = ggml_cast(ctx, b, GGML_TYPE_F32); // pos must be F32
            bool is_neox = (mode == GGML_ROPE_TYPE_NEOX);
            float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
            ggml_tensor * x = ggml_rope_comp(
                ctx, a, b, n_dims,
                theta_scale,
                is_neox ? GGML_ROPE_ORDERING_NEOX : GGML_ROPE_ORDERING_NORMAL);
            if (ext_factor != 0.0f) {
                // apply yarn mscale
                attn_factor *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            x = ggml_rope_comp_set_yarn(ctx, x, n_ctx_orig, n_dims,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
            if (c) {
                x = ggml_rope_comp_set_freq_factors(ctx, x, c);
            }
            return x;
        } else {
            return ggml_rope_ext(
                ctx, a, b, c, n_dims, mode, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
        }
    }

    ggml_tensor * a    = nullptr;
    ggml_tensor * freq = nullptr;
    ggml_tensor * pos  = nullptr;

    void build_input(ggml_context * ctx) {
        GGML_ASSERT(a == nullptr);
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (forward) {
                ggml_set_param(a);
            }
            ggml_set_input(a);
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (forward) {
                ggml_set_param(a);
            }
            ggml_set_input(a);
            ggml_set_name(a, "a");
        }

        const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

        if (is_mrope || is_vision) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2] * 4);
        } else {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        }
        ggml_set_input(pos);
        ggml_set_name(pos, "pos");

        if (ff) {
            freq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2);
            ggml_set_input(freq);
            ggml_set_name(freq, "freq");
        }
    }

    ggml_tensor * build_graph(ggml_context * ctx) {
        const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
        ggml_tensor * out = nullptr;
        if (is_mrope) {
            if (is_vision) {
                GGML_ASSERT(n_dims/4 > 0);
                int rope_sections[4] = {n_dims/4, n_dims/4, 0, 0}; // Vision-RoPE only use first two dimension for image (x, y) coordinate
                if (forward) {
                    if (inplace) {
                        //out = _ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    } else {
                        out = _ggml_rope_multi(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    }
                } else {
                    //out = _ggml_rope_multi_back(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            } else {
                GGML_ASSERT(n_dims/3 > 0);
                int rope_sections[4] = {n_dims/3, n_dims/3, n_dims/3, 0};
                if (forward) {
                    if (inplace) {
                        //out = _ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    } else {
                        out = _ggml_rope_multi(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    }
                } else {
                    //out = _ggml_rope_multi_back(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            }
        } else {
            if (forward) {
                if (inplace) {
                    //out = _ggml_rope_ext_inplace(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                } else {
                    out = _ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            } else {
                //out = _ggml_rope_ext_back(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }
        }

        if (out) {
            ggml_set_name(out, "out");
        }

        return out;
    }

    void init_tensor_uniform(ggml_tensor * tensor, float fmin = -1.0f, float fmax = 1.0f) {
        const size_t n_elements = ggml_nelements(tensor);
        switch (tensor->type) {
            case GGML_TYPE_F32:
                {
                    float * data = (float *)tensor->data;
                    for (size_t i = 0; i < n_elements; i++) {
                        data[i] = frand()*(fmax - fmin) + fmin;
                    }
                } break;
            case GGML_TYPE_F16:
                {
                    ggml_fp16_t * data = (ggml_fp16_t *)tensor->data;
                    for (size_t i = 0; i < n_elements; i++) {
                        float v = frand()*(fmax - fmin) + fmin;
                        data[i] = ggml_fp32_to_fp16(v);
                    }
                } break;
            default:
                assert(false);
        }
    }

    void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if ((t->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }
            if (t->type == GGML_TYPE_I32) {
                // pos
                const int num_pos_ids = (mode & GGML_ROPE_TYPE_MROPE) ? ne_a[2] * 4 : ne_a[2];
                std::vector<int> data(num_pos_ids);
                for (int i = 0; i < num_pos_ids; i++) {
                    data[i] = rand() % n_ctx;
                }
                // printf("init pos tensor %s\n", ggml_get_name(t));
                memcpy(t->data, data.data(), num_pos_ids * sizeof(int));
            } else {
                if (t->ne[0] == n_dims/2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    // printf("init freq tensor %s\n", ggml_get_name(t));
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    // printf("init param tensor %s\n", ggml_get_name(t));
                    init_tensor_uniform(t);
                }
            }
        }
    }
};

static void test_rope_comp() {
    ggml_init_params params = {
        /* .mem_size   = */ 128*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<test_rope *> test_cases;

    bool all = true;
    bool fw  = true;
    for (float fs : { 1.0f, 1.4245f }) {
        for (float ef : { 0.0f, 0.7465f }) {
            for (float af : { 1.0f, 1.4245f }) {
                for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                    for (bool ff : {false, true}) { // freq_factors
                        for (float v : { 0, 1 }) {
                            test_cases.emplace_back(new test_rope(type, {128,  32, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 7B

                            if (all) {
                                test_cases.emplace_back(new test_rope(type, {128,  40, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 13B
                                test_cases.emplace_back(new test_rope(type, {128,  52, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 30B
                                test_cases.emplace_back(new test_rope(type, {128,  64, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 65B
                            }

                            if (all) {
                                test_cases.emplace_back(new test_rope(type, { 64,   1, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                test_cases.emplace_back(new test_rope(type, { 64,  71, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                test_cases.emplace_back(new test_rope(type, { 64,   8, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)

                                test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, { 80,  32, 4, 1},  32, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));

                                test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (stablelm)
                                test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                test_cases.emplace_back(new test_rope(type, { 80,  32, 4, 1},  32, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                            }

                            if (all) {
                                test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 2B)
                                test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 7B)
                                test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1},  20, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1},  32, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1}, 128, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,imrope (qwen3vl 2B)
                                test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1}, 128, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,imrope (qwen3vl 7B)
                                test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1},  20, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1},  32, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, { 80,  16, 2, 1},  80, GGML_ROPE_TYPE_VISION, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl ViT)
                                test_cases.emplace_back(new test_rope(type, {128,  16, 2, 1}, 128, GGML_ROPE_TYPE_IMROPE, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen3vl)
                            }

                            test_cases.emplace_back(new test_rope(type, { 64, 128, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                        }
                    }

                    all = false;
                }
            }
        }
    }

    std::vector<uint8_t> work_buffer;

    size_t n_passed = 0;

    for (size_t i = 0; i < test_cases.size(); i++) {
        test_rope * tc = test_cases[i];

        ggml_context * ctx0 = ggml_init(params);
        ggml_cgraph * gf = ggml_new_graph(ctx0);

        tc->build_input(ctx0);
        tc->initialize_tensors(ctx0);

        ggml_tensor * out0 = tc->build_graph(ctx0);
        tc->use_comp = true;
        ggml_tensor * out1 = tc->build_graph(ctx0);

        if (out0 == nullptr || out1 == nullptr) {
            GGML_PRINT("test_rope_comp \x1b[33mSKIPPED\x1b[0m: %s\n", tc->vars().c_str());
            ggml_free(ctx0);
            delete tc;
            continue;
        }

        // calculate nmse between out0 and out1
        ggml_tensor * diff    = ggml_sub(ctx0, out0, out1);
        ggml_tensor * mse_a_b = ggml_sum(ctx0, ggml_sqr(ctx0, diff));
        ggml_tensor * mse_a_0 = ggml_sum(ctx0, ggml_sqr(ctx0, out0));
        ggml_tensor * out     = ggml_div(ctx0, mse_a_b, mse_a_0);
        out = ggml_cast(ctx0, out, GGML_TYPE_F32);

        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_helper(work_buffer, gf, 4);

        GGML_ASSERT(ggml_nelements(out) == 1);
        float nmse = ((float *)out->data)[0];
        const float nmse_threshold = 1e-3f;
        if (nmse > nmse_threshold) {
            GGML_PRINT("test_rope_comp \x1b[31mFAILED\x1b[0m: nmse=%f > %f for %s\n",  nmse, nmse_threshold, tc->vars().c_str());
        } else {
            GGML_PRINT("test_rope_comp OK    : nmse=%f <= %f for %s\n", nmse, nmse_threshold, tc->vars().c_str());
            n_passed++;
        }

        ggml_free(ctx0);
        delete tc;
    }

    GGML_ASSERT(n_passed == test_cases.size());
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
    for (int m = 0; m < 5; ++m) {
        const int ndims = 4;

        const int64_t n_rot = 128;
        const int64_t ne[4] = { 2*n_rot, 32, 73, 1 };

        const int n_past_0 = 100;
        const int n_past_2 = 33;

        struct ggml_tensor * r0;
        struct ggml_tensor * r1;
        struct ggml_tensor * r2;
        x = get_random_tensor_f32(ctx0, ndims, ne, -1.0f, 1.0f);
        int mode = -1;

        if (m < 2) {
            struct ggml_tensor * p0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);
            struct ggml_tensor * p1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);
            struct ggml_tensor * p2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]);

            for (int i = 0; i < ne[2]; ++i) {
                ((int32_t *) p0->data)[i] = n_past_0 + i;
                ((int32_t *) p1->data)[i] = n_past_2 - n_past_0;
                ((int32_t *) p2->data)[i] = n_past_2 + i;
            }
            // test mode 0, 2  (standard, GPT-NeoX)
            mode = m == 0 ? GGML_ROPE_TYPE_NORMAL : GGML_ROPE_TYPE_NEOX;

            // 100, 101, 102, ..., 172
            r0 = ggml_rope(ctx0, x,  p0, n_rot, mode);
            // -67, -67, -67, ..., -67
            r1 = ggml_rope(ctx0, r0, p1, n_rot, mode); // "context swap", i.e. forget n_past_0 - n_past_2 tokens

            //  33,  34,  35, ..., 105
            r2 = ggml_rope(ctx0, x,  p2, n_rot, mode);
        } else {
            // testing multi-dimension rope position embedding mode
            struct ggml_tensor * p0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2] * 4);
            struct ggml_tensor * p1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2] * 4);
            struct ggml_tensor * p2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2] * 4);

            int sections[4] = {16, 24, 24, 0};

            mode = (m == 2) ? GGML_ROPE_TYPE_MROPE : (m == 3) ? GGML_ROPE_TYPE_VISION : GGML_ROPE_TYPE_IMROPE;

            for (int i = 0; i < ne[2]; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ((int32_t *) p0->data)[i + ne[2] * j] = n_past_0 + i + j;
                    ((int32_t *) p1->data)[i + ne[2] * j] = n_past_2 - n_past_0;
                    ((int32_t *) p2->data)[i + ne[2] * j] = n_past_2 + i + j;
                }
            }

            // [[100, 101, 102, ..., 172],
            // [101, 102, 103, ..., 173],
            // [102, 103, 104, ..., 174]]
            r0 = ggml_rope_multi(
                ctx0, x, p0, nullptr,
                n_rot, sections, mode, 32768, 1000000, 1, 0, 1, 32, 1);
            // [[-67, -67, -67, ..., -67]
            // [-67, -67, -67, ..., -67]
            // [-67, -67, -67, ..., -67]]
            r1 = ggml_rope_multi(
                ctx0, r0, p1, nullptr,
                n_rot, sections, mode, 32768, 1000000, 1, 0, 1, 32, 1);

            //  [[33,  34,  35, ..., 105]
            //  [34,  35,  36, ..., 106]
            //  [35,  36,  37, ..., 107]]
            r2 = ggml_rope_multi(
                ctx0, x, p2, nullptr,
                n_rot, sections, mode, 32768, 1000000, 1, 0, 1, 32, 1);
        }

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

    test_rope_comp();

    return 0;
}
