// Q6_K MMQ coalesced GPU unit test
// Tests the production MMQ kernel path using coalesced layout (batch > 1)
//
// Build: cmake --build build --target test-mmq-q6k-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-q6k-coalesced

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"

#define QK_K 256
#define WARP_SIZE 32
#define MMVQ_COALESCED_TILE_BLOCKS WARP_SIZE
#define QI6_K (QK_K / 8)
#ifndef QI8_1
#define QI8_1 (QK8_1 / 4)
#endif

static inline int get_int_from_uint8(const uint8_t * x8, const int i32) {
    const uint16_t * x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0];
    x32 |= (int) x16[1] << 16;
    return x32;
}

static inline int get_int_from_int8(const int8_t * x8, const int i32) {
    const uint16_t * x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0];
    x32 |= (int) x16[1] << 16;
    return x32;
}

static void cpu_mul_mat_q6k_f32(const void* x_data, const float* y,
                                float* dst, int nrows, int ncols, int n_tokens) {
    const block_q6_K* x = (const block_q6_K*)x_data;
    const int blocks_per_row = ncols / QK_K;

    std::vector<float> x_f32(ncols);
    for (int row = 0; row < nrows; row++) {
        dequantize_row_q6_K(&x[row * blocks_per_row], x_f32.data(), ncols);
        for (int tok = 0; tok < n_tokens; tok++) {
            float sum = 0.0f;
            for (int k = 0; k < ncols; k++) {
                sum += x_f32[k] * y[tok * ncols + k];
            }
            dst[tok * nrows + row] = sum;
        }
    }
}

static void dequantize_row_q8_1_ref(const block_q8_1* x, float* y, int64_t k) {
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_1; ++j) {
            y[i * QK8_1 + j] = d * x[i].qs[j];
        }
    }
}

static void cpu_mul_mat_q6k_q8_1(const void* x_data, const float* y,
                                 float* dst, int nrows, int ncols, int n_tokens) {
    const block_q6_K* x = (const block_q6_K*)x_data;
    const int blocks_per_row_x = ncols / QK_K;
    const int blocks_per_row_y = ncols / QK8_1;

    std::vector<block_q8_1> y_q8(blocks_per_row_y * n_tokens);
    std::vector<float> y_q8_f32(ncols * n_tokens);

    for (int tok = 0; tok < n_tokens; ++tok) {
        block_q8_1* y_row = &y_q8[tok * blocks_per_row_y];
        quantize_row_q8_1_ref(y + tok * ncols, y_row, ncols);
        dequantize_row_q8_1_ref(y_row, y_q8_f32.data() + tok * ncols, ncols);
    }

    std::vector<float> x_f32(ncols);
    for (int row = 0; row < nrows; ++row) {
        dequantize_row_q6_K(&x[row * blocks_per_row_x], x_f32.data(), ncols);
        for (int tok = 0; tok < n_tokens; ++tok) {
            const float* y_row = y_q8_f32.data() + tok * ncols;
            float sum = 0.0f;
            for (int i = 0; i < ncols; ++i) {
                sum += x_f32[i] * y_row[i];
            }
            dst[tok * nrows + row] = sum;
        }
    }
}

static bool test_mmq_q6k_coalesced(int n_tokens, bool verbose) {
    printf("\n--- MMQ Q6_K coalesced batch=%d ---\n", n_tokens);

    setenv("GGML_SYCL_LAYOUT_OVERRIDE", "coalesced", 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }
    printf("  Backend: %s\n", ggml_backend_name(backend));

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Coalesced layout requires blocks_per_row multiple of MMVQ_COALESCED_TILE_BLOCKS.
    const int n_embd = QK_K * MMVQ_COALESCED_TILE_BLOCKS;  // 8192
    const int n_ff = 256;

    const int blocks_per_row = n_embd / QK_K;
    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("  SKIP: blocks_per_row=%d not divisible by %d\n",
               blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        ggml_backend_free(backend);
        return true;
    }

    printf("  Weight: Q6_K [%d x %d], Input: F32 [%d x %d]\n",
           n_embd, n_ff, n_embd, n_tokens);

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAIL: Could not create ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff);
    ggml_set_name(weight, "test_q6k_weight_coalesced");

    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "test_input");

    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "test_output");

    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        printf("  FAIL: Could not allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        printf("  FAIL: Could not allocate compute buffer\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    std::mt19937 rng(42 + n_tokens);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int weight_floats = n_ff * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    printf("  Quantizing weights to Q6_K...\n");
    const int total_blocks = n_ff * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_ff, n_embd, nullptr);

    printf("  Uploading weights to GPU...\n");
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    printf("\n  === COALESCED LAYOUT CHECK ===\n");
    printf("  weight->extra = %p\n", (void*)weight->extra);
    if (!weight->extra) {
        printf("  FAIL: tensor->extra is NULL - coalesced path will NOT be used\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const ggml_tensor_layout * layout = weight->layout;
    if (!layout) {
        printf("  FAIL: tensor->layout is NULL - coalesced path will NOT be used\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    printf("  layout->mode = %d (0=AoS, 1=SoA, 2=Coalesced)\n", (int) layout->mode);
    if (layout->mode != GGML_LAYOUT_COALESCED) {
        printf("  FAIL: expected COALESCED layout (2), got %d\n", (int) layout->mode);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    const int row_quants_bytes = tiles_per_row * MMVQ_COALESCED_TILE_BLOCKS *
        ((QK_K / 2) + (QK_K / 4) + (QK_K / 16));
    const size_t d_offset = (size_t)n_ff * (size_t)row_quants_bytes;

    std::vector<uint8_t> gpu_raw_data(total_blocks * sizeof(block_q6_K));
    ggml_backend_tensor_get(weight, gpu_raw_data.data(), 0, total_blocks * sizeof(block_q6_K));

    if (verbose) {
        auto sub_32 = [](int x) {
            int out = 0;
            for (int i = 0; i < 4; ++i) {
                const int8_t v = (int8_t)((x >> (i * 8)) & 0xFF);
                const int8_t r = (int8_t)(v - 32);
                out |= ((int)(uint8_t)r) << (i * 8);
            }
            return out;
        };

        const int row = 0;
        const int block = 0;
        const block_q6_K * bq6 = &weight_q6k[row * blocks_per_row + block];

        constexpr int tile_ql_stride = 2 * QI6_K + 1;
        int x_ql_aos[tile_ql_stride] = {0};
        int x_ql_coa[tile_ql_stride] = {0};

        const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
        const int ql_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 2);
        const int qh_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 4);
        const int sc_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 16);
        const int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;
        const int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;

        const uint8_t * coa_base = gpu_raw_data.data();
        const size_t row_stride = (size_t)tiles_per_row * (size_t)tile_total;
        const uint8_t * tile_base = coa_base + row * row_stride;
        const uint8_t * tile_ql = tile_base;
        const uint8_t * tile_qh = tile_ql + ql_tile_bytes;
        const int8_t * tile_sc = (const int8_t *)(tile_qh + qh_tile_bytes);

        for (int k = 0; k < QI6_K; ++k) {
            const int kqsx = k;
            const int ql = get_int_from_uint8(bq6->ql, kqsx);
            const int ql0 = (ql >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql >> 4) & 0x0F0F0F0F;

            const int qh_idx = (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4);
            const int qh = get_int_from_uint8(bq6->qh, qh_idx);
            const int qh_shift = 2 * ((kqsx % (QI6_K/2)) / (QI6_K/4));
            const int qh0 = ((qh >> qh_shift) << 4) & 0x30303030;
            const int qh1 = (qh >> qh_shift) & 0x30303030;

            const int kq0 = 2 * kqsx - kqsx % (QI6_K/2);
            const int kq1 = kq0 + (QI6_K/2);

            x_ql_aos[kq0] = sub_32(ql0 | qh0);
            x_ql_aos[kq1] = sub_32(ql1 | qh1);

            const int ql_offset = kqsx * word_stride;
            const int qh_offset = qh_idx * word_stride;
            const int ql_coa = *((const int *)(tile_ql + ql_offset));
            const int qh_coa = *((const int *)(tile_qh + qh_offset));
            const int qh_coa0 = ((qh_coa >> qh_shift) << 4) & 0x30303030;
            const int ql_coa0 = (ql_coa >> 0) & 0x0F0F0F0F;
            const int ql_coa1 = (ql_coa >> 4) & 0x0F0F0F0F;
            const int qh_coa1 = (qh_coa >> qh_shift) & 0x30303030;

            x_ql_coa[kq0] = sub_32(ql_coa0 | qh_coa0);
            x_ql_coa[kq1] = sub_32(ql_coa1 | qh_coa1);
        }

        printf("  [CPU DBG] x_ql_aos[0..13]=0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x\n",
               x_ql_aos[0], x_ql_aos[1], x_ql_aos[2], x_ql_aos[3],
               x_ql_aos[4], x_ql_aos[5], x_ql_aos[6], x_ql_aos[7],
               x_ql_aos[8], x_ql_aos[9], x_ql_aos[10], x_ql_aos[11],
               x_ql_aos[12], x_ql_aos[13]);
        printf("  [CPU DBG] x_ql_coa[0..13]=0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x\n",
               x_ql_coa[0], x_ql_coa[1], x_ql_coa[2], x_ql_coa[3],
               x_ql_coa[4], x_ql_coa[5], x_ql_coa[6], x_ql_coa[7],
               x_ql_coa[8], x_ql_coa[9], x_ql_coa[10], x_ql_coa[11],
               x_ql_coa[12], x_ql_coa[13]);

        const float d_aos = ggml_fp16_to_fp32(bq6->d);
        const int sc_aos = get_int_from_int8(bq6->scales, 0);
        const int sc_coa = *((const int *)(tile_sc + 0));
        printf("  [CPU DBG] x_d=%.6f x_sc_aos=0x%08x x_sc_coa=0x%08x\n",
               d_aos, sc_aos, sc_coa);
    }

    ggml_half expected_d = weight_q6k[0].d;
    ggml_half got_d = 0;
    memcpy(&got_d, &gpu_raw_data[d_offset], sizeof(ggml_half));
    if (got_d != expected_d) {
        printf("  FAIL: d[0] mismatch at coalesced d_offset %zu: expected 0x%04x got 0x%04x\n",
               d_offset, (unsigned)expected_d, (unsigned)got_d);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    printf("  PASS: d[0] matches at coalesced d_offset=%zu\n", d_offset);
    printf("  === END COALESCED LAYOUT CHECK ===\n\n");

    std::vector<float> input_f32(n_embd * n_tokens);
    for (int i = 0; i < n_embd * n_tokens; i++) {
        input_f32[i] = dist(rng);
    }
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * n_tokens * sizeof(float));

    if (verbose) {
        auto sub_32 = [](int x) {
            int out = 0;
            for (int i = 0; i < 4; ++i) {
                const int8_t v = (int8_t)((x >> (i * 8)) & 0xFF);
                const int8_t r = (int8_t)(v - 32);
                out |= ((int)(uint8_t)r) << (i * 8);
            }
            return out;
        };

        const int row = 0;
        const int block = 0;
        const block_q6_K * bq6 = &weight_q6k[row * blocks_per_row + block];
        constexpr int tile_ql_stride = 2 * QI6_K + 1;
        int x_ql_aos[tile_ql_stride] = {0};

        for (int k = 0; k < QI6_K; ++k) {
            const int kqsx = k;
            const int ql = get_int_from_uint8(bq6->ql, kqsx);
            const int ql0 = (ql >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql >> 4) & 0x0F0F0F0F;

            const int qh_idx = (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4);
            const int qh = get_int_from_uint8(bq6->qh, qh_idx);
            const int qh_shift = 2 * ((kqsx % (QI6_K/2)) / (QI6_K/4));
            const int qh0 = ((qh >> qh_shift) << 4) & 0x30303030;
            const int qh1 = (qh >> qh_shift) & 0x30303030;

            const int kq0 = 2 * kqsx - kqsx % (QI6_K/2);
            const int kq1 = kq0 + (QI6_K/2);

            x_ql_aos[kq0] = sub_32(ql0 | qh0);
            x_ql_aos[kq1] = sub_32(ql1 | qh1);
        }

        const int kx = n_embd;
        const int kx_padded = kx;
        const int blocks_per_row_y = kx / QK8_1;
        const size_t row_stride = (size_t)(kx_padded / QK8_1) * sizeof(block_q8_1);

        std::vector<block_q8_1> y_q8(blocks_per_row_y * n_tokens);
        std::vector<uint8_t> y_soa(row_stride * n_tokens, 0);

        for (int row = 0; row < n_tokens; ++row) {
            quantize_row_q8_1_ref(input_f32.data() + row * kx, &y_q8[row * blocks_per_row_y], kx);
            uint8_t * row_base = y_soa.data() + row * row_stride;
            for (int col = 0; col < blocks_per_row_y; ++col) {
                const block_q8_1 & b = y_q8[row * blocks_per_row_y + col];
                std::memcpy(row_base + col * QK8_1, b.qs, QK8_1);
                const ggml_half ds_pair[2] = { b.d, b.s };
                std::memcpy(row_base + kx + col * sizeof(ds_pair), ds_pair, sizeof(ds_pair));
            }
        }

        const uint8_t * row0 = y_soa.data();
        const int8_t * y_block_qs = (const int8_t *)(row0 + 0 * QK8_1);
        const ggml_half * y_block_ds = (const ggml_half *)(row0 + kx);

        const int y_word0 = *((const int *)(y_block_qs + 0 * 4));
        const int y_word1 = *((const int *)(y_block_qs + 1 * 4));
        const int y_word2 = *((const int *)(y_block_qs + 2 * 4));
        const int y_word3 = *((const int *)(y_block_qs + 3 * 4));
        printf("  [CPU DBG] y_qs_word[0..3]=0x%08x,0x%08x,0x%08x,0x%08x\n",
               y_word0, y_word1, y_word2, y_word3);
        printf("  [CPU DBG] y_ds[0]=[%.6f,%.6f]\n",
               ggml_fp16_to_fp32(y_block_ds[0]),
               ggml_fp16_to_fp32(y_block_ds[1]));

        // CPU vec_dot for k=0 using the same layout as tile_y_qs.
        auto dp4a = [](int a, int b, int c) {
            const int8_t * ap = (const int8_t *) &a;
            const int8_t * bp = (const int8_t *) &b;
            int sum = c;
            for (int i = 0; i < 4; ++i) {
                sum += (int) ap[i] * (int) bp[i];
            }
            return sum;
        };

        int y_tile[WARP_SIZE];
        for (int idx = 0; idx < WARP_SIZE; ++idx) {
            const int block = idx / QI8_1;
            const int word = idx % QI8_1;
            const int8_t * block_qs = y_block_qs + block * QK8_1;
            y_tile[idx] = *((const int *)(block_qs + word * 4));
        }

        const int x_sc_word = get_int_from_int8(bq6->scales, 0);
        const int8_t * sc = (const int8_t *) &x_sc_word;
        const float d6 = ggml_fp16_to_fp32(bq6->d);
        const float d8_0 = ggml_fp16_to_fp32(y_block_ds[0]);
        const float d8_1 = ggml_fp16_to_fp32(y_block_ds[2]);

        float sumf_d = 0.0f;
        for (int i0 = 0; i0 < 8; i0 += 4) {
            int sumi_x = 0;
            int sumi_y = 0;
            for (int i = i0; i < i0 + 2; ++i) {
                sumi_x = dp4a(x_ql_aos[2 * i + 0], y_tile[2 * i + 0], sumi_x);
                sumi_x = dp4a(x_ql_aos[2 * i + 1], y_tile[2 * i + 1], sumi_x);
                sumi_y = dp4a(x_ql_aos[2 * i + 4], y_tile[2 * i + 4], sumi_y);
                sumi_y = dp4a(x_ql_aos[2 * i + 5], y_tile[2 * i + 5], sumi_y);
            }
            const float d8 = (i0 == 0) ? d8_0 : d8_1;
            sumf_d += d8 * (sc[i0 / 2 + 0] * sumi_x + sc[i0 / 2 + 1] * sumi_y);
        }

        const float cpu_vecdot = d6 * sumf_d;
        printf("  [CPU DBG] vec_dot k=0 -> %.6f\n", cpu_vecdot);
    }

    printf("  Building compute graph...\n");
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph) {
        printf("  FAIL: Could not create graph\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, output);

    printf("  Executing MMQ graph...\n");
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> gpu_output(n_ff * n_tokens);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_ff * n_tokens * sizeof(float));

    printf("  Computing CPU reference...\n");
    std::vector<float> cpu_output(n_ff * n_tokens);
    cpu_mul_mat_q6k_q8_1(weight_q6k.data(), input_f32.data(),
                         cpu_output.data(), n_ff, n_embd, n_tokens);

    const float ABS_TOL = 0.1f;
    const float REL_TOL = 0.15f;

    int mismatches = 0;
    float max_rel_error = 0.0f;
    float max_abs_error = 0.0f;
    int mismatch_tok = -1, mismatch_row = -1;
    float mismatch_gpu = 0, mismatch_cpu = 0;

    for (int tok = 0; tok < n_tokens; tok++) {
        for (int row = 0; row < n_ff; row++) {
            int idx = tok * n_ff + row;
            float gpu_val = gpu_output[idx];
            float cpu_val = cpu_output[idx];

            float abs_diff = std::abs(gpu_val - cpu_val);
            float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;

            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
                max_abs_error = abs_diff;
                mismatch_tok = tok;
                mismatch_row = row;
                mismatch_gpu = gpu_val;
                mismatch_cpu = cpu_val;
            }

            bool abs_fail = abs_diff > ABS_TOL;
            bool rel_fail = rel_error > REL_TOL;
            if (abs_fail && rel_fail) {
                if (verbose && mismatches < 5) {
                    printf("  MISMATCH tok=%d row=%d: GPU=%.6f CPU=%.6f diff=%.6f (%.2f%%)\n",
                           tok, row, gpu_val, cpu_val, abs_diff, rel_error * 100);
                }
                mismatches++;
            }
        }
    }

    printf("  Max relative error: %.4f%% at tok=%d row=%d (GPU=%.6f CPU=%.6f)\n",
           max_rel_error * 100, mismatch_tok, mismatch_row, mismatch_gpu, mismatch_cpu);
    printf("  Thresholds: abs>%.2f AND rel>%.0f%%\n", ABS_TOL, REL_TOL * 100);

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (mismatches > 0) {
        printf("  FAIL: %d mismatches in %d values (%.2f%%)\n",
               mismatches, n_ff * n_tokens, (float)mismatches / (n_ff * n_tokens) * 100);
        return false;
    }

    printf("  PASS: All values within tolerance (abs>%.2f AND rel>%.0f%%)\n", ABS_TOL, REL_TOL * 100);
    return true;
}

int main() {
    printf("Q6_K MMQ Coalesced GPU Unit Test\n");
    printf("================================\n");
    printf("\nThis tests the MMQ kernel path using coalesced layout (batch > 1)\n");
    printf("Batch size is set > %d to force MMQ.\n\n", 8);

    int passed = 0;
    int failed = 0;

    printf("Test 1: Coalesced batch=16\n");
    if (test_mmq_q6k_coalesced(16, true)) passed++; else failed++;

    printf("\n================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
