// Unit tests for coalesced MMVQ kernels (Q4_0 and Q6_K)
// Build: cmake --build build --target test-mmvq-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmvq-coalesced

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#include <cstdint>

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Constants from ggml-common.h
#define QK4_0 32
#define QK_K 256
#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QI6_K 32
#define QR6_K 2
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#define WARP_SIZE GGML_SYCL_WARP_SIZE
#define GGML_SYCL_MMV_Y 1
#define MMVQ_COALESCED_TILE_BLOCKS WARP_SIZE

static inline float half_to_float(sycl::half h) {
    return (float) h;
}

// Minimal block structures matching ggml layout (host + device)
struct block_q4_0 {
    sycl::half d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    sycl::half d;
};

struct block_q8_1 {
    sycl::half d;
    sycl::half s;
    int8_t qs[QK8_1];
};

static void reorder_q8_1_to_qs_ds(const std::vector<block_q8_1> & src, std::vector<uint8_t> & dst) {
    const size_t blocks = src.size();
    const size_t qs_bytes = blocks * QK8_1;
    const size_t ds_bytes = blocks * sizeof(sycl::half2);
    dst.assign(qs_bytes + ds_bytes, 0);

    uint8_t * qs_dst = dst.data();
    uint8_t * ds_dst = dst.data() + qs_bytes;

    for (size_t i = 0; i < blocks; ++i) {
        std::memcpy(qs_dst + i * QK8_1, src[i].qs, QK8_1);
        sycl::half2 ds = sycl::half2(src[i].d, src[i].s);
        std::memcpy(ds_dst + i * sizeof(sycl::half2), &ds, sizeof(sycl::half2));
    }
}

// Helpers for Q6_K reference
static inline int get_int_from_int8_aligned(const int8_t * x8, const int i32) {
    return *((const int *)(x8 + sizeof(int) * i32));
}

static inline int get_int_from_uint8(const uint8_t * x8, const int i32) {
    const uint16_t * x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0];
    x32 |= (int) x16[1] << 16;
    return x32;
}

static float cpu_vec_dot_q6_K_q8_1(const block_q6_K * bq6_K, const block_q8_1 * bq8_1, int iqs) {
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
    const int8_t * scales = bq6_K->scales + scale_offset;

    auto dp4a = [](int a, int b, int c) {
        const int8_t * ap = (const int8_t *) &a;
        const int8_t * bp = (const int8_t *) &b;
        int sum = c;
        for (int i = 0; i < 4; ++i) {
            sum += (int)ap[i] * (int)bp[i];
        }
        return sum;
    };
    auto sub_32 = [](int x) {
        int out = 0;
        for (int i = 0; i < 4; ++i) {
            const int8_t v = (int8_t)((x >> (i * 8)) & 0xFF);
            const int8_t r = (int8_t)(v - 32);
            out |= ((int)(uint8_t)r) << (i * 8);
        }
        return out;
    };

    float sumf = 0.0f;
    for (int i = 0; i < QR6_K; ++i) {
        const int u = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        const float d8 = half_to_float(bq8_1[bq8_offset + 2 * i].d);
        const int sc = scales[4 * i];

        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = sub_32(vil | vih);
        sumf += d8 * (dp4a(vi, u, 0) * sc);
    }
    return half_to_float(bq6_K->d) * sumf;
}

static float cpu_row_dot_q6_K(const block_q6_K * x_row, const block_q8_1 * y, int ncols) {
    const int blocks_per_row = ncols / QK_K;
    float sum = 0.0f;
    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const block_q6_K * bx = &x_row[ib];
        const block_q8_1 * by = &y[ib * (QK_K / QK8_1)];
        for (int iqs = 0; iqs < QI6_K; ++iqs) {
            sum += cpu_vec_dot_q6_K_q8_1(bx, by, iqs);
        }
    }
    return sum;
}

static void reorder_q4_0_to_soa(const std::vector<block_q4_0> & aos, int ncols, int nrows, std::vector<uint8_t> & soa) {
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = blocks_per_row * nrows;
    const size_t qs_bytes = nblocks * (QK4_0 / 2);
    const size_t d_bytes = nblocks * sizeof(sycl::half);

    soa.assign(qs_bytes + d_bytes, 0);
    uint8_t * qs_dst = soa.data();
    uint8_t * d_dst = soa.data() + qs_bytes;

    for (int ib = 0; ib < nblocks; ++ib) {
        const block_q4_0 & b = aos[ib];
        std::memcpy(qs_dst + ib * (QK4_0 / 2), b.qs, QK4_0 / 2);
    std::memcpy(d_dst + ib * sizeof(sycl::half), &b.d, sizeof(sycl::half));
    }
}

static void coalesce_q4_0_qs(const std::vector<uint8_t> & soa, int ncols, int nrows, std::vector<uint8_t> & coalesced) {
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    const int bytes_per_row = ncols / 2;
    const size_t qs_bytes = nrows * bytes_per_row;
    const size_t d_bytes = nrows * blocks_per_row * sizeof(sycl::half);

    coalesced.assign(qs_bytes + d_bytes, 0);
    std::memcpy(coalesced.data() + qs_bytes, soa.data() + qs_bytes, d_bytes);

    for (int row = 0; row < nrows; ++row) {
        for (int tile = 0; tile < tiles_per_row; ++tile) {
            for (int block_in_tile = 0; block_in_tile < MMVQ_COALESCED_TILE_BLOCKS; ++block_in_tile) {
                for (int word_in_block = 0; word_in_block < 4; ++word_in_block) {
                    const int src_offset = row * bytes_per_row +
                                           tile * (MMVQ_COALESCED_TILE_BLOCKS * 16) +
                                           block_in_tile * 16 +
                                           word_in_block * 4;
                    const int dst_offset = row * bytes_per_row +
                                           tile * (MMVQ_COALESCED_TILE_BLOCKS * 16) +
                                           word_in_block * (MMVQ_COALESCED_TILE_BLOCKS * 4) +
                                           block_in_tile * 4;
                    std::memcpy(coalesced.data() + dst_offset, soa.data() + src_offset, 4);
                }
            }
        }
    }
}

static void coalesce_q6_k_from_aos(const std::vector<block_q6_K> & aos, int ncols, int nrows, std::vector<uint8_t> & coalesced) {
    constexpr int WORD_PLANE_STRIDE = MMVQ_COALESCED_TILE_BLOCKS * 4;
    constexpr int ql_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 2);
    constexpr int qh_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 4);
    constexpr int sc_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 16);
    constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;

    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    const int nblocks = blocks_per_row * nrows;
    const size_t tile_bytes = nrows * tiles_per_row * tile_total;
    const size_t d_bytes = nblocks * sizeof(sycl::half);

    coalesced.assign(tile_bytes + d_bytes, 0);
    uint8_t * dst = coalesced.data();
    uint8_t * dst_d = coalesced.data() + tile_bytes;

    for (int ib = 0; ib < nblocks; ++ib) {
        const block_q6_K & b = aos[ib];
        const int row = ib / blocks_per_row;
        const int col_block = ib % blocks_per_row;
        const int tile = col_block / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile = col_block % MMVQ_COALESCED_TILE_BLOCKS;

        const int tile_base = row * tiles_per_row * tile_total + tile * tile_total;

        for (int word = 0; word < 32; ++word) {
            const int ql_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            std::memcpy(dst + ql_offset, b.ql + word * 4, 4);
        }

        const int qh_base = tile_base + ql_tile_bytes;
        for (int word = 0; word < 16; ++word) {
            const int qh_offset = qh_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            std::memcpy(dst + qh_offset, b.qh + word * 4, 4);
        }

        const int sc_base = qh_base + qh_tile_bytes;
        for (int word = 0; word < 4; ++word) {
            const int sc_offset = sc_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            std::memcpy(dst + sc_offset, b.scales + word * 4, 4);
        }

        std::memcpy(dst_d + ib * sizeof(sycl::half), &b.d, sizeof(sycl::half));
    }
}

static float cpu_row_dot_q4_0(const block_q4_0 * x_row, const block_q8_1 * y, int ncols) {
    const int blocks_per_row = ncols / QK4_0;
    float sum = 0.0f;
    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const block_q4_0 * bx = &x_row[ib];
        const block_q8_1 * by = &y[ib];

        int sumi = 0;
        for (int i = 0; i < QK4_0 / 2; ++i) {
            const int lo = bx->qs[i] & 0x0F;
            const int hi = (bx->qs[i] >> 4) & 0x0F;
            sumi += lo * by->qs[i];
            sumi += hi * by->qs[i + 16];
        }

        const float d4 = half_to_float(bx->d);
        const float d8 = half_to_float(by->d);
        const float s8 = half_to_float(by->s);
        sum += d4 * (sumi * d8 - 8.0f * s8);
    }
    return sum;
}

static float cpu_row_dot_q4_0_coalesced(const uint8_t * x_coal, const uint8_t * y_reordered, int ncols, int nrows, int row) {
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    const int x_row_stride = ncols / 2;
    const sycl::half * x_d = (const sycl::half *)(x_coal + nrows * x_row_stride);
    const int8_t * y_qs = (const int8_t *) y_reordered;
    const sycl::half2 * y_ds = (const sycl::half2 *) (y_reordered + ncols);
    const int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;

    auto dp4a = [](int a, int b, int c) {
        const int8_t * ap = (const int8_t *) &a;
        const int8_t * bp = (const int8_t *) &b;
        int sum = c;
        for (int i = 0; i < 4; ++i) {
            sum += (int)ap[i] * (int)bp[i];
        }
        return sum;
    };

    float sum = 0.0f;
    for (int tile = 0; tile < tiles_per_row; ++tile) {
        const int tile_base = row * x_row_stride + tile * (MMVQ_COALESCED_TILE_BLOCKS * 16);
        for (int block_in_tile = 0; block_in_tile < MMVQ_COALESCED_TILE_BLOCKS; ++block_in_tile) {
            const int block_idx = row * blocks_per_row + tile * MMVQ_COALESCED_TILE_BLOCKS + block_in_tile;
            const float d4 = half_to_float(x_d[block_idx]);
            const int y_block = tile * MMVQ_COALESCED_TILE_BLOCKS + block_in_tile;
            const int y_base = y_block * QK8_1;
            const sycl::half2 ds8 = y_ds[y_block];
            const float d8 = (float) ds8.x();
            const float s8 = (float) ds8.y();

            for (int is_upper_half = 0; is_upper_half < 2; ++is_upper_half) {
                const int word_base = is_upper_half * (2 * word_stride);
                const int word0_offset = word_base + block_in_tile * 4;
                const int word1_offset = word_base + word_stride + block_in_tile * 4;

                const int v0 = *((const int *)(x_coal + tile_base + word0_offset));
                const int v1 = *((const int *)(x_coal + tile_base + word1_offset));

                const int y_offset = is_upper_half * 8;
                const int u0 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4);
                const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 4);
                const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);
                const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 5);

                const int vi0_0 = (v0 >> 0) & 0x0F0F0F0F;
                const int vi1_0 = (v0 >> 4) & 0x0F0F0F0F;
                const int vi0_1 = (v1 >> 0) & 0x0F0F0F0F;
                const int vi1_1 = (v1 >> 4) & 0x0F0F0F0F;

                int sumi = 0;
                sumi = dp4a(vi0_0, u0, sumi);
                sumi = dp4a(vi1_0, u1, sumi);
                sumi = dp4a(vi0_1, u2, sumi);
                sumi = dp4a(vi1_1, u3, sumi);

                sum += d4 * (sumi * d8 - 4.0f * s8);
            }
        }
    }
    return sum;
}

static float cpu_row_dot_q6_k_coalesced(const uint8_t * x_coal, const uint8_t * y_reordered, int ncols, int nrows, int row) {
    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    constexpr int ql_tile_bytes = TILE_BLOCKS * (QK_K / 2);
    constexpr int qh_tile_bytes = TILE_BLOCKS * (QK_K / 4);
    constexpr int sc_tile_bytes = TILE_BLOCKS * (QK_K / 16);
    constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;
    constexpr int word_plane_stride = TILE_BLOCKS * 4;

    const uint8_t * x_base = x_coal;
    const int x_row_stride = tiles_per_row * tile_total;
    const sycl::half * x_d = (const sycl::half *)(x_base + nrows * x_row_stride);
    const int8_t * y_qs = (const int8_t *) y_reordered;
    const sycl::half2 * y_ds = (const sycl::half2 *) (y_reordered + ncols);

    auto dp4a = [](int a, int b, int c) {
        const int8_t * ap = (const int8_t *) &a;
        const int8_t * bp = (const int8_t *) &b;
        int sum = c;
        for (int i = 0; i < 4; ++i) {
            sum += (int)ap[i] * (int)bp[i];
        }
        return sum;
    };
    auto sub_32 = [](int x) {
        int out = 0;
        for (int i = 0; i < 4; ++i) {
            const int8_t v = (int8_t)((x >> (i * 8)) & 0xFF);
            const int8_t r = (int8_t)(v - 32);
            out |= ((int)(uint8_t)r) << (i * 8);
        }
        return out;
    };

    float sum = 0.0f;
    for (int tile = 0; tile < tiles_per_row; ++tile) {
        const int tile_base = row * x_row_stride + tile * tile_total;
        const uint8_t * tile_ql = x_base + tile_base;
        const uint8_t * tile_qh = tile_ql + ql_tile_bytes;
        const int8_t * tile_sc = (const int8_t *)(tile_qh + qh_tile_bytes);

        for (int block_in_tile = 0; block_in_tile < TILE_BLOCKS; ++block_in_tile) {
            const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
            const float d = half_to_float(x_d[block_idx]);
            const int y_block_base = (tile * TILE_BLOCKS + block_in_tile) * 8;

            for (int iqs = 0; iqs < QI6_K; ++iqs) {
                const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
                const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
                const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
                const int u_index = iqs % QI8_1;

                const int ql_offset = iqs * word_plane_stride + block_in_tile * 4;
                const int qh_word_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
                const int qh_offset = qh_word_idx * word_plane_stride + block_in_tile * 4;
                const int vl = *((const int *)(tile_ql + ql_offset));
                const int vh_raw = *((const int *)(tile_qh + qh_offset));
                const int vh = vh_raw >> vh_shift;

                const int sc_idx0 = scale_offset;
                const int sc_idx1 = scale_offset + 4;
                const int sc_word0 = sc_idx0 / 4;
                const int sc_byte0 = sc_idx0 % 4;
                const int sc_word1 = sc_idx1 / 4;
                const int sc_byte1 = sc_idx1 % 4;
                const int sc_offset0 = sc_word0 * word_plane_stride + block_in_tile * 4 + sc_byte0;
                const int sc_offset1 = sc_word1 * word_plane_stride + block_in_tile * 4 + sc_byte1;
                const int8_t sc0 = tile_sc[sc_offset0];
                const int8_t sc1 = tile_sc[sc_offset1];

                const int y_block0 = y_block_base + bq8_offset;
                const int y_block1 = y_block0 + 2;
                const int u0 = get_int_from_int8_aligned(y_qs + y_block0 * QK8_1, u_index);
                const int u1 = get_int_from_int8_aligned(y_qs + y_block1 * QK8_1, u_index);
                const sycl::half2 ds0 = y_ds[y_block0];
                const sycl::half2 ds1 = y_ds[y_block1];
                const float d8_0 = (float) ds0.x();
                const float d8_1 = (float) ds1.x();

                float sumf = 0.0f;
                const int vil0 = (vl >> 0) & 0x0F0F0F0F;
                const int vih0 = ((vh >> 0) << 4) & 0x30303030;
                const int vi0 = sub_32(vil0 | vih0);
                sumf += d8_0 * (dp4a(vi0, u0, 0) * sc0);

                const int vil1 = (vl >> 4) & 0x0F0F0F0F;
                const int vih1 = ((vh >> 4) << 4) & 0x30303030;
                const int vi1 = sub_32(vil1 | vih1);
                sumf += d8_1 * (dp4a(vi1, u1, 0) * sc1);

                sum += d * sumf;
            }
        }
    }
    return sum;
}

static bool run_q4_0_coalesced_test(sycl::queue & q) {
    const int nrows = 4;
    const int ncols = 1024;
    const int blocks_per_row = ncols / QK4_0;

    std::vector<float> x_f(nrows * ncols);
    std::vector<float> y_f(ncols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float & v : x_f) v = dist(rng);
    for (float & v : y_f) v = dist(rng);

    std::vector<block_q4_0> x_aos(nrows * blocks_per_row);
    for (int ib = 0; ib < (int)x_aos.size(); ++ib) {
        x_aos[ib].d = sycl::half(0.5f);
        for (int i = 0; i < QK4_0 / 2; ++i) {
            const uint8_t lo = (uint8_t)(rng() % 16);
            const uint8_t hi = (uint8_t)(rng() % 16);
            x_aos[ib].qs[i] = (hi << 4) | lo;
        }
    }

    std::vector<block_q8_1> y_q8(blocks_per_row);
    for (int ib = 0; ib < blocks_per_row; ++ib) {
        int sum = 0;
        for (int i = 0; i < QK8_1; ++i) {
            const int8_t v = (int8_t)(rng() % 255 - 127);
            y_q8[ib].qs[i] = v;
            sum += v;
        }
        const float d = 0.01f;
        y_q8[ib].d = sycl::half(d);
        y_q8[ib].s = sycl::half(d * sum);
    }
    std::vector<uint8_t> y_q8_reordered;
    reorder_q8_1_to_qs_ds(y_q8, y_q8_reordered);

    std::vector<uint8_t> x_soa;
    reorder_q4_0_to_soa(x_aos, ncols, nrows, x_soa);
    std::vector<uint8_t> x_coal;
    coalesce_q4_0_qs(x_soa, ncols, nrows, x_coal);

    std::vector<float> ref(nrows);
    for (int r = 0; r < nrows; ++r) {
        ref[r] = cpu_row_dot_q4_0(&x_aos[r * blocks_per_row], y_q8.data(), ncols);
    }

    for (int r = 0; r < nrows; ++r) {
        const float ref_coal = cpu_row_dot_q4_0_coalesced(x_coal.data(), y_q8_reordered.data(), ncols, nrows, r);
        const float diff = std::abs(ref_coal - ref[r]);
        const float denom = std::max(1.0f, std::abs(ref[r]));
        if (diff / denom > 1e-5f) {
            printf("Q4_0 coalesced CPU mismatch row %d: coal=%f ref=%f\n", r, ref_coal, ref[r]);
            return false;
        }
    }

    uint8_t * d_x = sycl::malloc_device<uint8_t>(x_coal.size(), q);
    uint8_t * d_y = sycl::malloc_device<uint8_t>(y_q8_reordered.size(), q);
    float * d_out = sycl::malloc_device<float>(nrows, q);

    q.memcpy(d_x, x_coal.data(), x_coal.size()).wait();
    q.memcpy(d_y, y_q8_reordered.data(), y_q8_reordered.size()).wait();

    constexpr int num_subgroups = 16;
    const int block_num_y = ceil_div(nrows, num_subgroups);
    const sycl::range<3> global_size(1, 1, block_num_y * num_subgroups * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, 1, num_subgroups * WARP_SIZE);

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const auto sg = nd_item.get_sub_group();
            const int sg_range = sg.get_group_linear_range();
            const int workgroup_id = nd_item.get_group_linear_id();
            const int sg_id = sg.get_group_linear_id();
            const int lane_id = sg.get_local_linear_id();
            const int row = workgroup_id * sg_range + sg_id;
            if (row >= nrows) return;

            constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
            const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
            const uint8_t * x_qs = d_x;
            const int x_row_stride = ncols / 2;
            const sycl::half * x_d = (const sycl::half *)(x_qs + nrows * x_row_stride);

            const int8_t * y_qs = (const int8_t *) d_y;
            const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)d_y + ncols);

            float partial_sum = 0.0f;
            for (int tile = 0; tile < tiles_per_row; ++tile) {
                const int tile_base = row * x_row_stride + tile * (TILE_BLOCKS * 16);
                const int word_stride = TILE_BLOCKS * 4;

                for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
                    const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                    const float d4 = half_to_float(x_d[block_idx]);
                    const int y_block = tile * TILE_BLOCKS + block_in_tile;
                    const int y_base = y_block * QK8_1;

                    for (int half = 0; half < 2; ++half) {
                        const int word_base = half * (2 * word_stride);
                        const int word0_offset = word_base + block_in_tile * 4;
                        const int word1_offset = word_base + word_stride + block_in_tile * 4;

                        const int v0 = *((const int *)(x_qs + tile_base + word0_offset));
                        const int v1 = *((const int *)(x_qs + tile_base + word1_offset));

                        const int y_offset = half * 8;
                        const int u0 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4);
                        const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 4);
                        const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);
                        const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 5);

                        const int vi0_0 = (v0 >> 0) & 0x0F0F0F0F;
                        const int vi1_0 = (v0 >> 4) & 0x0F0F0F0F;
                        const int vi0_1 = (v1 >> 0) & 0x0F0F0F0F;
                        const int vi1_1 = (v1 >> 4) & 0x0F0F0F0F;

                        auto dp4a = [](int a, int b, int c) {
                            const int8_t * ap = (const int8_t *) &a;
                            const int8_t * bp = (const int8_t *) &b;
                            int sum = c;
                            for (int i = 0; i < 4; ++i) {
                                sum += (int)ap[i] * (int)bp[i];
                            }
                            return sum;
                        };

                        int sumi = 0;
                        sumi = dp4a(vi0_0, u0, sumi);
                        sumi = dp4a(vi1_0, u1, sumi);
                        sumi = dp4a(vi0_1, u2, sumi);
                        sumi = dp4a(vi1_1, u3, sumi);

                        const sycl::half2 ds8 = y_ds[y_block];
                        const float d8 = (float) ds8.x();
                        const float s8 = (float) ds8.y();
                        partial_sum += d4 * (sumi * d8 - 4.0f * s8);
                    }
                }
            }

            const float sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());
            if (sg.leader()) {
                d_out[row] = sum;
            }
        });
    }).wait();

    std::vector<float> out(nrows);
    q.memcpy(out.data(), d_out, nrows * sizeof(float)).wait();

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    sycl::free(d_out, q);

    bool pass = true;
    for (int r = 0; r < nrows; ++r) {
        const float diff = std::abs(out[r] - ref[r]);
        const float denom = std::max(1.0f, std::abs(ref[r]));
        if (diff / denom > 1e-2f) {
            printf("Q4_0 coalesced mismatch row %d: gpu=%f ref=%f\n", r, out[r], ref[r]);
            pass = false;
        }
    }
    return pass;
}

static bool run_q6_k_coalesced_test(sycl::queue & q) {
    const int nrows = 2;
    const int ncols = 8192;
    const int blocks_per_row = ncols / QK_K;

    std::vector<float> x_f(nrows * ncols);
    std::vector<float> y_f(ncols);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float & v : x_f) v = dist(rng);
    for (float & v : y_f) v = dist(rng);

    std::vector<block_q6_K> x_aos(nrows * blocks_per_row);
    for (int ib = 0; ib < (int)x_aos.size(); ++ib) {
        x_aos[ib].d = sycl::half(0.25f);
        for (int i = 0; i < QK_K / 2; ++i) x_aos[ib].ql[i] = (uint8_t) (rng() & 0xFF);
        for (int i = 0; i < QK_K / 4; ++i) x_aos[ib].qh[i] = (uint8_t) (rng() & 0xFF);
        for (int i = 0; i < QK_K / 16; ++i) x_aos[ib].scales[i] = (int8_t)(rng() % 31 - 15);
    }

    std::vector<block_q8_1> y_q8(blocks_per_row * (QK_K / QK8_1));
    for (int ib = 0; ib < (int)y_q8.size(); ++ib) {
        int sum = 0;
        for (int i = 0; i < QK8_1; ++i) {
            const int8_t v = (int8_t)(rng() % 255 - 127);
            y_q8[ib].qs[i] = v;
            sum += v;
        }
        const float d = 0.02f;
        y_q8[ib].d = sycl::half(d);
        y_q8[ib].s = sycl::half(d * sum);
    }
    std::vector<uint8_t> y_q8_reordered;
    reorder_q8_1_to_qs_ds(y_q8, y_q8_reordered);

    std::vector<uint8_t> x_coal;
    coalesce_q6_k_from_aos(x_aos, ncols, nrows, x_coal);

    std::vector<float> ref(nrows);
    for (int r = 0; r < nrows; ++r) {
        ref[r] = cpu_row_dot_q6_K(&x_aos[r * blocks_per_row], y_q8.data(), ncols);
    }

    for (int r = 0; r < nrows; ++r) {
        const float ref_coal = cpu_row_dot_q6_k_coalesced(x_coal.data(), y_q8_reordered.data(), ncols, nrows, r);
        const float diff = std::abs(ref_coal - ref[r]);
        const float denom = std::max(1.0f, std::abs(ref[r]));
        if (diff / denom > 1e-5f) {
            printf("Q6_K coalesced CPU mismatch row %d: coal=%f ref=%f\n", r, ref_coal, ref[r]);
            return false;
        }
    }

    uint8_t * d_x = sycl::malloc_device<uint8_t>(x_coal.size(), q);
    uint8_t * d_y = sycl::malloc_device<uint8_t>(y_q8_reordered.size(), q);
    float * d_out = sycl::malloc_device<float>(nrows, q);

    q.memcpy(d_x, x_coal.data(), x_coal.size()).wait();
    q.memcpy(d_y, y_q8_reordered.data(), y_q8_reordered.size()).wait();

    constexpr int num_subgroups = 16;
    const int block_num_y = ceil_div(nrows, num_subgroups);
    const sycl::range<3> global_size(1, 1, block_num_y * num_subgroups * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, 1, num_subgroups * WARP_SIZE);

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const auto sg = nd_item.get_sub_group();
            const int sg_range = sg.get_group_linear_range();
            const int workgroup_id = nd_item.get_group_linear_id();
            const int sg_id = sg.get_group_linear_id();
            const int lane_id = sg.get_local_linear_id();
            const int row = workgroup_id * sg_range + sg_id;
            if (row >= nrows) return;

            constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
            constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;
            constexpr int ql_tile_bytes = TILE_BLOCKS * (QK_K / 2);
            constexpr int qh_tile_bytes = TILE_BLOCKS * (QK_K / 4);
            constexpr int sc_tile_bytes = TILE_BLOCKS * (QK_K / 16);
            constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;

            const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
            const uint8_t * x_base = d_x;
            const int x_row_stride = tiles_per_row * tile_total;
            const sycl::half * x_d = (const sycl::half *)(x_base + nrows * x_row_stride);
            const int8_t * y_qs = (const int8_t *) d_y;
            const sycl::half2 * y_ds = (const sycl::half2 *)((const char *)d_y + ncols);

            float partial_sum = 0.0f;
            for (int tile = 0; tile < tiles_per_row; ++tile) {
                const int tile_base = row * x_row_stride + tile * tile_total;
                const uint8_t * tile_ql = x_base + tile_base;
                const uint8_t * tile_qh = tile_ql + ql_tile_bytes;
                const int8_t * tile_sc = (const int8_t *)(tile_qh + qh_tile_bytes);

                for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
                    const int block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                    const float d = half_to_float(x_d[block_idx]);

                    const int y_block_base = (tile * TILE_BLOCKS + block_in_tile) * 8;
                    for (int iqs = 0; iqs < QI6_K; ++iqs) {
                        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
                        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
                        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
                        const int u_index = iqs % QI8_1;

                        const int ql_offset = iqs * WORD_PLANE_STRIDE + block_in_tile * 4;
                        const int qh_word_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
                        const int qh_offset = qh_word_idx * WORD_PLANE_STRIDE + block_in_tile * 4;
                        const int vl = *((const int *)(tile_ql + ql_offset));
                        const int vh_raw = *((const int *)(tile_qh + qh_offset));
                        const int vh = vh_raw >> vh_shift;

                        const int sc_idx0 = scale_offset;
                        const int sc_idx1 = scale_offset + 4;
                        const int sc_word0 = sc_idx0 / 4;
                        const int sc_byte0 = sc_idx0 % 4;
                        const int sc_word1 = sc_idx1 / 4;
                        const int sc_byte1 = sc_idx1 % 4;
                        const int sc_offset0 = sc_word0 * WORD_PLANE_STRIDE + block_in_tile * 4 + sc_byte0;
                        const int sc_offset1 = sc_word1 * WORD_PLANE_STRIDE + block_in_tile * 4 + sc_byte1;
                        const int8_t sc0 = tile_sc[sc_offset0];
                        const int8_t sc1 = tile_sc[sc_offset1];

                        const int y_block0 = y_block_base + bq8_offset;
                        const int y_block1 = y_block0 + 2;
                        const int u0 = get_int_from_int8_aligned(y_qs + y_block0 * QK8_1, u_index);
                        const int u1 = get_int_from_int8_aligned(y_qs + y_block1 * QK8_1, u_index);
                        const sycl::half2 ds0 = y_ds[y_block0];
                        const sycl::half2 ds1 = y_ds[y_block1];
                        const float d8_0 = (float) ds0.x();
                        const float d8_1 = (float) ds1.x();

                        auto dp4a = [](int a, int b, int c) {
                            const int8_t * ap = (const int8_t *) &a;
                            const int8_t * bp = (const int8_t *) &b;
                            int sum = c;
                            for (int i = 0; i < 4; ++i) {
                                sum += (int)ap[i] * (int)bp[i];
                            }
                            return sum;
                        };
                        auto sub_32 = [](int x) {
                            int out = 0;
                            for (int i = 0; i < 4; ++i) {
                                const int8_t v = (int8_t)((x >> (i * 8)) & 0xFF);
                                const int8_t r = (int8_t)(v - 32);
                                out |= ((int)(uint8_t)r) << (i * 8);
                            }
                            return out;
                        };

                        const int vil0 = (vl >> 0) & 0x0F0F0F0F;
                        const int vih0 = ((vh >> 0) << 4) & 0x30303030;
                        const int vi0 = sub_32(vil0 | vih0);
                        const int sum0 = dp4a(vi0, u0, 0);

                        const int vil1 = (vl >> 4) & 0x0F0F0F0F;
                        const int vih1 = ((vh >> 4) << 4) & 0x30303030;
                        const int vi1 = sub_32(vil1 | vih1);
                        const int sum1 = dp4a(vi1, u1, 0);

                        partial_sum += d * (d8_0 * (sum0 * sc0) + d8_1 * (sum1 * sc1));
                    }
                }
            }

            const float sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());
            if (sg.leader()) {
                d_out[row] = sum;
            }
        });
    }).wait();

    std::vector<float> out(nrows);
    q.memcpy(out.data(), d_out, nrows * sizeof(float)).wait();

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    sycl::free(d_out, q);

    bool pass = true;
    for (int r = 0; r < nrows; ++r) {
        const float diff = std::abs(out[r] - ref[r]);
        const float denom = std::max(1.0f, std::abs(ref[r]));
        if (diff / denom > 1e-2f) {
            printf("Q6_K coalesced mismatch row %d: gpu=%f ref=%f\n", r, out[r], ref[r]);
            pass = false;
        }
    }
    return pass;
}

int main() {
    sycl::queue q;
    bool pass = true;

    printf("Running Q4_0 coalesced MMVQ test...\n");
    if (!run_q4_0_coalesced_test(q)) {
        pass = false;
    } else {
        printf("Q4_0 coalesced test PASS\n");
    }

    printf("Running Q6_K coalesced MMVQ test...\n");
    if (!run_q6_k_coalesced_test(q)) {
        pass = false;
    } else {
        printf("Q6_K coalesced test PASS\n");
    }

    if (!pass) {
        return 1;
    }
    printf("All coalesced MMVQ tests PASS\n");
    return 0;
}
