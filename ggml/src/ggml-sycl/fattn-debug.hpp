//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_DEBUG_HPP
#define GGML_SYCL_FATTN_DEBUG_HPP

#include <sys/stat.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

// Environment variable to enable FA debug dumping
// Set GGML_SYCL_FA_DEBUG=1 to enable
// Set GGML_SYCL_FA_DEBUG=2 for verbose mode (dumps all heads)
inline int fattn_debug_level() {
    static int level = -1;
    if (level < 0) {
        const char * env = getenv("GGML_SYCL_FA_DEBUG");
        level            = env ? atoi(env) : 0;
        if (level > 0) {
            // Create debug directory
            mkdir("/tmp/fa_debug", 0755);
            fprintf(stderr, "[FA-DEBUG] Debug level %d enabled, output to /tmp/fa_debug/\n", level);
        }
    }
    return level;
}

// Debug context to track FA calls
struct FattnDebugContext {
    int   call_id;     // Sequential call number
    int   n_queries;   // ne01
    int   n_heads;     // ne02
    int   n_kv_heads;  // ne12
    int   n_kv;        // ne11 (sequence length)
    int   D;           // head dimension
    float scale;
    bool  is_fa_on;    // true for FA path, false for standard attention

    // File for this call's output
    FILE * file;

    FattnDebugContext() : call_id(0), file(nullptr) {}

    void open_file(const char * suffix) {
        if (fattn_debug_level() <= 0) {
            return;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "/tmp/fa_debug/fa_%s_call%04d.txt", suffix, call_id);
        file = fopen(filename, "w");
        if (file) {
            fprintf(file, "# FA Debug Dump - Call %d\n", call_id);
            fprintf(file, "# Mode: %s\n", is_fa_on ? "FA_ON" : "FA_OFF");
            fprintf(file, "# n_queries=%d, n_heads=%d, n_kv_heads=%d, n_kv=%d, D=%d, scale=%.6f\n", n_queries, n_heads,
                    n_kv_heads, n_kv, D, scale);
            fprintf(file, "#\n");
        }
    }

    void close_file() {
        if (file) {
            fclose(file);
            file = nullptr;
        }
    }
};

// Global debug context (thread-local for safety)
inline FattnDebugContext & get_fattn_debug_ctx() {
    thread_local FattnDebugContext ctx;
    return ctx;
}

// Dump Q tensor values
inline void fattn_debug_dump_Q(dpct::queue_ptr stream,
                               const char *    Q_ptr,
                               int             Q_type_size,  // sizeof(float) or sizeof(half)
                               int             D,
                               int             n_queries,
                               int             n_heads,
                               int             nb01,
                               int             nb02,
                               float           scale) {
    if (fattn_debug_level() <= 0) {
        return;
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== Q INPUT (scaled by %.6f) ===\n", scale);

    // Only dump first 2 heads in level 1, all heads in level 2
    int heads_to_dump = (fattn_debug_level() >= 2) ? n_heads : std::min(2, n_heads);

    std::vector<float> host_Q(D);

    for (int h = 0; h < heads_to_dump; h++) {
        for (int q = 0; q < n_queries; q++) {
            // Copy Q[h][q] to host
            const char * q_row = Q_ptr + nb02 * h + nb01 * q;

            if (Q_type_size == sizeof(float)) {
                stream->memcpy(host_Q.data(), q_row, D * sizeof(float)).wait();
            } else {
                // Half precision - need to convert
                std::vector<sycl::half> host_Q_h(D);
                stream->memcpy(host_Q_h.data(), q_row, D * sizeof(sycl::half)).wait();
                for (int d = 0; d < D; d++) {
                    host_Q[d] = static_cast<float>(host_Q_h[d]);
                }
            }

            fprintf(ctx.file, "Q[h=%d,q=%d]: ", h, q);
            // Dump first 8 and last 4 values
            for (int d = 0; d < std::min(8, D); d++) {
                fprintf(ctx.file, "%.4f ", host_Q[d] * scale);
            }
            if (D > 12) {
                fprintf(ctx.file, "... ");
                for (int d = D - 4; d < D; d++) {
                    fprintf(ctx.file, "%.4f ", host_Q[d] * scale);
                }
            }
            fprintf(ctx.file, "\n");
        }
    }
}

// Dump K tensor values (first few KV positions)
inline void
fattn_debug_dump_K(dpct::queue_ptr stream, const char * K_ptr, int D, int n_kv, int n_kv_heads, int nb11, int nb12) {
    if (fattn_debug_level() <= 0) {
        return;
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== K INPUT (first %d positions) ===\n", std::min(16, n_kv));

    int kv_heads_to_dump = (fattn_debug_level() >= 2) ? n_kv_heads : std::min(2, n_kv_heads);
    int kv_to_dump       = std::min(16, n_kv);

    std::vector<sycl::half> host_K(D);

    for (int kv_h = 0; kv_h < kv_heads_to_dump; kv_h++) {
        for (int kv = 0; kv < kv_to_dump; kv++) {
            const sycl::half * k_row = reinterpret_cast<const sycl::half *>(K_ptr + nb12 * kv_h + nb11 * kv);
            stream->memcpy(host_K.data(), k_row, D * sizeof(sycl::half)).wait();

            fprintf(ctx.file, "K[kv_h=%d,kv=%d]: ", kv_h, kv);
            for (int d = 0; d < std::min(8, D); d++) {
                fprintf(ctx.file, "%.4f ", static_cast<float>(host_K[d]));
            }
            if (D > 12) {
                fprintf(ctx.file, "... ");
                for (int d = D - 4; d < D; d++) {
                    fprintf(ctx.file, "%.4f ", static_cast<float>(host_K[d]));
                }
            }
            fprintf(ctx.file, "\n");
        }
    }
}

// Dump V tensor values (first few KV positions)
inline void
fattn_debug_dump_V(dpct::queue_ptr stream, const char * V_ptr, int D, int n_kv, int n_kv_heads, int nb21, int nb22) {
    if (fattn_debug_level() <= 0) {
        return;
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== V INPUT (first %d positions) ===\n", std::min(16, n_kv));

    int kv_heads_to_dump = (fattn_debug_level() >= 2) ? n_kv_heads : std::min(2, n_kv_heads);
    int kv_to_dump       = std::min(16, n_kv);

    std::vector<sycl::half> host_V(D);

    for (int kv_h = 0; kv_h < kv_heads_to_dump; kv_h++) {
        for (int kv = 0; kv < kv_to_dump; kv++) {
            const sycl::half * v_row = reinterpret_cast<const sycl::half *>(V_ptr + nb22 * kv_h + nb21 * kv);
            stream->memcpy(host_V.data(), v_row, D * sizeof(sycl::half)).wait();

            fprintf(ctx.file, "V[kv_h=%d,kv=%d]: ", kv_h, kv);
            for (int d = 0; d < std::min(8, D); d++) {
                fprintf(ctx.file, "%.4f ", static_cast<float>(host_V[d]));
            }
            if (D > 12) {
                fprintf(ctx.file, "... ");
                for (int d = D - 4; d < D; d++) {
                    fprintf(ctx.file, "%.4f ", static_cast<float>(host_V[d]));
                }
            }
            fprintf(ctx.file, "\n");
        }
    }
}

// Dump mask values
inline void fattn_debug_dump_mask(dpct::queue_ptr stream,
                                  const char *    mask_ptr,
                                  int             n_kv,
                                  int             n_queries,
                                  int             nb31,
                                  int             ne30) {
    if (fattn_debug_level() <= 0 || !mask_ptr) {
        return;
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== MASK (first %d KV positions) ===\n", std::min(32, n_kv));

    int                     kv_to_dump = std::min(32, n_kv);
    std::vector<sycl::half> host_mask(kv_to_dump);

    for (int q = 0; q < n_queries; q++) {
        const sycl::half * mask_row = reinterpret_cast<const sycl::half *>(mask_ptr + nb31 * q);
        stream->memcpy(host_mask.data(), mask_row, kv_to_dump * sizeof(sycl::half)).wait();

        fprintf(ctx.file, "mask[q=%d]: ", q);
        for (int kv = 0; kv < kv_to_dump; kv++) {
            float mv = static_cast<float>(host_mask[kv]);
            if (mv < -1e10f) {
                fprintf(ctx.file, "-inf ");
            } else {
                fprintf(ctx.file, "%.1f ", mv);
            }
        }
        fprintf(ctx.file, "\n");
    }
}

// Dump attention output
inline void fattn_debug_dump_output(dpct::queue_ptr stream, const float * dst_ptr, int D, int n_queries, int n_heads) {
    if (fattn_debug_level() <= 0) {
        return;
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== FA OUTPUT ===\n");

    int heads_to_dump = (fattn_debug_level() >= 2) ? n_heads : std::min(4, n_heads);

    // Output layout: dst[d + D*(h + n_heads*q)]
    size_t             output_size = (size_t) D * n_queries * n_heads;
    std::vector<float> host_dst(output_size);
    stream->memcpy(host_dst.data(), dst_ptr, output_size * sizeof(float)).wait();

    for (int h = 0; h < heads_to_dump; h++) {
        for (int q = 0; q < n_queries; q++) {
            fprintf(ctx.file, "out[h=%d,q=%d]: ", h, q);

            // First 8 values
            for (int d = 0; d < std::min(8, D); d++) {
                float val = host_dst[d + D * (h + n_heads * q)];
                fprintf(ctx.file, "%.6f ", val);
            }
            if (D > 12) {
                fprintf(ctx.file, "... ");
                // Last 4 values
                for (int d = D - 4; d < D; d++) {
                    float val = host_dst[d + D * (h + n_heads * q)];
                    fprintf(ctx.file, "%.6f ", val);
                }
            }
            fprintf(ctx.file, "\n");
        }
    }

    // Also compute and dump statistics per head
    fprintf(ctx.file, "\n=== OUTPUT STATISTICS ===\n");
    for (int h = 0; h < heads_to_dump; h++) {
        float sum = 0, sum_sq = 0, min_val = FLT_MAX, max_val = -FLT_MAX;
        for (int q = 0; q < n_queries; q++) {
            for (int d = 0; d < D; d++) {
                float val = host_dst[d + D * (h + n_heads * q)];
                sum += val;
                sum_sq += val * val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
        int   count = n_queries * D;
        float mean  = sum / count;
        float var   = sum_sq / count - mean * mean;
        fprintf(ctx.file, "head[%d]: mean=%.6f, var=%.6f, min=%.6f, max=%.6f\n", h, mean, var, min_val, max_val);
    }
}

// Dump attention scores (QK) - for detailed debugging
inline void fattn_debug_dump_QK(dpct::queue_ptr stream,
                                const float *   QK_ptr,  // [n_queries][n_kv] or portion thereof
                                int             n_queries,
                                int             n_kv_batch,
                                int             kv_start,
                                int             head) {
    if (fattn_debug_level() < 2) {
        return;  // Only in verbose mode
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== QK SCORES [head=%d, kv_start=%d] ===\n", head, kv_start);

    std::vector<float> host_QK(n_queries * n_kv_batch);
    stream->memcpy(host_QK.data(), QK_ptr, host_QK.size() * sizeof(float)).wait();

    for (int q = 0; q < n_queries; q++) {
        fprintf(ctx.file, "QK[q=%d]: ", q);
        int kv_to_dump = std::min(16, n_kv_batch);
        for (int k = 0; k < kv_to_dump; k++) {
            fprintf(ctx.file, "%.4f ", host_QK[q * n_kv_batch + k]);
        }
        if (n_kv_batch > kv_to_dump) {
            fprintf(ctx.file, "...");
        }
        fprintf(ctx.file, "\n");
    }
}

// Dump softmax weights
inline void fattn_debug_dump_softmax(dpct::queue_ptr    stream,
                                     const sycl::half * S_ptr,  // [n_queries][n_kv_batch]
                                     int                n_queries,
                                     int                n_kv_batch,
                                     int                kv_start,
                                     int                head,
                                     int                S_stride) {
    if (fattn_debug_level() < 2) {
        return;  // Only in verbose mode
    }

    auto & ctx = get_fattn_debug_ctx();
    if (!ctx.file) {
        return;
    }

    fprintf(ctx.file, "\n=== SOFTMAX WEIGHTS [head=%d, kv_start=%d] ===\n", head, kv_start);

    std::vector<sycl::half> host_S(n_queries * S_stride);
    stream->memcpy(host_S.data(), S_ptr, host_S.size() * sizeof(sycl::half)).wait();

    for (int q = 0; q < n_queries; q++) {
        fprintf(ctx.file, "S[q=%d]: ", q);
        float sum        = 0;
        int   kv_to_dump = std::min(16, n_kv_batch);
        for (int k = 0; k < kv_to_dump; k++) {
            float w = static_cast<float>(host_S[q * S_stride + k]);
            fprintf(ctx.file, "%.4f ", w);
            sum += w;
        }
        // Also show sum to verify it's close to 1.0
        for (int k = kv_to_dump; k < n_kv_batch; k++) {
            sum += static_cast<float>(host_S[q * S_stride + k]);
        }
        fprintf(ctx.file, " (sum=%.4f)\n", sum);
    }
}

#endif  // GGML_SYCL_FATTN_DEBUG_HPP
