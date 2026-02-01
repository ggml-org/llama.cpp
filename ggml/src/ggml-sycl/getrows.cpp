//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "getrows.hpp"

#include "common.hpp"
#include "dequantize.hpp"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static const ggml_tensor * get_storage_tensor(const ggml_tensor * t) {
    const ggml_tensor * current = t;
    while (current && current->view_src != nullptr) {
        current = current->view_src;
    }
    return current;
}

static size_t get_view_byte_offset(const ggml_tensor * t) {
    size_t offset = 0;
    const ggml_tensor * current = t;
    while (current && current->view_src != nullptr) {
        offset += current->view_offs;
        current = current->view_src;
    }
    return offset;
}

static int64_t get_view_row_offset(const ggml_tensor * t) {
    int64_t offset = 0;
    const ggml_tensor * current = t;
    while (current && current->view_src != nullptr) {
        if (current->nb[1] > 0) {
            offset += static_cast<int64_t>(current->view_offs / current->nb[1]);
        }
        current = current->view_src;
    }
    return offset;
}

static bool ggml_sycl_debug_getrows_tokens_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_DEBUG_GET_ROWS_TOKENS");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static const char * ggml_sycl_layout_mode_name_local(layout_mode mode) {
    switch (mode) {
        case GGML_LAYOUT_AOS:         return "aos";
        case GGML_LAYOUT_SOA:         return "soa";
        case GGML_LAYOUT_COALESCED:   return "coalesced";
        case GGML_LAYOUT_XMX_TILED:   return "xmx_tiled";
        case GGML_LAYOUT_XMX_GEMM_TILED: return "xmx_gemm_tiled";
        case GGML_LAYOUT_ONEDNN_PACKED: return "onednn_packed";
        default:                      return "unknown";
    }
}

static const char * ggml_sycl_usm_alloc_name(sycl::usm::alloc alloc) {
    switch (alloc) {
        case sycl::usm::alloc::host:   return "host";
        case sycl::usm::alloc::shared: return "shared";
        case sycl::usm::alloc::device: return "device";
        default:                       return "unknown";
    }
}

static bool ggml_sycl_cpu_get_rows_direct(ggml_backend_sycl_context & ctx,
                                          ggml_tensor * dst,
                                          const void * src0_override,
                                          const char * reason) {
    if (!dst || !dst->src[0] || !dst->src[1]) {
        return false;
    }

    auto * stream = ctx.stream();
    if (!stream) {
        return false;
    }

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    auto ptr_is_device = [&](const void * ptr) -> bool {
        if (!ptr) {
            return false;
        }
        try {
            return sycl::get_pointer_type(const_cast<void *>(ptr), stream->get_context()) == sycl::usm::alloc::device;
        } catch (...) {
            return false;
        }
    };

    void * src0_orig = src0->data;
    void * src1_orig = src1->data;
    void * dst_orig  = dst->data;

    std::vector<uint8_t> src0_host;
    std::vector<uint8_t> src1_host;
    std::vector<uint8_t> dst_host;
    void *               src0_host_ptr   = nullptr;
    void *               src1_host_ptr   = nullptr;
    void *               dst_host_ptr    = nullptr;
    bool                 src0_host_pinned = false;
    bool                 src1_host_pinned = false;
    bool                 dst_host_pinned  = false;
    const size_t src0_bytes = ggml_nbytes(src0);
    const size_t src1_bytes = ggml_nbytes(src1);
    const size_t dst_bytes  = ggml_nbytes(dst);

    bool ok = false;
    try {
        if (src0_override) {
            src0->data = const_cast<void *>(src0_override);
        } else if (ptr_is_device(src0_orig)) {
            src0_host_ptr = ggml_sycl_host_malloc(src0_bytes);
            if (src0_host_ptr) {
                src0_host_pinned = true;
            } else {
                src0_host.resize(src0_bytes);
                src0_host_ptr = src0_host.data();
            }
            stream->memcpy(src0_host_ptr, src0_orig, src0_bytes).wait();
            src0->data = src0_host_ptr;
        }

        if (ptr_is_device(src1_orig)) {
            src1_host_ptr = ggml_sycl_host_malloc(src1_bytes);
            if (src1_host_ptr) {
                src1_host_pinned = true;
            } else {
                src1_host.resize(src1_bytes);
                src1_host_ptr = src1_host.data();
            }
            stream->memcpy(src1_host_ptr, src1_orig, src1_bytes).wait();
            src1->data = src1_host_ptr;
        }

        if (ptr_is_device(dst_orig)) {
            dst_host_ptr = ggml_sycl_host_malloc(dst_bytes);
            if (dst_host_ptr) {
                dst_host_pinned = true;
            } else {
                dst_host.resize(dst_bytes);
                dst_host_ptr = dst_host.data();
            }
            dst->data = dst_host_ptr;
        }

        ggml_compute_params params = {};
        params.ith        = 0;
        params.nth        = 1;
        params.wsize      = 0;
        params.wdata      = nullptr;
        params.threadpool = nullptr;

        GGML_LOG_WARN("[SYCL] CPU direct get_rows (%s)\n", reason ? reason : "unknown");
        ggml_compute_forward_get_rows(&params, dst);

        if (ptr_is_device(dst_orig)) {
            stream->memcpy(dst_orig, dst->data, dst_bytes).wait();
        }
        ok = true;
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[SYCL] CPU direct get_rows failed (%s): %s\n", reason ? reason : "unknown", e.what());
    } catch (...) {
        GGML_LOG_WARN("[SYCL] CPU direct get_rows failed (%s)\n", reason ? reason : "unknown");
    }

    src0->data = src0_orig;
    src1->data = src1_orig;
    dst->data  = dst_orig;
    if (src0_host_pinned && src0_host_ptr) {
        ggml_sycl_host_free(src0_host_ptr);
    }
    if (src1_host_pinned && src1_host_ptr) {
        ggml_sycl_host_free(src1_host_ptr);
    }
    if (dst_host_pinned && dst_host_ptr) {
        ggml_sycl_host_free(dst_host_ptr);
    }

    return ok;
}

static bool ggml_sycl_wait_after_get_rows_q6_k_soa() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = std::getenv("GGML_SYCL_WAIT_AFTER_GET_ROWS_Q6K");
        cached = (env && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

static bool ggml_sycl_get_rows_trace_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_GET_ROWS_TRACE");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    return enabled != 0;
}

struct get_rows_stream_segment {
    size_t src_base      = 0;
    size_t bytes_per_row = 0;
};

struct get_rows_stream_ctx {
    ggml_backend_sycl_context * backend_ctx     = nullptr;
    const ggml_tensor *         src0            = nullptr;
    ggml_tensor *               dst             = nullptr;
    const int32_t *             row_indices     = nullptr;
    size_t                      row_count       = 0;
    int64_t                     row_base        = 0;
    layout_mode                 layout          = GGML_LAYOUT_AOS;
    const uint8_t *             src_base        = nullptr;
    size_t                      row_total_bytes = 0;
    int                         segment_count   = 0;
    get_rows_stream_segment     segments[4]     = {};
    int                         device_id       = -1;
    const int32_t *             seq_device      = nullptr;
    size_t                      seq_count       = 0;
    float *                     dst_base        = nullptr;
    int64_t                     dst_row_stride  = 0;
};

static bool get_rows_parse_env_mb_value(const char * name, size_t & out_mb) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    long   mb  = std::strtol(env, &end, 10);
    if (end == env || mb < 0) {
        GGML_LOG_WARN("[GET_ROWS] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_mb = static_cast<size_t>(mb);
    return true;
}

static bool get_rows_parse_env_count_value(const char * name, size_t & out_count) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end   = nullptr;
    long   count = std::strtol(env, &end, 10);
    if (end == env || count < 0) {
        GGML_LOG_WARN("[GET_ROWS] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_count = static_cast<size_t>(count);
    return true;
}

static bool get_rows_is_device_oom(const sycl::exception & e) {
    const char * msg = e.what();
    if (!msg) {
        return false;
    }
    return std::strstr(msg, "OUT_OF_DEVICE_MEMORY") != nullptr ||
           std::strstr(msg, "OUT_OF_RESOURCES") != nullptr;
}

static size_t get_rows_min_slice_bytes(size_t row_bytes) {
    size_t min_mb = 0;
    if (!get_rows_parse_env_mb_value("GGML_SYCL_GET_ROWS_MIN_SLICE_MB", min_mb)) {
        min_mb = 4;
    }
    size_t bytes = min_mb * 1024ULL * 1024ULL;
    if (row_bytes > 0) {
        size_t rows = bytes / row_bytes;
        if (rows < 1) {
            rows = 1;
        }
        bytes = rows * row_bytes;
    }
    return bytes;
}

static void get_rows_resolve_dma_params(size_t row_bytes, size_t & slice_bytes, size_t & buffer_count) {
    size_t slice_mb = 1024;
    size_t buffers  = 2;
    size_t env_val  = 0;
    bool   slice_env_set = false;
    bool   buffers_env_set = false;

    slice_env_set = get_rows_parse_env_mb_value("GGML_SYCL_DMA_SLICE_MB", env_val);
    if (slice_env_set) {
        slice_mb = env_val;
    }
    buffers_env_set = get_rows_parse_env_count_value("GGML_SYCL_DMA_BUFFERS", env_val) ||
                      get_rows_parse_env_count_value("GGML_SYCL_DMA_SLICES", env_val);
    if (buffers_env_set) {
        buffers = env_val;
    }
    if (!slice_env_set && !buffers_env_set && ggml_backend_sycl_weights_evictable()) {
        // Use smaller defaults for evictable weights to reduce staging OOM risk.
        slice_mb = std::min<size_t>(slice_mb, 32);
        buffers  = std::min<size_t>(buffers, 1);
    }

    if (slice_bytes == 0) {
        slice_bytes = slice_mb * 1024ULL * 1024ULL;
    }
    if (buffer_count == 0) {
        buffer_count = buffers;
    }
    if (row_bytes > 0) {
        size_t rows_per_slice = slice_bytes / row_bytes;
        if (rows_per_slice < 1) {
            rows_per_slice = 1;
        }
        slice_bytes = rows_per_slice * row_bytes;
    }
}

static size_t get_rows_q6_k_coalesced_row_quants_bytes(int blocks_per_row) {
    size_t row_quants_bytes = 0;
    int    remaining        = blocks_per_row;
    while (remaining > 0) {
        int ts = 1;
        while (ts * 2 <= remaining && ts < 32) {
            ts *= 2;
        }
        row_quants_bytes += static_cast<size_t>(ts) * (128 + 64 + 16);
        remaining -= ts;
    }
    return row_quants_bytes;
}

static bool get_rows_build_stream_segments(const ggml_tensor * src0,
                                           layout_mode         layout,
                                           int64_t             ncols,
                                           int64_t             total_rows,
                                           get_rows_stream_ctx & ctx) {
    ctx.segment_count   = 0;
    ctx.row_total_bytes = 0;

    const auto add_segment = [&](size_t bytes_per_row) {
        GGML_ASSERT(ctx.segment_count < 4);
        ctx.segments[ctx.segment_count].src_base      = 0;
        ctx.segments[ctx.segment_count].bytes_per_row = bytes_per_row;
        ctx.segment_count++;
        ctx.row_total_bytes += bytes_per_row;
    };

    if (layout == GGML_LAYOUT_AOS) {
        const size_t row_bytes = ggml_row_size(src0->type, ncols);
        add_segment(row_bytes);
        return true;
    }

    const size_t blocks_per_row = static_cast<size_t>(ncols) / ggml_blck_size(src0->type);
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            {
                const size_t q_bytes = static_cast<size_t>(ncols) / 2;
                const size_t d_bytes = blocks_per_row * sizeof(ggml_half);
                add_segment(q_bytes);
                add_segment(d_bytes);
            }
            break;
        case GGML_TYPE_Q8_0:
            {
                const size_t q_bytes = static_cast<size_t>(ncols);
                const size_t d_bytes = blocks_per_row * sizeof(ggml_half);
                add_segment(q_bytes);
                add_segment(d_bytes);
            }
            break;
        case GGML_TYPE_Q6_K:
            {
                if (layout == GGML_LAYOUT_COALESCED) {
                    const size_t q_bytes = get_rows_q6_k_coalesced_row_quants_bytes(static_cast<int>(blocks_per_row));
                    const size_t d_bytes = blocks_per_row * sizeof(ggml_half);
                    add_segment(q_bytes);
                    add_segment(d_bytes);
                } else {
                    const size_t ql_bytes     = blocks_per_row * (QK_K / 2);
                    const size_t qh_bytes     = blocks_per_row * (QK_K / 4);
                    const size_t scales_bytes = blocks_per_row * (QK_K / 16);
                    const size_t d_bytes      = blocks_per_row * sizeof(ggml_half);
                    add_segment(ql_bytes);
                    add_segment(qh_bytes);
                    add_segment(scales_bytes);
                    add_segment(d_bytes);
                }
            }
            break;
        default:
            GGML_ABORT("GET_ROWS streaming: unsupported layout/type");
    }

    // Fix src_base offsets for non-AoS segments.
    size_t src_base = 0;
    for (int i = 0; i < ctx.segment_count; ++i) {
        ctx.segments[i].src_base = src_base;
        src_base += static_cast<size_t>(total_rows) * ctx.segments[i].bytes_per_row;
    }
    return true;
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(const void *                            src0,
                       const int32_t *                         src1,
                       dst_t *                                 dst,
                       int64_t                                 ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
                       /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
                       /*size_t s0,*/ size_t                   s1,
                       size_t                                  s2,
                       size_t                                  s3,
                       /*size_t nb00,*/ size_t                 nb01,
                       size_t                                  nb02,
                       size_t                                  nb03,
                       size_t                                  s10,
                       size_t                                  s11,
                       size_t                                  s12,
                       const sycl::nd_item<3> &                item_ct1 /*, size_t s13*/) {
    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

    dst_t *      dst_row  = dst + i10 * s1 + i11 * s2 + i12 * s3;
    const void * src0_row = (const char *) src0 + i01 * nb01 + i11 * nb02 + i12 * nb03;

    const int ib       = i00 / qk;         // block index
    const int iqs      = (i00 % qk) / qr;  // quant index
    const int iybs     = i00 - i00 % qk;   // dst block start index
    const int y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0]        = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

// SoA (Structure of Arrays) version of k_get_rows for reordered quantized tensors
// In SoA layout: all qs bytes come first, then all d (scale) bytes
// This requires computing separate pointers for qs and d data
template <int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder, typename dst_t>
static void k_get_rows_reorder(const void *             src0,
                               const int32_t *          src1,
                               dst_t *                  dst,
                               int64_t                  ne00,
                               int64_t                  ne01,
                               int64_t                  ne12,
                               size_t                   s1,
                               size_t                   s2,
                               size_t                   s3,
                               size_t                   nb01,
                               size_t                   nb02,
                               size_t                   nb03,
                               size_t                   s10,
                               size_t                   s11,
                               size_t                   s12,
                               int64_t                  row_offset,
                               int64_t                  d_offset,
                               const sycl::nd_item<3> & item_ct1) {
    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = static_cast<int64_t>(src1[i10 * s10 + i11 * s11 + i12 * s12]) + row_offset;

    dst_t * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

    // For SoA layout, compute block index within the full tensor
    const int64_t blocks_per_row = ne00 / qk;
    const int     ib_local       = i00 / qk;                         // block index within row
    const int64_t ib_global      = i01 * blocks_per_row + ib_local;  // global block index

    const int iqs      = (i00 % qk) / qr;                            // quant index within block
    const int iybs     = i00 - i00 % qk;                             // dst block start index
    const int y_offset = qr == 1 ? 1 : qk / 2;

    // In SoA layout:
    // - qs data is contiguous from start: offset = row * bytes_per_row + block_offset + elem_offset
    // - d data starts at d_offset: offset = d_offset + global_block_index * sizeof(half)
    // Byte sizes depend on quantization type:
    //   Q4_0 (qr=2): 4 bits/elem → ne00/2 bytes/row, qk/2 bytes/block
    //   Q8_0 (qr=1): 8 bits/elem → ne00 bytes/row, qk bytes/block
    const size_t bytes_per_row   = ne00 / qr;
    const size_t bytes_per_block = qk / qr;
    const void * d_ptr           = (const char *) src0 + d_offset;
    const void * qs_ptr          = (const char *) src0 + i01 * bytes_per_row + ib_local * bytes_per_block + iqs;

    // dequantize using SoA kernel
    dfloat2 v;
    dequantize_kernel_reorder(d_ptr, ib_global, qs_ptr, 0, v);

    dst_row[iybs + iqs + 0]        = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

// Specialized Q6_K AoS kernel for GET_ROWS (standard block layout)
template <typename dst_t>
static void k_get_rows_q6_k_aos(const void *             src0,
                                const int32_t *          src1,
                                dst_t *                  dst,
                                int64_t                  ne00,
                                int64_t                  ne01,
                                int64_t                  ne12,
                                size_t                   s1,
                                size_t                   s2,
                                size_t                   s3,
                                size_t                   nb01,
                                size_t                   nb02,
                                size_t                   nb03,
                                size_t                   s10,
                                size_t                   s11,
                                size_t                   s12,
                                const sycl::nd_item<3> & item_ct1) {
    // Each thread processes 4 values (Q6_K block structure)
    // Thread layout: tid = ip * 32 + il, where ip in {0,1}, il in {0..31}
    const int tid = item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    // Block index within row (each block has QK_K=256 elements)
    const int     block_in_row   = item_ct1.get_group(2);
    const int64_t blocks_per_row = ne00 / QK_K;

    if (block_in_row >= blocks_per_row) {
        return;
    }

    const int i01     = src1[i10 * s10 + i11 * s11 + i12 * s12];  // Row index
    dst_t *   dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

    // Get pointer to the Q6_K block (AoS layout)
    const block_q6_K * x  = (const block_q6_K *) ((const char *) src0 + i01 * nb01 + i11 * nb02 + i12 * nb03);
    const block_q6_K * bx = x + block_in_row;

    // Thread decomposition: ip (0 or 1), il (0..31)
    const int ip = tid / 32;
    const int il = tid % 32;
    const int is = 8 * ip + il / 16;

    // Destination position for this thread's 4 values
    dst_t * y = dst_row + block_in_row * QK_K + 128 * ip + il;

    // Read data for this thread from AoS block
    const uint8_t * ql = bx->ql + 64 * ip + il;
    const uint8_t   qh = bx->qh[32 * ip + il];
    const int8_t *  sc = bx->scales + is;
    const float     d  = static_cast<float>(bx->d);

    // Dequantize 4 values
    y[0]  = d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K AoS GET_ROWS
static void get_rows_q6_k_aos_sycl(ggml_backend_sycl_context & ctx,
                                   const ggml_tensor *         src0,
                                   const ggml_tensor *         src1,
                                   ggml_tensor *               dst,
                                   const void *                src0_dd,
                                   const int32_t *             src1_dd,
                                   float *                     dst_dd,
                                   queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K uses 64 threads (2 phases * 32 threads) per block
    const sycl::range<3> block_dims(1, 1, 64);
    // One work-group per Q6_K block in each row
    const int64_t        blocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, blocks_per_row);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        k_get_rows_q6_k_aos<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11,
                                   s12, item_ct1);
    });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Specialized Q6_K SoA kernel for GET_ROWS
// Q6_K has 4 sections: [all ql (n*128)][all qh (n*64)][all scales (n*16)][all d (n*2)]
template <typename dst_t>
static void k_get_rows_q6_k_soa(const void *             src0,
                                const int32_t *          src1,
                                dst_t *                  dst,
                                int64_t                  ne00,
                                int64_t                  ne01,
                                int64_t                  ne12,
                                size_t                   s1,
                                size_t                   s2,
                                size_t                   s3,
                                size_t                   nb01,
                                size_t                   nb02,
                                size_t                   nb03,
                                size_t                   s10,
                                size_t                   s11,
                                size_t                   s12,
                                int64_t                  row_offset,
                                int64_t                  total_nrows,
                                const sycl::nd_item<3> & item_ct1) {
    // Each thread processes 4 values (Q6_K block structure)
    // Thread layout: tid = ip * 32 + il, where ip in {0,1}, il in {0..31}
    const int tid = item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    // Block index within row (each block has QK_K=256 elements)
    const int     block_in_row   = item_ct1.get_group(2);
    const int64_t blocks_per_row = ne00 / QK_K;

    if (block_in_row >= blocks_per_row) {
        return;
    }

    const int64_t i01 = static_cast<int64_t>(src1[i10 * s10 + i11 * s11 + i12 * s12]) + row_offset;  // Row index
    dst_t *   dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

    // Global block index for SoA offset calculation
    const int64_t n_blocks  = total_nrows * blocks_per_row;  // Total blocks in tensor
    const int64_t ib_global = i01 * blocks_per_row + block_in_row;

    // Thread decomposition: ip (0 or 1), il (0..31)
    const int ip = tid / 32;
    const int il = tid % 32;
    const int is = 8 * ip + il / 16;

    // SoA layout offsets
    const uint8_t * base_ptr      = static_cast<const uint8_t *>(src0);
    const int64_t   ql_offset     = ib_global * (QK_K / 2);  // 128 bytes per block
    const int64_t   qh_offset     = (QK_K / 2) * n_blocks + (QK_K / 4) * ib_global;
    const int64_t   scales_offset = (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks + (QK_K / 16) * ib_global;
    const int64_t   d_offset      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks;

    const uint8_t *    ql_ptr     = base_ptr + ql_offset;
    const uint8_t *    qh_ptr     = base_ptr + qh_offset;
    const uint8_t *    scales_ptr = base_ptr + scales_offset;
    const sycl::half * d_ptr      = (const sycl::half *) (base_ptr + d_offset) + ib_global;

    // Destination position for this thread's 4 values
    dst_t * y = dst_row + block_in_row * QK_K + 128 * ip + il;

    // Read data for this thread
    const uint8_t * ql = ql_ptr + 64 * ip + il;
    const uint8_t   qh = *(qh_ptr + 32 * ip + il);
    const int8_t *  sc = reinterpret_cast<const int8_t *>(scales_ptr + is);
    const float     d  = *d_ptr;

    // Dequantize 4 values
    y[0]  = d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K SoA GET_ROWS
static void get_rows_q6_k_soa_sycl(ggml_backend_sycl_context & ctx,
                                   const ggml_tensor *         src0,
                                   const ggml_tensor *         src1,
                                   ggml_tensor *               dst,
                                   const void *                src0_dd,
                                   const int32_t *             src1_dd,
                                   float *                     dst_dd,
                                   int64_t                     row_offset,
                                   int64_t                     total_nrows,
                                   queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K uses 64 threads (2 phases * 32 threads) per block
    const sycl::range<3> block_dims(1, 1, 64);
    // One work-group per Q6_K block in each row
    const int64_t        blocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, blocks_per_row);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    sycl::event evt = stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
        k_get_rows_q6_k_soa<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11,
                                   s12, row_offset, total_nrows, item_ct1);
    });
    if (ggml_sycl_wait_after_get_rows_q6_k_soa()) {
        evt.wait_and_throw();
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Q6_K Coalesced layout kernel for GET_ROWS
// Coalesced layout: variable tile decomposition with word-major ordering
//   - Variable power-of-2 tiles (max 32 blocks), e.g., 56 blocks = 32 + 16 + 8
//   - Within each tile: word-major ordering for memory coalescing
//   - D values follow all quant tiles contiguously
template <typename dst_t>
static void k_get_rows_q6_k_coalesced(const void *             src0,
                                      const int32_t *          src1,
                                      dst_t *                  dst,
                                      int64_t                  ne00,
                                      int64_t                  ne01,
                                      int64_t                  ne12,
                                      size_t                   s1,
                                      size_t                   s2,
                                      size_t                   s3,
                                      size_t                   nb01,
                                      size_t                   nb02,
                                      size_t                   nb03,
                                      size_t                   s10,
                                      size_t                   s11,
                                      size_t                   s12,
                                      int64_t                  row_offset,
                                      int64_t                  d_offset,
                                      int64_t                  row_quants_bytes,
                                      const sycl::nd_item<3> & item_ct1) {
    // Thread layout: same as Q6_K SoA (64 threads: 2 phases x 32 threads)
    const int tid = item_ct1.get_local_id(2);
    const int ip  = tid / 32;
    const int il  = tid % 32;
    const int is  = 8 * ip + il / 16;

    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    // Block index within row (each block has QK_K=256 elements)
    const int     block_in_row   = item_ct1.get_group(2);
    const int64_t blocks_per_row = ne00 / QK_K;

    if (block_in_row >= blocks_per_row) {
        return;
    }

    const int64_t i01 = static_cast<int64_t>(src1[i10 * s10 + i11 * s11 + i12 * s12]) + row_offset;  // Row index
    dst_t *   dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

    // Global block index
    const int64_t ib_global = i01 * blocks_per_row + block_in_row;

    // === Variable tile decomposition (matches CPU reorder) ===
    // Find which tile contains this block and where within the tile
    int tile_size        = 0;
    int tile_byte_offset = 0;
    int block_in_tile    = 0;
    {
        int remaining = blocks_per_row;
        int block_idx = 0;
        while (remaining > 0) {
            // Find largest power-of-2 tile size <= remaining, max 32
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) {
                ts *= 2;
            }
            // Check if block_in_row is in this tile
            if (block_in_row < block_idx + ts) {
                tile_size     = ts;
                block_in_tile = block_in_row - block_idx;
                break;
            }
            // Advance to next tile
            tile_byte_offset += ts * (128 + 64 + 16);  // ql + qh + scales per block
            block_idx += ts;
            remaining -= ts;
        }
    }

    // Word plane stride for this tile
    const int word_plane_stride = tile_size * 4;

    // Base pointer
    const uint8_t *    base_ptr = static_cast<const uint8_t *>(src0);
    const sycl::half * d_ptr    = (const sycl::half *) (base_ptr + d_offset) + ib_global;

    // Destination position
    dst_t * y = dst_row + block_in_row * QK_K + 128 * ip + il;

    // Tile base for this row
    const int64_t row_base  = i01 * row_quants_bytes;
    const int64_t tile_base = row_base + tile_byte_offset;

    // Component offsets within this tile
    const int64_t ql_tile_base = tile_base;
    const int64_t qh_tile_base = tile_base + tile_size * 128;         // After ql
    const int64_t sc_tile_base = tile_base + tile_size * (128 + 64);  // After ql + qh

    // === Read two ql bytes ===
    const int     ql_pos0    = 64 * ip + il;
    const int     ql_pos1    = ql_pos0 + 32;
    const int     ql_word0   = ql_pos0 / 4;
    const int     ql_byte0   = ql_pos0 % 4;
    const int     ql_word1   = ql_pos1 / 4;
    const int     ql_byte1   = ql_pos1 % 4;
    const int64_t ql_offset0 = ql_tile_base + ql_word0 * word_plane_stride + block_in_tile * 4 + ql_byte0;
    const int64_t ql_offset1 = ql_tile_base + ql_word1 * word_plane_stride + block_in_tile * 4 + ql_byte1;
    const uint8_t ql0        = base_ptr[ql_offset0];
    const uint8_t ql1        = base_ptr[ql_offset1];

    // === Read qh byte ===
    const int     qh_pos    = 32 * ip + il;
    const int     qh_word   = qh_pos / 4;
    const int     qh_byte   = qh_pos % 4;
    const int64_t qh_offset = qh_tile_base + qh_word * word_plane_stride + block_in_tile * 4 + qh_byte;
    const uint8_t qh        = base_ptr[qh_offset];

    // === Read scales ===
    const int     sc_idx0    = is + 0;
    const int     sc_idx2    = is + 2;
    const int     sc_idx4    = is + 4;
    const int     sc_idx6    = is + 6;
    const int     sc_word0   = sc_idx0 / 4;
    const int     sc_byte0   = sc_idx0 % 4;
    const int     sc_word2   = sc_idx2 / 4;
    const int     sc_byte2   = sc_idx2 % 4;
    const int     sc_word4   = sc_idx4 / 4;
    const int     sc_byte4   = sc_idx4 % 4;
    const int     sc_word6   = sc_idx6 / 4;
    const int     sc_byte6   = sc_idx6 % 4;
    const int64_t sc_offset0 = sc_tile_base + sc_word0 * word_plane_stride + block_in_tile * 4 + sc_byte0;
    const int64_t sc_offset2 = sc_tile_base + sc_word2 * word_plane_stride + block_in_tile * 4 + sc_byte2;
    const int64_t sc_offset4 = sc_tile_base + sc_word4 * word_plane_stride + block_in_tile * 4 + sc_byte4;
    const int64_t sc_offset6 = sc_tile_base + sc_word6 * word_plane_stride + block_in_tile * 4 + sc_byte6;
    const int8_t  sc0        = static_cast<int8_t>(base_ptr[sc_offset0]);
    const int8_t  sc2        = static_cast<int8_t>(base_ptr[sc_offset2]);
    const int8_t  sc4        = static_cast<int8_t>(base_ptr[sc_offset4]);
    const int8_t  sc6        = static_cast<int8_t>(base_ptr[sc_offset6]);
    const float   d          = *d_ptr;

    // Dequantize (same logic as SoA)
    y[0]  = d * sc0 * ((int8_t) ((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc2 * ((int8_t) ((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc4 * ((int8_t) ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc6 * ((int8_t) ((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K Coalesced GET_ROWS (legacy - uses inline tile computation)
static void get_rows_q6_k_coalesced_sycl(ggml_backend_sycl_context & ctx,
                                         const ggml_tensor *         src0,
                                         const ggml_tensor *         src1,
                                         ggml_tensor *               dst,
                                         const void *                src0_dd,
                                         const int32_t *             src1_dd,
                                         float *                     dst_dd,
                                         int64_t                     row_offset,
                                         int64_t                     d_offset,
                                         int64_t                     row_quants_bytes,
                                         queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K uses 64 threads (2 phases x 32 threads) per block
    const sycl::range<3> block_dims(1, 1, 64);
    const int64_t        blocks_per_row = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, blocks_per_row);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        k_get_rows_q6_k_coalesced<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2, s3, nb01, nb02, nb03, s10,
                                         s11, s12, row_offset, d_offset, row_quants_bytes, item_ct1);
    });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// =============================================================================
// Q6_K VARIABLE TILE GET_ROWS KERNEL
// Handles arbitrary block counts using power-of-2 tile decomposition
// Each work-group processes one Q6_K block, computing its tile membership
// Matches the layout used by mul_mat_vec_q6_k_variable_tile
// =============================================================================
template <typename dst_t>
static void k_get_rows_q6_k_coalesced_variable(const void * __restrict__ x,
                                               const int32_t * __restrict__ indices,
                                               dst_t * __restrict__ dst,
                                               const int64_t            blocks_per_row,
                                               const int64_t            row_quants_bytes,
                                               const int64_t            total_nrows,
                                               const int64_t            row_offset,
                                               const sycl::nd_item<3> & item) {
    // Thread layout: 64 threads (2 phases x 32 threads) per Q6_K block
    const int tid = item.get_local_id(2);
    const int ip  = tid / 32;                // Phase: 0 or 1
    const int il  = tid % 32;                // Lane within phase
    const int is  = 8 * ip + il / 16;        // Scale index base

    const int row_idx  = item.get_group(2);  // Which row (from indices)
    const int block_id = item.get_group(1);  // Which block in the row

    const int64_t src_row = static_cast<int64_t>(indices[row_idx]) + row_offset;

    // === Variable tile decomposition ===
    // Find which tile this block belongs to
    int tile_size        = 0;
    int tile_byte_offset = 0;
    int block_in_tile    = 0;
    {
        int remaining = blocks_per_row;
        int block_idx = 0;
        while (remaining > 0) {
            // Find largest power-of-2 tile size <= remaining, max 32
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) {
                ts *= 2;
            }
            // Check if block_id is in this tile
            if (block_id < block_idx + ts) {
                tile_size     = ts;
                block_in_tile = block_id - block_idx;
                break;
            }
            // Advance to next tile
            tile_byte_offset += ts * (128 + 64 + 16);  // ql + qh + scales per tile
            block_idx += ts;
            remaining -= ts;
        }
    }

    // Word plane stride for this tile
    const int word_plane_stride = tile_size * 4;

    // Base pointers
    const uint8_t * x_base = (const uint8_t *) x;

    // D values at end of tensor (after all rows' quant data)
    const ggml_half * x_d = (const ggml_half *) (x_base + total_nrows * row_quants_bytes);
    const float       d   = x_d[src_row * blocks_per_row + block_id];

    // Tile base for this row
    const int64_t row_base  = src_row * row_quants_bytes;
    const int64_t tile_base = row_base + tile_byte_offset;

    // Component offsets within this tile
    const uint8_t * tile_ql = x_base + tile_base;
    const uint8_t * tile_qh = tile_ql + tile_size * 128;                    // After ql
    const int8_t *  tile_sc = (const int8_t *) (tile_qh + tile_size * 64);  // After ql + qh

    // === Read two ql bytes ===
    const int     ql_pos0    = 64 * ip + il;
    const int     ql_pos1    = ql_pos0 + 32;
    const int     ql_word0   = ql_pos0 / 4;
    const int     ql_byte0   = ql_pos0 % 4;
    const int     ql_word1   = ql_pos1 / 4;
    const int     ql_byte1   = ql_pos1 % 4;
    const int64_t ql_offset0 = ql_word0 * word_plane_stride + block_in_tile * 4 + ql_byte0;
    const int64_t ql_offset1 = ql_word1 * word_plane_stride + block_in_tile * 4 + ql_byte1;
    const uint8_t ql0        = tile_ql[ql_offset0];
    const uint8_t ql1        = tile_ql[ql_offset1];

    // === Read qh byte ===
    const int     qh_pos    = 32 * ip + il;
    const int     qh_word   = qh_pos / 4;
    const int     qh_byte   = qh_pos % 4;
    const int64_t qh_offset = qh_word * word_plane_stride + block_in_tile * 4 + qh_byte;
    const uint8_t qh        = tile_qh[qh_offset];

    // === Read scales (word-major access) ===
    const int     sc_idx0    = is + 0;
    const int     sc_idx2    = is + 2;
    const int     sc_idx4    = is + 4;
    const int     sc_idx6    = is + 6;
    const int     sc_word0   = sc_idx0 / 4;
    const int     sc_byte0   = sc_idx0 % 4;
    const int     sc_word2   = sc_idx2 / 4;
    const int     sc_byte2   = sc_idx2 % 4;
    const int     sc_word4   = sc_idx4 / 4;
    const int     sc_byte4   = sc_idx4 % 4;
    const int     sc_word6   = sc_idx6 / 4;
    const int     sc_byte6   = sc_idx6 % 4;
    const int64_t sc_offset0 = sc_word0 * word_plane_stride + block_in_tile * 4 + sc_byte0;
    const int64_t sc_offset2 = sc_word2 * word_plane_stride + block_in_tile * 4 + sc_byte2;
    const int64_t sc_offset4 = sc_word4 * word_plane_stride + block_in_tile * 4 + sc_byte4;
    const int64_t sc_offset6 = sc_word6 * word_plane_stride + block_in_tile * 4 + sc_byte6;
    const int8_t  sc0        = tile_sc[sc_offset0];
    const int8_t  sc2        = tile_sc[sc_offset2];
    const int8_t  sc4        = tile_sc[sc_offset4];
    const int8_t  sc6        = tile_sc[sc_offset6];

    // Destination position for this thread's 4 values
    dst_t * out = dst + row_idx * blocks_per_row * QK_K + block_id * QK_K + 128 * ip + il;

    // Dequantize 4 values (same logic as SoA/AoS kernels)
    out[0]  = d * sc0 * ((int8_t) ((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    out[32] = d * sc2 * ((int8_t) ((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    out[64] = d * sc4 * ((int8_t) ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    out[96] = d * sc6 * ((int8_t) ((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

// Dispatch function for Q6_K Variable Tile GET_ROWS
// Uses power-of-2 tile decomposition matching MMVQ variable tile kernel
template <typename dst_t>
static void get_rows_q6_k_coalesced_variable_sycl(ggml_backend_sycl_context & ctx,
                                                  const ggml_tensor *         src0,
                                                  const ggml_tensor *         src1,
                                                  ggml_tensor *               dst,
                                                  const void *                src0_dd,
                                                  const int32_t *             src1_dd,
                                                  dst_t *                     dst_dd,
                                                  int64_t                     row_offset,
                                                  int64_t                     total_nrows,
                                                  dpct::queue_ptr             stream) {
    const int64_t blocks_per_row = src0->ne[0] / QK_K;
    const int64_t nrows          = src1->ne[0];

    // Compute variable row_quants_bytes using tile decomposition
    // This matches the MMVQ variable tile kernel calculation
    int64_t row_quants_bytes = 0;
    {
        int remaining = blocks_per_row;
        while (remaining > 0) {
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) {
                ts *= 2;
            }
            row_quants_bytes += ts * (128 + 64 + 16);  // ql + qh + scales per tile
            remaining -= ts;
        }
    }

    // Grid: one work-group per (row, block) pair
    // 64 threads per work-group (2 phases x 32 threads for Q6_K block structure)
    sycl::range<3> grid(1, blocks_per_row, nrows);
    sycl::range<3> block(1, 1, 64);

    stream->parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) {
        k_get_rows_q6_k_coalesced_variable<dst_t>(src0_dd, src1_dd, dst_dd, blocks_per_row, row_quants_bytes,
                                                  total_nrows, row_offset, item);
    });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Q4_0 Coalesced layout kernel for GET_ROWS
// Coalesced layout: word-major within tiles
//   - For word w of block b in tile: offset = tile_base + w*stride + b*4
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - Scales (d values) follow all quant tiles contiguously
template <typename dst_t>
static void k_get_rows_q4_0_coalesced(const void *             src0,
                                      const int32_t *          src1,
                                      dst_t *                  dst,
                                      int64_t                  ne00,
                                      int64_t                  ne01,
                                      int64_t                  ne12,
                                      size_t                   s1,
                                      size_t                   s2,
                                      size_t                   s3,
                                      size_t                   nb01,
                                      size_t                   nb02,
                                      size_t                   nb03,
                                      size_t                   s10,
                                      size_t                   s11,
                                      size_t                   s12,
                                      int64_t                  row_offset,
                                      int64_t                  d_offset,
                                      const sycl::nd_item<3> & item_ct1) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK4_0 / 2;        // 16 bytes of quants per block
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;  // bytes between word planes

    // Each thread dequantizes 2 values (like k_get_rows)
    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = static_cast<int64_t>(src1[i10 * s10 + i11 * s11 + i12 * s12]) + row_offset;

    dst_t * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

    // Calculate block/element positions
    const int64_t blocks_per_row = ne00 / QK4_0;
    const int     ib_local       = i00 / QK4_0;                      // Block index within row
    const int64_t ib_global      = i01 * blocks_per_row + ib_local;  // Global block index

    const int iqs      = (i00 % QK4_0) / 2;                          // Quant index within block (0-15)
    const int iybs     = i00 - i00 % QK4_0;                          // Dst block start index
    const int y_offset = QK4_0 / 2;                                  // 16

    // Calculate coalesced layout positions
    const int64_t tile          = ib_local / TILE_BLOCKS;
    const int     block_in_tile = ib_local % TILE_BLOCKS;

    // Which word contains our quant byte?
    const int word_idx     = iqs / 4;  // Which of 4 words (0-3)
    const int byte_in_word = iqs % 4;  // Byte within word (0-3)

    // Row's quant section base
    const int64_t row_quants_bytes = ne00 / 2;  // bytes per row of quants
    const int64_t row_base         = i01 * row_quants_bytes;

    // Tile base within row
    const int64_t tile_base = row_base + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

    // Coalesced offset: word-major within tile
    const int64_t qs_offset = tile_base + word_idx * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word;

    // Read the quant byte from coalesced layout
    const uint8_t * qs_ptr  = (const uint8_t *) src0 + qs_offset;
    const uint8_t   qs_byte = *qs_ptr;

    // Read scale from contiguous d array after all quants
    const sycl::half * d_ptr = (const sycl::half *) ((const char *) src0 + d_offset) + ib_global;
    const float        d     = (float) (*d_ptr);

    // Dequantize: Q4_0 stores 2 values per byte (low 4 bits, high 4 bits)
    const int vl = (qs_byte & 0xF) - 8;
    const int vh = (qs_byte >> 4) - 8;

    dst_row[iybs + iqs + 0]        = d * vl;
    dst_row[iybs + iqs + y_offset] = d * vh;
}

// Dispatch function for Q4_0 Coalesced GET_ROWS
static void get_rows_q4_0_coalesced_sycl(ggml_backend_sycl_context & ctx,
                                         const ggml_tensor *         src0,
                                         const ggml_tensor *         src1,
                                         ggml_tensor *               dst,
                                         const void *                src0_dd,
                                         const int32_t *             src1_dd,
                                         float *                     dst_dd,
                                         int64_t                     row_offset,
                                         int64_t                     d_offset,
                                         queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int            block_num_x = (ne00 + 2 * SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2 * SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        k_get_rows_q4_0_coalesced<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2, s3, nb01, nb02, nb03, s10,
                                         s11, s12, row_offset, d_offset, item_ct1);
    });

    // DEBUG: avoid queue waits during graph recording

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// Q8_0 Coalesced layout kernel for GET_ROWS
// Coalesced layout: word-major within tiles
//   - For word w of block b in tile: offset = tile_base + w*stride + b*4
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - Scales (d values) follow all quant tiles contiguously
template <typename dst_t>
static void k_get_rows_q8_0_coalesced(const void *             src0,
                                      const int32_t *          src1,
                                      dst_t *                  dst,
                                      int64_t                  ne00,
                                      int64_t                  ne01,
                                      int64_t                  ne12,
                                      size_t                   s1,
                                      size_t                   s2,
                                      size_t                   s3,
                                      size_t                   nb01,
                                      size_t                   nb02,
                                      size_t                   nb03,
                                      size_t                   s10,
                                      size_t                   s11,
                                      size_t                   s12,
                                      int64_t                  row_offset,
                                      int64_t                  d_offset,
                                      const sycl::nd_item<3> & item_ct1) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2)) * 2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = static_cast<int64_t>(src1[i10 * s10 + i11 * s11 + i12 * s12]) + row_offset;

    dst_t * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;

    const int64_t blocks_per_row = ne00 / QK8_0;
    const int     ib_local       = i00 / QK8_0;
    const int64_t ib_global      = i01 * blocks_per_row + ib_local;

    const int iqs  = i00 % QK8_0;
    const int iybs = i00 - i00 % QK8_0;

    const int64_t tile          = ib_local / TILE_BLOCKS;
    const int     block_in_tile = ib_local % TILE_BLOCKS;

    const int word_idx0     = iqs / 4;
    const int byte_in_word0 = iqs % 4;
    const int word_idx1     = (iqs + 1) / 4;
    const int byte_in_word1 = (iqs + 1) % 4;

    const int64_t row_quants_bytes = ne00;
    const int64_t row_base         = i01 * row_quants_bytes;
    const int64_t tile_base        = row_base + tile * (TILE_BLOCKS * QK8_0);

    const int64_t qs_offset0 = tile_base + word_idx0 * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word0;
    const int64_t qs_offset1 = tile_base + word_idx1 * WORD_PLANE_STRIDE + block_in_tile * 4 + byte_in_word1;

    const int8_t q0 = (int8_t) *((const uint8_t *) src0 + qs_offset0);
    const int8_t q1 = (int8_t) *((const uint8_t *) src0 + qs_offset1);

    const sycl::half * d_ptr = (const sycl::half *) ((const char *) src0 + d_offset) + ib_global;
    const float        d     = (float) (*d_ptr);

    dst_row[iybs + iqs + 0] = d * (float) q0;
    dst_row[iybs + iqs + 1] = d * (float) q1;
}

// Dispatch function for Q8_0 Coalesced GET_ROWS
static void get_rows_q8_0_coalesced_sycl(ggml_backend_sycl_context & ctx,
                                         const ggml_tensor *         src0,
                                         const ggml_tensor *         src1,
                                         ggml_tensor *               dst,
                                         const void *                src0_dd,
                                         const int32_t *             src1_dd,
                                         float *                     dst_dd,
                                         int64_t                     row_offset,
                                         int64_t                     d_offset,
                                         queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int            block_num_x = (ne00 + 2 * SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2 * SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        k_get_rows_q8_0_coalesced<float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2, s3, nb01, nb02, nb03, s10,
                                         s11, s12, row_offset, d_offset, item_ct1);
    });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// SoA dispatch function for reordered Q4_0/Q8_0 tensors
template <int qk, int qr, dequantize_kernel_t_reorder dq_reorder>
static void get_rows_sycl_reorder(ggml_backend_sycl_context & ctx,
                                  const ggml_tensor *         src0,
                                  const ggml_tensor *         src1,
                                  ggml_tensor *               dst,
                                  const void *                src0_dd,
                                  const int32_t *             src1_dd,
                                  float *                     dst_dd,
                                  int64_t                     row_offset,
                                  int64_t                     d_offset,
                                  queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int            block_num_x = (ne00 + 2 * SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2 * SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        k_get_rows_reorder<qk, qr, dq_reorder, float>(src0_dd, src1_dd, dst_dd, ne00, ne01, ne12, s1, s2, s3, nb01,
                                                      nb02, nb03, s10, s11, s12, row_offset, d_offset, item_ct1);
    });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t, typename dst_t>
static void k_get_rows_float(const src0_t *                          src0,
                             const int32_t *                         src1,
                             dst_t *                                 dst,
                             int64_t                                 ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
                             /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
                             /*size_t s0,*/ size_t                   s1,
                             size_t                                  s2,
                             size_t                                  s3,
                             /*size_t nb00,*/ size_t                 nb01,
                             size_t                                  nb02,
                             size_t                                  nb03,
                             size_t                                  s10,
                             size_t                                  s11,
                             size_t                                  s12,
                             const sycl::nd_item<3> &                item_ct1 /*, size_t s13*/) {
    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) / ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0)) % ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];

    dst_t *        dst_row  = dst + i10 * s1 + i11 * s2 + i12 * s3;
    const src0_t * src0_row = (const src0_t *) ((const char *) src0 + i01 * nb01 + i11 * nb02 + i12 * nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx,
                          const ggml_tensor *         src0,
                          const ggml_tensor *         src1,
                          ggml_tensor *               dst,
                          const void *                src0_dd,
                          const int32_t *             src1_dd,
                          float *                     dst_dd,
                          queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int            block_num_x = (ne00 + 2 * SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2 * SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        k_get_rows<qk, qr, dq>(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12,
                               item_ct1);
    });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                ggml_tensor *               dst,
                                const src0_t *              src0_dd,
                                const int32_t *             src1_dd,
                                float *                     dst_dd,
                                queue_ptr                   stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int            block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12,
                             item_ct1);
        });
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

static void ggml_sycl_get_rows_dispatch_slice(ggml_backend_sycl_context & ctx,
                                              const ggml_tensor *         src0,
                                              const ggml_tensor *         src1,
                                              ggml_tensor *               dst,
                                              const void *                src0_dd,
                                              const int32_t *             src1_dd,
                                              float *                     dst_dd,
                                              layout_mode                 layout,
                                              int64_t                     row_offset,
                                              int64_t                     storage_rows,
                                              dpct::queue_ptr             stream) {
    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, src0, src1, dst, (const sycl::half *) src0_dd, src1_dd, dst_dd, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, src0, src1, dst, (const float *) src0_dd, src1_dd, dst_dd, stream);
            break;
        case GGML_TYPE_Q4_0:
            {
                const int64_t ne00     = src0->ne[0];
                const int64_t d_offset = storage_rows * ne00 / 2;
                if (layout == GGML_LAYOUT_SOA) {
                    get_rows_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                        ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, row_offset, d_offset, stream);
                } else if (layout == GGML_LAYOUT_COALESCED) {
                    get_rows_q4_0_coalesced_sycl(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, row_offset, d_offset,
                                                 stream);
                } else {
                    get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
                }
            }
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
            break;
        case GGML_TYPE_Q8_0:
            {
                const int64_t ne00     = src0->ne[0];
                const int64_t d_offset = storage_rows * ne00;
                if (layout == GGML_LAYOUT_SOA) {
                    get_rows_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(
                        ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, row_offset, d_offset, stream);
                } else if (layout == GGML_LAYOUT_COALESCED) {
                    get_rows_q8_0_coalesced_sycl(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, row_offset, d_offset,
                                                 stream);
                } else {
                    get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
                }
            }
            break;
        case GGML_TYPE_Q6_K:
            if (layout == GGML_LAYOUT_COALESCED) {
                get_rows_q6_k_coalesced_variable_sycl<float>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, row_offset,
                                                             storage_rows, stream);
            } else if (layout == GGML_LAYOUT_SOA) {
                get_rows_q6_k_soa_sycl(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, row_offset, storage_rows, stream);
            } else {
                get_rows_q6_k_aos_sycl(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, stream);
            }
            break;
        default:
            GGML_LOG_ERROR("%s: unsupported type for streaming: %s\n", __func__, ggml_type_name(src0->type));
            GGML_ABORT("fatal error");
    }
}

struct ggml_sycl_get_rows_marker_kernel;

static sycl::event get_rows_stream_copy(sycl::queue &                    queue,
                                        void *                           device_slice,
                                        size_t                           slice_bytes,
                                        size_t                           offset_bytes,
                                        const void *                     src_ptr,
                                        size_t                           src_size,
                                        const void *                     ctx_void,
                                        const std::vector<sycl::event> & deps) {
    GGML_UNUSED(src_ptr);
    GGML_UNUSED(src_size);
    const auto * ctx = static_cast<const get_rows_stream_ctx *>(ctx_void);
    if (!ctx || ctx->row_total_bytes == 0 || ctx->segment_count == 0) {
        return ggml_sycl_submit_marker<ggml_sycl_get_rows_marker_kernel>(queue);
    }

    GGML_ASSERT(offset_bytes % ctx->row_total_bytes == 0);
    GGML_ASSERT(slice_bytes % ctx->row_total_bytes == 0);

    const size_t row_start = offset_bytes / ctx->row_total_bytes;
    const size_t row_count = slice_bytes / ctx->row_total_bytes;

    const sycl::usm::alloc src_alloc = sycl::get_pointer_type(ctx->src_base, queue.get_context());
    if (src_alloc != sycl::usm::alloc::device) {
        uint8_t * host_slice =
            static_cast<uint8_t *>(ggml_sycl_malloc_host_tracked_bytes(slice_bytes, queue, "get_rows:host_stage"));
        if (!host_slice) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                  "GET_ROWS stream: host staging allocation failed");
        }
        size_t dst_segment_offset = 0;
        for (int seg_idx = 0; seg_idx < ctx->segment_count; ++seg_idx) {
            const auto & seg = ctx->segments[seg_idx];
            for (size_t i = 0; i < row_count; ++i) {
                const int32_t row_idx = ctx->row_indices[row_start + i] + static_cast<int32_t>(ctx->row_base);
                const uint8_t * src = ctx->src_base + seg.src_base + static_cast<size_t>(row_idx) * seg.bytes_per_row;
                void * dst = host_slice + dst_segment_offset + i * seg.bytes_per_row;
                std::memcpy(dst, src, seg.bytes_per_row);
            }
            dst_segment_offset += row_count * seg.bytes_per_row;
        }
        GGML_ASSERT(dst_segment_offset == slice_bytes);
        sycl::event evt;
        try {
            evt = queue.memcpy(device_slice, host_slice, slice_bytes, deps);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[GET_ROWS] stream copy enqueue failed: %s\n", e.what());
            throw;
        }
        if (auto * cache = ggml_sycl::get_unified_cache(queue)) {
            cache->defer_host_free(host_slice, slice_bytes, evt);
        } else {
            if (!ggml_sycl_graph_recording_active()) {
                evt.wait();
            }
            ggml_sycl_free_host_tracked_bytes(host_slice, slice_bytes, queue);
        }
        return evt;
    }

    size_t                   dst_segment_offset = 0;
    std::vector<sycl::event> cur_deps           = deps;
    sycl::event              last_evt;

    for (int seg_idx = 0; seg_idx < ctx->segment_count; ++seg_idx) {
        const auto & seg = ctx->segments[seg_idx];
        for (size_t i = 0; i < row_count; ++i) {
            const int32_t row_idx = ctx->row_indices[row_start + i] + static_cast<int32_t>(ctx->row_base);
            const uint8_t * src = ctx->src_base + seg.src_base + static_cast<size_t>(row_idx) * seg.bytes_per_row;
            void * dst = static_cast<uint8_t *>(device_slice) + dst_segment_offset + i * seg.bytes_per_row;
            try {
                last_evt = queue.memcpy(dst, src, seg.bytes_per_row, cur_deps);
            } catch (const sycl::exception & e) {
                GGML_LOG_ERROR("[GET_ROWS] stream copy enqueue failed: %s\n", e.what());
                throw;
            }
            cur_deps.assign(1, last_evt);
        }
        dst_segment_offset += row_count * seg.bytes_per_row;
    }

    return last_evt;
}

static sycl::event get_rows_stream_slice(sycl::queue &                    queue,
                                         void *                           device_slice,
                                         size_t                           slice_bytes,
                                         size_t                           offset_bytes,
                                         const void *                     ctx_void,
                                         const std::vector<sycl::event> & deps) {
    const auto * ctx = static_cast<const get_rows_stream_ctx *>(ctx_void);
    if (!ctx || ctx->row_total_bytes == 0) {
        return ggml_sycl_submit_marker<ggml_sycl_get_rows_marker_kernel>(queue);
    }

    if (!deps.empty()) {
        sycl::event dep_evt = ggml_sycl_submit_marker<ggml_sycl_get_rows_marker_kernel>(queue, deps);
        if (!queue.has_property<sycl::property::queue::in_order>() && !ggml_sycl_graph_recording_active()) {
            dep_evt.wait();
        }
    }

    GGML_ASSERT(offset_bytes % ctx->row_total_bytes == 0);
    GGML_ASSERT(slice_bytes % ctx->row_total_bytes == 0);

    const size_t row_start = offset_bytes / ctx->row_total_bytes;
    const size_t row_count = slice_bytes / ctx->row_total_bytes;
    GGML_ASSERT(row_count <= ctx->seq_count);

    ggml_tensor src1_fake{};
    src1_fake.type = GGML_TYPE_I32;
    src1_fake.ne[0] = static_cast<int64_t>(row_count);
    src1_fake.ne[1] = 1;
    src1_fake.ne[2] = 1;
    src1_fake.ne[3] = 1;
    src1_fake.nb[0] = sizeof(int32_t);
    src1_fake.nb[1] = src1_fake.nb[0] * src1_fake.ne[0];
    src1_fake.nb[2] = src1_fake.nb[1];
    src1_fake.nb[3] = src1_fake.nb[2];

    float * dst_ptr = ctx->dst_base + static_cast<int64_t>(row_start) * ctx->dst_row_stride;

    try {
        ggml_sycl_get_rows_dispatch_slice(*ctx->backend_ctx,
                                          ctx->src0,
                                          &src1_fake,
                                          ctx->dst,
                                          device_slice,
                                          ctx->seq_device,
                                          dst_ptr,
                                          ctx->layout,
                                          0,
                                          static_cast<int64_t>(row_count),
                                          &queue);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[GET_ROWS] stream kernel enqueue failed: %s\n", e.what());
        throw;
    }

    return ggml_sycl_submit_marker<ggml_sycl_get_rows_marker_kernel>(queue);
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int           device       = ctx.device;
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * storage      = get_storage_tensor(src0);
    const int64_t       storage_rows = ggml_nrows(storage);
    const int64_t       row_offset   = get_view_row_offset(src0);
    const size_t        view_offset  = get_view_byte_offset(src0);
    const void *        aos_base     = nullptr;
    const void *        src0_d       = nullptr;
    const int32_t *     src1_i32     = nullptr;
    float *             dst_d        = nullptr;

    ggml_tensor_extra_gpu * extra     = (ggml_tensor_extra_gpu *) src0->extra;
    const layout_mode       mode      = get_effective_layout_mode(extra);
    const bool              layout_ok = (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0 ||
                                         src0->type == GGML_TYPE_Q6_K);
    layout_mode             layout    = GGML_LAYOUT_AOS;
    layout_mode             chosen    = GGML_LAYOUT_AOS;
    if (ggml_sycl_get_layout_choice_for_tensor(src0, device, &chosen)) {
        layout = chosen;
    } else if (layout_ok && (mode == GGML_LAYOUT_SOA || mode == GGML_LAYOUT_COALESCED)) {
        layout = mode;
    }
    const void *            layout_base = nullptr;
    if (layout != GGML_LAYOUT_AOS) {
        layout_base = ggml_sycl_get_layout_ptr_for(storage, device, layout);
        if (!layout_base) {
            layout = GGML_LAYOUT_AOS;
        }
    }
    if (layout == GGML_LAYOUT_AOS) {
        aos_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_AOS);
        if (!aos_base) {
            if (ggml_backend_sycl_weights_evictable() && storage->buffer &&
                ggml_backend_buffer_is_host(storage->buffer)) {
                aos_base = storage->data;
            } else {
                aos_base = ggml_sycl_get_layout_ptr(storage, device);
            }
        }
        src0_d = aos_base ? (const char *) aos_base + view_offset : nullptr;
    }

    ggml_sycl::unified_cache * cache =
        ggml_sycl::unified_cache_enabled() ? ggml_sycl::get_unified_cache(*ctx.stream()) : nullptr;
    ggml_sycl_cache_id cache_key =
        cache ? ggml_backend_sycl_get_weight_cache_key(src0, device) : ggml_sycl_cache_id{};
    ggml_sycl::cache_ptr_view cache_view{};
    bool                      cache_view_valid = false;

    if (cache && cache_key.valid && ggml_sycl_tensor_is_weight(src0)) {
        cache_view = cache->get_view(cache_key, layout);

        const void * base_ptr = (layout == GGML_LAYOUT_AOS) ? aos_base : layout_base;
        if (!cache_view.ptr && base_ptr) {
            cache_view.ptr      = const_cast<void *>(base_ptr);
            cache_view.size     = ggml_row_size(src0->type, src0->ne[0]) * static_cast<size_t>(storage_rows);
            cache_view.layout   = layout;
            cache_view.type     = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
            const sycl::usm::alloc alloc = sycl::get_pointer_type(cache_view.ptr, ctx.stream()->get_context());
            if (alloc == sycl::usm::alloc::device) {
                cache_view.location = ggml_sycl::cache_location::DEVICE;
            } else if (alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared) {
                cache_view.location = ggml_sycl::cache_location::HOST_PINNED;
            } else {
                cache_view.location = ggml_sycl::cache_location::HOST_MMAP;
            }
        }
        cache_view_valid = cache_view.ptr != nullptr;
        if (cache_view_valid && cache_view.location == ggml_sycl::cache_location::HOST_MMAP &&
            ggml_backend_sycl_weights_evictable()) {
            GGML_LOG_WARN("[GET_ROWS] host-mmap weights; using streamed get_rows\n");
        }
    }

    src1_i32 = (const int32_t *) ggml_sycl_get_data_ptr(dst->src[1], device);
    dst_d    = (float *) ggml_sycl_get_data_ptr(dst, device);

    if (ggml_sycl_get_rows_trace_enabled()) {
        size_t free_mem  = 0;
        size_t total_mem = 0;
        ggml_backend_sycl_get_device_memory(device, &free_mem, &total_mem);
        const sycl::context & sycl_ctx = ctx.stream()->get_context();
        sycl::usm::alloc src0_alloc = sycl::usm::alloc::unknown;
        if (src0->data != nullptr) {
            src0_alloc = sycl::get_pointer_type(src0->data, sycl_ctx);
        }
        const sycl::usm::alloc src1_alloc = sycl::get_pointer_type(src1_i32, sycl_ctx);
        const sycl::usm::alloc dst_alloc  = sycl::get_pointer_type(dst_d, sycl_ctx);
        GGML_LOG_INFO(
            "[GET_ROWS] entry: tensor=%s type=%s mode=%d layout=%d rows=%lld ncols=%lld src0_alloc=%d src1_alloc=%d dst_alloc=%d "
            "free=%.1fMB total=%.1fMB\n",
            src0->name ? src0->name : "unknown",
            ggml_type_name(src0->type),
            (int) mode,
            (int) layout,
            (long long) ggml_nrows(storage),
            (long long) src0->ne[0],
            (int) src0_alloc,
            (int) src1_alloc,
            (int) dst_alloc,
            free_mem / (1024.0 * 1024.0),
            total_mem / (1024.0 * 1024.0));
    }

    const int64_t n_rows_total = dst->src[1]->ne[0];
    const bool    index_is_1d  = (dst->src[1]->ne[1] == 1 && dst->src[1]->ne[2] == 1 && dst->src[1]->ne[3] == 1);

    if (ggml_sycl_debug_getrows_tokens_enabled() && src0->name &&
        std::strstr(src0->name, "token_embd.weight") != nullptr && index_is_1d && n_rows_total > 0) {
        static int dbg_left = 8;
        if (dbg_left > 0) {
            int32_t host_ids[4] = { -1, -1, -1, -1 };
            int32_t direct_ids[4] = { -1, -1, -1, -1 };
            const int64_t n_copy = std::min<int64_t>(n_rows_total, 4);
            bool copied = false;
            const char * buft_name = "(no-buft)";
            if (dst->src[1] && dst->src[1]->buffer) {
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dst->src[1]->buffer);
                if (buft) {
                    buft_name = ggml_backend_buft_name(buft);
                }
            }
            const char * alloc_name = "unknown";
            if (ctx.stream()) {
                const sycl::context & sycl_ctx = ctx.stream()->get_context();
                const sycl::usm::alloc alloc = sycl::get_pointer_type(src1_i32, sycl_ctx);
                alloc_name = ggml_sycl_usm_alloc_name(alloc);
                if (alloc != sycl::usm::alloc::unknown) {
                    ctx.stream()->memcpy(host_ids, src1_i32, sizeof(int32_t) * (size_t) n_copy).wait();
                    copied = true;
                    if (alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared) {
                        for (int64_t i = 0; i < n_copy; ++i) {
                            direct_ids[(size_t) i] = src1_i32[i];
                        }
                    }
                }
            }
            if (!copied) {
                for (int64_t i = 0; i < n_copy; ++i) {
                    host_ids[(size_t) i] = src1_i32[i];
                    direct_ids[(size_t) i] = src1_i32[i];
                }
            }
            GGML_LOG_INFO(
                "[GET_ROWS] token_embd: buft=%s alloc=%s layout=%s mode=%s rows=%lld ids=[%d,%d,%d,%d] direct=[%d,%d,%d,%d] ptr=%p\n",
                buft_name,
                alloc_name,
                ggml_sycl_layout_mode_name_local(layout),
                ggml_sycl_layout_mode_name_local(mode),
                (long long) n_rows_total,
                host_ids[0], host_ids[1], host_ids[2], host_ids[3],
                direct_ids[0], direct_ids[1], direct_ids[2], direct_ids[3],
                (const void *) src1_i32);
            dbg_left--;
        }
    }

    if (cache_view_valid && index_is_1d && n_rows_total > 0) {
        if (ggml_sycl_get_rows_trace_enabled()) {
            GGML_LOG_INFO("[GET_ROWS] cache view: tensor=%s layout=%d ptr=%p loc=%d size=%zu\n",
                          src0->name ? src0->name : "unknown",
                          (int) layout,
                          cache_view.ptr,
                          (int) cache_view.location,
                          cache_view.size);
        }

        if (cache_view.ptr && cache_view.location != ggml_sycl::cache_location::DEVICE) {
            const bool weights_evictable = ggml_backend_sycl_weights_evictable();
            get_rows_stream_ctx stream_ctx{};
            const bool segments_ok =
                get_rows_build_stream_segments(src0, layout, src0->ne[0], storage_rows, stream_ctx);
            bool streamed = false;
            if (ggml_sycl_get_rows_trace_enabled()) {
                GGML_LOG_INFO("[GET_ROWS] stream setup: segments_ok=%d row_bytes=%zu layout=%d tensor=%s\n",
                              segments_ok ? 1 : 0,
                              stream_ctx.row_total_bytes,
                              (int) layout,
                              src0->name ? src0->name : "unknown");
            }
            if (segments_ok && stream_ctx.row_total_bytes > 0) {
                try {
                    std::vector<int32_t> row_indices(n_rows_total);
                    const sycl::usm::alloc idx_alloc =
                        sycl::get_pointer_type(src1_i32, ctx.stream()->get_context());
                    if (idx_alloc == sycl::usm::alloc::device) {
                        if (ggml_sycl_graph_recording_active()) {
                            ctx.graphs_disabled = true;
                            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                                  "GET_ROWS requires host indices; disabling graphs");
                        }
                        ctx.stream()->memcpy(row_indices.data(), src1_i32, n_rows_total * sizeof(int32_t)).wait();
                    } else {
                        std::memcpy(row_indices.data(), src1_i32, n_rows_total * sizeof(int32_t));
                    }

                    if (ggml_sycl_get_rows_trace_enabled() && n_rows_total > 0) {
                        int32_t min_idx = row_indices[0];
                        int32_t max_idx = row_indices[0];
                        for (int64_t i = 1; i < n_rows_total; ++i) {
                            min_idx = std::min(min_idx, row_indices[static_cast<size_t>(i)]);
                            max_idx = std::max(max_idx, row_indices[static_cast<size_t>(i)]);
                        }
                        GGML_LOG_INFO("[GET_ROWS] indices: rows=%lld min=%d max=%d\n",
                                      (long long) n_rows_total,
                                      min_idx,
                                      max_idx);
                    }

                    size_t slice_bytes  = 0;
                    size_t buffer_count = 0;
                    get_rows_resolve_dma_params(stream_ctx.row_total_bytes, slice_bytes, buffer_count);
                    const size_t rows_per_slice = slice_bytes / stream_ctx.row_total_bytes;
                    if (ggml_sycl_get_rows_trace_enabled()) {
                        size_t free_mem  = 0;
                        size_t total_mem = 0;
                        ggml_backend_sycl_get_device_memory(device, &free_mem, &total_mem);
                        GGML_LOG_INFO(
                            "[GET_ROWS] stream: tensor=%s layout=%d loc=%d rows=%lld row_bytes=%zu total_bytes=%zu "
                            "slice=%zu buffers=%zu rows_per_slice=%zu free=%.1fMB total=%.1fMB\n",
                            src0->name ? src0->name : "unknown",
                            (int) layout,
                        (int) cache_view.location,
                        (long long) n_rows_total,
                        stream_ctx.row_total_bytes,
                        stream_ctx.row_total_bytes * static_cast<size_t>(n_rows_total),
                        slice_bytes,
                        buffer_count,
                            rows_per_slice,
                            free_mem / (1024.0 * 1024.0),
                            total_mem / (1024.0 * 1024.0));
                    }

                    ggml_sycl_pool_alloc<int32_t> seq_device_alloc(ctx.pool());
                    int32_t * seq_device = seq_device_alloc.alloc(rows_per_slice);
                    if (!seq_device) {
                        GGML_LOG_WARN("[GET_ROWS] DMA index staging allocation failed (rows=%zu)\n", rows_per_slice);
                        if (weights_evictable) {
                            GGML_LOG_WARN("[GET_ROWS] Falling back to CPU get_rows (DMA index staging failed)\n");
                            if (ggml_sycl_cpu_fallback_graph(ctx, dst, "get_rows seq staging")) {
                                return;
                            }
                        }
                    } else {
                        std::vector<int32_t> seq_host(rows_per_slice);
                        for (size_t i = 0; i < rows_per_slice; ++i) {
                            seq_host[i] = static_cast<int32_t>(i);
                        }
                        sycl::event seq_evt =
                            ctx.stream()->memcpy(seq_device, seq_host.data(), rows_per_slice * sizeof(int32_t));
                        std::vector<sycl::event> stream_deps;
                        if (ggml_sycl_graph_recording_active()) {
                            stream_deps.push_back(seq_evt);
                        } else {
                            seq_evt.wait();
                        }

                        stream_ctx.backend_ctx    = &ctx;
                        stream_ctx.src0           = src0;
                        stream_ctx.dst            = dst;
                        stream_ctx.row_indices    = row_indices.data();
                        stream_ctx.row_count      = static_cast<size_t>(n_rows_total);
                        stream_ctx.row_base       = row_offset;
                        stream_ctx.layout         = layout;
                    stream_ctx.src_base       = static_cast<const uint8_t *>(cache_view.ptr);
                        stream_ctx.device_id      = device;
                        stream_ctx.seq_device     = seq_device;
                        stream_ctx.seq_count      = rows_per_slice;
                        stream_ctx.dst_base       = dst_d;
                        stream_ctx.dst_row_stride = dst->nb[1] / sizeof(float);

                        const size_t total_bytes = stream_ctx.row_total_bytes * static_cast<size_t>(n_rows_total);
                        cache->pin(cache_key, layout);
                        ggml_sycl::unified_cache::dma_stream_result result{};
                        const size_t min_slice_bytes = get_rows_min_slice_bytes(stream_ctx.row_total_bytes);
                        size_t try_slice_bytes = slice_bytes;
                        size_t try_buffers = buffer_count;
                        size_t retry_count = 0;
                        size_t retry_limit = 2;
                        (void) get_rows_parse_env_count_value("GGML_SYCL_GET_ROWS_DMA_RETRIES", retry_limit);
                        auto clamp_slice = [&](size_t bytes) {
                            if (bytes < min_slice_bytes) {
                                bytes = min_slice_bytes;
                            }
                            if (stream_ctx.row_total_bytes > 0) {
                                size_t rows = bytes / stream_ctx.row_total_bytes;
                                if (rows < 1) {
                                    rows = 1;
                                }
                                bytes = rows * stream_ctx.row_total_bytes;
                            }
                            return bytes;
                        };
                        try_slice_bytes = clamp_slice(try_slice_bytes);
                        bool streamed_ok = false;
                        for (; retry_count <= retry_limit; ++retry_count) {
                            bool should_retry = false;
                            try {
                                result = cache->stream_dma(cache_view,
                                                           total_bytes,
                                                           try_slice_bytes,
                                                           try_buffers,
                                                           get_rows_stream_slice,
                                                           &stream_ctx,
                                                           stream_deps,
                                                           get_rows_stream_copy);
                            } catch (const sycl::exception & e) {
                                if (get_rows_is_device_oom(e) && try_slice_bytes > min_slice_bytes) {
                                    should_retry = true;
                                } else {
                                    cache->unpin(cache_key, layout);
                                    throw;
                                }
                            }
                            if (!should_retry && result.ok) {
                                streamed_ok = true;
                                break;
                            }
                            if (!should_retry && !result.ok && try_slice_bytes > min_slice_bytes) {
                                should_retry = true;
                            }
                            if (!should_retry) {
                                break;
                            }
                            const size_t next_slice = clamp_slice(try_slice_bytes / 2);
                            if (next_slice == try_slice_bytes) {
                                break;
                            }
                            GGML_LOG_WARN("[GET_ROWS] DMA retry: slice %.1f MB -> %.1f MB (buffers=%zu)\n",
                                          try_slice_bytes / (1024.0 * 1024.0),
                                          next_slice / (1024.0 * 1024.0),
                                          try_buffers);
                            try_slice_bytes = next_slice;
                            try_buffers = 1;
                        }
                        if (streamed_ok) {
                            cache->unpin_on_event(cache_key, layout, result.event);
                        } else {
                            cache->unpin(cache_key, layout);
                        }
                        if (!streamed_ok) {
                            bool fallback_ok = false;
                            if (result.mmap_direct_failed) {
                                GGML_LOG_WARN("[GET_ROWS] DMA from mmap failed, falling back to CPU (%s)\n",
                                              dst->name ? dst->name : "unknown");
                                fallback_ok = ggml_sycl_cpu_fallback_graph(ctx, dst, "get_rows streaming");
                            }
                            if (!fallback_ok && weights_evictable) {
                                GGML_LOG_WARN("[GET_ROWS] DMA streaming failed, falling back to CPU (%s)\n",
                                              dst->name ? dst->name : "unknown");
                                fallback_ok = ggml_sycl_cpu_fallback_graph(ctx, dst, "get_rows streaming");
                            }
                            if (!fallback_ok && !weights_evictable) {
                                GGML_LOG_WARN("[GET_ROWS] DMA streaming failed, falling back to CPU (%s)\n",
                                              dst->name ? dst->name : "unknown");
                                fallback_ok = ggml_sycl_cpu_fallback_graph(ctx, dst, "get_rows streaming");
                            }
                            if (fallback_ok) {
                                return;
                            }
                            GGML_ABORT("GET_ROWS streaming failed");
                        }
                        GGML_UNUSED(src0_d);
                        GGML_UNUSED(src1_i32);
                        streamed = true;
                        return;
                    }
                } catch (const sycl::exception & e) {
                    GGML_LOG_WARN("[GET_ROWS] streaming threw (%s), falling back to CPU\n", e.what());
                    if (ggml_sycl_cpu_fallback_graph(ctx, dst, "get_rows streaming exception")) {
                        return;
                    }
                    GGML_ABORT("GET_ROWS streaming exception");
                }
            }
            if (!segments_ok || stream_ctx.row_total_bytes == 0) {
                if (ggml_sycl_get_rows_trace_enabled()) {
                    GGML_LOG_WARN("[GET_ROWS] streaming unavailable: segments_ok=%d row_bytes=%zu layout=%d tensor=%s\n",
                                  segments_ok ? 1 : 0,
                                  stream_ctx.row_total_bytes,
                                  (int) layout,
                                  src0->name ? src0->name : "unknown");
                }
                GGML_LOG_WARN("[GET_ROWS] Falling back to CPU get_rows (streaming unavailable)\n");
                if (ggml_sycl_cpu_fallback_graph(ctx, dst, "get_rows no stream")) {
                    return;
                }
            }
            GGML_UNUSED(streamed);
            GGML_ABORT("GET_ROWS host/mmap weights path did not stream or CPU-fallback");
        }
    }

    // TP DEBUG (controlled by GGML_SYCL_TP_DEBUG environment variable)
    bool       is_tok_embd    = dst->src[0]->name && strstr(dst->src[0]->name, "token_embd");
    int64_t    n_rows         = dst->src[1]->ne[0];  // Number of rows to fetch
    static int getrows_b1_dbg = 0;
    if (g_ggml_sycl_tp_debug && is_tok_embd && n_rows == 1 && getrows_b1_dbg++ < 5) {
        // Read token ID
        int32_t tok_id;
        ctx.stream()->memcpy(&tok_id, src1_i32, sizeof(int32_t)).wait();
        fprintf(stderr, "TP DEBUG GET_ROWS tok_embd batch=1: device=%d, tok_id=%d, src0_d=%p, src1_i32=%p, dst_d=%p\n",
                device, tok_id, src0_d, (void *) src1_i32, (void *) dst_d);

        // Check if embedding table has per-device pointers (extra->data_device)
        ggml_tensor *           emb_tensor = dst->src[0];
        ggml_tensor_extra_gpu * extra      = (ggml_tensor_extra_gpu *) emb_tensor->extra;
        if (extra) {
            fprintf(stderr, "TP DEBUG GET_ROWS: emb extra=%p, data_device[0]=%p, data_device[1]=%p\n", extra,
                    extra->data_device[0], extra->data_device[1]);
        } else {
            fprintf(stderr, "TP DEBUG GET_ROWS: emb extra=NULL (no per-device pointers)\n");
        }

        // Check embedding table values at tok_id position
        // Embedding table has shape [vocab_size, n_embd] = [32000, 4096]
        // For F32: row_offset = tok_id * n_embd * sizeof(float)
        // For F16: row_offset = tok_id * n_embd * sizeof(sycl::half)
        int64_t n_embd     = dst->src[0]->ne[0];
        size_t  elem_size  = ggml_type_size(dst->src[0]->type);
        size_t  row_offset = tok_id * n_embd * elem_size;

        fprintf(stderr,
                "TP DEBUG GET_ROWS embd table: ne[0]=%lld (n_embd), ne[1]=%lld (vocab), type=%s, elem_size=%zu\n",
                (long long) dst->src[0]->ne[0], (long long) dst->src[0]->ne[1], ggml_type_name(dst->src[0]->type),
                elem_size);

        // Read Q4_0 block at tok_id row (first block)
        // Q4_0 block: 2 bytes scale (fp16) + 16 bytes quants (32 4-bit values)
        // Block offset = tok_id * blocks_per_row * block_size = tok_id * (4096/32) * 18
        struct block_q4_0_t {
            sycl::half d;
            uint8_t    qs[16];
        };

        int64_t blocks_per_row = n_embd / 32;  // 4096/32 = 128 blocks per row
        size_t  q4_row_offset  = tok_id * blocks_per_row * sizeof(block_q4_0_t);

        block_q4_0_t blk;
        const char * q4_ptr = (const char *) src0_d + q4_row_offset;
        ctx.stream()->memcpy(&blk, q4_ptr, sizeof(blk)).wait();

        float d_val = (float) blk.d;
        // Dequantize first 4 values
        int   v0    = (blk.qs[0] & 0xF) - 8;
        int   v1    = (blk.qs[0] >> 4) - 8;
        int   v2    = (blk.qs[1] & 0xF) - 8;
        int   v3    = (blk.qs[1] >> 4) - 8;
        fprintf(stderr,
                "TP DEBUG GET_ROWS Q4_0[tok=%d]: ptr=%p, d=%.6f, qs[0-1]=0x%02x%02x, deq=[%.6f,%.6f,%.6f,%.6f]\n",
                tok_id, q4_ptr, d_val, blk.qs[0], blk.qs[1], v0 * d_val, v1 * d_val, v2 * d_val, v3 * d_val);

        // Also check token 0 and token 1 to verify embedding table has data
        block_q4_0_t blk0, blk1;
        ctx.stream()->memcpy(&blk0, src0_d, sizeof(blk0)).wait();  // Token 0, block 0
        ctx.stream()
            ->memcpy(&blk1, (const char *) src0_d + blocks_per_row * sizeof(block_q4_0_t), sizeof(blk1))
            .wait();  // Token 1, block 0
        fprintf(stderr, "TP DEBUG GET_ROWS tok0: d=%.6f, qs[0]=0x%02x | tok1: d=%.6f, qs[0]=0x%02x\n", (float) blk0.d,
                blk0.qs[0], (float) blk1.d, blk1.qs[0]);

        // Check which device the queue is actually on
        sycl::device queue_dev = ctx.stream()->get_device();
        fprintf(stderr, "TP DEBUG GET_ROWS: ctx.device=%d, queue_device='%s'\n", device,
                queue_dev.get_info<sycl::info::device::name>().c_str());

        // Also check tensor->data vs resolved pointer
        fprintf(stderr, "TP DEBUG GET_ROWS: tensor->data=%p, resolved src0_d=%p (match=%d)\n", dst->src[0]->data,
                src0_d, (dst->src[0]->data == src0_d));
    }

    // DEBUG: Check F32 get_rows for inp_out_ids reduction (attention output to FFN)
    // This is triggered when src0 is F32 and batch reduces from >1 to 1
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int f32_getrows_dbg = 0;
    bool       is_f32          = dst->src[0]->type == GGML_TYPE_F32;
    bool       is_reduction    = dst->src[0]->ne[1] > 1 && n_rows == 1;  // Batch >1 reduced to 1
    if (g_ggml_sycl_tp_debug && is_f32 && is_reduction && f32_getrows_dbg++ < 10) {
        const char * name = dst->src[0]->name ? dst->src[0]->name : "?";
        int32_t      row_idx;
        ctx.stream()->memcpy(&row_idx, src1_i32, sizeof(int32_t)).wait();

        // Read values at the row being extracted
        int64_t       ne0        = dst->src[0]->ne[0];  // Row width
        size_t        row_byte_offset = row_idx * ne0 * sizeof(float);
        float         src_vals[4];
        const float * row_ptr = (const float *) ((const char *) src0_d + row_byte_offset);
        ctx.stream()->memcpy(src_vals, row_ptr, 4 * sizeof(float)).wait();

        fprintf(stderr,
                "TP DEBUG GET_ROWS F32 reduction: src0=%s ne=[%lldx%lld], extracting row %d, values=[%f,%f,%f,%f]\n",
                name, (long long) ne0, (long long) dst->src[0]->ne[1], row_idx, src_vals[0], src_vals[1], src_vals[2],
                src_vals[3]);

        // Check if any values are NaN
        bool has_nan = false;
        for (int i = 0; i < 4; i++) {
            if (std::isnan(src_vals[i])) {
                has_nan = true;
            }
        }
        if (has_nan) {
            fprintf(stderr, "TP DEBUG GET_ROWS F32: WARNING - NaN found in source row %d!\n", row_idx);
        }
    }

    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *) src0_d, src1_i32, dst_d,
                                ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *) src0_d, src1_i32, dst_d,
                                ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            {
                // Check reorder mode for proper kernel dispatch
                ggml_tensor_extra_gpu * extra  =
                    (ggml_tensor_extra_gpu *) (src0->extra ? src0->extra : storage->extra);
                bool                    is_soa = ggml_sycl_layout_is_soa(extra);

                if (is_soa) {
                    // SoA layout: all qs first, then all d
                    // d_offset = total qs bytes = storage_rows * ne00 / 2 (for Q4_0: 16 bytes qs per 32 values)
                    const int64_t ne00     = src0->ne[0];
                    const int64_t d_offset = storage_rows * ne00 / 2;
                    const void *  soa_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_SOA);
                    if (soa_base == nullptr) {
                        get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, src0, dst->src[1], dst,
                                                                     (const float *) src0_d, src1_i32, dst_d,
                                                                     ctx.stream());
                    } else {
                        get_rows_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                            ctx, src0, dst->src[1], dst, soa_base, src1_i32, dst_d, row_offset, d_offset,
                            ctx.stream());
                    }
                } else if (ggml_sycl_layout_is_coalesced(extra)) {
                    // Coalesced layout: word-major within tiles
                    // d_offset = total qs bytes = storage_rows * ne00 / 2 (for Q4_0: 16 bytes qs per 32 values)
                    const int64_t ne00     = src0->ne[0];
                    const int64_t d_offset = storage_rows * ne00 / 2;
                    const void *  coa_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_COALESCED);
                    if (coa_base == nullptr) {
                        get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, src0, dst->src[1], dst,
                                                                     (const float *) src0_d, src1_i32, dst_d,
                                                                     ctx.stream());
                    } else {
                        get_rows_q4_0_coalesced_sycl(ctx, src0, dst->src[1], dst, coa_base, src1_i32, dst_d,
                                                     row_offset, d_offset, ctx.stream());
                    }
                } else {
                    // AoS (original) layout
                    get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, src0, dst->src[1], dst,
                                                                 (const float *) src0_d, src1_i32, dst_d, ctx.stream());
                }
            }
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *) src0_d,
                                                         src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *) src0_d,
                                                         src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *) src0_d,
                                                         src1_i32, dst_d, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            {
                // Check reorder mode for proper kernel dispatch
                ggml_tensor_extra_gpu * extra =
                    (ggml_tensor_extra_gpu *) (src0->extra ? src0->extra : storage->extra);
                if (ggml_sycl_layout_is_soa(extra)) {
                    // SoA layout: all qs first, then all d
                    // d_offset = total qs bytes = storage_rows * ne00 (for Q8_0: 32 bytes qs per 32 values)
                    const int64_t ne00     = src0->ne[0];
                    const int64_t d_offset = storage_rows * ne00;
                    const void *  soa_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_SOA);
                    if (soa_base == nullptr) {
                        get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, src0, dst->src[1], dst,
                                                                     (const float *) src0_d, src1_i32, dst_d,
                                                                     ctx.stream());
                    } else {
                        get_rows_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(
                            ctx, src0, dst->src[1], dst, soa_base, src1_i32, dst_d, row_offset, d_offset,
                            ctx.stream());
                    }
                } else if (ggml_sycl_layout_is_coalesced(extra)) {
                    // Coalesced layout: word-major within tiles
                    // d_offset = total qs bytes = storage_rows * ne00 (for Q8_0: 32 bytes qs per 32 values)
                    const int64_t ne00     = src0->ne[0];
                    const int64_t d_offset = storage_rows * ne00;
                    const void *  coa_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_COALESCED);
                    if (coa_base == nullptr) {
                        get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, src0, dst->src[1], dst,
                                                                     (const float *) src0_d, src1_i32, dst_d,
                                                                     ctx.stream());
                    } else {
                        get_rows_q8_0_coalesced_sycl(ctx, src0, dst->src[1], dst, coa_base, src1_i32, dst_d,
                                                     row_offset, d_offset, ctx.stream());
                    }
                } else {
                    // AoS (original) layout
                    get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, src0, dst->src[1], dst,
                                                                 (const float *) src0_d, src1_i32, dst_d, ctx.stream());
                }
            }
            break;
        case GGML_TYPE_Q6_K:
            {
                // Check reorder mode for proper kernel dispatch
                ggml_tensor_extra_gpu * extra =
                    (ggml_tensor_extra_gpu *) (src0->extra ? src0->extra : storage->extra);
                if (ggml_sycl_layout_is_coalesced(extra)) {
                    // Coalesced layout: variable tile decomposition with word-major ordering
                    // Uses new variable tile kernel that computes row_quants_bytes correctly
                    GGML_SYCL_DEBUG("Calling get_rows_q6_k_coalesced_variable_sycl\n");
                    const void * coa_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_COALESCED);
                    if (coa_base == nullptr) {
                        get_rows_q6_k_aos_sycl(ctx, src0, dst->src[1], dst, src0_d, src1_i32, dst_d, ctx.stream());
                    } else {
                        get_rows_q6_k_coalesced_variable_sycl<float>(ctx, src0, dst->src[1], dst, coa_base, src1_i32,
                                                                     dst_d, row_offset, storage_rows, ctx.stream());
                    }
                } else if (ggml_sycl_layout_is_soa(extra)) {
                    // Standard SoA layout: [all ql (n*128)][all qh (n*64)][all scales (n*16)][all d (n*2)]
                    const void * soa_base = ggml_sycl_get_layout_ptr_for(storage, device, GGML_LAYOUT_SOA);
                    if (soa_base == nullptr) {
                        get_rows_q6_k_aos_sycl(ctx, src0, dst->src[1], dst, src0_d, src1_i32, dst_d, ctx.stream());
                    } else {
                        get_rows_q6_k_soa_sycl(ctx, src0, dst->src[1], dst, soa_base, src1_i32, dst_d, row_offset,
                                               storage_rows, ctx.stream());
                    }
                } else {
                    // AoS (original) layout - uses specialized kernel due to Q6_K block complexity
                    get_rows_q6_k_aos_sycl(ctx, src0, dst->src[1], dst, src0_d, src1_i32, dst_d, ctx.stream());
                }
            }
            break;
        default:
            // TODO: other k-quants (Q2_K, Q3_K, Q4_K, Q5_K)
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }

    // DEBUG: Check output after kernel for token embedding batch=1
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int getrows_out_dbg = 0;
    bool       is_tok_embd_out = dst->src[0]->name && strstr(dst->src[0]->name, "token_embd");
    int64_t    n_rows_out      = dst->src[1]->ne[0];
    if (g_ggml_sycl_tp_debug && is_tok_embd_out && n_rows_out == 1 && getrows_out_dbg++ < 5) {
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst_d, std::min((size_t) 8 * sizeof(float), dst->ne[0] * sizeof(float))).wait();

        // Check for zeros
        int zero_count = 0;
        for (int i = 0; i < 8; i++) {
            if (out_vals[i] == 0.0f) {
                zero_count++;
            }
        }

        fprintf(stderr,
                "TP DEBUG GET_ROWS tok_embd OUTPUT batch=1: device=%d, "
                "dst[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f zeros=%d/8\n",
                device, out_vals[0], out_vals[1], out_vals[2], out_vals[3], out_vals[4], out_vals[5], out_vals[6],
                out_vals[7], zero_count);
    }

    // Optional numeric spot-check for AoS Q4_0 GET_ROWS correctness.
    // Enable with GGML_SYCL_GET_ROWS_COMPARE=1 and optionally filter by name with
    // GGML_SYCL_GET_ROWS_COMPARE_TENSOR=<substring>.
    static int compare_enabled = -1;
    if (compare_enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_GET_ROWS_COMPARE");
        compare_enabled  = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    if (compare_enabled && layout == GGML_LAYOUT_AOS && src0->type == GGML_TYPE_Q4_0 && src0_d && src1_i32 && dst_d) {
        const char * tensor_filter = std::getenv("GGML_SYCL_GET_ROWS_COMPARE_TENSOR");
        const bool filter_ok = !tensor_filter || (src0->name && std::strstr(src0->name, tensor_filter) != nullptr);
        static std::atomic<int> compare_remaining{ 1 };
        if (filter_ok) {
            const int remaining = compare_remaining.fetch_sub(1);
            if (remaining > 0) {
                ctx.stream()->wait();

                const int64_t sample_rows = std::min<int64_t>(n_rows_total, 2);
                const int64_t sample_cols = std::min<int64_t>(dst->ne[0], 64);
                if (sample_rows > 0 && sample_cols > 0) {
                    std::vector<int32_t> indices_host(static_cast<size_t>(n_rows_total));
                    ctx.stream()->memcpy(indices_host.data(), src1_i32, n_rows_total * sizeof(int32_t)).wait();

                    const int64_t ncols = src0->ne[0];
                    const size_t row_size = ggml_row_size(src0->type, ncols);
                    const int64_t blocks_per_row = ncols / QK4_0;

                    std::vector<float> out_host(static_cast<size_t>(sample_rows * sample_cols), 0.0f);
                    for (int64_t r = 0; r < sample_rows; ++r) {
                        const size_t dst_offset = static_cast<size_t>(r) * (dst->nb[1] / sizeof(float));
                        ctx.stream()->memcpy(&out_host[static_cast<size_t>(r * sample_cols)],
                                             dst_d + dst_offset,
                                             static_cast<size_t>(sample_cols) * sizeof(float)).wait();
                    }

                    float max_abs_err = 0.0f;
                    for (int64_t r = 0; r < sample_rows; ++r) {
                        const int32_t row_idx = indices_host[static_cast<size_t>(r)];
                        if (row_idx < 0 || row_idx >= storage_rows) {
                            continue;
                        }
                        const uint8_t * row_ptr =
                            static_cast<const uint8_t *>(src0_d) + static_cast<size_t>(row_idx) * src0->nb[1];
                        std::vector<uint8_t> row_host(row_size);
                        ctx.stream()->memcpy(row_host.data(), row_ptr, row_size).wait();
                        const auto * row_blocks = reinterpret_cast<const block_q4_0 *>(row_host.data());

                        for (int64_t c = 0; c < sample_cols; ++c) {
                            const int64_t block_idx = c / QK4_0;
                            const int idx_in_block = static_cast<int>(c % QK4_0);
                            const block_q4_0 & block = row_blocks[block_idx];
                            const float d = static_cast<float>(block.d);
                            const int qs_val =
                                (idx_in_block < 16) ? (block.qs[idx_in_block] & 0x0F)
                                                    : (block.qs[idx_in_block - 16] >> 4);
                            const float ref = static_cast<float>(qs_val - 8) * d;
                            const float got = out_host[static_cast<size_t>(r * sample_cols + c)];
                            max_abs_err = std::max(max_abs_err, std::fabs(got - ref));
                        }
                    }

                    GGML_LOG_INFO(
                        "[GET_ROWS-COMPARE] tensor=%s rows=%lld cols=%lld max_abs_err=%.6g\n",
                        src0->name ? src0->name : "unknown", (long long) sample_rows, (long long) sample_cols,
                        max_abs_err);
                }
            }
        }
    }
}
