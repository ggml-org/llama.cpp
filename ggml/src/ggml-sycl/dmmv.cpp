#include "convert.hpp"
#include "dmmv.hpp"
#include "dequantize.hpp"
#include "presets.hpp"
#include "dmmv-esimd.hpp"
#include "quants.hpp"
#include "ggml-sycl/quantize.hpp"

#include <cstdlib>
#include <cstring>

//============================================================================
// Unified DMMV SoA kernels following Q6_K pattern
// All quant types use the same structure: compute pointers directly in kernel
//============================================================================

struct ggml_sycl_q8_0_cache {
    block_q8_0 * ptr = nullptr;
    int ncols = 0;
    int device = -1;
};

struct ggml_sycl_q8_1_cache {
    block_q8_1 * ptr = nullptr;
    int ncols = 0;
    int device = -1;
};

static block_q8_0 * ggml_sycl_get_q8_0_cache(int ncols, int device, const dpct::queue_ptr & stream) {
    static thread_local ggml_sycl_q8_0_cache cache;
    const int blocks_per_row = ncols / QK8_0;

    if (cache.ptr != nullptr && (cache.ncols != ncols || cache.device != device)) {
        sycl::free(cache.ptr, *stream);
        cache.ptr = nullptr;
    }

    if (cache.ptr == nullptr) {
        cache.ptr = sycl::malloc_device<block_q8_0>(blocks_per_row, *stream);
        if (!cache.ptr) {
            GGML_ABORT("DMMV Q4_0: failed to allocate q8_0 cache buffer");
        }
        cache.ncols = ncols;
        cache.device = device;
    }

    return cache.ptr;
}

static block_q8_1 * ggml_sycl_get_q8_1_cache(int ncols, int device, const dpct::queue_ptr & stream) {
    static thread_local ggml_sycl_q8_1_cache cache;
    const int blocks_per_row = ncols / QK8_1;

    if (cache.ptr != nullptr && (cache.ncols != ncols || cache.device != device)) {
        sycl::free(cache.ptr, *stream);
        cache.ptr = nullptr;
    }

    if (cache.ptr == nullptr) {
        cache.ptr = sycl::malloc_device<block_q8_1>(blocks_per_row, *stream);
        if (!cache.ptr) {
            GGML_ABORT("DMMV: failed to allocate q8_1 cache buffer");
        }
        cache.ncols = ncols;
        cache.device = device;
    }

    return cache.ptr;
}

// Q4_0 SoA DMMV kernel - follows Q6_K pattern exactly
// SoA layout: [all qs: nblocks * 16 bytes] [all d: nblocks * 2 bytes]
// Each byte contains 2 x 4-bit values: low nibble = value[i], high nibble = value[i+16]
static void dequantize_mul_mat_vec_q4_0_soa_direct(
    const void* __restrict__ vx,
    const float* __restrict__ y,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const int64_t d_offset,  // Pre-calculated on host
    const int row_low,
    const sycl::nd_item<3>& item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item_ct1.get_local_id(2);
    const int num_blocks_per_row = ncols / QK4_0;
    const int global_row = row_low + row;
    const int ib0 = global_row * num_blocks_per_row;

    // SoA base pointers - exactly like Q6_K
    const uint8_t* qs_base = static_cast<const uint8_t*>(vx);
    const sycl::half* d_base = reinterpret_cast<const sycl::half*>(
        static_cast<const char*>(vx) + d_offset);

    float tmp = 0.0f;

    // Each thread processes blocks, similar to Q6_K pattern
    for (int block_in_row = tid; block_in_row < num_blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = ib0 + block_in_row;
        const float d = static_cast<float>(d_base[block_idx]);

        // SoA: calculate pointer for this block's qs
        const uint8_t* qs = qs_base + block_idx * (QK4_0 / 2);  // 16 bytes per block

        // y values for this block
        const float* y_block = y + block_in_row * QK4_0;

        // Process all 32 values in this block (16 bytes, 2 values per byte)
        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uint8_t qs_byte = qs[j];
            const float v0 = ((float)(qs_byte & 0xF) - 8.0f) * d;
            const float v1 = ((float)(qs_byte >> 4) - 8.0f) * d;

            // Q4_0 layout: low nibble = position j, high nibble = position j+16
            block_sum += v0 * y_block[j];
            block_sum += v1 * y_block[j + 16];
        }
        tmp += block_sum;
    }

    // Warp reduction
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// Q8_0 SoA DMMV kernel - follows Q6_K pattern exactly
// SoA layout: [all qs: nblocks * 32 bytes] [all d: nblocks * 2 bytes]
static void dequantize_mul_mat_vec_q8_0_soa_direct(
    const void* __restrict__ vx,
    const float* __restrict__ y,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const int64_t d_offset,  // Pre-calculated on host
    const int row_low,
    const sycl::nd_item<3>& item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item_ct1.get_local_id(2);
    const int num_blocks_per_row = ncols / QK8_0;
    const int global_row = row_low + row;
    const int ib0 = global_row * num_blocks_per_row;

    // SoA base pointers - exactly like Q6_K
    const int8_t* qs_base = static_cast<const int8_t*>(vx);
    const sycl::half* d_base = reinterpret_cast<const sycl::half*>(
        static_cast<const char*>(vx) + d_offset);

    float tmp = 0.0f;

    // Each thread processes blocks, similar to Q6_K pattern
    for (int block_in_row = tid; block_in_row < num_blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = ib0 + block_in_row;
        const float d = static_cast<float>(d_base[block_idx]);

        // SoA: calculate pointer for this block's qs
        const int8_t* qs = qs_base + block_idx * QK8_0;  // 32 bytes per block

        // y values for this block
        const float* y_block = y + block_in_row * QK8_0;

        // Process all 32 values in this block (32 int8 values)
        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < QK8_0; ++j) {
            const float v = static_cast<float>(qs[j]) * d;
            block_sum += v * y_block[j];
        }
        tmp += block_sum;
    }

    // Warp reduction
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void convert_f16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::half *x = (const sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

static void convert_f32(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const float * x = (const float *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

// Helper to traverse view_src chain and get the underlying storage tensor
// This is critical for SoA layout: offsets must be calculated using the storage tensor's
// dimensions (ne[1]), not the view's dimensions, since SoA layout is not sliceable.
static const ggml_tensor * get_storage_tensor(const ggml_tensor * t) {
    const ggml_tensor * current = t;
    while (current->view_src != nullptr) {
        current = current->view_src;
    }
    return current;
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;

        // Bounds check: skip if this thread's column is out of range
        if (col >= ncols) {
            continue;
        }

        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

// Q8_0 AoS debug kernel - captures comprehensive values for comparison with SoA kernel
// Debug buffer layout (256 floats):
// [0]: ncols, [1]: nrows, [2]: iter_stride, [3]: vals_per_iter
// [4-131]: Per-iteration data (32 iterations max, 4 floats each): i, ib, partial_sum, tmp_after
// [132]: final_tmp (before warp reduction), [133]: final_dst (after warp reduction)
// [134-197]: First 32 qs values, [198-229]: First 32 d values, [230-255]: First 26 y values
static void dequantize_mul_mat_vec_q8_0_aos_debug_kernel(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst,
                                   const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1, float * __restrict__ debug_buf) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = 1;  // QR8_0 == 1

    float tmp = 0.0f;

    // Debug: capture for row 0, tid 0
    bool do_debug = (row == 0 && tid == 0 && debug_buf != nullptr);

    if (do_debug) {
        debug_buf[0] = (float)ncols;
        debug_buf[1] = (float)nrows;
        debug_buf[2] = (float)iter_stride;
        debug_buf[3] = (float)vals_per_iter;
    }

    int debug_iter = 0;
    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;

        // Bounds check: skip if this thread's column is out of range
        if (col >= ncols) {
            continue;
        }

        const int ib = (row*ncols + col)/QK8_0;
        const int iqs = (col%QK8_0)/1;  // QR8_0 == 1
        const int iybs = col - col%QK8_0;

        float iter_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_q8_0(vx, ib, iqs + j/1, v);

            float product_x = v.x() * y[iybs + iqs + j / 1 + 0];
            float product_y = v.y() * y[iybs + iqs + j / 1 + y_offset];
            iter_sum += product_x + product_y;
            tmp += product_x + product_y;
        }

        // Capture per-iteration data (first 32 iterations)
        if (do_debug && debug_iter < 32) {
            int base = 4 + debug_iter * 4;
            debug_buf[base + 0] = (float)i;          // iteration i
            debug_buf[base + 1] = (float)ib;         // block index
            debug_buf[base + 2] = iter_sum;          // sum added this iteration
            debug_buf[base + 3] = tmp;               // running total
            debug_iter++;
        }
    }

    if (do_debug) {
        debug_buf[132] = tmp;  // final tmp before warp reduction

        // Capture raw data
        const block_q8_0* x = (const block_q8_0*)vx;
        for (int i = 0; i < 32 && i < ncols/QK8_0; i++) {
            debug_buf[134 + i] = (float)x[i].qs[0];  // first qs of each block
            debug_buf[198 + i] = (float)x[i].d;      // d of each block
        }
        for (int i = 0; i < 26 && i < ncols; i++) {
            debug_buf[230 + i] = y[i];  // first 26 y values
        }
    }

    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (do_debug) {
        debug_buf[133] = tmp;  // final after warp reduction
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// Simple reference SoA kernel - uses AoS-style iteration (iter_stride=64) but reads SoA data
// Enable with GGML_SYCL_DMMV_SIMPLE_SOA=1 for debugging
// d_offset: pre-calculated offset to scale values (calculated on host using storage tensor dimensions)
template <int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
static void dequantize_mul_mat_vec_reorder_simple(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst,
                                   const int ncols, const int nrows, const int64_t d_offset,
                                   const int row_low, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    // Use same iteration pattern as AoS kernel
    const int iter_stride = 2*GGML_SYCL_DMMV_X;  // 64 instead of 512
    const int vals_per_iter = iter_stride / WARP_SIZE; // 4 instead of 32
    const int y_offset = qr == 1 ? 1 : qk/2;

    float tmp = 0.0f;
    // d_offset is pre-calculated on host using storage tensor dimensions
    const char *d_ptr = (const char*)vx + d_offset;

    // Global row index in full tensor
    const int global_row = row_low + row;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;

        // Bounds check: skip if this thread's column is out of range
        if (col >= ncols) {
            continue;
        }

        const int ib = (global_row*ncols + col)/qk; // use global_row for SoA layout
        const int iqs = (col%qk)/qr;
        const int iybs = col - col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * (qk / qr) + iqs + j/qr, v);

            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
        }
    }

    // Warp reduction
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// d_offset: pre-calculated offset to scale values (calculated on host using storage tensor dimensions)
template <int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
static void dequantize_mul_mat_vec_reorder(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst,
                                   const int ncols, const int nrows, const int64_t d_offset,
                                   const int row_low, const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    // d_offset = pre-calculated byte offset from vx to scale values
    // row_low = starting row for this slice
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    // Use same simple iteration pattern as AoS kernel - no complex remainder handling
    const int iter_stride = 2*GGML_SYCL_DMMV_X;  // 64 - match AoS for precision parity
    const int vals_per_iter = iter_stride / WARP_SIZE; // 2 values per thread per iteration
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16
    // d_offset is pre-calculated on host using storage tensor dimensions
    const char *d_ptr = (const char*)vx + d_offset;

    // Global row index in full tensor (for correct block index in SoA layout)
    const int global_row = row_low + row;

    // Simple loop like AoS kernel - process all columns
    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        if (col >= ncols) continue;  // Bounds check for edge case ncols < iter_stride
        const int ib = (global_row*ncols + col)/qk; // x block index using global row
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * (qk / qr) + iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

// Q8_0 SoA kernel - uses soa_base (full tensor) + row_low for correct slicing
// Q8_0: 32 int8 values per block (32 bytes qs), unlike Q4_0 which packs 2 nibbles per byte
// SoA layout: [all qs: storage_rows * ncols bytes][all d: storage_rows * (ncols/qk) * 2 bytes]
// d_offset = pre-calculated byte offset from vx to scale values, row_low = starting row for this slice
template <int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
static void dequantize_mul_mat_vec_q8_0_reorder_kernel(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst,
                                   const int ncols, const int nrows, const int64_t d_offset,
                                   const int row_low, const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = qr == 1 ? 1 : qk/2;

    float tmp = 0.0f;

    // d_offset is pre-calculated on host using storage tensor dimensions
    const char *d_ptr = (const char*)vx + d_offset;

    // Global row index in full tensor (for correct block index in SoA layout)
    const int global_row = row_low + row;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;

        // Bounds check: skip if this thread's column is out of range
        if (col >= ncols) {
            continue;
        }

        // Use global_row for block index (correct for SoA layout)
        const int ib = (global_row*ncols + col)/qk;
        const int iqs = (col%qk)/qr;
        const int iybs = col - col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            // Q8_0: each block has QK8_0 (32) bytes of qs data
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * QK8_0 + iqs + j/qr, v);

            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
        }
    }

    // Warp reduction
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// Coalesced DMMV kernel for Q4_0
// Matches the coalesced MMVQ layout: word w of block b at tile_offset + w*64 + b*4
// Tile = TILE_BLOCKS blocks of 16-byte quants
// Scales are after all quants, block-sequential (not coalesced)
//
// Parameters for slicing support:
// - nrows: number of rows to process
// - nrows_full: full tensor rows (for scale offset calculation)
// - row_low: starting row offset (for scale indexing)
static void dequantize_mul_mat_vec_q4_0_coalesced(
    const void * __restrict__ vx,
    const dfloat * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    const sycl::nd_item<3> & nd_item)
{
    const int row = nd_item.get_group(2) * nd_item.get_local_range(1) +
                    nd_item.get_local_id(1);
    const int lane_id = nd_item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK4_0;
    const int word_stride = TILE_BLOCKS * 4;

    // X base pointers (coalesced layout: quants first, then scales)
    const uint8_t * x_qs = (const uint8_t *)vx;
    const int x_row_stride = ncols / 2;  // bytes per row of quants

    // Scales are after all quants in the FULL tensor
    const sycl::half * x_d = (const sycl::half *)((const char *)vx + nrows_full * x_row_stride);

    float partial_sum = 0.0f;
    const int global_row = row_low + row;
    const int row_quants_bytes = ncols / 2;
    const int y_offset = QK4_0 / 2;
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * lane_id;
        if (col >= ncols) {
            continue;
        }

        const int ib = col / QK4_0;
        const int iqs = (col % QK4_0) / 2;
        const int iybs = col - col % QK4_0;

        const int tile = ib / TILE_BLOCKS;
        const int block_in_tile = ib % TILE_BLOCKS;
        const int word_idx = iqs / 4;
        const int byte_in_word = iqs % 4;

        const int64_t row_base = (int64_t)global_row * row_quants_bytes;
        const int64_t tile_base = row_base + (int64_t)tile * MMVQ_COALESCED_TILE_BYTES;
        const int64_t qs_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;
        const uint8_t qs_byte = x_qs[qs_offset];

        const float d = (float)x_d[global_row * blocks_per_row + ib];
        const float v0 = (float)((qs_byte & 0xF) - 8);
        const float v1 = (float)((qs_byte >> 4) - 8);

        partial_sum += d * (v0 * (float)y[iybs + iqs + 0] + v1 * (float)y[iybs + iqs + y_offset]);
    }

    // Warp reduction
    float sum = partial_sum;
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        sum += dpct::permute_sub_group_by_xor(nd_item.get_sub_group(), sum, mask);
    }

    if (lane_id == 0) {
        dst[row] = sum;
    }
}

// Q4_0 coalesced DMMV with Q8_0-quantized Y (CPU-backend parity)
static void dequantize_mul_mat_vec_q4_0_coalesced_q8_0(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    const sycl::nd_item<3> & nd_item)
{
    const int row = nd_item.get_group(2) * nd_item.get_local_range(1) +
                    nd_item.get_local_id(1);
    const int lane_id = nd_item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK4_0;
    const int word_stride = TILE_BLOCKS * 4;

    const uint8_t * x_qs = (const uint8_t *)vx;
    const int row_quants_bytes = ncols / 2;
    const ggml_half * x_d = (const ggml_half *)((const char *)vx + nrows_full * row_quants_bytes);

    const int global_row = row_low + row;
    const int row_base = global_row * row_quants_bytes;

    float tmp = 0.0f;
    for (int block_in_row = lane_id; block_in_row < blocks_per_row; block_in_row += WARP_SIZE) {
        const int tile = block_in_row / TILE_BLOCKS;
        const int block_in_tile = block_in_row % TILE_BLOCKS;
        const int tile_base = row_base + tile * MMVQ_COALESCED_TILE_BYTES;

        int sumi = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int word_idx = j / 4;
            const int byte_in_word = j % 4;
            const int qs_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;
            const uint8_t qs_byte = x_qs[qs_offset];

            const int v0 = (qs_byte & 0xF) - 8;
            const int v1 = (qs_byte >> 4) - 8;
            const block_q8_0 * y_blk = y + block_in_row;
            sumi += v0 * y_blk->qs[j];
            sumi += v1 * y_blk->qs[j + QK4_0 / 2];
        }

        const int block_idx = global_row * blocks_per_row + block_in_row;
        const float d = (float)x_d[block_idx];
        const float yd = (float)y[block_in_row].d;
        tmp += (float)sumi * d * yd;
    }

    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(nd_item.get_sub_group(), tmp, mask);
    }

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_0_sycl_coalesced_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    dpct::queue_ptr stream)
{
    GGML_ASSERT((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q4_0_coalesced_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q4_0_coalesced_q8_0(vx, y, dst, ncols, nrows, nrows_full, row_low, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q4_0_soa_q8_0(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int64_t d_offset,
    const int row_low,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int num_blocks_per_row = ncols / QK4_0;
    const int global_row = row_low + row;
    const int ib0 = global_row * num_blocks_per_row;

    const uint8_t * qs_base = static_cast<const uint8_t *>(vx);
    const ggml_half * d_base = reinterpret_cast<const ggml_half *>(
        static_cast<const char *>(vx) + d_offset);

    float tmp = 0.0f;

    for (int block_in_row = tid; block_in_row < num_blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = ib0 + block_in_row;
        const uint8_t * qs = qs_base + block_idx * (QK4_0 / 2);
        const block_q8_0 * y_blk = y + block_in_row;

        int sumi = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uint8_t qs_byte = qs[j];
            const int v0 = (qs_byte & 0xF) - 8;
            const int v1 = (qs_byte >> 4) - 8;
            sumi += v0 * y_blk->qs[j];
            sumi += v1 * y_blk->qs[j + QK4_0 / 2];
        }

        const float d = (float)d_base[block_idx];
        const float yd = (float)y_blk->d;
        tmp += (float)sumi * d * yd;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_0_sycl_reorder_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    const int64_t d_offset, const int row_low,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q4_0_soa_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q4_0_soa_q8_0(vx, y, dst, ncols, nrows, d_offset, row_low, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q4_0_aos_q8_0(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int num_blocks_per_row = ncols / QK4_0;
    const block_q4_0 * x = (const block_q4_0 *)vx;

    float tmp = 0.0f;

    for (int block_in_row = tid; block_in_row < num_blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = row * num_blocks_per_row + block_in_row;
        const block_q4_0 * xb = x + block_idx;
        const block_q8_0 * y_blk = y + block_in_row;

        int sumi = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uint8_t qs_byte = xb->qs[j];
            const int v0 = (qs_byte & 0xF) - 8;
            const int v1 = (qs_byte >> 4) - 8;
            sumi += v0 * y_blk->qs[j];
            sumi += v1 * y_blk->qs[j + QK4_0 / 2];
        }

        const float d = (float)xb->d;
        const float yd = (float)y_blk->d;
        tmp += (float)sumi * d * yd;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_0_sycl_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q4_0_aos_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q4_0_aos_q8_0(vx, y, dst, ncols, nrows, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q4_1_q8_1(
    const void * __restrict__ vx,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int blocks_per_row = ncols / QK4_1;
    const block_q4_1 * x = (const block_q4_1 *)vx;

    float tmp = 0.0f;
    for (int block_in_row = tid; block_in_row < blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = row * blocks_per_row + block_in_row;
        const block_q4_1 * xb = x + block_idx;
        const block_q8_1 * yb = y + block_in_row;

        int sumi = 0;
        for (int j = 0; j < QK4_1 / 2; ++j) {
            const int v0 = (xb->qs[j] & 0xF);
            const int v1 = (xb->qs[j] >> 4);
            sumi += v0 * yb->qs[j];
            sumi += v1 * yb->qs[j + QK4_1 / 2];
        }

        const ggml_half2 xdm = xb->dm;
        const ggml_half2 yds = yb->ds;
        const float xd = (float)xdm.x();
        const float xm = (float)xdm.y();
        const float yd = (float)yds.x();
        const float ys = (float)yds.y();
        tmp += (xd * yd) * (float)sumi + xm * ys;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_1_sycl_q8_1(
    const void *vx, const block_q8_1 *y, float *dst,
    const int ncols, const int nrows,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q4_1_aos_q8_1_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q4_1_q8_1(vx, y, dst, ncols, nrows, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q5_0_q8_0(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int blocks_per_row = ncols / QK5_0;
    const block_q5_0 * x = (const block_q5_0 *)vx;

    float tmp = 0.0f;
    for (int block_in_row = tid; block_in_row < blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = row * blocks_per_row + block_in_row;
        const block_q5_0 * xb = x + block_idx;
        const block_q8_0 * yb = y + block_in_row;

        uint32_t qh;
        memcpy(&qh, xb->qh, sizeof(qh));

        int sumi = 0;
        for (int j = 0; j < QK5_0 / 2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int x0 = (int8_t)(((xb->qs[j] & 0x0F) | xh_0) - 16);
            const int x1 = (int8_t)(((xb->qs[j] >> 4) | xh_1) - 16);

            sumi += x0 * yb->qs[j];
            sumi += x1 * yb->qs[j + QK5_0 / 2];
        }

        tmp += (float)sumi * (float)xb->d * (float)yb->d;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q5_0_sycl_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q5_0_aos_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q5_0_q8_0(vx, y, dst, ncols, nrows, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q5_1_q8_1(
    const void * __restrict__ vx,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int blocks_per_row = ncols / QK5_1;
    const block_q5_1 * x = (const block_q5_1 *)vx;

    float tmp = 0.0f;
    for (int block_in_row = tid; block_in_row < blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = row * blocks_per_row + block_in_row;
        const block_q5_1 * xb = x + block_idx;
        const block_q8_1 * yb = y + block_in_row;

        uint32_t qh;
        memcpy(&qh, xb->qh, sizeof(qh));

        int sumi = 0;
        for (int j = 0; j < QK5_1 / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (xb->qs[j] & 0xF) | xh_0;
            const int x1 = (xb->qs[j] >> 4) | xh_1;

            sumi += x0 * yb->qs[j];
            sumi += x1 * yb->qs[j + QK5_1 / 2];
        }

        const ggml_half2 xdm = xb->dm;
        const ggml_half2 yds = yb->ds;
        const float xd = (float)xdm.x();
        const float xm = (float)xdm.y();
        const float yd = (float)yds.x();
        const float ys = (float)yds.y();
        tmp += (xd * yd) * (float)sumi + xm * ys;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q5_1_sycl_q8_1(
    const void *vx, const block_q8_1 *y, float *dst,
    const int ncols, const int nrows,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q5_1_aos_q8_1_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q5_1_q8_1(vx, y, dst, ncols, nrows, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q8_0_q8_0_aos(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int blocks_per_row = ncols / QK8_0;
    const block_q8_0 * x = (const block_q8_0 *)vx;

    float tmp = 0.0f;
    for (int block_in_row = tid; block_in_row < blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = row * blocks_per_row + block_in_row;
        const block_q8_0 * xb = x + block_idx;
        const block_q8_0 * yb = y + block_in_row;

        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += xb->qs[j] * yb->qs[j];
        }

        tmp += (float)sumi * (float)xb->d * (float)yb->d;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q8_0_aos_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q8_0_q8_0_aos(vx, y, dst, ncols, nrows, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q8_0_q8_0_soa(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int64_t d_offset,
    const int row_low,
    const sycl::nd_item<3> & item_ct1)
{
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int num_blocks_per_row = ncols / QK8_0;
    const int global_row = row_low + row;
    const int ib0 = global_row * num_blocks_per_row;

    const int8_t * qs_base = static_cast<const int8_t *>(vx);
    const ggml_half * d_base = reinterpret_cast<const ggml_half *>(
        static_cast<const char *>(vx) + d_offset);

    float tmp = 0.0f;
    for (int block_in_row = tid; block_in_row < num_blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = ib0 + block_in_row;
        const int8_t * qs = qs_base + block_idx * QK8_0;
        const block_q8_0 * yb = y + block_in_row;

        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += (int)qs[j] * yb->qs[j];
        }

        tmp += (float)sumi * (float)d_base[block_idx] * (float)yb->d;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl_reorder_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    const int64_t d_offset, const int row_low,
    dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q8_0_soa_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q8_0_q8_0_soa(vx, y, dst, ncols, nrows, d_offset, row_low, item_ct1);
            });
    });
}

static void dequantize_mul_mat_vec_q8_0_q8_0_coalesced(
    const void * __restrict__ vx,
    const block_q8_0 * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    const sycl::nd_item<3> & nd_item)
{
    const int row = nd_item.get_group(2) * nd_item.get_local_range(1) +
                    nd_item.get_local_id(1);
    const int lane_id = nd_item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK8_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int word_stride = TILE_BLOCKS * 4;

    const uint8_t * x_qs = (const uint8_t *)vx;
    const int x_row_stride = ncols;
    const ggml_half * x_d = (const ggml_half *)((const char *)vx + nrows_full * x_row_stride);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES_Q8_0;

        for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
            const int block_idx = (row_low + row) * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
            const float d = (float)x_d[block_idx];

            const int y_block = tile * TILE_BLOCKS + block_in_tile;
            const block_q8_0 * yb = y + y_block;

            int sumi = 0;
            for (int word = 0; word < 8; ++word) {
                const int word_offset = word * word_stride + block_in_tile * 4;
                const uint8_t * word_ptr = x_qs + tile_base + word_offset;
                sumi += ((int8_t)word_ptr[0]) * yb->qs[word * 4 + 0];
                sumi += ((int8_t)word_ptr[1]) * yb->qs[word * 4 + 1];
                sumi += ((int8_t)word_ptr[2]) * yb->qs[word * 4 + 2];
                sumi += ((int8_t)word_ptr[3]) * yb->qs[word * 4 + 3];
            }

            partial_sum += (float)sumi * d * (float)yb->d;
        }
    }

    float sum = partial_sum;
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        sum += dpct::permute_sub_group_by_xor(nd_item.get_sub_group(), sum, mask);
    }

    if (lane_id == 0) {
        dst[row] = sum;
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl_coalesced_q8_0(
    const void *vx, const block_q8_0 *y, float *dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    dpct::queue_ptr stream)
{
    GGML_ASSERT((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q8_0_coalesced_q8_0_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q8_0_q8_0_coalesced(vx, y, dst, ncols, nrows, nrows_full, row_low, item_ct1);
            });
    });
}

// Dispatch function for coalesced Q4_0 DMMV with slicing support
static void dequantize_mul_mat_vec_q4_0_sycl_coalesced(
    const void *vx, const dfloat *y, float *dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    dpct::queue_ptr stream)
{
    GGML_ASSERT((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q4_0_coalesced_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q4_0_coalesced(vx, y, dst, ncols, nrows, nrows_full, row_low, item_ct1);
            });
    });
}

// Coalesced DMMV kernel for Q8_0
// Matches the coalesced MMVQ layout: word w of block b at tile_offset + w*stride + b*4
// Scales are after all quants, block-sequential (not coalesced)
static void dequantize_mul_mat_vec_q8_0_coalesced(
    const void * __restrict__ vx,
    const dfloat * __restrict__ y,
    float * __restrict__ dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    const sycl::nd_item<3> & nd_item)
{
    const int row = nd_item.get_group(2) * nd_item.get_local_range(1) +
                    nd_item.get_local_id(1);
    const int lane_id = nd_item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS = MMVQ_COALESCED_TILE_BLOCKS;
    const int blocks_per_row = ncols / QK8_0;
    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int word_stride = TILE_BLOCKS * 4;

    // X base pointers (coalesced layout: quants first, then scales)
    const uint8_t * x_qs = (const uint8_t *)vx;
    const int x_row_stride = ncols;  // bytes per row of quants

    // Scales are after all quants in the FULL tensor
    const sycl::half * x_d = (const sycl::half *)((const char *)vx + nrows_full * x_row_stride);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES_Q8_0;

        for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
            const int block_idx = (row_low + row) * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
            const float d = (float)x_d[block_idx];

            const int y_base = (tile * TILE_BLOCKS + block_in_tile) * QK8_0;

            uint8_t qs[QK8_0];
#pragma unroll
            for (int word = 0; word < 8; ++word) {
                const int word_offset = word * word_stride + block_in_tile * 4;
                const uint8_t * word_ptr = x_qs + tile_base + word_offset;
                qs[word * 4 + 0] = word_ptr[0];
                qs[word * 4 + 1] = word_ptr[1];
                qs[word * 4 + 2] = word_ptr[2];
                qs[word * 4 + 3] = word_ptr[3];
            }

            float block_sum = 0.0f;
#pragma unroll
            for (int j = 0; j < QK8_0; ++j) {
                block_sum += (float)((int8_t)qs[j]) * (float)y[y_base + j];
            }

            partial_sum += d * block_sum;
        }
    }

    float sum = partial_sum;
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        sum += dpct::permute_sub_group_by_xor(nd_item.get_sub_group(), sum, mask);
    }

    if (lane_id == 0) {
        dst[row] = sum;
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl_coalesced(
    const void *vx, const dfloat *y, float *dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    dpct::queue_ptr stream)
{
    GGML_ASSERT((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dmmv_q8_0_coalesced_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_q8_0_coalesced(vx, y, dst, ncols, nrows, nrows_full, row_low, item_ct1);
            });
    });
}

__dpct_inline__ static uint8_t load_q6k_coalesced_byte(
    const uint8_t * base,
    const int64_t tile_base,
    const int block_in_tile,
    const int byte_index) {
    const int word = byte_index / 4;
    const int byte_in_word = byte_index % 4;
    const int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;
    const int64_t offset = tile_base + word * word_stride + block_in_tile * 4 + byte_in_word;
    return base[offset];
}

static void dequantize_mul_mat_vec_q6_k_coalesced(const void * __restrict__ vx,
                                                  const float * __restrict__ yy,
                                                  float * __restrict__ dst,
                                                  const int ncols, int nrows,
                                                  const int nrows_full,
                                                  const int row_low,
                                                  const sycl::nd_item<3> &item_ct1) {
    static_assert(16 % K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int global_row = row_low + row;
    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;

    constexpr int ql_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 2);
    constexpr int qh_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 4);
    constexpr int sc_tile_bytes = MMVQ_COALESCED_TILE_BLOCKS * (QK_K / 16);
    constexpr int tile_total = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;

    const uint8_t * base = static_cast<const uint8_t *>(vx);
    const int64_t tile_row_base = static_cast<int64_t>(global_row) * tiles_per_row * tile_total;
    const int64_t d_offset = static_cast<int64_t>(nrows_full) * tiles_per_row * tile_total;
    const sycl::half * d_base = reinterpret_cast<const sycl::half *>(base + d_offset);

#if QK_K == 256
    const int tid = item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION;
    const int ix = item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION;

    const int step = 16 / K_QUANTS_PER_ITERATION;

    const int im = tid / step;
    const int in = tid - step * im;

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION * in;
    const int is = 0;
#else
    const int l0 = 4 * in;
    const int is = in / 4;
#endif
    const int ql_offset_local = 64 * im + l0;
    const int qh_offset_local = 32 * im + l0;
    const int s_offset = 8 * im + is;
    const int y_offset = 128 * im + l0;

    float tmp = 0.0f;

    for (int i = ix; i < blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int tile = i / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile = i % MMVQ_COALESCED_TILE_BLOCKS;
        const int64_t ql_tile_base = tile_row_base + tile * tile_total;
        const int64_t qh_tile_base = ql_tile_base + ql_tile_bytes;
        const int64_t sc_tile_base = qh_tile_base + qh_tile_bytes;

        const int block_idx = global_row * blocks_per_row + i;
        const float d = static_cast<float>(d_base[block_idx]);
        const float * y = yy + i * QK_K + y_offset;

#if K_QUANTS_PER_ITERATION == 1
        const uint8_t ql0 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql_offset_local + 0);
        const uint8_t ql1 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql_offset_local + 16);
        const uint8_t ql2 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql_offset_local + 32);
        const uint8_t ql3 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql_offset_local + 48);

        const uint8_t qh0 = load_q6k_coalesced_byte(base, qh_tile_base, block_in_tile, qh_offset_local + 0);
        const uint8_t qh1 = load_q6k_coalesced_byte(base, qh_tile_base, block_in_tile, qh_offset_local + 16);

        const int8_t s0 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 0));
        const int8_t s1 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 1));
        const int8_t s2 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 2));
        const int8_t s3 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 3));
        const int8_t s4 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 4));
        const int8_t s5 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 5));
        const int8_t s6 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 6));
        const int8_t s7 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_offset + 7));

        int8_t q0 = (int8_t)((ql0 & 0xF) | ((qh0 & 0x03) << 4)) - 32;
        int8_t q1 = (int8_t)((ql1 & 0xF) | ((qh1 & 0x03) << 4)) - 32;
        int8_t q2 = (int8_t)((ql2 & 0xF) | ((qh0 & 0x0c) << 2)) - 32;
        int8_t q3 = (int8_t)((ql3 & 0xF) | ((qh1 & 0x0c) << 2)) - 32;
        int8_t q4 = (int8_t)((ql0 >> 4) | ((qh0 & 0x30) >> 0)) - 32;
        int8_t q5 = (int8_t)((ql1 >> 4) | ((qh1 & 0x30) >> 0)) - 32;
        int8_t q6 = (int8_t)((ql2 >> 4) | ((qh0 & 0xc0) >> 2)) - 32;
        int8_t q7 = (int8_t)((ql3 >> 4) | ((qh1 & 0xc0) >> 2)) - 32;

        float sum = y[ 0] * s0 * d * q0
                  + y[16] * s1 * d * q1
                  + y[32] * s2 * d * q2
                  + y[48] * s3 * d * q3
                  + y[64] * s4 * d * q4
                  + y[80] * s5 * d * q5
                  + y[96] * s6 * d * q6
                  + y[112] * s7 * d * q7;

        tmp += sum;
#else
        float sum = 0.0f;
        for (int l = 0; l < 4; ++l) {
            const int ql0_idx = ql_offset_local + l;
            const int ql1_idx = ql_offset_local + l + 32;
            const int qh_idx = qh_offset_local + l;

            const uint8_t ql0 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql0_idx);
            const uint8_t ql1 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql1_idx);
            const uint8_t qh = load_q6k_coalesced_byte(base, qh_tile_base, block_in_tile, qh_idx);

            int8_t q0 = (int8_t)((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
            int8_t q1 = (int8_t)((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32;

            const int s_base = s_offset;
            const int8_t s0 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_base + 0));
            const int8_t s2 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_base + 2));
            const int8_t s4 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_base + 4));
            const int8_t s6 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, s_base + 6));

            sum += y[l +  0] * s0 * d * q0
                 + y[l + 32] * s2 * d * q1
                 + y[l + 64] * s4 * d * q2
                 + y[l + 96] * s6 * d * q3;
        }
        tmp += sum;
#endif
    }
#else
    const int tid = item_ct1.get_local_id(2) / (2 * K_QUANTS_PER_ITERATION);
    const int ix = item_ct1.get_local_id(2) % (2 * K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0.0f;
    for (int i = ix; i < blocks_per_row; i += 2 * K_QUANTS_PER_ITERATION) {
        const int tile = i / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile = i % MMVQ_COALESCED_TILE_BLOCKS;
        const int64_t ql_tile_base = tile_row_base + tile * tile_total;
        const int64_t qh_tile_base = ql_tile_base + ql_tile_bytes;
        const int64_t sc_tile_base = qh_tile_base + qh_tile_bytes;

        const int block_idx = global_row * blocks_per_row + i;
        const float d = static_cast<float>(d_base[block_idx]);
        const float * y = yy + i * QK_K + step;

        float sum = 0.0f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const int ql0_idx = step + j;
            const int ql1_idx = step + j + 16;
            const int qh_idx = step + j;

            const uint8_t ql0 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql0_idx);
            const uint8_t ql1 = load_q6k_coalesced_byte(base, ql_tile_base, block_in_tile, ql1_idx);
            const uint8_t qh = load_q6k_coalesced_byte(base, qh_tile_base, block_in_tile, qh_idx);

            const int8_t s0 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, 0));
            const int8_t s1 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, 1));
            const int8_t s2 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, 2));
            const int8_t s3 = static_cast<int8_t>(load_q6k_coalesced_byte(base, sc_tile_base, block_in_tile, 3));

            sum += y[j +  0] * s0 * d * ((int8_t)((ql0 & 0xF) | ((qh & 0x03) << 4)) - 32)
                 + y[j + 16] * s1 * d * ((int8_t)((ql1 & 0xF) | ((qh & 0x0c) << 2)) - 32)
                 + y[j + 32] * s2 * d * ((int8_t)((ql0 >> 4) | ((qh & 0x30) >> 0)) - 32)
                 + y[j + 48] * s3 * d * ((int8_t)((ql1 >> 4) | ((qh & 0xc0) >> 2)) - 32);
        }
        tmp += sum;
    }
#endif

    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q6_K_sycl_coalesced(
    const void * vx, const float * y, float * dst,
    const int ncols, const int nrows,
    const int nrows_full, const int row_low,
    dpct::queue_ptr stream) {
    GGML_ASSERT((ncols / QK_K) % MMVQ_COALESCED_TILE_BLOCKS == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k_coalesced(vx, y, dst, ncols, nrows, nrows_full, row_low, item_ct1);
        });
}

static void convert_mul_mat_vec_f16_sycl(const void *vx, const dfloat *y,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, item_ct1);
            });
    }
}

/*
DPCT1110:4: The total declared local variable size in device function
dequantize_mul_mat_vec_q2_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q2_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp += dall * sum1 - dmin * sum2;

    }
#else
    const int tid = item_ct1.get_local_id(2) /
                    (2 * K_QUANTS_PER_ITERATION); // 0...15 or 0...7
    const int ix = item_ct1.get_local_id(2) %
                   (2 * K_QUANTS_PER_ITERATION); // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;


    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const sycl::float2 dall =
            x[i].dm.convert<float, sycl::rounding_mode::automatic>();

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

/*
DPCT1110:5: The total declared local variable size in device function
dequantize_mul_mat_vec_q3_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q3_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;

    }
#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

/*
DPCT1110:6: The total declared local variable size in device function
dequantize_mul_mat_vec_q4_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q4_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il  = tid/step;                            // 0...3
    const int ir  = tid - step*il;                       // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
            s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                       s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
               dmin * smin;
#else
        const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
        const uint16_t * q2 = q1 + 32;

        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[0] & 0xf0f0;
        q16[2] = q2[0] & 0x0f0f;
        q16[3] = q2[0] & 0xf0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 2; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+2];
            s.z += y2[l] * q4[l+4]; s.w += y2[l+32] * q4[l+6];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#endif

    }
#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

/*
DPCT1110:7: The total declared local variable size in device function
dequantize_mul_mat_vec_q5_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q5_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2);
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = item_ct1.get_local_id(2) / 2; // 0...15
    const int ix = item_ct1.get_local_id(2) % 2;

    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            sum.x() +=
                y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
            sum.y() +=
                y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
            sum.z() +=
                y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
            sum.w() +=
                y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                       sum.w() * sc[5]) -
               dmin * smin;
    }

#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        // Decode individual q values for debugging
        int8_t q0 = (int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32;
        int8_t q1 = (int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32;
        int8_t q2 = (int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32;
        int8_t q3 = (int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32;
        int8_t q4 = (int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32;
        int8_t q5 = (int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32;
        int8_t q6 = (int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32;
        int8_t q7 = (int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32;

        float sum = y[ 0] * s[0] * d * q0
                  + y[16] * s[1] * d * q1
                  + y[32] * s[2] * d * q2
                  + y[48] * s[3] * d * q3
                  + y[64] * s[4] * d * q4
                  + y[80] * s[5] * d * q5
                  + y[96] * s[6] * d * q6
                  +y[112] * s[7] * d * q7;

        tmp += sum;
#else
        // K_QUANTS_PER_ITERATION == 2 path
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            int8_t q0 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q1 = (int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += y[l+ 0] * s[0] * d * q0
                 + y[l+32] * s[2] * d * q1
                 + y[l+64] * s[4] * d * q2
                 + y[l+96] * s[6] * d * q3;
        }
        tmp += sum;
#endif

    }

#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// Q4_0 SoA dispatch function - uses soa_base (full tensor) + row_low for correct slicing
// vx: storage tensor base pointer from ggml_sycl_get_data_ptr()
// d_offset: pre-calculated byte offset from vx to scale values (using storage tensor dimensions)
// row_low: starting row for this slice (relative to storage tensor)
static void dequantize_mul_mat_vec_q4_0_sycl_reorder(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows, const int64_t d_offset,
                                             const int row_low, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);

    // Debug: trace SoA DMMV parameters
    static int soa_debug_count = 0;
    if (std::getenv("GGML_SYCL_SOA_DEBUG") && soa_debug_count++ < 50) {
        fprintf(stderr, "[Q4_0_SOA_DMMV] #%d vx=%p ncols=%d nrows=%d d_offset=%lld row_low=%d\n",
                soa_debug_count, vx, ncols, nrows, (long long)d_offset, row_low);
        fflush(stderr);
    }
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        // Use new direct SoA kernel following Q6_K pattern
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class dmmv_q4_0_soa_direct_kernel>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec_q4_0_soa_direct(
                        vx, y, dst, ncols, nrows, d_offset, row_low, item_ct1);
                });
        });
    }
}

// Q8_0 SoA dispatch function - uses soa_base (full tensor) + row_low for correct slicing
// vx: storage tensor base pointer from ggml_sycl_get_data_ptr()
// d_offset: pre-calculated byte offset from vx to scale values (using storage tensor dimensions)
// row_low: starting row for this slice (relative to storage tensor)
static void dequantize_mul_mat_vec_q8_0_sycl_reorder(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows, const int64_t d_offset,
                                             const int row_low, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        // Use new direct SoA kernel following Q6_K pattern
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class dmmv_q8_0_soa_direct_kernel>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec_q8_0_soa_direct(
                        vx, y, dst, ncols, nrows, d_offset, row_low, item_ct1);
                });
        });
    }
}

static void dequantize_mul_mat_vec_q4_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q4_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q5_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q5_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    // Debug buffer for GPU-side debugging - no limit, always debug when env set
    static int aos_kernel_debug_count = 0;
    float* debug_buf = nullptr;
    bool do_debug = std::getenv("GGML_SYCL_DMMV_SOA_DEBUG") != nullptr;

    if (do_debug) {
        debug_buf = sycl::malloc_device<float>(256, *stream);
        stream->memset(debug_buf, 0, 256 * sizeof(float)).wait();
    }

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        if (do_debug) {
            stream->submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class dmmv_q8_0_aos_debug_kernel>(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        dequantize_mul_mat_vec_q8_0_aos_debug_kernel(
                            vx, y, dst, ncols, nrows, item_ct1, debug_buf);
                    });
            });
        } else {
            stream->parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                        vx, y, dst, ncols, nrows, item_ct1);
                });
        }
    }

    if (do_debug) {
        aos_kernel_debug_count++;
        std::vector<float> h(256);
        stream->memcpy(h.data(), debug_buf, 256 * sizeof(float)).wait();
        sycl::free(debug_buf, *stream);

        fprintf(stderr, "\n========== GPU KERNEL AoS DEBUG #%d ==========\n", aos_kernel_debug_count);
        fprintf(stderr, "ncols=%d nrows=%d iter_stride=%d vals_per_iter=%d\n",
                (int)h[0], (int)h[1], (int)h[2], (int)h[3]);

        // Per-iteration data
        fprintf(stderr, "\n--- Per-iteration accumulation (thread 0, row 0) ---\n");
        fprintf(stderr, "iter | i     | ib    | iter_sum       | running_total\n");
        fprintf(stderr, "-----|-------|-------|----------------|---------------\n");
        for (int iter = 0; iter < 32; iter++) {
            int base = 4 + iter * 4;
            if (h[base] == 0 && h[base+1] == 0 && h[base+2] == 0 && h[base+3] == 0 && iter > 0) break;
            fprintf(stderr, "%4d | %5d | %5d | %14.6f | %14.6f\n",
                    iter, (int)h[base], (int)h[base+1], h[base+2], h[base+3]);
        }

        fprintf(stderr, "\nfinal_tmp (before warp reduce) = %.6f\n", h[132]);
        fprintf(stderr, "final_dst (after warp reduce)  = %.6f\n", h[133]);

        // Raw data
        fprintf(stderr, "\n--- Raw qs values (first of each block) ---\n");
        for (int i = 0; i < 16; i++) {
            fprintf(stderr, "block[%2d].qs[0]=%4d  ", i, (int)h[134+i]);
            if ((i+1) % 4 == 0) fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n--- Raw d values (scale per block) ---\n");
        for (int i = 0; i < 16; i++) {
            fprintf(stderr, "d[%2d]=%.6f  ", i, h[198+i]);
            if ((i+1) % 4 == 0) fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n--- Y vector (first 26) ---\n");
        for (int i = 0; i < 26; i++) {
            fprintf(stderr, "y[%2d]=%.4f ", i, h[230+i]);
            if ((i+1) % 8 == 0) fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n========== END AoS DEBUG ==========\n\n");
        fflush(stderr);
    }
}

static void dequantize_mul_mat_vec_q2_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q3_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q4_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q5_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const sycl::range<3> block_dims(1, 1, QK_WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, item_ct1);
        });
}

void dequantize_mul_mat_vec_q6_K_sycl(const void *vx, const float *y,
                                      float *dst, const int ncols,
                                      const int nrows,
                                      dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

// Q6_K SoA (Structure of Arrays) DMMV kernel
// SoA layout: [all ql] [all qh] [all scales] [all d]
// ql: 128 bytes per block (QK_K/2)
// qh: 64 bytes per block (QK_K/4)
// scales: 16 bytes per block (QK_K/16)
// d: 2 bytes per block (sizeof(ggml_half))
static void dequantize_mul_mat_vec_q6_k_soa(const void * __restrict__ vx, const float * __restrict__ yy,
                                            float * __restrict__ dst, const int ncols, int nrows,
                                            const int64_t qh_offset, const int64_t scales_offset, const int64_t d_offset,
                                            const int row_low,
                                            const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    // For SoA layout, use absolute row index (row_low + row) for block addressing
    const int ib0 = (row_low + row) * num_blocks_per_row;

    // SoA base pointers
    const uint8_t * ql_base = (const uint8_t *)vx;
    const uint8_t * qh_base = ql_base + qh_offset;
    const int8_t * scales_base = (const int8_t *)(ql_base + scales_offset);
    const sycl::half * d_base = (const sycl::half *)(ql_base + d_offset);

#if QK_K == 256

    const int tid = item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION;
    const int ix = item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION;

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;
    const int in = tid - step*im;

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;
    const int is = 0;
#else
    const int l0 = 4 * in;
    const int is = in / 4;
#endif
    const int ql_offset_local = 64*im + l0;
    const int qh_offset_local = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int block_idx = ib0 + i;

        // SoA: calculate offsets for this block
        const uint8_t * ql = ql_base + block_idx * (QK_K/2) + ql_offset_local;
        const uint8_t * qh = qh_base + block_idx * (QK_K/4) + qh_offset_local;
        const int8_t  * s  = scales_base + block_idx * (QK_K/16) + s_offset;
        const float d = static_cast<float>(d_base[block_idx]);

        const float * y = yy + i * QK_K + y_offset;

#if K_QUANTS_PER_ITERATION == 1
        int8_t q0 = (int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32;
        int8_t q1 = (int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32;
        int8_t q2 = (int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32;
        int8_t q3 = (int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32;
        int8_t q4 = (int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32;
        int8_t q5 = (int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32;
        int8_t q6 = (int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32;
        int8_t q7 = (int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32;

        float sum = y[ 0] * s[0] * d * q0
                  + y[16] * s[1] * d * q1
                  + y[32] * s[2] * d * q2
                  + y[48] * s[3] * d * q3
                  + y[64] * s[4] * d * q4
                  + y[80] * s[5] * d * q5
                  + y[96] * s[6] * d * q6
                  +y[112] * s[7] * d * q7;

        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            int8_t q0 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q1 = (int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += y[l+ 0] * s[0] * d * q0
                 + y[l+32] * s[2] * d * q1
                 + y[l+64] * s[4] * d * q2
                 + y[l+96] * s[6] * d * q3;
        }
        tmp += sum;
#endif
    }

#else
    // QK_K == 64 path
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const int block_idx = ib0 + i;

        const uint8_t * ql = ql_base + block_idx * (QK_K/2) + step;
        const uint8_t * qh = qh_base + block_idx * (QK_K/4) + step;
        const int8_t  * s  = scales_base + block_idx * (QK_K/16);
        const float d = static_cast<float>(d_base[block_idx]);

        const float * y = yy + i * QK_K + step;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = QK_WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

// Q6_K SoA dispatch function
void dequantize_mul_mat_vec_q6_K_sycl_soa(const void *vx, const float *y,
                                          float *dst, const int ncols,
                                          const int nrows, const int64_t ne01,
                                          const int row_low,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Calculate SoA offsets using full tensor dimensions (ne01)
    // Q6_K SoA layout: [all ql] [all qh] [all scales] [all d]
    const int64_t nblocks = ne01 * (ncols / QK_K);
    const int64_t ql_size = nblocks * (QK_K / 2);     // 128 bytes per block
    const int64_t qh_size = nblocks * (QK_K / 4);     // 64 bytes per block
    const int64_t scales_size = nblocks * (QK_K / 16); // 16 bytes per block

    const int64_t qh_offset = ql_size;
    const int64_t scales_offset = ql_size + qh_size;
    const int64_t d_offset = ql_size + qh_size + scales_size;

    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, QK_WARP_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(QK_WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k_soa(vx, y, dst, ncols, nrows, qh_offset, scales_offset, d_offset, row_low, item_ct1);
        });
}

void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = nullptr; // dfloat == half

    bool src1_convert_f16 =
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

    if (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    // Debug: DMMV function is being called
    static int dmmv_call_count = 0;
    if (dmmv_call_count < 5 && std::getenv("GGML_SYCL_MMQ_DEBUG")) {
        fprintf(stderr, "[DMMV] Called for type=%d ncols=%ld nrows=%ld\n",
                (int)src0->type, (long)ne00, (long)row_diff);
        dmmv_call_count++;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            {
                // Use src0->extra directly, not dst->src[0]->extra
                // Both should be the same pointer, but src0 is passed directly from the graph node
                ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)src0->extra;
                reorder_mode mode = extra ? extra->optimized_feature.get_reorder() : reorder_mode::NONE;
                // Unconditional debug for AOS path investigation
                static int q4_0_dispatch_count = 0;
                if (std::getenv("GGML_SYCL_AOS_DEBUG") && q4_0_dispatch_count++ < 50) {
                    fprintf(stderr, "[Q4_0_DISPATCH] #%d tensor=%s mode=%d ne00=%lld row_diff=%lld extra=%p src0_dd_i=%p\n",
                            q4_0_dispatch_count, src0->name ? src0->name : "?", (int)mode,
                            (long long)ne00, (long long)row_diff, (void*)extra, (void*)src0_dd_i);
                    fflush(stderr);
                }

                if (std::getenv("GGML_SYCL_MMQ_DEBUG")) {
                    static int q4_0_mode_debug = 0;
                    if (q4_0_mode_debug++ < 10) {
                        fprintf(stderr, "[DMMV] Q4_0 reorder mode=%d (0=NONE,1=SOA,2=COALESCED)\n",
                                (int)mode);
                    }
                }

                {
                    block_q8_0 * src1_q8 = ggml_sycl_get_q8_0_cache(ne00, ctx.device, stream);
                    quantize_row_q8_0_sycl(src1_ddf_i, src1_q8, ne00, 1, ne00, stream);
                    stream->wait();

                    if (mode == reorder_mode::COALESCED) {
                        const int64_t ne01 = src0->ne[1];
                        GGML_SYCL_KTRACE("dmmv_q4_0_coalesced_q8_0", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long)ne00, (long long)row_diff, (long long)ne01, (long long)row_low);
                        dequantize_mul_mat_vec_q4_0_sycl_coalesced_q8_0(src0_dd_i, src1_q8,
                                                                       dst_dd_i, ne00, row_diff, ne01, row_low, stream);
                    } else if (mode == reorder_mode::SOA) {
                        const ggml_tensor * storage = get_storage_tensor(src0);
                        const int64_t storage_ne01 = storage->ne[1];
                        const int64_t ncols = storage->ne[0];
                        const int64_t nblocks = storage_ne01 * (ncols / QK4_0);
                        const int64_t total_qs_bytes = nblocks * (QK4_0 / 2);
                        const int64_t d_offset = total_qs_bytes;

                        const void * storage_base = ggml_sycl_get_data_ptr(storage, ctx.device);

                        int64_t view_row_offset = 0;
                        if (src0->view_src != nullptr) {
                            view_row_offset = src0->view_offs / src0->nb[1];
                        }
                        const int global_row_low = row_low + view_row_offset;

                        GGML_SYCL_KTRACE("dmmv_q4_0_soa_q8_0", " ne00=%lld row_diff=%lld storage_ne01=%lld global_row_low=%lld d_offset=%lld",
                                         (long long)ne00, (long long)row_diff, (long long)storage_ne01, (long long)global_row_low, (long long)d_offset);
                        dequantize_mul_mat_vec_q4_0_sycl_reorder_q8_0(storage_base, src1_q8, dst_dd_i,
                                                                     ne00, row_diff, d_offset, global_row_low, stream);
                    } else {
                        GGML_SYCL_KTRACE("dmmv_q4_0_aos_q8_0", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                        dequantize_mul_mat_vec_q4_0_sycl_q8_0(src0_dd_i, src1_q8, dst_dd_i, ne00, row_diff, stream);
                    }
                }
            }
            break;
        case GGML_TYPE_Q4_1:
            {
                block_q8_1 * src1_q8 = ggml_sycl_get_q8_1_cache(ne00, ctx.device, stream);
                quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_q8, ne00, 1, ne00, stream);
                stream->wait();
                GGML_SYCL_KTRACE("dmmv_q4_1_q8_1", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                dequantize_mul_mat_vec_q4_1_sycl_q8_1(src0_dd_i, src1_q8, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q5_0:
            {
                block_q8_0 * src1_q8 = ggml_sycl_get_q8_0_cache(ne00, ctx.device, stream);
                quantize_row_q8_0_sycl(src1_ddf_i, src1_q8, ne00, 1, ne00, stream);
                stream->wait();
                GGML_SYCL_KTRACE("dmmv_q5_0_q8_0", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                dequantize_mul_mat_vec_q5_0_sycl_q8_0(src0_dd_i, src1_q8, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q5_1:
            {
                block_q8_1 * src1_q8 = ggml_sycl_get_q8_1_cache(ne00, ctx.device, stream);
                quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_q8, ne00, 1, ne00, stream);
                stream->wait();
                GGML_SYCL_KTRACE("dmmv_q5_1_q8_1", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                dequantize_mul_mat_vec_q5_1_sycl_q8_1(src0_dd_i, src1_q8, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q8_0:
            {
                // Use src0->extra directly (same as Q4_0 fix)
                ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)src0->extra;
                reorder_mode mode = extra ? extra->optimized_feature.get_reorder() : reorder_mode::NONE;

                // DEBUG: Track SoA dispatch for Q8_0
                if (g_ggml_sycl_debug) {
                    fprintf(stderr, "[DMMV-Q8_0-DBG] src0='%s' extra=%p mode=%d (0=NONE,1=SOA,2=COALESCED)\n",
                            src0->name ? src0->name : "?", extra, (int)mode);
                }

                block_q8_0 * src1_q8 = ggml_sycl_get_q8_0_cache(ne00, ctx.device, stream);
                quantize_row_q8_0_sycl(src1_ddf_i, src1_q8, ne00, 1, ne00, stream);
                stream->wait();

                if (mode == reorder_mode::COALESCED) {
                    const int64_t ne01 = src0->ne[1];
                    GGML_SYCL_KTRACE("dmmv_q8_0_coalesced_q8_0", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                     (long long)ne00, (long long)row_diff, (long long)ne01, (long long)row_low);
                    dequantize_mul_mat_vec_q8_0_sycl_coalesced_q8_0(src0_dd_i, src1_q8,
                                                                   dst_dd_i, ne00, row_diff, ne01, row_low, stream);
                } else if (mode == reorder_mode::SOA) {
                    // Q8_0 SoA layout: [all qs: nblocks * 32 bytes][all d: nblocks * 2 bytes]
                    // CRITICAL: Must use storage tensor dimensions for offset calculation, not view dimensions
                    const ggml_tensor * storage = get_storage_tensor(src0);
                    const int64_t storage_ne01 = storage->ne[1];  // storage tensor rows (not view rows!)
                    const int64_t ncols = storage->ne[0];
                    // Q8_0: explicit nblocks pattern (matches Q6_K)
                    const int64_t nblocks = storage_ne01 * (ncols / QK8_0);
                    const int64_t total_qs_bytes = nblocks * QK8_0;  // 32 bytes per block
                    const int64_t d_offset = total_qs_bytes;

                    // Get storage tensor base pointer
                    const void * storage_base = ggml_sycl_get_data_ptr(storage, ctx.device);

                    // Calculate global row_low: row_low is relative to src0, need to add view offset
                    int64_t view_row_offset = 0;
                    if (src0->view_src != nullptr) {
                        // src0 is a view - calculate its row offset within storage
                        // Use nb[1] (row stride) to calculate row offset from byte offset
                        view_row_offset = src0->view_offs / src0->nb[1];
                    }
                    const int global_row_low = row_low + view_row_offset;

                    GGML_SYCL_KTRACE("dmmv_q8_0_soa_q8_0", " ne00=%lld row_diff=%lld storage_ne01=%lld global_row_low=%lld d_offset=%lld",
                                     (long long)ne00, (long long)row_diff, (long long)storage_ne01, (long long)global_row_low, (long long)d_offset);
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_q8_0(storage_base, src1_q8, dst_dd_i,
                                                                  ne00, row_diff, d_offset, global_row_low, stream);
                } else {
                    GGML_SYCL_KTRACE("dmmv_q8_0_aos_q8_0", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                    dequantize_mul_mat_vec_q8_0_sycl_q8_0(src0_dd_i, src1_q8, dst_dd_i, ne00, row_diff, stream);
                }
            }
            break;
        case GGML_TYPE_Q2_K:
            GGML_SYCL_KTRACE("dmmv_q2_k", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
            dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            GGML_SYCL_KTRACE("dmmv_q3_k", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
            dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            {
                // Use src0->extra directly (same as Q4_0/Q8_0 fix)
                ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)src0->extra;
                if (extra && extra->optimized_feature.is_reordered()) {
                    // reorder is currently not supported for dmmv
                    GGML_ABORT("Unimplemented dequantize case case for q4_k reorder");
                } else {
                    GGML_SYCL_KTRACE("dmmv_q4_k_aos", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                    dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            }
            break;
        case GGML_TYPE_Q5_K:
            GGML_SYCL_KTRACE("dmmv_q5_k", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
            dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            {
                ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)src0->extra;
                reorder_mode mode = extra ? extra->optimized_feature.get_reorder() : reorder_mode::NONE;

                if (mode == reorder_mode::COALESCED) {
                    const ggml_tensor * storage = get_storage_tensor(src0);
                    const int64_t storage_ne01 = storage->ne[1];

                    const void * storage_base = ggml_sycl_get_data_ptr(storage, ctx.device);

                    int64_t view_row_offset = 0;
                    if (src0->view_src != nullptr) {
                        view_row_offset = src0->view_offs / src0->nb[1];
                    }
                    const int global_row_low = row_low + view_row_offset;

                    GGML_SYCL_KTRACE("dmmv_q6_k_coalesced", " ne00=%lld row_diff=%lld storage_ne01=%lld global_row_low=%lld",
                                     (long long)ne00, (long long)row_diff, (long long)storage_ne01, (long long)global_row_low);
                    dequantize_mul_mat_vec_q6_K_sycl_coalesced(storage_base, src1_ddf_i, dst_dd_i,
                                                              ne00, row_diff, storage_ne01, global_row_low, stream);
                } else if (mode == reorder_mode::SOA) {
                    // Q6_K SoA layout: [all ql] [all qh] [all scales] [all d]
                    // CRITICAL: Must use storage tensor dimensions for offset calculation, not view dimensions
                    const ggml_tensor * storage = get_storage_tensor(src0);
                    const int64_t storage_ne01 = storage->ne[1];  // storage tensor rows (not view rows!)

                    // Get storage tensor base pointer
                    const void * storage_base = ggml_sycl_get_data_ptr(storage, ctx.device);

                    // Calculate global row_low: row_low is relative to src0, need to add view offset
                    int64_t view_row_offset = 0;
                    if (src0->view_src != nullptr) {
                        // src0 is a view - calculate its row offset within storage
                        // Use nb[1] (row stride) to calculate row offset from byte offset
                        view_row_offset = src0->view_offs / src0->nb[1];
                    }
                    const int global_row_low = row_low + view_row_offset;

                    GGML_SYCL_KTRACE("dmmv_q6_k_soa", " ne00=%lld row_diff=%lld storage_ne01=%lld global_row_low=%lld",
                                     (long long)ne00, (long long)row_diff, (long long)storage_ne01, (long long)global_row_low);
                    dequantize_mul_mat_vec_q6_K_sycl_soa(storage_base, src1_ddf_i, dst_dd_i, ne00, row_diff, storage_ne01, global_row_low, stream);
                } else {
                    GGML_SYCL_KTRACE("dmmv_q6_k_aos", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
                    dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            }
            break;
        case GGML_TYPE_F16:
            GGML_SYCL_KTRACE("dmmv_f16", " ne00=%lld row_diff=%lld", (long long)ne00, (long long)row_diff);
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        default:
            printf("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(ctx);
}
