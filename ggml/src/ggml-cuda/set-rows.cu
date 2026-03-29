#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

// ---- TurboQuant3 set_rows: GROUP_SIZE-element groups with WHT rotation + norm correction ----
//
// Templated on GROUP_SIZE (128 or 64).
// Parallel kernel: one CUDA block per group, GROUP_SIZE threads per block.
// Thread j handles element j within the group.
//
// Steps (all parallel):
//   1. Load element j from global memory
//   2. Parallel L2 norm (warp reduce + inter-warp via shared memory)
//   3. Normalize
//   4. Forward WHT (log2(GROUP_SIZE) butterfly stages, shared memory)
//   5. Quantize element j to 3-bit centroid index
//   6. Pack qs (warp shuffle) and signs (__ballot_sync) into turbo3 block, no atomics
//   7. Parallel reconstruction norm (same pattern as step 2)
//   8. Write corrected norm (one thread per sub-block)

template <typename idx_t, int GROUP_SIZE>
__launch_bounds__(128)  // max of 128 or 64
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    static_assert(GROUP_SIZE == 128 || GROUP_SIZE == 64, "GROUP_SIZE must be 128 or 64");

    // blockIdx.x = flat group index; threadIdx.x = element within group (0..GROUP_SIZE-1)
    const int j = threadIdx.x;

    // Decode blockIdx.x → (i_grp, i01, i02, i03)
    constexpr int blocks_per_group = GROUP_SIZE / QK_TURBO3;
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo3_0 * blk_base    = dst_row_ptr + i_grp * blocks_per_group;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[GROUP_SIZE];
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    __syncthreads();

    // ---- InnerQ: calibrate on original (unscaled) values ----
    if (d_innerq_calibrating) {
        atomicAdd(&d_innerq_sq_accum[j], x[j] * x[j]);
        if (j == 0) atomicAdd(&d_innerq_count, 1);
    }

    // ---- InnerQ: apply channel scale (only when active) ----
    if (d_innerq_active) {
        x[j] *= d_innerq_scale[j];
    }
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = GROUP_SIZE / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 → butterfly → signs2, normalized) ----
    if (GROUP_SIZE == 128) {
        x[j] *= TURBO_WHT_SIGNS1[j];
    } else {
        x[j] *= TURBO_WHT_SIGNS1_64[j];
    }
    __syncthreads();

#define WHT_STAGE_SHARED(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    // Butterfly stages: loop from h=1 to h<GROUP_SIZE, doubling each time
    WHT_STAGE_SHARED(1)
    WHT_STAGE_SHARED(2)
    WHT_STAGE_SHARED(4)
    WHT_STAGE_SHARED(8)
    WHT_STAGE_SHARED(16)
    WHT_STAGE_SHARED(32)
    if (GROUP_SIZE == 128) { WHT_STAGE_SHARED(64) }
#undef WHT_STAGE_SHARED

    constexpr float inv_sqrt_group = (GROUP_SIZE == 128) ? 0.08838834764831845f : 0.125f;
    if (GROUP_SIZE == 128) {
        x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j];
    } else {
        x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2_64[j];
    }
    __syncthreads();

    // ---- Step 5: Quantize element j ----
    const float rv = x[j];
    const uint8_t idx = turbo_nearest_centroid_3bit(rv);

    // ---- Step 6: Pack qs and signs (warp-cooperative, no atomics) ----
    // Warp warp_id handles turbo3 sub-block warp_id (elements warp_id*32 .. warp_id*32+31).
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;
    block_turbo3_0 * blk = blk_base + warp_id;

    // Pack qs: 4 elements per byte, 2 bits each.
    // All 4 threads in a qs-group gather their low2 bits via shuffle.
    const int qs_byte_idx = lane / 4;
    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_low2, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[qs_byte_idx] = qs_byte;

    // Pack signs: 8 elements per byte, 1 bit each.  __ballot_sync across warp.
    const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
    const int signs_byte_idx = lane / 8;
    const uint8_t signs_byte = (uint8_t)((ballot >> (signs_byte_idx * 8)) & 0xFF);
    if (lane % 8 == 0) blk->signs[signs_byte_idx] = signs_byte;

    // ---- Step 7: Reconstruction norm (parallel, same pattern as step 2) ----
    const float c = TURBO_CENTROIDS_3BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm (one thread per turbo3 sub-block) ----
    if (lane == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// ---- TurboQuant3 tail kernel: straight 3-bit quantize without WHT rotation ----
//
// For head dims not divisible by 128 (e.g. 576 = 4*128 + 64), the remainder
// elements can't use the 128-element WHT. They are quantised directly into
// standard turbo3 blocks.  Q is also NOT rotated for these positions (the graph
// guards on ne[0] % 128), so <Q_tail, K_tail> stays in the original space.
//
// One CUDA block per row, with tail_size threads (must be multiple of 32).

template <typename idx_t>
static __global__ void k_set_rows_turbo3_tail(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int tail_size) {

    const int j = threadIdx.x;  // 0 .. tail_size-1

    // Decode blockIdx.x → (i01, i02, i03)
    int64_t tmp = blockIdx.x;
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne12;
    const int64_t i03 = tmp / ne12;

    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    // Tail starts after all full 128-element groups
    const int64_t n_full = ne00 / QK_TURBO3_GROUP;
    const int64_t tail_start = n_full * QK_TURBO3_GROUP;
    block_turbo3_0 * blk_base = dst_row_ptr + n_full * (QK_TURBO3_GROUP / QK_TURBO3);

    // ---- Load ----
    const float val = src_row[tail_start + j];

    // ---- L2 norm over the tail group (warp reduce + inter-warp) ----
    const int n_warps = tail_size / WARP_SIZE;
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;

    __shared__ float warp_accum[4];  // max 3 warps (tail ≤ 96)
    float v2 = val * val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (lane == 0) warp_accum[warp_id] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Normalize (no WHT!) ----
    const float rv = val * inv_norm;

    // ---- Quantize ----
    const uint8_t idx = turbo_nearest_centroid_3bit(rv);

    // ---- Pack qs and signs (same warp-cooperative logic) ----
    block_turbo3_0 * blk = blk_base + warp_id;

    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_low2, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[lane / 4] = qs_byte;

    const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
    const int signs_byte_idx = lane / 8;
    const uint8_t signs_byte = (uint8_t)((ballot >> (signs_byte_idx * 8)) & 0xFF);
    if (lane % 8 == 0) blk->signs[signs_byte_idx] = signs_byte;

    // ---- Reconstruction norm ----
    const float c = TURBO_CENTROIDS_3BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (lane == 0) warp_accum[warp_id] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (lane == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turbo3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBO3 == 0);  // must be block-aligned (32)

    cudaStream_t stream = ctx.stream();

    // Read WHT group size from op_params (set by llama-kv-cache.cpp based on head_dim).
    // Default to 128 if not set (backward compat with head_dim=128 models).
    int group_size = 128;
    memcpy(&group_size, dst->op_params, sizeof(int));
    if (group_size != 64 && group_size != 128) group_size = 128;
    GGML_ASSERT(ne00 % group_size == 0);

    const int64_t n_full_groups   = ne00 / group_size;
    const int     tail_size       = (int)(ne00 % group_size);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    // InnerQ: check/finalize calibration before kernel launch
    turbo_innerq_check_finalize(group_size, ne00);

    // Launch 1: full groups with WHT rotation
    if (n_full_groups > 0) {
        const int64_t ne_total = n_full_groups * ne01 * ne02 * ne03;
        if (group_size == 128) {
            k_set_rows_turbo3<idx_t, 128><<<(int)ne_total, 128, 0, stream>>>(
                src0_d, src1_d, (block_turbo3_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        } else {
            k_set_rows_turbo3<idx_t, 64><<<(int)ne_total, 64, 0, stream>>>(
                src0_d, src1_d, (block_turbo3_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        }
    }

    // Launch 2: tail elements (no WHT, straight quantize)
    // Not needed for 64-aligned dims but kept for potential future use
    if (tail_size > 0) {
        GGML_ASSERT(tail_size % QK_TURBO3 == 0);  // tail must be block-aligned
        const int64_t n_rows = ne01 * ne02 * ne03;
        k_set_rows_turbo3_tail<idx_t><<<(int)n_rows, tail_size, 0, stream>>>(
            src0_d, src1_d, (block_turbo3_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3, tail_size);
    }
}

// ---- TurboQuant2 set_rows: GROUP_SIZE-element groups with WHT rotation + norm correction ----
//
// Same structure as turbo3 but 2-bit quantization only (no signs byte).

template <typename idx_t, int GROUP_SIZE>
__launch_bounds__(128)
static __global__ void k_set_rows_turbo2(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    static_assert(GROUP_SIZE == 128 || GROUP_SIZE == 64, "GROUP_SIZE must be 128 or 64");

    const int j = threadIdx.x;

    constexpr int blocks_per_group = GROUP_SIZE / QK_TURBO2;
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo2_0 * dst_row_ptr = (block_turbo2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo2_0 * blk_base    = dst_row_ptr + i_grp * blocks_per_group;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[GROUP_SIZE];
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    __syncthreads();

    // ---- InnerQ: calibrate on original (unscaled) values ----
    if (d_innerq_calibrating) {
        atomicAdd(&d_innerq_sq_accum[j], x[j] * x[j]);
        if (j == 0) atomicAdd(&d_innerq_count, 1);
    }

    // ---- InnerQ: apply channel scale (only when active) ----
    if (d_innerq_active) {
        x[j] *= d_innerq_scale[j];
    }
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = GROUP_SIZE / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT ----
    if (GROUP_SIZE == 128) {
        x[j] *= TURBO_WHT_SIGNS1[j];
    } else {
        x[j] *= TURBO_WHT_SIGNS1_64[j];
    }
    __syncthreads();

#define WHT_STAGE_SHARED_T2(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T2(1)
    WHT_STAGE_SHARED_T2(2)
    WHT_STAGE_SHARED_T2(4)
    WHT_STAGE_SHARED_T2(8)
    WHT_STAGE_SHARED_T2(16)
    WHT_STAGE_SHARED_T2(32)
    if (GROUP_SIZE == 128) { WHT_STAGE_SHARED_T2(64) }
#undef WHT_STAGE_SHARED_T2

    constexpr float inv_sqrt_group = (GROUP_SIZE == 128) ? 0.08838834764831845f : 0.125f;
    if (GROUP_SIZE == 128) {
        x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j];
    } else {
        x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2_64[j];
    }
    __syncthreads();

    // ---- Step 5: Quantize element j to 2-bit centroid ----
    const float rv = x[j];
    const uint8_t idx = turbo_nearest_centroid_2bit(rv);

    // ---- Step 6: Pack qs (warp-cooperative, no atomics) ----
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;
    block_turbo2_0 * blk = blk_base + warp_id;

    // Pack qs: 4 elements per byte, 2 bits each.
    const uint8_t my_bits = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_bits, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[lane / 4] = qs_byte;

    // No signs packing needed for turbo2

    // ---- Step 7: Reconstruction norm ----
    const float c = TURBO_CENTROIDS_2BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm ----
    if (lane == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// ---- TurboQuant2 tail kernel: straight 2-bit quantize without WHT rotation ----

template <typename idx_t>
static __global__ void k_set_rows_turbo2_tail(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int tail_size) {

    const int j = threadIdx.x;

    int64_t tmp = blockIdx.x;
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne12;
    const int64_t i03 = tmp / ne12;

    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo2_0 * dst_row_ptr = (block_turbo2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    const int64_t n_full = ne00 / QK_TURBO2_GROUP;
    const int64_t tail_start = n_full * QK_TURBO2_GROUP;
    block_turbo2_0 * blk_base = dst_row_ptr + n_full * (QK_TURBO2_GROUP / QK_TURBO2);

    // ---- Load ----
    const float val = src_row[tail_start + j];

    // ---- L2 norm ----
    const int n_warps = tail_size / WARP_SIZE;
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;

    __shared__ float warp_accum[4];
    float v2 = val * val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (lane == 0) warp_accum[warp_id] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Normalize (no WHT!) ----
    const float rv = val * inv_norm;

    // ---- Quantize ----
    const uint8_t idx = turbo_nearest_centroid_2bit(rv);

    // ---- Pack qs ----
    block_turbo2_0 * blk = blk_base + warp_id;

    const uint8_t my_bits = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_bits, (lane & ~3) + k);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[lane / 4] = qs_byte;

    // ---- Reconstruction norm ----
    const float c = TURBO_CENTROIDS_2BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (lane == 0) warp_accum[warp_id] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (lane == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(ne00);
}

template<typename idx_t>
static void set_rows_cuda_turbo2(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBO2 == 0);

    cudaStream_t stream = ctx.stream();

    int group_size = 128;
    memcpy(&group_size, dst->op_params, sizeof(int));
    if (group_size != 64 && group_size != 128) group_size = 128;
    GGML_ASSERT(ne00 % group_size == 0);

    const int64_t n_full_groups   = ne00 / group_size;
    const int     tail_size       = (int)(ne00 % group_size);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    // InnerQ: check/finalize calibration before kernel launch
    turbo_innerq_check_finalize(group_size, ne00);

    if (n_full_groups > 0) {
        const int64_t ne_total = n_full_groups * ne01 * ne02 * ne03;
        if (group_size == 128) {
            k_set_rows_turbo2<idx_t, 128><<<(int)ne_total, 128, 0, stream>>>(
                src0_d, src1_d, (block_turbo2_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        } else {
            k_set_rows_turbo2<idx_t, 64><<<(int)ne_total, 64, 0, stream>>>(
                src0_d, src1_d, (block_turbo2_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        }
    }

    if (tail_size > 0) {
        GGML_ASSERT(tail_size % QK_TURBO2 == 0);
        const int64_t n_rows = ne01 * ne02 * ne03;
        k_set_rows_turbo2_tail<idx_t><<<(int)n_rows, tail_size, 0, stream>>>(
            src0_d, src1_d, (block_turbo2_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3, tail_size);
    }
}

// ---- TurboQuant4 set_rows: 128-element groups with WHT rotation + 4-bit quantization ----
//
// turbo4 block size IS the WHT group size (128), so 1 CUDA block = 1 turbo4 block.
// 128 threads per block, thread j handles element j.
// 4-bit centroids (16 values), nibble packed: qs[j/2] |= (idx & 0xF) << ((j%2)*4)

template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turbo4(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    // blockIdx.x = flat block index; threadIdx.x = element within block (0..127)
    const int j = threadIdx.x;

    // Decode blockIdx.x → (i_blk, i01, i02, i03)
    const int64_t n_blocks_per_row = ne00 / QK_TURBO4;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 * dst_row_ptr = (block_turbo4_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo4_0 * blk = dst_row_ptr + i_blk;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBO4 + j];
    __syncthreads();

    // ---- InnerQ: calibrate on original (unscaled) values ----
    if (d_innerq_calibrating) {
        atomicAdd(&d_innerq_sq_accum[j], x[j] * x[j]);
        if (j == 0) atomicAdd(&d_innerq_count, 1);
    }

    // ---- InnerQ: apply channel scale (only when active) ----
    if (d_innerq_active) {
        x[j] *= d_innerq_scale[j];
    }
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = 128 / WARP_SIZE;  // = 4
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 → butterfly → signs2, normalized) ----
    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T4(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T4(1)
    WHT_STAGE_SHARED_T4(2)
    WHT_STAGE_SHARED_T4(4)
    WHT_STAGE_SHARED_T4(8)
    WHT_STAGE_SHARED_T4(16)
    WHT_STAGE_SHARED_T4(32)
    WHT_STAGE_SHARED_T4(64)
#undef WHT_STAGE_SHARED_T4

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Step 5: Quantize element j to 4-bit centroid ----
    const float rv = x[j];
    const uint8_t idx = turbo_nearest_centroid_4bit(rv);

    // ---- Step 6: Pack qs (nibble packed, warp-cooperative) ----
    // 2 elements per byte, 4 bits each.
    // Thread pairs (j, j+1) share a qs byte.
    const int lane = j % WARP_SIZE;
    const uint8_t my_nibble = idx & 0xF;
    uint8_t qs_byte = 0;
    // Gather nibble from partner thread
    uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1);
    if (j % 2 == 0) {
        qs_byte = my_nibble | (partner_nibble << 4);
        blk->qs[j / 2] = qs_byte;
    }

    // ---- Step 7: Reconstruction norm (parallel) ----
    const float c = TURBO_CENTROIDS_4BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm and zero rnorm (one thread) ----
    if (j == 0) {
        blk->norm  = __float2half(corrected_norm);
        blk->rnorm = __float2half(0.0f);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turbo4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBO4 == 0);  // must be block-aligned (128)

    cudaStream_t stream = ctx.stream();

    // turbo4 block size = WHT group size = 128, always
    const int64_t n_blocks = ne00 / QK_TURBO4;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    // InnerQ: check/finalize calibration before kernel launch
    turbo_innerq_check_finalize(QK_TURBO4, ne00);

    if (n_blocks > 0) {
        const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;
        k_set_rows_turbo4<idx_t><<<(int)ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turbo4_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO2_0) {
        set_rows_cuda_turbo2<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
