#include "common.cuh"
#include "vecdotq.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);


struct __builtin_align__(32) float8
{
    float x; float y; float z; float w;
    float p; float q; float r; float s;
};

static __device__ __forceinline__ float vec_dot_nvfp4_q8_1_repacked_full(
                                        const void * __restrict__ vbq,
                                        const block_q8_1 * __restrict__ bq8_1,
                                        const int32_t & kbx,
                                        const uint3 & blocks_per_matrix) {
    const int32_t matrix = fastdiv((uint32_t) kbx, blocks_per_matrix);
    const int32_t ib     = kbx - matrix * blocks_per_matrix.z;

    const uint8_t * matrix_base = (const uint8_t *) vbq + matrix * blocks_per_matrix.z * sizeof(block_nvfp4);
    const uint8_t * qs_base     = matrix_base;
    const uint8_t * d_base      = qs_base + blocks_per_matrix.z * (QK_NVFP4 / 2);

    const uint8_t * qs = qs_base + ib * (QK_NVFP4 / 2);
    const uint8_t * d  = d_base  + ib * (QK_NVFP4 / QK_NVFP4_SUB);

    int qs_i[QI_NVFP4];
    ggml_cuda_memcpy_1<16>(qs_i + 0, qs +  0);
    ggml_cuda_memcpy_1<16>(qs_i + 4, qs + 16);

    float sum = 0.0f;
#pragma unroll
    for (int is = 0; is < QK_NVFP4 / QK_NVFP4_SUB; ++is) {
        const int2 v0 = get_int_from_table_16(qs_i[2*is + 0], kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(qs_i[2*is + 1], kvalues_mxfp4);
        const block_q8_1 * bq8 = bq8_1 + (is >> 1);
        const int32_t i8 = ((is & 1) << 2);

        int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);

        const float scale = ggml_cuda_ue4m3_to_fp32(d[is]) * __low2float(bq8->ds);
        sum += scale * float(sumi);
    }

    return sum;
}

static __device__ __forceinline__ float vec_dot_nvfp4x2_f32(
                                        const __nv_fp4x2_storage_t v,
                                        const float2 & f) {
    const __half2_raw hraw2 = __nv_cvt_fp4x2_to_halfraw2(v, __NV_E2M1);
    const __half2 h2 = static_cast<__half2>(hraw2);
    const float2 vf = __half22float2(h2);
    return vf.x*f.x + vf.y*f.y;
}

static __device__ __forceinline__ float vec_dot_nvfp4_f32_repacked_subblock(
                                        const void * __restrict__ vbq,
                                        const float * __restrict__ y,
                                        const int32_t & kbx,
                                        const uint3 & blocks_per_matrix,
                                        const int32_t & lane) {
    static_assert(ggml_cuda_get_physical_warp_size() == 32, "subblock NVFP4 F32 path assumes 32 lanes");
    static_assert(QK_NVFP4 == 64, "subblock NVFP4 F32 path assumes 64-element NVFP4 blocks");
    static_assert(QK_NVFP4_SUB == 16, "subblock NVFP4 F32 path assumes 16-element NVFP4 sub-blocks");

    const int32_t group = lane >> 2;
    const int32_t sub   = lane & 3;
    const int32_t kbx_g = kbx + group;

    const int32_t matrix = fastdiv((uint32_t) kbx_g, blocks_per_matrix);
    const int32_t ib     = kbx_g - matrix * blocks_per_matrix.z;

    const uint8_t * matrix_base = (const uint8_t *) vbq + matrix * blocks_per_matrix.z * sizeof(block_nvfp4);
    const uint8_t * qs_base     = matrix_base;
    const uint8_t * d_base      = qs_base + blocks_per_matrix.z * (QK_NVFP4 / 2);

    const uint8_t * qs = qs_base + ib * (QK_NVFP4 / 2) + sub * (QK_NVFP4_SUB / 2);
    const uint8_t * d  = d_base  + ib * (QK_NVFP4 / QK_NVFP4_SUB);

    uint64_t qs_u64;
    ggml_cuda_memcpy_1<8>(&qs_u64, qs);

    const float4 * yf = (const float4 *) (y + group * QK_NVFP4 + sub * QK_NVFP4_SUB);
    const float8 y0 = *((const float8 *) &yf[0]);
    const float8 y2 = *((const float8 *) &yf[2]);

    const uint8_t * qs_bytes = (const uint8_t *) &qs_u64;
    float sumf = 0.0f;
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[0]), make_float2(y0.x, y0.y));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[1]), make_float2(y0.z, y0.w));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[2]), make_float2(y0.p, y0.q));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[3]), make_float2(y0.r, y0.s));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[4]), make_float2(y2.x, y2.y));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[5]), make_float2(y2.z, y2.w));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[6]), make_float2(y2.p, y2.q));
    sumf += vec_dot_nvfp4x2_f32(static_cast<__nv_fp4x2_storage_t>(qs_bytes[7]), make_float2(y2.r, y2.s));

    return ggml_cuda_ue4m3_to_fp32(d[sub]) * sumf;
}