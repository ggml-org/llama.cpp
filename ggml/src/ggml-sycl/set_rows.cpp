#include "set_rows.hpp"
#include "cpy.hpp"
#include "turbo-quants.hpp"

namespace utils {
template<typename T>
static constexpr bool is_arithmetic_v() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ifdef GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#endif
        ;
}
}

template<typename TIn, typename TOut>
static inline std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), void>
convert (const char* src, char* dst) {
    auto src_val = *reinterpret_cast<const TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

template <typename idx_t, int GROUP_SIZE, typename block_t, int QK, void (*quantize_fn)(float, block_t *, const sycl::nd_item<1> &), uint8_t (*nearest_centroid_fn)(float), const float * CENTROIDS>
static void k_set_rows_turbo_generic(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t n_indices,
        const int64_t s1,
        const sycl::nd_item<1> &item_ct1,
        float *shared_mem) {

    const int j = item_ct1.get_local_id(0);
    const int sg_id = item_ct1.get_sub_group().get_group_id()[0];
    const int n_sg = item_ct1.get_sub_group().get_group_range()[0];

    // Total groups = (n_indices) * n_groups_per_row
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = item_ct1.get_group(0);
    
    if (g >= n_indices * n_groups_per_row) return;

    const int64_t i_row = g / n_groups_per_row;
    const int64_t i_grp = g % n_groups_per_row;

    const idx_t dst_row = src1[i_row];
    const float * src_row = src0 + i_row * ne00;
    
    block_t * blk_base = (block_t *)((char *)dst + dst_row*s1) + i_grp * (GROUP_SIZE / QK);

    float *x = shared_mem;
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    float v = x[j];
    float v2 = v * v;
    auto sg = item_ct1.get_sub_group();
    v2 = sycl::reduce_over_group(sg, v2, sycl::plus<>());
    
    float *warp_accum = x + 128; 
    if (sg.get_local_id()[0] == 0) {
        warp_accum[sg_id] = v2;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    float grp_norm;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_sg; w++) total += warp_accum[w];
        warp_accum[0] = total; 
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    grp_norm = sycl::sqrt(warp_accum[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    v *= inv_norm;
    v *= TURBO_WHT_SIGNS1[j];
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    turbo_wht<128>(v, item_ct1, x);
    v *= TURBO_WHT_SIGNS2[j];
    v *= (float)(1.0f / sycl::sqrt(128.0f));

    block_t * blk = blk_base + (j / QK);
    quantize_fn(v, blk, item_ct1);

    const uint8_t idx = nearest_centroid_fn(v);
    const float c = CENTROIDS[idx];
    float rc = c * c;
    rc = sycl::reduce_over_group(sg, rc, sycl::plus<>());
    if (sg.get_local_id()[0] == 0) {
        warp_accum[sg_id] = rc;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    float recon_norm;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_sg; w++) total += warp_accum[w];
        warp_accum[0] = total;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    recon_norm = sycl::sqrt(warp_accum[0]);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    const int elem_in_block = j % QK;
    if (elem_in_block == 0) blk->norm = (sycl::half)corrected_norm;
}

template <typename idx_t, typename block_t, int QK, void (*quantize_fn)(float, block_t *, const sycl::nd_item<1> &), uint8_t (*nearest_centroid_fn)(float), const float * CENTROIDS>
static void set_rows_sycl_turbo(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
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
        queue_ptr stream) {

    const int64_t n_groups_per_row = ne00 / 128;
    const int64_t n_indices = ne11; 
    
    const int64_t grid_size = n_indices * n_groups_per_row;
    const int block_size = 128;

    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(128 + 32), h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
            k_set_rows_turbo_generic<idx_t, 128, block_t, QK, quantize_fn, nearest_centroid_fn, CENTROIDS>(
                src0, src1, dst,
                ne00, n_indices, s1,
                item_ct1,
                shared_mem.get_multi_ptr<sycl::access::decorated::no>().get()
            );
        });
    });
}

template <typename TIdx, typename blockType, int qk, cpy_kernel_t cpyblck>
static void set_rows_sycl_q(const char * __restrict__ src0_d,
                            const TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            const int64_t ne00,
                            const int64_t ne01,
                            const int64_t ne02,
                            const int64_t ne03,
                            const int64_t ne10,
                            const int64_t ne11,
                            const int64_t ne12,
                            const int64_t ne13,
                            // strides for src0
                            const size_t  nb00,
                            const size_t  nb01,
                            const size_t  nb02,
                            const size_t  nb03,
                            // strides for src1
                            const size_t  nb10,
                            const size_t  nb11,
                            const size_t  nb12,
                            const size_t  nb13,
                            // strides for dst
                            const size_t  nb1,
                            const size_t  nb2,
                            const size_t  nb3,
                            queue_ptr     stream) {
    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr int block_size   = 256;
    const int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        const int64_t i = item_ct1.get_global_linear_id();
        if (i >= total_blocks) {
            return;
        }
        const int64_t i_base      = i * qk;
        const int64_t i03         = i_base / (ne00 * ne01 * ne02);
        const int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        const int64_t i02         = rem1 / (ne00 * ne01);
        const int64_t rem2        = rem1 - i02 * ne00 * ne01;
        const int64_t i01         = rem2 / ne00;
        const int64_t i00         = rem2 - i01 * ne00;
        const int64_t i12         = i03 % ne12;
        const int64_t i11         = i02 % ne11;
        const int64_t i10         = i01;
        const size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        const char *  src_block   = src0_d + src_offset + i00 * sizeof(float);
        const size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        const int64_t dst_row     = src1_d[src1_offset / sizeof(TIdx)];
        const size_t  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * sizeof(blockType);
        char * dst_block = reinterpret_cast<char *>(reinterpret_cast<char *>(dst_d) + dst_offset);
        cpyblck(src_block, dst_block);
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template<typename TIn, typename TIdx, typename TOut>
static void k_set_rows(
        const char * __restrict__ src0, const TIdx * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        const int64_t total_elements,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_linear_id();
    if (i >= total_elements) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(const TIdx *)((const char *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    const char * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    const char * src_elem = src0_row + i00 * src_type_size;
    char * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    char * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template<typename TIn, typename TIdx, typename TOut>
static void set_rows_sycl(
        const char * src0_d, const TIdx * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        queue_ptr stream) {

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr int block_size = 64;
    const int64_t grid_size = ceil_div(total_elements, block_size);

    stream->parallel_for(
        sycl::nd_range<1>(grid_size * block_size, block_size),
        [=](sycl::nd_item<1> item_ct1) {
            k_set_rows<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_size, dst_type_size,
                total_elements,
                item_ct1
            );
        }
    );
}

template<typename TIn, typename TIdx>
static void set_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const char * src0_d = (const char *)src0->data;
    const TIdx * src1_d = (const TIdx *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, float>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(float),
                stream
            );
            break;
        case GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::half),
                stream
            );
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::ext::oneapi::bfloat16),
                stream
            );
            break;
#endif
        case GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(src0_d, src1_d, (block_q8_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(src0_d, src1_d, (block_q5_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(src0_d, src1_d, (block_q5_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(src0_d, src1_d, (block_q4_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(src0_d, src1_d, (block_q4_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(src0_d, src1_d, (block_iq4_nl *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO2_0:
            set_rows_sycl_turbo<TIdx, block_turbo2_0, QK_TURBO2, quantize_turbo2_0<1>, turbo_nearest_centroid_2bit, TURBO_CENTROIDS_2BIT>(
                (const float *)src0_d, src1_d, (block_turbo2_0 *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
            break;
        case GGML_TYPE_TURBO3_0:
            set_rows_sycl_turbo<TIdx, block_turbo3_0, QK_TURBO3, quantize_turbo3_0<1>, turbo_nearest_centroid_3bit, TURBO_CENTROIDS_3BIT>(
                (const float *)src0_d, src1_d, (block_turbo3_0 *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
            break;
        case GGML_TYPE_TURBO4_0:
            set_rows_sycl_turbo<TIdx, block_turbo4_0, QK_TURBO4, quantize_turbo4_0<1>, turbo_nearest_centroid_4bit, TURBO_CENTROIDS_4BIT>(
                (const float *)src0_d, src1_d, (block_turbo4_0 *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
            break;

        default:
            GGML_ABORT("Unsupported tensor type!");
            break;
    }
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TurboQuant KV cache is unsupported on the SYCL backend: the turbo store and
    // attention kernels miscompile / fault on Intel Arc. This SET_ROWS store is the
    // first turbo op submitted for a turbo KV graph, so abort here on the host before
    // any broken turbo kernel reaches the GPU queue (the GGML_OP_TURBO_WHT executor
    // aborts too, as a backstop).
    if (dst->type == GGML_TYPE_TURBO2_0 || dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0) {
        GGML_ABORT("ggml_sycl: TurboQuant KV cache (turbo2/turbo3/turbo4) is not supported on the SYCL backend - it miscompiles / produces incorrect output on Intel Arc. Use --cache-type-k/v q8_0 (or f16), or run turbo KV on CPU with -ngl 0.");
    }

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_sycl<float, int32_t>(ctx, src0, src1, dst);
    }
}
