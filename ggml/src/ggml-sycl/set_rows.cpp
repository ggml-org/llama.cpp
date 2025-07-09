#include "set_rows.hpp"

namespace utils {
template<typename T>
static constexpr bool is_arithmetic_v() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half> || std::is_same_v<T, sycl::ext::oneapi::bfloat16>;
}
}
template<typename TIn, typename TOut>
static inline std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), void>
convert (const char* src, char* dst) {
    auto src_val = *reinterpret_cast<const TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;;
}

template<typename TIn, typename TOut>
static void k_set_rows(
        const char * __restrict__ src0, const int64_t * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        const sycl::nd_item<3> & item_ct1) {

    const int i03 = item_ct1.get_group(0);
    const int i02 = item_ct1.get_group(1);
    const int i01 = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);  // Row index

    if (i01 >= ne01) {
        return;
    }

    const int i12 = i03 % ne12;
    const int i11 = i02 % ne11;
    const int i10 = i01;

    const int64_t dst_row = *(const int64_t *)((const char *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    const char * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    char * dst_row_ptr    = dst + dst_row*nb1 + i02*nb2 + i03*nb3;

    for (int col = item_ct1.get_local_id(0); col < ne00; col += item_ct1.get_local_range(0)) {
        const char * src_elem = src0_row + col * src_type_size;
        char * dst_elem       = dst_row_ptr + col * dst_type_size;
        convert<TIn, TOut>(src_elem, dst_elem);
    }
}

template<typename TIn, typename TOut>
static void set_rows_sycl(
        const char * src0_d, const int64_t * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        queue_ptr stream) {

    constexpr int max_threads_per_row = 64; // KEEPING 64 for now
    const int threads_per_row     = std::min((int)ne00, max_threads_per_row);

    constexpr int max_threads_per_block = 64;
    const int rows_per_block        = std::max(1, max_threads_per_block / threads_per_row);

    const sycl::range<3> block_size(1, rows_per_block, threads_per_row);
    const sycl::range<3> grid_size(ne03, ne02, (ne01 + rows_per_block - 1) / rows_per_block);

        sycl_parallel_for(
            stream,
            sycl::nd_range<3>(grid_size * block_size, block_size),
            [=](sycl::nd_item<3> item_ct1) {
                k_set_rows<TIn, TOut>(
                    src0_d, src1_d, dst_d,
                    ne00, ne01, ne11, ne12,
                    nb01, nb02, nb03,
                    nb10, nb11, nb12,
                    nb1, nb2, nb3,
                    src_type_size, dst_type_size,
                    item_ct1
                );
            }
        );
}


void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t * src1_dd = static_cast<const int64_t *>(src1->data);

    dpct::queue_ptr stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            set_rows_sycl<float, float>(
                (const char *)src0->data, src1_dd, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(float), sizeof(float),
                stream
            );
            break;
        case GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<float, sycl::half>(
                (const char *)src0->data, src1_dd, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(float), sizeof(sycl::half),
                stream
        );
            break;
        default:
            GGML_ABORT("Unsupported tensor type!");
            break;
    }
}
