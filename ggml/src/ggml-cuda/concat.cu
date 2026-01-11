#include "concat.cuh"

// contiguous kernels
template <typename T>
static __global__ void concat_dim0(const T * x, const T * y, T * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static __global__ void concat_dim1(const T * x, const T * y, T * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned)ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static __global__ void concat_dim2(const T * x, const T * y, T * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static void concat_cuda(const char * x, const char * y, char * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>((const T *) x, (const T *) y, (T *) dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>((const T *) x, (const T *) y, (T *) dst, ne0, ne01);
        return;
    }
    concat_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>((const T *) x, (const T *) y, (T *) dst, ne0, ne02);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const T *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const T *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type && src1->type == dst->type);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const char * src0_d = (const char *)src0->data;
        const char * src1_d = (const char *)src1->data;

        char * dst_d = (char *)dst->data;

        decltype(concat_cuda<float>) * concat_cuda_func;
        switch (dst->type) {
            case GGML_TYPE_F32:
                concat_cuda_func = concat_cuda<float>;
                break;
            case GGML_TYPE_I32:
                concat_cuda_func = concat_cuda<int32_t>;
                break;
            case GGML_TYPE_F16:
                concat_cuda_func = concat_cuda<half>;
                break;
            case GGML_TYPE_BF16:
                concat_cuda_func = concat_cuda<nv_bfloat16>;
                break;
            case GGML_TYPE_I16:
                concat_cuda_func = concat_cuda<int16_t>;
                break;
            case GGML_TYPE_I8:
                concat_cuda_func = concat_cuda<int8_t>;
                break;
            default:
                GGML_ABORT("concat: unsupported type %s", ggml_type_name(dst->type));
        }

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_cuda_func(
                        src0_d + i3 * src0->nb[3],
                        src1_d + i3 * src1->nb[3],
                        dst_d + i3 * dst->nb[3],
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,         src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            decltype(concat_non_cont<float, dim>) * concat_kernel;
            switch (dst->type) {
                case GGML_TYPE_F32:
                    concat_kernel = concat_non_cont<float, dim>;
                    break;
                case GGML_TYPE_I32:
                    concat_kernel = concat_non_cont<int32_t, dim>;
                    break;
                case GGML_TYPE_F16:
                    concat_kernel = concat_non_cont<half, dim>;
                    break;
                case GGML_TYPE_BF16:
                    concat_kernel = concat_non_cont<nv_bfloat16, dim>;
                    break;
                case GGML_TYPE_I16:
                    concat_kernel = concat_non_cont<int16_t, dim>;
                    break;
                case GGML_TYPE_I8:
                    concat_kernel = concat_non_cont<int8_t, dim>;
                    break;
                default:
                    GGML_ABORT("concat: unsupported type %s", ggml_type_name(dst->type));
            }

            concat_kernel<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}
