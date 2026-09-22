#include "conv2d.cuh"
#include "convert.cuh"
#include "mma.cuh"

struct conv_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

__device__ __forceinline__ int64_t max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int64_t out_coord,
                                                     int64_t kern_coord,
                                                     int64_t stride,
                                                     int64_t dilation,
                                                     int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ static int64_t kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params & P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ static int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ static void unpack_indices(int64_t             global_idx,
                                          const conv_params & P,
                                          int64_t &           n,
                                          int64_t &           c,
                                          int64_t &           out_y,
                                          int64_t &           out_x) {
        out_x = global_idx % P.OW;
        out_y = (global_idx / P.OW) % P.OH;
        c     = (global_idx / (P.OW * P.OH)) % P.OC;
        n     = global_idx / (P.OW * P.OH * P.OC);
    }
};

template <typename T, typename Layout>
static __global__ void conv2d_kernel(const float * __restrict__ input,
                                     const T * __restrict__ kernel,
                                     float * __restrict__ output,
                                     const conv_params P) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c_out, out_y, out_x;
    Layout::unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val = input[Layout::input_index(n, c_in, in_y, in_x, P)];
                const T kernel_val = kernel[Layout::kernel_index(c_out, c_in, ky, kx, P)];
                acc += (input_val * ggml_cuda_cast<float>(kernel_val));
            }
        }
    }

    // [N, OC, OH, OW]
    output[Layout::output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_cuda(const float * X_D, const T * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    const int blocks = (P.TOTAL + CUDA_CONV2D_BLOCK_SIZE - 1) / CUDA_CONV2D_BLOCK_SIZE;
    conv2d_kernel<T, whcn_layout><<<blocks, CUDA_CONV2D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, P);
}

static __global__ void
conv2d_pad_f16(const float * input, half * output, int iw, int ih, int pw, int ph, int px, int py, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int x = i % pw - px, y = i / pw % ph - py, nc = i / (pw * ph);
    output[i] = __float2half(
        (unsigned) x < (unsigned) iw && (unsigned) y < (unsigned) ih ? input[(nc * ih + y) * iw + x] : 0.0f);
}

template <int KW, int KH, bool use_mma>
static __global__ void conv2d_implicit_gemm_f16(const half * __restrict__ input,
                                                const half * __restrict__ weight,
                                                float * __restrict__ output,
                                                const conv_params P,
                                                const int         split_k) {
    using namespace ggml_cuda_mma;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = 4 * warp_size;
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int AS = BK / 2 + 4;
    constexpr int BS = BN / 2 + 4;
    __shared__ __align__(16) half2 a_s[BM][AS];
    __shared__ __align__(16) half2 b_s[BK][BS];

    const int tid = threadIdx.y * warp_size + threadIdx.x;
    const int iw = int(P.IW), ih = int(P.IH), ow = int(P.OW), oh = int(P.OH);
    const int kw = KW ? KW : int(P.KW), kh = KH ? KH : int(P.KH);
    const int ic = int(P.IC), oc = int(P.OC);
    const int sx = int(P.ST_X), sy = int(P.ST_Y);
    const int dx = int(P.DL_X), dy = int(P.DL_Y);
    const int n = blockIdx.z / split_k, split = blockIdx.z % split_k;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    const int k_total   = ic * kw * kh;
    const int load_lane = warp_size == 32 ? threadIdx.x : threadIdx.x % (BN / 2);
    const int load_row  = threadIdx.y * (warp_size / (BN / 2)) + (warp_size == 32 ? 0 : threadIdx.x / (BN / 2));
    const int spatial   = n0 + 2 * load_lane;
    const int spatial0 = min(spatial, ow * oh - 1), spatial1 = min(spatial + 1, ow * oh - 1);
    const int y0 = spatial0 / ow, x0 = spatial0 % ow;
    const int y1 = spatial1 / ow, x1 = spatial1 % ow;
    const int pos0 = y0 * sy * iw + x0 * sx, pos1 = y1 * sy * iw + x1 * sx;

    [[maybe_unused]] const int wm = threadIdx.y / 2 * 32, wn = threadIdx.y % 2 * 32;
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    using tile_ab = tile<16, 8, half2, get_input_data_layout()>;
#    if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    // AMD accumulator fragments transpose the input fragment's row/column mapping.
    using tile_c = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
#    else
    using tile_c = tile<16, 16, float>;
#    endif
    [[maybe_unused]] tile_c c[2][2];
#else
    if constexpr (use_mma) {
        NO_DEVICE_CODE;
        return;
    }
#endif
    constexpr int              RM = 4, RN = BM * BN / (nthreads * RM);
    [[maybe_unused]] const int simt_m = tid / (BN / RN) * RM, simt_n = tid % (BN / RN) * RN;
    [[maybe_unused]] float     c_simt[RM][RN] = {};
    const int                  tiles          = (k_total + BK - 1) / BK;
    const int                  begin          = int(int64_t(tiles) * split / split_k) * BK;
    const int                  end            = int(int64_t(tiles) * (split + 1) / split_k) * BK;
    for (int k0 = begin; k0 < end; k0 += BK) {
        if (k_total % 8 == 0 && uintptr_t(weight) % 16 == 0) {
#pragma unroll
            for (int i = tid; i < BM * BK / 8; i += nthreads) {
                const int  row = i / (BK / 8), col = 8 * (i % (BK / 8));
                const int4 v                 = m0 + row < oc && k0 + col < k_total ?
                                                   ((const int4 *) weight)[((m0 + row) * k_total + k0 + col) / 8] :
                                                   make_int4(0, 0, 0, 0);
                *(int4 *) &a_s[row][col / 2] = v;
            }
        } else {
#pragma unroll
            for (int i = tid; i < BM * BK / 2; i += nthreads) {
                const int row = i / (BK / 2), col = 2 * (i % (BK / 2));
                half      lo = __float2half(0.0f), hi = lo;
                if (m0 + row < oc && k0 + col < k_total) {
                    lo = weight[(m0 + row) * k_total + k0 + col];
                    if (k0 + col + 1 < k_total) {
                        hi = weight[(m0 + row) * k_total + k0 + col + 1];
                    }
                }
                a_s[row][col / 2] = __halves2half2(lo, hi);
            }
        }
#pragma unroll
        for (int k = load_row; k < BK; k += nthreads / (BN / 2)) {
            const int ki = k0 + k;
            const int ci = ki / (kw * kh), ky = ki / kw % kh, kx = ki % kw;
            const int offset = ki < k_total ? (n * ic + ci) * ih * iw + ky * dy * iw + kx * dx : 0;
            half      lo = __float2half(0.0f), hi = lo;
            if (ki < k_total && spatial < ow * oh) {
                lo = input[offset + pos0];
            }
            if (ki < k_total && spatial + 1 < ow * oh) {
                hi = input[offset + pos1];
            }
            b_s[k][load_lane] = __halves2half2(lo, hi);
        }
        __syncthreads();
        if constexpr (use_mma) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#    pragma unroll
            for (int k = 0; k < BK; k += 16) {
                tile_ab a[2], b[2];
#    pragma unroll
                for (int i = 0; i < 2; ++i) {
                    load_ldmatrix(a[i], &a_s[wm + 16 * i][k / 2], AS);
                    load_ldmatrix_trans(b[i], &b_s[k][(wn + 16 * i) / 2], BS);
                }
#    pragma unroll
                for (int i = 0; i < 2; ++i) {
#    pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        mma(c[i][j], a[i], b[j]);
                    }
                }
            }
#endif
        } else {
#pragma unroll 4
            for (int k = 0; k < BK; ++k) {
                float a[RM], b[RN];
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    a[i] = __half2float(((const half *) a_s[simt_m + i])[k]);
                }
#pragma unroll
                for (int j = 0; j < RN; ++j) {
                    b[j] = __half2float(((const half *) b_s[k])[simt_n + j]);
                }
#pragma unroll
                for (int i = 0; i < RM; ++i) {
#pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        c_simt[i][j] += a[i] * b[j];
                    }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (use_mma) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#    pragma unroll
        for (int i = 0; i < 2; ++i) {
#    pragma unroll
            for (int j = 0; j < 2; ++j) {
#    pragma unroll
                for (int l = 0; l < c[i][j].ne; ++l) {
                    const int co  = m0 + wm + 16 * i + c[i][j].get_i(l);
                    const int pos = n0 + wn + 16 * j + c[i][j].get_j(l);
                    if (co < oc && pos < ow * oh) {
                        output[(int64_t(blockIdx.z) * oc + co) * ow * oh + pos] = c[i][j].x[l];
                    }
                }
            }
        }
#endif
    } else {
#pragma unroll
        for (int i = 0; i < RM; ++i) {
#pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int co = m0 + simt_m + i, pos = n0 + simt_n + j;
                if (co < oc && pos < ow * oh) {
                    output[(int64_t(blockIdx.z) * oc + co) * ow * oh + pos] = c_simt[i][j];
                }
            }
        }
    }
}

static __global__ void conv2d_reduce_split_k(const float * __restrict__ partial,
                                             float * __restrict__ output,
                                             const int total,
                                             const int per_batch,
                                             const int split_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int     n   = i / per_batch;
    const float * src = partial + int64_t(n) * (split_k - 1) * per_batch + i;
    float         sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += src[int64_t(k) * per_batch];
    }
    output[i] = sum;
}

template <bool use_mma>
static void conv2d_launch_implicit_gemm(const half *        input,
                                        const half *        weight,
                                        float *             output,
                                        const conv_params & params,
                                        int                 split_k,
                                        dim3                grid,
                                        dim3                block,
                                        cudaStream_t        stream) {
    if (params.KW == 3 && params.KH == 3) {
        conv2d_implicit_gemm_f16<3, 3, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    } else if (params.KW == 1 && params.KH == 1) {
        conv2d_implicit_gemm_f16<1, 1, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    } else {
        conv2d_implicit_gemm_f16<0, 0, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    }
}

static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<float>(X_D, K_D, Y_D, P, st);
}

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int64_t IW = input->ne[0];   // input_w
    const int64_t IH = input->ne[1];   // input_h
    const int64_t OW = dst->ne[0];     // output_w
    const int64_t OH = dst->ne[1];     // output_h
    const int64_t KW = kernel->ne[0];  // kernel_w
    const int64_t KH = kernel->ne[1];  // kernel_h
    const int64_t IC = input->ne[2];   // input_channels
    const int64_t OC = kernel->ne[3];  // ouptut_chanles
    const int64_t B  = input->ne[3];   // n_batches

    const int64_t total  = B * OC * OH * OW;
    conv_params   params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    const auto & device = ggml_cuda_info().devices[ctx.device];
    const bool   use_mma =
        turing_mma_available(device.cc) || amd_wmma_available(device.cc) || amd_mfma_available(device.cc);
    // MUSA can share the tiling without a native fragment implementation in mma.cuh.
    const bool use_simt   = GGML_CUDA_CC_IS_MTHREADS(device.cc);
    const bool pointwise  = KW == 1 && KH == 1 && ST_X == 1 && ST_Y == 1 && PD_X == 0 && PD_Y == 0;
    const bool use_blas   = pointwise && fast_fp16_hardware_available(device.cc);
    // Short reductions on small maps do not amortize conversion and launch costs.
    const bool small_conv = IC * KW * KH < 64 && OW * OH < 512;

    const int64_t limit    = INT_MAX - 256;
    const int64_t padded_w = IW + 2 * int64_t(PD_X), padded_h = IH + 2 * int64_t(PD_Y);
    const bool    padded_fits = padded_w > 0 && padded_w <= limit && padded_h > 0 && padded_h <= limit &&
                             padded_w * padded_h <= limit && IC * B <= limit / (padded_w * padded_h);
    if (kernel->type == GGML_TYPE_F16 && (use_mma || use_blas || use_simt) && (use_blas || !small_conv) &&
        ggml_nelements(input) <= limit && ggml_nelements(kernel) <= limit && total <= limit && padded_fits &&
        PD_X >= 0 && PD_Y >= 0 && ST_X > 0 && ST_Y > 0 && DL_X > 0 && DL_Y > 0 &&
        (OW - 1) * ST_X + (KW - 1) * DL_X < padded_w && (OH - 1) * ST_Y + (KH - 1) * DL_Y < padded_h &&
        (OC + 63) / 64 <= 65535 && B <= 65535) {
        const int pw = int(padded_w), ph = int(padded_h);
        const int padded_total = int(padded_w * padded_h * IC * B);

        ggml_cuda_pool_alloc<half> x_half(ctx.pool(), padded_total);
        // Match im2col's F16 input precision, but expand patches only in shared memory and accumulate in F32.
        if (PD_X == 0 && PD_Y == 0) {
            ggml_get_to_fp16_cuda(input->type)(X_D, x_half.get(), padded_total, st);
        } else {
            conv2d_pad_f16<<<(padded_total + 255) / 256, 256, 0, st>>>(X_D, x_half.get(), int(IW), int(IH), pw, ph,
                                                                       PD_X, PD_Y, padded_total);
        }
        const conv_params padded_params = { pw, ph, OW, OH, KW, KH, ST_X, ST_Y, 0, 0, DL_X, DL_Y, IC, OC, B, total };
        if (use_blas) {
            const float    alpha = 1.0f, beta = 0.0f;
            const int      positions = int(OW * OH);
            cublasHandle_t cublas_h  = ctx.cublas_handle();
            for (int n = 0; n < B; ++n) {
                CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, positions, int(OC), int(IC), &alpha,
                                          x_half.get() + int64_t(n) * IC * positions, CUDA_R_16F, positions, K_D,
                                          CUDA_R_16F, int(IC), &beta, Y_D + int64_t(n) * OC * positions, CUDA_R_32F,
                                          positions, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            return;
        }
        const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
        const int     target  = 8 * ggml_cuda_info().devices[ctx.device].nsm;
        // Split long reductions so small spatial maps still occupy the GPU.
        const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH + 63) / 64,
                                               std::max(int64_t(1), (target + blocks - 1) / blocks) }));

        ggml_cuda_pool_alloc<float> partial(ctx.pool());
        float *                     result = split_k == 1 ? Y_D : partial.alloc(total * split_k);
        const dim3                  block(device.warp_size, 4);
        const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
        if (use_mma) {
            conv2d_launch_implicit_gemm<true>(x_half.get(), (const half *) K_D, result, padded_params, split_k, grid,
                                              block, st);
        } else {
            conv2d_launch_implicit_gemm<false>(x_half.get(), (const half *) K_D, result, padded_params, split_k, grid,
                                               block, st);
        }
        if (split_k > 1) {
            conv2d_reduce_split_k<<<(total + 255) / 256, 256, 0, st>>>(result, Y_D, int(total), int(OC * OW * OH),
                                                                       split_k);
        }
        return;
    }

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, (half *) K_D, Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}
