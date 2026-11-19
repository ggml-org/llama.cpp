#include "conv3d.cuh"
#include "convert.cuh"
#include "mma.cuh"

struct conv3d_params {
    int64_t IW, IH, ID;
    int64_t OW, OH, OD;
    int64_t KW, KH, KD;
    int64_t ST_X, ST_Y, ST_Z;
    int64_t PD_X, PD_Y, PD_Z;
    int64_t DL_X, DL_Y, DL_Z;
    int64_t IC, OC, B, TOTAL;
};

template <typename T>
static __global__ void conv3d_kernel(const float * input, const T * weight, float * output, const conv3d_params P) {
    const int64_t spatial = P.OW * P.OH * P.OD;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < P.TOTAL;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t x = i % P.OW, y = i / P.OW % P.OH, z = i / (P.OW * P.OH) % P.OD;
        const int64_t co = i / spatial % P.OC, n = i / (spatial * P.OC);
        float         sum = 0.0f;
        for (int64_t ci = 0; ci < P.IC; ++ci) {
            for (int64_t kz = 0; kz < P.KD; ++kz) {
                const int64_t iz = z * P.ST_Z + kz * P.DL_Z - P.PD_Z;
                if (iz < 0 || iz >= P.ID) {
                    continue;
                }
                for (int64_t ky = 0; ky < P.KH; ++ky) {
                    const int64_t iy = y * P.ST_Y + ky * P.DL_Y - P.PD_Y;
                    if (iy < 0 || iy >= P.IH) {
                        continue;
                    }
                    for (int64_t kx = 0; kx < P.KW; ++kx) {
                        const int64_t ix = x * P.ST_X + kx * P.DL_X - P.PD_X;
                        if (ix >= 0 && ix < P.IW) {
                            const int64_t xi = (((n * P.IC + ci) * P.ID + iz) * P.IH + iy) * P.IW + ix;
                            const int64_t wi = (((co * P.IC + ci) * P.KD + kz) * P.KH + ky) * P.KW + kx;
                            sum += input[xi] * ggml_cuda_cast<float>(weight[wi]);
                        }
                    }
                }
            }
        }
        output[i] = sum;
    }
}

static __global__ void conv3d_pad_f16(const float * input,
                                      half *        output,
                                      int           iw,
                                      int           ih,
                                      int           id,
                                      int           pw,
                                      int           ph,
                                      int           pd,
                                      int           px,
                                      int           py,
                                      int           pz,
                                      int           total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int x = i % pw - px, y = i / pw % ph - py, z = i / (pw * ph) % pd - pz;
    const int nc = i / (pw * ph * pd);
    output[i] =
        __float2half((unsigned) x < (unsigned) iw && (unsigned) y < (unsigned) ih && (unsigned) z < (unsigned) id ?
                         input[((nc * id + z) * ih + y) * iw + x] :
                         0.0f);
}

template <int KW, int KH, int KD, bool use_mma>
static __global__ void conv3d_implicit_gemm_f16(const half * __restrict__ input,
                                                const half * __restrict__ weight,
                                                float * __restrict__ output,
                                                const conv3d_params P,
                                                const int           split_k,
                                                const bool          aligned_weights) {
    using namespace ggml_cuda_mma;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = 4 * warp_size;
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int AS = BK / 2 + 4;
    constexpr int BS = BN / 2 + 4;
    static_assert(AS * sizeof(half2) % sizeof(int4) == 0, "shared weight rows must be 16-byte aligned");
    __shared__ __align__(16) half2 a_s[BM][AS];
    __shared__ __align__(16) half2 b_s[BK][BS];
    const int                      tid = threadIdx.y * warp_size + threadIdx.x;
    const int                      iw = int(P.IW), ih = int(P.IH), id = int(P.ID);
    const int                      ow = int(P.OW), oh = int(P.OH), od = int(P.OD);
    const int                      kw = KW ? KW : int(P.KW), kh = KH ? KH : int(P.KH), kd = KD ? KD : int(P.KD);
    const int                      ic = int(P.IC), oc = int(P.OC);
    const int                      sx = int(P.ST_X), sy = int(P.ST_Y), sz = int(P.ST_Z);
    const int                      dx = int(P.DL_X), dy = int(P.DL_Y), dz = int(P.DL_Z);
    const int                      n = blockIdx.z / split_k, split = blockIdx.z % split_k;
    const int                      m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int                      k_total   = ic * kw * kh * kd;
    const int                      load_lane = warp_size == 32 ? threadIdx.x : threadIdx.x % (BN / 2);
    const int load_row = threadIdx.y * (warp_size / (BN / 2)) + (warp_size == 32 ? 0 : threadIdx.x / (BN / 2));
    const int spatial  = n0 + 2 * load_lane;
    const int spatial0 = min(spatial, ow * oh * od - 1), spatial1 = min(spatial + 1, ow * oh * od - 1);
    const int z0 = spatial0 / (ow * oh), y0 = spatial0 / ow % oh, x0 = spatial0 % ow;
    const int z1 = spatial1 / (ow * oh), y1 = spatial1 / ow % oh, x1 = spatial1 % ow;
    const int pos0                = (z0 * sz * ih + y0 * sy) * iw + x0 * sx;
    const int pos1                = (z1 * sz * ih + y1 * sy) * iw + x1 * sx;
    [[maybe_unused]] const int wm = threadIdx.y / 2 * 32, wn = threadIdx.y % 2 * 32;
    using tile_ab = tile<16, 8, half2, get_input_data_layout()>;
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    // AMD accumulator fragments transpose the input fragment's row/column mapping.
    using tile_c = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
#else
    using tile_c = tile<16, 16, float>;
#endif
    [[maybe_unused]] tile_c    c[2][2];
    constexpr int              RM = 4, RN = BM * BN / (nthreads * RM);
    [[maybe_unused]] const int simt_m = tid / (BN / RN) * RM, simt_n = tid % (BN / RN) * RN;
    [[maybe_unused]] float     c_simt[RM][RN] = {};
    const int                  tiles          = (k_total + BK - 1) / BK;
    const int                  begin          = int(int64_t(tiles) * split / split_k) * BK;
    const int                  end            = int(int64_t(tiles) * (split + 1) / split_k) * BK;
    for (int k0 = begin; k0 < end; k0 += BK) {
        if (aligned_weights) {
#pragma unroll
            for (int i0 = 0; i0 < BM * BK / 8; i0 += nthreads) {
                const int  i   = i0 + tid;
                const int  row = i / (BK / 8), col = 8 * (i % (BK / 8));
                const int4 v                 = m0 + row < oc && k0 + col < k_total ?
                                                   ((const int4 *) weight)[((m0 + row) * k_total + k0 + col) / 8] :
                                                   make_int4(0, 0, 0, 0);
                *(int4 *) &a_s[row][col / 2] = v;
            }
        } else {
#pragma unroll
            for (int i0 = 0; i0 < BM * BK / 2; i0 += nthreads) {
                const int i   = i0 + tid;
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
        for (int kb = 0; kb < BK; kb += nthreads / (BN / 2)) {
            const int k  = kb + load_row;
            const int ki = k0 + k;
            const int ci = ki / (kw * kh * kd), kz = ki / (kw * kh) % kd, ky = ki / kw % kh, kx = ki % kw;
            const int offset = ki < k_total ? ((n * ic + ci) * id + kz * dz) * ih * iw + ky * dy * iw + kx * dx : 0;
            half      lo = __float2half(0.0f), hi = lo;
            if (ki < k_total && spatial < ow * oh * od) {
                lo = input[offset + pos0];
            }
            if (ki < k_total && spatial + 1 < ow * oh * od) {
                hi = input[offset + pos1];
            }
            b_s[k][load_lane] = __halves2half2(lo, hi);
        }
        __syncthreads();
        if constexpr (use_mma) {
#pragma unroll
            for (int k = 0; k < BK; k += 16) {
                tile_ab a[2], b[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    load_ldmatrix(a[i], &a_s[wm + 16 * i][k / 2], AS);
                    load_ldmatrix_trans(b[i], &b_s[k][(wn + 16 * i) / 2], BS);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        mma(c[i][j], a[i], b[j]);
                    }
                }
            }
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
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int l = 0; l < c[i][j].ne; ++l) {
                    const int co  = m0 + wm + 16 * i + c[i][j].get_i(l);
                    const int pos = n0 + wn + 16 * j + c[i][j].get_j(l);
                    if (co < oc && pos < ow * oh * od) {
                        output[(int64_t(blockIdx.z) * oc + co) * ow * oh * od + pos] = c[i][j].x[l];
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < RM; ++i) {
#pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int co = m0 + simt_m + i, pos = n0 + simt_n + j;
                if (co < oc && pos < ow * oh * od) {
                    output[(int64_t(blockIdx.z) * oc + co) * ow * oh * od + pos] = c_simt[i][j];
                }
            }
        }
    }
}

static __global__ void conv3d_reduce_split_k(const float * __restrict__ partial,
                                             float * __restrict__ output,
                                             const int total,
                                             const int per_batch,
                                             const int split_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int     n   = i / per_batch;
    // Partial slices are ordered as [batch, split, output channel, spatial position].
    const float * src = partial + int64_t(n) * (split_k - 1) * per_batch + i;
    float         sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += src[int64_t(k) * per_batch];
    }
    output[i] = sum;
}

template <bool use_mma>
static void conv3d_launch_implicit_gemm(const half *          input,
                                        const half *          weight,
                                        float *               output,
                                        const conv3d_params & params,
                                        int                   split_k,
                                        dim3                  grid,
                                        dim3                  block,
                                        cudaStream_t          stream) {
    // Vector loads require both the base pointer and each weight row to be 16-byte aligned.
    const bool aligned_weights = uintptr_t(weight) % sizeof(int4) == 0 &&
                                 (params.IC * params.KW * params.KH * params.KD) % (sizeof(int4) / sizeof(half)) == 0;
    if (params.KW == 3 && params.KH == 3 && params.KD == 3) {
        conv3d_implicit_gemm_f16<3, 3, 3, use_mma>
            <<<grid, block, 0, stream>>>(input, weight, output, params, split_k, aligned_weights);
    } else if (params.KW == 1 && params.KH == 1 && params.KD == 3) {
        conv3d_implicit_gemm_f16<1, 1, 3, use_mma>
            <<<grid, block, 0, stream>>>(input, weight, output, params, split_k, aligned_weights);
    } else if (params.KW == 1 && params.KH == 1 && params.KD == 1) {
        conv3d_implicit_gemm_f16<1, 1, 1, use_mma>
            <<<grid, block, 0, stream>>>(input, weight, output, params, split_k, aligned_weights);
    } else {
        conv3d_implicit_gemm_f16<0, 0, 0, use_mma>
            <<<grid, block, 0, stream>>>(input, weight, output, params, split_k, aligned_weights);
    }
}

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(kernel) && ggml_is_contiguous(dst));

    const int32_t * p  = dst->op_params;
    const int64_t   IW = input->ne[0], IH = input->ne[1], ID = input->ne[2];
    const int64_t   OW = dst->ne[0], OH = dst->ne[1], OD = dst->ne[2];
    const int64_t   KW = kernel->ne[0], KH = kernel->ne[1], KD = kernel->ne[2];
    const int64_t   IC = p[9], B = p[10], OC = p[11];
    GGML_ASSERT(IC > 0 && B > 0 && OC > 0 && input->ne[3] == IC * B && kernel->ne[3] == IC * OC);
    GGML_ASSERT(dst->ne[3] == OC * B && p[0] > 0 && p[1] > 0 && p[2] > 0 && p[6] > 0 && p[7] > 0 && p[8] > 0);
    const int64_t       total  = ggml_nelements(dst);
    const conv3d_params params = { IW,   IH,   ID,   OW,   OH,   OD,   KW,   KH, KD, p[0], p[1],
                                   p[2], p[3], p[4], p[5], p[6], p[7], p[8], IC, OC, B,    total };
    const float *       x      = (const float *) input->data;
    const half *        w      = (const half *) kernel->data;
    float *             y      = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();
    const auto &        device = ggml_cuda_info().devices[ctx.device];
    const bool          use_mma =
        turing_mma_available(device.cc) || amd_wmma_available(device.cc) || amd_mfma_available(device.cc);
    const bool pointwise =
        KW == 1 && KH == 1 && KD == 1 && p[0] == 1 && p[1] == 1 && p[2] == 1 && p[3] == 0 && p[4] == 0 && p[5] == 0;
    const bool    use_blas = pointwise && fast_fp16_hardware_available(device.cc);
    const int64_t limit    = INT_MAX - 256;
    const int64_t pw = IW + 2 * int64_t(p[3]), ph = IH + 2 * int64_t(p[4]), pd = ID + 2 * int64_t(p[5]);
    const bool    padded_fits = pw > 0 && pw <= limit && ph > 0 && ph <= limit && pd > 0 && pd <= limit &&
                             pw * ph <= limit / pd && IC * B <= limit / (pw * ph * pd);
    if (kernel->type == GGML_TYPE_F16 && KW > 0 && KH > 0 && KD > 0 && ggml_nelements(input) <= limit &&
        ggml_nelements(kernel) <= limit && total <= limit && padded_fits && p[3] >= 0 && p[4] >= 0 && p[5] >= 0 &&
        (OW - 1) * p[0] + (KW - 1) * p[6] < pw && (OH - 1) * p[1] + (KH - 1) * p[7] < ph &&
        (OD - 1) * p[2] + (KD - 1) * p[8] < pd && (OC + 63) / 64 <= 65535 && B <= 65535) {
        const int                  padded_total = int(pw * ph * pd * IC * B);
        ggml_cuda_pool_alloc<half> x_half(ctx.pool(), padded_total);
        // Match im2col's F16 input precision without materializing all patches in global memory.
        if (p[3] == 0 && p[4] == 0 && p[5] == 0) {
            ggml_get_to_fp16_cuda(input->type)(x, x_half.get(), padded_total, stream);
        } else {
            conv3d_pad_f16<<<(padded_total + 255) / 256, 256, 0, stream>>>(
                x, x_half.get(), int(IW), int(IH), int(ID), int(pw), int(ph), int(pd), p[3], p[4], p[5], padded_total);
        }
        const conv3d_params padded_params = { pw,   ph, pd, OW, OH,   OD,   KW,   KH, KD, p[0], p[1],
                                              p[2], 0,  0,  0,  p[6], p[7], p[8], IC, OC, B,    total };
        const int           positions     = int(OW * OH * OD);
        if (use_blas) {
            const float    alpha = 1.0f, beta = 0.0f;
            cublasHandle_t cublas_h = ctx.cublas_handle();
            for (int n = 0; n < B; ++n) {
                CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, positions, int(OC), int(IC), &alpha,
                                          x_half.get() + int64_t(n) * IC * positions, CUDA_R_16F, positions, w,
                                          CUDA_R_16F, int(IC), &beta, y + int64_t(n) * OC * positions, CUDA_R_32F,
                                          positions, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            return;
        }
        const int64_t blocks  = ((positions + 63) / 64) * ((OC + 63) / 64) * B;
        const int     target  = 8 * device.nsm;
        const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH * KD + 63) / 64,
                                               std::max(int64_t(1), (target + blocks - 1) / blocks) }));
        ggml_cuda_pool_alloc<float> partial(ctx.pool());
        float *                     result = split_k == 1 ? y : partial.alloc(total * split_k);
        const dim3                  block(device.warp_size, 4);
        const dim3 grid(unsigned((positions + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
        if (use_mma) {
            conv3d_launch_implicit_gemm<true>(x_half.get(), w, result, padded_params, split_k, grid, block, stream);
        } else {
            conv3d_launch_implicit_gemm<false>(x_half.get(), w, result, padded_params, split_k, grid, block, stream);
        }
        if (split_k > 1) {
            conv3d_reduce_split_k<<<unsigned((total + 255) / 256), 256, 0, stream>>>(result, y, int(total),
                                                                                     int(OC * positions), split_k);
        }
        return;
    }
    const int blocks = int(std::min(int64_t(65535), (total + 255) / 256));
    if (kernel->type == GGML_TYPE_F16) {
        conv3d_kernel<<<blocks, 256, 0, stream>>>(x, w, y, params);
    } else {
        conv3d_kernel<<<blocks, 256, 0, stream>>>(x, (const float *) kernel->data, y, params);
    }
}
