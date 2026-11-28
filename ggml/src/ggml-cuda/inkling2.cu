#include "inkling2.cuh"

// the fused kernels keep the operation order of the unfused graph where it is cheap to do so,
// __fmul_rn/__fadd_rn stop the compiler from contracting a separate MUL and ADD into one FMA

static __global__ void situ_f32(
        const float * gate, const float * up, float * dst, const int ne0, const int ne1, const int ne2, const int64_t nrows,
        const int64_t g_s1, const int64_t g_s2, const int64_t g_s3, const int64_t u_s1, const int64_t u_s2, const int64_t u_s3,
        const float s0, const float b0, const float s1, const float b1, const float s2, const float b2, const float s3, const float b3) {
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();
    if (i0 >= ne0) {
        return;
    }

    for (int64_t row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        const int64_t i3 = row / ((int64_t) ne1*ne2);

        const float g = gate[i3*g_s3 + i2*g_s2 + i1*g_s1 + i0];
        const float u = up  [i3*u_s3 + i2*u_s2 + i1*u_s1 + i0];

        const float act = s1*tanhf(s0*g + b0) + b1;
        const float sig = 1.0f/(1.0f + expf(-g));
        const float lin = s3*tanhf(s2*u + b2) + b3;

        dst[row*ne0 + i0] = __fmul_rn(__fmul_rn(act, sig), lin);
    }
}

void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
        const float * s, const float * b, ggml_tensor * dst) {
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->nb[0] == sizeof(float) && up->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, dst));

    const int64_t ne0   = gate->ne[0];
    const int64_t nrows = ggml_nrows(gate);

    const int  block_size = 256;
    const dim3 grid((ne0 + block_size - 1)/block_size, std::min<int64_t>(nrows, 65535), 1);

    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(situ_f32, params,
            (const float *) gate->data, (const float *) up->data, (float *) dst->data, (int) ne0, (int) gate->ne[1], (int) gate->ne[2], nrows,
            (int64_t) (gate->nb[1]/sizeof(float)), (int64_t) (gate->nb[2]/sizeof(float)), (int64_t) (gate->nb[3]/sizeof(float)),
            (int64_t) (up->nb[1]/sizeof(float)), (int64_t) (up->nb[2]/sizeof(float)), (int64_t) (up->nb[3]/sizeof(float)),
            s[0], b[0], s[1], b[1], s[2], b[2], s[3], b[3]);
}

// one thread per (channel, run of tokens of one sequence). the last K columns of x stay in registers, right-aligned in
// M = GGML_CUDA_SHORT_CONV_MAX_K slots with zero weights in front, so K needs no template and the sums keep their order.
// a thread reads cur at a token before it writes dst there and never again, so dst may be cur itself when a thread has
// the whole sequence
template <bool has_dyn, bool has_res>
static __global__ void short_conv_f32(const ggml_cuda_short_conv_args a) {
    ggml_cuda_pdl_lc();
    constexpr int M = GGML_CUDA_SHORT_CONV_MAX_K;

    const int K  = a.K;
    const int K1 = K - 1;
    const int c  = blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();
    if (c >= a.C) {
        return;
    }

    float w[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        w[m] = m >= M - K ? a.w[c*K + m - (M - K)] : 0.0f;
    }

    const int n_runs = (a.Ts + a.run - 1)/a.run;
    for (int r = blockIdx.y; r < a.S*n_runs; r += gridDim.y) {
        const int s  = r / n_runs;
        const int t0 = (r - s*n_runs)*a.run;
        const int t1 = min(t0 + a.run, a.Ts);

        const float * pc = a.past + c*a.past_s1 + s*a.past_s2;
        const float * xc = a.cur + (int64_t) s*a.Ts*a.C + c;

        // x[M - K + k] is column ts + k of [past; cur^T]
        float x[M];
#pragma unroll
        for (int m = 0; m < M - 1; ++m) {
            const int j = t0 + m - (M - K);
            x[m] = j < 0 ? 0.0f : j < K1 ? pc[j] : xc[(int64_t) (j - K1)*a.C];
        }

        for (int ts = t0; ts < t1; ++ts) {
            const int64_t t = (int64_t) s*a.Ts + ts;

            x[M - 1] = xc[(int64_t) ts*a.C];

            float y = 0.0f;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                y += x[m]*w[m];
            }

            if constexpr (has_dyn) {
                const float * d = a.dyn + t*a.dyn_s1 + (c / (a.C / a.n_heads))*K;
#pragma unroll
                for (int lag = 0; lag < M; ++lag) {
                    if (lag < K) {
                        y = __fadd_rn(y, __fmul_rn(x[M - 1 - lag], d[lag]));
                    }
                }
            }

            float o = a.scale*__fadd_rn(x[M - 1], y) + a.bias;
            if constexpr (has_res) {
                o = __fadd_rn(a.res[t*a.C + c], o);
            }
            a.dst[t*a.C + c] = o;

#pragma unroll
            for (int m = 0; m < M - 1; ++m) {
                x[m] = x[m + 1];
            }

            // x now holds the K - 1 columns that end at token ts: the conv state of slot Ts - 1 - ts, if it is kept
            if (ts >= a.Ts - a.kw) {
                float * st = a.state + (a.Ts - 1 - ts)*a.state_slot_s + c*a.state_ch_s + s*a.state_seq_s;
#pragma unroll
                for (int m = 0; m < M - 1; ++m) {
                    if (m >= M - K) {
                        st[m - (M - K)] = x[m];
                    }
                }
            }
        }
    }
}

void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_short_conv_args & a) {
    GGML_ASSERT(a.K >= 2 && a.K <= GGML_CUDA_SHORT_CONV_MAX_K);

    const int  block_size = 128;
    const int  n_runs     = (a.Ts + a.run - 1)/a.run;
    const dim3 grid((a.C + block_size - 1)/block_size, std::min(a.S*n_runs, 65535), 1);

    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    if (a.dyn) {
        if (a.res) {
            ggml_cuda_kernel_launch(short_conv_f32<true, true>, params, a);
        } else {
            ggml_cuda_kernel_launch(short_conv_f32<true, false>, params, a);
        }
    } else {
        if (a.res) {
            ggml_cuda_kernel_launch(short_conv_f32<false, true>, params, a);
        } else {
            ggml_cuda_kernel_launch(short_conv_f32<false, false>, params, a);
        }
    }
}

// one block per (query, token): the dot products of the B stack rows and of cur with the query, the softmax over the
// B + 1 rows and the weighted sum; the norm of cur is a scale of its dot product
template <int block_size>
static __global__ void depth_mix_f32(
        const float * stack, const float * cur, const float * query, float * dst, const int n, const int B, const int Q,
        const int64_t stack_s1, const int64_t stack_s2, const int64_t cur_s1, const int64_t query_s1,
        const bool norm_cur, const float eps, const float scale) {
    ggml_cuda_pdl_lc();
    constexpr int MAX = GGML_CUDA_DEPTH_MIX_MAX_ROWS;
    constexpr int NW  = block_size/WARP_SIZE;

    const int q    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    const float * qv = query + q*query_s1;
    const float * st = stack + t*stack_s2;
    const float * xc = cur   + t*cur_s1;

    // [0, B): the stack rows, MAX: cur, MAX + 1: the sum of squares of cur
    float acc[MAX + 2];
#pragma unroll
    for (int r = 0; r < MAX + 2; ++r) {
        acc[r] = 0.0f;
    }

    ggml_cuda_pdl_sync();

    for (int i = tid; i < n; i += block_size) {
        const float qi = qv[i];
#pragma unroll
        for (int r = 0; r < MAX; ++r) {
            if (r < B) {
                acc[r] += st[r*stack_s1 + i]*qi;
            }
        }
        const float xi = xc[i];
        acc[MAX]     += xi*qi;
        acc[MAX + 1] += xi*xi;
    }

    __shared__ float s_acc[NW][MAX + 2];
#pragma unroll
    for (int r = 0; r < MAX + 2; ++r) {
        if (r < B || r >= MAX) {
            acc[r] = warp_reduce_sum(acc[r]);
            if (lane == 0) {
                s_acc[warp][r] = acc[r];
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < MAX + 2; ++r) {
        if (r < B || r >= MAX) {
            acc[r] = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                acc[r] += s_acc[w][r];
            }
        }
    }

    const float sc = norm_cur ? rsqrtf(acc[MAX + 1]/n + eps) : 1.0f;

    float mx = scale*(sc*acc[MAX]);
#pragma unroll
    for (int r = 0; r < MAX; ++r) {
        if (r < B) {
            acc[r] = scale*acc[r];
            mx = fmaxf(mx, acc[r]);
        }
    }

    float p_cur = expf(scale*(sc*acc[MAX]) - mx);
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < MAX; ++r) {
        if (r < B) {
            acc[r] = expf(acc[r] - mx);
            sum += acc[r];
        }
    }
    sum += p_cur;

    const float inv = 1.0f/sum;
#pragma unroll
    for (int r = 0; r < MAX; ++r) {
        acc[r] *= inv;
    }
    p_cur *= inv*sc;

    for (int i = tid; i < n; i += block_size) {
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < MAX; ++r) {
            if (r < B) {
                o += acc[r]*st[r*stack_s1 + i];
            }
        }
        o += p_cur*xc[i];
        dst[((int64_t) t*n + i)*Q + q] = o;
    }
}

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * stack, const ggml_tensor * cur,
        bool norm_cur, float eps, const ggml_tensor * query, float scale, ggml_tensor * dst) {
    const int64_t n = cur->ne[0];
    const int64_t B = stack->ne[1];
    const int64_t T = stack->ne[2];
    const int64_t Q = query->ne[1];

    GGML_ASSERT(B >= 1 && B <= GGML_CUDA_DEPTH_MIX_MAX_ROWS);
    GGML_ASSERT(stack->nb[0] == sizeof(float) && cur->nb[0] == sizeof(float) && query->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst) && dst->ne[0] == Q && dst->ne[1] == n && dst->ne[2] == T);
    GGML_ASSERT(T <= 65535 && Q <= 65535);

    const int  block_size = 256;
    const dim3 grid(Q, T, 1);

    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(depth_mix_f32<block_size>, params,
            (const float *) stack->data, (const float *) cur->data, (const float *) query->data, (float *) dst->data,
            (int) n, (int) B, (int) Q,
            (int64_t) (stack->nb[1]/sizeof(float)), (int64_t) (stack->nb[2]/sizeof(float)),
            (int64_t) (cur->nb[2]/sizeof(float)), (int64_t) (query->nb[1]/sizeof(float)),
            norm_cur, eps, scale);
}

// one block per (rank row, token): the sums of squares of cur and msg and their dot products with the two halves of the row
template <int block_size>
static __global__ void loop_ctrl_down_f32(
        const float * cur, const float * msg, const float * w, float * gate, const int n,
        const int64_t cur_s1, const int64_t msg_s1, const int64_t w_s1, const int64_t gate_s1, const float eps) {
    ggml_cuda_pdl_lc();
    const int r   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    const float * xc = cur + t*cur_s1;
    const float * xm = msg + t*msg_s1;
    const float * wr = w   + r*w_s1;

    ggml_cuda_pdl_sync();

    float2 ss = make_float2(0.0f, 0.0f);
    float2 dd = make_float2(0.0f, 0.0f);
    for (int i = tid; i < n; i += block_size) {
        const float c = xc[i];
        const float m = xm[i];
        ss.x += c*c;
        ss.y += m*m;
        dd.x += wr[i]*c;
        dd.y += wr[n + i]*m;
    }

    __shared__ float2 s_ss[32];
    __shared__ float2 s_dd[32];
    ss = block_reduce<block_reduce_method::SUM, block_size>(ss, s_ss);
    dd = block_reduce<block_reduce_method::SUM, block_size>(dd, s_dd);

    if (tid == 0) {
        gate[t*gate_s1 + r] = rsqrtf(ss.x/n + eps)*dd.x + rsqrtf(ss.y/n + eps)*dd.y;
    }
}

// one warp per (output channel, token): the gain and forget dot products over the rank and the combine
static __global__ void loop_ctrl_up_f32(
        const float * cur, const float * msg, const float * gate, const float * wg, const float * wf, float * dst,
        const int n, const int rank, const int64_t cur_s1, const int64_t msg_s1, const int64_t gate_s1,
        const int64_t wg_s1, const int64_t wf_s1, const int64_t dst_s1) {
    ggml_cuda_pdl_lc();
    const int i    = blockIdx.x*blockDim.y + threadIdx.y;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (i >= n) {
        return;
    }

    const float * gt = gate + t*gate_s1;

    float g = 0.0f;
    float f = 0.0f;
    for (int k = lane; k < rank; k += WARP_SIZE) {
        const float x = gt[k];
        g += wg[i*wg_s1 + k]*x;
        f += wf[i*wf_s1 + k]*x;
    }
    g = warp_reduce_sum(g);
    f = warp_reduce_sum(f);

    if (lane == 0) {
        const float c = cur[t*cur_s1 + i];
        const float m = msg[t*msg_s1 + i];
        const float a = __fsub_rn(c, __fmul_rn(tanhf(f), c));
        dst[t*dst_s1 + i] = __fadd_rn(a, __fmul_rn(tanhf(g), m));
    }
}

void ggml_cuda_op_loop_ctrl(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg, float eps,
        const ggml_tensor * w_down, const ggml_tensor * w_gain, const ggml_tensor * w_forget, ggml_tensor * gate, ggml_tensor * dst) {
    const int64_t n    = cur->ne[0];
    const int64_t T    = cur->ne[1];
    const int64_t rank = w_down->ne[1];

    GGML_ASSERT(cur->nb[0] == sizeof(float) && msg->nb[0] == sizeof(float) && gate->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    GGML_ASSERT(w_down->ne[0] == 2*n && w_gain->ne[0] == rank && w_forget->ne[0] == rank && w_gain->ne[1] == n && w_forget->ne[1] == n);
    GGML_ASSERT(T <= GGML_CUDA_LOOP_CTRL_MAX_TOKENS);

    cudaStream_t stream = ctx.stream();

    {
        const int block_size = 256;
        const ggml_cuda_kernel_launch_params params(dim3(rank, T, 1), dim3(block_size, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_ctrl_down_f32<block_size>, params,
                (const float *) cur->data, (const float *) msg->data, (const float *) w_down->data, (float *) gate->data, (int) n,
                (int64_t) (cur->nb[1]/sizeof(float)), (int64_t) (msg->nb[1]/sizeof(float)),
                (int64_t) (w_down->nb[1]/sizeof(float)), (int64_t) (gate->nb[1]/sizeof(float)), eps);
    }
    {
        const int rows_per_block = 8;
        const ggml_cuda_kernel_launch_params params(dim3((n + rows_per_block - 1)/rows_per_block, T, 1), dim3(WARP_SIZE, rows_per_block, 1), 0, stream);
        ggml_cuda_kernel_launch(loop_ctrl_up_f32, params,
                (const float *) cur->data, (const float *) msg->data, (const float *) gate->data,
                (const float *) w_gain->data, (const float *) w_forget->data, (float *) dst->data, (int) n, (int) rank,
                (int64_t) (cur->nb[1]/sizeof(float)), (int64_t) (msg->nb[1]/sizeof(float)), (int64_t) (gate->nb[1]/sizeof(float)),
                (int64_t) (w_gain->nb[1]/sizeof(float)), (int64_t) (w_forget->nb[1]/sizeof(float)), (int64_t) (dst->nb[1]/sizeof(float)));
    }
}
