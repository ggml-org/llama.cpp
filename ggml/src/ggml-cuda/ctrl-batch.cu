#include "ctrl-batch.cuh"
#include "ctrl-mix.cuh"
#include "convert.cuh"

#define CTRL_BATCH_MAX 4

namespace {

struct ctrl_nodes {
    int concat, down, mix_first, mix_last;
};

// what one controller reads and writes
struct ctrl_tensors {
    const ggml_tensor * a; const ggml_tensor * b;          // the concatenated halves
    const ggml_tensor * wd;                                // down weight [n_a + n_b, rank]
    const ggml_tensor * g;                                 // down output [rank]
    const ggml_tensor * wf; const ggml_tensor * wg;        // forget and gain weights [rank, n]
    const ggml_tensor * cur; const ggml_tensor * msg;      // mixed vectors [n]
    const ggml_tensor * out;                               // [n]
};

bool is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// the next compute node at or after k, -1 if none
int next_compute(const ggml_cgraph * cgraph, int k) {
    for (; k < cgraph->n_nodes; ++k) {
        if (!is_view(cgraph->nodes[k])) {
            return k;
        }
    }
    return -1;
}

// the block size mul_mat_vec_f picks for ncols: the down products copy its arithmetic, so they must use the same
int mmvf_block_size(int64_t ncols, int warp_size, int cc) {
    int64_t best = warp_size;
    int64_t niter_best = (ncols + 2*warp_size - 1)/(2*warp_size);
    int64_t max_bs = 256;
    if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
        max_bs = 128;
    }
    for (int64_t bs = 2*warp_size; bs <= max_bs; bs += warp_size) {
        const int64_t niter = (ncols + 2*bs - 1)/(2*bs);
        if (niter < niter_best) {
            niter_best = niter;
            best = bs;
        }
    }
    return (int) best;
}

void tensors_of(const ggml_cgraph * cgraph, const ctrl_nodes & c, ctrl_tensors & t) {
    const ggml_tensor * concat = cgraph->nodes[c.concat];
    const ggml_tensor * down   = cgraph->nodes[c.down];
    const ggml_tensor * mf     = cgraph->nodes[c.mix_first];
    const ggml_tensor * add    = cgraph->nodes[c.mix_last];
    const ggml_tensor * sub    = add->src[0]->op == GGML_OP_SUB ? add->src[0] : add->src[1];
    const ggml_tensor * mulg   = add->src[0]->op == GGML_OP_SUB ? add->src[1] : add->src[0];
    const ggml_tensor * tg     = mulg->src[0]->op == GGML_OP_UNARY ? mulg->src[0] : mulg->src[1];
    t.a   = concat->src[0];
    t.b   = concat->src[1];
    t.wd  = down->src[0];
    t.g   = down;
    t.wf  = mf->src[0];
    t.wg  = tg->src[0]->src[0];
    t.cur = sub->src[0];
    t.msg = mulg->src[0] == tg ? mulg->src[1] : mulg->src[0];
    t.out = add;
}

// one controller from node k on: CONCAT, MUL_MAT (down), then a ctrl_mix whose g is the down output
bool match_one(const ggml_cgraph * cgraph, int k, ctrl_nodes & c) {
    const ggml_tensor * concat = cgraph->nodes[k];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
            ggml_nrows(concat) != 1 || !(concat->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return false;
    }
    const ggml_tensor * a = concat->src[0];
    const ggml_tensor * b = concat->src[1];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
            ggml_nrows(a) != 1 || ggml_nrows(b) != 1 || a->ne[0] % 2 || b->ne[0] % 2) {
        return false;
    }
    const int kd = next_compute(cgraph, k + 1);
    if (kd < 0) {
        return false;
    }
    const ggml_tensor * down = cgraph->nodes[kd];
    const ggml_tensor * wd   = down->src[0];
    if (down->op != GGML_OP_MUL_MAT || down->src[1] != concat || wd->type != GGML_TYPE_BF16 || !ggml_is_contiguous(wd) ||
            wd->ne[0] != concat->ne[0] || wd->ne[2] != 1 || wd->ne[3] != 1 || down->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(down) || !(down->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, k) != 1 || (concat->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (down->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int km = next_compute(cgraph, kd + 1);
    if (km < 0) {
        return false;
    }
    const int last = ggml_cuda_match_ctrl_mix(cgraph, km, false);
    if (last < 0 || cgraph->nodes[km]->src[1] != down) {
        return false;
    }
    c = { k, kd, km, last };
    return true;
}

int match_all(const ggml_cgraph * cgraph, int i, ctrl_nodes * cs) {
    int n = 0;
    int k = i;
    while (n < CTRL_BATCH_MAX && k >= 0 && match_one(cgraph, k, cs[n])) {
        // all of one shape
        if (n > 0) {
            ctrl_tensors t0, t;
            tensors_of(cgraph, cs[0], t0);
            tensors_of(cgraph, cs[n], t);
            if (!ggml_are_same_shape(t.wd, t0.wd) || !ggml_are_same_shape(t.wf, t0.wf) || t.wf->type != t0.wf->type ||
                    t.a->ne[0] != t0.a->ne[0]) {
                break;
            }
        }
        k = next_compute(cgraph, cs[n].mix_last + 1);
        ++n;
    }
    if (n == 0) {
        return 0;
    }
    // a controller may not read what an earlier one in the batch writes: they run at the same time
    for (int x = 0; x < n; ++x) {
        ctrl_tensors tx;
        tensors_of(cgraph, cs[x], tx);
        for (int y = 0; y < n; ++y) {
            ctrl_tensors ty;
            tensors_of(cgraph, cs[y], ty);
            for (const ggml_tensor * in : { tx.a, tx.b, tx.cur, tx.msg }) {
                if (in == ty.out || in == ty.g) {
                    return 0;
                }
            }
        }
    }
    return n;
}

struct ctrl_batch_params {
    const nv_bfloat16 * wd[CTRL_BATCH_MAX];
    const float * a[CTRL_BATCH_MAX];
    const float * b[CTRL_BATCH_MAX];
    float * g[CTRL_BATCH_MAX];
    const void * wf[CTRL_BATCH_MAX];
    const void * wg[CTRL_BATCH_MAX];
    const float * cur[CTRL_BATCH_MAX];
    const float * msg[CTRL_BATCH_MAX];
    float * out[CTRL_BATCH_MAX];
    int n_a2;     // float2 elements of a
    int ncols2;   // float2 elements of [a; b]
    int64_t wd_s1; // elements between down-weight rows
    int rank;     // rows of the down product, columns of the mix
    int64_t w_s1; // elements between mix-weight rows
    int nrows;    // rows of the mix
};

// mul_mat_vec_f for one BF16 row and one F32 vector (the HIP path), the vector read from two places
template <int block_size>
__global__ void ctrl_batch_down(const ctrl_batch_params p) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row = blockIdx.x;
    const int c   = blockIdx.y;
    const int tid = threadIdx.x;

    const int    * x2 = (const int *) (p.wd[c] + row*p.wd_s1);
    const float2 * a2 = (const float2 *) p.a[c];
    const float2 * b2 = (const float2 *) p.b[c];

    __shared__ float buf_iw[warp_size];
    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf = 0.0f;
    for (int col2 = tid; col2 < p.ncols2; col2 += block_size) {
        const int    tmpx = x2[col2];
        const float2 tmpy = col2 < p.n_a2 ? a2[col2] : b2[col2 - p.n_a2];
        const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
        const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
        ggml_cuda_mad(sumf, tmpx0, tmpy.x);
        ggml_cuda_mad(sumf, tmpx1, tmpy.y);
    }

    sumf = warp_reduce_sum<warp_size>(sumf);
    if (block_size > warp_size) {
        buf_iw[tid/warp_size] = sumf;
        __syncthreads();
        if (tid < warp_size) {
            sumf = buf_iw[tid];
            sumf = warp_reduce_sum<warp_size>(sumf);
        }
    }
    if (tid == 0) {
        p.g[c][row] = sumf;
    }
}

template <typename T> __device__ __forceinline__ float ctrl_load(const T * x, int64_t i);
template <> __device__ __forceinline__ float ctrl_load<float>(const float * x, int64_t i) { return x[i]; }
template <> __device__ __forceinline__ float ctrl_load<half>(const half * x, int64_t i) { return __half2float(x[i]); }
template <> __device__ __forceinline__ float ctrl_load<nv_bfloat16>(const nv_bfloat16 * x, int64_t i) {
    return __uint_as_float(((uint32_t) reinterpret_cast<const uint16_t *>(x)[i]) << 16);
}

// ctrl_mix_kernel (ctrl-mix.cu) for every controller of the batch: a warp per output row
template <typename T>
__global__ void ctrl_batch_mix(const ctrl_batch_params p) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row  = blockIdx.x*(blockDim.x/warp_size) + threadIdx.x/warp_size;
    const int lane = threadIdx.x % warp_size;
    const int c    = blockIdx.y;
    if (row >= p.nrows) {
        return;
    }
    const T     * wf = (const T *) p.wf[c];
    const T     * wg = (const T *) p.wg[c];
    const float * g  = p.g[c];

    float f  = 0.0f;
    float gg = 0.0f;
    for (int k = lane; k < p.rank; k += warp_size) {
        const float gc = g[k];
        f  += ctrl_load(wf, row*p.w_s1 + k)*gc;
        gg += ctrl_load(wg, row*p.w_s1 + k)*gc;
    }
    f  = warp_reduce_sum<warp_size>(f);
    gg = warp_reduce_sum<warp_size>(gg);

    if (lane == 0) {
        const float x = p.cur[c][row];
        const float m = p.msg[c][row];
        p.out[c][row] = (x - tanhf(f)*x) + m*tanhf(gg);
    }
}

} // namespace

int ggml_cuda_match_ctrl_batch(const ggml_cgraph * cgraph, int i, bool check_memory) {
    ctrl_nodes cs[CTRL_BATCH_MAX];
    const int n = match_all(cgraph, i, cs);
    if (n == 0) {
        return -1;
    }
    {
        ctrl_tensors t0;
        tensors_of(cgraph, cs[0], t0);
        const int dev = ggml_cuda_get_device();
        if (mmvf_block_size(t0.wd->ne[0], ggml_cuda_info().devices[dev].warp_size, ggml_cuda_info().devices[dev].cc) % 32 != 0) {
            return -1;
        }
    }
    if (check_memory) {
        // every output against everything the batch reads and every other output; a mix output may only be exactly
        // its own element-wise cur or msg
        for (int x = 0; x < n; ++x) {
            ctrl_tensors tx;
            tensors_of(cgraph, cs[x], tx);
            for (int y = 0; y < n; ++y) {
                ctrl_tensors ty;
                tensors_of(cgraph, cs[y], ty);
                for (const ggml_tensor * in : { ty.a, ty.b, ty.wd, ty.wf, ty.wg, ty.cur, ty.msg }) {
                    if (ggml_cuda_fusion_overlap(tx.g, in)) {
                        return -1;
                    }
                    const bool own = x == y && (in == ty.cur || in == ty.msg);
                    if (own ? !ggml_cuda_fusion_elementwise_ok(tx.out, in) : ggml_cuda_fusion_overlap(tx.out, in)) {
                        return -1;
                    }
                }
                if (ggml_cuda_fusion_overlap(tx.out, ty.g) || (x != y && (ggml_cuda_fusion_overlap(tx.g, ty.g) ||
                        ggml_cuda_fusion_overlap(tx.out, ty.out)))) {
                    return -1;
                }
            }
        }
    }
    return cs[n - 1].mix_last;
}

void ggml_cuda_op_ctrl_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    ctrl_nodes cs[CTRL_BATCH_MAX];
    const int n = match_all(cgraph, i, cs);
    GGML_ASSERT(n > 0 && cs[n - 1].mix_last == last);

    ctrl_batch_params p = {};
    ctrl_tensors t;
    for (int c = 0; c < n; ++c) {
        tensors_of(cgraph, cs[c], t);
        p.wd[c]  = (const nv_bfloat16 *) t.wd->data;
        p.a[c]   = (const float *) t.a->data;
        p.b[c]   = (const float *) t.b->data;
        p.g[c]   = (float *) t.g->data;
        p.wf[c]  = t.wf->data;
        p.wg[c]  = t.wg->data;
        p.cur[c] = (const float *) t.cur->data;
        p.msg[c] = (const float *) t.msg->data;
        p.out[c] = (float *) t.out->data;
    }
    tensors_of(cgraph, cs[0], t);
    p.n_a2   = (int) (t.a->ne[0]/2);
    p.ncols2 = (int) (t.wd->ne[0]/2);
    p.wd_s1  = t.wd->nb[1]/ggml_type_size(t.wd->type);
    p.rank   = (int) t.wd->ne[1];
    p.w_s1   = t.wf->nb[1]/ggml_type_size(t.wf->type);
    p.nrows  = (int) t.wf->ne[1];

    cudaStream_t stream = ctx.stream();
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int bs = mmvf_block_size(t.wd->ne[0], warp_size, ggml_cuda_info().devices[ctx.device].cc);
    const dim3 grid_down(p.rank, n);
    switch (bs) {
        case  32: ctrl_batch_down< 32><<<grid_down,  32, 0, stream>>>(p); break;
        case  64: ctrl_batch_down< 64><<<grid_down,  64, 0, stream>>>(p); break;
        case  96: ctrl_batch_down< 96><<<grid_down,  96, 0, stream>>>(p); break;
        case 128: ctrl_batch_down<128><<<grid_down, 128, 0, stream>>>(p); break;
        case 160: ctrl_batch_down<160><<<grid_down, 160, 0, stream>>>(p); break;
        case 192: ctrl_batch_down<192><<<grid_down, 192, 0, stream>>>(p); break;
        case 224: ctrl_batch_down<224><<<grid_down, 224, 0, stream>>>(p); break;
        case 256: ctrl_batch_down<256><<<grid_down, 256, 0, stream>>>(p); break;
        default: GGML_ABORT("ctrl_batch: block size %d", bs);
    }

    constexpr int block_size = 256;
    const dim3 grid_mix((p.nrows + block_size/warp_size - 1)/(block_size/warp_size), n);
    switch (t.wf->type) {
        case GGML_TYPE_F32:  ctrl_batch_mix<float>      <<<grid_mix, block_size, 0, stream>>>(p); break;
        case GGML_TYPE_F16:  ctrl_batch_mix<half>       <<<grid_mix, block_size, 0, stream>>>(p); break;
        case GGML_TYPE_BF16: ctrl_batch_mix<nv_bfloat16><<<grid_mix, block_size, 0, stream>>>(p); break;
        default: GGML_ABORT("ctrl_batch: weight type");
    }
}
