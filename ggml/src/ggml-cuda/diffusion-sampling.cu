#include "diffusion-sampling.cuh"

#include <cfloat>
#include <map>
#include <mutex>

// One block per canvas position. Parallel max->argmax, parallel Z and T (T=sum d*e), entropy=logZ-T/Z,
// then thread 0 walks the multinomial CDF with r=u[row]*Z. Only the reduction order differs from the host
// worker, so argmax is exact and Z/entropy match to FP reduction tolerance (~1e-4 rel). Dense, top_k==0.
static __global__ void diffusion_dense_sample_kernel(
        const float * __restrict__ logits,
        const float * __restrict__ u,
        int   * __restrict__ argmax,
        float * __restrict__ entropy,
        int   * __restrict__ sampled,
        const int   n_vocab,
        const float inv_temp) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_val[256];
    __shared__ float s_sum[256];
    __shared__ int   s_idx[256];

    const float * row_logits = logits + (size_t) row * n_vocab;

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float x = row_logits[v] * inv_temp;
        if (x > local_max) { local_max = x; local_idx = v; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    const float max_l = s_val[0];
    const int   amax  = s_idx[0];

    float local_sum = 0.0f;
    float local_t   = 0.0f;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float d = row_logits[v] * inv_temp - max_l;
        const float e = expf(d);
        local_sum += e;
        local_t   += d * e;
    }
    s_sum[tid] = local_sum;
    s_val[tid] = local_t;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_val[tid] += s_val[tid + stride];
        }
        __syncthreads();
    }
    const float z = s_sum[0];
    const float t = s_val[0];
    if (tid == 0) {
        argmax[row]  = amax;
        entropy[row] = logf(z) - t / z;
    }
    __syncthreads();

    // multinomial draw (first v with cumulative exp(d) >= r, in vocab order). Split the vocab into 256
    // contiguous slices: each thread sums its slice, we exclusive-scan the slice sums, then only the thread
    // whose slice spans r walks its ~ceil(n_vocab/256) elements serially. Serial work drops from n_vocab to
    // one slice; pick the same vocab-order first-crossing as the host (FP reduction order aside).
    const float r = u[row] * z;
    const int   chunk = (n_vocab + blockDim.x - 1) / blockDim.x;
    const int   beg   = tid * chunk;
    const int   end   = min(beg + chunk, n_vocab);

    float slice_sum = 0.0f;
    for (int v = beg; v < end; ++v) {
        slice_sum += expf(row_logits[v] * inv_temp - max_l);
    }
    s_sum[tid] = slice_sum;
    __syncthreads();

    __shared__ int s_tok;
    if (tid == 0) {
        s_tok    = n_vocab - 1;                // host default if cum never reaches r (FP guard)
        s_idx[0] = -1;                         // no crossing slice -> no thread walks, default stands
        float pref = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) { // exclusive scan + locate the crossing slice (256 iters)
            const float next = pref + s_sum[i];
            if (next >= r) { s_idx[0] = i; s_val[0] = pref; break; }
            pref = next;
        }
    }
    __syncthreads();

    if (tid == s_idx[0]) {                      // only the crossing thread walks its slice from its prefix
        float cum = s_val[0];
        for (int v = beg; v < end; ++v) {
            cum += expf(row_logits[v] * inv_temp - max_l);
            if (cum >= r) { s_tok = v; break; }
        }
    }
    __syncthreads();
    if (tid == 0) { sampled[row] = s_tok; }
}

// Per-device scratch: u + the 3 outputs, grow-only and cached so the steady state has no cudaMalloc.
struct dg_devsample_scratch {
    float * u       = nullptr;
    int   * argmax  = nullptr;
    float * entropy = nullptr;
    int   * sampled = nullptr;
    int     cap     = 0;
};

static std::mutex g_dg_devsample_mutex;
static std::map<int, dg_devsample_scratch> g_dg_devsample;

static void dg_devsample_reserve(dg_devsample_scratch & s, int n) {
    if (s.cap >= n) { return; }
    if (s.u)       { CUDA_CHECK(cudaFree(s.u)); }
    if (s.argmax)  { CUDA_CHECK(cudaFree(s.argmax)); }
    if (s.entropy) { CUDA_CHECK(cudaFree(s.entropy)); }
    if (s.sampled) { CUDA_CHECK(cudaFree(s.sampled)); }
    CUDA_CHECK(cudaMalloc((void **) &s.u,       (size_t) n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s.argmax,  (size_t) n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &s.entropy, (size_t) n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s.sampled, (size_t) n * sizeof(int)));
    s.cap = n;
}

bool ggml_cuda_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp) {
    if (!logits || !u_host || !argmax_host || !entropy_host || !sampled_host || n_tokens <= 0) {
        return false;
    }
    if (logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) || logits->data == nullptr) {
        return false;
    }
    const int n_vocab = (int) logits->ne[0];
    if (n_vocab <= 0 || (int) ggml_nrows(logits) < n_tokens) {
        return false;
    }
    if (!logits->buffer || ggml_backend_buffer_is_host(logits->buffer)) {
        return false;  // host tensor -> caller falls back to the host path
    }
    const float * logits_d = (const float *) logits->data;

    // gated to a single CUDA device, so the tensor is on the current device; run there on the default
    // stream (the caller has already synchronized the backend).
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    std::lock_guard<std::mutex> lock(g_dg_devsample_mutex);
    dg_devsample_scratch & s = g_dg_devsample[device];
    dg_devsample_reserve(s, n_tokens);

    CUDA_CHECK(cudaMemcpyAsync(s.u, u_host, (size_t) n_tokens * sizeof(float), cudaMemcpyHostToDevice, 0));
    diffusion_dense_sample_kernel<<<n_tokens, 256, 0, 0>>>(
            logits_d, s.u, s.argmax, s.entropy, s.sampled, n_vocab, inv_temp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(argmax_host,  s.argmax,  (size_t) n_tokens * sizeof(int),   cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(entropy_host, s.entropy, (size_t) n_tokens * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(sampled_host, s.sampled, (size_t) n_tokens * sizeof(int),   cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    return true;
}
