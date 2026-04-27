/**
 * turboquant_hip.hip
 * ROCm / HIP implementation of TurboQuant kernels
 *
 * Build:
 *   hipcc -O3 -arch=gfx1201 turboquant_hip.hip turboquant.cpp -o turboquant.o
 *
 * Kernels:
 *   1. tq_rotate_whd_kernel   - Walsh-Hadamard + sign-flip random rotation
 *   2. tq_quantize_kernel     - Lloyd-Max scalar quantization (per coordinate)
 *   3. tq_qjl_kernel          - 1-bit QJL residual sign projection
 *   4. tq_attn_logits_kernel  - Direct Memory Access (no LDS, uint4 + __ldg)
 *   5. tq_attn_output_kernel  - softmax + weighted sum (values)
 */

#include "turboquant.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <cmath>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <cstring>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t e = (cmd); \
        if (e != hipSuccess) { \
            fprintf(stderr, "[TurboQuant] HIP error %s at %s:%d\n", \
                    hipGetErrorString(e), __FILE__, __LINE__); \
            return -1; \
        } \
    } while(0)

#ifdef __gfx1201__
#  define WARP_SIZE  32   // RDNA4: RX 9070 / R9700
#elif defined(__gfx1200__)
#  define WARP_SIZE  32   // RDNA4: RX 9060 XT
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1030__)
#  define WARP_SIZE  32   // RDNA2/3
#else
#  define WARP_SIZE  64   // CDNA / GCN fallback
#endif

#define MAX_DIM      256

// ─────────────────────────────────────────────────────────────────────────────
// Device-side codebook (loaded from host constants, read-only)
// ─────────────────────────────────────────────────────────────────────────────
__constant__ float d_cb2_levels[4] = {
    -0.7071f, -0.2357f,  0.2357f,  0.7071f
};
__constant__ float d_cb2_thresh[5] = {
    -1.0f, -0.5f, 0.0f, 0.5f, 1.0f
};

__constant__ float d_cb3_levels[8] = {
    -0.9009f, -0.6235f, -0.3612f, -0.1205f,
     0.1205f,  0.3612f,  0.6235f,  0.9009f
};
__constant__ float d_cb3_thresh[9] = {
    -1.0f, -0.7654f, -0.5000f, -0.2393f, 0.0f,
     0.2393f, 0.5000f, 0.7654f, 1.0f
};

__constant__ float d_cb4_levels[16] = {
    -0.9613f, -0.8315f, -0.6895f, -0.5345f,
    -0.3681f, -0.1951f, -0.0980f,  0.0000f,
     0.0980f,  0.1951f,  0.3681f,  0.5345f,
     0.6895f,  0.8315f,  0.9613f,  0.9903f
};
__constant__ float d_cb4_thresh[17] = {
    -1.0f,    -0.9009f, -0.7654f, -0.6124f,
    -0.4540f, -0.2820f, -0.1464f, -0.0490f,
     0.0490f,  0.1464f,  0.2820f,  0.4540f,
     0.6124f,  0.7654f,  0.9009f,  0.9808f, 1.0f
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Walsh-Hadamard Random Rotation
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tq_rotate_whd_kernel(
    float       *__restrict__ x,
    const float *__restrict__ sign1,
    const float *__restrict__ sign2,
    int head_dim,
    int n_heads,
    float inv_sqrt_d)
{
    __shared__ float smem[MAX_DIM];

    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x;

    if (tid >= head_dim) return;

    int base = (tok * n_heads + head) * head_dim;
    float val = x[base + tid] * sign1[tid];

    for (int stride = 1; stride < WARP_SIZE && stride < head_dim; stride <<= 1) {
        float other = __shfl_xor(val, stride);
        if ((tid & stride) == 0) val = val + other;
        else                     val = other - val;
    }

    if (head_dim > WARP_SIZE) {
        smem[tid] = val;
        __syncthreads();

        for (int stride = WARP_SIZE; stride < head_dim; stride <<= 1) {
            int pair = tid ^ stride;
            float a = smem[tid];
            float b = smem[pair];
            __syncthreads();
            if ((tid & stride) == 0) {
                smem[tid]  = a + b;
                smem[pair] = a - b;
            }
            __syncthreads();
        }
        val = smem[tid];
    }

    x[base + tid] = val * sign2[tid] * inv_sqrt_d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Lloyd-Max Scalar Quantization
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
int tq_scalar_quantize(float v, int bits) {
    v = fmaxf(-1.0f, fminf(1.0f, v));

    if (bits == 2) {
        if (v < d_cb2_thresh[1]) return 0;
        if (v < d_cb2_thresh[2]) return 1;
        if (v < d_cb2_thresh[3]) return 2;
        return 3;
    } else if (bits == 3) {
        int lo = 0, hi = 7;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (v < d_cb3_thresh[mid + 1]) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    } else {
        int lo = 0, hi = 15;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (v < d_cb4_thresh[mid + 1]) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
}

__device__ __forceinline__
float tq_scalar_reconstruct(int idx, int bits) {
    if (bits == 2) return d_cb2_levels[idx];
    if (bits == 3) return d_cb3_levels[idx];
    return d_cb4_levels[idx];
}

__global__ void tq_quantize_kernel(
    const float   *__restrict__ x_rot,
    uint8_t       *__restrict__ q_out,
    float         *__restrict__ recon,
    int head_dim,
    int n_heads,
    int bits)
{
    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x;

    if (tid >= head_dim) return;

    int vec_idx = tok * n_heads + head;
    float v = x_rot[vec_idx * head_dim + tid];

    int q_idx = tq_scalar_quantize(v, bits);
    float r   = tq_scalar_reconstruct(q_idx, bits);

    if (recon) recon[vec_idx * head_dim + tid] = r;

    int   byte_pos = (tid * bits) / 8;
    int   bit_off  = (tid * bits) % 8;
    int   bytes_per_vec = (head_dim * bits + 7) / 8;
    uint8_t *base = q_out + vec_idx * bytes_per_vec;

    uint32_t mask = ((1u << bits) - 1u);
    uint32_t val  = (uint32_t)q_idx & mask;

    if (bits <= 8) {
        atomicOr((unsigned int *)(base + byte_pos),
                 (unsigned int)(val << bit_off));
        if (bit_off + bits > 8) {
            int overflow = bit_off + bits - 8;
            atomicOr((unsigned int *)(base + byte_pos + 1),
                     (unsigned int)(val >> (bits - overflow)));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Quantized Johnson-Lindenstrauss (QJL) Residual
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tq_qjl_kernel(
    const float   *__restrict__ x_orig,
    const float   *__restrict__ recon_rot,
    const float   *__restrict__ sign1,
    const float   *__restrict__ sign2,
    const float   *__restrict__ qjl_proj,
    uint8_t       *__restrict__ qjl_out,
    int head_dim,
    int n_heads,
    float inv_sqrt_d)
{
    __shared__ float smem[MAX_DIM];

    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x;
    int vec  = tok * n_heads + head;

    if (tid >= head_dim) return;

    smem[tid] = recon_rot[vec * head_dim + tid] * sign2[tid];
    __syncthreads();

    for (int stride = 1; stride < head_dim; stride <<= 1) {
        int pair = tid ^ stride;
        float a = smem[tid], b = smem[pair];
        __syncthreads();
        if ((tid & stride) == 0) {
            smem[tid]  = a + b;
            smem[pair] = a - b;
        }
        __syncthreads();
    }
    float recon_orig = smem[tid] * sign1[tid] * inv_sqrt_d;

    float residual = x_orig[vec * head_dim + tid] - recon_orig;

    smem[tid] = residual;
    __syncthreads();

    float proj = 0.0f;
    const float *S_row = qjl_proj + tid * head_dim;
    for (int i = 0; i < head_dim; i++) {
        proj += S_row[i] * smem[i];
    }

    int   qjl_sign = (proj >= 0.0f) ? 1 : 0;
    int   bytes_per_vec = (head_dim + 7) / 8;
    uint8_t *base = qjl_out + vec * bytes_per_vec;
    atomicOr((unsigned int *)(base + tid / 8),
             (unsigned int)(qjl_sign << (tid % 8)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: Direct Memory Access (No LDS, uint4 + __ldg())
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Direct Memory Access Kernel for RX 9070 (gfx1201)
 * - No LDS (shared memory) - pure direct global memory access
 * - uint4 x 128-bit vector loads
 * - 128-byte alignment
 * - __ldg() for Infinity Cache optimization
 */
__global__ void tq_attn_logits_kernel(
    const half    *__restrict__ query,
    const uint8_t *__restrict__ keys_q,
    const uint8_t *__restrict__ keys_qjl,
    const float   *__restrict__ sign1,
    const float   *__restrict__ sign2,
    const float   *__restrict__ qjl_proj,
    float         *__restrict__ logits,
    int head_dim,
    int n_heads,
    int n_tokens,
    int key_bits,
    float inv_sqrt_d,
    float scale)
{
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    int lane = tid & 31;

    if (tid >= head_dim) return;

    // Rotate query with WHT (register-based, no LDS)
    float q_orig = __half2float(query[head * head_dim + tid]);
    float q_rot  = q_orig * sign1[tid];

    for (int stride = 1; stride < head_dim; stride <<= 1) {
        float other = __shfl_xor(q_rot, stride);
        if ((tid & stride) == 0) q_rot += other;
        else                     q_rot = other - q_rot;
    }
    q_rot = q_rot * sign2[tid] * inv_sqrt_d;

    // QJL projection: Sq = S * q_orig
    float sq = 0.0f;
    const float *S_row = qjl_proj + tid * head_dim;
    #pragma unroll 4
    for (int i = 0; i < head_dim; i++) {
        sq += S_row[i] * q_orig;
    }

    // Memory alignment parameters
    int bytes_per_key = (head_dim * key_bits + 7) / 8;
    int qjl_bytes    = (head_dim + 7) / 8;
    int align_bytes  = (bytes_per_key + 127) & ~127;
    int align_qjl    = (qjl_bytes + 127) & ~127;
    float qjl_scale  = (float)(M_PI / 2.0f) / (float)head_dim;

    // Token loop: Direct Memory Access
    for (int tok = blockIdx.y; tok < n_tokens; tok += gridDim.y) {
        // 128-byte aligned addresses
        int base_off = ((tok * n_heads + head) * align_bytes);
        int qjl_off  = ((tok * n_heads + head) * align_qjl);

        const uint8_t *k_ptr = keys_q + base_off;
        const uint8_t *s_ptr = keys_qjl + qjl_off;

        // Direct 128-bit (16 bytes) vector loads per thread lane
        int vec_base = lane * 16;
        if (vec_base < align_bytes) {
            uint4 v0 = __ldg((const uint4*)(k_ptr + vec_base + 0));
            uint4 v1 = __ldg((const uint4*)(k_ptr + vec_base + 16));
            uint4 v2 = __ldg((const uint4*)(k_ptr + vec_base + 32));
            uint4 v3 = __ldg((const uint4*)(k_ptr + vec_base + 48));
            (void)v0; (void)v1; (void)v2; (void)v3;
        }

        // Bit-unpack key[tid]
        int bit_pos  = tid * key_bits;
        int byte_off = bit_pos >> 3;
        int bit_off  = bit_pos & 7;

        uint32_t bits = 0;
        if (byte_off < align_bytes) {
            bits = __ldg((const uint32_t*)(k_ptr + byte_off));
            if (bit_off + key_bits > 8 && byte_off + 1 < align_bytes) {
                bits |= ((uint32_t)__ldg((const uint8_t*)(k_ptr + byte_off + 1))) << 8;
            }
        }
        uint32_t raw = (bits >> bit_off) & ((1u << key_bits) - 1u);
        float k_rot = tq_scalar_reconstruct((int)raw, key_bits);

        // QJL sign bit
        int sign_byte = tid >> 3;
        uint8_t sign_val = 0;
        if (sign_byte < align_qjl) {
            sign_val = __ldg((const uint8_t*)(s_ptr + sign_byte));
        }
        int sign_bit = (sign_val >> (tid & 7)) & 1;
        float k_sign = sign_bit ? 1.0f : -1.0f;

        // Inner products (all in registers)
        float ip1 = q_rot * k_rot;
        float ip2 = sq * k_sign * qjl_scale;
        float ip  = ip1 + ip2;

        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            ip += __shfl_xor(ip, offset);
        }

        // Write result (lane 0)
        if (lane == 0) {
            logits[head * n_tokens + tok] = ip * scale;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: Softmax + Weighted Value Sum
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tq_attn_output_kernel(
    const float   *__restrict__ logits,
    const uint8_t *__restrict__ vals_q,
    const uint8_t *__restrict__ vals_qjl,
    half          *__restrict__ out,
    int head_dim,
    int n_heads,
    int n_tokens,
    int val_bits)
{
    int head = blockIdx.x;
    int dim  = blockIdx.y;

    if (head >= n_heads || dim >= head_dim) return;

    int bytes_per_val = (head_dim * val_bits + 7) / 8;

    const float *lgt = logits + head * n_tokens;
    float max_l = lgt[0];
    for (int t = 1; t < n_tokens; t++) max_l = fmaxf(max_l, lgt[t]);

    float sum_exp = 0.0f;
    for (int t = 0; t < n_tokens; t++) sum_exp += expf(lgt[t] - max_l);

    int   bit_pos  = dim * val_bits;
    int   byte_pos = bit_pos / 8;
    int   bit_off  = bit_pos % 8;
    uint32_t mask  = (1u << val_bits) - 1u;

    float acc = 0.0f;
    for (int tok = 0; tok < n_tokens; tok++) {
        int   head_tok = tok * n_heads + head;
        const uint8_t *vq = vals_q + head_tok * bytes_per_val;

        uint32_t raw = (uint32_t)vq[byte_pos] >> bit_off;
        if (bit_off + val_bits > 8)
            raw |= (uint32_t)vq[byte_pos + 1] << (8 - bit_off);
        raw &= mask;

        float v_recon = tq_scalar_reconstruct((int)raw, val_bits);
        float weight  = expf(lgt[tok] - max_l) / sum_exp;
        acc += weight * v_recon;
    }

    out[head * head_dim + dim] = __float2half(acc);
}

// ─────────────────────────────────────────────────────────────────────────────
// fp16 to float conversion kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tq_fp16_to_float_kernel(
    const half *__restrict__ src,
    float      *__restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher functions
// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

int tq_launch_rotate(
    float *d_x, const float *d_sign1, const float *d_sign2,
    int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream)
{
    if ((head_dim & (head_dim - 1)) != 0 || head_dim > MAX_DIM) {
        fprintf(stderr, "[TurboQuant] head_dim must be power of 2 and <= %d\n", MAX_DIM);
        return -1;
    }
    dim3 grid(n_tokens, n_heads);
    dim3 block(head_dim);
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    hipLaunchKernelGGL(tq_rotate_whd_kernel, grid, block, 0, (hipStream_t)stream,
        d_x, d_sign1, d_sign2, head_dim, n_heads, inv_sqrt_d);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_quantize(
    const float *d_x_rot, uint8_t *d_q_out, float *d_recon,
    int n_tokens, int n_heads, int head_dim, int bits, tq_hip_stream_t stream)
{
    dim3 grid(n_tokens, n_heads);
    dim3 block(head_dim);
    hipLaunchKernelGGL(tq_quantize_kernel, grid, block, 0, (hipStream_t)stream,
        d_x_rot, d_q_out, d_recon, head_dim, n_heads, bits);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_qjl(
    const float *d_x_orig, const float *d_recon_rot,
    const float *d_sign1, const float *d_sign2, const float *d_qjl_proj,
    uint8_t *d_qjl_out,
    int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream)
{
    dim3 grid(n_tokens, n_heads);
    dim3 block(head_dim);
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    hipLaunchKernelGGL(tq_qjl_kernel, grid, block, 0, (hipStream_t)stream,
        d_x_orig, d_recon_rot, d_sign1, d_sign2, d_qjl_proj,
        d_qjl_out, head_dim, n_heads, inv_sqrt_d);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_attn_logits(
    const void *d_query, const uint8_t *d_keys_q, const uint8_t *d_keys_qjl,
    const float *d_sign1, const float *d_sign2, const float *d_qjl_proj,
    float *d_logits,
    int n_heads, int head_dim, int n_tokens, int key_bits, tq_hip_stream_t stream)
{
    int blocks_y = (n_tokens + 127) / 128; 
    if (blocks_y > 256) blocks_y = 256;

    dim3 grid(n_heads, blocks_y);
    dim3 block(head_dim);
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    hipLaunchKernelGGL(tq_attn_logits_kernel, grid, block,
        head_dim * sizeof(float) * 4, (hipStream_t)stream,
        (const half*)d_query, d_keys_q, d_keys_qjl,
        d_sign1, d_sign2, d_qjl_proj,
        d_logits, head_dim, n_heads, n_tokens, key_bits,
        inv_sqrt_d, attn_scale);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_attn_output(
    const float *d_logits, const uint8_t *d_vals_q, const uint8_t *d_vals_qjl,
    void *d_output,
    int n_heads, int head_dim, int n_tokens, int val_bits, tq_hip_stream_t stream)
{
    dim3 grid(n_heads, head_dim);
    dim3 block(1);
    hipLaunchKernelGGL(tq_attn_output_kernel, grid, block, 0, (hipStream_t)stream,
        d_logits, d_vals_q, d_vals_qjl, (half*)d_output,
        head_dim, n_heads, n_tokens, val_bits);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_fp16_to_float(
    const void *d_src, float *d_dst, int n, tq_hip_stream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    hipLaunchKernelGGL(tq_fp16_to_float_kernel, grid, block, 0, (hipStream_t)stream,
        (const half*)d_src, d_dst, n);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

}