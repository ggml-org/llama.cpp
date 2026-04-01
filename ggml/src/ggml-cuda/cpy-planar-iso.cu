/*
 * CUDA kernels for F16 → PlanarQuant/IsoQuant bulk conversion.
 * Used by ggml_cpy to convert deferred F16 KV cache to quantized format.
 *
 * Four conversions: F16→planar3, F16→planar4, F16→iso3, F16→iso4
 * Each: read F16 → convert to F32 → apply rotation → quantize → pack
 */

#include "common.cuh"
#include "ggml-common.h"

#include <cmath>

// ── Rotation constants (must match Python exactly) ────────────────────
// Generated from: torch.manual_seed(42); torch.rand(64) * 2π → cos/sin
// And: torch.randn(32,4, generator=seed42) → normalize → quaternions

// Planar: 64 cos/sin pairs for 2D Givens rotation
__constant__ float d_planar_cos[64];
__constant__ float d_planar_sin[64];

// IsoQuant: 32 unit quaternions (w,x,y,z) for 4D rotation
__constant__ float d_iso_qw[32];
__constant__ float d_iso_qx[32];
__constant__ float d_iso_qy[32];
__constant__ float d_iso_qz[32];

// 3-bit centroids (8 levels) — same as turbo3
__constant__ float d_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// 4-bit centroids (16 levels) — same as turbo4
__constant__ float d_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// 3-bit midpoints for fast quantization
__constant__ float d_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// ── Device helpers ───────────────────────────────────────────────────

__device__ __forceinline__ uint8_t quantize_3bit(float val) {
    uint8_t idx = 0;
    if      (val < d_mid_3bit[0]) idx = 0;
    else if (val < d_mid_3bit[1]) idx = 1;
    else if (val < d_mid_3bit[2]) idx = 2;
    else if (val < d_mid_3bit[3]) idx = 3;
    else if (val < d_mid_3bit[4]) idx = 4;
    else if (val < d_mid_3bit[5]) idx = 5;
    else if (val < d_mid_3bit[6]) idx = 6;
    else                          idx = 7;
    return idx;
}

__device__ __forceinline__ uint8_t quantize_4bit(float val) {
    uint8_t best = 0;
    float best_d = fabsf(val - d_centroids_4bit[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float d = fabsf(val - d_centroids_4bit[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

// ── Planar3: F16 → block_planar3_0 (2D Givens + 3-bit) ─────────────

__global__ void kernel_cpy_f16_planar3(
    const half * __restrict__ src,
    block_planar3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_PLANAR3;
    block_planar3_0 * blk = &dst[ib];

    // Load and compute norm
    float buf[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    // Forward Givens rotation per pair
    float rotated[128];
    for (int p = 0; p < 64; p++) {
        float c = d_planar_cos[p], s_val = d_planar_sin[p];
        rotated[p*2]   = c * buf[p*2] - s_val * buf[p*2+1];
        rotated[p*2+1] = s_val * buf[p*2] + c * buf[p*2+1];
    }

    // Quantize + pack (3-bit: 2-bit qs + 1-bit signs)
    for (int j = 0; j < QK_PLANAR3/4; j++) blk->qs[j] = 0;
    for (int j = 0; j < QK_PLANAR3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += d_centroids_3bit[idx] * d_centroids_3bit[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    float corrected = recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm;
    blk->norm = __float2half(corrected);
}

// ── Planar4: F16 → block_planar4_0 (2D Givens + 4-bit nibble) ──────

__global__ void kernel_cpy_f16_planar4(
    const half * __restrict__ src,
    block_planar4_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_PLANAR4;
    block_planar4_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR4; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_PLANAR4; j++) buf[j] *= inv_norm;

    float rotated[128];
    for (int p = 0; p < 64; p++) {
        float c = d_planar_cos[p], s_val = d_planar_sin[p];
        rotated[p*2]   = c * buf[p*2] - s_val * buf[p*2+1];
        rotated[p*2+1] = s_val * buf[p*2] + c * buf[p*2+1];
    }

    for (int j = 0; j < 64; j++) blk->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = quantize_4bit(rotated[j]);
        blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += d_centroids_4bit[idx] * d_centroids_4bit[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    blk->rnorm = __float2half(0.0f);
}

// ── Iso3: F16 → block_iso3_0 (quaternion 4D + 3-bit) ───────────────

__global__ void kernel_cpy_f16_iso3(
    const half * __restrict__ src,
    block_iso3_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_ISO3;
    block_iso3_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_ISO3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO3; j++) buf[j] *= inv_norm;

    // Forward quaternion rotation per 4D group
    float rotated[128];
    for (int g = 0; g < 32; g++) {
        float qw = d_iso_qw[g], qx = d_iso_qx[g], qy = d_iso_qy[g], qz = d_iso_qz[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    for (int j = 0; j < QK_ISO3/4; j++) blk->qs[j] = 0;
    for (int j = 0; j < QK_ISO3/8; j++) blk->signs[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_ISO3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4)*2);
        if (idx & 0x4) blk->signs[j/8] |= (1 << (j%8));
        recon_sq += d_centroids_3bit[idx] * d_centroids_3bit[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
}

// ── Iso4: F16 → block_iso4_0 (quaternion 4D + 4-bit nibble) ────────

__global__ void kernel_cpy_f16_iso4(
    const half * __restrict__ src,
    block_iso4_0 * __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const half * s = src + ib * QK_ISO4;
    block_iso4_0 * blk = &dst[ib];

    float buf[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_ISO4; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < QK_ISO4; j++) buf[j] *= inv_norm;

    float rotated[128];
    for (int g = 0; g < 32; g++) {
        float qw = d_iso_qw[g], qx = d_iso_qx[g], qy = d_iso_qy[g], qz = d_iso_qz[g];
        float v0 = buf[g*4], v1 = buf[g*4+1], v2 = buf[g*4+2], v3 = buf[g*4+3];
        rotated[g*4]   = qw*v0 - qx*v1 - qy*v2 - qz*v3;
        rotated[g*4+1] = qw*v1 + qx*v0 + qy*v3 - qz*v2;
        rotated[g*4+2] = qw*v2 - qx*v3 + qy*v0 + qz*v1;
        rotated[g*4+3] = qw*v3 + qx*v2 - qy*v1 + qz*v0;
    }

    for (int j = 0; j < 64; j++) blk->qs[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = quantize_4bit(rotated[j]);
        blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
        recon_sq += d_centroids_4bit[idx] * d_centroids_4bit[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    blk->norm = __float2half(recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm);
    blk->rnorm = __float2half(0.0f);
}

// ── Host dispatch functions (called from cpy.cu) ────────────────────

static bool constants_initialized = false;

void ggml_cuda_init_planar_iso_constants() {
    if (constants_initialized) return;

    // Hardcoded rotation constants — must match planar-iso-constants.cuh exactly.
    // Generated from LCG PRNG with seed=42 (verified against C reference code).
    static const float h_cos[64] = {-0.9095053397f,0.1535578452f,-0.8537489227f,-0.6827218011f,-0.4249387949f,0.9864510046f,0.9906673944f,0.5752363372f,-0.9866459035f,0.9878848090f,-0.6215683804f,-0.9835597698f,0.8777263755f,-0.4624640047f,0.2843135922f,-0.7739960698f,0.2385234222f,0.9121914932f,-0.8815003943f,-0.2639699512f,-0.5517087300f,-0.9035294557f,-0.8520543188f,-0.5600635985f,-0.7667286376f,-0.9877949369f,-0.9781949787f,-0.9953372831f,-0.8622053901f,-0.7382118186f,0.9136037642f,-0.2558504503f,-0.8541000475f,-0.6159335408f,0.9861256679f,-0.6758560284f,0.4249571682f,-0.6219544719f,0.9130573430f,-0.5948161096f,0.5759782996f,0.9729901203f,0.6535998325f,0.9222195491f,-0.7668084044f,0.5116178563f,-0.7848786574f,0.9902111051f,0.1997167840f,0.7173003220f,-0.9999998006f,-0.9557868691f,0.5594852693f,-0.9980111824f,0.9782398557f,-0.9150004329f,-0.4084754305f,0.0071549185f,0.9558482753f,-0.0971921648f,-0.9469334002f,0.9999492419f,0.6100589016f,0.0350818915f};
    static const float h_sin[64] = {-0.4156922383f,0.9881396603f,0.5206849114f,-0.7306784124f,-0.9052220836f,0.1640561354f,0.1363015542f,0.8179872593f,0.1628798979f,0.1551889303f,0.7833599099f,-0.1805828875f,-0.4791621957f,0.8866380571f,-0.9587313395f,0.6331904010f,-0.9711367448f,0.4097641756f,0.4721832852f,-0.9645309040f,0.8340368561f,0.4285259884f,0.5234533769f,0.8284496156f,0.6419713361f,-0.1557599517f,-0.2076886701f,0.0964556523f,0.5065588468f,-0.6745689815f,-0.4066056591f,-0.9667163736f,0.5201087471f,-0.7877981171f,0.1660005034f,-0.7370336688f,0.9052134584f,0.7830534049f,-0.4078312009f,-0.8038618014f,0.8174649829f,-0.2308467584f,-0.7568403127f,-0.3866666566f,0.6418760557f,-0.8592131104f,0.6196494922f,0.1395778183f,0.9798536657f,0.6967641265f,-0.0006314605f,0.2940603015f,0.8288402943f,-0.0630371303f,0.2074771907f,0.4034528570f,0.9127693152f,-0.9999744032f,0.2938606379f,0.9952656344f,0.3214298299f,0.0100754012f,-0.7923560668f,-0.9993844410f};
    CUDA_CHECK(cudaMemcpyToSymbol(d_planar_cos, h_cos, sizeof(h_cos)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_planar_sin, h_sin, sizeof(h_sin)));

    static const float h_qw[32] = {0.8350809813f,-0.1648498178f,0.1283752173f,0.2897698581f,-0.1820549369f,0.9549587369f,-0.8741137385f,0.8988990188f,-0.1312584430f,-0.3990598321f,-0.2694816887f,-0.1181898862f,0.1363395452f,0.2665117681f,-0.8263269663f,-0.1834189594f,0.3098247349f,0.2804697454f,-0.5655074716f,-0.1627507508f,0.8684155941f,0.2233296037f,-0.1291671842f,0.6606932878f,-0.5694432259f,-0.2782760859f,0.5113853812f,-0.5139024258f,0.7489815354f,-0.3037399948f,-0.4143463373f,-0.3524050117f};
    static const float h_qx[32] = {0.3547102809f,-0.5782636404f,-0.8299785256f,0.5694668293f,-0.8199930191f,0.1259543896f,-0.3090814352f,-0.2613596618f,-0.1660282463f,-0.5143862963f,0.5898610353f,-0.8277072310f,-0.6826571226f,-0.1740629375f,0.1416199356f,0.4648889899f,0.3485621810f,0.8982698917f,-0.3015249372f,0.4990116358f,0.2398942262f,-0.7447698116f,0.4783197045f,0.0735855624f,-0.2975912094f,-0.0700704753f,0.2975627482f,-0.2652103305f,-0.1539765000f,0.0849994123f,-0.1069803685f,-0.5753474832f};
    static const float h_qy[32] = {0.2416850179f,-0.4488199651f,0.3478420675f,0.5024775267f,0.1696543097f,0.1760476083f,0.0254505407f,0.2389279008f,-0.9429193735f,0.3925755024f,-0.2757458389f,-0.1485267133f,0.5530825853f,-0.8936085105f,0.2953715622f,-0.5285226703f,0.7939327955f,0.0139789311f,-0.2555710375f,0.4543992281f,-0.2698826790f,-0.4736968279f,0.4361720681f,-0.3461222053f,0.0792116225f,0.8827795386f,0.7416539788f,-0.3826399446f,-0.3534849286f,-0.8696597815f,-0.6908422709f,0.2082736641f};
    static const float h_qz[32] = {0.3038694561f,0.4734756052f,-0.3878843784f,0.5831694603f,-0.5054479241f,-0.1731694490f,-0.3737666607f,0.2328704894f,0.2621760964f,0.6239953637f,-0.7082104683f,0.5308507681f,-0.4413037896f,-0.2802782655f,-0.4522367120f,-0.6698107123f,-0.3752456903f,-0.3359423280f,0.7181019187f,0.7106907368f,0.3100073636f,0.4016827941f,0.7350437641f,-0.6607965231f,0.7619289756f,0.3648703992f,-0.3040413559f,0.7213236690f,0.5280022621f,-0.3742936850f,-0.5760775208f,0.7015634775f};
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qw, h_qw, sizeof(h_qw)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qx, h_qx, sizeof(h_qx)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qy, h_qy, sizeof(h_qy)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_iso_qz, h_qz, sizeof(h_qz)));

    constants_initialized = true;
}

void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_PLANAR3;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_planar3<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_planar3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_PLANAR4;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_planar4<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_planar4_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_ISO3;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_iso3<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_iso3_0 *)dst, n_blocks);
}

void ggml_cuda_cpy_f16_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_init_planar_iso_constants();
    const int64_t n_blocks = ne / QK_ISO4;
    const int threads = 256;
    const int blocks = (n_blocks + threads - 1) / threads;
    kernel_cpy_f16_iso4<<<blocks, threads, 0, stream>>>(
        (const half *)src, (block_iso4_0 *)dst, n_blocks);
}
