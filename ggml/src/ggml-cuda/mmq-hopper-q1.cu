// Hopper (sm_90a) wgmma MMQ path for Q1_0: dequant-in-SMEM + int8 wgmma with exact per-block scaling.
// Experimental opt-in path (env GGML_HOPPER_Q1) targeting large-batch prefill on sm_90a.
// Activations are quantized fp32 -> int8 with a per-128-K absmax scale (coarser than q8_1's per-32;
// flagged for KLD validation). Dispatched only when M,N,K % 128 == 0 and cc >= 900; otherwise the
// caller falls through to the standard MMQ path.
#include "common.cuh"
#include <unordered_map>

#if defined(GGML_USE_HOPPER_Q1)  // built only when CUTLASS include dir is provided
#include <cuda_pipeline.h>
#include <cute/tensor.hpp>
using namespace cute;

namespace hopper_q1 {

static constexpr int bM = 128, bN = 128, bK = 128;
using MmaAtom = GMMA::MMA_64x64x32_S32S8S8_SS_TN;

// fp32 -> int8 with per-128 absmax scale. Warp-per-group: 32 lanes x 4 floats (float4 loads),
// shuffle reduction, vectorized 4x-int8 stores. 8 groups per 256-thread block.
__global__ void quant_act_per128(const float* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ d,
                                 int M, int K) {
  const int ngroups = M * (K / 128);
  const int g = blockIdx.x * 8 + threadIdx.x / 32;  // group = (m, kc)
  if (g >= ngroups) return;
  const int lane = threadIdx.x % 32;
  const int m = g / (K / 128), kc = g % (K / 128);
  const float4* xs = reinterpret_cast<const float4*>(x + (size_t)m * K + kc * 128) + lane;
  float4 v = *xs;
  float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  const float scale = amax / 127.0f;
  const float inv = scale > 0.f ? 1.0f / scale : 0.f;
  char4 out = make_char4((char)lrintf(v.x * inv), (char)lrintf(v.y * inv), (char)lrintf(v.z * inv),
                         (char)lrintf(v.w * inv));
  *(reinterpret_cast<char4*>(q + (size_t)m * K + kc * 128) + lane) = out;
  if (lane == 0) d[(size_t)m * (K / 128) + kc] = scale;
}


// one-time repack: interleaved block_q1_0 (18 B blocks) -> dense bit words + fp32 scales.
// Weights are static; runs once per tensor, then every GEMM reads coalesced dense arrays.
__global__ void repack_q1_dense(const block_q1_0* __restrict__ W, unsigned* __restrict__ bits,
                                float* __restrict__ dw, long nblocks_total) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks_total) return;
  const uint16_t* u16 = reinterpret_cast<const uint16_t*>(W + b);  // [0]=d, [1..8]=bit halves
  dw[b] = __half2float(*reinterpret_cast<const __half*>(W + b));
#pragma unroll
  for (int w = 0; w < 4; ++w)
    bits[b * 4 + w] = (unsigned)u16[1 + 2 * w] | ((unsigned)u16[2 + 2 * w] << 16);
}

struct DenseQ1 { unsigned* bits; float* dw; };
// experiment-grade cache: weights static for process lifetime; freed at exit by the driver.
static DenseQ1 get_dense_q1(const void* wdata, long N, long K, cudaStream_t stream) {
  static std::unordered_map<const void*, DenseQ1> cache;
  auto it = cache.find(wdata);
  if (it != cache.end()) return it->second;
  DenseQ1 d{};
  const long nb = N * (K / 128);
  cudaMalloc(&d.bits, nb * 16);
  cudaMalloc(&d.dw, nb * sizeof(float));
  repack_q1_dense<<<(unsigned)((nb + 255) / 256), 256, 0, stream>>>((const block_q1_0*)wdata, d.bits, d.dw, nb);
  cache.emplace(wdata, d);
  return d;
}

__global__ __launch_bounds__(512) void q1_wgmma_ggml(const int8_t* __restrict__ Aq, const float* __restrict__ dA,
                                                     const unsigned* __restrict__ Wbits, const float* __restrict__ Wd,
                                                     float* __restrict__ C,
                                                     int M, int N, int K) {
  using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, Shape<Int<bM>, Int<bK>>{}));
  using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, Shape<Int<bN>, Int<bK>>{}));
  extern __shared__ __align__(128) int8_t smem[];
  int8_t* sA[2] = {smem, smem + cosize_v<SmemLayoutA>};
  int8_t* sB[2] = {smem + 2 * cosize_v<SmemLayoutA>, smem + 2 * cosize_v<SmemLayoutA> + cosize_v<SmemLayoutB>};
  float* sDa[2];
  float* sDw[2];
  {
    float* p = reinterpret_cast<float*>(smem + 2 * cosize_v<SmemLayoutA> + 2 * cosize_v<SmemLayoutB>);
    sDa[0] = p; sDa[1] = p + bM; sDw[0] = p + 2 * bM; sDw[1] = p + 2 * bM + bN;
  }
  const int mblk = blockIdx.x, nblk = blockIdx.y;
  const int wg = threadIdx.x / 128;
  const int wgm = wg / 2, wgn = wg % 2;
  TiledMMA mma = make_tiled_mma(MmaAtom{});
  auto thr = mma.get_slice(threadIdx.x % 128);
  Tensor acc_i32 = partition_fragment_C(mma, Shape<Int<64>, Int<64>>{});
  auto acc_f32 = make_fragment_like<float>(acc_i32);
  clear(acc_i32);
  clear(acc_f32);
  auto cAcc = thr.partition_C(make_identity_tensor(Shape<Int<64>, Int<64>>{}));

  const int nblocks_row = K / 128;  // q1_0 blocks per weight row

  auto load_stage = [&](int kc, int buf) {
    Tensor tA = make_tensor(make_smem_ptr(sA[buf]), SmemLayoutA{});
    Tensor tB = make_tensor(make_smem_ptr(sB[buf]), SmemLayoutB{});
    // A: cp.async (global->SMEM, no register round-trip); latency hides under the B unpack below.
    // A synchronous copy here costs ~1.8x end-to-end (exposed global latency per k-chunk).
    constexpr int A_CHUNKS = bM * (bK / 16);
    for (int i = threadIdx.x; i < A_CHUNKS; i += 512) {
      int r = i / (bK / 16), c16 = (i % (bK / 16)) * 16;
      __pipeline_memcpy_async(&tA(r, c16), Aq + (size_t)(mblk * bM + r) * K + kc * bK + c16, 16);
    }
    __pipeline_commit();
    constexpr int B_WORDS = bN * (bK / 32);
    for (int i = threadIdx.x; i < B_WORDS; i += 512) {
      int r = i / (bK / 32), w = i % (bK / 32);
      unsigned bits = Wbits[((size_t)(nblk * bN + r) * nblocks_row + kc) * 4 + w];
      unsigned out[8];
#pragma unroll
      for (int nib = 0; nib < 8; ++nib) {
        // branchless bit -> {+1,-1} byte expansion. NOT a __constant__ LUT: divergent indices
        // serialize the constant cache (one address per warp per cycle) and real weight bits are
        // high-entropy, costing ~1.7x end-to-end. Synthetic uniform test data hides this entirely.
        unsigned nb = (bits >> (nib * 4)) & 0xF;
        unsigned spread = (nb & 1u) | ((nb & 2u) << 7) | ((nb & 4u) << 14) | ((nb & 8u) << 21);
        out[nib] = __vadd4(0xFFFFFFFFu, spread << 1);  // per-byte 0xFF + 2*bit, no cross-byte carry
      }
      *reinterpret_cast<int4*>(&tB(r, w * 32)) = make_int4(out[0], out[1], out[2], out[3]);
      *reinterpret_cast<int4*>(&tB(r, w * 32 + 16)) = make_int4(out[4], out[5], out[6], out[7]);
    }
    for (int i = threadIdx.x; i < bM; i += 512) sDa[buf][i] = dA[(size_t)(mblk * bM + i) * nblocks_row + kc];
    for (int i = threadIdx.x; i < bN; i += 512) sDw[buf][i] = Wd[(size_t)(nblk * bN + i) * nblocks_row + kc];
  };

  const int nchunks = K / bK;
  load_stage(0, 0);
  for (int kc = 0; kc < nchunks; ++kc) {
    int cur = kc & 1, nxt = cur ^ 1;
    __pipeline_wait_prior(0);  // stage `cur` cp.async complete before it is published
    __syncthreads();
    using SmemLayoutH = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, Shape<Int<64>, Int<bK>>{}));
    Tensor tAh = make_tensor(make_smem_ptr(sA[cur] + wgm * (64 * bK)), SmemLayoutH{});
    Tensor tBh = make_tensor(make_smem_ptr(sB[cur] + wgn * (64 * bK)), SmemLayoutH{});
    Tensor tCsA = thr.partition_A(tAh);
    Tensor tCsB = thr.partition_B(tBh);
    warpgroup_fence_operand(acc_i32);
    warpgroup_arrive();
    gemm(mma, tCsA, tCsB, acc_i32);
    warpgroup_commit_batch();
    if (kc + 1 < nchunks) load_stage(kc + 1, nxt);
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_i32);
    CUTE_UNROLL
    for (int e = 0; e < size(acc_i32); ++e) {
      int ml = get<0>(cAcc(e)) + wgm * 64;
      int nl = get<1>(cAcc(e)) + wgn * 64;
      acc_f32(e) += float(acc_i32(e)) * (sDa[cur][ml] * sDw[cur][nl]);
    }
    clear(acc_i32);
  }
  CUTE_UNROLL
  for (int e = 0; e < size(acc_f32); ++e) {
    int m = get<0>(cAcc(e)) + mblk * bM + wgm * 64;
    int n = get<1>(cAcc(e)) + nblk * bN + wgn * 64;
    C[(size_t)m * N + n] = acc_f32(e);
  }
}


// stream-K variant: persistent work-centric loop; scaled fp32 partials atomicAdd'd into pre-zeroed C
__global__ __launch_bounds__(512) void q1_wgmma_ggml_sk(const int8_t* __restrict__ Aq, const float* __restrict__ dA,
                                                        const unsigned* __restrict__ Wbits, const float* __restrict__ Wd,
                                                     float* __restrict__ C,
                                                        int M, int N, int K, int ntn, long total, int ncta) {
  using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, Shape<Int<bM>, Int<bK>>{}));
  using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, Shape<Int<bN>, Int<bK>>{}));
  extern __shared__ __align__(128) int8_t smem[];
  int8_t* sA[2] = {smem, smem + cosize_v<SmemLayoutA>};
  int8_t* sB[2] = {smem + 2 * cosize_v<SmemLayoutA>, smem + 2 * cosize_v<SmemLayoutA> + cosize_v<SmemLayoutB>};
  float* sDa[2]; float* sDw[2];
  { float* p = reinterpret_cast<float*>(smem + 2 * cosize_v<SmemLayoutA> + 2 * cosize_v<SmemLayoutB>);
    sDa[0]=p; sDa[1]=p+bM; sDw[0]=p+2*bM; sDw[1]=p+2*bM+bN; }
  int mblk = 0, nblk = 0;
  const int wg = threadIdx.x / 128; const int wgm = wg / 2, wgn = wg % 2;
  TiledMMA mma = make_tiled_mma(MmaAtom{});
  auto thr = mma.get_slice(threadIdx.x % 128);
  Tensor acc_i32 = partition_fragment_C(mma, Shape<Int<64>, Int<64>>{});
  auto acc_f32 = make_fragment_like<float>(acc_i32);
  clear(acc_i32); clear(acc_f32);
  auto cAcc = thr.partition_C(make_identity_tensor(Shape<Int<64>, Int<64>>{}));
  const int nblocks_row = K / 128;
  auto load_stage = [&](int kc, int buf) {
    Tensor tA = make_tensor(make_smem_ptr(sA[buf]), SmemLayoutA{});
    Tensor tB = make_tensor(make_smem_ptr(sB[buf]), SmemLayoutB{});
    constexpr int A_CHUNKS = bM * (bK / 16);  // A via cp.async (see fixed-grid kernel note)
    for (int i = threadIdx.x; i < A_CHUNKS; i += 512) {
      int r = i / (bK / 16), c16 = (i % (bK / 16)) * 16;
      __pipeline_memcpy_async(&tA(r, c16), Aq + (size_t)(mblk * bM + r) * K + kc * bK + c16, 16);
    }
    __pipeline_commit();
    constexpr int B_WORDS = bN * (bK / 32);
    for (int i = threadIdx.x; i < B_WORDS; i += 512) {
      int r = i / (bK / 32), w = i % (bK / 32);
      unsigned bits = Wbits[((size_t)(nblk * bN + r) * nblocks_row + kc) * 4 + w];
      unsigned out[8];
#pragma unroll
      for (int nib = 0; nib < 8; ++nib) {
        // branchless bit -> {+1,-1} byte expansion. NOT a __constant__ LUT: divergent indices
        // serialize the constant cache (one address per warp per cycle) and real weight bits are
        // high-entropy, costing ~1.7x end-to-end. Synthetic uniform test data hides this entirely.
        unsigned nb = (bits >> (nib * 4)) & 0xF;
        unsigned spread = (nb & 1u) | ((nb & 2u) << 7) | ((nb & 4u) << 14) | ((nb & 8u) << 21);
        out[nib] = __vadd4(0xFFFFFFFFu, spread << 1);  // per-byte 0xFF + 2*bit, no cross-byte carry
      }
      *reinterpret_cast<int4*>(&tB(r, w * 32)) = make_int4(out[0], out[1], out[2], out[3]);
      *reinterpret_cast<int4*>(&tB(r, w * 32 + 16)) = make_int4(out[4], out[5], out[6], out[7]);
    }
    for (int i = threadIdx.x; i < bM; i += 512) sDa[buf][i] = dA[(size_t)(mblk * bM + i) * nblocks_row + kc];
    for (int i = threadIdx.x; i < bN; i += 512) sDw[buf][i] = Wd[(size_t)(nblk * bN + i) * nblocks_row + kc];
  };
  const int nchunks = K / 128;
  auto flush = [&](bool owned) {
    CUTE_UNROLL
    for (int e = 0; e < size(acc_f32); ++e) {
      int m = get<0>(cAcc(e)) + mblk * bM + wgm * 64;
      int n = get<1>(cAcc(e)) + nblk * bN + wgn * 64;
      if (owned) C[(size_t)m * N + n] = acc_f32(e);          // whole tile in this CTA's span
      else atomicAdd(&C[(size_t)m * N + n], acc_f32(e));      // split tile (C pre-zeroed)
    }
    clear(acc_f32);
  };
  auto tile_owned = [&](int tile, long lo, long hi) {
    long first = (long)tile * nchunks, last = first + nchunks - 1;
    return first >= lo && last < hi;
  };
  const int cta = blockIdx.x;
  long u0 = (long)cta * total / ncta, u1 = (long)(cta + 1) * total / ncta;
  int cur_tile = -1, buf = 0;
  for (long u = u0; u < u1; ++u) {
    int tile = (int)(u / nchunks), kc = (int)(u % nchunks);
    if (tile != cur_tile) {
      if (cur_tile >= 0) flush(tile_owned(cur_tile, u0, u1));
      cur_tile = tile; mblk = tile / ntn; nblk = tile % ntn; buf = 0;
      load_stage(kc, buf);
    }
    __pipeline_wait_prior(0);  // stage `buf` cp.async complete before it is published
    __syncthreads();
    {
      using SmemLayoutH = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, Shape<Int<64>, Int<bK>>{}));
      Tensor tAh = make_tensor(make_smem_ptr(sA[buf] + wgm * (64 * bK)), SmemLayoutH{});
      Tensor tBh = make_tensor(make_smem_ptr(sB[buf] + wgn * (64 * bK)), SmemLayoutH{});
      Tensor tCsA = thr.partition_A(tAh);
      Tensor tCsB = thr.partition_B(tBh);
      warpgroup_fence_operand(acc_i32);
      warpgroup_arrive();
      gemm(mma, tCsA, tCsB, acc_i32);
      warpgroup_commit_batch();
    }
    bool next_same = (u + 1 < u1) && ((u + 1) / nchunks == tile);
    if (next_same) load_stage(kc + 1, buf ^ 1);
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_i32);
    CUTE_UNROLL
    for (int e = 0; e < size(acc_i32); ++e) {
      int ml = get<0>(cAcc(e)) + wgm * 64;
      int nl = get<1>(cAcc(e)) + wgn * 64;
      acc_f32(e) += float(acc_i32(e)) * (sDa[buf][ml] * sDw[buf][nl]);
    }
    clear(acc_i32);
    if (next_same) buf ^= 1;
    __syncthreads();
  }
  if (cur_tile >= 0) flush(tile_owned(cur_tile, u0, u1));
}

}  // namespace hopper_q1
#endif  // GGML_USE_HOPPER_Q1

// returns false if the shape/arch is unsupported (caller falls through to standard MMQ)
bool ggml_cuda_mul_mat_q1_hopper(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1,
                                 ggml_tensor* dst) {
#if defined(GGML_USE_HOPPER_Q1)
  static const bool enabled = getenv("GGML_HOPPER_Q1") != nullptr;
  if (!enabled) return false;
  const int cc = ggml_cuda_info().devices[ctx.device].cc;
  const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
  if (cc < 900 ||  // 900 = Hopper; no GGML_CUDA_CC_HOPPER macro in this tree
       src0->type != GGML_TYPE_Q1_0 || src1->type != GGML_TYPE_F32 ||
      dst->type != GGML_TYPE_F32 || src1->ne[2] * src1->ne[3] != 1 || src0->ne[2] * src0->ne[3] != 1 ||
      (M % 128) || (N % 128) || (K % 128) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
    return false;
  }
  cudaStream_t stream = ctx.stream();
  hopper_q1::DenseQ1 wq = hopper_q1::get_dense_q1(src0->data, (long)N, (long)K, stream);
  ggml_cuda_pool_alloc<int8_t> act_q(ctx.pool(), (size_t)M * K);
  ggml_cuda_pool_alloc<float> act_d(ctx.pool(), (size_t)M * (K / 128));
  {
    const int ngroups = (int)(M * (K / 128));
    hopper_q1::quant_act_per128<<<(ngroups + 7) / 8, 256, 0, stream>>>((const float*)src1->data, act_q.get(),
                                                                        act_d.get(), (int)M, (int)K);
  }
  constexpr int SMEM_BYTES = 2 * (hopper_q1::bM * hopper_q1::bK) + 2 * (hopper_q1::bN * hopper_q1::bK) +
                             2 * (hopper_q1::bM + hopper_q1::bN) * (int)sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(hopper_q1::q1_wgmma_ggml, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    attr_set = true;
  }
  const int ntm = (int)(M / hopper_q1::bM), ntn = (int)(N / hopper_q1::bN);
  const int ntiles = ntm * ntn;
  static int NSM = 0;
  if (NSM == 0) cudaDeviceGetAttribute(&NSM, cudaDevAttrMultiProcessorCount, ctx.device);
  if (ntiles < 8 * NSM) {
    // starved grid -> stream-K (persistent; fp32-additive atomic flush into zeroed C)
    static bool attr_sk = false;
    if (!attr_sk) {
      cudaFuncSetAttribute(hopper_q1::q1_wgmma_ggml_sk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
      attr_sk = true;
    }
    const long total = (long)ntiles * (K / 128);
    cudaMemsetAsync(dst->data, 0, (size_t)M * N * sizeof(float), stream);
    hopper_q1::q1_wgmma_ggml_sk<<<NSM, 512, SMEM_BYTES, stream>>>(act_q.get(), act_d.get(),
        wq.bits, wq.dw, (float*)dst->data, (int)M, (int)N, (int)K, ntn, total, NSM);
  } else {
    dim3 grid(ntm, ntn);
    hopper_q1::q1_wgmma_ggml<<<grid, 512, SMEM_BYTES, stream>>>(act_q.get(), act_d.get(),
        wq.bits, wq.dw, (float*)dst->data, (int)M, (int)N, (int)K);
  }
  return true;
#else
  GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
  return false;
#endif
}
