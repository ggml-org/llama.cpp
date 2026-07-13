// test_fa — standalone correctness check for libppu_fa.so's ppu_flash_attn_fwd, independent of ggml.
// Uses a simple packed [b, s, h, d] half layout and compares the .so output against an fp32 CPU reference.
//
//   ./test_fa            # non-causal, hd128, 1x64x64, 4 heads
//   ./test_fa 1          # causal
//   ./test_fa 1 128      # causal, head_dim 128
// Build: linked against ppu_fa by ppu_so/CMakeLists.txt.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" int ppu_flash_attn_fwd(
    const void * q, const void * k, const void * v, void * o,
    int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
    long long q_batch_stride, long long q_head_stride, long long q_row_stride,
    long long k_batch_stride, long long k_head_stride, long long k_row_stride,
    long long v_batch_stride, long long v_head_stride, long long v_row_stride,
    long long o_batch_stride, long long o_head_stride, long long o_row_stride,
    float scale, float logit_softcap, int is_causal, int dtype, void * stream);

int main(int argc, char ** argv) {
    const int is_causal = argc > 1 ? atoi(argv[1]) : 0;
    const int D         = argc > 2 ? atoi(argv[2]) : 128;
    const int B = 1, SQ = 64, SK = 80, H = 8, HKV = 2;   // SQ != SK exercises bottom-right causal; H/HKV=4 exercises GQA
    const float scale = 1.0f / std::sqrt((float) D);

    // packed [b, s, h, d]; q/o use H heads and SQ rows, k/v use HKV heads and SK rows.
    auto qidx = [&](int b, int s, int h, int d) { return (size_t) ((b * SQ + s) * H   + h) * D + d; };
    auto kidx = [&](int b, int s, int h, int d) { return (size_t) ((b * SK + s) * HKV + h) * D + d; };
    const size_t nQ = (size_t) B * SQ * H * D, nK = (size_t) B * SK * HKV * D, nO = nQ;

    std::vector<float> hq(nQ), hk(nK), hv(nK), href(nO);
    auto rnd = [](int i) { return std::sin(0.1f * i) * 0.5f; };
    for (size_t i = 0; i < nQ; ++i) hq[i] = rnd((int) i);
    for (size_t i = 0; i < nK; ++i) { hk[i] = rnd((int) i + 7); hv[i] = rnd((int) i + 13); }

    // fp32 CPU reference. GQA: query head h attends to kv head h / (H/HKV). Causal is BOTTOM-RIGHT aligned.
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
        const int hk_ = h / (H / HKV);
        for (int i = 0; i < SQ; ++i) {
            std::vector<float> sc(SK, 0.f);
            float mx = -1e30f;
            const int jmax = is_causal ? (i + (SK - SQ) + 1) : SK;
            for (int j = 0; j < SK; ++j) {
                if (j >= jmax) { sc[j] = -1e30f; continue; }
                float acc = 0.f;
                for (int d = 0; d < D; ++d) acc += hq[qidx(b,i,h,d)] * hk[kidx(b,j,hk_,d)];
                sc[j] = acc * scale; mx = std::fmax(mx, sc[j]);
            }
            float sum = 0.f;
            for (int j = 0; j < SK; ++j) { sc[j] = (sc[j] <= -1e29f) ? 0.f : std::exp(sc[j] - mx); sum += sc[j]; }
            for (int d = 0; d < D; ++d) {
                float acc = 0.f;
                for (int j = 0; j < SK; ++j) acc += sc[j] * hv[kidx(b,j,hk_,d)];
                href[qidx(b,i,h,d)] = acc / sum;
            }
        }
    }

    auto to_half = [](const std::vector<float> & f) {
        std::vector<__half> h(f.size());
        for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
        return h;
    };
    auto dq = to_half(hq), dk = to_half(hk), dv = to_half(hv);
    __half *Q, *K, *V, *O;
    cudaMalloc(&Q, nQ*2); cudaMalloc(&K, nK*2); cudaMalloc(&V, nK*2); cudaMalloc(&O, nO*2);
    cudaMemcpy(Q, dq.data(), nQ*2, cudaMemcpyHostToDevice);
    cudaMemcpy(K, dk.data(), nK*2, cudaMemcpyHostToDevice);
    cudaMemcpy(V, dv.data(), nK*2, cudaMemcpyHostToDevice);

    // packed [b,s,h,d] strides (elements)
    const long long qrs = (long long) H*D,   qhs = D, qbs = (long long) SQ*H*D;
    const long long krs = (long long) HKV*D, khs = D, kbs = (long long) SK*HKV*D;
    const long long ors = (long long) H*D,   ohs = D, obs = (long long) SQ*H*D;

    const int rc = ppu_flash_attn_fwd(Q, K, V, O, B, SQ, SK, H, HKV, D,
        qbs,qhs,qrs, kbs,khs,krs, kbs,khs,krs, obs,ohs,ors,
        scale, 0.0f, is_causal, /*dtype=*/0, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    if (rc != 0) { printf("ppu_flash_attn_fwd rc=%d (unsupported)\n", rc); return 1; }

    std::vector<__half> ho(nO);
    cudaMemcpy(ho.data(), O, nO*2, cudaMemcpyDeviceToHost);
    double maxerr = 0, l2 = 0;
    for (size_t i = 0; i < nO; ++i) { double e = std::fabs(__half2float(ho[i]) - href[i]); maxerr = std::fmax(maxerr, e); l2 += e*e; }
    printf("causal=%d  d=%d  max_err=%.4g  rms=%.4g  -> %s\n", is_causal, D, maxerr, std::sqrt(l2/nO),
           maxerr < 2e-2 ? "PASS" : "FAIL");
    return maxerr < 2e-2 ? 0 : 1;
}
