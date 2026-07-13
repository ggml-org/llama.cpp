// test_moe — standalone correctness check for libppu_moe.so's ppu_moe_grouped_gemm_bf16_nopad, independent of ggml.
//
//   out[i, :] = A[i, :] @ B[m_indices[i]]^T     A:[m,k] bf16, B:[G,n,k] bf16, out:[m,n] bf16
//
//   ./test_moe                 # G=4, rows_per_expert=128, n=256, k=128
//   ./test_moe 8 256 512 256   # G rows_per_expert n k
//
// Rows are expert-grouped and expert-ordered, which is the layout the ggml hook produces (see
// ggml_cuda_mul_mat_id_ppu_so). Compares against an fp32 CPU reference computed from the same bf16 inputs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" int ppu_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);
extern "C" int ppu_moe_row_alignment(void);

int main(int argc, char ** argv) {
    const int G   = argc > 1 ? atoi(argv[1]) : 4;
    const int RPE = argc > 2 ? atoi(argv[2]) : 128;   // rows per expert
    const int N   = argc > 3 ? atoi(argv[3]) : 256;
    const int K   = argc > 4 ? atoi(argv[4]) : 128;
    // Each expert's row segment must be padded up to ppu_moe_row_alignment() -- the same contract the ggml hook
    // implements. Pad rows carry their expert's id; their output rows are computed but never read.
    const int ALIGN = ppu_moe_row_alignment();
    const int RPE_P = (RPE + ALIGN - 1) / ALIGN * ALIGN;   // padded rows per expert
    const int M     = G * RPE_P;                            // padded total rows

    auto rnd = [](int i) { return std::sin(0.037f * i) * 0.5f; };

    std::vector<float> hA((size_t) M * K), hB((size_t) G * N * K);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = rnd((int) i);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = rnd((int) i + 11);

    // round-trip through bf16 so the CPU reference sees exactly the values the kernel sees
    auto to_bf16 = [](const std::vector<float> & f) {
        std::vector<__nv_bfloat16> b(f.size());
        for (size_t i = 0; i < f.size(); ++i) b[i] = __float2bfloat16(f[i]);
        return b;
    };
    auto bA = to_bf16(hA), bB = to_bf16(hB);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __bfloat162float(bA[i]);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = __bfloat162float(bB[i]);

    std::vector<int> mi(M);
    for (int i = 0; i < M; ++i) mi[i] = i / RPE_P;   // expert-grouped, expert-ordered, padded segments

    // real rows of expert e are [e*RPE_P, e*RPE_P + RPE); the rest is padding we must not check
    auto is_real = [&](int row) { return (row % RPE_P) < RPE; };

    // fp32 CPU reference (real rows only)
    std::vector<float> ref((size_t) M * N, 0.f);
    for (int i = 0; i < M; ++i) {
        if (!is_real(i)) continue;
        const int e = mi[i];
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int kk = 0; kk < K; ++kk) acc += hA[(size_t) i*K + kk] * hB[((size_t) e*N + j)*K + kk];
            ref[(size_t) i*N + j] = acc;
        }
    }

    void *dA, *dB, *dO; int * dMI;
    cudaMalloc(&dA, bA.size()*2); cudaMalloc(&dB, bB.size()*2);
    cudaMalloc(&dO, (size_t) M*N*2); cudaMalloc(&dMI, (size_t) M*sizeof(int));
    cudaMemcpy(dA, bA.data(), bA.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, bB.data(), bB.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dMI, mi.data(), (size_t) M*sizeof(int), cudaMemcpyHostToDevice);

    const int rc = ppu_moe_grouped_gemm_bf16_nopad(dA, dB, dO, dMI, M, N, K, G, M / G, /*stream=*/nullptr);
    if (rc != 0) { printf("ppu_moe_grouped_gemm_bf16_nopad rc=%d (unsupported / JIT failed)\n", rc); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("kernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }

    std::vector<__nv_bfloat16> ho((size_t) M*N);
    cudaMemcpy(ho.data(), dO, (size_t) M*N*2, cudaMemcpyDeviceToHost);

    // bf16 output has ~3 decimal digits; scale tolerance with the accumulation magnitude.
    double maxrel = 0, l2 = 0, refl2 = 0;
    for (size_t i = 0; i < ho.size(); ++i) {
        if (!is_real((int) (i / N))) continue;      // pad rows hold garbage by design
        const double got = __bfloat162float(ho[i]), want = ref[i];
        const double e = std::fabs(got - want);
        maxrel = std::fmax(maxrel, e / std::fmax(1.0, std::fabs(want)));
        l2 += e*e; refl2 += want*want;
    }
    const double rel_rms = std::sqrt(l2 / std::fmax(refl2, 1e-12));
    printf("G=%d rows/expert=%d(->%d) M=%d N=%d K=%d align=%d  max_rel=%.4g  rel_rms=%.4g  -> %s\n",
           G, RPE, RPE_P, M, N, K, ALIGN, maxrel, rel_rms,
           rel_rms < 1e-2 ? "PASS" : "FAIL");
    return rel_rms < 1e-2 ? 0 : 1;
}
