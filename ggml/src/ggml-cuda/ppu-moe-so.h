#pragma once
// Thin C ABI for the external MoE grouped-GEMM .so (wraps DeepGEMM; cutlass/nvrtc stay inside the .so, llama.cpp
// sees only these declarations -- the cuBLAS model). Build libppu_moe.so from DeepGEMM's JIT runtime + this entry.
//
// out/A/B are bf16. A = [total_rows, K] bf16 (gathered, expert-grouped and expert-ordered), B = [n_experts, N, K]
// bf16, out = [total_rows, N] bf16, m_indices[row] = expert id of compact row `row`.
// `total_rows` must already include the per-expert padding required by ppu_moe_row_alignment() below.
// `expected_m` is an advisory average-rows-per-expert hint; the current implementation ignores it.
// Returns 0 on success, non-zero if the .so has no kernel for this shape/arch (caller falls back to the inline path).
#ifdef __cplusplus
extern "C" {
#endif

// Required per-expert row alignment of the compact layout (DeepGEMM pins BLOCK_M to it and reads one expert id per
// BLOCK_M block). Each expert's row segment must start and end on a multiple of this, or the GEMM silently computes
// a straddling block against the wrong expert's weights. Typically 128.
int ppu_moe_row_alignment(void);

int ppu_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

#ifdef __cplusplus
}
#endif
