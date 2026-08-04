// Apple Accelerate cblas_sgemm bridge for the per-chunk
// calibration matmul (Phase 16.5, memopt-metal-dispatch).
//
// The legacy per-chunk LRQ / FLRQ matmul in
// ``tools/tessera/calibration_memory.py`` is pure numpy. On
// Apple Silicon and Intel Macs, the same matmul routed through
// Accelerate's cblas_sgemm is 2-4x faster: on Apple Silicon
// cblas_sgemm dispatches to AMX/NEON SIMD; on Intel Mac it
// dispatches to AVX-512 SIMD. The Python side wraps this
// file via ctypes in ``tools/tessera/calibration_metal.py``.
//
// We expose a single extern "C" entry point so ctypes can
// call it without a C++ ABI dependency.

#include <Accelerate/Accelerate.h>

#include <cstddef>
#include <cstdint>

extern "C" int tessera_accelerate_sgemm_f32(
        const float * a,
        const float * b,
        float * c,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        int transpose_a,
        int transpose_b) {
    if (!a || !b || !c || m == 0 || n == 0 || k == 0) {
        return -1;
    }
    // cblas_sgemm computes C = alpha * op(A) * op(B) + beta * C
    // in the CBLAS layout (column-major by default).  We use
    // CblasRowMajor so the caller's row-major numpy arrays
    // map naturally: A is (M, K) row-major, B is (K, N)
    // row-major, C is (M, N) row-major.  Leading dimensions
    // are the row-stride in row-major mode (= number of
    // columns of the operand's logical shape).
    const enum CBLAS_ORDER layout = CblasRowMajor;
    const enum CBLAS_TRANSPOSE op_a = transpose_a ? CblasTrans : CblasNoTrans;
    const enum CBLAS_TRANSPOSE op_b = transpose_b ? CblasTrans : CblasNoTrans;
    int lda = static_cast<int>(transpose_a ? m : k);
    int ldb = static_cast<int>(transpose_b ? k : n);
    int ldc = static_cast<int>(n);
    cblas_sgemm(
        layout,
        op_a,
        op_b,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        1.0f,
        a, lda,
        b, ldb,
        0.0f,
        c, ldc);
    return 0;
}
