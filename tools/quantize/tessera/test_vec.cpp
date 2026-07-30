//
// test_vec.cpp
//
// Smoke test for tessera-vec.h. Exercises every entry point against
// hand-computed values and returns non-zero on any mismatch.
//

#include "tessera-vec.h"

#include <cmath>
#include <cstdio>

static int g_fail = 0;

static void check_close(const char * name, float got, float want) {
    const float tol = 1e-4f * (std::fabs(want) + 1.0f);
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %-16s got %.7g want %.7g\n", name, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-16s %.7g\n", name, (double)got);
    }
}

static void check_arr(const char * name, const float * got, const float * want, int n) {
    for (int i = 0; i < n; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s[%d]", name, i);
        check_close(buf, got[i], want[i]);
    }
}

int main() {
#if defined(__APPLE__)
    std::printf("backend: Accelerate/vDSP\n");
#elif defined(GGML_USE_OPENBLAS)
    std::printf("backend: OpenBLAS\n");
#else
    std::printf("backend: naive\n");
#endif

    const float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float b[4] = { 5.0f, 6.0f, 7.0f, 8.0f };
    const int64_t n = 4;

    // reductions
    check_close("meanv",  ts_vec_meanv(a, n),  2.5f);                 // 10/4
    check_close("measqv", ts_vec_measqv(a, n), 7.5f);                 // 30/4
    check_close("maxv",   ts_vec_maxv(a, n),   4.0f);
    check_close("minv",   ts_vec_minv(a, n),   1.0f);
    check_close("sve",    ts_vec_sve(a, n),    10.0f);
    check_close("dotpr",  ts_vec_dotpr(a, b, n), 70.0f);              // 5+12+21+32
    check_close("norm2",  ts_vec_norm2(a, n),  std::sqrt(30.0f));

    // empty-input guards (match the Python fallback semantics)
    check_close("meanv(empty)", ts_vec_meanv(a, 0), 0.0f);
    check_close("maxv(empty)",  ts_vec_maxv(a, 0),  -INFINITY);
    check_close("minv(empty)",  ts_vec_minv(a, 0),  INFINITY);
    check_close("dotpr(empty)", ts_vec_dotpr(a, b, 0), 0.0f);

    // elementwise
    float out[4];
    const float exp_vsmul[4] = { 2.0f, 4.0f, 6.0f, 8.0f };
    const float exp_vmul[4]  = { 5.0f, 12.0f, 21.0f, 32.0f };
    const float exp_vadd[4]  = { 6.0f, 8.0f, 10.0f, 12.0f };
    const float exp_vsub[4]  = { -4.0f, -4.0f, -4.0f, -4.0f };

    ts_vec_vsmul(a, 2.0f, out, n);
    check_arr("vsmul", out, exp_vsmul, 4);

    ts_vec_vmul(a, b, out, n);
    check_arr("vmul", out, exp_vmul, 4);

    ts_vec_vadd(a, b, out, n);
    check_arr("vadd", out, exp_vadd, 4);

    ts_vec_vsub(a, b, out, n);
    check_arr("vsub", out, exp_vsub, 4);

    float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float exp_scale[4] = { 3.0f, 6.0f, 9.0f, 12.0f };
    ts_vec_scale(x, 3.0f, n);
    check_arr("scale", x, exp_scale, 4);

    // mat_mul: C(M x N) = A(M x K) @ B(K x N)
    const int64_t M = 2, K = 3, N = 2;
    const float A[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };   // 2 x 3
    const float B[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }; // 3 x 2
    const float exp_mul[4] = { 58.0f, 64.0f, 139.0f, 154.0f };
    float C[4];
    ts_mat_mul(A, B, C, M, K, N);
    check_arr("mat_mul", C, exp_mul, 4);

    // mat_mul_at: C(M x N) = A^T @ B, A stored row-major as (K x M)
    const float At[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };  // 3 x 2
    const float exp_mul_at[4] = { 89.0f, 98.0f, 116.0f, 128.0f };
    ts_mat_mul_at(At, B, C, M, K, N);
    check_arr("mat_mul_at", C, exp_mul_at, 4);

    if (g_fail == 0) {
        std::printf("\nall tests passed\n");
        return 0;
    }
    std::printf("\n%d check(s) failed\n", g_fail);
    return 1;
}
