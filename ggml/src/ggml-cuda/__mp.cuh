#pragma once

// #define GGML_PERF_ON

static __device__ void atomicAddUint64(uint64_t *address, uint64_t val) {
    atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

#ifdef GGML_PERF_ON
#define GGML_PERF_CLOCK(t)                  std::chrono::system_clock::time_point t = std::chrono::system_clock::now()
#define GGML_PERF_CLOCK_NOW(t)              t = std::chrono::system_clock::now()
#define GGML_PERF_CLOCK_COUNT(t)            std::chrono::duration<double>(std::chrono::system_clock::now() - t).count()
#define GGML_PERF_CLOCK_COUNT_ADD(s, t)     s += std::chrono::duration<double>(std::chrono::system_clock::now() - t).count()
#define GGML_PERF_GPU_CLOCK(t)              uint64_t t = clock64()
#define GGML_PERF_GPU_CLOCK_NOW(t)          t = clock64()
#define GGML_PERF_GPU_CLOCK_COUNT(t)        clock64() - t
#define GGML_PERF_GPU_CLOCK_COUNT_ADD(s, t) s += (clock64() - t)
#else
#define GGML_PERF_CLOCK(t)
#define GGML_PERF_CLOCK_NOW(t)
#define GGML_PERF_CLOCK_COUNT(t)
#define GGML_PERF_CLOCK_COUNT_ADD(s, t)
#define GGML_PERF_GPU_CLOCK(t)
#define GGML_PERF_GPU_CLOCK_NOW(t)
#define GGML_PERF_GPU_CLOCK_COUNT(t)
#define GGML_PERF_GPU_CLOCK_COUNT_ADD(s, t)
#endif // GGML_PERF_ON


#include "common.cuh"
#include "vecdotq.cuh"
#include <cstdint>

static __device__ uint64_t __ticks_total = 0, __ticks1 = 0, __ticks2 = 0, __ticks3 = 0, __ticks4 = 0, __ticks5 = 0;
static __device__ __forceinline__ float __vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    GGML_PERF_GPU_CLOCK(tick_start);

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    GGML_PERF_GPU_CLOCK(_tick1);

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    GGML_PERF_GPU_CLOCK(_tick2);

    // const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t scales[K_SCALE_SIZE/2];
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    GGML_PERF_GPU_CLOCK(_tick3);

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }
    GGML_PERF_GPU_CLOCK(_tick4);

    float ret = vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);

    GGML_PERF_GPU_CLOCK(tick_end);

    // Stats: __ticks_total  |  __ticks1  |  __ticks2  |  __ticks3  |  __ticks4  |  __ticks5
    //          161989088    |   5698656  |   1895872  |   68142496 |  14016416  | 72235648
    //                       |    3.52%   |    1.17%   |    42.07%  |    8.65%   |  44.59%
    // ----------------------------------------------------------------------------------------
    //           62014000    |   10536672 |    568288  |    493632  |    1359488 |  49060384
    //                       |    17.00%  |    0.91%   |    0.80%   |    2.19%   |  79.11%
    // ----------------------------------------------------------------------------------------
#ifdef GGML_PERF_ON
    atomicAddUint64(&__ticks1,      _tick1   - tick_start);
    atomicAddUint64(&__ticks2,      _tick2   - _tick1);
    atomicAddUint64(&__ticks3,      _tick3   - _tick2);
    atomicAddUint64(&__ticks4,      _tick4   - _tick3);
    atomicAddUint64(&__ticks5,      tick_end - _tick4);
    atomicAddUint64(&__ticks_total, tick_end - tick_start);
    printf(">> [dotq] __ticks_total = %12llu, __ticks1 = %12llu, __ticks2 = %12llu, __ticks3 = %12llu, __ticks4 = %12llu, __ticks5 = %12llu\n",
        __ticks_total, __ticks1, __ticks2, __ticks3, __ticks4, __ticks5
    );
#endif // GGML_PERF_ON

    return ret;
}
