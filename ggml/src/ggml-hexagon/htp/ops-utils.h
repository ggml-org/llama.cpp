#ifndef OPS_UTILS_H
#define OPS_UTILS_H

#include "htp-msg.h"

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static inline uint64_t htp_get_cycles() {
    uint64_t cycles = 0;
    asm volatile(" %0 = c15:14\n" : "=r"(cycles));
    return cycles;
}

static inline uint64_t htp_get_pktcnt() {
    uint64_t pktcnt;
    asm volatile(" %0 = c19:18\n" : "=r"(pktcnt));
    return pktcnt;
}

static inline int32_t htp_is_aligned(void * addr, uint32_t align) {
    return ((size_t) addr & (align - 1)) == 0;
}

static inline uint32_t htp_round_up(uint32_t n, uint32_t m) {
    return m * ((n + m - 1) / m);
}

static inline void htp_l2fetch(const void * p, uint32_t height, uint32_t width, uint32_t stride) {
    const uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    asm volatile(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

static inline int32_t htp_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;
}

static inline void htp_dump_int8_line(char * pref, const int8_t * x, int n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static inline void htp_dump_uint8_line(char * pref, const uint8_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n && p < p_end; i++) {
        p += snprintf(p, p_end - p, "%d, ", x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static inline void htp_dump_int32_line(char * pref, const int32_t * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%d, ", (int) x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static inline void htp_dump_fp16_line(char * pref, const __fp16 * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", (float) x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static inline void htp_dump_fp32_line(char * pref, const float * x, uint32_t n) {
    char str[1024], *p = str, *p_end = str + sizeof(str);
    p += snprintf(p, p_end - p, "%s: ", pref);
    for (int i = 0; i < n; i++) {
        p += snprintf(p, p_end - p, "%.6f, ", x[i]);
    }
    FARF(HIGH, "%s\n", str);
}

static inline void htp_dump_f32(char * pref, const float * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    for (; i < n0; i++) {
        htp_dump_fp32_line(pref, x + (16 * i), 16);
    }
    if (n1) {
        htp_dump_fp32_line(pref, x + (16 * i), n1);
    }
}

static inline void htp_dump_f16(char * pref, const __fp16 * x, uint32_t n) {
    uint32_t n0 = n / 16;
    uint32_t n1 = n % 16;

    uint32_t i = 0;
    for (; i < n0; i++) {
        htp_dump_fp16_line(pref, x + (16 * i), 16);
    }
    if (n1) {
        htp_dump_fp16_line(pref, x + (16 * i), n1);
    }
}

#endif /* OPS_UTILS_H */
