#ifndef Q125_Q4KP_VNNI_KERNEL_H
#define Q125_Q4KP_VNNI_KERNEL_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Same P6 layout and one-vector contract as q4kp_gemv. Caller additionally
 * requires AVX512VNNI and AVX512VL. Non-saturating integer dot instructions
 * preserve the original int32 result and every FP32 accumulation operation.
 */
int q4kp_vnni_supported(void);
void q4kp_vnni_gemv(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
#ifdef __cplusplus
}
#endif
#endif
