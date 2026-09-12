#ifndef Q125_Q4KP_WIDE_KERNEL_H
#define Q125_Q4KP_WIDE_KERNEL_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* P6 GEMV for two independent 8-row groups in 512-bit lanes. Same inputs and
 * output order as q4kp_gemv, nr == 1, n multiple 256 and nc multiple 8.
 * Full 16-row tiles share Q8 broadcasts; an 8-row tail uses frozen VNNI GEMV.
 * Caller must first check q4kp_wide_supported() and own the P6 tensor layout.
 */
int q4kp_wide_supported(void);
void q4kp_wide_gemv(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
#ifdef __cplusplus
}
#endif
#endif
