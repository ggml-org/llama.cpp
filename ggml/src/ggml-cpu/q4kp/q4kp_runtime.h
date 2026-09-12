#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int q4kp_runtime_enabled(void);
// 0 recoded tensors; 1 metadata bytes recoded in place (not extra memory);
// 2 matrix operations using new kernels; 3 extra allocation bytes (zero).
uint64_t q4kp_runtime_stat(int index);
#ifdef __cplusplus
}
#endif
