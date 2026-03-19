#ifndef HMX_QUANTS_H
#define HMX_QUANTS_H

// x4x2 super-block constants (must match Host-side QK_* from htp-msg.h)
#define HMX_QK_Q4x4x2  256   // elements per x4x2 logical block (8 × QK4_0)
#define HMX_QK_Q8x4x2  256   // elements per x4x2 logical block (8 × QK8_0)

// Scales per x4x2 logical block: 8 × sizeof(__fp16) = 16 bytes
#define HMX_X4X2_SCALES_PER_BLK  8
#define HMX_X4X2_DBLK_SIZE       16  // 8 * 2 bytes

#endif // HMX_QUANTS_H
