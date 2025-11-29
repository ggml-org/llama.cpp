//
// Test for AVX instruction set.
//
vzeroall
vmovapd  %ymm0, %ymm1
vmulpd   %ymm0, %ymm0, %ymm1
