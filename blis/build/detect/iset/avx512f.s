//
// Test for AVX-512f instruction set.
//
vzeroall
vmovapd  %zmm0, %zmm1
vmulpd %zmm0, %zmm0, %zmm1
vfmadd213pd 0x400(%rax,%rsi,8) {1to8}, %zmm1, %zmm2
