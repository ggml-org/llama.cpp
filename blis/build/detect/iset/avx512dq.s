//
// Test for AVX-512dq instruction set.
//
vzeroall
vpmullq %zmm0, %zmm0, %zmm1
vpmullw %zmm0, %zmm0, %zmm1
