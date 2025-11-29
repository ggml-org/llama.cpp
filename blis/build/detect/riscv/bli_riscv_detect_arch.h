/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/

/* Construct a RISC-V architecture string based on available features. */

#if __riscv

#if __riscv_arch_test

#if __riscv_i
#define RISCV_I i
#else
#define RISCV_I
#endif

#if __riscv_e
#define RISCV_E e
#else
#define RISCV_E
#endif

#if __riscv_m
#define RISCV_M m
#else
#define RISCV_M
#endif

#if __riscv_a
#define RISCV_A a
#else
#define RISCV_A
#endif

#if __riscv_f
#define RISCV_F f
#else
#define RISCV_F
#endif

#if __riscv_d
#define RISCV_D d
#else
#define RISCV_D
#endif

#if __riscv_flen >= 128
#define RISCV_Q q
#else
#define RISCV_Q
#endif

#if __riscv_c
#define RISCV_C c
#else
#define RISCV_C
#endif

#if __riscv_p
#define RISCV_P p
#else
#define RISCV_P
#endif

/* FORCE_RISCV_VECTOR is a Clang workaround */
#if __riscv_v || FORCE_RISCV_VECTOR
#define RISCV_V v
#else
#define RISCV_V
#endif

/* No test currently for Zicsr, which was removed from the base ISA,
   but F implies Zicsr */
#if __riscv_f
#define RISCV_ZICSR _zicsr
#else
#define RISCV_ZICSR
#endif

/* No test currently for Zifencei, which was removed from the base ISA */
#define RISCV_ZIFENCEI

#if __riscv_zba
#define RISCV_ZBA _zba
#else
#define RISCV_ZBA
#endif

#if __riscv_zbb
#define RISCV_ZBB _zbb
#else
#define RISCV_ZBB
#endif

#if __riscv_zbc
#define RISCV_ZBC _zbc
#else
#define RISCV_ZBC
#endif

#if __riscv_zbs
#define RISCV_ZBS _zbs
#else
#define RISCV_ZBS
#endif

#if __riscv_zfh
#define RISCV_ZFH _zfh
#else
#define RISCV_ZFH
#endif

#else /* __riscv_arch_test */

/* We assume I and E are exclusive when __riscv_arch_test isn't defined */
#if __riscv_32e
#define RISCV_I
#define RISCV_E e
#else
#define RISCV_I i
#define RISCV_E
#endif

#if __riscv_mul
#define RISCV_M m
#else
#define RISCV_M
#endif

#if __riscv_atomic
#define RISCV_A a
#else
#define RISCV_A
#endif

#if __riscv_flen >= 32
#define RISCV_F f
#else
#define RISCV_F
#endif

#if __riscv_flen >= 64
#define RISCV_D d
#else
#define RISCV_D
#endif

#if __riscv_flen >= 128
#define RISCV_Q q
#else
#define RISCV_Q
#endif

#if __riscv_compressed
#define RISCV_C c
#else
#define RISCV_C
#endif

#define RISCV_P

/* FORCE_RISCV_VECTOR is a Clang workaround */
#if __riscv_vector || FORCE_RISCV_VECTOR
#define RISCV_V v
#else
#define RISCV_V
#endif

/* No test currently for Zicsr, which was removed from the base ISA, but
   F implies Zicsr */
#if __riscv_flen >= 32
#define RISCV_ZICSR _zicsr
#else
#define RISCV_ZICSR
#endif

#define RISCV_ZIFENCEI
#define RISCV_ZBA
#define RISCV_ZBB
#define RISCV_ZBC
#define RISCV_ZBS
#define RISCV_ZFH

#endif /* __riscv_arch_test */

#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)

CAT(rv, CAT(__riscv_xlen, CAT(RISCV_I, CAT(RISCV_E, CAT(RISCV_M, CAT(RISCV_A,
CAT(RISCV_F, CAT(RISCV_D, CAT(RISCV_Q, CAT(RISCV_C, CAT(RISCV_P, CAT(RISCV_V,
CAT(RISCV_ZICSR, CAT(RISCV_ZIFENCEI, CAT(RISCV_ZBA, CAT(RISCV_ZBB,
CAT(RISCV_ZBC, CAT(RISCV_ZBS, RISCV_ZFH))))))))))))))))))

#endif /* __riscv */
