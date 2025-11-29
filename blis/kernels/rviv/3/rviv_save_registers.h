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

// 128-bit RISC-V is assumed to support the __riscv_xlen test macro
#if __riscv_xlen == 128  // false if !defined(__riscv_xlen)

	addi sp, sp, -128
	sq s7, 112(sp)
	sq s6,  96(sp)
	sq s5,  80(sp)
	sq s4,  64(sp)
	sq s3,  48(sp)
	sq s2,  32(sp)
	sq s1,  16(sp)
	sq s0,   0(sp)

// 64-bit RISC-V can be indicated by either __riscv_xlen == 64 or
// RISCV_SIZE == 64, to support toolchains which do not currently
// support __riscv_xlen. If a macro is undefined, it is considered 0.
#elif __riscv_xlen == 64 || RISCV_SIZE == 64

	addi sp, sp, -64
	sd s7, 56(sp)
	sd s6, 48(sp)
	sd s5, 40(sp)
	sd s4, 32(sp)
	sd s3, 24(sp)
	sd s2, 16(sp)
	sd s1,  8(sp)
	sd s0,  0(sp)

#else
// else 32-bit RISC-V is assumed

	addi sp, sp, -32
	sw s7, 28(sp)
	sw s6, 24(sp)
	sw s5, 20(sp)
	sw s4, 16(sp)
	sw s3, 12(sp)
	sw s2,  8(sp)
	sw s1,  4(sp)
	sw s0,  0(sp)

#endif
