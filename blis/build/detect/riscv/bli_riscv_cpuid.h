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

/* RISC-V autodetection code which works with native or cross-compilers.
   Compile with $CC -E and ignore all output lines starting with #.  On RISC-V
   it may return rv32i (base 32-bit integer RISC-V), rv32iv (rv32i plus vector
   extensions), rv64i (base 64-bit integer RISC-V), or rv64iv (rv64i plus
   vector extensions). On 128-bit integer RISC-V, it falls back to generic
   for now. For toolchains which do not yet support RISC-V feature-detection
   macros, it will fall back on generic, so the BLIS configure script may need
   the RISC-V configuration to be explicitly specified. */

// false if !defined(__riscv) || !defined(__riscv_xlen)
#if __riscv && __riscv_xlen == 64

#if __riscv_vector // false if !defined(__riscv_vector)
rv64iv
#else
rv64i
#endif

// false if !defined(__riscv) || !defined(__riscv_xlen) || __riscv_e32 != 0
#elif __riscv && __riscv_xlen == 32 && !__riscv_e32

#if __riscv_vector // false if !defined(__riscv_vector)
rv32iv
#else
rv32i
#endif

#else

generic  // fall back on BLIS runtime CPUID autodetection algorithm

#endif
