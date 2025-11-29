#!/bin/sh
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2018, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of copyright holder(s) nor the names
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# This script regenerates a list of symbols for use when building
# Windows-compatible DLLs. We assume that this script will be run after
# running configure as:
#
#   ./configure --enable-cblas haswell
#
# and compiling BLIS normally. (Notice that we also prune out all
# haswell/zen-related context initialization and reference kernels.)
#

libblis='lib/haswell/libblis.so'
symfile='build/libblis-symbols.def'

echo "EXPORTS"                                                  > def.exports
#nm -g ${libblis} | grep -o " D BLIS_.*"        | cut -f2- "-dD" > def.blis_const
nm -g ${libblis} | grep -o " T bli_.*"         | cut -f2- "-dT" > def.blis
nm -g ${libblis} | grep -o " T bla_.*"         | cut -f2- "-dT" > def.blis_bla
nm -g ${libblis} | grep -o " T cblas_.*"       | cut -f2- "-dT" > def.blis_cblas
nm -g ${libblis} | grep -o " T s[acdgnrst].*"  | cut -f2- "-dT" > def.blas_s
nm -g ${libblis} | grep -o " T d[acdgnrstz].*" | cut -f2- "-dT" > def.blas_d
nm -g ${libblis} | grep -o " T c[acdghrst].*"  | cut -f2- "-dT" > def.blas_c
nm -g ${libblis} | grep -o " T z[acdghrst].*"  | cut -f2- "-dT" > def.blas_z
nm -g ${libblis} | grep -o " T i[cdsz].*"      | cut -f2- "-dT" > def.blas_i

cat def.exports \
    def.blis \
    def.blis_bla \
    def.blas_s \
    def.blas_d \
    def.blas_c \
    def.blas_z \
    def.blas_i \
    def.blis_cblas \
    | cut -f2- "-d " \
    | grep -v init_haswell \
    | grep -v haswell_ref \
    | grep -v zen_ref \
    > ${symfile}

rm -f \
    def.exports \
    def.blis \
    def.blis_bla \
    def.blas_s \
    def.blas_d \
    def.blas_c \
    def.blas_z \
    def.blas_i \
    def.blis_cblas

