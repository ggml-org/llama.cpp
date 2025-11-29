/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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

#define SVE512_IN_REG_TRANSPOSE_d8x8_PREPARE(XTMP,PT,P2C,P4C,P6C,PTFTF,P4,P6) \
  "ptrue   " #PT".d \n\t" \
  "mov     " #XTMP", #2 \n\t" \
  "whilelo " #P2C".d, xzr, " #XTMP" \n\t" \
  "mov     " #XTMP", #4 \n\t" \
  "whilelo " #P4".d, xzr, " #XTMP" \n\t" \
  "mov     " #XTMP", #6 \n\t" \
  "whilelo " #P6".d, xzr, " #XTMP" \n\t" \
 \
  "eor     " #PTFTF".b, " #PT"/z, " #P6".b, " #P4".b \n\t" /***** o o | o */ \
  "orr     " #PTFTF".b, " #PT"/z, " #PTFTF".b, " #P2C".b \n\t" /* | o | o */ \
 \
  "not     " #P2C".b, " #PT"/z, " #P2C".b \n\t" \
  "not     " #P4C".b, " #PT"/z, " #P4".b \n\t" \
  "not     " #P6C".b, " #PT"/z, " #P6".b \n\t" \

#define SVE512_IN_REG_TRANSPOSE_d8x8(DST0,DST1,DST2,DST3,DST4,DST5,DST6,DST7,SRC0,SRC1,SRC2,SRC3,SRC4,SRC5,SRC6,SRC7,PT,P2C,P4C,P6C,PTFTF,P4,P6) \
  "trn1    " #DST0".d, " #SRC0".d, " #SRC1".d \n\t" \
  "trn2    " #DST1".d, " #SRC0".d, " #SRC1".d \n\t" \
  "trn1    " #DST2".d, " #SRC2".d, " #SRC3".d \n\t" \
  "trn2    " #DST3".d, " #SRC2".d, " #SRC3".d \n\t" \
  "trn1    " #DST4".d, " #SRC4".d, " #SRC5".d \n\t" \
  "trn2    " #DST5".d, " #SRC4".d, " #SRC5".d \n\t" \
  "trn1    " #DST6".d, " #SRC6".d, " #SRC7".d \n\t" \
  "trn2    " #DST7".d, " #SRC6".d, " #SRC7".d \n\t" \
 \
  "compact " #SRC0".d, " #P2C", " #DST0".d \n\t" \
  "compact " #SRC2".d, " #P2C", " #DST1".d \n\t" \
  "ext     " #SRC1".b, " #SRC1".b, " #DST2".b, #48 \n\t" \
  "ext     " #SRC3".b, " #SRC3".b, " #DST3".b, #48 \n\t" \
  "compact " #SRC4".d, " #P2C", " #DST4".d \n\t" \
  "compact " #SRC6".d, " #P2C", " #DST5".d \n\t" \
  "ext     " #SRC5".b, " #SRC5".b, " #DST6".b, #48 \n\t" \
  "ext     " #SRC7".b, " #SRC7".b, " #DST7".b, #48 \n\t" \
 \
  "sel     " #DST0".d, " #PTFTF", " #DST0".d, " #SRC1".d \n\t" \
  "sel     " #DST2".d, " #PTFTF", " #SRC0".d, " #DST2".d \n\t" \
  "sel     " #DST1".d, " #PTFTF", " #DST1".d, " #SRC3".d \n\t" \
  "sel     " #DST3".d, " #PTFTF", " #SRC2".d, " #DST3".d \n\t" \
  "sel     " #DST4".d, " #PTFTF", " #DST4".d, " #SRC5".d \n\t" \
  "sel     " #DST6".d, " #PTFTF", " #SRC4".d, " #DST6".d \n\t" \
  "sel     " #DST5".d, " #PTFTF", " #DST5".d, " #SRC7".d \n\t" \
  "sel     " #DST7".d, " #PTFTF", " #SRC6".d, " #DST7".d \n\t" \
 \
  "compact " #SRC0".d, " #P4C", " #DST0".d \n\t" \
  "compact " #SRC1".d, " #P4C", " #DST1".d \n\t" \
  "compact " #SRC2".d, " #P4C", " #DST2".d \n\t" \
  "compact " #SRC3".d, " #P4C", " #DST3".d \n\t" \
  "ext     " #SRC4".b, " #SRC4".b, " #DST4".b, #32 \n\t" \
  "ext     " #SRC5".b, " #SRC5".b, " #DST5".b, #32 \n\t" \
  "ext     " #SRC6".b, " #SRC6".b, " #DST6".b, #32 \n\t" \
  "ext     " #SRC7".b, " #SRC7".b, " #DST7".b, #32 \n\t" \
 \
  "sel     " #DST0".d, " #P4", " #DST0".d, " #SRC4".d \n\t" \
  "sel     " #DST1".d, " #P4", " #DST1".d, " #SRC5".d \n\t" \
  "sel     " #DST2".d, " #P4", " #DST2".d, " #SRC6".d \n\t" \
  "sel     " #DST3".d, " #P4", " #DST3".d, " #SRC7".d \n\t" \
  "sel     " #DST4".d, " #P4", " #SRC0".d, " #DST4".d \n\t" \
  "sel     " #DST5".d, " #P4", " #SRC1".d, " #DST5".d \n\t" \
  "sel     " #DST6".d, " #P4", " #SRC2".d, " #DST6".d \n\t" \
  "sel     " #DST7".d, " #P4", " #SRC3".d, " #DST7".d \n\t"

