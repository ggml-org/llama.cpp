/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2015, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#include <stdio.h>
#include <string.h>

#define VENDOR_UNKNOWN       0
#define VENDOR_INTEL         1
#define VENDOR_AMD           2

#define CPUNAME_GENERIC      0
#define CPUNAME_PENRYN       1
#define CPUNAME_SANDYBRIDGE  2
#define CPUNAME_HASWELL      3
#define CPUNAME_KNC          4
#define CPUNAME_KNL          5
#define CPUNAME_BULLDOZER    6
#define CPUNAME_PILEDRIVER   7
#define CPUNAME_STEAMROLLER  8
#define CPUNAME_EXCAVATOR    9
#define CPUNAME_ZEN         10

static char *cpuname[] = {
  "generic",
  "penryn",
  "sandybridge",
  "haswell",
  "knc",
  "knl",
  "bulldozer",
  "piledriver",
  "steamroller",
  "excavator",
  "zen",
};

#define BITMASK(a, b, c) ((((a) >> (b)) & (c)))

static inline void cpuid(int op, int *eax, int *ebx, int *ecx, int *edx){
#if defined(__i386__) && defined(__PIC__)
  __asm__ __volatile__
    ("mov %%ebx, %%edi;"
     "cpuid;"
     "xchgl %%ebx, %%edi;"
     : "=a" (*eax), "=D" (*ebx), "=c" (*ecx), "=d" (*edx) : "a" (op) : "cc");
#else
  __asm__ __volatile__
    ("cpuid": "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx) : "a" (op) : "cc");
#endif
}

static inline int have_cpuid(void){
  int eax, ebx, ecx, edx;

  cpuid(0, &eax, &ebx, &ecx, &edx);
  return eax;
}


int get_vendor(void){
  int eax, ebx, ecx, edx;
  char vendor[13];

  cpuid(0, &eax, &ebx, &ecx, &edx);

  *(int *)(&vendor[0]) = ebx;
  *(int *)(&vendor[4]) = edx;
  *(int *)(&vendor[8]) = ecx;
  vendor[12] = (char)0;

  if (!strcmp(vendor, "GenuineIntel")) return VENDOR_INTEL;
  if (!strcmp(vendor, "AuthenticAMD")) return VENDOR_AMD;

  if ((eax == 0) || ((eax & 0x500) != 0)) return VENDOR_INTEL;

  return VENDOR_UNKNOWN;
}


static inline void xgetbv(int op, int * eax, int * edx){
  //Use binary code for xgetbv
  __asm__ __volatile__
    (".byte 0x0f, 0x01, 0xd0": "=a" (*eax), "=d" (*edx) : "c" (op) : "cc");
}

int support_avx(){
  int eax, ebx, ecx, edx;
  int ret=0;

  cpuid(1, &eax, &ebx, &ecx, &edx);
  if ((ecx & (1 << 28)) != 0 && (ecx & (1 << 27)) != 0 && (ecx & (1 << 26)) != 0){
    xgetbv(0, &eax, &edx);
    if((eax & 6) == 6){
      ret=1;  //OS support AVX
    }
  }
  return ret;
}

int support_avx512(){
  int eax, ebx, ecx, edx;
  int ret=0;

  cpuid(1, &eax, &ebx, &ecx, &edx);
  if ((ecx & (1 << 28)) != 0 && (ecx & (1 << 27)) != 0 && (ecx & (1 << 26)) != 0){
    xgetbv(0, &eax, &edx);
    if((eax & 0xE6) == 0xE6){
      ret=1;  //OS support AVX-512
    }
  }
  return ret;
}

int cpu_detect()
{
  int eax, ebx, ecx, edx;
  int vendor, family, extend_family, model, extend_model;

  if ( !have_cpuid() ) return CPUNAME_GENERIC;

  vendor = get_vendor();

  cpuid( 1, &eax, &ebx, &ecx, &edx );

  extend_family = BITMASK( eax, 20, 0xff );
  extend_model  = BITMASK( eax, 16, 0x0f );
  family        = BITMASK( eax,  8, 0x0f );
  model         = BITMASK( eax,  4, 0x0f );

  if (vendor == VENDOR_INTEL){
    model |= extend_model<<4;
    switch (family) {
    case 0x6:
      switch (model) {
        case 0x0F: //Core2
        case 0x16: //Core2
        case 0x17: //Penryn
        case 0x1D: //Penryn
        case 0x1A: //Nehalem
        case 0x1E: //Nehalem
        case 0x2E: //Nehalem
        case 0x25: //Westmere
        case 0x2C: //Westmere
        case 0x2F: //Westmere
          return CPUNAME_PENRYN;
        case 0x2A: //Sandy Bridge
        case 0x2D: //Sandy Bridge
        case 0x3A: //Ivy Bridge
        case 0x3E: //Ivy Bridge
          if(support_avx()) {
            return CPUNAME_SANDYBRIDGE;
          }else{
            return CPUNAME_GENERIC; //OS doesn't support AVX
          }
        case 0x3C: //Haswell
        case 0x3F: //Haswell
        case 0x3D: //Broadwell
        case 0x47: //Broadwell
        case 0x4F: //Broadwell
        case 0x56: //Broadwell
        case 0x4E: //Skylake
        case 0x5E: //Skylake
          if(support_avx()) {
            return CPUNAME_HASWELL;
          }else{
            return CPUNAME_GENERIC; //OS doesn't support AVX
          }
        case 0x57: //KNL
          if(support_avx512()) {
            return CPUNAME_KNL;
          }else{
            return CPUNAME_GENERIC; //OS doesn't support AVX
          }
      }
      break;
    case 0xB:
      switch (model) {
        case 0x01: //KNC
          return CPUNAME_KNC;
      }
    }
  }else if (vendor == VENDOR_AMD){
    switch (family) {
    case 0xf:
      switch (extend_family) {
      case 6:
        switch (model) {
        case 1:
          if(support_avx())
            return CPUNAME_BULLDOZER;
          else
            return CPUNAME_GENERIC; //OS don't support AVX.
        case 2:
          if(support_avx())
            return CPUNAME_PILEDRIVER;
          else
            return CPUNAME_GENERIC; //OS don't support AVX.
        case 0:
          // Steamroller. Temp use Piledriver.
          if(support_avx())
            return CPUNAME_STEAMROLLER;
          else
            return CPUNAME_GENERIC; //OS don't support AVX.
        }
      case 8:
	switch (model){
	case 1:
          if(support_avx())
	    return CPUNAME_ZEN;
          else
            return CPUNAME_REFERENCE; //OS don't support AVX.
	}
      }
      break;
    }
  }

  return CPUNAME_GENERIC;
}


int main()
{
  int cpuname_id;

  cpuname_id=cpu_detect();

  printf("%s\n", cpuname[cpuname_id]);
  return 0;
}
