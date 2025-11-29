/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

#ifndef BLIS_SYSTEM_H
#define BLIS_SYSTEM_H

// NOTE: If not yet defined, we define _POSIX_C_SOURCE to make sure that
// various parts of POSIX are defined and made available.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <float.h>
#include <errno.h>
#include <ctype.h>

#ifdef _MSC_VER
  #define strncasecmp _strnicmp
  #define strcasecmp _stricmp
#else
  #include <strings.h>
#endif

// Determine the compiler (hopefully) and define conveniently named macros
// accordingly.
#if   defined(__ICC) || defined(__INTEL_COMPILER)
  #define BLIS_ICC
#elif defined(__clang__)
  #define BLIS_CLANG
#elif defined(__GNUC__)
  #define BLIS_GCC
#endif

// Determine if we are on a 64-bit or 32-bit architecture.
#if defined(_M_X64) || defined(__x86_64) || defined(__aarch64__) || \
    defined(_ARCH_PPC64) || defined(__s390x__) || defined(_LP64)
  #define BLIS_ARCH_64
#else
  #define BLIS_ARCH_32
#endif

// Determine the target operating system.
#if defined(BLIS_ENABLE_SYSTEM)
  #if defined(_WIN32) || defined(__CYGWIN__)
    #define BLIS_OS_WINDOWS 1
  #elif defined(__gnu_hurd__)
    #define BLIS_OS_GNU 1
  #elif defined(__APPLE__) || defined(__MACH__)
    #define BLIS_OS_OSX 1
  #elif defined(__ANDROID__)
    #define BLIS_OS_ANDROID 1
  #elif defined(__linux__)
    #define BLIS_OS_LINUX 1
  #elif defined(__bgq__)
    #define BLIS_OS_BGQ 1
  #elif defined(__bg__)
    #define BLIS_OS_BGP 1
  #elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
        defined(__bsdi__) || defined(__DragonFly__) || \
        defined(__FreeBSD_kernel__) || defined(__HAIKU__)
    #define BLIS_OS_BSD 1
  #elif defined(EMSCRIPTEN)
    #define BLIS_OS_EMSCRIPTEN
  #else
    #error "Cannot determine operating system"
  #endif
#else // #if defined(BLIS_DISABLE_SYSTEM)
  #define BLIS_OS_NONE
#endif

// A few changes that may be necessary in Windows environments.
#if BLIS_OS_WINDOWS

  // Include Windows header file.
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <windows.h>

  #if !defined(__clang__) && !defined(__GNUC__)
    // Undefine attribute specifiers in Windows.
    #define __attribute__(x)

    // Undefine restrict.
    #define restrict
  #endif

#endif

// time.h provides clock_gettime().
#if BLIS_OS_WINDOWS
  #include <time.h>
#elif BLIS_OS_OSX
  #include <mach/mach_time.h>
#else
  //#include <sys/time.h>

  #include <time.h>
#endif


#endif
