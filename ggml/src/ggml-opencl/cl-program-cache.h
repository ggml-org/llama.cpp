// On-disk cache for OpenCL cl_program binaries. Lets a fresh process skip the
// expensive clBuildProgram-from-source step when a binary for the exact same
// (source, compile options, device, driver, platform) was previously saved.
//
// Activation: default on via GGML_OPENCL_KERNEL_CACHE_DIR:
//   unset / empty / "1" / "default"      : platform default cache dir
//                                          (%LOCALAPPDATA%\llama.cpp\cl-cache,
//                                          ~/Library/Caches/llama.cpp/cl-cache,
//                                          <temp dir>/llama.cpp/cl-cache elsewhere)
//   "0" / "off" / "none" / "disable(d)"  : disabled (all functions no-op)
//   any other value                      : used verbatim as the cache path
// If the chosen directory cannot be created/used, the cache silently disables
// itself for the process and falls back to source compile.
// GGML_OPENCL_KERNEL_CACHE_DEBUG=1 prints a HIT/MISS/SAVE trace (with a running
// tally) straight to stderr — visible even in tools that filter INFO/WARN logs;
// redirect stderr to record it.
//
// Cache key (SHA-256 hex):
//   sha256(source_bytes || '\x00' ||
//          compile_opts  || '\x00' ||
//          CL_DEVICE_NAME || '\x00' ||
//          CL_DRIVER_VERSION || '\x00' ||
//          CL_PLATFORM_VERSION || '\x00' ||
//          CL_PROGRAM_CACHE_FORMAT_VERSION)
//
// The key fully captures everything that can affect the produced binary,
// without needing the host source revision (a kernel source change shows up
// in source_bytes; a compile-option change shows up in compile_opts).
//
// File layout per cache entry: <cache_dir>/<sha256-hex>.clbin
//   bytes [0..7]   : magic "GGMLCLBC"
//   bytes [8..11]  : uint32_t format version (CL_PROGRAM_CACHE_FORMAT_VERSION)
//   bytes [12..15] : uint32_t reserved (0)
//   bytes [16..]   : raw cl_program binary as returned by
//                    clGetProgramInfo(CL_PROGRAM_BINARIES)
//
// A separate <sha256-hex>.clrej marker records that a flash-attention kernel
// (or a whole optional program) compiled and then failed its work-group
// check ("will fall back"). The next process skips that rebuild. The hash
// includes the same device/driver/source/options material as .clbin, plus
// the kernel name and required work-group size. An empty kernel name marks
// the whole program. A driver or source change misses both files.
//
// Concurrency: writes go to <name>.tmp.<pid> then atomic rename. On race,
// last-writer-wins. No locks.

#pragma once

#include <CL/cl.h>
#include <cstddef>
#include <string>

// Bumped manually if host-side OpenCL API usage changes in a way that
// affects compile semantics but does not show up in source_bytes /
// compile_opts (e.g. switching from clCreateProgramWithSource to
// clCompileProgram + clLinkProgram, or changing how multiple sources
// are concatenated). Most commits — including kernel changes — do NOT
// require bumping this; the source bytes already capture those.
#define CL_PROGRAM_CACHE_FORMAT_VERSION 1u

struct cl_program_cache_state {
    // Empty string means cache is disabled.
    std::string dir;
    // Concatenated device/driver/platform identity + cache format version,
    // computed once at init and folded into every key.
    std::string key_suffix;
};

cl_program_cache_state cl_program_cache_init(cl_device_id device);

cl_program cl_program_cache_try_load(
    const cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    const char *                   source,
    const std::string &            compile_opts);

void cl_program_cache_try_save(
    const cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   device,
    const char *                   source,
    const std::string &            compile_opts);

// kernel_name == nullptr or "" and required_wg == 0 mark the whole program:
// it was built and none of its kernels could be registered. A non-empty name
// marks that one kernel. Both are no-ops when the cache is disabled.
bool cl_program_cache_wg_reject_load(
    const cl_program_cache_state & state,
    const char *                   source,
    const std::string &            compile_opts,
    const char *                   kernel_name,
    size_t                         required_wg);

void cl_program_cache_wg_reject_save(
    const cl_program_cache_state & state,
    const char *                   source,
    const std::string &            compile_opts,
    const char *                   kernel_name,
    size_t                         required_wg);
