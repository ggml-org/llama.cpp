// On-disk cache for OpenCL cl_program binaries. Lets a fresh process skip the
// expensive clBuildProgram-from-source step when a binary for the exact same
// (source, compile options, device, driver, platform) was previously saved.
//
// Activation: set env var GGML_OPENCL_KERNEL_CACHE_DIR to a writable directory
// path. If unset, cache is disabled and all functions are no-ops. If set but
// the directory cannot be created/used, cache silently disables itself for the
// rest of the process and the build path falls back to source compile.
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
// Concurrency: writes go to <name>.tmp.<pid> then atomic rename. On race,
// last-writer-wins. No locks.

#pragma once

// Caller is responsible for setting CL_TARGET_OPENCL_VERSION before
// including this header (typically done at the top of ggml-opencl.cpp).
// We do not set it here to avoid forcing a version on the includer.
#include <CL/cl.h>

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

// Initialise the cache. Reads GGML_OPENCL_KERNEL_CACHE_DIR. Logs (info) the
// directory and key suffix when enabled, or a one-line note when disabled.
// Safe to call multiple times; returns the same state for the same device.
cl_program_cache_state cl_program_cache_init(cl_device_id device);

// Try to load a cached binary and create a built cl_program from it. Returns
// nullptr on cache miss, on disabled cache, or on any failure (load, parse,
// clCreateProgramWithBinary, clBuildProgram). On nullptr the caller falls
// back to compiling from source.
cl_program cl_program_cache_try_load(
    const cl_program_cache_state & state,
    cl_context                     context,
    cl_device_id                   device,
    const char *                   source,
    const std::string &            compile_opts);

// Save a successfully-built cl_program's binary to the cache. Best-effort:
// failures are logged at info level and ignored. No-op if the cache is
// disabled.
void cl_program_cache_try_save(
    const cl_program_cache_state & state,
    cl_program                     program,
    cl_device_id                   device,
    const char *                   source,
    const std::string &            compile_opts);
