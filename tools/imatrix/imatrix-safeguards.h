// Crash-safety / long-run safeguards for llama-imatrix (Phase 16.9,
// 2026-08-04). Header-only so test binaries can unit-test the
// dynamic_save_freq_ladder and physmem_bytes functions without
// dragging in the full imatrix.cpp (which has llama_init and
// ggml dependencies that the test does not want).
//
// The four guards in main():
//   1. Memory precheck (refuse if model > physmem * fraction)
//   2. PID file (write <output>.pid)
//   3. Wall-time cap (SIGALRM via --max-minutes)
//   4. Dynamic save-frequency ladder (start paranoid, relax over time)
//
// All four previously lived in tools/tessera/smoke_imatrix.py.
// They moved to the binary in this commit.

#pragma once

#include <cstdint>
#include <string>
#include <sys/stat.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <io.h>
#  define getpid _getpid
#else
#  include <unistd.h>
#endif

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#elif defined(__linux__)
#  include <sys/sysinfo.h>
#endif

namespace tessera_imatrix_safeguards {

// physmem_bytes: best-effort physical memory probe. Returns 0 on
// unknown platforms (Windows / BSD) so the caller can detect and skip
// the precheck (we never want to false-positive a refusal on a host
// we cannot probe).
inline int64_t physmem_bytes() {
#if defined(__APPLE__)
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    int64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctl(mib, 2, &bytes, &len, nullptr, 0) != 0) {
        return 0;
    }
    return bytes;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) != 0) {
        return 0;
    }
    return (int64_t) si.totalram * (int64_t) si.mem_unit;
#else
    return 0;
#endif
}

inline int64_t file_size_or_zero(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return (int64_t) st.st_size;
}

// Dynamic save-frequency ladder: returns the current N-chunks-between-
// saves based on wall-time elapsed since the start of compute_imatrix.
// Starts paranoid (every 8 chunks) and relaxes as runtime stability is
// proven. The full curve is 8 -> 16 -> 32 -> 64 -> 128 over the first
// 25 minutes. The default for llama-imatrix is to call this when the
// user did not pass --save-frequency AND --no-dynamic-save is off.
inline int32_t dynamic_save_freq_ladder(double elapsed_sec) {
    if (elapsed_sec <  5.0 * 60.0) return  8;  // <5min: paranoid
    if (elapsed_sec < 10.0 * 60.0) return 16;  // 5-10min: relax
    if (elapsed_sec < 15.0 * 60.0) return 32;  // 10-15min: nominal
    if (elapsed_sec < 25.0 * 60.0) return 64;  // 15-25min: relaxed
    return 128;                                // 25min+: minimal
}

}  // namespace tessera_imatrix_safeguards
