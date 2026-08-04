// Cross-backend IOSurface buffer test (the data plane for lock-free
// CPU/Metal/ANE dispatch).
//
// Validates that ggml_backend_ane_iosurface_buffer_alloc produces a
// buffer whose memory is shared between the CPU (locked CVPixelBuffer
// base), Metal (lazily-wrapped MTLBuffer), and (transitively) ANE
// (raw IOSurfaceRef). The contract is "no copies": writing through the
// CPU view is observable through the Metal view, and vice versa.
//
// This is the foundation of the lock-free cross-backend dispatch
// (per the prism-engine IOSurface arena / SharedEventContract pattern,
// mapped to llama.cpp). The control plane (MTLSharedEvent) and the
// dispatch API build on top of this primitive in subsequent commits.

#include "ggml-ane.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>

namespace {

constexpr uint32_t kN = 4096;

bool cpu_write_metal_read() {
    ggml_backend_buffer_t buf = ggml_backend_ane_iosurface_buffer_alloc(kN * sizeof(float));
    if (!buf) {
        std::fprintf(stderr, "alloc failed\n");
        return false;
    }
    if (!ggml_backend_ane_iosurface_buffer_check(buf)) {
        std::fprintf(stderr, "check failed: not an ANE IOSurface buffer\n");
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buf);
    if (!base) {
        std::fprintf(stderr, "base is null\n");
        return false;
    }
    void * surface = ggml_backend_ane_iosurface_buffer_get_iosurface(buf);
    if (!surface) {
        std::fprintf(stderr, "iosurface is null\n");
        return false;
    }
    void * mtl_buf_void = ggml_backend_ane_iosurface_buffer_get_mtl_buffer(buf);
    if (!mtl_buf_void) {
        std::fprintf(stderr, "metal buffer is null\n");
        return false;
    }
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>) mtl_buf_void;
    if (mtl_buf.length < kN * sizeof(float)) {
        std::fprintf(stderr, "metal buffer too small (got %zu, expected >= %zu)\n",
                     (size_t) mtl_buf.length, (size_t) (kN * sizeof(float)));
        return false;
    }
    if (mtl_buf.contents == nullptr) {
        std::fprintf(stderr, "metal buffer contents pointer is null\n");
        return false;
    }

    // The CPU base, the IOSurface base, and the MTLBuffer contents must
    // all point to the same physical pages. Per Apple's docs, the IOSurface
    // is process-shared and the MTLBuffer (created with
    // newBufferWithBytesNoCopy) shares the same memory.
    void * isurf_base = IOSurfaceGetBaseAddress((IOSurfaceRef) surface);
    if (isurf_base != base) {
        std::fprintf(stderr, "iosurface base (%p) does not match CPU base (%p)\n",
                     isurf_base, base);
        return false;
    }
    if (mtl_buf.contents != base) {
        std::fprintf(stderr, "metal buffer contents (%p) does not match CPU base (%p)\n",
                     mtl_buf.contents, base);
        return false;
    }

    // Write a deterministic pattern through the CPU view, read it back
    // through the Metal view, and verify they match.
    auto * cpu = static_cast<float *>(base);
    auto * mtl = static_cast<float *>(mtl_buf.contents);
    for (uint32_t i = 0; i < kN; ++i) {
        cpu[i] = static_cast<float>(i) * 0.5f - 1.0f;
    }
    bool ok = true;
    for (uint32_t i = 0; i < kN; ++i) {
        if (mtl[i] != cpu[i]) {
            std::fprintf(stderr, "mismatch at %u (cpu=%.4f metal=%.4f)\n",
                         i, cpu[i], mtl[i]);
            ok = false;
            break;
        }
    }
    if (!ok) {
        return false;
    }

    // Reverse: write through Metal, read through CPU.
    for (uint32_t i = 0; i < kN; ++i) {
        mtl[i] = static_cast<float>(i) * -0.25f + 3.5f;
    }
    for (uint32_t i = 0; i < kN; ++i) {
        if (cpu[i] != static_cast<float>(i) * -0.25f + 3.5f) {
            std::fprintf(stderr, "cpu view did not see metal write at %u (cpu=%.4f)\n",
                         i, cpu[i]);
            ok = false;
            break;
        }
    }
    if (!ok) {
        return false;
    }

    // Calling get_mtl_buffer twice should return the same MTLBuffer
    // (lazy-cached).
    void * mtl_buf_2 = ggml_backend_ane_iosurface_buffer_get_mtl_buffer(buf);
    if (mtl_buf_2 != mtl_buf_void) {
        std::fprintf(stderr, "mtl_buffer not cached (got %p then %p)\n",
                     mtl_buf_void, mtl_buf_2);
        ok = false;
    }

    ggml_backend_buffer_free(buf);
    return ok;
}

} // namespace

int main() {
    std::printf("data plane: cross-backend IOSurface buffer\n");
    std::printf("  cpu_write_metal_read         ... ");
    if (cpu_write_metal_read()) {
        std::printf("OK\n");
        return 0;
    }
    std::printf("FAIL\n");
    return 1;
}
