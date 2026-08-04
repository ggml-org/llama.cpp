// Dispatch API test: encode_wait / encode_signal for Metal command
// buffers. Validates the cross-timeline coordination between the
// CPU and Metal GPU through a shared event, with the IOSurface
// buffer as the data plane.
//
// Test flow:
//   1. CPU writes a deterministic pattern to the IOSurface buffer.
//   2. CPU signals shared event at 0.
//   3. Metal command buffer: wait(event, 0) -> blit from IOSurface
//      to a Metal-only buffer -> signal(event, 1).
//   4. CPU commits the command buffer, waits for completion.
//   5. CPU try_wait(event, 1) must be true (Metal signaled).
//   6. CPU compares the Metal-only buffer against the IOSurface.
//      The values must match (the blit is the bridge; both views
//      are zero-copy from the same physical pages for the IOSurface
//      portion, but the Metal-only buffer is a separate allocation).
//
// This is the smallest end-to-end test of the lock-free dispatch
// primitive: CPU writes -> event signal -> Metal waits -> Metal
// reads from IOSurface -> Metal signals -> CPU waits -> CPU reads.
// No CPU-GPU synchronization other than the shared event.

#include "ggml-ane.h"
#include "ggml-metal.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>

namespace {

constexpr uint32_t kN = 4096;

bool cpu_signal_metal_wait_metal_signal() {
    ggml_mtl_shared_event_t event = ggml_mtl_shared_event_new();
    if (!event) {
        std::fprintf(stderr, "shared event creation failed\n");
        return false;
    }
    ggml_backend_buffer_t iosurface_buf = ggml_backend_ane_iosurface_buffer_alloc(kN * sizeof(float));
    if (!iosurface_buf) {
        std::fprintf(stderr, "iosurface buffer alloc failed\n");
        return false;
    }
    void * mtl_iosurface_void = ggml_backend_ane_iosurface_buffer_get_mtl_buffer(iosurface_buf);
    if (!mtl_iosurface_void) {
        std::fprintf(stderr, "iosurface as MTLBuffer failed\n");
        return false;
    }
    id<MTLBuffer> mtl_iosurface = (__bridge id<MTLBuffer>) mtl_iosurface_void;

    // The Metal-only destination buffer (separate allocation).
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "MTLCreateSystemDefaultDevice failed\n");
        return false;
    }
    id<MTLBuffer> mtl_dst = [device newBufferWithLength:kN * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!mtl_dst) {
        std::fprintf(stderr, "destination MTLBuffer alloc failed\n");
        return false;
    }

    // 1. CPU writes a pattern to the IOSurface (zero-copy via base()).
    void * base = ggml_backend_buffer_get_base(iosurface_buf);
    if (!base) {
        std::fprintf(stderr, "iosurface base is null\n");
        return false;
    }
    auto * cpu = static_cast<float *>(base);
    for (uint32_t i = 0; i < kN; ++i) {
        cpu[i] = static_cast<float>(i) * 0.125f - 4.0f;
    }

    // 2. CPU signals the event.
    ggml_mtl_shared_event_signal(event, 0);

    // 3. Build the Metal command buffer: wait -> blit -> signal.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        std::fprintf(stderr, "newCommandQueue failed\n");
        return false;
    }
    id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
    if (!cmd_buf) {
        std::fprintf(stderr, "commandBuffer failed\n");
        return false;
    }
    ggml_mtl_shared_event_encode_wait(event, (__bridge void *) cmd_buf, 0);

    id<MTLBlitCommandEncoder> blit = [cmd_buf blitCommandEncoder];
    if (!blit) {
        std::fprintf(stderr, "blitCommandEncoder failed\n");
        return false;
    }
    [blit copyFromBuffer:mtl_iosurface
            sourceOffset:0
                toBuffer:mtl_dst
        destinationOffset:0
                     size:kN * sizeof(float)];
    [blit endEncoding];
    ggml_mtl_shared_event_encode_signal(event, (__bridge void *) cmd_buf, 1);

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    // 4. CPU: the event should be at 1.
    if (ggml_mtl_shared_event_get_value(event) != 1) {
        std::fprintf(stderr, "event should be at 1 after Metal signal, got %llu\n",
                     (unsigned long long) ggml_mtl_shared_event_get_value(event));
        return false;
    }

    // 5. CPU: try_wait(1) is true; try_wait(2) is false.
    if (!ggml_mtl_shared_event_try_wait(event, 1)) {
        std::fprintf(stderr, "try_wait(1) should be true after Metal signal\n");
        return false;
    }
    if (ggml_mtl_shared_event_try_wait(event, 2)) {
        std::fprintf(stderr, "try_wait(2) should be false after Metal signal(1)\n");
        return false;
    }

    // 6. Compare the destination buffer against the source.
    auto * dst = static_cast<float *>(mtl_dst.contents);
    bool ok = true;
    for (uint32_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>(i) * 0.125f - 4.0f;
        if (dst[i] != expected) {
            std::fprintf(stderr, "mismatch at %u (expected %.4f, got %.4f)\n",
                         i, expected, dst[i]);
            ok = false;
            break;
        }
    }

    ggml_mtl_shared_event_free(event);
    ggml_backend_buffer_free(iosurface_buf);
    return ok;
}

bool metal_signal_cpu_wait() {
    // Reverse: Metal writes, signals; CPU waits, then reads.
    ggml_mtl_shared_event_t event = ggml_mtl_shared_event_new();
    if (!event) return false;

    ggml_backend_buffer_t iosurface_buf = ggml_backend_ane_iosurface_buffer_alloc(kN * sizeof(float));
    if (!iosurface_buf) return false;
    void * mtl_iosurface_void = ggml_backend_ane_iosurface_buffer_get_mtl_buffer(iosurface_buf);
    if (!mtl_iosurface_void) return false;
    id<MTLBuffer> mtl_iosurface = (__bridge id<MTLBuffer>) mtl_iosurface_void;

    // Pre-fill the IOSurface with a sentinel via the CPU view.
    void * base = ggml_backend_buffer_get_base(iosurface_buf);
    auto * cpu = static_cast<float *>(base);
    for (uint32_t i = 0; i < kN; ++i) cpu[i] = 0.0f;

    // The Metal-only source buffer.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> mtl_src = [device newBufferWithLength:kN * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    auto * src = static_cast<float *>(mtl_src.contents);
    for (uint32_t i = 0; i < kN; ++i) {
        src[i] = static_cast<float>(i) * -0.0625f + 7.5f;
    }

    // Metal command buffer: blit from source to IOSurface, then signal.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd_buf blitCommandEncoder];
    [blit copyFromBuffer:mtl_src
            sourceOffset:0
                toBuffer:mtl_iosurface
        destinationOffset:0
                     size:kN * sizeof(float)];
    [blit endEncoding];
    ggml_mtl_shared_event_encode_signal(event, (__bridge void *) cmd_buf, 1);
    [cmd_buf commit];

    // CPU waits for the event.
    ggml_mtl_shared_event_wait(event, 1);
    if (ggml_mtl_shared_event_get_value(event) != 1) {
        std::fprintf(stderr, "event should be at 1 after wait\n");
        return false;
    }

    // CPU reads the IOSurface; should match the source.
    bool ok = true;
    for (uint32_t i = 0; i < kN; ++i) {
        const float expected = static_cast<float>(i) * -0.0625f + 7.5f;
        if (cpu[i] != expected) {
            std::fprintf(stderr, "cpu view mismatch at %u (expected %.4f, got %.4f)\n",
                         i, expected, cpu[i]);
            ok = false;
            break;
        }
    }

    ggml_mtl_shared_event_free(event);
    ggml_backend_buffer_free(iosurface_buf);
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    std::printf("dispatch API: cross-timeline CPU <-> Metal via MTLSharedEvent\n");

    std::printf("  cpu_signal_metal_wait_metal_signal ... ");
    if (cpu_signal_metal_wait_metal_signal()) { std::printf("OK\n"); }
    else { std::printf("FAIL\n"); ok = false; }

    std::printf("  metal_signal_cpu_wait                ... ");
    if (metal_signal_cpu_wait()) { std::printf("OK\n"); }
    else { std::printf("FAIL\n"); ok = false; }

    return ok ? 0 : 1;
}
