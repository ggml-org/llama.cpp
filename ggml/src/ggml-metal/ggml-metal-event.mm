// Cross-backend MTLSharedEvent implementation (the control plane for
// lock-free CPU/Metal/ANE dispatch).
//
// Apple MTLSharedEvent is a cross-process counter: any thread (CPU or
// Metal GPU command buffer) can increment the value, and any thread
// can wait for it to reach a target value. This is the cross-backend
// primitive that ties the data plane (ggml_backend_ane_iosurface_buffer_t)
// to a producer/consumer handshake. The dispatch layer encodes
// wait(value) and signal(value) into a Metal command buffer for fully
// on-GPU synchronization, and the CPU side uses the same event to
// gate the ANE leg (ANE itself does not consume MTLSharedEvent; the
// CPU is the sequencer for ANE-Metal handoffs).
//
// This file is .mm so it can include Metal/Metal.h. The .c-style API
// in ggml-metal.h is implemented here.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml-metal.h"

#include <atomic>
#include <cstdlib>

struct ggml_mtl_shared_event {
    id<MTLSharedEvent> mtl_event;
    // Cross-thread cached value for the CPU side. Apple's
    // [event signaledValue] is itself thread-safe and lock-free; we
    // only cache it here as a hint for the try_wait fast path. The
    // authoritative value lives in the underlying MTLSharedEvent.
    std::atomic<uint64_t> cached_value;
};

GGML_BACKEND_API ggml_mtl_shared_event_t ggml_mtl_shared_event_new(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return nullptr;
    }
    // The `supportsSharedEvents` query is on MTLDevice but is not
    // reliably exposed on every device class (notably the simulator's
    // AGX device; Apple does not document the simulation contract for
    // shared events). We treat the absence of the selector as "try
    // anyway"; if `newSharedEvent` returns nil we surface that as
    // the failure mode.
    id<MTLSharedEvent> event = nil;
    if ([device respondsToSelector:@selector(newSharedEvent)]) {
        event = [device newSharedEvent];
    }
    if (!event) {
        return nullptr;
    }
    auto * wrapper = new ggml_mtl_shared_event;
    wrapper->mtl_event = event;
    wrapper->cached_value.store(0);
    return wrapper;
}

GGML_BACKEND_API void ggml_mtl_shared_event_free(ggml_mtl_shared_event_t event) {
    if (!event) {
        return;
    }
    // The mtl_event is a bridged id<MTLSharedEvent>; CFBridgingRetain
    // was not used at construction (newSharedEvent returns +1 retain),
    // so we release it directly.
    [event->mtl_event release];
    delete event;
}

GGML_BACKEND_API void ggml_mtl_shared_event_signal(ggml_mtl_shared_event_t event, uint64_t value) {
    if (!event) {
        return;
    }
    [event->mtl_event setSignaledValue:value];
    event->cached_value.store(value, std::memory_order_release);
}

GGML_BACKEND_API void ggml_mtl_shared_event_wait(ggml_mtl_shared_event_t event, uint64_t value) {
    if (!event) {
        return;
    }
    // The single-argument `waitUntilSignaledValue:` is iOS 16.0+ /
    // macOS 13.0+. On older systems the API takes a timeout in
    // milliseconds. Use the verbose form which is portable; the
    // timeout is large (effectively forever) so the behavior matches
    // the single-arg form for our purposes.
    if ([event->mtl_event respondsToSelector:@selector(waitUntilSignaledValue:)]) {
        [event->mtl_event waitUntilSignaledValue:value];
    } else {
        // 30s timeout: long enough to be "effectively forever" for
        // any reasonable producer; the Metal docs treat anything
        // beyond a few seconds as a hang.
        constexpr uint64_t kTimeoutMs = 30ULL * 1000ULL * 1000ULL;
        [event->mtl_event waitUntilSignaledValue:value timeoutMS:kTimeoutMs];
    }
    event->cached_value.store(value, std::memory_order_release);
}

GGML_BACKEND_API bool ggml_mtl_shared_event_try_wait(ggml_mtl_shared_event_t event, uint64_t value) {
    if (!event) {
        return false;
    }
    // Fast path: cached value already meets the target.
    if (event->cached_value.load(std::memory_order_acquire) >= value) {
        return true;
    }
    // Slow path: read the authoritative value from the underlying
    // MTLSharedEvent. The MTLSharedEvent's `signaledValue` is a
    // relaxed atomic load; the comparison is correct.
    const uint64_t current = event->mtl_event.signaledValue;
    event->cached_value.store(current, std::memory_order_release);
    return current >= value;
}

GGML_BACKEND_API uint64_t ggml_mtl_shared_event_get_value(ggml_mtl_shared_event_t event) {
    if (!event) {
        return 0;
    }
    const uint64_t current = event->mtl_event.signaledValue;
    event->cached_value.store(current, std::memory_order_release);
    return current;
}

GGML_BACKEND_API void * ggml_mtl_shared_event_get_mtl_event(ggml_mtl_shared_event_t event) {
    if (!event) {
        return nullptr;
    }
    return (__bridge void *) event->mtl_event;
}

GGML_BACKEND_API void ggml_mtl_shared_event_encode_wait(
        ggml_mtl_shared_event_t event, void * cmd_buf, uint64_t value) {
    if (!event || !cmd_buf) {
        return;
    }
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) cmd_buf;
    [cb encodeWaitForEvent:event->mtl_event value:value];
}

GGML_BACKEND_API void ggml_mtl_shared_event_encode_signal(
        ggml_mtl_shared_event_t event, void * cmd_buf, uint64_t value) {
    if (!event || !cmd_buf) {
        return;
    }
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) cmd_buf;
    [cb encodeSignalEvent:event->mtl_event value:value];
}
