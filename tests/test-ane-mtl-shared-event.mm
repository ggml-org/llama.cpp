// Cross-backend MTLSharedEvent test (the control plane for lock-free
// CPU/Metal/ANE dispatch).
//
// Validates the four basic operations on a shared event:
//   - create / free
//   - signal(value) followed by wait(value) is non-blocking
//   - try_wait(value) returns false before signal, true after
//   - get_value() reflects the last signal
//   - get_mtl_event() returns a non-null pointer that the caller can
//     use to encode wait/signal into a Metal command buffer (the
//     command buffer integration is tested via a follow-on commit;
//     this test exercises the CPU-side contract only).

#include "ggml-metal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>

#import <Metal/Metal.h>

namespace {

bool create_free() {
    ggml_mtl_shared_event_t event = ggml_mtl_shared_event_new();
    if (!event) {
        std::fprintf(stderr, "create failed\n");
        return false;
    }
    if (!ggml_mtl_shared_event_get_mtl_event(event)) {
        std::fprintf(stderr, "get_mtl_event returned null on a fresh event\n");
        return false;
    }
    ggml_mtl_shared_event_free(event);
    return true;
}

bool signal_then_wait() {
    ggml_mtl_shared_event_t event = ggml_mtl_shared_event_new();
    if (!event) return false;

    // Initial value is 0; try_wait(0) is true, try_wait(1) is false.
    if (!ggml_mtl_shared_event_try_wait(event, 0)) {
        std::fprintf(stderr, "try_wait(0) at start should be true\n");
        return false;
    }
    if (ggml_mtl_shared_event_try_wait(event, 1)) {
        std::fprintf(stderr, "try_wait(1) before signal should be false\n");
        return false;
    }

    ggml_mtl_shared_event_signal(event, 5);
    if (ggml_mtl_shared_event_get_value(event) != 5) {
        std::fprintf(stderr, "value should be 5 after signal\n");
        return false;
    }

    // try_wait(5) is true; try_wait(6) is false (strict less-than? no,
    // >=, so 5 is enough).
    if (!ggml_mtl_shared_event_try_wait(event, 5)) {
        std::fprintf(stderr, "try_wait(5) after signal should be true\n");
        return false;
    }
    if (ggml_mtl_shared_event_try_wait(event, 6)) {
        std::fprintf(stderr, "try_wait(6) after signal(5) should be false\n");
        return false;
    }

    // wait(5) returns immediately (already at 5).
    const auto t0 = std::chrono::steady_clock::now();
    ggml_mtl_shared_event_wait(event, 5);
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (us > 100.0) {
        std::fprintf(stderr, "wait(5) when already at 5 took %.2fus (expected < 100us)\n", us);
        return false;
    }

    ggml_mtl_shared_event_free(event);
    return true;
}

bool wait_blocks_until_signal() {
    ggml_mtl_shared_event_t event = ggml_mtl_shared_event_new();
    if (!event) return false;

    // A separate thread will signal after a delay. The main thread
    // blocks on wait(10). The wait must return only after the signal
    // lands, demonstrating the cross-thread contract.
    std::atomic<bool> signal_done{false};
    std::thread signaller([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ggml_mtl_shared_event_signal(event, 10);
        signal_done.store(true);
    });

    const auto t0 = std::chrono::steady_clock::now();
    ggml_mtl_shared_event_wait(event, 10);
    const auto t1 = std::chrono::steady_clock::now();

    if (!signal_done.load()) {
        std::fprintf(stderr, "wait returned before signaller reported done\n");
        return false;
    }
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (us < 4'000.0) {
        // The signaller sleeps 5ms. wait() should have blocked for at
        // least ~4ms (allowing a small scheduling slack).
        std::fprintf(stderr, "wait returned too early: %.2fus (signaller sleeps 5ms)\n", us);
        return false;
    }
    signaller.join();
    ggml_mtl_shared_event_free(event);
    return true;
}

bool try_wait_after_value() {
    ggml_mtl_shared_event_t event = ggml_mtl_shared_event_new();
    if (!event) return false;
    for (uint64_t v = 1; v <= 8; ++v) {
        ggml_mtl_shared_event_signal(event, v);
        if (ggml_mtl_shared_event_get_value(event) != v) {
            std::fprintf(stderr, "value should be %llu after signal\n",
                         (unsigned long long) v);
            return false;
        }
        if (!ggml_mtl_shared_event_try_wait(event, v)) {
            std::fprintf(stderr, "try_wait(%llu) after signal should be true\n",
                         (unsigned long long) v);
            return false;
        }
    }
    ggml_mtl_shared_event_free(event);
    return true;
}

} // namespace

int main() {
    bool ok = true;
    std::printf("control plane: cross-backend MTLSharedEvent\n");

    std::printf("  create_free                  ... ");
    if (create_free()) { std::printf("OK\n"); }
    else { std::printf("FAIL\n"); ok = false; }

    std::printf("  signal_then_wait              ... ");
    if (signal_then_wait()) { std::printf("OK\n"); }
    else { std::printf("FAIL\n"); ok = false; }

    std::printf("  wait_blocks_until_signal      ... ");
    if (wait_blocks_until_signal()) { std::printf("OK\n"); }
    else { std::printf("FAIL\n"); ok = false; }

    std::printf("  try_wait_after_value          ... ");
    if (try_wait_after_value()) { std::printf("OK\n"); }
    else { std::printf("FAIL\n"); ok = false; }

    return ok ? 0 : 1;
}
