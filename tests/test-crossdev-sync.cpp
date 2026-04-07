// test-crossdev-sync.cpp
// Quick test: does cross-device depends_on work with a shared SYCL context?
//
// Root cause of the old failure (Feb 20): dual SYCL context — dpct context vs
// shared context = different ze_context_handle_t. L0 events invalid across contexts.
// We removed explicit context creation. Test if it's fixed.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -o test-crossdev-sync tests/test-crossdev-sync.cpp
//
// Run (MUST expose 2+ L0 GPUs):
//   ONEAPI_DEVICE_SELECTOR='level_zero:0;level_zero:1' ./test-crossdev-sync
//
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>

struct timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    timer() : t0(clock::now()) {}
    double us() const {
        return std::chrono::duration<double, std::micro>(clock::now() - t0).count();
    }
};

int main() {
    printf("=== Cross-Device SYCL Sync Test ===\n\n");

    // ─── Find 2 L0 GPU devices ────────────────────────
    auto platform_list = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;

    for (auto & p : platform_list) {
        auto backend = p.get_backend();
        if (backend != sycl::backend::ext_oneapi_level_zero) continue;
        for (auto & d : p.get_devices(sycl::info::device_type::gpu)) {
            gpus.push_back(d);
        }
    }

    if (gpus.size() < 2) {
        printf("SKIP: Need 2+ Level Zero GPU devices, found %zu\n", gpus.size());
        printf("Run with: ONEAPI_DEVICE_SELECTOR='level_zero:0;level_zero:1'\n");
        return 1;
    }

    printf("Device 0: %s\n", gpus[0].get_info<sycl::info::device::name>().c_str());
    printf("Device 1: %s\n", gpus[1].get_info<sycl::info::device::name>().c_str());

    // ─── Test 1: Shared context + IOQ + depends_on ────
    printf("\n--- Test 1: Shared context + IOQ + depends_on ---\n");
    {
        bool pass = false;
        try {
            sycl::context shared_ctx({gpus[0], gpus[1]});
            sycl::queue q0(shared_ctx, gpus[0], sycl::property::queue::in_order());
            sycl::queue q1(shared_ctx, gpus[1], sycl::property::queue::in_order());

            printf("  Shared context created OK\n");

            const int N = 1024;
            // Host-pinned buffer (accessible by both devices)
            auto * buf = sycl::malloc_host<int>(N, shared_ctx);
            memset(buf, 0, N * sizeof(int));

            // Device 0 writes pattern
            auto evt0 = q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    buf[i] = static_cast<int>(i[0]) + 42;
                });
            });

            // Device 1 depends on device 0's event, then reads
            auto * result = sycl::malloc_host<int>(1, shared_ctx);
            result[0] = 0;

            q1.submit([&](sycl::handler & h) {
                h.depends_on(evt0);  // <-- THE CRITICAL CALL
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    // Verify device 0's writes are visible
                    int expected = static_cast<int>(i[0]) + 42;
                    if (buf[i] != expected) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> aref(result[0]);
                        aref.fetch_add(1);
                    }
                });
            }).wait();

            int errors = result[0];
            pass = (errors == 0);
            printf("  depends_on(cross_device_event): %s (%d errors)\n",
                pass ? "PASS" : "FAIL", errors);

            sycl::free(buf, shared_ctx);
            sycl::free(result, shared_ctx);
        } catch (sycl::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        } catch (std::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        }
        printf("  Result: %s\n", pass ? "WORKING" : "BROKEN");
    }

    // ─── Test 2: Shared context + OOQ + depends_on ────
    printf("\n--- Test 2: Shared context + OOQ + depends_on ---\n");
    {
        bool pass = false;
        try {
            sycl::context shared_ctx({gpus[0], gpus[1]});
            sycl::queue q0(shared_ctx, gpus[0]);  // OOQ (no in_order)
            sycl::queue q1(shared_ctx, gpus[1]);  // OOQ

            const int N = 1024;
            auto * buf = sycl::malloc_host<int>(N, shared_ctx);
            memset(buf, 0, N * sizeof(int));

            auto evt0 = q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    buf[i] = static_cast<int>(i[0]) + 99;
                });
            });

            auto * result = sycl::malloc_host<int>(1, shared_ctx);
            result[0] = 0;

            q1.submit([&](sycl::handler & h) {
                h.depends_on(evt0);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    if (buf[i] != static_cast<int>(i[0]) + 99) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> aref(result[0]);
                        aref.fetch_add(1);
                    }
                });
            }).wait();

            int errors = result[0];
            pass = (errors == 0);
            printf("  OOQ depends_on(cross_device_event): %s (%d errors)\n",
                pass ? "PASS" : "FAIL", errors);

            sycl::free(buf, shared_ctx);
            sycl::free(result, shared_ctx);
        } catch (sycl::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        } catch (std::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        }
        printf("  Result: %s\n", pass ? "WORKING" : "BROKEN");
    }

    // ─── Test 3: Shared context + ext_oneapi_submit_barrier ─
    printf("\n--- Test 3: Shared context + submit_barrier ---\n");
    {
        bool pass = false;
        try {
            sycl::context shared_ctx({gpus[0], gpus[1]});
            sycl::queue q0(shared_ctx, gpus[0], sycl::property::queue::in_order());
            sycl::queue q1(shared_ctx, gpus[1], sycl::property::queue::in_order());

            const int N = 1024;
            auto * buf = sycl::malloc_host<int>(N, shared_ctx);
            memset(buf, 0, N * sizeof(int));

            auto evt0 = q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    buf[i] = static_cast<int>(i[0]) + 77;
                });
            });

            // Use barrier instead of depends_on
            q1.ext_oneapi_submit_barrier({evt0});

            auto * result = sycl::malloc_host<int>(1, shared_ctx);
            result[0] = 0;

            q1.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    if (buf[i] != static_cast<int>(i[0]) + 77) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> aref(result[0]);
                        aref.fetch_add(1);
                    }
                });
            }).wait();

            int errors = result[0];
            pass = (errors == 0);
            printf("  submit_barrier(cross_device_event): %s (%d errors)\n",
                pass ? "PASS" : "FAIL", errors);

            sycl::free(buf, shared_ctx);
            sycl::free(result, shared_ctx);
        } catch (sycl::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        } catch (std::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        }
        printf("  Result: %s\n", pass ? "WORKING" : "BROKEN");
    }

    // ─── Test 4: Separate contexts (old broken pattern) ──
    printf("\n--- Test 4: Separate contexts + depends_on (OLD BROKEN PATTERN) ---\n");
    {
        bool pass = false;
        try {
            sycl::context ctx0({gpus[0]});
            sycl::context ctx1({gpus[1]});
            sycl::queue q0(ctx0, gpus[0], sycl::property::queue::in_order());
            sycl::queue q1(ctx1, gpus[1], sycl::property::queue::in_order());

            printf("  Separate contexts created\n");

            const int N = 1024;
            // Must use malloc_host from a shared-capable allocation
            // With separate contexts, we use host malloc from ctx0
            auto * buf = sycl::malloc_host<int>(N, ctx0);
            memset(buf, 0, N * sizeof(int));

            auto evt0 = q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    buf[i] = static_cast<int>(i[0]) + 55;
                });
            });

            auto * result = sycl::malloc_host<int>(1, ctx1);
            result[0] = 0;

            q1.submit([&](sycl::handler & h) {
                h.depends_on(evt0);  // cross-context event!
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    // buf was allocated from ctx0, not ctx1
                    // This may crash or produce wrong results
                    if (buf[i] != static_cast<int>(i[0]) + 55) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> aref(result[0]);
                        aref.fetch_add(1);
                    }
                });
            }).wait();

            int errors = result[0];
            pass = (errors == 0);
            printf("  Separate-context depends_on: %s (%d errors)\n",
                pass ? "PASS" : "FAIL", errors);

            sycl::free(buf, ctx0);
            sycl::free(result, ctx1);
        } catch (sycl::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        } catch (std::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        }
        printf("  Result: %s\n", pass ? "WORKING" : "BROKEN (expected)");
    }

    // ─── Test 5: Latency measurement ──────────────────
    printf("\n--- Test 5: Cross-device depends_on latency ---\n");
    {
        try {
            sycl::context shared_ctx({gpus[0], gpus[1]});
            sycl::queue q0(shared_ctx, gpus[0], sycl::property::queue::in_order());
            sycl::queue q1(shared_ctx, gpus[1], sycl::property::queue::in_order());

            auto * buf = sycl::malloc_host<int>(1, shared_ctx);
            buf[0] = 0;

            const int ROUNDS = 100;

            // Warmup
            for (int i = 0; i < 10; i++) {
                auto e = q0.submit([&](sycl::handler & h) {
                    h.single_task([=]() { buf[0] = i; });
                });
                q1.submit([&](sycl::handler & h) {
                    h.depends_on(e);
                    h.single_task([=]() { buf[0] += 1; });
                }).wait();
            }

            timer t;
            for (int i = 0; i < ROUNDS; i++) {
                auto e = q0.submit([&](sycl::handler & h) {
                    h.single_task([=]() { buf[0] = i; });
                });
                q1.submit([&](sycl::handler & h) {
                    h.depends_on(e);
                    h.single_task([=]() { buf[0] += 1; });
                }).wait();
            }
            double total_us = t.us();
            printf("  Cross-device round-trip: %.1f us/round (%d rounds)\n",
                total_us / ROUNDS, ROUNDS);

            sycl::free(buf, shared_ctx);
        } catch (sycl::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        }
    }

    // ─── Test 6: Cross-device memcpy bandwidth ────────
    printf("\n--- Test 6: Cross-device H2D/D2H bandwidth ---\n");
    {
        try {
            sycl::context shared_ctx({gpus[0], gpus[1]});
            sycl::queue q0(shared_ctx, gpus[0], sycl::property::queue::in_order());
            sycl::queue q1(shared_ctx, gpus[1], sycl::property::queue::in_order());

            size_t sz = 4 * 1024 * 1024;  // 4 MB
            auto * dev0 = sycl::malloc_device<uint8_t>(sz, q0);
            auto * dev1 = sycl::malloc_device<uint8_t>(sz, q1);
            auto * host = sycl::malloc_host<uint8_t>(sz, shared_ctx);

            // D0 → host → D1 (staged)
            q0.memcpy(host, dev0, sz).wait();  // warmup
            timer t1;
            for (int i = 0; i < 10; i++) {
                q0.memcpy(host, dev0, sz).wait();
                q1.memcpy(dev1, host, sz).wait();
            }
            double staged_ms = t1.us() / 10000.0;
            printf("  D0→host→D1 (staged, 4MB): %.3f ms (%.2f GB/s effective)\n",
                staged_ms, sz / (staged_ms * 1e6));

            // D0 → D1 direct (peer-to-peer if supported)
            try {
                q1.memcpy(dev1, dev0, sz).wait();  // warmup — may throw
                timer t2;
                for (int i = 0; i < 10; i++) {
                    q1.memcpy(dev1, dev0, sz).wait();
                }
                double p2p_ms = t2.us() / 10000.0;
                printf("  D0→D1 (direct P2P, 4MB):  %.3f ms (%.2f GB/s)\n",
                    p2p_ms, sz / (p2p_ms * 1e6));
            } catch (...) {
                printf("  D0→D1 (direct P2P): NOT SUPPORTED\n");
            }

            sycl::free(dev0, shared_ctx);
            sycl::free(dev1, shared_ctx);
            sycl::free(host, shared_ctx);
        } catch (sycl::exception & e) {
            printf("  EXCEPTION: %s\n", e.what());
        }
    }

    printf("\n=== SUMMARY ===\n");
    printf("If Tests 1-3 PASS: cross-device sync works with shared context.\n");
    printf("  → Expert parallelism becomes viable for multi-GPU MoE.\n");
    printf("  → P4.5 can consider expert parallelism, not just layer parallelism.\n");
    printf("If Tests 1-3 FAIL: still broken, layer parallelism only.\n");

    return 0;
}
