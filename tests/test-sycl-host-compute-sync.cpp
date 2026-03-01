// test-sycl-host-compute-sync.cpp — Minimal test for HOST_COMPUTE sync
//
// Tests whether host-pinned USM memory is correctly shared between:
//   - Level Zero GPU queue (kernel writes)
//   - OpenCL CPU queue (kernel reads/writes)
//   - Level Zero GPU queue (kernel reads back)
//
// This mimics the HOST_COMPUTE=1 data flow in llama.cpp:
//   GPU op → host-pinned buffer → CPU op → host-pinned buffer → GPU op
//
// Build:
//   icpx -fsycl -o test-sycl-host-compute-sync test-sycl-host-compute-sync.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./test-sycl-host-compute-sync
//
// MIT license — Intel Corporation 2024-2026

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>

static constexpr int N = 4096;  // Typical hidden dim for Mistral 7B

// Find a GPU queue (Level Zero preferred)
sycl::queue make_gpu_queue() {
    for (auto & p : sycl::platform::get_platforms()) {
        for (auto & d : p.get_devices(sycl::info::device_type::gpu)) {
            auto name = d.get_info<sycl::info::device::name>();
            printf("[GPU] Found: %s\n", name.c_str());
            return sycl::queue(d, sycl::property::queue::in_order{});
        }
    }
    fprintf(stderr, "ERROR: No GPU device found\n");
    exit(1);
}

// Find a CPU queue (OpenCL preferred)
sycl::queue make_cpu_queue() {
    for (auto & p : sycl::platform::get_platforms()) {
        for (auto & d : p.get_devices(sycl::info::device_type::cpu)) {
            auto name = d.get_info<sycl::info::device::name>();
            auto plat = p.get_info<sycl::info::platform::name>();
            printf("[CPU] Found: %s (%s)\n", name.c_str(), plat.c_str());
            return sycl::queue(d, sycl::property::queue::in_order{});
        }
    }
    fprintf(stderr, "ERROR: No CPU device found\n");
    exit(1);
}

bool check_values(const float * buf, int n, float expected, const char * label) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (std::fabs(buf[i] - expected) > 1e-5f) {
            if (errors < 5) {
                printf("  FAIL %s: buf[%d] = %.6f, expected %.6f\n", label, i, buf[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  PASS %s: all %d values = %.1f\n", label, n, expected);
    } else {
        printf("  FAIL %s: %d/%d values wrong\n", label, errors, n);
    }
    return errors == 0;
}

int main() {
    printf("=== HOST_COMPUTE Synchronization Test ===\n\n");

    auto gpu_q = make_gpu_queue();
    auto cpu_q = make_cpu_queue();

    // Allocate host-pinned memory on GPU context (like HOST_COMPUTE does)
    float * buf = sycl::malloc_host<float>(N, gpu_q);
    if (!buf) {
        fprintf(stderr, "ERROR: sycl::malloc_host failed\n");
        return 1;
    }
    printf("\nAllocated %d floats (%zu bytes) as host-pinned USM\n", N, N * sizeof(float));
    printf("  pointer type on GPU: %d\n", (int)sycl::get_pointer_type(buf, gpu_q.get_context()));
    printf("  pointer type on CPU: %d\n", (int)sycl::get_pointer_type(buf, cpu_q.get_context()));

    std::vector<float> host_check(N);
    bool all_pass = true;

    // =========================================================
    // Test 1: GPU write → GPU read (sanity check)
    // =========================================================
    printf("\n--- Test 1: GPU write → GPU read ---\n");
    memset(buf, 0, N * sizeof(float));

    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 1.0f;
    });
    gpu_q.wait();

    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 1.0f, "GPU→GPU");

    // =========================================================
    // Test 2: GPU write → CPU read (like GPU→CPU boundary)
    // =========================================================
    printf("\n--- Test 2: GPU write → wait → CPU read ---\n");
    memset(buf, 0, N * sizeof(float));

    // GPU writes 2.0
    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 2.0f;
    });
    gpu_q.wait();  // boundary sync (like graph_compute_impl GPU→CPU)

    // CPU reads — should see 2.0 (verified via host memcpy below)
    cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        // Touch buffer to force CPU cache line fetch — value checked below
        (void)buf[i];
    });
    cpu_q.wait();

    // Verify via host memcpy
    memcpy(host_check.data(), buf, N * sizeof(float));
    all_pass &= check_values(host_check.data(), N, 2.0f, "GPU→CPU");

    // =========================================================
    // Test 3: CPU write → GPU read (like CPU→GPU boundary)
    // This is the suspected failure mode!
    // =========================================================
    printf("\n--- Test 3: CPU write → wait → GPU read ---\n");
    memset(buf, 0, N * sizeof(float));

    // CPU writes 3.0
    cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 3.0f;
    });
    cpu_q.wait();  // boundary sync (like cpu_staging_drain)

    // GPU reads — should see 3.0
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 3.0f, "CPU→GPU");

    // =========================================================
    // Test 4: Full chain: GPU → CPU → GPU (the HOST_COMPUTE flow)
    // =========================================================
    printf("\n--- Test 4: GPU write → CPU modify → GPU read (full chain) ---\n");
    memset(buf, 0, N * sizeof(float));

    // Step 1: GPU writes 4.0
    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 4.0f;
    });
    gpu_q.wait();  // GPU→CPU boundary

    // Step 2: CPU adds 1.0 (result should be 5.0)
    cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = buf[i] + 1.0f;
    });
    cpu_q.wait();  // CPU→GPU boundary

    // Step 3: GPU reads — should see 5.0
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 5.0f, "GPU→CPU→GPU");

    // =========================================================
    // Test 5: host_task on GPU queue → GPU read
    // (MUL_MAT uses host_task, not parallel_for on cpu_q)
    // =========================================================
    printf("\n--- Test 5: host_task(gpu_q) write → GPU read ---\n");
    memset(buf, 0, N * sizeof(float));

    // host_task writes 6.0 on GPU queue
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) {
                buf[i] = 6.0f;
            }
        });
    });
    // Next GPU kernel on same in-order queue should see writes
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 6.0f, "host_task→GPU");

    // =========================================================
    // Test 6: Mixed: host_task(gpu_q) → parallel_for(cpu_q) → GPU read
    // This is the exact pattern: MUL_MAT(host_task) → RMS_NORM(cpu_q) → GPU island
    // =========================================================
    printf("\n--- Test 6: host_task(gpu_q) → parallel_for(cpu_q) → GPU read ---\n");
    memset(buf, 0, N * sizeof(float));

    // host_task writes 7.0 (like cpu_mul_mat)
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) {
                buf[i] = 7.0f;
            }
        });
    });
    gpu_q.wait();  // boundary sync

    // CPU parallel_for adds 1.0 (like cpu_rms_norm) → result should be 8.0
    cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = buf[i] + 1.0f;
    });
    cpu_q.wait();  // CPU→GPU boundary

    // GPU reads — should see 8.0
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 8.0f, "host_task→cpu_q→GPU");

    // =========================================================
    // Test 7: NO explicit wait between cpu_q and gpu_q
    // This tests what happens WITHOUT proper boundary sync
    // =========================================================
    printf("\n--- Test 7: cpu_q write → GPU read WITHOUT wait (should fail?) ---\n");
    memset(buf, 0, N * sizeof(float));

    // CPU writes 9.0
    cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 9.0f;
    });
    // NO cpu_q.wait() here!

    // GPU reads immediately — may or may not see 9.0
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    bool test7 = check_values(host_check.data(), N, 9.0f, "cpu_q→GPU(no wait)");
    if (!test7) {
        printf("  ^ Expected: this failure confirms cross-queue sync is needed\n");
    } else {
        printf("  ^ Interesting: GPU saw CPU writes even without explicit sync\n");
    }

    // =========================================================
    // Test 8: cpu_q allocates its OWN host memory (different context)
    // =========================================================
    printf("\n--- Test 8: malloc_host on CPU context → GPU read ---\n");
    float * cpu_buf = nullptr;
    try {
        cpu_buf = sycl::malloc_host<float>(N, cpu_q);
    } catch (...) {
        printf("  SKIP: sycl::malloc_host on CPU context failed\n");
    }
    if (cpu_buf) {
        memset(cpu_buf, 0, N * sizeof(float));
        cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            cpu_buf[i] = 10.0f;
        });
        cpu_q.wait();

        // Try GPU read of CPU-context host memory
        try {
            gpu_q.memcpy(host_check.data(), cpu_buf, N * sizeof(float)).wait();
            all_pass &= check_values(host_check.data(), N, 10.0f, "cpu_malloc→GPU");
        } catch (const sycl::exception & e) {
            printf("  FAIL: GPU can't read CPU-context USM: %s\n", e.what());
        }
        sycl::free(cpu_buf, cpu_q);
    }

    // =========================================================
    // Test 9: ALL ops via host_task on gpu_q (no cpu_q at all!)
    // User question: does avoiding the SYCL CPU device eliminate issues?
    // =========================================================
    printf("\n--- Test 9: All ops via host_task on gpu_q (no cpu_q) ---\n");
    memset(buf, 0, N * sizeof(float));

    // Step 1: GPU kernel writes 11.0
    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 11.0f;
    });
    // Step 2: host_task "MUL_MAT" multiplies by 2 (→ 22.0)
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] *= 2.0f;
        });
    });
    // Step 3: host_task "RMS_NORM" adds 1 (→ 23.0)
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] += 1.0f;
        });
    });
    // Step 4: GPU reads — should see 23.0
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 23.0f, "all-host_task(gpu_q)");

    // =========================================================
    // Test 10: 4-op chain alternating gpu_q/cpu_q WITH event deps
    // Mimics: MUL_MAT(host_task,gpu_q) → RMS_NORM(cpu_q) → SILU(cpu_q) → GPU island
    // =========================================================
    printf("\n--- Test 10: 4-op chain alternating gpu_q/cpu_q with event deps ---\n");
    memset(buf, 0, N * sizeof(float));

    // GPU writes initial value
    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 10.0f;
    });
    gpu_q.wait();  // GPU→CPU boundary

    // Op 1: host_task(gpu_q) — like cpu_mul_mat
    sycl::event ev1 = gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] *= 2.0f;  // → 20.0
        });
    });

    // Op 2: parallel_for(cpu_q) — like cpu_rms_norm
    // Must wait for ev1 (cross-queue dep: gpu_q → cpu_q)
    ev1.wait();  // eagerly wait (cross-runtime events can deadlock)
    sycl::event ev2 = cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] += 3.0f;  // → 23.0
    });

    // Op 3: parallel_for(cpu_q) — like cpu_silu (same queue, in-order)
    sycl::event ev3 = cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] += 7.0f;  // → 30.0
    });

    // CPU→GPU boundary
    ev3.wait();

    // GPU reads
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 30.0f, "4-op-chain(events)");

    // =========================================================
    // Test 11: Same 4-op chain but ALL via host_task on gpu_q
    // (user's suggestion: avoid cpu_q entirely)
    // =========================================================
    printf("\n--- Test 11: Same 4-op chain ALL via host_task (no cpu_q) ---\n");
    memset(buf, 0, N * sizeof(float));

    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 10.0f;
    });
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] *= 2.0f;  // → 20.0
        });
    });
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] += 3.0f;  // → 23.0
        });
    });
    gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] += 7.0f;  // → 30.0
        });
    });
    gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
    all_pass &= check_values(host_check.data(), N, 30.0f, "all-host_task-chain");

    // =========================================================
    // Test 12: 4-op chain with event dep chain (NO eager wait)
    // Tests if SYCL event dependencies work across L0↔OpenCL
    // =========================================================
    printf("\n--- Test 12: Cross-queue deps via depends_on (no eager wait) ---\n");
    memset(buf, 0, N * sizeof(float));

    gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        buf[i] = 10.0f;
    });
    gpu_q.wait();

    sycl::event d1 = gpu_q.submit([&](sycl::handler & h) {
        h.host_task([=]() {
            for (int i = 0; i < N; i++) buf[i] *= 2.0f;
        });
    });

    // Try depends_on with cross-runtime event (L0 event → OpenCL queue)
    sycl::event d2;
    bool depends_on_worked = true;
    try {
        d2 = cpu_q.submit([&](sycl::handler & h) {
            h.depends_on(d1);  // Cross-runtime dependency!
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                buf[i] += 3.0f;
            });
        });
        cpu_q.wait();
    } catch (const sycl::exception & e) {
        printf("  SKIP: depends_on cross-runtime failed: %s\n", e.what());
        depends_on_worked = false;
    }

    if (depends_on_worked) {
        gpu_q.memcpy(host_check.data(), buf, N * sizeof(float)).wait();
        bool test12 = check_values(host_check.data(), N, 23.0f, "cross-queue-depends_on");
        if (!test12) {
            printf("  ^ Cross-runtime depends_on doesn't ensure visibility!\n");
        }
    }

    // =========================================================
    // Test 13: Larger buffer (1M floats) to stress PCIe coherency
    // =========================================================
    printf("\n--- Test 13: Large buffer (1M floats, 4MB) stress test ---\n");
    constexpr int BIG = 1024 * 1024;
    float * big_buf = sycl::malloc_host<float>(BIG, gpu_q);
    if (big_buf) {
        memset(big_buf, 0, BIG * sizeof(float));
        std::vector<float> big_check(BIG);

        // GPU writes
        gpu_q.parallel_for(sycl::range<1>(BIG), [=](sycl::id<1> i) {
            big_buf[i] = 5.0f;
        });
        gpu_q.wait();

        // host_task modifies
        gpu_q.submit([&](sycl::handler & h) {
            h.host_task([=]() {
                for (int i = 0; i < BIG; i++) big_buf[i] += 3.0f;
            });
        });
        gpu_q.wait();

        // CPU parallel_for modifies
        cpu_q.parallel_for(sycl::range<1>(BIG), [=](sycl::id<1> i) {
            big_buf[i] += 2.0f;
        });
        cpu_q.wait();

        // GPU reads back
        gpu_q.memcpy(big_check.data(), big_buf, BIG * sizeof(float)).wait();
        all_pass &= check_values(big_check.data(), BIG, 10.0f, "1M-float-stress");
        sycl::free(big_buf, gpu_q);
    } else {
        printf("  SKIP: failed to allocate 4MB host-pinned buffer\n");
    }

    // =========================================================
    // Test 14: malloc_shared compute buffer (auto-migrating)
    // User question: can we use shared USM that works from both
    // VRAM and host, with runtime managing placement?
    // =========================================================
    printf("\n--- Test 14: malloc_shared compute buffer (auto-migrating) ---\n");
    float * shared_buf = sycl::malloc_shared<float>(N, gpu_q);
    if (shared_buf) {
        printf("  shared_buf pointer type on GPU: %d\n",
               (int)sycl::get_pointer_type(shared_buf, gpu_q.get_context()));
        printf("  shared_buf pointer type on CPU: %d\n",
               (int)sycl::get_pointer_type(shared_buf, cpu_q.get_context()));
        memset(shared_buf, 0, N * sizeof(float));

        // GPU writes → host_task reads/modifies → cpu_q reads/modifies → GPU reads
        gpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            shared_buf[i] = 5.0f;
        });
        gpu_q.wait();

        gpu_q.submit([&](sycl::handler & h) {
            h.host_task([=]() {
                for (int i = 0; i < N; i++) shared_buf[i] += 3.0f;  // → 8.0
            });
        });
        gpu_q.wait();

        cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            shared_buf[i] += 2.0f;  // → 10.0
        });
        cpu_q.wait();

        gpu_q.memcpy(host_check.data(), shared_buf, N * sizeof(float)).wait();
        all_pass &= check_values(host_check.data(), N, 10.0f, "malloc_shared");
        sycl::free(shared_buf, gpu_q);
    } else {
        printf("  SKIP: sycl::malloc_shared failed\n");
    }

    // =========================================================
    // Test 15: Performance comparison — host_task vs cpu_q vs direct
    // 1000 iterations of add-1.0 on 4096 floats
    // =========================================================
    printf("\n--- Test 15: Performance comparison (1000 iters, %d floats) ---\n", N);
    constexpr int ITERS = 1000;
    float * perf_buf = sycl::malloc_host<float>(N, gpu_q);
    if (perf_buf) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Approach A: All host_task on gpu_q (user's suggestion)
        memset(perf_buf, 0, N * sizeof(float));
        gpu_q.wait();
        t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERS; iter++) {
            gpu_q.submit([&](sycl::handler & h) {
                h.host_task([=]() {
                    for (int i = 0; i < N; i++) perf_buf[i] += 1.0f;
                });
            });
        }
        gpu_q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double host_task_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        check_values(perf_buf, N, (float)ITERS, "host_task-perf");
        printf("  host_task(gpu_q)  : %.2f ms (%.1f µs/iter)\n", host_task_ms, host_task_ms * 1000 / ITERS);

        // Approach B: parallel_for on cpu_q (current approach for some ops)
        memset(perf_buf, 0, N * sizeof(float));
        cpu_q.wait();
        t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERS; iter++) {
            cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                perf_buf[i] += 1.0f;
            });
        }
        cpu_q.wait();
        t1 = std::chrono::high_resolution_clock::now();
        double cpu_q_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        check_values(perf_buf, N, (float)ITERS, "cpu_q-perf");
        printf("  parallel_for(cpu_q): %.2f ms (%.1f µs/iter)\n", cpu_q_ms, cpu_q_ms * 1000 / ITERS);

        // Approach C: Direct host (no SYCL, just for loop)
        memset(perf_buf, 0, N * sizeof(float));
        t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERS; iter++) {
            for (int i = 0; i < N; i++) perf_buf[i] += 1.0f;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double direct_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        check_values(perf_buf, N, (float)ITERS, "direct-perf");
        printf("  direct host loop  : %.2f ms (%.1f µs/iter)\n", direct_ms, direct_ms * 1000 / ITERS);

        // Approach D: Interleaved host_task(gpu_q) → cpu_q with event chain
        memset(perf_buf, 0, N * sizeof(float));
        gpu_q.wait();
        t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERS; iter++) {
            sycl::event ev = gpu_q.submit([&](sycl::handler & h) {
                h.host_task([=]() {
                    for (int i = 0; i < N; i++) perf_buf[i] += 0.5f;
                });
            });
            ev.wait();  // cross-queue boundary
            cpu_q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                perf_buf[i] += 0.5f;
            });
            cpu_q.wait();  // back to gpu_q boundary
        }
        t1 = std::chrono::high_resolution_clock::now();
        double interleaved_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        check_values(perf_buf, N, (float)ITERS, "interleaved-perf");
        printf("  interleaved(both) : %.2f ms (%.1f µs/iter)\n", interleaved_ms, interleaved_ms * 1000 / ITERS);

        sycl::free(perf_buf, gpu_q);
    }

    // Cleanup
    sycl::free(buf, gpu_q);

    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
