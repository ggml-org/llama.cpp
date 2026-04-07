// test-expert-routing-roundtrip.cpp
// Validates P4.5: Full MoE expert routing across 2 GPUs
//
// Pattern: GPU0 gating → route tokens → GPU1 expert compute → merge back on GPU0
// This tests the ACTUAL data flow, not just sync primitives.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -o test-expert-routing-roundtrip tests/test-expert-routing-roundtrip.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR='level_zero:0;level_zero:1' ./test-expert-routing-roundtrip
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
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Multi-GPU Expert Routing Round-Trip Test (P4.5 Gate)   ║\n");
    printf("║  GPU0 gating → GPU1 expert → merge back on GPU0        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ─── Find 2 L0 GPUs ──────────────────────────────
    std::vector<sycl::device> gpus;
    for (auto & p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (auto & d : p.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(d);
    }

    if (gpus.size() < 2) {
        printf("SKIP: Need 2 L0 GPUs, found %zu\n", gpus.size());
        return 1;
    }

    sycl::context ctx({gpus[0], gpus[1]});
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order());
    sycl::queue q1(ctx, gpus[1], sycl::property::queue::in_order());

    printf("GPU0: %s\n", gpus[0].get_info<sycl::info::device::name>().c_str());
    printf("GPU1: %s\n", gpus[1].get_info<sycl::info::device::name>().c_str());

    // ─── Setup: Simulate MoE with 8 experts ──────────
    // 4 experts on GPU0, 4 on GPU1
    // Each expert "weight" is a small buffer (64 KB — simulates a projection)
    // Tokens: batch of 8, each routed to 2 experts (top-2)

    const int N_EXPERTS = 8;
    const int N_TOKENS = 8;
    const int TOP_K = 2;
    const int HIDDEN_DIM = 512;  // Elements per token
    const int EXPERT_SIZE = HIDDEN_DIM * sizeof(float);  // 2 KB per expert "weight"

    // Expert weights: experts 0-3 on GPU0, experts 4-7 on GPU1
    float * expert_w_gpu0[4], * expert_w_gpu1[4];
    for (int i = 0; i < 4; i++) {
        expert_w_gpu0[i] = sycl::malloc_device<float>(HIDDEN_DIM, q0);
        expert_w_gpu1[i] = sycl::malloc_device<float>(HIDDEN_DIM, q1);
        // Init weights: expert value = expert_id + 1
        float val0 = (float)(i + 1);
        float val1 = (float)(i + 5);
        q0.submit([&, v=val0, p=expert_w_gpu0[i]](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) { p[j] = v; });
        });
        q1.submit([&, v=val1, p=expert_w_gpu1[i]](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) { p[j] = v; });
        });
    }
    q0.wait(); q1.wait();

    // Token activations (on host-pinned for cross-device access)
    float * tokens = sycl::malloc_host<float>(N_TOKENS * HIDDEN_DIM, ctx);
    for (int t = 0; t < N_TOKENS; t++)
        for (int h = 0; h < HIDDEN_DIM; h++)
            tokens[t * HIDDEN_DIM + h] = 1.0f;

    // Routing table: token → expert assignments (predetermined for test)
    // Each token gets 2 experts. Deliberately cross-device assignments.
    struct route { int token; int expert; };
    route routes[] = {
        {0, 0}, {0, 4},  // Token 0 → expert 0 (GPU0) + expert 4 (GPU1)
        {1, 1}, {1, 5},  // Token 1 → expert 1 (GPU0) + expert 5 (GPU1)
        {2, 2}, {2, 6},
        {3, 3}, {3, 7},
        {4, 0}, {4, 7},  // Token 4 → expert 0 (GPU0) + expert 7 (GPU1)
        {5, 2}, {5, 5},
        {6, 3}, {6, 4},
        {7, 1}, {7, 6},
    };
    int n_routes = sizeof(routes) / sizeof(routes[0]);

    // Output buffer (host-pinned for merging)
    float * output = sycl::malloc_host<float>(N_TOKENS * HIDDEN_DIM, ctx);
    memset(output, 0, N_TOKENS * HIDDEN_DIM * sizeof(float));

    // ─── Test 1: Correctness ─────────────────────────
    printf("\n--- Test 1: Cross-device expert routing correctness ---\n");
    {
        // For each route, compute: output[token] += token_activation * expert_weight
        // This is a simplified MoE forward: token × expert = scaled output

        for (int r = 0; r < n_routes; r++) {
            int tok = routes[r].token;
            int exp_id = routes[r].expert;
            float * tok_ptr = tokens + tok * HIDDEN_DIM;

            if (exp_id < 4) {
                // Expert on GPU0
                float * w = expert_w_gpu0[exp_id];
                float * out_dev = sycl::malloc_device<float>(HIDDEN_DIM, q0);

                auto evt = q0.submit([&, tok_ptr, w, out_dev](sycl::handler & h) {
                    h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                        out_dev[j] = tok_ptr[j] * w[j];
                    });
                });

                // Copy result to host output (accumulate)
                float * staging = sycl::malloc_host<float>(HIDDEN_DIM, ctx);
                q0.memcpy(staging, out_dev, HIDDEN_DIM * sizeof(float)).wait();
                for (int j = 0; j < HIDDEN_DIM; j++)
                    output[tok * HIDDEN_DIM + j] += staging[j];

                sycl::free(staging, ctx);
                sycl::free(out_dev, q0);
            } else {
                // Expert on GPU1
                float * w = expert_w_gpu1[exp_id - 4];
                float * out_dev = sycl::malloc_device<float>(HIDDEN_DIM, q1);

                auto evt = q1.submit([&, tok_ptr, w, out_dev](sycl::handler & h) {
                    h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                        out_dev[j] = tok_ptr[j] * w[j];
                    });
                });

                float * staging = sycl::malloc_host<float>(HIDDEN_DIM, ctx);
                q1.memcpy(staging, out_dev, HIDDEN_DIM * sizeof(float)).wait();
                for (int j = 0; j < HIDDEN_DIM; j++)
                    output[tok * HIDDEN_DIM + j] += staging[j];

                sycl::free(staging, ctx);
                sycl::free(out_dev, q1);
            }
        }

        // Verify: each token's output should be sum of its 2 expert weights
        // (since token activation = 1.0, output[tok][j] = expert_w1 + expert_w2)
        int errors = 0;
        for (int t = 0; t < N_TOKENS; t++) {
            float expected = 0;
            for (int r = 0; r < n_routes; r++) {
                if (routes[r].token == t)
                    expected += (float)(routes[r].expert + 1);
            }
            float actual = output[t * HIDDEN_DIM];
            if (std::abs(actual - expected) > 0.01f) {
                printf("  Token %d: expected %.1f, got %.1f\n", t, expected, actual);
                errors++;
            }
        }
        printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    }

    // ─── Test 2: Latency with depends_on pipeline ────
    printf("\n--- Test 2: Pipelined expert routing latency ---\n");
    {
        // Simulate the real pattern:
        // 1. GPU0 computes gating (determines routing)
        // 2. GPU0 sends routed tokens to GPU1 via host staging
        // 3. GPU1 computes experts, depends_on the transfer
        // 4. GPU0 waits for GPU1 result, merges

        float * staging = sycl::malloc_host<float>(HIDDEN_DIM, ctx);
        float * gpu0_out = sycl::malloc_device<float>(HIDDEN_DIM, q0);
        float * gpu1_out = sycl::malloc_device<float>(HIDDEN_DIM, q1);

        const int ITERS = 100;

        // Warmup
        for (int i = 0; i < 10; i++) {
            auto e0 = q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                    gpu0_out[j] = 1.0f;
                });
            });
            q0.memcpy(staging, gpu0_out, HIDDEN_DIM * sizeof(float));
            auto e_copy = q0.submit([](sycl::handler & h) { h.single_task([]{}); });

            q1.submit([&](sycl::handler & h) {
                h.depends_on(e_copy);
                h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                    gpu1_out[j] = staging[j] * 2.0f;
                });
            });
            q1.memcpy(staging, gpu1_out, HIDDEN_DIM * sizeof(float));

            q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                    gpu0_out[j] += staging[j];
                });
            });
            q0.wait(); q1.wait();
        }

        timer t;
        for (int i = 0; i < ITERS; i++) {
            // Step 1: GPU0 gating (simulated)
            auto e_gate = q0.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                    gpu0_out[j] = 1.0f;
                });
            });

            // Step 2: Transfer to staging
            q0.memcpy(staging, gpu0_out, HIDDEN_DIM * sizeof(float));
            auto e_staged = q0.submit([](sycl::handler & h) { h.single_task([]{}); });

            // Step 3: GPU1 expert compute (depends on transfer)
            q1.submit([&](sycl::handler & h) {
                h.depends_on(e_staged);
                h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                    gpu1_out[j] = staging[j] * expert_w_gpu1[0][j];
                });
            });

            // Step 4: Transfer result back
            q1.memcpy(staging, gpu1_out, HIDDEN_DIM * sizeof(float));
            auto e_result = q1.submit([](sycl::handler & h) { h.single_task([]{}); });

            // Step 5: GPU0 merge (depends on GPU1 result)
            q0.submit([&](sycl::handler & h) {
                h.depends_on(e_result);
                h.parallel_for(sycl::range<1>(HIDDEN_DIM), [=](sycl::id<1> j) {
                    gpu0_out[j] += staging[j];
                });
            });
            q0.wait();
        }
        double total_us = t.us();
        printf("  Full round-trip: %.1f us/iter (%d iters)\n", total_us / ITERS, ITERS);
        printf("  Breakdown estimate: gating ~10us + staging ~20us + expert ~10us + merge ~10us\n");

        sycl::free(staging, ctx);
        sycl::free(gpu0_out, q0);
        sycl::free(gpu1_out, q1);
    }

    // ─── Test 3: Realistic data size (128 KB per expert activation) ─
    printf("\n--- Test 3: Realistic activation transfer size ---\n");
    {
        size_t sizes[] = {2*1024, 16*1024, 128*1024, 1024*1024};
        const char * names[] = {"2 KB", "16 KB", "128 KB", "1 MB"};

        for (int s = 0; s < 4; s++) {
            size_t nbytes = sizes[s];
            auto * host_buf = sycl::malloc_host<uint8_t>(nbytes, ctx);
            auto * dev0_buf = sycl::malloc_device<uint8_t>(nbytes, q0);
            auto * dev1_buf = sycl::malloc_device<uint8_t>(nbytes, q1);

            // D0 → host → D1 → host → D0 (full round trip)
            q0.memset(dev0_buf, 0x42, nbytes).wait();

            const int ITERS = 50;
            timer t;
            for (int i = 0; i < ITERS; i++) {
                q0.memcpy(host_buf, dev0_buf, nbytes);     // D0 → host
                auto e = q0.submit([](sycl::handler & h) { h.single_task([]{}); });
                q1.submit([&](sycl::handler & h) {
                    h.depends_on(e);
                }).wait();
                q1.memcpy(dev1_buf, host_buf, nbytes);     // host → D1
                q1.memcpy(host_buf, dev1_buf, nbytes);     // D1 → host
                auto e2 = q1.submit([](sycl::handler & h) { h.single_task([]{}); });
                q0.submit([&](sycl::handler & h) {
                    h.depends_on(e2);
                }).wait();
                q0.memcpy(dev0_buf, host_buf, nbytes).wait(); // host → D0
            }
            double us_per = t.us() / ITERS;
            printf("  %6s round-trip: %8.1f us (%.2f GB/s effective)\n",
                names[s], us_per, (4.0 * nbytes) / (us_per * 1e3));

            sycl::free(host_buf, ctx);
            sycl::free(dev0_buf, q0);
            sycl::free(dev1_buf, q1);
        }
    }

    // ─── Cleanup ──────────────────────────────────────
    for (int i = 0; i < 4; i++) {
        sycl::free(expert_w_gpu0[i], q0);
        sycl::free(expert_w_gpu1[i], q1);
    }
    sycl::free(tokens, ctx);
    sycl::free(output, ctx);

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  If Test 1 PASS + Test 2 < 500 us:                      ║\n");
    printf("║    Expert parallelism is viable for multi-GPU MoE.      ║\n");
    printf("║    P4.5 hybrid design confirmed.                        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
