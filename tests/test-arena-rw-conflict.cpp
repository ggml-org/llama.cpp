// Reproducer: Does a GPU kernel hang when reading from one sub-region
// of a large sycl::malloc_device while a previous kernel wrote to
// a DIFFERENT sub-region of the SAME allocation?
//
// This mirrors the arena setup:
//   - Compute zone (offset 0, 256 MB): written by Q8_1 quantization kernels
//   - Weight zone (offset ~8 GB): read by MMVQ kernels
//   - Both within one sycl::malloc_device block
//
// Build:  icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -o test-arena-rw test-arena-rw-conflict.cpp
// Run:    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-arena-rw

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    auto dev = q.get_device();
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Allocate 8 GB — mirrors the arena
    const size_t ARENA_SIZE = 8ULL * 1024 * 1024 * 1024;
    const size_t COMPUTE_ZONE_SIZE = 256ULL * 1024 * 1024;  // First 256 MB
    const size_t WEIGHT_OFFSET = ARENA_SIZE - 64ULL * 1024 * 1024;  // Last 64 MB
    const size_t N = 1024 * 1024;  // 1M floats = 4 MB

    printf("Allocating %.1f GB arena...\n", ARENA_SIZE / (1024.0 * 1024.0 * 1024.0));
    void * arena = sycl::malloc_device(ARENA_SIZE, q);
    if (!arena) {
        printf("FAILED: malloc_device returned nullptr\n");
        return 1;
    }
    printf("Arena at %p\n", arena);

    float * compute_zone = (float *)arena;  // Offset 0
    float * weight_zone = (float *)((char *)arena + WEIGHT_OFFSET);  // Near end

    // Upload weight data to the weight zone
    printf("\nUploading %zu floats to weight zone (offset %.1f GB)...\n",
           N, WEIGHT_OFFSET / (1024.0 * 1024.0 * 1024.0));
    float * host_weights = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) host_weights[i] = (float)(i % 1000) + 1.0f;
    q.memcpy(weight_zone, host_weights, N * sizeof(float)).wait();
    printf("Upload OK.\n");

    // Allocate separate output buffer
    float * output = sycl::malloc_device<float>(N, q);

    // ========================================================
    // Test 1: Read-only from weight zone (should work like standalone test)
    // ========================================================
    printf("\n=== Test 1: Read-only from weight zone ===\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        output[i] = weight_zone[i] * 2.0f;
    }).wait();
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("  PASS (%.2f ms)\n", ms);

    // ========================================================
    // Test 2: Write to compute zone, THEN read from weight zone
    //         (same allocation, different sub-regions, sequential)
    // ========================================================
    printf("\n=== Test 2: Write compute zone → Read weight zone (sequential) ===\n");
    t1 = std::chrono::high_resolution_clock::now();
    // Write kernel: fill compute zone with data
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        compute_zone[i] = (float)i * 0.5f;
    });
    // Read kernel: read from weight zone (DIFFERENT sub-region, SAME allocation)
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        output[i] = weight_zone[i] * 3.0f;
    });
    q.wait();  // Wait for both
    t2 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("  PASS (%.2f ms)\n", ms);

    // ========================================================
    // Test 3: Rapid alternating write+read (mirrors TG dispatch pattern)
    //         Multiple rounds of: write to compute zone, read from weight zone
    // ========================================================
    printf("\n=== Test 3: Rapid alternating write+read (10 rounds) ===\n");
    t1 = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < 10; round++) {
        // Write: quantize src1 into compute zone (simulates Q8_1 quantization)
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            compute_zone[i] = (float)(i + round) * 0.1f;
        });
        // Read: MMVQ reads weights from weight zone + quantized from compute zone
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            output[i] = weight_zone[i] * compute_zone[i];
        });
    }
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("  PASS (%.2f ms)\n", ms);

    // ========================================================
    // Test 4: memset compute zone (arena_reset), then read weight zone
    //         (simulates arena_reset between tokens)
    // ========================================================
    printf("\n=== Test 4: memset(compute zone, 0) → Read weight zone ===\n");
    t1 = std::chrono::high_resolution_clock::now();
    q.memset(compute_zone, 0, COMPUTE_ZONE_SIZE);
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        output[i] = weight_zone[i] * 4.0f;
    });
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("  PASS (%.2f ms)\n", ms);

    // ========================================================
    // Test 5: Graph recording + replay with arena pointers
    //         (simulates SYCL graph replay for TG)
    // ========================================================
    printf("\n=== Test 5: Graph record + replay with arena pointers ===\n");
    bool graph_ok = false;
    try {
        namespace ext = sycl::ext::oneapi::experimental;
        ext::command_graph graph(q.get_context(), q.get_device());

        graph.begin_recording(q);
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            compute_zone[i] = (float)i * 0.2f;
        });
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            output[i] = weight_zone[i] + compute_zone[i];
        });
        graph.end_recording(q);

        auto exec = graph.finalize();

        t1 = std::chrono::high_resolution_clock::now();
        for (int replay = 0; replay < 5; replay++) {
            q.ext_oneapi_graph(exec);
        }
        q.wait();
        t2 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        printf("  PASS (%.2f ms for 5 replays)\n", ms);
        graph_ok = true;
    } catch (const sycl::exception & e) {
        printf("  SKIP (graph not supported: %s)\n", e.what());
    }

    // Verify final output
    float * result = (float *)malloc(N * sizeof(float));
    q.memcpy(result, output, N * sizeof(float)).wait();
    bool verify_ok = true;
    for (size_t i = 0; i < 10; i++) {
        if (result[i] == 0.0f) {
            printf("  WARNING: output[%zu] = 0 (expected non-zero)\n", i);
            verify_ok = false;
        }
    }
    if (verify_ok) printf("\nOutput verification: OK\n");

    sycl::free(output, q);
    sycl::free(arena, q);
    free(host_weights);
    free(result);
    printf("\nAll tests completed.\n");
    return 0;
}
