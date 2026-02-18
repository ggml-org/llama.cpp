#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>
#include <immintrin.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

static const size_t BUF_SIZE = 4ULL * 1024 * 1024 * 1024; // 4 GB
static const int    ITERS    = 3;
static const int    THREADS[] = {4, 8, 12, 16, 18, 20};
static const int    N_THREAD_CONFIGS = sizeof(THREADS) / sizeof(THREADS[0]);

static double read_bandwidth(char * buf, size_t size, int nthreads) {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nthreads);

    double best = 0.0;
    for (int iter = 0; iter < ITERS; iter++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, size, size / nthreads),
            [&](const tbb::blocked_range<size_t> & r) {
                __m256i sum = _mm256_setzero_si256();
                for (size_t i = r.begin(); i < r.end(); i += 32) {
                    sum = _mm256_add_epi64(sum, _mm256_load_si256((__m256i *)(buf + i)));
                }
                volatile int64_t sink = _mm256_extract_epi64(sum, 0);
                (void)sink;
            }
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double gbps = (double)size / (1e9 * secs);
        best = std::max(best, gbps);
    }
    return best;
}

int main() {
    printf("=== DRAM Sequential Read Bandwidth (Arrow Lake DDR5-5600) ===\n");
    printf("Buffer: %zu MB, Iterations: %d (best of %d)\n\n", BUF_SIZE / (1024*1024), ITERS, ITERS);

    // Allocate + populate (pre-fault all pages)
    char * buf = (char *)mmap(nullptr, BUF_SIZE,
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                              -1, 0);
    if (buf == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // Fill with data to ensure pages are physically backed
    memset(buf, 0x42, BUF_SIZE);

    double results_no_thp[N_THREAD_CONFIGS];
    double results_thp[N_THREAD_CONFIGS];

    // --- No THP ---
    printf("Running without THP...\n");
    for (int t = 0; t < N_THREAD_CONFIGS; t++) {
        results_no_thp[t] = read_bandwidth(buf, BUF_SIZE, THREADS[t]);
        printf("  %2d threads: %.2f GB/s\n", THREADS[t], results_no_thp[t]);
        sleep(2);
    }

    // --- With THP ---
    printf("\nEnabling THP (madvise + re-touch + khugepaged settle)...\n");
    madvise(buf, BUF_SIZE, MADV_HUGEPAGE);
    // Re-touch pages to trigger THP coalescing
    for (size_t i = 0; i < BUF_SIZE; i += 4096) {
        buf[i] = 0x43;
    }
    printf("Waiting 5s for khugepaged...\n");
    sleep(5);

    printf("Running with THP...\n");
    for (int t = 0; t < N_THREAD_CONFIGS; t++) {
        results_thp[t] = read_bandwidth(buf, BUF_SIZE, THREADS[t]);
        printf("  %2d threads: %.2f GB/s\n", THREADS[t], results_thp[t]);
        sleep(2);
    }

    munmap(buf, BUF_SIZE);

    // --- Summary ---
    printf("\n%-8s  %-15s  %-16s  %s\n", "Threads", "No-THP (GB/s)", "With-THP (GB/s)", "THP Gain");
    double peak = 0.0;
    int peak_threads = 0;
    const char * peak_mode = "";

    for (int t = 0; t < N_THREAD_CONFIGS; t++) {
        double gain = 100.0 * (results_thp[t] - results_no_thp[t]) / results_no_thp[t];
        printf("%6d    %10.2f       %10.2f        %+.1f%%\n",
               THREADS[t], results_no_thp[t], results_thp[t], gain);
        if (results_no_thp[t] > peak) {
            peak = results_no_thp[t];
            peak_threads = THREADS[t];
            peak_mode = "without THP";
        }
        if (results_thp[t] > peak) {
            peak = results_thp[t];
            peak_threads = THREADS[t];
            peak_mode = "with THP";
        }
    }
    printf("\nPeak bandwidth: %.2f GB/s (%d threads, %s)\n", peak, peak_threads, peak_mode);

    return 0;
}
