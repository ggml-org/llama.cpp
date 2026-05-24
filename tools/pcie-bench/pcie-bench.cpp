#include "ggml-cpu/pinned.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static double bench_raw_bandwidth(size_t buffer_size_mb, int warmup, int iterations) {
    size_t size = buffer_size_mb * 1024 * 1024;
    void* host_ptr = ggml_cpu_pinned_alloc(size);
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate %zu MB pinned\n", buffer_size_mb);
        return -1;
    }

    // Fill with pattern
    memset(host_ptr, 0xAB, size);

    void* device_ptr = nullptr;
    cuda_check(cudaMalloc(&device_ptr, size), "cudaMalloc");

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cuda_check(cudaMemcpyAsync(device_ptr, host_ptr, size,
                                    cudaMemcpyHostToDevice, stream), "warmup copy");
        cuda_check(cudaStreamSynchronize(stream), "warmup sync");
    }

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create");
    cuda_check(cudaEventCreate(&stop), "event create");

    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        cuda_check(cudaEventRecord(start, stream), "record start");
        cuda_check(cudaMemcpyAsync(device_ptr, host_ptr, size,
                                    cudaMemcpyHostToDevice, stream), "copy");
        cuda_check(cudaEventRecord(stop, stream), "record stop");
        cuda_check(cudaEventSynchronize(stop), "sync");

        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        total_time += ms;
    }

    double avg_ms = total_time / iterations;
    double bandwidth_gb_s = (double)size / (1024.0 * 1024.0 * 1024.0) / (avg_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(device_ptr);
    ggml_cpu_pinned_free(host_ptr, size);

    return bandwidth_gb_s;
}

static void bench_layer_overlap(size_t layer_size_mb, int num_layers) {
    size_t size = layer_size_mb * 1024 * 1024;

    // Allocate two layers for ping-pong overlap
    void* host_ptrs[2];
    void* device_ptrs[2];
    void* device_compute[2];

    for (int i = 0; i < 2; i++) {
        host_ptrs[i] = ggml_cpu_pinned_alloc(size);
        if (!host_ptrs[i]) {
            fprintf(stderr, "Failed to allocate layer %d\n", i);
            return;
        }
        memset(host_ptrs[i], 0xAB, size);

        cuda_check(cudaMalloc(&device_ptrs[i], size), "cudaMalloc transfer");
        cuda_check(cudaMalloc(&device_compute[i], size), "cudaMalloc compute");
    }

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "stream create");

    double total_transfer = 0;
    double total_compute = 0;
    double total_overlap = 0;

    // Warmup
    {
        cuda_check(cudaMemcpyAsync(device_ptrs[0], host_ptrs[0], size,
                                    cudaMemcpyHostToDevice, stream), "warmup");
        cuda_check(cudaMemsetAsync(device_compute[0], 0, size, stream), "warmup compute");
        cuda_check(cudaStreamSynchronize(stream), "warmup sync");
    }

    for (int layer = 0; layer < num_layers; layer++) {
        int idx = layer % 2;
        cudaEvent_t transfer_start, transfer_end;
        cudaEvent_t compute_start, compute_end;
        cuda_check(cudaEventCreate(&transfer_start), "event");
        cuda_check(cudaEventCreate(&transfer_end), "event");
        cuda_check(cudaEventCreate(&compute_start), "event");
        cuda_check(cudaEventCreate(&compute_end), "event");

        cuda_check(cudaEventRecord(transfer_start, stream), "record");

        // Transfer next layer
        cuda_check(cudaMemcpyAsync(device_ptrs[idx], host_ptrs[idx], size,
                                    cudaMemcpyHostToDevice, stream), "layer transfer");

        cuda_check(cudaEventRecord(transfer_end, stream), "record");

        // Compute on previously transferred layer (memset as async work proxy)
        cuda_check(cudaEventRecord(compute_start, stream), "record");
        cuda_check(cudaMemsetAsync(device_compute[idx], 0, size, stream), "compute");
        cuda_check(cudaEventRecord(compute_end, stream), "record");

        cuda_check(cudaEventSynchronize(compute_end), "sync");

        float t_ms = 0, c_ms = 0;
        cuda_check(cudaEventElapsedTime(&t_ms, transfer_start, transfer_end), "time");
        cuda_check(cudaEventElapsedTime(&c_ms, compute_start, compute_end), "time");
        total_transfer += t_ms;
        total_compute += c_ms;

        // Overlap = how much of transfer was hidden by compute
        if (c_ms > t_ms) total_overlap += t_ms;  // fully hidden
        else total_overlap += c_ms;               // partially hidden

        cudaEventDestroy(transfer_start);
        cudaEventDestroy(transfer_end);
        cudaEventDestroy(compute_start);
        cudaEventDestroy(compute_end);
    }

    double avg_transfer = total_transfer / num_layers;
    double avg_compute = total_compute / num_layers;
    double transfer_bw = (double)size / (1024.0*1024.0*1024.0) / (avg_transfer / 1000.0);
    double overlap_pct = (total_overlap / total_transfer) * 100.0;

    cudaStreamDestroy(stream);
    for (int i = 0; i < 2; i++) {
        cudaFree(device_ptrs[i]);
        cudaFree(device_compute[i]);
        ggml_cpu_pinned_free(host_ptrs[i], size);
    }

    printf("  Layer size: %.1f MB\n", (double)layer_size_mb);
    printf("  Transfer time: %.1f ms (%.1f GB/s)\n", avg_transfer, transfer_bw);
    printf("  Compute time: %.1f ms\n", avg_compute);
    printf("  Overlap efficiency: %.0f%% (perfect = 100%% hidden transfer)\n", overlap_pct);
}

int main(int argc, char** argv) {
    int device = 0;
    size_t buffer_size_mb = 1024;  // default 1GB
    int warmup = 10;
    int iterations = 50;
    size_t layer_size_mb = 256;
    int num_layers = 20;
    const char* mode = "raw";

    static struct option long_options[] = {
        {"device",   required_argument, NULL, 'd'},
        {"size",     required_argument, NULL, 's'},
        {"warmup",   required_argument, NULL, 'w'},
        {"iter",     required_argument, NULL, 'n'},
        {"layer",    required_argument, NULL, 'l'},
        {"layers",   required_argument, NULL, 'L'},
        {"raw",      no_argument,       NULL, 'r'},
        {"overlap",  no_argument,       NULL, 'o'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:s:w:n:l:L:ro", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': device = atoi(optarg); break;
            case 's': buffer_size_mb = atoi(optarg); break;
            case 'w': warmup = atoi(optarg); break;
            case 'n': iterations = atoi(optarg); break;
            case 'l': layer_size_mb = atoi(optarg); break;
            case 'L': num_layers = atoi(optarg); break;
            case 'r': mode = "raw"; break;
            case 'o': mode = "overlap"; break;
            default:
                fprintf(stderr, "Usage: %s [--raw|--overlap] [options]\n", argv[0]);
                fprintf(stderr, "  --raw       Raw PCIe bandwidth test (default)\n");
                fprintf(stderr, "  --overlap   Layer transfer overlap test\n");
                fprintf(stderr, "  -s MB       Buffer size in MB (default: 1024)\n");
                fprintf(stderr, "  -w N        Warmup iterations (default: 10)\n");
                fprintf(stderr, "  -n N        Timed iterations (default: 50)\n");
                fprintf(stderr, "  -l MB       Layer size for overlap test (default: 256)\n");
                fprintf(stderr, "  -L N        Number of layers (default: 20)\n");
                fprintf(stderr, "  -d N        GPU device (default: 0)\n");
                return 1;
        }
    }

    cuda_check(cudaSetDevice(device), "set device");

    cudaDeviceProp props;
    cuda_check(cudaGetDeviceProperties(&props, device), "device props");
    printf("GPU: %s (compute %d.%d)\n", props.name, props.major, props.minor);

    if (strcmp(mode, "raw") == 0) {
        printf("Mode: raw PCIe bandwidth\n");
        printf("Buffer: %zu MB, warmup: %d, iterations: %d\n",
               buffer_size_mb, warmup, iterations);

        double bw = bench_raw_bandwidth(buffer_size_mb, warmup, iterations);
        if (bw > 0) {
            printf("Pinned RAM -> GPU (cudaMemcpyAsync): %.1f GB/s\n", bw);
            printf("Expected ceiling: ~63 GB/s (PCIe Gen5 x16)\n");
            printf("Expected realistic: ~50 GB/s\n");
        }
    } else {
        printf("Mode: layer transfer overlap\n");
        bench_layer_overlap(layer_size_mb, num_layers);
    }

    return 0;
}
