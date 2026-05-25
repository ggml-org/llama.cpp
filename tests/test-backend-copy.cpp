#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

struct bench_params {
    std::vector<std::string> backend_filters;
    std::vector<size_t>      sizes;
    int64_t min_time_us = 1000000;
    int     warmup      = 3;
    int     min_iters   = 5;
    bool    do_h2d      = true;
    bool    do_d2h      = true;
    bool    do_d2d      = true;
    bool    do_async    = true;
    bool    csv         = false;
};

struct device_info {
    ggml_backend_dev_t          dev;
    ggml_backend_ptr            backend;
    ggml_backend_buffer_ptr     buffer;
    std::string                 name;
    std::string                 description;
    enum ggml_backend_dev_type  type;
    size_t                      mem_free;
    size_t                      mem_total;
};

struct bench_result {
    std::string src_name;
    std::string dst_name;
    std::string method;
    size_t      size_bytes;
    double      avg_time_us;
    double      min_time_us;
    double      bandwidth_gbs;
    int         iterations;
};

static const std::vector<size_t> default_sizes = {
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
};

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s [options]\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help           show this help message\n");
    fprintf(stderr, "  -b NAME              benchmark only device NAME (repeatable)\n");
    fprintf(stderr, "  -s BYTES             add a transfer size in bytes (repeatable, replaces defaults)\n");
    fprintf(stderr, "  --min-time-ms MS     minimum benchmark duration per measurement (default: 1000)\n");
    fprintf(stderr, "  --warmup N           warmup iterations (default: 3)\n");
    fprintf(stderr, "  --min-iters N        minimum timed iterations (default: 5)\n");
    fprintf(stderr, "  --no-h2d             skip host-to-device tests\n");
    fprintf(stderr, "  --no-d2h             skip device-to-host tests\n");
    fprintf(stderr, "  --no-d2d             skip device-to-device tests\n");
    fprintf(stderr, "  --no-async           skip async copy tests\n");
    fprintf(stderr, "  --output FORMAT      output format: table (default), csv\n");
}

static bool parse_args(int argc, char ** argv, bench_params & params) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return false;
        } else if (strcmp(argv[i], "-b") == 0) {
            if (++i >= argc) { usage(argv[0]); return false; }
            params.backend_filters.push_back(argv[i]);
        } else if (strcmp(argv[i], "-s") == 0) {
            if (++i >= argc) { usage(argv[0]); return false; }
            size_t s = (size_t)atoll(argv[i]);
            if (s == 0) { fprintf(stderr, "invalid size: %s\n", argv[i]); return false; }
            // round up to multiple of sizeof(float)
            s = (s + sizeof(float) - 1) / sizeof(float) * sizeof(float);
            params.sizes.push_back(s);
        } else if (strcmp(argv[i], "--min-time-ms") == 0) {
            if (++i >= argc) { usage(argv[0]); return false; }
            params.min_time_us = atoll(argv[i]) * 1000;
        } else if (strcmp(argv[i], "--warmup") == 0) {
            if (++i >= argc) { usage(argv[0]); return false; }
            params.warmup = atoi(argv[i]);
        } else if (strcmp(argv[i], "--min-iters") == 0) {
            if (++i >= argc) { usage(argv[0]); return false; }
            params.min_iters = atoi(argv[i]);
        } else if (strcmp(argv[i], "--no-h2d") == 0) {
            params.do_h2d = false;
        } else if (strcmp(argv[i], "--no-d2h") == 0) {
            params.do_d2h = false;
        } else if (strcmp(argv[i], "--no-d2d") == 0) {
            params.do_d2d = false;
        } else if (strcmp(argv[i], "--no-async") == 0) {
            params.do_async = false;
        } else if (strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) { usage(argv[0]); return false; }
            if (strcmp(argv[i], "csv") == 0) {
                params.csv = true;
            } else if (strcmp(argv[i], "table") == 0) {
                params.csv = false;
            } else {
                fprintf(stderr, "unknown output format: %s\n", argv[i]);
                return false;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return false;
        }
    }

    if (params.sizes.empty()) {
        params.sizes = default_sizes;
    }
    std::sort(params.sizes.begin(), params.sizes.end());

    return true;
}

static std::string format_size(size_t bytes) {
    char buf[64];
    if (bytes >= 1024 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%zu GB", bytes / (1024 * 1024 * 1024));
    } else if (bytes >= 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%zu MB", bytes / (1024 * 1024));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%zu KB", bytes / 1024);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

static bool device_matches_filter(const char * name, const std::vector<std::string> & filters) {
    if (filters.empty()) {
        return true;
    }
    for (const auto & f : filters) {
        if (f == name) {
            return true;
        }
    }
    return false;
}

static bench_result benchmark_copy(
        const char * src_name,
        const char * dst_name,
        const char * method,
        size_t size_bytes,
        const bench_params & params,
        const std::function<void()> & sync_before,
        const std::function<void()> & copy_fn,
        const std::function<void()> & sync_after) {

    for (int i = 0; i < params.warmup; i++) {
        sync_before();
        copy_fn();
        sync_after();
    }

    int64_t total_time_us = 0;
    int64_t min_time      = INT64_MAX;
    int     iterations    = 0;

    while (total_time_us < params.min_time_us || iterations < params.min_iters) {
        sync_before();
        int64_t t0 = ggml_time_us();
        copy_fn();
        sync_after();
        int64_t t1 = ggml_time_us();

        int64_t elapsed = t1 - t0;
        total_time_us += elapsed;
        min_time = std::min(min_time, elapsed);
        iterations++;
    }

    double avg_us = (double)total_time_us / iterations;
    double bw_gbs = (double)size_bytes / avg_us * 1e6 / (1024.0 * 1024.0 * 1024.0);

    return { src_name, dst_name, method, size_bytes, avg_us, (double)min_time, bw_gbs, iterations };
}

static void print_csv_header() {
    printf("src,dst,method,size_bytes,avg_time_us,min_time_us,bandwidth_gbs,iterations\n");
}

static void print_csv_row(const bench_result & r) {
    printf("%s,%s,%s,%zu,%.2f,%.2f,%.4f,%d\n",
        r.src_name.c_str(), r.dst_name.c_str(), r.method.c_str(),
        r.size_bytes, r.avg_time_us, r.min_time_us, r.bandwidth_gbs, r.iterations);
}

static void print_table_header(size_t size) {
    printf("\n--- %s ---\n", format_size(size).c_str());
    printf("  %-14s %-14s %-12s %10s %10s %12s %6s\n",
        "Source", "Destination", "Method", "Avg (us)", "Min (us)", "BW (GB/s)", "Iters");
}

static void print_table_row(const bench_result & r) {
    printf("  %-14s %-14s %-12s %10.1f %10.1f %12.4f %6d\n",
        r.src_name.c_str(), r.dst_name.c_str(), r.method.c_str(),
        r.avg_time_us, r.min_time_us, r.bandwidth_gbs, r.iterations);
}

struct tensor_on_device {
    ggml_context_ptr ctx;
    ggml_tensor *    tensor;
};

static tensor_on_device create_tensor_on_buffer(ggml_backend_buffer_t buffer, size_t size_bytes, const char * name) {
    int64_t n_elements = (int64_t)(size_bytes / sizeof(float));
    if (n_elements == 0) {
        n_elements = 1;
    }

    ggml_init_params init_params = {
        /* .mem_size   = */ ggml_tensor_overhead() + 64,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(init_params));
    if (!ctx) {
        return { nullptr, nullptr };
    }

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_elements);
    ggml_set_name(tensor, name);

    struct ggml_tallocr talloc = ggml_tallocr_new(buffer);
    ggml_tallocr_alloc(&talloc, tensor);

    return { std::move(ctx), tensor };
}

int main(int argc, char ** argv) {
    bench_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    ggml_backend_load_all();

    // Enumerate devices
    std::vector<device_info> devices;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto dev_type = ggml_backend_dev_type(dev);

        if (dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL ||
            dev_type == GGML_BACKEND_DEVICE_TYPE_META) {
            continue;
        }

        const char * dev_name = ggml_backend_dev_name(dev);

        if (!device_matches_filter(dev_name, params.backend_filters)) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "warning: failed to init backend for %s, skipping\n", dev_name);
            continue;
        }

        size_t mem_free = 0, mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);

        device_info di;
        di.dev         = dev;
        di.backend     = ggml_backend_ptr(backend);
        di.buffer      = nullptr;
        di.name        = dev_name;
        di.description = ggml_backend_dev_description(dev);
        di.type        = dev_type;
        di.mem_free    = mem_free;
        di.mem_total   = mem_total;
        devices.push_back(std::move(di));
    }

    if (devices.empty()) {
        fprintf(stderr, "error: no devices found\n");
        return 1;
    }

    // Print device table
    if (!params.csv) {
        printf("=== ggml Backend Copy Benchmark ===\n\n");
        printf("Devices:\n");
        for (size_t i = 0; i < devices.size(); i++) {
            printf("  [%zu] %-10s : %s (%zu MB / %zu MB)\n",
                i, devices[i].name.c_str(), devices[i].description.c_str(),
                devices[i].mem_free / (1024 * 1024),
                devices[i].mem_total / (1024 * 1024));
        }
        printf("\n");
    }

    // Separate CPU and GPU devices
    std::vector<size_t> gpu_indices;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].type != GGML_BACKEND_DEVICE_TYPE_CPU) {
            gpu_indices.push_back(i);
        }
    }

    if (gpu_indices.empty() && !params.csv) {
        printf("No GPU devices found. Only CPU-to-CPU results available.\n\n");
    }

    // Allocate buffers on each device (sized for the largest test)
    size_t max_size = params.sizes.back();
    for (auto & dev : devices) {
        size_t alloc_size = max_size + 4096;
        if (dev.mem_total > 0 && alloc_size > dev.mem_free) {
            fprintf(stderr, "warning: max test size (%s) exceeds free memory on %s (%zu MB), will skip large sizes\n",
                format_size(max_size).c_str(), dev.name.c_str(), dev.mem_free / (1024 * 1024));
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(dev.backend.get(), alloc_size);
        if (!buf) {
            fprintf(stderr, "warning: failed to allocate buffer on %s\n", dev.name.c_str());
        }
        dev.buffer.reset(buf);
    }

    // Host data buffer
    std::vector<uint8_t> host_data(max_size, 0xAB);

    if (params.csv) {
        print_csv_header();
    }

    // Run benchmarks for each size
    for (size_t size : params.sizes) {
        if (!params.csv) {
            print_table_header(size);
        }

        // Create tensors for this size on each device
        struct dev_tensor {
            size_t dev_idx;
            tensor_on_device td;
        };
        std::vector<dev_tensor> tensors;

        for (size_t i = 0; i < devices.size(); i++) {
            if (!devices[i].buffer) {
                continue;
            }
            if (devices[i].mem_total > 0 && size + 4096 > devices[i].mem_free) {
                continue;
            }
            char name[64];
            snprintf(name, sizeof(name), "%s_%s", devices[i].name.c_str(), format_size(size).c_str());
            auto td = create_tensor_on_buffer(devices[i].buffer.get(), size, name);
            if (!td.tensor) {
                fprintf(stderr, "warning: failed to create tensor on %s for size %s\n",
                    devices[i].name.c_str(), format_size(size).c_str());
                continue;
            }
            tensors.push_back({ i, std::move(td) });
        }

        // Build index: dev_idx -> tensors array index
        auto find_tensor = [&](size_t dev_idx) -> ggml_tensor * {
            for (auto & t : tensors) {
                if (t.dev_idx == dev_idx) {
                    return t.td.tensor;
                }
            }
            return nullptr;
        };

        // Host-to-Device
        if (params.do_h2d) {
            for (size_t gi : gpu_indices) {
                ggml_tensor * dst = find_tensor(gi);
                if (!dst) { continue; }
                ggml_backend_t be = devices[gi].backend.get();

                auto r = benchmark_copy("Host", devices[gi].name.c_str(), "set", size, params,
                    [&]{ ggml_backend_synchronize(be); },
                    [&]{ ggml_backend_tensor_set(dst, host_data.data(), 0, size); },
                    [&]{ ggml_backend_synchronize(be); });

                params.csv ? print_csv_row(r) : print_table_row(r);
            }
        }

        // Device-to-Host
        if (params.do_d2h) {
            for (size_t gi : gpu_indices) {
                ggml_tensor * src = find_tensor(gi);
                if (!src) { continue; }
                ggml_backend_t be = devices[gi].backend.get();

                auto r = benchmark_copy(devices[gi].name.c_str(), "Host", "get", size, params,
                    [&]{ ggml_backend_synchronize(be); },
                    [&]{ ggml_backend_tensor_get(src, host_data.data(), 0, size); },
                    [&]{ ggml_backend_synchronize(be); });

                params.csv ? print_csv_row(r) : print_table_row(r);
            }
        }

        // Device-to-Device (sync)
        if (params.do_d2d) {
            for (size_t si = 0; si < devices.size(); si++) {
                for (size_t di = 0; di < devices.size(); di++) {
                    if (si == di) { continue; }

                    ggml_tensor * src = find_tensor(si);
                    ggml_tensor * dst = find_tensor(di);
                    if (!src || !dst) { continue; }

                    ggml_backend_t be_src = devices[si].backend.get();
                    ggml_backend_t be_dst = devices[di].backend.get();

                    auto r = benchmark_copy(
                        devices[si].name.c_str(), devices[di].name.c_str(), "copy_sync", size, params,
                        [&]{
                            ggml_backend_synchronize(be_src);
                            ggml_backend_synchronize(be_dst);
                        },
                        [&]{ ggml_backend_tensor_copy(src, dst); },
                        [&]{
                            ggml_backend_synchronize(be_src);
                            ggml_backend_synchronize(be_dst);
                        });

                    params.csv ? print_csv_row(r) : print_table_row(r);
                }
            }
        }

        // Device-to-Device (async)
        if (params.do_d2d && params.do_async) {
            for (size_t si = 0; si < devices.size(); si++) {
                for (size_t di = 0; di < devices.size(); di++) {
                    if (si == di) { continue; }

                    ggml_tensor * src = find_tensor(si);
                    ggml_tensor * dst = find_tensor(di);
                    if (!src || !dst) { continue; }

                    ggml_backend_t be_src = devices[si].backend.get();
                    ggml_backend_t be_dst = devices[di].backend.get();

                    auto r = benchmark_copy(
                        devices[si].name.c_str(), devices[di].name.c_str(), "copy_async", size, params,
                        [&]{
                            ggml_backend_synchronize(be_src);
                            ggml_backend_synchronize(be_dst);
                        },
                        [&]{ ggml_backend_tensor_copy_async(be_src, be_dst, src, dst); },
                        [&]{
                            ggml_backend_synchronize(be_src);
                            ggml_backend_synchronize(be_dst);
                        });

                    params.csv ? print_csv_row(r) : print_table_row(r);
                }
            }
        }
    }

    if (!params.csv) {
        printf("\nDone.\n");
    }

    return 0;
}
