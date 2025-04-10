// Benchmark quantization specific functions on synthetic data

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <algorithm>
#include <assert.h>
#include <functional>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_ALIGNMENT 64
#define QK 32
#define WARMUP 5
#define ITERATIONS 10
#define MAX_ITERATIONS 100000000

#define L1_SIZE      32*128
#define L2_SIZE     32*2048
#define L3_SIZE    32*20480
#define MEM_SIZE 32*2048000

struct quantize_perf_params {
    std::vector<std::string> include_types;
    std::vector<size_t> test_sizes;
    size_t alignment_offset = 0;
    bool op_quantize_row_q_reference = false;
    bool op_quantize_row_q = false;
    bool op_dequantize_row_q = false;
    bool op_quantize_row_q_dot = false;
    bool op_vec_dot_q = false;
    int64_t iterations = ITERATIONS;
};

#if defined(__x86_64__) || defined(__i386__)

#include <x86intrin.h>
inline int64_t cpu_cycles() {
// Rough way to detect new-ish CPUs
#ifdef __POPCNT__
    unsigned int dummy;
    return __rdtscp(&dummy);
#else
    return __rdtsc();
#endif
}

#else

#define cpu_cycles() 0

#endif


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

static float gigabytes_per_second(size_t bytes, int64_t usecs) {
    return bytes / (float) usecs * 1000000 / (1024*1024*1024);
}

static void * align_with_offset(void * ptr, int offset) {
    size_t dummy_size = MAX_ALIGNMENT * 4;
    return (char *) std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, ptr, dummy_size) + offset;
}

static void benchmark_function(size_t size, size_t q_size, int64_t iterations, const std::function<float(void)> & func) {
    int64_t min_time_us = INT64_MAX;
    int64_t total_time_us = 0;
    int64_t min_time_cycles = INT64_MAX;
    int64_t total_time_cycles = 0;

    for (int i = 0; i < WARMUP; i++) {
        func();
    }

    for (int i = 0; i < iterations; i++) {
        const int64_t start_time = ggml_time_us();
        const int64_t start_cycles = cpu_cycles();

        func();

        const int64_t end_cycles = cpu_cycles();
        const int64_t end_time = ggml_time_us();

        total_time_cycles += end_cycles - start_cycles;
        min_time_cycles = std::min(min_time_cycles, end_cycles - start_cycles);
        total_time_us += end_time - start_time;
        min_time_us = std::min(min_time_us, end_time - start_time);
    }

    printf("      min cycles/%d vals   : %9.2f\n",  QK, QK * min_time_cycles / (float) size);
    printf("      avg cycles/%d vals   : %9.2f\n",  QK, QK * total_time_cycles / (float) (size * iterations));
    printf("      float32 throughput   : %9.2f GB/s\n",  gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n",  gigabytes_per_second(q_size * iterations, total_time_us));
}

static void usage(char * argv[]) {
    printf("Benchmark quantization specific functions on synthetic data\n");
    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options: (default)\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --size SIZE           set test size, divisible by 32 (L1_SIZE:%d)\n", L1_SIZE);
    printf("  -3                    use size as L1, L2, L3 sizes (L1:%d L2:%d L3:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE);
    printf("  -4                    use size as L1, L2, L3, MEM sizes (L1:%d L2:%d L3:%d MEM:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE, MEM_SIZE);
    printf("  --op OP               set test operation as quantize_row_q_reference, quantize_row_q, dequantize_row_q,\n");
    printf("                        quantize_row_q_dot, vec_dot_q (all)\n");
    printf("  --type TYPE           set test type as");
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (ggml_type_name(type) != NULL) {
            if (qfns_cpu->from_float && qfns->to_float) {
                printf(" %s", ggml_type_name(type));
            }
        }
    }
    printf(" (all)\n");
    printf("  --alignment-offset OFFSET\n");
    printf("                        set alignment offset as OFFSET (0)\n");
    printf("  -i NUM, --iterations NUM\n");
    printf("                        set test iteration number (%d)\n", ITERATIONS);
}

int main(int argc, char * argv[]) {
    quantize_perf_params params {};

    // read command line

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "--size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            size_t size = std::stoi(argv[i]);
            if (size % 32 != 0) {
                fprintf(stderr, "error: size %zu not divisible by 32\n", size);
                invalid_param = true;
                break;
            }
            params.test_sizes.push_back(size);
        } else if (arg == "-3") {
            // quick select sizes that probably fit in CPU caches
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
        } else if (arg == "-4") {
            // quick select cache sizes + memory
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
            params.test_sizes.push_back(MEM_SIZE);
        } else if (arg == "--op") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string op {argv[i]};
            if (op == "quantize_row_q_reference") {
                params.op_quantize_row_q_reference = true;
            } else if (op == "quantize_row_q") {
                params.op_quantize_row_q = true;
            } else if (op == "dequantize_row_q") {
                params.op_dequantize_row_q = true;
            } else if (op == "quantize_row_q_dot") {
                params.op_quantize_row_q_dot = true;
            } else if (op == "vec_dot_q") {
                params.op_vec_dot_q = true;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_types.push_back(argv[i]);
        } else if (arg == "--alignment-offset") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int alignment = std::stoi(argv[i]);
            if (alignment < 0 || alignment > MAX_ALIGNMENT) {
            fprintf(stderr, "error: alignment-offset must be less than %d\n", MAX_ALIGNMENT);
                invalid_param = true;
                break;
            }
            params.alignment_offset = alignment;
        } else if ((arg == "-i") || (arg == "--iterations")) {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int number = std::stoi(argv[i]);
            if (number < 0 || number > MAX_ITERATIONS) {
            fprintf(stderr, "error: iterations must be less than %d\n", MAX_ITERATIONS);
                invalid_param = true;
                break;
            }
            params.iterations = number;
        } else if ((arg == "-h") || (arg == "--help")) {
            usage(argv);
            return 1;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return 1;
    }

    if (params.test_sizes.empty()) {
        params.test_sizes.push_back(L1_SIZE);
    }
    if (!(params.op_quantize_row_q_reference || params.op_quantize_row_q || params.op_dequantize_row_q || params.op_quantize_row_q_dot || params.op_vec_dot_q)) {
        params.op_quantize_row_q_reference = params.op_quantize_row_q = params.op_dequantize_row_q = params.op_quantize_row_q_dot = params.op_vec_dot_q = true;
    }

    std::sort(params.test_sizes.begin(), params.test_sizes.end());
    size_t largest = params.test_sizes.back();

    std::vector<uint8_t> test_data1_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_data2_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q1_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q2_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_out_v  (largest*4 + MAX_ALIGNMENT*2);

    float * test_data1 = (float *) align_with_offset(test_data1_v.data(), params.alignment_offset);
    float * test_data2 = (float *) align_with_offset(test_data2_v.data(), params.alignment_offset);
    float * test_q1    = (float *) align_with_offset(test_q1_v.data(),    params.alignment_offset);
    float * test_q2    = (float *) align_with_offset(test_q2_v.data(),    params.alignment_offset);
    float * test_out   = (float *) align_with_offset(test_out_v.data(),   params.alignment_offset);

    generate_data(0, largest, test_data1);
    generate_data(1, largest, test_data2);

    int64_t iterations = params.iterations;


    // Initialize GGML, ensures float conversion tables are initialized
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (!params.include_types.empty() && ggml_type_name(type) && std::find(params.include_types.begin(), params.include_types.end(), ggml_type_name(type)) == params.include_types.end()) {
            continue;
        }

        if (qfns_cpu->from_float && qfns->to_float) {
            printf("%s\n", ggml_type_name(type));

            ggml_quantize_init(type);

            if (params.op_quantize_row_q_reference) {
                printf("  quantize_row_q_reference\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns->from_float_ref(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q) {
                printf("  quantize_row_q\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns_cpu->from_float(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_dequantize_row_q) {
                printf("  dequantize_row_q\n");
                qfns_cpu->from_float(test_data1, test_q1, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        qfns->to_float(test_q1, test_out, size);
                        return test_out[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_quantize_row_q_dot) {
                printf("  quantize_row_q_dot\n");
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
                        vdot->from_float(test_data1, test_q1, size);
                        return test_q1[0];
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }

            if (params.op_vec_dot_q) {
                printf("  vec_dot_q\n");
                qfns_cpu->from_float(test_data1, test_q1, largest);
                qfns_cpu->from_float(test_data2, test_q2, largest);
                for (size_t size : params.test_sizes) {
                    printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                    auto quantize_fn = [&](void) -> float {
                        float result;
                        qfns_cpu->vec_dot(size, &result, 0, test_q1, 0, test_q2, 0, 1);
                        return result;
                    };
                    size_t quantized_size = ggml_row_size(type, size);
                    benchmark_function(size, quantized_size, iterations, quantize_fn);
                }
                printf("\n");
            }
        }
    }

    ggml_free(ctx);

    return 0;
}
