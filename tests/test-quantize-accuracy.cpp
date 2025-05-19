// Test the accuracy of quantization and dequantization

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define MAX_ALIGNMENT 64
#define QK 32

// Data pattern types
enum DataPattern {
    PATTERN_RANDOM,     // Random values - only pattern we'll use
};

// Parameters for the test
struct quantize_accuracy_params {
    std::vector<std::string> include_types;
    size_t test_size = 1024;     // Default test size
    size_t alignment_offset = 0;
    bool verbose = false;        // Whether to print all values or just statistics
    bool csv_output = true;      // Output in CSV format
};

// Generate random data
static void generate_random_data(size_t n, float * dst) {
    // Random values between -2 and 2
    srand(42); // Fixed seed for reproducibility
    for (size_t i = 0; i < n; i++) {
        dst[i] = -2.0f + 4.0f * (rand() / (float)RAND_MAX);
    }
}

// Align memory to a specific boundary with offset
static void * align_with_offset(void * ptr, int offset) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned = (addr + MAX_ALIGNMENT - 1) & ~(MAX_ALIGNMENT - 1);
    return (void*)(aligned + offset);
}

// Calculate error metrics
static void calculate_error_metrics(const float * original, const float * reconstructed, size_t n,
                                   float & max_error, float & avg_error, float & rms_error,
                                   float & max_rel_error, float & avg_rel_error) {
    max_error = 0.0f;
    avg_error = 0.0f;
    rms_error = 0.0f;
    max_rel_error = 0.0f;
    avg_rel_error = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float error = fabsf(original[i] - reconstructed[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
        rms_error += error * error;

        // Calculate relative error (avoid division by zero)
        if (fabsf(original[i]) > 1e-6f) {
            float rel_error = error / fabsf(original[i]);
            max_rel_error = std::max(max_rel_error, rel_error);
            avg_rel_error += rel_error;
        }
    }

    avg_error /= n;
    rms_error = sqrtf(rms_error / n);
    avg_rel_error /= n;
}

// Get SNR (signal-to-noise ratio) in decibels
static float calculate_snr(const float * original, const float * reconstructed, size_t n) {
    float signal_power = 0.0f;
    float noise_power = 0.0f;

    for (size_t i = 0; i < n; i++) {
        signal_power += original[i] * original[i];
        float noise = original[i] - reconstructed[i];
        noise_power += noise * noise;
    }

    // Avoid division by zero
    if (noise_power < 1e-10f) return 100.0f; // arbitrary high value for near-zero noise

    return 10.0f * log10f(signal_power / noise_power);
}

static void usage(char * argv[]) {
    printf("Test the accuracy of quantization and dequantization with random data\n");
    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options: (default)\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --size SIZE           set test size, divisible by 32 (1024)\n");
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
    printf("  -v, --verbose         print all values\n");
    printf("  --no-csv              disable CSV output format\n");
}

static void print_csv_header() {
    printf("type,bits_per_val,compression_ratio,max_abs_error,avg_abs_error,rms_error,max_rel_error_percent,avg_rel_error_percent,snr_db\n");
}

static void run_test_for_type(ggml_type type, const float * input_data, float * quantized_data, float * output_data, size_t test_size, bool verbose, bool csv_output) {
    const auto * qfns     = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

    if (!csv_output) {
        printf("=== Testing %s ===\n", ggml_type_name(type));
    }

    // Initialize quantization for this type
    ggml_quantize_init(type);

    // Quantize using CPU implementation
    qfns_cpu->from_float(input_data, quantized_data, test_size);

    // Dequantize back to float
    qfns->to_float(quantized_data, output_data, test_size);

    // Calculate errors
    float max_error, avg_error, rms_error, max_rel_error, avg_rel_error;
    calculate_error_metrics(input_data, output_data, test_size, max_error, avg_error, rms_error, max_rel_error, avg_rel_error);

    // Calculate SNR
    float snr = calculate_snr(input_data, output_data, test_size);

    // Calculate compression ratio
    size_t float_size = test_size * sizeof(float);
    size_t quantized_size = ggml_row_size(type, test_size);
    float compression_ratio = float_size / (float)quantized_size;
    float bits_per_val = 8.0f * quantized_size / test_size;

    if (csv_output) {
        // Output in CSV format
        printf("%s,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n",
               ggml_type_name(type),
               bits_per_val,
               compression_ratio,
               max_error,
               avg_error,
               rms_error,
               max_rel_error * 100.0f,
               avg_rel_error * 100.0f,
               snr);
    } else {
        // Print error metrics in human-readable format
        printf("Max absolute error: %.6f\n", max_error);
        printf("Avg absolute error: %.6f\n", avg_error);
        printf("RMS error:          %.6f\n", rms_error);
        printf("Max relative error: %.6f%%\n", max_rel_error * 100.0f);
        printf("Avg relative error: %.6f%%\n", avg_rel_error * 100.0f);
        printf("SNR:                %.2f dB\n", snr);
        printf("Compression ratio: %.2f:1 (%.2f bits per value)\n",
              compression_ratio, bits_per_val);

        // Print the original/reconstructed values if verbose
        if (verbose) {
            printf("\nOriginal vs Reconstructed values:\n");
            for (size_t j = 0; j < std::min(test_size, size_t(20)); j++) {
                printf("[%4zu] %.6f -> %.6f (error: %.6f)\n",
                      j, input_data[j], output_data[j], fabsf(input_data[j] - output_data[j]));
            }

            // If test size is large, print the last few values
            if (test_size > 20) {
                printf("...\n");
                for (size_t j = test_size - 5; j < test_size; j++) {
                    printf("[%4zu] %.6f -> %.6f (error: %.6f)\n",
                          j, input_data[j], output_data[j], fabsf(input_data[j] - output_data[j]));
                }
            }
        }

        printf("\n");
    }
}

int main(int argc, char * argv[]) {
    quantize_accuracy_params params {};

    // Parse command line arguments
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
            params.test_size = size;
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
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--no-csv") {
            params.csv_output = false;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return 1;
    }

    // Allocate memory for test data
    std::vector<uint8_t> input_data_v(params.test_size*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> quantized_data_v(params.test_size*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> output_data_v(params.test_size*4 + MAX_ALIGNMENT*2);

    float * input_data = (float *) align_with_offset(input_data_v.data(), params.alignment_offset);
    float * quantized_data = (float *) align_with_offset(quantized_data_v.data(), params.alignment_offset);
    float * output_data = (float *) align_with_offset(output_data_v.data(), params.alignment_offset);

    // Generate random test data
    generate_random_data(params.test_size, input_data);

    // Initialize GGML context
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    if (!params.csv_output) {
        printf("Testing quantization/dequantization accuracy with %zu random values\n\n", params.test_size);
    } else {
        print_csv_header();
    }

    // Test each quantization type
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // Skip if type not included or not a quantizable type
        if (!params.include_types.empty() &&
            ggml_type_name(type) &&
            std::find(params.include_types.begin(), params.include_types.end(), ggml_type_name(type)) == params.include_types.end()) {
            // printf("skipping %s due to NOT in include_types.\n", ggml_type_name(type));
            continue;
        }

        if (qfns_cpu->from_float && qfns->to_float) {
            run_test_for_type(type, input_data, quantized_data, output_data, params.test_size, params.verbose, params.csv_output);
        } else {
            // printf("skipping %s due to NO to_float.\n", ggml_type_name(type));
        }
    }

    ggml_free(ctx);

    return 0;
}
