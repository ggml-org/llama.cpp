// This file defines tests for various GGML ops and backends.
// For the forward pass it asserts that the results of multiple backends computing the same GGML ops are consistent.
// For the backward pass it asserts that the gradients from backpropagation are consistent
// with the gradients obtained via the method of finite differences ("grad" mode, this is optional).
// It is also possible to check the performance ("perf" mode).
//
// this file has three sections: Section 1 does general setup, section 2 defines the GGML ops to be tested,
// and section 3 defines which tests to run.
// Quick start for adding a new GGML op: Go to section 2 and create a struct that inherits from test_case,
// then go to section 3 and add an instantiation of your struct.


// ##############################
// ## Section 1: General Setup ##
// ##############################


#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <future>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = std::thread::hardware_concurrency();
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto & gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        std::vector<std::future<void>> tasks;
        tasks.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) {
            size_t start =     i*nels/n_threads;
            size_t end   = (i+1)*nels/n_threads;
            tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
        }
        for (auto & t : tasks) {
            t.get();
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

         // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
                    start * blck_size, end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency()/2,
                                                      std::max<size_t>(1, n_blocks / min_blocks_per_thread));
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start =     i*n_blocks/n_threads;
                size_t end   = (i+1)*n_blocks/n_threads;
                tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = ggml_nbytes(tensor)/2;
        ggml_backend_tensor_set(tensor, data.data(), 0*nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1*nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

// generate an F16 mask where certain blocks are randomly masked with -INF value
static void init_tensor_kq_mask(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS( int32_t, ne, tensor, ne);

    std::vector<float>       data_f32(ne0*ne1*ne2*ne3);
    std::vector<ggml_fp16_t> data_f16(ne0*ne1*ne2*ne3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (size_t i = 0; i < data_f32.size(); i++) {
        data_f32[i] = dis(gen);
    }

    // block size
    const int blck0 = 128;
    const int blck1 = 64;

    // number of INF blocks
    const int n_inf_blocks = 0.1*(ne0*ne1*ne2*ne3)/(blck0*blck1);

    for (int b = 0; b < n_inf_blocks; b++) {
        const int p3 = (rd() % ne3);
        const int p2 = (rd() % ne2);
        const int p1 = (rd() % ne1);
        const int p0 = (rd() % ne0);

        for (int i1 = 0; i1 < blck1 && p1 + i1 < ne1; i1++) {
            const int idx = p3*ne2*ne1*ne0 + p2*ne1*ne0 + (p1 + i1)*ne0 + p0;

            for (int i0 = 0; i0 < blck0 && p0 + i0 < ne0; i0++) {
                data_f32[idx + i0] = -INFINITY;
            }
        }
    }

    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), ne0*ne1*ne2*ne3);

    ggml_backend_tensor_set(tensor, data_f16.data(), 0, data_f16.size()*sizeof(ggml_fp16_t));
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const auto * tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float)*(int64_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the numerical calculation can fail.
static double mean_abs_asymm(const float * a, const float * b, const size_t n, const std::vector<float> & expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float & ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum/nvalid;
}

// utils for printing the variables of the test cases

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_prec prec) {
    return prec == GGML_PREC_F32 ? "f32" : "def";
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

static std::string var_to_str(ggml_scale_mode mode) {
    switch (mode) {
        case GGML_SCALE_MODE_NEAREST:  return "nearest";
        case GGML_SCALE_MODE_BILINEAR: return "bilinear";
        default:                      return std::to_string(mode);
    }
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)
#define VARS_TO_STR13(a, b, c, d, e, f, g, h, i, j, k, l, m) VAR_TO_STR(a) + "," + VARS_TO_STR12(b, c, d, e, f, g, h, i, j, k, l, m)
#define VARS_TO_STR14(a, b, c, d, e, f, g, h, i, j, k, l, m, n) VAR_TO_STR(a) + "," + VARS_TO_STR13(b, c, d, e, f, g, h, i, j, k, l, m, n)
#define VARS_TO_STR15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) VAR_TO_STR(a) + "," + VARS_TO_STR14(b, c, d, e, f, g, h, i, j, k, l, m, n, o)
#define VARS_TO_STR16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) VAR_TO_STR(a) + "," + VARS_TO_STR15(b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
    MODE_SUPPORT,
};

// Output format support similar to llama-bench
enum output_formats { CONSOLE, SQL, CSV };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case CONSOLE:
            return "console";
        case SQL:
            return "sql";
        case CSV:
            return "csv";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "console") {
        format = CONSOLE;
    } else if (s == "sql") {
        format = SQL;
    } else if (s == "csv") {
        format = CSV;
    } else {
        return false;
    }
    return true;
}

// Test result structure for SQL output
struct test_result {
    std::string test_time;
    std::string build_commit;
    std::string backend_name;
    std::string op_name;
    std::string op_params;
    std::string test_mode;
    bool        supported;
    bool        passed;
    std::string error_message;
    double      time_us;
    double      flops;
    double      bandwidth_gb_s;
    size_t      memory_kb;
    int         n_runs;
    std::string device_description;
    std::string backend_reg_name;

    test_result() {
        // Initialize with default values
        time_us        = 0.0;
        flops          = 0.0;
        bandwidth_gb_s = 0.0;
        memory_kb      = 0;
        n_runs         = 0;
        supported      = false;
        passed         = false;

        // Set test time
        time_t t = time(NULL);
        char   buf[32];
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // Set build info
        build_commit = ggml_commit();
    }

    test_result(const std::string & backend_name, const std::string & op_name, const std::string & op_params,
                const std::string & test_mode, bool supported, bool passed, const std::string & error_message = "",
                double time_us = 0.0, double flops = 0.0, double bandwidth_gb_s = 0.0, size_t memory_kb = 0,
                int n_runs = 0, const std::string & device_description = "", const std::string & backend_reg_name = "") :
        backend_name(backend_name),
        op_name(op_name),
        op_params(op_params),
        test_mode(test_mode),
        supported(supported),
        passed(passed),
        error_message(error_message),
        time_us(time_us),
        flops(flops),
        bandwidth_gb_s(bandwidth_gb_s),
        memory_kb(memory_kb),
        n_runs(n_runs),
        device_description(device_description),
        backend_reg_name(backend_reg_name) {
        // Set test time
        time_t t = time(NULL);
        char   buf[32];
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // Set build info
        build_commit = ggml_commit();
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "test_time", "build_commit",  "backend_name", "op_name", "op_params",      "test_mode", "supported",
            "passed",    "error_message", "time_us",      "flops",   "bandwidth_gb_s", "memory_kb", "n_runs",
            "device_description", "backend_reg_name"
        };
        return fields;
    }

    enum field_type { STRING, BOOL, INT, FLOAT };

    static field_type get_field_type(const std::string & field) {
        if (field == "supported" || field == "passed") {
            return BOOL;
        }
        if (field == "memory_kb" || field == "n_runs") {
            return INT;
        }
        if (field == "time_us" || field == "flops" || field == "bandwidth_gb_s") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        return { test_time,
                 build_commit,
                 backend_name,
                 op_name,
                 op_params,
                 test_mode,
                 std::to_string(supported),
                 std::to_string(passed),
                 error_message,
                 std::to_string(time_us),
                 std::to_string(flops),
                 std::to_string(bandwidth_gb_s),
                 std::to_string(memory_kb),
                 std::to_string(n_runs),
                 device_description,
                 backend_reg_name };
    }
};

// Printer classes for different output formats
enum class test_status_t { NOT_SUPPORTED, OK, FAIL, SKIPPED };

struct test_operation_info {
    std::string   op_name;
    std::string   op_params;
    std::string   backend_name;
    test_status_t status = test_status_t::OK;
    std::string   failure_reason;

    // Additional information fields that were previously in separate structs
    std::string error_component;
    std::string error_details;

    // Gradient info
    int64_t     gradient_index = -1;
    std::string gradient_param_name;
    float       gradient_value = 0.0f;

    // MAA error info
    double maa_error     = 0.0;
    double maa_threshold = 0.0;

    // Flags for different types of information
    bool has_error            = false;
    bool has_gradient_info    = false;
    bool has_maa_error        = false;
    bool is_compare_failure   = false;
    bool is_large_tensor_skip = false;

    test_operation_info() = default;

    test_operation_info(const std::string & op_name, const std::string & op_params, const std::string & backend_name,
                        test_status_t status = test_status_t::OK, const std::string & failure_reason = "") :
        op_name(op_name),
        op_params(op_params),
        backend_name(backend_name),
        status(status),
        failure_reason(failure_reason) {}

    // Set error information
    void set_error(const std::string & component, const std::string & details) {
        has_error       = true;
        error_component = component;
        error_details   = details;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set gradient information
    void set_gradient_info(int64_t index, const std::string & param_name, float value) {
        has_gradient_info   = true;
        gradient_index      = index;
        gradient_param_name = param_name;
        gradient_value      = value;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set MAA error information
    void set_maa_error(double error, double threshold) {
        has_maa_error = true;
        maa_error     = error;
        maa_threshold = threshold;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set compare failure
    void set_compare_failure() {
        is_compare_failure = true;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set large tensor skip
    void set_large_tensor_skip() { is_large_tensor_skip = true; }
};

struct test_summary_info {
    size_t tests_passed;
    size_t tests_total;
    bool   is_backend_summary = false;  // true for backend summary, false for test summary

    test_summary_info() = default;

    test_summary_info(size_t tests_passed, size_t tests_total, bool is_backend_summary = false) :
        tests_passed(tests_passed),
        tests_total(tests_total),
        is_backend_summary(is_backend_summary) {}
};

struct testing_start_info {
    size_t device_count;

    testing_start_info() = default;

    testing_start_info(size_t device_count) : device_count(device_count) {}
};

struct backend_init_info {
    size_t      device_index;
    size_t      total_devices;
    std::string device_name;
    bool        skipped = false;
    std::string skip_reason;
    std::string description;
    size_t      memory_total_mb = 0;
    size_t      memory_free_mb  = 0;
    bool        has_memory_info = false;

    backend_init_info() = default;

    backend_init_info(size_t device_index, size_t total_devices, const std::string & device_name, bool skipped = false,
                      const std::string & skip_reason = "", const std::string & description = "",
                      size_t memory_total_mb = 0, size_t memory_free_mb = 0, bool has_memory_info = false) :
        device_index(device_index),
        total_devices(total_devices),
        device_name(device_name),
        skipped(skipped),
        skip_reason(skip_reason),
        description(description),
        memory_total_mb(memory_total_mb),
        memory_free_mb(memory_free_mb),
        has_memory_info(has_memory_info) {}
};

struct backend_status_info {
    std::string   backend_name;
    test_status_t status;

    backend_status_info() = default;

    backend_status_info(const std::string & backend_name, test_status_t status) :
        backend_name(backend_name),
        status(status) {}
};

struct overall_summary_info {
    size_t backends_passed;
    size_t backends_total;
    bool   all_passed;

    overall_summary_info() = default;

    overall_summary_info(size_t backends_passed, size_t backends_total, bool all_passed) :
        backends_passed(backends_passed),
        backends_total(backends_total),
        all_passed(all_passed) {}
};

struct printer {
    virtual ~printer() {}

    FILE * fout = stdout;

    virtual void print_header() {}

    virtual void print_test_result(const test_result & result) = 0;

    virtual void print_footer() {}

    virtual void print_operation(const test_operation_info & info) { (void) info; }

    virtual void print_summary(const test_summary_info & info) { (void) info; }

    virtual void print_testing_start(const testing_start_info & info) { (void) info; }

    virtual void print_backend_init(const backend_init_info & info) { (void) info; }

    virtual void print_backend_status(const backend_status_info & info) { (void) info; }

    virtual void print_overall_summary(const overall_summary_info & info) { (void) info; }

    virtual void print_failed_tests(const std::vector<std::string> & failed_tests) { (void) failed_tests; }
};

struct console_printer : public printer {
    void print_test_result(const test_result & result) override {
        if (result.test_mode == "test") {
            print_test_console(result);
        } else if (result.test_mode == "perf") {
            print_perf_console(result);
        } else if (result.test_mode == "support") {
            print_support_console(result);
        }
    }

    void print_operation(const test_operation_info & info) override {
        printf("  %s(%s): ", info.op_name.c_str(), info.op_params.c_str());
        fflush(stdout);

        // Handle large tensor skip first
        if (info.is_large_tensor_skip) {
            printf("skipping large tensors for speed \n");
            return;
        }

        // Handle not supported status
        if (info.status == test_status_t::NOT_SUPPORTED) {
            if (!info.failure_reason.empty()) {
                printf("not supported [%s]\n", info.failure_reason.c_str());
            } else {
                printf("not supported [%s]\n", info.backend_name.c_str());
            }
            return;
        }

        // Handle errors and additional information
        if (info.has_error) {
            if (info.error_component == "allocation") {
                fprintf(stderr, "failed to allocate tensors [%s] ", info.backend_name.c_str());
            } else if (info.error_component == "backend") {
                fprintf(stderr, "  Failed to initialize %s backend\n", info.backend_name.c_str());
            } else {
                fprintf(stderr, "Error in %s: %s\n", info.error_component.c_str(), info.error_details.c_str());
            }
        }

        // Handle gradient info
        if (info.has_gradient_info) {
            printf("[%s] nonfinite gradient at index %" PRId64 " (%s=%f) ", info.op_name.c_str(), info.gradient_index,
                   info.gradient_param_name.c_str(), info.gradient_value);
        }

        // Handle MAA error
        if (info.has_maa_error) {
            printf("[%s] MAA = %.9f > %.9f ", info.op_name.c_str(), info.maa_error, info.maa_threshold);
        }

        // Handle compare failure
        if (info.is_compare_failure) {
            printf("compare failed ");
        }

        // Print final status
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_summary(const test_summary_info & info) override {
        if (info.is_backend_summary) {
            printf("%zu/%zu backends passed\n", info.tests_passed, info.tests_total);
        } else {
            printf("  %zu/%zu tests passed\n", info.tests_passed, info.tests_total);
        }
    }

    void print_backend_status(const backend_status_info & info) override {
        printf("  Backend %s: ", info.backend_name.c_str());
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_testing_start(const testing_start_info & info) override {
        printf("Testing %zu devices\n\n", info.device_count);
    }

    void print_backend_init(const backend_init_info & info) override {
        printf("Backend %zu/%zu: %s\n", info.device_index + 1, info.total_devices, info.device_name.c_str());

        if (info.skipped) {
            printf("  %s\n", info.skip_reason.c_str());
            return;
        }

        if (!info.description.empty()) {
            printf("  Device description: %s\n", info.description.c_str());
        }

        if (info.has_memory_info) {
            printf("  Device memory: %zu MB (%zu MB free)\n", info.memory_total_mb, info.memory_free_mb);
        }

        printf("\n");
    }

    void print_overall_summary(const overall_summary_info & info) override {
        printf("%zu/%zu backends passed\n", info.backends_passed, info.backends_total);
        if (info.all_passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_failed_tests(const std::vector<std::string> & failed_tests) override {
        if (failed_tests.empty()) {
            return;
        }

        printf("\nFailing tests:\n");
        for (const auto & test_name : failed_tests) {
            printf("  %s\n", test_name.c_str());
        }
    }

  private:
    void print_test_console(const test_result & result) {
        printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported [%s] ", result.backend_name.c_str());
            printf("\n");
            return;
        }

        if (result.passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_perf_console(const test_result & result) {
        int len = printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported\n");
            return;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last  = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        printf("    %8d runs - %8.2f us/run - ", result.n_runs, result.time_us);

        if (result.flops > 0) {
            auto format_flops = [](double flops) -> std::string {
                char buf[256];
                if (flops >= 1e12) {
                    snprintf(buf, sizeof(buf), "%6.2f TFLOP", flops / 1e12);
                } else if (flops >= 1e9) {
                    snprintf(buf, sizeof(buf), "%6.2f GFLOP", flops / 1e9);
                } else if (flops >= 1e6) {
                    snprintf(buf, sizeof(buf), "%6.2f MFLOP", flops / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%6.2f kFLOP", flops / 1e3);
                }
                return buf;
            };
            uint64_t op_flops_per_run = result.flops * result.time_us / 1e6;
            printf("%s/run - \033[1;34m%sS\033[0m", format_flops(op_flops_per_run).c_str(),
                   format_flops(result.flops).c_str());
        } else {
            printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m", result.memory_kb, result.bandwidth_gb_s);
        }
        printf("\n");
    }

    void print_support_console(const test_result & result) {
        printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (result.supported) {
            printf("\033[1;32mSUPPORTED\033[0m\n");
        } else {
            printf("\033[1;31mNOT SUPPORTED\033[0m\n");
        }
    }
};

struct sql_printer : public printer {
    static std::string get_sql_field_type(const std::string & field) {
        switch (test_result::get_field_type(field)) {
            case test_result::STRING:
                return "TEXT";
            case test_result::BOOL:
            case test_result::INT:
                return "INTEGER";
            case test_result::FLOAT:
                return "REAL";
            default:
                GGML_ABORT("invalid field type");
        }
    }

    void print_header() override {
        std::vector<std::string> fields = test_result::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS test_backend_ops (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields[i].c_str(), get_sql_field_type(fields[i]).c_str(),
                    i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n\n");
    }

    void print_test_result(const test_result & result) override {
        fprintf(fout, "INSERT INTO test_backend_ops (");
        std::vector<std::string> fields = test_result::get_fields();
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "%s%s", fields[i].c_str(), i < fields.size() - 1 ? ", " : "");
        }
        fprintf(fout, ") VALUES (");
        std::vector<std::string> values = result.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values[i].c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

struct csv_printer : public printer {
    void print_header() override {

        std::vector<std::string> fields     = test_result::get_fields();
        std::vector<std::string> fields_csv = get_fields_csv();
        for (size_t i = 0; i < fields.size(); i++) {
            if (std::find(std::begin(fields_csv), std::end(fields_csv), fields[i]) == std::end(fields_csv)) {
                continue;
            }
            printf("\"%s\"%s", fields[i].c_str(), i < fields.size() - 1 ? "," : "");
        }
        printf("\n");
    }

    void print_test_result(const test_result & result) override {

        std::vector<std::string> values     = result.get_values();
        std::vector<std::string> fields     = test_result::get_fields();
        std::vector<std::string> fields_csv = get_fields_csv();

        for (size_t i = 0; i < values.size(); i++) {

            if (std::find(std::begin(fields_csv), std::end(fields_csv), fields[i]) == std::end(fields_csv)) {
                continue;
            }

            // Escape quotes and wrap in quotes for CSV
            std::string escaped_value = values[i];
            size_t pos = 0;
            while ((pos = escaped_value.find("\"", pos)) != std::string::npos) {
                escaped_value.replace(pos, 1, "\"\"");
                pos += 2;
            }
            printf("\"%s\"%s", escaped_value.c_str(), i < values.size() - 1 ? "," : "");
        }
        printf("\n");
    }

    static std::vector<std::string> get_fields_csv() {
        return {
            "op_name",
            "op_params",
            "supported",
            "error_message",
            "test_mode",
            "backend_reg_name",
            "backend_name",
        };
    }

};

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case CONSOLE:
            return std::make_unique<console_printer>();
        case SQL:
            return std::make_unique<sql_printer>();
        case CSV:
            return std::make_unique<csv_printer>();
    }
    GGML_ABORT("invalid output format");
}

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual double max_maa_err() {
        return 1e-4;
    }

    virtual float grad_eps() {
        return 1e-1f;
    }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() {
        return false;
    }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed up the tests).
    virtual int64_t grad_nmax() {
        return 10000;
    }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the values.
    // Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using finite differences is unreliable.
    virtual std::vector<float> grad_expect() {
        return {};
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor * t) {
        GGML_UNUSED(t);
        return 0;
    }

    virtual bool run_whole_graph() { return false; }

    ggml_cgraph * gf = nullptr;
    ggml_cgraph * gb = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    std::string current_op_name;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF || mode == MODE_GRAD || mode == MODE_SUPPORT) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    // Checks an op against the test filter, which is a comma separated list of OP names or specific variations
    bool matches_filter(ggml_tensor * op, const char * op_names_filter) {
        if (op_names_filter) {
            const auto op_name = op_desc(op);
            const auto op_full_name = op_name + "(" + vars() + ")";
            std::string_view filter(op_names_filter);
            while (!filter.empty()) {
                auto comma_pos = filter.find_first_of(',');
                const auto lparen_pos = filter.find_first_of('(');
                if (lparen_pos < comma_pos) {
                    auto rparen_pos = filter.find_first_of(')');
                    comma_pos = filter.find_first_of(',', rparen_pos);
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_full_name) {
                        return true;
                    }
                } else {
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_name) {
                        return true;
                    }
                }
                filter = comma_pos != std::string_view::npos ? filter.substr(comma_pos + 1) : "";
            }
            return false;
        } else {
            return true;
        }
    }

    test_status_t eval(ggml_backend_t backend1,
                       ggml_backend_t backend2,
                       const char *   op_names_filter,
                       printer *      output_printer) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);
        current_op_name   = op_desc(out);

        if (!matches_filter(out, op_names_filter)) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return test_status_t::SKIPPED;
        }

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    supported = false;
                    break;
                }
            }
        }

        if (!supported) {
            // Create test result for unsupported operation
            test_result result(ggml_backend_name(backend1), current_op_name, vars(), "test",
                             false, false, "not supported");

            if (output_printer) {
                output_printer->print_test_result(result);
            }

            ggml_free(ctx);
            return test_status_t::NOT_SUPPORTED;
        }

        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);

        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return test_status_t::FAIL;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
            backend1,
            backend2
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, run_whole_graph() ? out : nullptr);

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        // Create test result
        bool        test_passed = ud.ok && cmp_ok;
        std::string error_msg   = test_passed ? "" : (!cmp_ok ? "compare failed" : "test failed");
        test_result result(ggml_backend_name(backend1), current_op_name, vars(), "test", supported, test_passed,
                           error_msg);

        if (output_printer) {
            output_printer->print_test_result(result);
        }

        return test_passed ? test_status_t::OK : test_status_t::FAIL;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        ggml_tensor * out             = build_graph(ctx.get());
        current_op_name               = op_desc(out);
        if (!matches_filter(out, op_names_filter)) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            return true;
        }

        if (!ggml_backend_supports_op(backend, out)) {
            // Create test result for unsupported performance test
            test_result result(ggml_backend_name(backend), current_op_name, vars(), "perf", false, false,
                               "not supported");

            output_printer->print_test_result(result);

            return true;
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr

        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx.get());

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return false;
        }

        // determine number of runs
        int n_runs;
        bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (op_flops(out) > 0) {
            // based on flops
            const uint64_t GFLOP = 1000 * 1000 * 1000;
            const uint64_t target_flops_cpu =   8ULL * GFLOP;
            const uint64_t target_flops_gpu = 100ULL * GFLOP;
            uint64_t target_flops = is_cpu ? target_flops_cpu : target_flops_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_flops / op_flops(out)) + 1;
        } else {
            // based on memory size
            const size_t GB = 1ULL << 30;
            const size_t target_size_cpu =  8 * GB;
            const size_t target_size_gpu = 32 * GB;
            size_t target_size = is_cpu ? target_size_cpu : target_size_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_size / op_size(out)) + 1;
        }

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_is_view_op(ggml_graph_node(gf, i)->op) || ggml_graph_node(gf, i) == out) {
                continue;
            }
            mem += tensor_op_size(ggml_graph_node(gf, i));
        }

        // run
        int64_t total_time_us = 0;
        int64_t total_mem = 0;
        int total_runs = 0;
        do {
            int64_t start_time = ggml_time_us();
            ggml_status status = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                return false;
            }
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_mem += mem;
            total_runs += n_runs;
        } while (total_time_us < 1000*1000); // run for at least 1 second

        // Create test result
        double avg_time_us      = (double) total_time_us / total_runs;
        double calculated_flops = (op_flops(out) > 0) ? (op_flops(out) * total_runs) / (total_time_us / 1e6) : 0.0;
        double calculated_bandwidth =
            (op_flops(out) == 0) ? total_mem / (total_time_us / 1e6) / 1024.0 / 1024.0 / 1024.0 : 0.0;
        size_t calculated_memory_kb = op_size(out) / 1024;

        test_result result(ggml_backend_name(backend), current_op_name, vars(), "perf", true, true, "", avg_time_us,
                           calculated_flops, calculated_bandwidth, calculated_memory_kb, total_runs);

        if (output_printer) {
            output_printer->print_test_result(result);
        }

        return true;
    }

    bool eval_support(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_SUPPORT;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        ggml_tensor * out = build_graph(ctx.get());
        current_op_name   = op_desc(out);

        if (!matches_filter(out, op_names_filter)) {
            return true;
        }

        bool supported = ggml_backend_supports_op(backend, out);

        std::string device_desc = ggml_backend_dev_description(ggml_backend_get_device(backend));
        std::string backend_reg_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)));

        test_result result(ggml_backend_name(backend), current_op_name, vars(), "support", supported, supported,
                           supported ? "yes" : "no", 0.0, 0.0, 0.0, 0, 0, device_desc, backend_reg_name);

        output_printer->print_test_result(result);

        return true;
    }

    bool eval_grad(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_GRAD;
        const std::vector<float> expect = grad_expect();

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + 2*ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);
        gb = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);

        ggml_tensor * out = build_graph(ctx.get());

        if (!matches_filter(out, op_names_filter) || out->op == GGML_OP_OPT_STEP_ADAMW) {
            return true;
        }

        if (out->type != GGML_TYPE_F32) {
            output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                test_status_t::NOT_SUPPORTED,
                                                                out->name + std::string("->type != FP32")));
            return true;
        }

        // Print operation info first
        output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend)));

        // check if the backend supports the ops
        bool        supported  = true;
        bool        any_params = false;
        std::string failure_reason;

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                supported      = false;
                failure_reason = ggml_backend_name(backend);
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM)) {
                any_params = true;
                if (t->type != GGML_TYPE_F32) {
                    supported      = false;
                    failure_reason = std::string(t->name) + "->type != FP32";
                    break;
                }
            }
        }
        if (!any_params) {
            supported      = false;
            failure_reason = op_desc(out);
        }

        if (!supported) {
            output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                test_status_t::NOT_SUPPORTED, failure_reason));
            return true;
        }

        int64_t ngrads = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) {
                ngrads += ggml_nelements(t);
            }
        }
        if (ngrads > grad_nmax()) {
            test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
            info.set_large_tensor_skip();
            output_printer->print_operation(info);
            return true;
        }


        if (!ggml_is_scalar(out)) {
            out = ggml_sum(ctx.get(), out);
            ggml_set_name(out, "sum_of_out");
        }
        ggml_set_loss(out);

        ggml_build_forward_expand(gf, out);
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx.get(), gb, nullptr);
        if (expect.size() != 1 || expect[0] != 0.0f) {
            GGML_ASSERT(ggml_graph_n_nodes(gb) > ggml_graph_n_nodes(gf));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
                GGML_ASSERT(!(t->flags & GGML_TENSOR_FLAG_PARAM) || ggml_graph_get_grad(gb, t)->op != GGML_OP_NONE);
            }
        }

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                    test_status_t::NOT_SUPPORTED,
                                                                    ggml_backend_name(backend)));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM) && t->type != GGML_TYPE_F32) {
                output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                    test_status_t::NOT_SUPPORTED,
                                                                    std::string(t->name) + "->type != FP32"));
                supported = false;
                break;
            }
        }
        if (!supported) {
            return true;
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr
        if (buf == NULL) {
            test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
            info.set_error("allocation", "");
            output_printer->print_operation(info);
            return false;
        }

        initialize_tensors(ctx.get()); // Randomizes all tensors (including gradients).
        ggml_graph_reset(gb);    // Sets gradients to 1 if loss, 0 otherwise.

        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return false;
        }
        status = ggml_backend_graph_compute(backend, gb);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return false;
        }

        bool ok = true;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!(t->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            const char * bn = ggml_backend_name(backend);
            const int64_t ne = ggml_nelements(t);

            std::vector<float> ga;
            struct ggml_tensor * grad = ggml_graph_get_grad(gb, t);
            if (grad) {
                ga = tensor_to_float(grad);
            } else {
                ga.resize(ne); // default value is 0.0f
            }

            for (int64_t i = 0; i < ne; ++i) { // gradient algebraic
                // check for nans
                if (!std::isfinite(ga[i])) {
                    test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
                    info.set_gradient_info(i, bn, ga[i]);
                    output_printer->print_operation(info);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }

            std::vector<float> gn(ne); // gradient numeric
            GGML_ASSERT(ga.size() == gn.size());

            std::vector<float> x0 = tensor_to_float(t); // original t data
            GGML_ASSERT(ggml_is_scalar(out));
            GGML_ASSERT(out->type == GGML_TYPE_F32);

            const float eps = grad_eps();
            for (int64_t i = 0; i < ne; ++i) {
                const float xiu  = x0[i] + 1.0f*eps; // x, index i, up
                const float xiuh = x0[i] + 0.5f*eps; // x, index i, up half
                const float xidh = x0[i] - 0.5f*eps; // x, index i, down half
                const float xid  = x0[i] - 1.0f*eps; // x, index i, down

                float fu, fuh, fdh, fd; // output values for xiu, xiuh, xid, xidh

                ggml_backend_tensor_set(t, &xiu, i*sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fu, 0, ggml_nbytes(out));

                ggml_backend_tensor_set(t, &xid, i*sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fd, 0, ggml_nbytes(out));

                if (grad_precise()) {
                    ggml_backend_tensor_set(t, &xiuh, i*sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fuh, 0, ggml_nbytes(out));

                    ggml_backend_tensor_set(t, &xidh, i*sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fdh, 0, ggml_nbytes(out));

                    gn[i] = (8.0*(double)fuh + (double)fd - (8.0*(double)fdh + (double)fu)) / (6.0*(double)eps);
                } else {
                    gn[i] = (fu - fd) / (2.0f*eps);
                }

                ggml_backend_tensor_set(t, x0.data(), 0, ggml_nbytes(t));
            }

            const double err = mean_abs_asymm(gn.data(), ga.data(), gn.size(), expect);
            if (err > max_maa_err()) {
                test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
                info.set_maa_error(err, max_maa_err());
                output_printer->print_operation(info);
                ok = false;
                break;
            }
            if (!ok) {
                break;
            }
        }

        // Create final test result
        test_operation_info final_info(op_desc(out), vars(), ggml_backend_name(backend));
        if (!ok) {
            final_info.set_compare_failure();
        }
        final_info.status = ok ? test_status_t::OK : test_status_t::FAIL;
        output_printer->print_operation(final_info);

        if (ok) {
            return true;
        }

        return false;
    }
};


// ###################################
// ## Section 2: GGML Op Defintions ##
// ###################################


// The following is an example showing the bare minimum for creating a test for a GGML op.

// GGML_OP_EXAMPLE
struct test_example : public test_case {
    // Always define these 2 or variants thereof:
    const ggml_type type; // The type of the input tensors.
    const std::array<int64_t, 4> ne; // The shape of the input tensors.
    // For some ops it's necessary to define multiple types or shapes for the inputs.
    // Or they may need additional parameters.

    // Put all parameters needed to fully define the test into one of the VARS_TO_STR macros.
    // In most cases these are just the properties of the struct that you defined above.
    // This is needed for info prints.
    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    // Define a constructor for the struct.
    // In most cases it will be sufficient to have the same arguments as the struct has properties
    // and just use initializer lists.
    test_example(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    // Define how a simple GGML compute graph can be constructed for the new GGML op.
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Step 1: create input tensors that don't depend on any other tensors:
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a"); // Setting names is optional but it's useful for debugging.

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        // Step 2: use the op that you want to test in the GGML compute graph.
        ggml_tensor * out = ggml_add(ctx, a, b); // For this example we're just doing a simple addition.
        ggml_set_name(out, "out");

        // Step 3: return the output tensor.
        return out;
    }
    // In order to also check the gradients for your op, add calls like ggml_set_param(a)
    // immediately after you create the tensors.
    // This is optional and only makes sense if a backward pass has actually been implemented for the new op.
};


// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, v);
    }

    test_unary(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
            int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const bool grad_supported = op == GGML_UNARY_OP_ABS || op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_NEG ||
            op == GGML_UNARY_OP_STEP || op == GGML_UNARY_OP_RELU || op == GGML_UNARY_OP_SILU;

        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (grad_supported) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (grad_supported) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");
        }

        ggml_tensor * out = ggml_unary(ctx, a, op);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }

    float grad_eps() override {
        return 15.0f;
    }

    std::vector<float> grad_expect() override {
        if (op == GGML_UNARY_OP_ABS) {
            return {-1.0f, 1.0f};
        }
        if (op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_STEP) {
            return {0.0f};
        }
        if (op == GGML_UNARY_OP_RELU) {
            return {0.0f, 1.0f};
        }
        return {};
    }

};

// GGML_OP_GLU
struct test_glu : public test_case {
    const ggml_glu_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)
    bool swapped;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, v, swapped);
    }

    test_glu(ggml_glu_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
            int v = 0,
            bool swapped = false)
        : op(op), type(type), ne_a(ne_a), v(v), swapped(swapped) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_name(a, "a");
        }

        ggml_tensor * out = ggml_glu(ctx, a, op, swapped);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

struct test_glu_split : public test_case {
    const ggml_glu_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, v) + ",split";
    }

    test_glu_split(ggml_glu_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
            int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        ggml_tensor * b;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");

            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_a[0], ne_a[1], ne_a[2], ne_a[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(a, "view_of_b");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            b = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_glu_split(ctx, a, b, op);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

struct test_swiglu_oai : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)
    float alpha;
    float limit;

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, v, alpha, limit);
    }

    test_swiglu_oai(ggml_type type = GGML_TYPE_F32,
                    std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
                    int v = 0,
                    float alpha = 1.702f,
                    float limit = 7.0f)
        : type(type), ne_a(ne_a), v(v), alpha(alpha), limit(limit) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        ggml_tensor * b;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");

            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_a[0], ne_a[1], ne_a[2], ne_a[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(a, "view_of_b");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            b = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_swiglu_oai(ctx, a, b, alpha, limit);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int be1; // batch size
    const int be2; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR7(type, n, m, r, be1, be2, v);
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int be1 = 1, int be2 = 1, bool v = false)
        : type(type), n(n), m(m), r(r), be1(be1), be2(be2), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_4d(ctx, type, n, m, be1, be2);
        ggml_set_name(in, "in");

        ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_3d(ctx, rows, r/2, be1, be2, rows->nb[1], rows->nb[2], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        const bool grad_supported = ggml_is_matrix(in) && ggml_is_vector(rows);
        if (grad_supported) {
            ggml_set_param(in);
            // rows is a constant input -> no gradients
        }

        ggml_tensor * out = ggml_get_rows(ctx, in, rows);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*be1*be2);
                for (int i = 0; i < r*be1*be2; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * be1 * be2 * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_GET_ROWS_BACK
struct test_get_rows_back : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR6(type, n, m, r, b, v);
    }

    test_get_rows_back(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in_forward = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_set_name(in_forward, "in_forward");

        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_2d(ctx, rows, r/2, b, rows->nb[1], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        ggml_tensor * grad = ggml_new_tensor_3d(ctx, type, n, r, b);
        ggml_set_name(grad, "grad");

        ggml_tensor * out = ggml_get_rows_back(ctx, grad, rows, in_forward);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*b);
                for (int i = 0; i < r*b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SET_ROWS
struct test_set_rows : public test_case {
    const ggml_type type;
    const ggml_type type_idx;
    const std::array<int64_t, 4> ne;
    const std::array<int, 2> nr23; // broadcast only dims 2 and 3
    const int r; // rows to set
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR6(type, type_idx, ne, nr23, r, v);
    }

    test_set_rows(ggml_type type,
            ggml_type type_idx,
            std::array<int64_t, 4> ne,
            std::array<int, 2> nr23,
            int r, bool v = false)
        : type(type), type_idx(type_idx), ne(ne), nr23(nr23), r(r), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * dst = ggml_new_tensor_4d(ctx, type,          ne[0], ne[1], ne[2]*nr23[0], ne[3]*nr23[1]);
        ggml_set_name(dst, "dst");

        ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], r,     ne[2]*nr23[0], ne[3]*nr23[1]);
        ggml_set_name(src, "src");

        ggml_tensor * row_idxs = ggml_new_tensor_3d(ctx, type_idx, r, ne[2], ne[3]);
        ggml_set_name(row_idxs, "row_idxs");

        if (v) {
            src      = ggml_view_4d(ctx, src, ne[0], r/2, ne[2]*nr23[0], ne[3]*nr23[1], src->nb[1], src->nb[2], src->nb[3], 0);
            row_idxs = ggml_view_3d(ctx, row_idxs, r/2, ne[2], ne[3], row_idxs->nb[1], row_idxs->nb[2], 0);
            ggml_set_name(row_idxs, "view_of_rows");
        }

        ggml_tensor * out = ggml_set_rows(ctx, dst, src, row_idxs);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I64 || t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) {
                    continue;
                }

                for (int i2 = 0; i2 < t->ne[2]; i2++) {
                    for (int i1 = 0; i1 < t->ne[1]; i1++) {
                        // generate a shuffled subset of row indices
                        std::vector<int64_t> data(ne[1]);
                        for (int i = 0; i < ne[1]; i++) {
                            data[i] = i;
                        }
                        std::shuffle(data.begin(), data.end(), rng);
                        data.resize(t->ne[0]);

                        const size_t offs = i1*t->nb[1] + i2*t->nb[2];
                        if (t->type == GGML_TYPE_I32) {
                            // TODO: Make a template or something
                            std::vector<int32_t> data_i32(t->ne[0]);
                            for (int i = 0; i < t->ne[0]; i++) {
                                data_i32[i] = static_cast<int32_t>(data[i]);
                            }
                            ggml_backend_tensor_set(t, data_i32.data(), offs, t->ne[0]*sizeof(int32_t));
                        } else {
                            ggml_backend_tensor_set(t, data.data(), offs, t->ne[0]*sizeof(int64_t));
                        }
                    }
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override {
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_IQ4_NL ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0) {
            // estimate what the max nmse error would be if one quantized value is
            // off by one. The test values are distributed in [-1,1], so it'll be
            // roughly (2.0 / 2^bits)^2, divided by the mean square value of the reference,
            // which is roughly 0.25 times the number of elements.
            double err_estimate = 1.0f/8.0f;
            if (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
                err_estimate /= 2.0f;
            }
            if (type == GGML_TYPE_Q8_0) {
                err_estimate /= 8.0f;
            }
            err_estimate *= err_estimate;
            err_estimate /= 0.25f*float(ne[0] * r * ne[2]*nr23[0] * ne[3]*nr23[1]);
            return err_estimate;
        }
        return 1e-7;
    }
};

// GGML_OP_ARGMAX
struct test_argmax : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_argmax(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 100, 1, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argmax(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override {
        return 0.0;
    }
};

// GGML_OP_COUNT_EQUAL
struct test_count_equal : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_count_equal(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {4, 500, 1, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * a_argmax = ggml_argmax(ctx, a);
        ggml_set_name(a_argmax, "a_argmax");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * b_argmax = ggml_argmax(ctx, b);
        ggml_set_name(b_argmax, "b_argmax");

        ggml_tensor * out = ggml_count_equal(ctx, a_argmax, b_argmax);
        ggml_set_name(out, "out");

        return out;
    }

    double max_nmse_err() override {
        return 0.0;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            std::array<int, 4> nr = {2, 2, 2, 2})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_set_name(target, "target");

        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        ggml_tensor * out = ggml_repeat(ctx, src, target);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_REPEAT_BACK
struct test_repeat_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;
    const bool v; // whether src is a noncontiguous view

    std::string vars() override {
        return VARS_TO_STR4(type, ne, nr, v);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {8, 6, 4, 2},
            std::array<int, 4> nr = {2, 2, 2, 2},
            bool v = false)
        : type(type), ne(ne), nr(nr), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_set_name(src, "src");

        if (v) {
            GGML_ASSERT(ne[0] % 2 == 0);
            GGML_ASSERT(ne[1] % 2 == 0);
            GGML_ASSERT(ne[2] % 2 == 0);
            GGML_ASSERT(ne[3] % 2 == 0);
            GGML_ASSERT(nr[0] % 2 == 0 || nr[0] == 1);
            GGML_ASSERT(nr[1] % 2 == 0 || nr[1] == 1);
            GGML_ASSERT(nr[2] % 2 == 0 || nr[2] == 1);
            GGML_ASSERT(nr[3] % 2 == 0 || nr[3] == 1);

            const int64_t ne00 = nr[0] == 1 ? src->ne[0] : src->ne[0] / 2;
            const int64_t ne01 = nr[1] == 1 ? src->ne[1] : src->ne[1] / 2;
            const int64_t ne02 = nr[2] == 1 ? src->ne[2] : src->ne[2] / 2;
            const int64_t ne03 = nr[3] == 1 ? src->ne[3] : src->ne[3] / 2;

            src = ggml_view_4d(ctx, src, ne00, ne01, ne02, ne03, src->nb[1], src->nb[2], src->nb[3], 0);
        }

        ggml_tensor * target = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(target, "target");

        ggml_tensor * out = ggml_repeat_back(ctx, src, target);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) v += "," + VAR_TO_STR(permute);
        return v;
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 20, 1},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), permute(permute),
            _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        if (_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(src, "src_permuted");
        }

        ggml_tensor * out = ggml_dup(ctx, src);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SET
struct test_set : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const int dim;

    std::string vars() override {
        return VARS_TO_STR4(type_src, type_dst, ne, dim);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_set(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {6, 5, 4, 3}, int dim = 1)
        : type_src(type_src), type_dst(type_dst), ne(ne), dim(dim) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        auto ne_dst = ne;
        for (int i = 0; i < dim; ++i) {
            ne_dst[i] *= 2;
        }
        ggml_tensor* dst = ggml_new_tensor(ctx, type_dst, 4, ne_dst.data());
        ggml_set_param(dst);
        ggml_set_name(dst, "dst");

        size_t offset = 0;
        for (int i = 0; i < dim; ++i) {
            offset += ((ne_dst[i] - ne[i])/2)*dst->nb[i];
        }
        ggml_tensor * out = ggml_set(ctx, dst, src,
            // The backward pass requires setting a contiguous region:
            src->nb[1], src->nb[2], src->nb[3], offset);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute_src;
    const std::array<int64_t, 4> permute_dst;
    bool _src_use_permute;
    bool _dst_use_permute;

    std::string vars() override {
        return VARS_TO_STR5(type_src, type_dst, ne, permute_src, permute_dst);
    }

    double max_nmse_err() override {
        if (type_src == type_dst) {
            return 0.0;
        }
        if (type_dst == GGML_TYPE_Q4_0 || type_dst == GGML_TYPE_Q4_1 || type_dst == GGML_TYPE_IQ4_NL ||
            type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1 || type_dst == GGML_TYPE_Q8_0) {
            // estimate what the max nmse error would be if one quantized value is
            // off by one. The test values are distributed in [-150,150], so it'll be
            // roughly (150*2.0 / 2^bits)^2, divided by the mean square value of the reference,
            // which is roughly 0.25*150^2 times the number of elements.
            double err_estimate = 1.0f/8.0f * 150.0f;
            if (type_dst == GGML_TYPE_IQ4_NL) {
                // iq4_nl values are a bit more spread out
                err_estimate *= 2.0f;
            }
            if (type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1) {
                err_estimate /= 2.0f;
            }
            if (type_dst == GGML_TYPE_Q8_0) {
                err_estimate /= 8.0f;
            }
            err_estimate *= err_estimate;
            err_estimate /= (150.0f*150.0f*0.25f)*float(ne[0] * ne[1] * ne[2] * ne[3]);
            return err_estimate;
        }
        return 1e-6;
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1},
            std::array<int64_t, 4> permute_src = {0, 0, 0, 0},
            std::array<int64_t, 4> permute_dst = {0, 0, 0, 0})
        : type_src(type_src), type_dst(type_dst), ne(ne), permute_src(permute_src), permute_dst(permute_dst),
          _src_use_permute(permute_src[0] + permute_src[1] + permute_src[2] + permute_src[3] > 0),
          _dst_use_permute(permute_dst[0] + permute_dst[1] + permute_dst[2] + permute_dst[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        if (_src_use_permute) {
            src = ggml_permute(ctx, src, permute_src[0], permute_src[1], permute_src[2], permute_src[3]);
            ggml_set_name(src, "src_permuted");
        }

        ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, src->ne);
        ggml_set_name(dst, "dst");

        if (_dst_use_permute) {
            dst = ggml_permute(ctx, dst, permute_dst[0], permute_dst[1], permute_dst[2], permute_dst[3]);
            ggml_set_name(dst, "dst_permuted");
        }

        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check if casting between f32 and i32 is consistent
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

// GGML_OP_CONT
struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        src = ggml_transpose(ctx, src);
        ggml_set_name(src, "src_transposed");

        ggml_tensor * out = ggml_cont(ctx, src);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ADD
// GGML_OP_SUB
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;
    int nf; // number of fused ops, nf == 1 -> single op (no fusion)

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne, nr, nf);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 3;
    }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1},
            int nf = 1)
        : op(op), type(type), ne(ne), nr(nr), nf(nf) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        GGML_ASSERT(nf <= 16);

        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_set_name(a, "a");

        ggml_tensor * b[16];
        for (int i = 0; i < nf; ++i) {
            b[i] = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b[i], (std::string("b") + std::to_string(i)).c_str());
        }

        // The backward pass supports broadcasting only for GGML_ADD:
        const bool grad_supported = op == ggml_add && ggml_are_same_shape(a, b[0]) && nf == 1;
        if (grad_supported) {
            ggml_set_param(a);
            ggml_set_param(b[0]);
        }

        ggml_tensor * out = a;

        for (int i = 0; i < nf; ++i) {
            out = op(ctx, out, b[i]);
        }

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_mul || op == ggml_div) {
                // MUL and DIV have numerical issues around zero:
                init_tensor_uniform(t, 0.9f, 1.1f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    float grad_eps() override {
        return 0.1f * (op == ggml_mul ? ne[0]*ne[1]*ne[2]*ne[3] : 1);
    }

    bool grad_precise() override {
        return op == ggml_div;
    }

    double max_maa_err() override {
        return op == ggml_add ? 1e-4 : 1e-3;
    }
};

// GGML_OP_ADD_ID
struct test_add_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t n_embd;
    const int64_t n_experts;
    const int64_t n_experts_used;
    const int64_t n_token;

    std::string vars() override {
        return VARS_TO_STR6(type_a, type_b, n_embd, n_experts, n_experts_used, n_token);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]) + ggml_nbytes(t->src[2]);
    }

    test_add_id(ggml_type type_a = GGML_TYPE_F32,
            ggml_type type_b = GGML_TYPE_F32,
            int64_t n_embd = 128,
            int64_t n_experts = 16,
            int64_t n_experts_used = 8,
            int64_t n_token = 10)
        : type_a(type_a), type_b(type_b), n_embd(n_embd),
          n_experts(n_experts), n_experts_used(n_experts_used), n_token(n_token) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_3d(ctx, type_a, n_embd, n_experts_used, n_token);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, type_b, n_embd, n_experts);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts, n_token);
        if (n_experts_used != n_experts) {
            ids = ggml_view_2d(ctx, ids, n_experts_used, n_token, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * out = ggml_add_id(ctx, a, b, ids);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                std::random_device rd;
                std::default_random_engine rng(rd());
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_experts;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_ADD1
struct test_add1 : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_add1(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor_1d(ctx, type, 1);
        // ggml_set_param(b); // TODO: implement
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_add1(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * ne[0]*ne[1]*ne[2]*ne[3];
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float scale;
    float bias;
    bool inplace;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, scale, bias, inplace);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float scale = 2.0f,
            float bias = 0.0f,
            bool inplace = false)
        : type(type), ne(ne), scale(scale), bias(bias), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out;
        if (inplace) {
            out = ggml_scale_bias_inplace(ctx, a, scale, bias);
        } else {
            out = ggml_scale_bias(ctx, a, scale, bias);
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
struct test_softcap : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float softcap;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SOFTCAP";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(type, ne, softcap);
    }

    test_softcap(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float softcap = 30.0f)
        : type(type), ne(ne), softcap(softcap) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());

        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, a, 1.0f / softcap)), softcap);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SILU_BACK
struct test_silu_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_silu_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(grad, "grad");

        ggml_tensor * out = ggml_silu_back(ctx, a, grad);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool v; // whether a is a non-contiguous view
    const float eps;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, v, eps);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            bool v = false,
            float eps = 1e-6f)
        : type(type), ne(ne), v(v), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0]/2, a->ne[1]/2, a->ne[2]/2, a->ne[3]/2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out = ggml_norm(ctx, a, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_NORM + GGML_OP_MUL + GGML_OP_ADD
struct test_norm_mul_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;
    const bool broadcast;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "NORM_MUL_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne, eps, broadcast);
    }

    test_norm_mul_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 2, 1, 1},
            float eps = 1e-5f,
            bool broadcast = false)
        : type(type), ne(ne), eps(eps), broadcast(broadcast) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        std::array<int64_t, 4> broadcast_dims = {ne[0], ne[1] * 2, ne[2] * 2, ne[3] * 2};

        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, broadcast ? broadcast_dims.data() : ne.data());
        ggml_tensor * w = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a); ggml_set_param(w); ggml_set_param(b);
        ggml_set_name(a, "a"); ggml_set_name(w, "w"); ggml_set_name(b, "b");

        // Use a, w and b early to avoid OP_NONE in graph
        a = ggml_add(ctx, ggml_add(ctx, a, w), b);

        ggml_tensor * n = ggml_norm(ctx, a, eps);
        ggml_tensor * m = ggml_mul(ctx, n, w);
        ggml_tensor * out = ggml_add(ctx, m, b);
        ggml_set_name(out, "out");
        return out;
    }
};
// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool v; // whether a is a non-contiguous view
    const float eps;
    const bool inplace; // whether to do the operation inplace

    std::string vars() override {
        return VARS_TO_STR5(type, ne, v, eps, inplace);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            bool v = false,
            float eps = 1e-6f,
            bool inplace = false)
        : type(type), ne(ne), v(v), eps(eps), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0]/2, a->ne[1]/2, a->ne[2]/2, a->ne[3]/2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out;
        if (inplace) {
            out = ggml_rms_norm_inplace(ctx, a, eps);
        } else {
            out = ggml_rms_norm(ctx, a, eps);
        }
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_RMS_NORM_BACK
struct test_rms_norm_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_rms_norm_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_rms_norm_back(ctx, a, b, eps);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }
};

// GGML_OP_RMS_NORM + GGML_OP_MUL + GGML_OP_ADD
struct test_rms_norm_mul_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;
    const bool broadcast;
    const bool multi_add; // test a sequence of adds feeding into rms_norm

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_MUL_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR5(type, ne, eps, broadcast, multi_add);
    }

    test_rms_norm_mul_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f, bool broadcast = false, bool multi_add = false)
        : type(type), ne(ne), eps(eps), broadcast(broadcast), multi_add(multi_add) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        std::array<int64_t, 4> broadcast_dims = {ne[0]*2, ne[1]*3, ne[2]*3, ne[3]*4};

        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, broadcast ? broadcast_dims.data() : ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * c = ggml_new_tensor(ctx, type, 4, ne.data());

        ggml_set_param(a);
        ggml_set_name(a, "a");
        ggml_set_param(b);
        ggml_set_name(b, "b");
        ggml_set_param(c);
        ggml_set_name(c, "c");

        // Use a, b and c early, so we don't end up with an OP_NONE between rms_norm and mul
        a = ggml_add(ctx, ggml_add(ctx, a, b), c);
        if (multi_add) {
            a = ggml_add(ctx, ggml_add(ctx, a, b), c);
        }
        ggml_tensor * out = ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, a, eps), b), c);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_SSM_CONV
struct test_ssm_conv : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, ne_b);
    }

    test_ssm_conv(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 1},
            std::array<int64_t, 4> ne_b = {3, 3, 1, 1})
        : type(type), ne_a(ne_a), ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a   = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b   = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_tensor * out = ggml_ssm_conv(ctx, a, b);
        return out;
    }
};

// GGML_OP_SSM_SCAN
struct test_ssm_scan : public test_case {
    const ggml_type type;

    const int64_t d_state;
    const int64_t head_dim;
    const int64_t n_head;
    const int64_t n_group;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR7(type, d_state, head_dim, n_head, n_group, n_seq_tokens, n_seqs);
    }

    test_ssm_scan(ggml_type type = GGML_TYPE_F32,
            int64_t d_state = 32,
            int64_t head_dim = 1, // non-zero for Mamba-2
            int64_t n_head  = 32,
            int64_t n_group = 1,
            int64_t n_seq_tokens = 32,
            int64_t n_seqs = 32)
        : type(type), d_state(d_state), head_dim(head_dim), n_head(n_head), n_group(n_group), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * s   = ggml_new_tensor_4d(ctx, type, d_state,  head_dim,     n_head,       n_seqs);
        ggml_tensor * x   = ggml_new_tensor_4d(ctx, type, head_dim, n_head,       n_seq_tokens, n_seqs);
        ggml_tensor * dt  = ggml_new_tensor_3d(ctx, type, n_head,   n_seq_tokens, n_seqs);
        ggml_tensor * A   = ggml_new_tensor_2d(ctx, type, (head_dim > 1) ? 1 : d_state, n_head);
        ggml_tensor * B   = ggml_new_tensor_4d(ctx, type, d_state,  n_group,      n_seq_tokens, n_seqs);
        ggml_tensor * C   = ggml_new_tensor_4d(ctx, type, d_state,  n_group,      n_seq_tokens, n_seqs);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,  n_seqs);
        ggml_tensor * out = ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids);
        return out;
    }

    // similar to test_mul_mat_id
    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_RWKV_WKV6
struct test_rwkv_wkv6 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_rwkv_wkv6(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * r   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * tf  = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size, head_count }.data());
        ggml_tensor * td  = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, s);
        return out;
    }
};

// GGML_OP_GATED_LINEAR_ATTN
struct test_gla : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_gla(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * q   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * g   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_gated_linear_attn(ctx, k, v, q, g, s, pow(head_size, -0.5));
        return out;
    }
};

// GGML_OP_RWKV_WKV7
struct test_rwkv_wkv7 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_rwkv_wkv7(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * r   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * w   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * a   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * b   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        // Outputs may become NaN with long seqlen without these normalization
        a = ggml_l2_norm(ctx, a, 1e-7F);
        b = ggml_l2_norm(ctx, b, 1e-7F);
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, s);
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions
    const bool v; // whether a and b are non-contiguous views
    const uint32_t o; // number of outputs

    std::string vars() override {
        return VARS_TO_STR10(type_a, type_b, m, n, k, bs, nr, per, v, o);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    int64_t grad_nmax() override {
        return 20000;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            std::array<int64_t, 4> per = {0, 1, 2, 3},
            bool v = false, uint32_t o = 1)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), per(per), v(v), o(o) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a;
        ggml_tensor * b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(!v); // not handled
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions given by m,n,k.
            const int64_t ne_a[4] = {k, m, bs[0],       bs[1]};
            const int64_t ne_b[4] = {k, n, bs[0]*nr[0], bs[1]*nr[1]};

            a = ggml_new_tensor_4d(ctx, type_a, ne_a[per[0]], ne_a[per[1]], ne_a[per[2]], ne_a[per[3]]);
            b = ggml_new_tensor_4d(ctx, type_b, ne_b[per[0]], ne_b[per[1]], ne_b[per[2]], ne_b[per[3]]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            ggml_set_name(a, "a_permuted");
            ggml_set_name(b, "b_permuted");
        } else {
            if (v) {
                a = ggml_new_tensor_4d(ctx, type_a, k*2, m, bs[0],       bs[1]);
                b = ggml_new_tensor_4d(ctx, type_b, k*2, n, bs[0]*nr[0], bs[1]*nr[1]);

                if (!ggml_is_quantized(type_a)) {
                    if (bs[1] == 1 && nr[1] == 1) {
                        ggml_set_param(a);
                    }
                    ggml_set_param(b);
                }

                a = ggml_view_4d(ctx, a, k, m, bs[0],       bs[1],       a->nb[1], a->nb[2], a->nb[3], 0);
                b = ggml_view_4d(ctx, b, k, n, bs[0]*nr[0], bs[1]*nr[1], b->nb[1], b->nb[2], b->nb[3], 0);
            } else {
                a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0],       bs[1]);
                b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);

                if (!ggml_is_quantized(type_a)) {
                    if (bs[1] == 1 && nr[1] == 1) {
                        ggml_set_param(a);
                    }
                    ggml_set_param(b);
                }
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");
        for (uint32_t i = 1; i < o; ++i) {
            ggml_tensor * out2 = ggml_mul_mat(ctx, a, b);
            ggml_set_name(out2, "out2");
            out = ggml_add(ctx, out, out2);
        }

        return out;
    }

    bool run_whole_graph() override { return o > 1; }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return ggml_op_name(GGML_OP_MUL_MAT);
    }
};

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // broadcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const uint32_t o; // number of outputs

    std::string vars() override {
        return VARS_TO_STR9(type_a, type_b, n_mats, n_used, b, m, n, k, o);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * k * n * n_used;
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32, uint32_t o = 1)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k), o(o) {
            GGML_ASSERT(n_used <= n_mats);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_set_name(as, "as");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        ggml_set_name(ids, "ids");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_set_name(out, "out");

        for (uint32_t i = 1; i < o; ++i) {
            ggml_tensor * a2 = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
            ggml_tensor * out2 = ggml_mul_mat_id(ctx, a2, b, ids);
            ggml_set_name(out2, "out2");
            out = ggml_add(ctx, out, out2);
        }

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                std::random_device rd;
                std::default_random_engine rng(rd());
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    bool run_whole_graph() override { return o > 1; }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return ggml_op_name(GGML_OP_MUL_MAT_ID);
    }
};

// GGML_OP_OUT_PROD
struct test_out_prod : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4
    const bool trans_b;

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, m, n, k, bs, nr, trans_b);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    test_out_prod(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            bool trans_b = false)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), trans_b(trans_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, m, k, bs[0], bs[1]);
        ggml_set_name(a, "a");

        ggml_tensor * b;
        if (trans_b) {
            b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
            b = ggml_transpose(ctx, b);
        } else {
            b = ggml_new_tensor_4d(ctx, type_b, n, k, bs[0]*nr[0], bs[1]*nr[1]);
        }
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_out_prod(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sqr(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * 0.25f*ne[0]*ne[1]*ne[2]*ne[3]; // 10% of expected value of sum.
    }
};

// GGML_OP_SQRT
struct test_sqrt : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqrt(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 3, 3, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sqrt(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // fill with positive values
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 50.0f, 100.0f);
        }
    }

    float grad_eps() override {
        return 20.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_LOG
struct test_log : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_log(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_log(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // log(1) == 0, cluster values there to keep the sum low for better precision in the backward pass:
            init_tensor_uniform(t, 0.9f, 1.1f);
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_SIN
struct test_sin : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sin(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sin(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -6.5f, 6.5f); // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    float grad_eps() override {
        return 0.2f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_COS
struct test_cos : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cos(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_cos(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -6.5f, 6.5f); // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    float grad_eps() override {
        return 0.2f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 1e-2f;
    }

    std::vector<float> grad_expect() override {
        return {0.0f, 1.0f};
    }
};

// GGML_OP_FLOOR
struct test_floor : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_floor(ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_floor(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_CEIL
struct test_ceil : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_ceil(ggml_type type = GGML_TYPE_F32,
              std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_ceil(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_ROUND
struct test_round : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_round(ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_round(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_TRUNC
struct test_trunc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_trunc(ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_trunc(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 3, 2},
            int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool mask;
    const bool sinks;
    const ggml_type m_prec;
    const std::array<int64_t, 2> nr23; // broadcast only dims 2 and 3
    const float scale;
    const float max_bias;
    const bool inplace;

    std::string vars() override {
        return VARS_TO_STR9(type, ne, mask, sinks, m_prec, nr23, scale, max_bias, inplace);
    }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override {
        return 1e-6;
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            bool mask = false,
            bool sinks = false,
            ggml_type m_prec = GGML_TYPE_F32,
            std::array<int64_t, 2> nr23 = {1, 1},
            float scale = 1.0f,
            float max_bias = 0.0f,
            bool inplace = false)
        : type(type), ne(ne), mask(mask), sinks(sinks), m_prec(m_prec), nr23(nr23), scale(scale), max_bias(max_bias), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2]*nr23[0], ne[3]*nr23[1]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * mask = nullptr;
        if (this->mask) {
            mask = ggml_new_tensor_4d(ctx, m_prec, ne[0], ne[1], ne[2], ne[3]);
            ggml_set_name(mask, "mask");
        }

        ggml_tensor * sinks = nullptr;
        if (this->sinks) {
            sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[2]*nr23[0]);
            ggml_set_name(sinks, "sinks");
        }

        ggml_tensor * out;
        if (inplace) {
            out = ggml_soft_max_ext_inplace(ctx, a, mask, scale, max_bias);
        } else {
            out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        }
        ggml_soft_max_add_sinks(out, sinks);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_SOFT_MAX_BACK
struct test_soft_max_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float scale;
    const float max_bias;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, scale, max_bias);
    }

    test_soft_max_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            float scale = 1.0f,
            float max_bias = 0.0f)
        : type(type), ne(ne), scale(scale), max_bias(max_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_soft_max_ext_back(ctx, a, b, scale, max_bias);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ROPE + GGML_OP_ROPE_BACK
struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx; // used to generate positions
    float fs; // freq_scale
    float ef; // ext_factor
    float af; // attn_factor
    bool ff;
    int v; // view (1 : non-contiguous a)
    bool forward;
    bool inplace;

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return VARS_TO_STR11(type, ne_a, n_dims, mode, n_ctx, fs, ef, af, ff, v, inplace);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 3, 1},
            int n_dims = 10, int mode = GGML_ROPE_TYPE_NORMAL, int n_ctx = 512, float fs = 1.0f,
            float ef = 0.0f, float af = 0.0f, bool ff = false, int v = 0, bool forward = true, bool inplace = false)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx), fs(fs), ef(ef), af(af), ff(ff), v(v), forward(forward), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (forward) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (forward) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");
        }

        const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

        ggml_tensor * pos;
        if (is_mrope || is_vision) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2] * 4);
        } else {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        }
        ggml_set_name(pos, "pos");

        ggml_tensor * freq = nullptr;
        if (ff) {
            freq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2);
            ggml_set_name(freq, "freq");
        }

        ggml_tensor * out;
        if (is_mrope) {
            if (is_vision) {
                GGML_ASSERT(n_dims/4 > 0);
                int rope_sections[4] = {n_dims/4, n_dims/4, 0, 0}; // Vision-RoPE only use first two dimension for image (x, y) coordinate
                if (forward) {
                    if (inplace) {
                        out = ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    } else {
                        out = ggml_rope_multi(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    }
                } else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            } else {
                GGML_ASSERT(n_dims/3 > 0);
                int rope_sections[4] = {n_dims/3, n_dims/3, n_dims/3, 0};
                if (forward) {
                    if (inplace) {
                        out = ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    } else {
                        out = ggml_rope_multi(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    }
                } else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            }
        } else {
            if (forward) {
                if (inplace) {
                    out = ggml_rope_ext_inplace(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                } else {
                    out = ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            } else {
                out = ggml_rope_ext_back(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }

            // TODO: add test with a non-contiguous view as input ; this case is needed for build_rope_2d in clip.cpp
        }
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                const int num_pos_ids = (mode & GGML_ROPE_TYPE_MROPE) ? ne_a[2] * 4 : ne_a[2];
                std::vector<int> data(num_pos_ids);
                for (int i = 0; i < num_pos_ids; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, num_pos_ids * sizeof(int));
            } else {
                if (t->ne[0] == n_dims/2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_POOL2D
struct test_pool2d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int k0;
    const int k1;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;

    std::string vars() override {
        return VARS_TO_STR9(pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1);
    }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
            ggml_type type_input = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            int k0 = 3, int k1 = 3,
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1)
        : pool_type(pool_type), type_input(type_input), ne_input(ne_input), k0(k0), k1(k1), s0(s0), s1(s1), p0(p0), p1(p1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        ggml_tensor * out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_1D
struct test_conv_transpose_1d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;

    const int s0; // stride
    const int p0; // padding
    const int d0; // dilation

    std::string vars() override {
        return VARS_TO_STR5(ne_input, ne_kernel, s0, p0, d0);
    }

    test_conv_transpose_1d(std::array<int64_t, 4> ne_input = {197, 32, 1, 1}, // [input_width, input_channels, 1 /* assert in cpu kernel*/, 1 (should be batch)]
                           std::array<int64_t, 4> ne_kernel = {16, 32, 32, 1}, // [kernel_width, output_channels, input_channels, 1 (should be batch)]
                           int s0 = 1, int p0 = 0, int d0 = 1)
        : ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), p0(p0), d0(d0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, s0, p0, d0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_2D
struct test_conv_transpose_2d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const int stride;

    std::string vars() override {
        return VARS_TO_STR3(ne_input, ne_kernel, stride);
    }

    double max_nmse_err() override {
        return 5e-4; // The default 1e-7 is too small for Vulkan.
    }

    test_conv_transpose_2d(std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
                           std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
                           int stride = 1)
        : ne_input(ne_input), ne_kernel(ne_kernel), stride(stride){}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F16, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_transpose_2d_p0(ctx, kernel, input, stride);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;
    // dilation
    const int d0;
    const int d1;
    // mode
    const bool is_2D;

    std::string vars() override {
        return VARS_TO_STR12(type_input, type_kernel, dst_type, ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1,
            int d0 = 1, int d1 = 1,
            bool is_2D = true)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D, dst_type);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_IM2COL_3D
struct test_im2col_3d : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    const int s2;
    // padding
    const int p0;
    const int p1;
    const int p2;
    // dilation
    const int d0;
    const int d1;
    const int d2;

    const int64_t IC;
    const bool v;

    std::string vars() override {
        return VARS_TO_STR16(type_input, type_kernel, dst_type, ne_input, ne_kernel, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, v);
    }

    test_im2col_3d(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
                std::array<int64_t, 4> ne_input = {10, 10, 10, 9}, // [OC*IC, KD, KH, KW]
                std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [N*IC, ID, IH, IW]
                int64_t IC = 3,
                int s0 = 1, int s1 = 1, int s2 = 1,
                int p0 = 1, int p1 = 1, int p2 = 1,
                int d0 = 1, int d1 = 1, int d2 = 1,
                bool v = false)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), IC(IC), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        if (v) {
            input = ggml_view_4d(ctx, input, ne_input[0] - 2, ne_input[1] - 2, ne_input[2] - 2, ne_input[3] - 2, input->nb[1], input->nb[2], input->nb[3], 0);
            ggml_set_name(input, "view_of_input");
        }

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_im2col_3d(ctx, kernel, input, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, dst_type);
        ggml_set_name(out, "out");

        return out;
    }
};

// CONV_2D
struct test_conv_2d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const ggml_type              type_kernel;
    const int                    stride0;
    const int                    stride1;
    const int                    padding0;
    const int                    padding1;
    const int                    dilation0;
    const int                    dilation1;
    // Whether the inputs are contiguous in the channel dim or the width dim
    const bool                   cwhn;

    // If true, the direct CONV_2D will be used in the graph, otherwise it
    // uses ggml_conv_2d:
    // * if the program is called with -o CONV_2D_DIRECT_IMPL, the
    // CONV_2D graph will be built, while
    // * if the program is called with -o CONV_2D_INDIRECT_IMPL, the
    // IM2COL -> MUL_MM graph will be built.

    std::string vars() override {
        return VARS_TO_STR10(ne_input, ne_kernel, type_kernel, stride0, stride1, padding0, padding1, dilation0, dilation1, cwhn);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Just counting matmul costs:
        // KxCRS @ CRSxNPQ = KxNPQ --> KxNPQx(CRS+CRS-1) flops

        // Copied from ggml.c: int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d)
        auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
            return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
        };

        int64_t W    = ne_input[0];
        int64_t H    = ne_input[1];
        int64_t KW   = ne_kernel[0];
        int64_t KH   = ne_kernel[1];
        int64_t Cin  = ne_kernel[2];
        int64_t Cout = ne_kernel[3];
        int64_t N    = ne_input[3];
        int64_t OH   = calc_conv_output_size(H, KH, stride0, padding0, dilation0);
        int64_t OW   = calc_conv_output_size(W, KW, stride0, padding0, dilation0);

        int64_t K   = Cout;
        int64_t CRS = Cin * KH * KW;
        int64_t NPQ = N * OH * OW;

        return K * NPQ * (2 * CRS - 1);
    }

    test_conv_2d(std::array<int64_t, 4> ne_input  = { 64, 64, 16, 1 },
                 std::array<int64_t, 4> ne_kernel = { 3, 3, 1, 16 }, ggml_type type_kernel = GGML_TYPE_F32, int stride0 = 1,
                 int stride1 = 1, int padding0 = 0, int padding1 = 0, int dilation0 = 1, int dilation1 = 1, bool cwhn = false) :
        ne_input(ne_input),
        ne_kernel(ne_kernel),
        type_kernel(type_kernel),
        stride0(stride0),
        stride1(stride1),
        padding0(padding0),
        padding1(padding1),
        dilation0(dilation0),
        dilation1(dilation1),
        cwhn(cwhn) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        if (cwhn) {
            // change memory layout to channel-most-contiguous (CWHN),
            // then permute it back so NE matches the original input
            input  = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
            input  = ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        ggml_tensor * out =
            ggml_conv_2d_direct(ctx, kernel, input, stride0, stride1, padding0, padding1, dilation0, dilation1);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CONV_2D_DW
struct test_conv_2d_dw : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const int stride;
    const int padding;
    const int dilation;
    const bool cwhn;

    std::string vars() override {
        return VARS_TO_STR6(ne_input, ne_kernel, stride, padding, dilation, cwhn);
    }

    test_conv_2d_dw(std::array<int64_t, 4> ne_input = {64, 64, 16, 1},
            std::array<int64_t, 4> ne_kernel = {3, 3, 1, 16},
            int stride = 1, int padding = 0, int dilation = 1, bool cwhn = false)
        : ne_input(ne_input), ne_kernel(ne_kernel), stride(stride), padding(padding), dilation(dilation), cwhn(cwhn) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        if (cwhn) {
            // change memory layout to channel-most-contiguous (CWHN),
            // then permute it back so NE matches the original input
            input = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
            input = ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        ggml_tensor * out = ggml_conv_2d_dw_direct(
            ctx, kernel, input,
            stride, stride, padding, padding, dilation, dilation);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CONV_3D
struct test_conv_3d : public test_case {
    // Logical 5D dimensions
    const int64_t N, IC, ID, IH, IW;
    const int64_t OC, KD, KH, KW;
    // Conv params
    const int s0, s1, s2;
    const int p0, p1, p2;
    const int d0, d1, d2;
    // Types
    const ggml_type type_kernel;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "CONV_3D";
    }

    std::string vars() override {
        return VARS_TO_STR11(N, IC, ID, IH, IW, OC, KD, KH, KW, s0, s1) + "," +
               VARS_TO_STR8(s2, p0, p1, p2, d0, d1, d2, type_kernel);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
            return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
        };
        const int64_t OD = calc_conv_output_size(ID, KD, s2, p2, d2);
        const int64_t OH = calc_conv_output_size(IH, KH, s1, p1, d1);
        const int64_t OW = calc_conv_output_size(IW, KW, s0, p0, d0);

        return (uint64_t)N * OC * OD * OH * OW * (2 * IC * KD * KH * KW - 1);
    }

    test_conv_3d(
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW,
        int64_t OC, int64_t KD, int64_t KH, int64_t KW,
        int s0, int s1, int s2,
        int p0, int p1, int p2,
        int d0, int d1, int d2,
        ggml_type type_kernel
    ) : N(N), IC(IC), ID(ID), IH(IH), IW(IW),
        OC(OC), KD(KD), KH(KH), KW(KW),
        s0(s0), s1(s1), s2(s2),
        p0(p0), p1(p1), p2(p2),
        d0(d0), d1(d1), d2(d2),
        type_kernel(type_kernel) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // GGML input tensor is packed as [W, H, D, C*N]
        const int64_t ne_input[] = {IW, IH, ID, IC * N};
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input);
        ggml_set_name(input, "input");

        // GGML kernel tensor is packed as [KW, KH, KD, IC*OC]
        const int64_t ne_kernel[] = {KW, KH, KD, IC * OC};
        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel);
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_3d_direct(ctx, kernel, input, s0, s1, s2, p0, p1, p2, d0, d1, d2, (int)IC, (int)N, (int)OC);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CONCAT
struct test_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int64_t ne_b_d;
    const int dim;
    const int v; // view (1 << 0: non-cont a, 1 << 1: non-cont b)

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, ne_b_d, dim, v);
    }

    test_concat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 5, 5},
            int64_t ne_b_d = 5,
            int dim = 2, int v = 0)
        : type(type), ne_a(ne_a), ne_b_d(ne_b_d), dim(dim), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto ne_b = ne_a;
        ne_b[dim] = ne_b_d;
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_name(a, "a");
        }
        ggml_tensor * b;
        if (v & 2) {
            auto ne = ne_b; ne[0] *= 3; ne[1] *= 2; ne[2] *= 4;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(b, "view_of_b");
        } else {
            b = ggml_new_tensor(ctx, type, 4, ne_b.data());
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_concat(ctx, a, b, dim);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ARGSORT
struct test_argsort : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order order;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, order);
    }

    test_argsort(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {16, 10, 10, 10},
            ggml_sort_order order = GGML_SORT_ORDER_ASC)
        : type(type), ne(ne), order(order) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argsort(ctx, a, order);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(ggml_nelements(t));
                for (int i = 0; i < ggml_nelements(t); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0]*ne[1]*ne[2]*ne[3] * sizeof(int));
            } else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                GGML_ABORT("fatal error");
            }
        }
    }
};

struct test_topk_moe: public test_case {
    const std::array<int64_t, 4> ne;
    const int n_expert_used;
    const bool with_norm;
    const bool                   delayed_softmax;

    test_topk_moe(std::array<int64_t, 4> ne              = { 10, 5, 1, 1 },
                  int                    n_expert_used   = 1,
                  bool                   with_norm       = false,
                  bool                   delayed_softmax = false) :
        ne(ne),
        n_expert_used(n_expert_used),
        with_norm(with_norm),
        delayed_softmax(delayed_softmax) {
        GGML_ASSERT(n_expert_used <= ne[0]);
        GGML_ASSERT(!(with_norm && delayed_softmax));
    }

    std::string vars() override { return VARS_TO_STR4(ne, n_expert_used, with_norm, delayed_softmax); }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "TOPK_MOE";
    }

    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int n_expert = ne[0];
        const int n_tokens = ne[1];

        ggml_tensor * logits = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
        ggml_tensor * probs            = delayed_softmax ? logits : ggml_soft_max(ctx, logits);
        ggml_tensor * selected_experts = ggml_top_k(ctx, probs, n_expert_used); // [n_expert_used, n_tokens]

        ggml_tensor * out = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]

        if (delayed_softmax) {
            out = ggml_reshape_2d(ctx, out, n_expert_used, n_tokens);
            out = ggml_soft_max(ctx, out);  // [n_expert_used, n_tokens]
            out = ggml_reshape_3d(ctx, out, 1, n_expert_used, n_tokens);
        }

        if (with_norm) {
            out = ggml_reshape_2d(ctx, out, n_expert_used, n_tokens);
            ggml_tensor * weights_sum = ggml_sum_rows(ctx, out); // [1, n_tokens]

            weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5, INFINITY);
            out = ggml_div(ctx, out, weights_sum); // [n_expert_used, n_tokens]
            out = ggml_reshape_3d(ctx, out, 1, n_expert_used, n_tokens);
        }

        ggml_set_name(out, "out");
        return out;
    }
};

struct test_mul_mat_vec_fusion : public test_case {
    const ggml_type type;
    const ggml_glu_op glu_op;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const bool use_id;
    const int n_mats;
    const int n_used;
    const bool b;        // broadcast b matrix (only for use_id)
    const bool with_bias;
    const bool with_gate;

    test_mul_mat_vec_fusion(ggml_type type, ggml_glu_op op, int64_t m, int64_t n, int64_t k,
                        bool use_id = false, int n_mats = 1, int n_used = 1, bool b = false, bool with_bias = false, bool with_gate = true)
    : type(type), glu_op(op), m(m), n(n), k(k), use_id(use_id), n_mats(n_mats), n_used(n_used), b(b), with_bias(with_bias), with_gate(with_gate) {
        if (use_id) {
            GGML_ASSERT(n_used <= n_mats);
        }
    }

    std::string vars() override {
        return VARS_TO_STR11(type, glu_op, m, n, k, use_id, n_mats, n_used, b, with_bias, with_gate);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_VEC_FUSION";
    }

    bool run_whole_graph() override { return true; }

    ggml_tensor * build_gate(ggml_context * ctx, ggml_tensor * ffn_gate, ggml_tensor * ffn_up) {
        ggml_tensor * out = nullptr;
        if (with_gate) {
            if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                out = ggml_swiglu_oai(ctx, ffn_gate, ffn_up, alpha, limit);
            } else {
                out = ggml_glu_split(ctx, ffn_gate, ffn_up, glu_op);
            }
        }
        return out;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        if (!use_id) {
            std::array<int64_t, 4> ne = {k, m, 1, 1};
            std::array<int64_t, 4> ne0 = {k, n, 1, 1};

            ggml_tensor * cur  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
            ggml_tensor * gate = with_gate ? ggml_new_tensor(ctx, type, 4, ne0.data()) : nullptr;
            ggml_tensor * up   = ggml_new_tensor(ctx, type, 4, ne0.data());

            ggml_tensor * ffn_up = ggml_mul_mat(ctx, up, cur);
            if (with_bias) {
                std::array<int64_t, 4> bias_ne = {ffn_up->ne[0], 1, 1, 1};
                ggml_tensor * up_bias = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, bias_ne.data());
                ffn_up = ggml_add(ctx, ffn_up, up_bias);
            }

            ggml_tensor * ffn_gate = with_gate ? ggml_mul_mat(ctx, gate, cur) : nullptr;
            if (with_bias && with_gate) {
                std::array<int64_t, 4> bias_ne = {ffn_gate->ne[0], 1, 1, 1};
                ggml_tensor * gate_bias = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, bias_ne.data());
                ffn_gate = ggml_add(ctx, ffn_gate, gate_bias);
            }

            ggml_tensor * out = with_gate ? build_gate(ctx, ffn_gate, ffn_up) : ffn_up;
            ggml_set_name(out, "out");
            return out;
        } else {
            ggml_tensor * gates = ggml_new_tensor_3d(ctx, type, k, n, n_mats);
            ggml_tensor * ups   = ggml_new_tensor_3d(ctx, type, k, n, n_mats);
            ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, m);

            if (n_used != n_mats) {
                ids = ggml_view_2d(ctx, ids, n_used, m, ids->nb[1], 0);
            }

            ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, this->b ? 1 : n_used, m);
            ggml_set_name(cur, "cur");

            ggml_tensor * ffn_up = ggml_mul_mat_id(ctx, ups, cur, ids);
            if (with_bias) {
                ggml_tensor * up_bias_param = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_up->ne[0], n_mats);
                ffn_up = ggml_add_id(ctx, ffn_up, up_bias_param, ids);
            }

            ggml_tensor * ffn_gate = with_gate? ggml_mul_mat_id(ctx, gates, cur, ids) : nullptr;
            if (with_bias && with_gate) {
                ggml_tensor * gate_bias_param = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_gate->ne[0], n_mats);
                ffn_gate = ggml_add_id(ctx, ffn_gate, gate_bias_param, ids);
            }

            ggml_tensor * out = with_gate ? build_gate(ctx, ffn_gate, ffn_up) : ffn_up;
            ggml_set_name(out, "out");
            return out;
        }
    }

    void initialize_tensors(ggml_context * ctx) override {
        if (!use_id) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                init_tensor_uniform(t);
            }
        } else {
            std::random_device rd;
            std::default_random_engine rng(rd());
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (t->type == GGML_TYPE_I32) {
                    if (ggml_is_view_op(t->op)) { continue; }
                    // ids
                    for (int64_t r = 0; r < ggml_nrows(t); r++) {
                        std::vector<int32_t> data(t->ne[0]);
                        for (int i = 0; i < t->ne[0]; i++) {
                            data[i] = i % n_mats;
                        }
                        std::shuffle(data.begin(), data.end(), rng);
                        ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                    }
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }

    double max_nmse_err() override {
        return 5e-3;
    }
};

// GGML_OP_SUM
struct test_sum : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) v += "," + VAR_TO_STR(permute);
        return v;
    }

    test_sum(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), permute(permute),
            _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        if (_use_permute) {
            a = ggml_permute(ctx, a, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(a, "a_permuted");
        }

        ggml_tensor * out = ggml_sum(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * sqrtf(ne[0]*ne[1]*ne[2]*ne[3]);
    }
};

// GGML_OP_SUM_ROWS
struct test_sum_rows : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool permute;
    const bool slice;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, permute, slice);
    }

    test_sum_rows(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            bool permute = false, bool slice = false)
        : type(type), ne(ne), permute(permute), slice(slice) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        if (slice) {
            a = ggml_view_4d(ctx, a,
                             ne[0], ne[1], ne[2] / 2, ne[3] - 1,
                             a->nb[1], a->nb[2] * 2, a->nb[3], /*offset=*/a->nb[3]);
        }
        if (permute) {
            a = ggml_permute(ctx, a, 0, 2, 3, 1);
        }

        ggml_tensor * out = ggml_sum_rows(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_MEAN
struct test_mean : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_mean(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_mean(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * ne[0]*ne[1]*ne[2]*ne[3];
    }
};

// GGML_OP_UPSCALE
struct test_upscale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t scale_factor;
    const bool transpose;
    const ggml_scale_mode mode;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, scale_factor, mode, transpose);
    }

    test_upscale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {512, 512, 3, 1},
            int32_t scale_factor = 2, ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST, bool transpose = false)
        : type(type), ne(ne), scale_factor(scale_factor), transpose(transpose), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        if (transpose) {
            a = ggml_transpose(ctx, a);
            ggml_set_name(a, "a_transposed");
        }

        ggml_tensor * out = ggml_upscale(ctx, a, scale_factor, mode);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_UPSCALE (via ggml_interpolate)
struct test_interpolate : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> ne_tgt;
    const uint32_t mode = GGML_SCALE_MODE_NEAREST;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, ne_tgt, mode);
    }

    test_interpolate(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne     = {2, 5,  7, 11},
            std::array<int64_t, 4> ne_tgt = {5, 7, 11, 13},
            uint32_t mode = GGML_SCALE_MODE_NEAREST)
        : type(type), ne(ne), ne_tgt(ne_tgt), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_interpolate(ctx, a, ne_tgt[0], ne_tgt[1],ne_tgt[2], ne_tgt[3], mode);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_GROUP_NORM
struct test_group_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t num_groups;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, num_groups, eps);
    }

    test_group_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            int32_t num_groups = 32,
            float eps = 1e-6f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_group_norm(ctx, a, num_groups, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_GROUP_NORM + GGML_OP_MUL + GGML_OP_ADD
struct test_group_norm_mul_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    int num_groups;
    float eps;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GROUP_NORM_MUL_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne, num_groups, eps);
    }

    test_group_norm_mul_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 1, 1, 1},
            int num_groups = 4,
            float eps = 1e-5f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * w = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a); ggml_set_param(w); ggml_set_param(b);
        ggml_set_name(a, "a"); ggml_set_name(w, "w"); ggml_set_name(b, "b");
        ggml_tensor * n = ggml_group_norm(ctx, a, num_groups, eps);
        ggml_tensor * m = ggml_mul(ctx, n, w);
        ggml_tensor * out = ggml_add(ctx, m, b);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_L2_NORM
struct test_l2_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_l2_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            float eps = 1e-12f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_l2_norm(ctx, a, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ACC
struct test_acc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, ne_b);
    }

    test_acc(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {256, 17, 1, 1},
            std::array<int64_t, 4> ne_b = {256, 16, 1, 1})
        : type(type), ne_a(ne_a), ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_set_param(b);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], b->nb[1]);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD
struct test_pad : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, pad_0, pad_1);
    }

    test_pad(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 512, 1, 1},
            int pad_0 = 1, int pad_1 = 1)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_pad(ctx, a, pad_0, pad_1, 0, 0);
        ggml_set_name(out, "out");

        return out;
    }
};

struct test_pad_ext : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int lp0;
    const int rp0;
    const int lp1;
    const int rp1;
    const int lp2;
    const int rp2;
    const int lp3;
    const int rp3;
    const bool v;

    std::string vars() override {
        return VARS_TO_STR11(type, ne_a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, v);
    }

    test_pad_ext(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 512, 3, 1},
            int lp0 = 1, int rp0 = 1, int lp1 = 1, int rp1 = 1,
            int lp2 = 1, int rp2 = 1, int lp3 = 1, int rp3 = 1,
            bool v = false)
        : type(type), ne_a(ne_a), lp0(lp0), rp0(rp0), lp1(lp1), rp1(rp1), lp2(lp2), rp2(rp2), lp3(lp3), rp3(rp3), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        if (v) {
            a = ggml_view_4d(ctx, a, (a->ne[0] + 1) / 2, (a->ne[1] + 1) / 2, (a->ne[2] + 1) / 2, (a->ne[3] + 1) / 2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD_REFLECT_1D
struct test_pad_reflect_1d : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, pad_0, pad_1);
    }

    test_pad_reflect_1d(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 34, 2, 1},
            int pad_0 = 10, int pad_1 = 9)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 2, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_pad_reflect_1d(ctx, a, pad_0, pad_1);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ROLL
struct test_roll : public test_case {
    const int shift0;
    const int shift1;
    const int shift3;
    const int shift4;

    std::string vars() override {
        return VARS_TO_STR4(shift0, shift1, shift3, shift4);
    }

    test_roll(int shift0 = 3, int shift1 = -2, int shift3 = 1, int shift4 = -1)
        : shift0(shift0), shift1(shift1), shift3(shift3), shift4(shift4) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        int64_t ne[4] = {10, 5, 4, 3};
        ggml_tensor * a = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_roll(ctx, a, shift0, shift1, shift3, shift4);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ARANGE
struct test_arange : public test_case {
    const ggml_type type;
    const float start;
    const float stop;
    const float step;

    std::string vars() override {
        return VARS_TO_STR4(type, start, stop, step);
    }

    test_arange(ggml_type type = GGML_TYPE_F32,
            float start = 0.f, float stop = 10.f, float step = 1.f)
        : type(type), start(start), stop(stop), step(step)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = ggml_arange(ctx, start, stop, step);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_TIMESTEP_EMBEDDING
struct test_timestep_embedding : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int dim;
    const int max_period;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, dim, max_period);
    }

    test_timestep_embedding(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {2, 1, 1, 1},
            int dim = 320, int max_period=10000)
        : type(type), ne_a(ne_a), dim(dim), max_period(max_period)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_timestep_embedding(ctx, a, dim, max_period);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_LEAKY_RELU
struct test_leaky_relu : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const float negative_slope;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, negative_slope);
    }

    test_leaky_relu(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 4, 3},
            float negative_slope = 0.1f)
        : type(type), ne_a(ne_a), negative_slope(negative_slope)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_leaky_relu(ctx, a, negative_slope, true);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_FLASH_ATTN_EXT
struct test_flash_attn_ext : public test_case {
    const int64_t hsk; // K head size
    const int64_t hsv; // V head size
    const int64_t nh; // num heads
    const std::array<int64_t, 2> nr23; // repeat in dim 2 and 3, tests for grouped-query attention
    const int64_t kv; // kv size
    const int64_t nb; // batch size

    const bool mask; // use mask
    const bool sinks; // use sinks

    const float max_bias; // ALiBi
    const float logit_softcap; // Gemma 2

    const ggml_prec prec;
    const ggml_type type_KV;
    std::array<int32_t, 4> permute;

    std::string vars() override {
        return VARS_TO_STR13(hsk, hsv, nh, nr23, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_KV, permute);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Just counting matmul costs:
        // Q*K^T is nb x hsk x kv, P*V is nb x kv x hsv, per head
        return (2 * nh*nr23[0] * nb * (hsk + hsv) * kv)*nr23[1];
    }

    test_flash_attn_ext(int64_t hsk = 128, int64_t hsv = 128, int64_t nh = 32, std::array<int64_t, 2> nr23 = {1, 1}, int64_t kv = 96, int64_t nb = 8,
                        bool mask = true, bool sinks = false, float max_bias = 0.0f, float logit_softcap = 0.0f, ggml_prec prec = GGML_PREC_F32,
                        ggml_type type_KV = GGML_TYPE_F16, std::array<int32_t, 4> permute = {0, 1, 2, 3})
        : hsk(hsk), hsv(hsv), nh(nh), nr23(nr23), kv(kv), nb(nb), mask(mask), sinks(sinks), max_bias(max_bias), logit_softcap(logit_softcap), prec(prec), type_KV(type_KV), permute(permute) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_KV));
        const int64_t hsv_padded = GGML_PAD(hsv, ggml_blck_size(type_KV));

        auto const &create_permuted = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, bool is_view) -> ggml_tensor * {
            int64_t ne[4] = {ne0, ne1, ne2, ne3};
            int64_t ne_perm[4];
            for (int i = 0; i < 4; ++i) {
                ne_perm[permute[i]] = ne[i];
            }
            ggml_tensor * t;
            if (is_view) {
                ggml_tensor * t0 = ggml_new_tensor_4d(ctx, type, ne_perm[0], 2*ne_perm[1], ne_perm[2], ne_perm[3]);
                t = ggml_view_4d(ctx, t0, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3], t0->nb[1], t0->nb[2], t0->nb[3], 0);
            } else {
                t = ggml_new_tensor_4d(ctx, type, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3]);
            }
            if (permute != std::array<int32_t, 4>{0, 1, 2, 3}) {
                t = ggml_permute(ctx, t, permute[0], permute[1], permute[2], permute[3]);
            }
            return t;
        };

        ggml_tensor * q = create_permuted(GGML_TYPE_F32, hsk_padded, nb, nh*nr23[0], nr23[1], false);
        ggml_set_name(q, "q");

        ggml_tensor * k = create_permuted(type_KV,       hsk_padded, kv, nh,         nr23[1], true); // the K tensor is usually a view of the K cache
        ggml_set_name(k, "k");

        ggml_tensor * v = create_permuted(type_KV,       hsv_padded, kv, nh,         nr23[1], true); // the V tensor is usually a view of the V cache
        ggml_set_name(v, "v");

        ggml_tensor * m = nullptr;
        if (mask) {
            m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, nr23[1]);
            ggml_set_name(m, "m");
        }

        ggml_tensor * s = nullptr;
        if (sinks) {
            s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q->ne[2]);
            ggml_set_name(s, "s");
        }

        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hsk), max_bias, logit_softcap);
        ggml_flash_attn_ext_add_sinks(out, s);
        ggml_flash_attn_ext_set_prec (out, prec);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "s") == 0) {
                // make the sink values more noticable in order to trigger a test failure when the implementation is wrong
                init_tensor_uniform(t, -10.0f, 10.0f);
            } else if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_CROSS_ENTROPY_LOSS
struct test_cross_entropy_loss : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cross_entropy_loss(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(logits);
        ggml_set_name(logits, "logits");

        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        // The labels are assumed to be constant -> no gradients.
        ggml_set_name(labels, "labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        ggml_set_name(labels, "labels_normalized");

        ggml_tensor * out = ggml_cross_entropy_loss(ctx, logits, labels);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // For larger abs. diffs between logits softmax is more linear, therefore more precise num. gradients.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -100.0f, 100.0f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_CROSS_ENTROPY_LOSS_BACK
struct test_cross_entropy_loss_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cross_entropy_loss_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * grad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(grad, "grad");

        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(logits, "logits");

        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(labels, "labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        ggml_set_name(labels, "labels_normalized");

        ggml_tensor * out = ggml_cross_entropy_loss_back(ctx, grad, logits, labels);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_OPT_STEP_ADAMW
struct test_opt_step_adamw : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_opt_step_adamw(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a); // Despite tensor a having gradients the output tensor will not.
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad, "grad");

        ggml_tensor * grad_m = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad_m, "grad_m");

        ggml_tensor * grad_v = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad_v, "grad_v");

        ggml_tensor * adamw_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
        ggml_set_name(adamw_params, "adamw_params");

        ggml_tensor * out = ggml_opt_step_adamw(ctx, a, grad, grad_m, grad_v, adamw_params);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 0.0f, 1.0f); // grad_v and adamw_params need non-negative values.
        }
    }

    bool grad_precise() override {
        return true;
    }
};

struct test_opt_step_sgd : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_opt_step_sgd(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);  // Despite tensor a having gradients the output tensor will not.
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad, "grad");

        ggml_tensor * sgd_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        ggml_set_name(sgd_params, "sgd_params");

        ggml_tensor * out = ggml_opt_step_sgd(ctx, a, grad, sgd_params);

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 0.0f, 1.0f);  // sgd_params need non-negative values.
        }
    }

    bool grad_precise() override {
        return true;
    }
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llama_hparams {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    static constexpr uint32_t n_layer = 1;
    uint32_t n_rot;
    uint32_t n_embd_head; // dimension of values (d_v)
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    // cparams
    static constexpr uint32_t n_ctx = 512; // user-specified context size
    static constexpr uint32_t n_ctx_orig = n_ctx;

    // batch
    int32_t n_tokens;

    // llm_build_context
    static constexpr int32_t n_kv    = 32; // size of KV cache to consider (n_kv <= n_ctx
    static constexpr int32_t kv_head = 1;  // index of where we store new KV data in the cache

    uint32_t n_embd_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head * n_head_kv;
    }
};

// LLM base class
struct test_llm : public test_case {
    llama_hparams hp;

protected:
    test_llm(llama_hparams hp)
        : hp(std::move(hp)) {
    }

public:
    struct ggml_tensor * llm_build_norm(
            struct ggml_context * ctx,
             struct ggml_tensor * cur,
             struct ggml_tensor * mw,
             struct ggml_tensor * mb,
                  llm_norm_type   type) {
        switch (type) {
            case LLM_NORM:     cur = ggml_norm    (ctx, cur, hp.f_norm_eps); break;
            case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps); break;
        }
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    void llm_build_kv_store(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * k_cur,
             struct ggml_tensor * v_cur) {
        // compute the transposed [n_tokens, n_embd] V matrix
        struct ggml_tensor * v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, hp.n_embd_gqa(), hp.n_tokens));

        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, k_l, hp.n_tokens*hp.n_embd_gqa(),
                (ggml_row_size(k_l->type, hp.n_embd_gqa()))*hp.kv_head);

        struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, v_l, hp.n_tokens, hp.n_embd_gqa(),
                (  hp.n_ctx)*ggml_element_size(v_l),
                (hp.kv_head)*ggml_element_size(v_l));

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_cpy(ctx, k_cur,   k_cache_view);
        ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    struct ggml_tensor * llm_build_kqv(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * q_cur,
             struct ggml_tensor * kq_mask,
                        float     kq_scale) {
        struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        struct ggml_tensor * k =
            ggml_view_3d(ctx, k_l,
                    hp.n_embd_head, hp.n_kv, hp.n_head_kv,
                    ggml_row_size(k_l->type, hp.n_embd_gqa()),
                    ggml_row_size(k_l->type, hp.n_embd_head),
                    0);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, v_l,
                    hp.n_kv, hp.n_embd_head, hp.n_head_kv,
                    ggml_element_size(v_l)*hp.n_ctx,
                    ggml_element_size(v_l)*hp.n_ctx*hp.n_embd_head,
                    0);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head*hp.n_head, hp.n_tokens);

        struct ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
        cur = ggml_mul_mat(ctx, wo, cur);

        return cur;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(hp.n_tokens);
                for (int i = 0; i < hp.n_tokens; i++) {
                    data[i] = rand() % hp.n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, hp.n_tokens * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Llama
struct test_llama : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;
    bool fused;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "LLAMA";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    bool run_whole_graph() override { return fused; }

    test_llama(int n_tokens = 1, bool fused = false)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 32,
            /*n_head_kv      =*/ 32,
            /*n_rot          =*/ 100,
            /*n_embd_head    =*/ 100,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 0.f,
            /*f_norm_rms_eps =*/ 1e-5f,
            /*n_tokens       =*/ n_tokens,
        })
        , fused(fused)
    {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            ggml_tensor * attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, inpL, attn_norm, nullptr, LLM_NORM_RMS);

            // self-attention
            {
                ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
                ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());
                ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);

                Qcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            // feed-forward network
            ggml_tensor * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, ffn_inp, ffn_norm, nullptr, LLM_NORM_RMS);

            ggml_tensor * ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff,   hp.n_embd);
            ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            struct ggml_tensor * tmp = ggml_mul_mat(ctx, ffn_up, cur);
            cur = ggml_mul_mat(ctx, ffn_gate, cur);
            cur = ggml_silu(ctx, cur);
            cur = ggml_mul(ctx, cur, tmp);
            cur = ggml_mul_mat(ctx, ffn_down, cur);

            cur = ggml_add(ctx, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, nullptr, LLM_NORM_RMS);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

// Falcon
struct test_falcon : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FALCON";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_falcon(int n_tokens = 1)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 50,
            /*n_head_kv      =*/ 1,
            /*n_rot          =*/ 64,
            /*n_embd_head    =*/ 64,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 1e-5f,
            /*f_norm_rms_eps =*/ 0.f,
            /*n_tokens       =*/ n_tokens,
        }) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            // norm
            ggml_tensor * attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm = llm_build_norm(ctx, inpL, attn_norm_w, attn_norm_b, LLM_NORM);

            // self-attention
            {
                cur = attn_norm;

                ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd + 2*hp.n_embd_gqa());

                cur = ggml_mul_mat(ctx, wqkv, cur);

                struct ggml_tensor * Qcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd,     hp.n_tokens, cur->nb[1], 0*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd + hp.n_embd_gqa())));

                Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx, Qcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, Kcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
                ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
                cur = attn_norm;
                cur = ggml_mul_mat(ctx, ffn_up, cur);
                cur = ggml_gelu(ctx, cur);
                cur = ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = ggml_add(ctx, cur, ffn_inp);

            cur = ggml_add(ctx, cur, inpL);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        ggml_tensor * output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, output_norm_b, LLM_NORM);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};


// ###########################################
// ## Section 3: GGML Op Test Instantiation ##
// ###########################################
static const ggml_type all_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_MXFP4,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
};

static const ggml_type base_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16,
    GGML_TYPE_Q8_0, // for I8MM tests
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, // for I8MM tests
    GGML_TYPE_Q4_K,
    GGML_TYPE_MXFP4, // TODO: or "other"
    GGML_TYPE_IQ2_XXS
};

static const ggml_type other_types[] = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
    GGML_TYPE_BF16,
};

// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // unary ops
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int v : {0, 1}) {
            for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }

    // glu ops
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int v : {0, 1}) {
            for (int op = 0; op < GGML_GLU_OP_COUNT; op++) {
                if (op == GGML_GLU_OP_SWIGLU_OAI) {
                    // SWIGLU_OAI is handled separately
                    continue;
                }

                for (bool swapped : {false, true}) {
                    test_cases.emplace_back(new test_glu((ggml_glu_op) op, type, { 128, 2, 2, 2 }, v, swapped));
                    test_cases.emplace_back(new test_glu((ggml_glu_op) op, type, { 5, 7, 11, 13 }, v, swapped));
                }

                test_cases.emplace_back(new test_glu_split((ggml_glu_op) op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_glu_split((ggml_glu_op) op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }

    for (int v : {0, 1}) {
        for (float alpha : {.5f, 1.702f}) {
            for (float limit : {2.0f, 7.0f}) {
                test_cases.emplace_back(new test_swiglu_oai(GGML_TYPE_F32, { 128, 2, 2, 2 }, v, alpha, limit));
            }
        }
    }

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_0}) {
        test_cases.emplace_back(new test_get_rows(type, 300*256,   5,         4,   1,   2, false));
        test_cases.emplace_back(new test_get_rows(type,     256,   80000, 70000,   2,   1, false));
        test_cases.emplace_back(new test_get_rows(type,     256,   5,         4, 700, 100, false));
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, 1, v));
            }
        }
    }
    for (int b : {1, 7}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 256, 5, 4, b, 1, v));
        }
    }

    test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows_back(type, 256, 5, 4, 1, v));
        }
    }
    for (bool v : {false, true}) {
        test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_I32, 256, 5, 4, 1, v));
    }

    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F32, GGML_TYPE_I64, { 1, 8, 1, 3 }, { 1, 1 }, 2, false));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F32, GGML_TYPE_I32, { 1, 8, 1, 3 }, { 1, 1 }, 2, false));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_Q8_0, GGML_TYPE_I32, { 256, 5, 1, 3 }, { 1, 1, }, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_set_rows(type, GGML_TYPE_I64, { 256, 5,  b, 3 }, { 1, 1, }, 1, v));
                test_cases.emplace_back(new test_set_rows(type, GGML_TYPE_I64, { 256, 11, 1, b }, { 2, 3, }, 7, v));

                test_cases.emplace_back(new test_set_rows(type, GGML_TYPE_I64, { 3*ggml_blck_size(type), 3, b, 1 }, { 2, 3, }, 2, v));

                if (ggml_blck_size(type) == 1) {
                    test_cases.emplace_back(new test_set_rows(type, GGML_TYPE_I64, { 31, 3, b, 1 }, { 2, 3, }, 2, v));
                    test_cases.emplace_back(new test_set_rows(type, GGML_TYPE_I64, { 33, 5, 1, b }, { 2, 3, }, 1, v));
                }
            }
        }
    }

    for (ggml_type type_input : {GGML_TYPE_F32}) {
        for (ggml_op_pool pool_type : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
            for (int k0 : {1, 3}) {
                for (int k1 : {1, 3}) {
                    for (int s0 : {1, 2}) {
                        for (int s1 : {1, 2}) {
                            for (int p0 : {0, 1}) {
                                for (int p1 : {0, 1}) {
                                    test_cases.emplace_back(new test_pool2d(pool_type, type_input, {10, 10, 3, 1}, k0, k1, s0, s1, p0, p1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#if 0
    // >4GB im2col destination. Too slow to run by default.
    // Test cases taken from Wan2.1 T2V 1.3B.
    test_cases.emplace_back(new test_im2col   (GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {832, 480, 192, 4}, {3, 3, 192, 96}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {834, 482, 6, 96},  {3, 3,3, 9216}, 96, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
#endif

    // im2col 1D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    for (int s0 : {1, 3}) {
        for (int p0 : {0, 3}) {
            for (int d0 : {1, 3}) {
                test_cases.emplace_back(new test_im2col(
                    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {20, 2, 2, 1}, {3, 2, 2, 1},
                    s0, 0, p0, 0, d0, 0, false));
            }
        }
    }

    // im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    for (int s0 : {1, 3}) {
        for (int s1 : {1, 3}) {
            for (int p0 : {0, 3}) {
                for (int p1 : {0, 3}) {
                    for (int d0 : {1, 3}) {
                        for (int d1 : {1, 3}) {
                            test_cases.emplace_back(new test_im2col(
                                GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {20, 20, 2, 2}, {3, 3, 2, 2},
                                s0, s1, p0, p1, d0, d1, true));
                        }
                    }
                }
            }
        }
    }

    // extra tests for im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 32}, {3, 3, 1, 32}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 32}, {3, 3, 2, 32}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 1024}, {3, 3, 1, 1024}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 1024}, {3, 3, 2, 1024}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 2048}, {3, 3, 1, 2048}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 2048}, {3, 3, 2, 2048}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 2560}, {3, 3, 1, 2560}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 2560}, {3, 3, 2, 2560}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {5, 5, 1, 32}, {3, 4, 1, 32}, 1, 1, 0, 0, 1, 1, true));

    // im2col 3D
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    for (int s0 : {1, 3}) {
        for (int s1 : {1, 3}) {
            for (int s2 : {1, 3}) {
                for (int p0 : {0, 3}) {
                    for (int p1 : {0, 3}) {
                        for (int p2 : {0, 3}) {
                            for (int d0 : {1, 3}) {
                                for (int d1 : {1, 3}) {
                                    for (int d2 : {1, 3}) {
                                        for (int IC : {1, 3}) {
                                            for (bool v : {false, true}) {
                                                test_cases.emplace_back(new test_im2col_3d(
                                                    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {20, 20, 10, 3}, {3, 3, 3, 3},
                                                    IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, v));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

// Conv_2D test cases
#ifdef DETAILED_TESTS
    // Probably we do not have enough time to execute these in the pipeline.
    uint32_t iwh_idx  = 0;
    uint32_t kwh_idx  = 1;
    uint32_t Cout_idx = 2;
    uint32_t Cin_idx  = 3;
    uint32_t B_idx    = 4;

    std::vector<std::array<int, 5>> cases = {
  //{IWH, KWH, Cout, Cin, B}
  // K=CRS=NPQ=4096 conv_2d matmul performance
        {19,   4, 4096, 256, 16},
 // K=128, CRS=128, NPQ=4096
        { 19,  4, 128,  8,   16},
 // K=130, CRS=128, NPQ=4096
        { 19,  4, 130,  8,   16},
 // Edge case: K x CRS is small
        { 19,  2, 4,    4,   16},
 // A ConvNet's first layer
        { 224, 3, 8,    3,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel
        { 224, 2, 8,    1,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel, several images in the batch
        { 224, 2, 8,    1,   8 },
 // A middle layer of a ConvNet
        { 58,  3, 64,   32,  1 },
 // A middle layer of a ConvNet, several images in the batch
        { 58,  3, 64,   32,  8 },
 // A deep layer of a ConvNet, several images in the batch
        { 16,  3, 256,  128, 8 }
    };

    for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (auto act_case : cases) {
            test_cases.emplace_back(new test_conv_2d(
                { act_case[iwh_idx], act_case[iwh_idx], act_case[Cin_idx], act_case[B_idx] },
                { act_case[kwh_idx], act_case[kwh_idx], act_case[Cin_idx], act_case[Cout_idx] },
                kernel_type, 1, 1, 0, 0, 1, 1, false));
        }
    }
#endif

    // CONV_2D:
    auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
    };

    //uint32_t s0 = 3;
    uint32_t s1 = 5;
    uint32_t p0 = 5;
    //uint32_t p1 = 2;
    uint32_t d0 = 2;
    uint32_t d1 = 4;

    for (uint32_t s0 : { 1, 3 }) {
        for (uint32_t p1 : { 2, 5 }) {
            for (uint32_t Cin : { 1, 25 }) {
                for (uint32_t Cout : { 1, 12 }) {
                    for (uint32_t KH : { 1, 2, 3, 11 }) {
                        for (uint32_t KW : { 1, 2, 3, 11 }) {
                            for (uint32_t H : { 1, 133 }) {
                                for (uint32_t W : { 1, 141 }) {
                                    if (calc_conv_output_size(W, KW, s0, p0, d0) > 0 &&
                                        calc_conv_output_size(H, KH, s1, p1, d1) > 0) {
                                        for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                                            test_cases.emplace_back(new test_conv_2d(
                                                { W, H, Cin, 2 }, { KW, KH, Cin, Cout }, kernel_type, s0, s1, p0, p1, d0, d1, false));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // sycl backend will limit task global_range < MAX_INT
    // test cases for 2D im2col with large input W and H (occurs in stable-diffusion)
    // however these cases need to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // these cases are verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));

    test_cases.emplace_back(new test_conv_2d_dw({17, 34, 9, 1}, {3, 3, 1, 9}, 1, 0, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({17, 34, 9, 1}, {3, 3, 1, 9}, 1, 0, 1, true));
    test_cases.emplace_back(new test_conv_2d_dw({32, 8, 64, 1}, {3, 3, 1, 64}, 2, 1, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({32, 8, 64, 1}, {3, 3, 1, 64}, 2, 1, 1, true));

    // CONV_3D
    auto calc_conv_output_size_3d = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
    };

    for (ggml_type kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int N : {1, 2}) {
            for (int IC : {1, 3}) {
                for (int OC : {1, 4}) {
                    for (int s0 : {1, 2}) {
                        for (int p1 : {0, 1}) {
                            for (int d2 : {1, 2}) {
                                int64_t IW = 20, IH = 22, ID = 18;
                                int64_t KW = 3,  KH = 3,  KD = 3;
                                int s1 = s0, s2 = s0;
                                int p0 = p1, p2 = p1;
                                int d0 = d2, d1 = d2;

                                if (calc_conv_output_size_3d(IW, KW, s0, p0, d0) <= 0 ||
                                    calc_conv_output_size_3d(IH, KH, s1, p1, d1) <= 0 ||
                                    calc_conv_output_size_3d(ID, KD, s2, p2, d2) <= 0) {
                                    continue;
                                }
                                test_cases.emplace_back(new test_conv_3d(
                                    N, IC, ID, IH, IW,
                                    OC, KD, KH, KW,
                                    s0, s1, s2, p0, p1, p2, d0, d1, d2,
                                    kernel_type));

                                // Asymmetric kernel and params
                                int64_t asym_KW = 5, asym_KH = 1, asym_KD = 3;
                                int asym_s0 = 2, asym_s1 = 1, asym_s2 = 1;
                                int asym_p0 = 2, asym_p1 = 0, asym_p2 = 1;
                                int asym_d0 = 1, asym_d1 = 1, asym_d2 = 2;

                                if (calc_conv_output_size_3d(IW, asym_KW, asym_s0, asym_p0, asym_d0) <= 0 ||
                                    calc_conv_output_size_3d(IH, asym_KH, asym_s1, asym_p1, asym_d1) <= 0 ||
                                    calc_conv_output_size_3d(ID, asym_KD, asym_s2, asym_p2, asym_d2) <= 0) {
                                    continue;
                                }
                                test_cases.emplace_back(new test_conv_3d(
                                    N, IC, ID, IH, IW,
                                    OC, asym_KD, asym_KH, asym_KW,
                                    asym_s0, asym_s1, asym_s2, asym_p0, asym_p1, asym_p2, asym_d0, asym_d1, asym_d2,
                                    kernel_type));
                            }
                        }
                    }
                }
            }
        }
        // Case with kernel size 1
        test_cases.emplace_back(new test_conv_3d(1, 4, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type));
    }

    for(uint32_t Cout : {1, 9}){
        for(uint32_t Cin : {1, 7}){
            for(uint32_t K : {1, 3, 1337}){
                for(uint32_t L : {1, 2, 13}){
                    for(uint32_t s0: {1, 2, 3}){
                        test_cases.emplace_back(new test_conv_transpose_1d({L,Cin,1,1}, {K,Cout,Cin,1}, s0, 0, 1));
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,1,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({2,1,1,1}, {3,1,1,1}, 1, 0, 1));

    test_cases.emplace_back(new test_conv_transpose_2d({3, 2, 3, 1}, {2, 2, 1, 3}, 1));
    test_cases.emplace_back(new test_conv_transpose_2d({10, 10, 9, 1}, {3, 3, 1, 9}, 2));

    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, {4,  500, 1, 1}));
    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, {4, 5000, 1, 1}));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32,    1, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32,  513, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {100,  10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {1024, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {1024, 12, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {2000, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {5438,  3, 1, 1}));

    for (int ne3 : {1, 3}) { // CUDA backward pass only supports ne3 == 1
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {2, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 2, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 1, 2, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 1, 1, 2}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I32, {10, 5, 4, ne3}, {2, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I16, {10, 5, 4, ne3}, {1, 1, 1, 2}));
    }

    for (bool view : {false, true}) {
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 1, 1, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {2, 1, 1, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 2, 1, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 1, 2, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 1, 1, 2}, view));
    }

    test_cases.emplace_back(new test_dup(GGML_TYPE_F32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {0, 2, 1, 3})); // dup by rows
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {1, 0, 2, 3})); // dup dst not-contiguous
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10,  8, 3, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10,  8, 3, 1}, {1, 2, 0, 3}));

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_F32, GGML_TYPE_F32, {6, 5, 4, 3}, dim));
    }

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_I32, GGML_TYPE_I32, {6, 5, 4, 3}, dim));
    }

    // same-type copy
    for (ggml_type type : all_types) {
        const auto nk = ggml_blck_size(type);

        for (int k = 1; k < 4; ++k) {
            test_cases.emplace_back(new test_cpy(type, type, {k*nk, 2, 3, 4}));
            test_cases.emplace_back(new test_cpy(type, type, {k*nk, 2, 3, 4}, {0, 2, 1, 3}));
            test_cases.emplace_back(new test_cpy(type, type, {k*nk, 2, 3, 4}, {0, 3, 1, 2}, {0, 2, 1, 3}));
        }
    }

    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : all_types) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 4, 4, 4}));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {0, 2, 1, 3})); // cpy by rows
        }
    }
    for (ggml_type type_src : all_types) {
        for (ggml_type type_dst : {GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 4, 4, 4}));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {0, 2, 1, 3})); // cpy by rows
        }
    }
    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {1, 0, 2, 3})); // cpy not-contiguous
        }
    }
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_I32, {256, 2, 3, 4}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_I32, {256, 2, 3, 4}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_I32, GGML_TYPE_F32, {256, 2, 3, 4}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_I32, GGML_TYPE_F32, {256, 2, 3, 4}, {1, 0, 2, 3}));

    test_cases.emplace_back(new test_cont());
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {2, 1, 1 ,1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {2, 1, 3 ,5}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {2, 3, 5 ,7}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, {2, 1, 1 ,1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, {2, 1, 3 ,5}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, {2, 3, 5 ,7}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, {2, 1, 1 ,1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, {2, 1, 3 ,5}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, {2, 3, 5 ,7}));

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr) {
        for (auto op : {ggml_add, ggml_sub, ggml_mul, ggml_div}) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr));
        }
    };
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        add_test_bin_bcast(type, {1, 1, 8, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1, 1}, {32, 1, 1, 1});
        add_test_bin_bcast(type, {1, 1, 320, 320}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {10, 5, 1, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {10, 5, 4, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {2, 1, 1, 1});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {1, 2, 1, 1});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {1, 1, 2, 1});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {1, 1, 1, 2});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {1, 1, 2, 2});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {1, 2, 2, 2});
        add_test_bin_bcast(type, {10, 5, 4, 3}, {2, 2, 2, 2});

        // test case for k_bin_bcast_unravel in CUDA backend
        add_test_bin_bcast(type, {1, 1, 65536, 1}, {256, 1, 1, 1});

        // stable diffusion
        add_test_bin_bcast(type, {1280, 1, 1, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1280, 1, 1, 1}, {1, 16, 16, 1});
        add_test_bin_bcast(type, {1280, 16, 16, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1280, 1, 1, 1}, {1, 256, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1280, 1}, {16, 16, 1, 1});
        add_test_bin_bcast(type, {16, 16, 1280, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1920, 1}, {16, 16, 1, 1});
        add_test_bin_bcast(type, {1, 1, 2560, 1}, {16, 16, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1280, 1}, {32, 32, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1920, 1}, {32, 32, 1, 1});
        add_test_bin_bcast(type, {1, 1, 640, 1}, {32, 32, 1, 1});
        add_test_bin_bcast(type, {5120, 1, 1, 1}, {1, 256, 1, 1});
        add_test_bin_bcast(type, {640, 1, 1, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {64, 262144, 1, 1}, {1, 1, 1, 1});
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {1, 1, 1, 1});
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {2, 1, 1, 1});
    }

    // single inplace tests, especially important for WebGPU backend since kernels for inplace vs. not are different
    test_cases.emplace_back(new test_bin_bcast(ggml_add_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));
    test_cases.emplace_back(new test_bin_bcast(ggml_mul_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));
    test_cases.emplace_back(new test_bin_bcast(ggml_sub_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));
    test_cases.emplace_back(new test_bin_bcast(ggml_div_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));

    // fusion
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {2, 1, 1, 1}, 2));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 2, 1, 1}, 3));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {1, 1, 2, 1}, 4));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 2}, 5));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {1, 1, 2, 2}, 6));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {1, 2, 2, 2}, 7));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {2, 2, 2, 2}, 8));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));

    test_cases.emplace_back(new test_add1());
    test_cases.emplace_back(new test_scale());
    test_cases.emplace_back(new test_scale(GGML_TYPE_F32, {10, 10, 10, 10}, 2.0f, 1.0f));
    test_cases.emplace_back(new test_scale(GGML_TYPE_F32, {10, 10, 10, 10}, 2.0f, 1.0f, true)); // inplace test
    test_cases.emplace_back(new test_scale(GGML_TYPE_F32, {100, 10, 10, 10}, 2.0f, 1.0f));
    test_cases.emplace_back(new test_softcap(GGML_TYPE_F32, {10, 10, 10, 10}, 50.0f));
    test_cases.emplace_back(new test_silu_back());

    for (float eps : {0.0f, 1e-6f, 1e-4f, 1e-1f}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_norm    (GGML_TYPE_F32, {64, 5, 4, 3}, v, eps));
            test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {64, 5, 4, 3}, v, eps));
        }
        test_cases.emplace_back(new test_rms_norm_back(GGML_TYPE_F32, {64, 5, 4, 3}, eps));
        test_cases.emplace_back(new test_l2_norm      (GGML_TYPE_F32, {64, 5, 4, 3}, eps));
    }

    // in-place tests
    test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {64, 5, 4, 3}, false, 1e-6f, true));

    for (float eps : {0.0f, 1e-6f, 1e-4f, 1e-1f, 1.0f}) {
        test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {64, 5, 4, 3}, eps, false));
        test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {64, 5, 4, 3}, eps, true));
        test_cases.emplace_back(new test_norm_mul_add(GGML_TYPE_F32, {64, 5, 4, 3}, eps, false));
        test_cases.emplace_back(new test_norm_mul_add(GGML_TYPE_F32, {64, 5, 4, 3}, eps, true));
    }
    for (uint32_t n : {1, 511, 1025, 8192, 33*512}) {
        for (bool multi_add : {false, true}) {
            test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {n, 1, 1, 1}, 1e-6f, false, multi_add));
        }
    }

    test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, {64, 5, 4, 3}, 1e-12f));

    for (int64_t d_conv : {3, 4}) {
        for (int64_t d_inner: {1024, 1536, 2048}) {
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {4, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}));
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {8, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}));
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {4, d_inner, 4, 1}, {d_conv, d_inner, 1, 1}));
        }
    }

    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 16, 1, 1024, 1, 32, 4)); // Mamba-1
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 32, 4)); // Mamba-2
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 256, 64,  8, 2, 32, 4)); // Falcon-H1

    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 128, 4));

#if 0
    // > 4GB A matrix. Too slow to be enabled by default.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16,  900000,  3, 2592, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 1700000, 96, 2592, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 1700000,  3, 2592, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 1700000,  1, 2592, {1, 1}, {1, 1}));
#endif

    for (ggml_type type_a : all_types) {
        for (int i = 1; i < 10; ++i) {
            test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 16,  i, 256, { 1,  1}, {1, 1}));
        }
    }

#if 0
    {
        // Test paths in OpenCL
        std::vector<int> ns = {32, 64, 128, 256, 512, 1024, 4096};
        std::vector<int> ks = {896, 1536, 4096};
        for (auto n : ns) {
            for (auto k : ks) {
                test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 1024, n, k, {1, 1}, {1, 1}));
            }
        }
    }
#endif

#if 1
    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            std::vector<int> ks = { 256 };
            if (ggml_blck_size(type_a) == 1) {
                ks.push_back(4);
            }
            for (auto k : ks) {
                // test cases without permutation
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {1, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {1, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {1, 1}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {2, 2}));

                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {1, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {1, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {1, 1}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {2, 2}));

                // test cases with permutation
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {2, 3}, {1, 1}, {0, 2, 1, 3}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {2, 3}, {1, 1}, {0, 1, 3, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));

                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, k, {2, 3}, {1, 1}, {0, 2, 1, 3}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, k, {2, 3}, {1, 1}, {0, 1, 3, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));

                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {2, 3}, {1, 1}, {0, 2, 1, 3}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {2, 3}, {1, 1}, {0, 1, 3, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));
            }

            // test cases with large ne00/ne10 to cover stream-k fixup
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, 1024, {3, 2}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, 1024, {3, 2}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 1024, {3, 2}, {1, 1}));

            // test cases with large batch size
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, {1536, 1}, {1, 1}));
        }
    }
    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            if (ggml_blck_size(type_a) != 256) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, ggml_blck_size(type_a), {1,  1}, {1, 1}));
            }
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {1,  1}, {1, 1}));
        }
    }
#else
    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1,  1}, {1, 1}));
            }
        }
    }
#endif

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,  128, { 8,  1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,  128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 45, 128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 45,  64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056, 1, 193, {1,  1}, {4, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056, 1, 67,  {1,  1}, {4, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 16, 32, 32, { 1,  1}, {1, 1}, {0, 1, 2, 3}, true, 3));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, 77, {12,1}, {1,1}));

#if 0
    // test the mat-mat path for Metal
    for (int k = 1; k < 512; ++k) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 127, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 127, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 77, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 128, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 128, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, false, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, true, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F32, GGML_TYPE_F32, 16, 16, false, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F32, GGML_TYPE_F32, 16, 16, true, 50, 200, k));
    }
#endif

    for (auto bs2 : {1,3}) {
        for (auto bs : {1,2,4,8}) {
            for (auto nr : {1,4}) {
                for (uint32_t m = 0; m < 2; ++m) {
                    for (uint32_t k = 0; k < 2; ++k) {
                        for (ggml_type type: {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
                            test_cases.emplace_back(new test_mul_mat(type, GGML_TYPE_F32, 1056 + m, 1, 128 + k,  {bs,  bs2}, {nr, 1}, {0, 2, 1, 3}));
                            test_cases.emplace_back(new test_mul_mat(type, GGML_TYPE_F32, 128 + m,  1, 1056 + k, {bs,  bs2}, {nr, 1}, {0, 1, 2, 3}, true));
                        }
                    }
                }
            }
        }
    }

    // sycl backend will limit task global_range < MAX_INT
    // test case for f16-type-convert-to-fp32 kernel with large k under fp32 compute dtype (occurs in stable-diffusion)
    // however this case needs to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // this case is verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 512, 262144, 9216, {1, 1}, {1, 1}));

    // test large experts*tokens
    for (bool b : {false, true}) {
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, b, 32, 1024, 16));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 2, 2, b, 32, 8192, 64));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, b, 50, 200, 64));
    }

    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 1, 1, false, 8, 16, 1));
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, false, 32, 32, 32, 3));

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (bool b : {false, true}) {
                        for (int n : {1, 4, 5, 17, 32, 129}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4}) {
                for (int n_used : {2}) {
                    for (bool b : {false}) {
                        for (int n : {1, 32}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (int n : {1, 16}) {
                for (int k : {1, 16}) {
                    for (int bs2 : {1, 3}) {
                        for (int bs3 : {1, 3}) {
                            for (int nr2 : {1, 2}) {
                                for (int nr3 : {1, 2}) {
                                    test_cases.emplace_back(new test_out_prod(type_a, type_b, 256, n, k, {bs2, bs3}, {nr2, nr3}));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // add_id
    for (ggml_type type_a : {GGML_TYPE_F32}) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (int n_embd : {32, 129}) {
                        for (int n_token : {1, 32, 129}) {
                            test_cases.emplace_back(new test_add_id(type_a, type_b, n_embd, n_mats, n_used, n_token));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_sqr       (type));
        test_cases.emplace_back(new test_sqrt      (type));
        test_cases.emplace_back(new test_log       (type));
        test_cases.emplace_back(new test_sin       (type));
        test_cases.emplace_back(new test_cos       (type));
        test_cases.emplace_back(new test_clamp     (type));
        test_cases.emplace_back(new test_leaky_relu(type));
        test_cases.emplace_back(new test_floor     (type));
        test_cases.emplace_back(new test_ceil      (type));
        test_cases.emplace_back(new test_round     (type));
        test_cases.emplace_back(new test_trunc     (type));
        test_cases.emplace_back(new test_sqr       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_sqrt      (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_log       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_sin       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_cos       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_clamp     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_leaky_relu(type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_floor     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_ceil      (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_round     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_trunc     (type, {7, 1, 5, 3}));
    }

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 1, 1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 3, 1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 3, 2}, 5));

#if 0
    std::uniform_int_distribution<> dist_ne1(1, 50);
    int exponent = 1;
    while (exponent < (1 << 17)) {
        std::uniform_int_distribution<> dist_ne0(exponent, 2*exponent);

        for (int n = 0; n < 10; ++n) {
            int64_t ne0 = dist_ne0(rng);
            int64_t ne1 = dist_ne1(rng);
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, GGML_TYPE_F32, {ne0, ne1, 1, 1}, n/2 == 0, 0.1f, ne0 < 1000 ? 4.0f : 0.0f));
        }

        exponent <<= 1;
    }
#endif
    for (bool mask : {false, true}) {
        for (bool sinks : {false, true}) {
            for (float max_bias : {0.0f, 8.0f}) {
                if (!mask && max_bias > 0.0f) continue;
                for (float scale : {1.0f, 0.1f}) {
                    for (int64_t ne0 : {16, 1024}) {
                        for (int64_t ne1 : {16, 1024}) {
                            if (mask) {
                                for (ggml_type m_prec : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                                    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, mask, sinks, m_prec, {1, 1}, scale, max_bias));
                                    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, sinks, m_prec, {1, 1}, scale, max_bias));

                                    if (ne0 <= 32 && ne1 <= 32) {
                                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 3}, mask, sinks, m_prec, {3, 1}, scale, max_bias));
                                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, sinks, m_prec, {2, 3}, scale, max_bias));
                                    }
                                }
                            } else {
                                /* The precision of mask here doesn't matter as boolean mask is false */
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, mask, sinks, GGML_TYPE_F32, {1, 1}, scale, max_bias));
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, sinks, GGML_TYPE_F32, {1, 1}, scale, max_bias));
                            }
                        }
                    }
                }
            }
            // inplace tests
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, mask, sinks, GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f, true));
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, mask, sinks, GGML_TYPE_F16, {1, 1}, 0.1f, 0.0f, true));
        }
    }
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, true,  true,  GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, true,  false, GGML_TYPE_F16, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, false, true,  GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  true,  GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  false, GGML_TYPE_F16, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  true,  GGML_TYPE_F32, {1, 1}, 0.1f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  true,  GGML_TYPE_F16, {1, 1}, 0.1f, 8.0f));

    for (float max_bias : {0.0f, 8.0f}) {
        for (float scale : {1.0f, 0.1f}) {
            for (int64_t ne0 : {16, 1024}) {
                for (int64_t ne1 : {16, 1024}) {
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, scale, max_bias));
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, scale, max_bias));
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, {ne0,   ne1,   2, 3}, scale, max_bias));
                }
            }
        }
    }

    for (bool fw : {true, false}) { // fw == forward
        bool all = true;

        for (float fs : { 1.0f, 1.4245f }) {
            for (float ef : { 0.0f, 0.7465f }) {
                for (float af : { 1.0f, 1.4245f }) {
                    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                        for (bool ff : {false, true}) { // freq_factors
                            for (float v : { 0, 1 }) {
                                test_cases.emplace_back(new test_rope(type, {128,  32, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 7B

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, {128,  40, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 13B
                                    test_cases.emplace_back(new test_rope(type, {128,  52, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 30B
                                    test_cases.emplace_back(new test_rope(type, {128,  64, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 65B
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 64,   1, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,  71, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,   8, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)

                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 4, 1},  32, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));

                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (stablelm)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 4, 1},  32, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 2B)
                                    test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 7B)
                                    test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1},  20, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1},  32, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, { 80,  16, 2, 1},  80, GGML_ROPE_TYPE_VISION, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl ViT)
                                }

                                test_cases.emplace_back(new test_rope(type, { 64, 128, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                            }
                        }

                        all = false;
                    }
                }
            }
        }
    }

    // single inplace test per type/mode/ff
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int mode : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_VISION}) {
            for (bool ff : {false, true}) {
                test_cases.emplace_back(new test_rope(type, {128,  32, 2, 1}, 128, mode, 512, 1.4245f, 0.7465f, 1.4245f, ff, 0, true, true));
            }
        }
    }

    for (int v : { 0, 1, 2, 3 }) {
        for (int dim : { 0, 1, 2, 3, }) {
            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I32, {11, 12, 13, 14}, 7, dim, v));
        }
    }

    for (ggml_sort_order order : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {8, 1, 1, 1}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {16, 10, 10, 10}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {60, 10, 10, 10}, order)); // qwen
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {1024, 1, 1, 1}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {16384, 1, 1, 1}, order)); // bailingmoe2 (group selection)
    }

    for (ggml_scale_mode mode : {GGML_SCALE_MODE_NEAREST, GGML_SCALE_MODE_BILINEAR}) {
        test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, {512, 512, 3, 2}, 2, mode));
        test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, {512, 512, 3, 2}, 2, mode, true));
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {2, 5,  7, 11}, {5, 7, 11, 13}, mode));
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {5, 7, 11, 13}, {2, 5,  7, 11}, mode));
    }
    test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {2, 5,  7, 11}, {5, 7, 11, 13}, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS));
    test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {1, 4, 3, 2}, {2, 8, 3, 2}, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS));
    test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {4, 1, 3, 2}, {1, 1, 3, 2}, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS));

    test_cases.emplace_back(new test_sum());
    test_cases.emplace_back(new test_sum_rows());
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, {11, 5, 6, 3}, {0, 2, 1, 3}));  // row-contiguous but non-contiguous
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, {11, 5, 6, 3}, {0, 3, 2, 1}));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, {11, 5, 6, 3}, {0, 1, 3, 2}));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 11, 5, 6, 3 }, true, false));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 11, 5, 6, 3 }, false, true));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 11, 5, 6, 3 }, true, true));
    test_cases.emplace_back(new test_mean());
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 1, 1, 1 }));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 33, 1, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 33, 1, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 1024, 1, 1 }));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 33, 1024, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 256, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 256, 1, 1 }, { 1, 0, 2, 3 })); // sum dst not-contiguous
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 33, 256, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 33, 256, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 32769, 1, 1, 1 }));
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, {64, 64, 320, 1}));
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, {9, 9, 1280, 1}));
    test_cases.emplace_back(new test_group_norm_mul_add(GGML_TYPE_F32, {64, 64, 320, 1}));
    test_cases.emplace_back(new test_group_norm_mul_add(GGML_TYPE_F32, {9, 9, 1280, 1}));
    test_cases.emplace_back(new test_acc());
    test_cases.emplace_back(new test_pad());
    test_cases.emplace_back(new test_pad_ext());
    test_cases.emplace_back(new test_pad_reflect_1d());
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 384, 4, 1}));
    test_cases.emplace_back(new test_roll());
    test_cases.emplace_back(new test_arange());
    test_cases.emplace_back(new test_timestep_embedding());
    test_cases.emplace_back(new test_leaky_relu());

    for (bool v : {false, true}) {
        test_cases.emplace_back(new test_pad_ext(GGML_TYPE_F32, {512, 512, 1, 1}, 0, 1, 0, 1, 0, 0, 0, 0, v));
        test_cases.emplace_back(new test_pad_ext(GGML_TYPE_F32, {11, 22, 33, 44}, 1, 2, 3, 4, 5, 6, 7, 8, v));
    }

    for (int hsk : { 40, 64, 80, 96, 128, 192, 256, 576 }) {
        for (int hsv : { 40, 64, 80, 96, 128, 192, 256, 512 }) {
            if (hsk != 192 && hsk != 576 && hsk != hsv) continue;
            if (hsk == 192 && (hsv != 128 && hsv != 192)) continue;
            if (hsk == 576 && hsv != 512) continue; // DeepSeek MLA

            for (bool mask : { true, false } ) {
                for (bool sinks : { true, false } ) {
                    for (float max_bias : { 0.0f, 8.0f }) {
                        if (!mask && max_bias > 0.0f) continue;
                        for (float logit_softcap : {0.0f, 10.0f}) {
                            if (hsk != 128 && logit_softcap != 0.0f) continue;
                            for (int nh : { 4, }) {
                                for (int nr3 : { 1, 3, }) {
                                    if (hsk > 64 && nr3 > 1) continue; // skip broadcast for large head sizes
                                    for (int nr2 : { 1, 4, 16 }) {
                                        if (nr2 == 16 && hsk != 128) continue;
                                        //for (int kv : { 1, 17, 31, 33, 61, 113, 65, 127, 129, 130, 255, 260, 371, 380, 407, 512, 1024, }) {
                                        for (int kv : { 113, 512, 1024, }) {
                                            if (nr2 != 1 && kv != 512) continue;
                                            for (int nb : { 1, 3, 32, 35, }) {
                                                for (ggml_prec prec : {GGML_PREC_F32, GGML_PREC_DEFAULT}) {
                                                    if (hsk != 128 && prec == GGML_PREC_DEFAULT) continue;
                                                    for (ggml_type type_KV : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
                                                        test_cases.emplace_back(new test_flash_attn_ext(
                                                                    hsk, hsv, nh, {nr2, nr3}, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_KV));
                                                        // run fewer test cases permuted
                                                        if (mask == true && max_bias == 0.0f && logit_softcap == 0 && kv == 512) {
                                                            test_cases.emplace_back(new test_flash_attn_ext(
                                                                        hsk, hsv, nh, {nr2, nr3}, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_KV, {0, 2, 1, 3}));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_cross_entropy_loss     (GGML_TYPE_F32, {   10, 5, 4, 3}));
    test_cases.emplace_back(new test_cross_entropy_loss     (GGML_TYPE_F32, {30000, 1, 1, 1}));
    test_cases.emplace_back(new test_cross_entropy_loss_back(GGML_TYPE_F32, {   10, 5, 4, 3}));
    test_cases.emplace_back(new test_cross_entropy_loss_back(GGML_TYPE_F32, {30000, 1, 1, 1}));

    test_cases.emplace_back(new test_opt_step_adamw(GGML_TYPE_F32, {10, 5, 4, 3}));
    test_cases.emplace_back(new test_opt_step_sgd(GGML_TYPE_F32, {10, 5, 4, 3}));

    for (ggml_type type : base_types) {
        for (bool with_gate : {false, true}) {
            for (bool use_id : {false, true}) {
                for (bool b : {false, true}) {
                    if (!use_id && b) {
                        continue;
                    }
                    for (bool with_bias : {false, true}) {
                        if (!with_gate && !with_bias) {
                            continue;
                        }
                        for (ggml_glu_op glu_op : {GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU}) {
                            if (!with_bias && glu_op == GGML_GLU_OP_SWIGLU_OAI) {
                                continue;
                            }
                            if (!with_gate && glu_op != GGML_GLU_OP_SWIGLU) {
                                continue;
                            }
                            test_cases.emplace_back(new test_mul_mat_vec_fusion(type, glu_op, 1, 32, 256,
                                use_id, 16, 8, b, with_bias, with_gate));
                        }
                    }
                }
            }
        }
    }

    for (bool with_norm : {false, true}) {
        test_cases.emplace_back(new test_topk_moe({8, 22, 1, 1}, 4, with_norm));
        test_cases.emplace_back(new test_topk_moe({32, 22, 1, 1}, 8, with_norm));
        test_cases.emplace_back(new test_topk_moe({128, 1, 1, 1}, 128, with_norm));
    }

    test_cases.emplace_back(new test_topk_moe({ 8, 22, 1, 1 }, 4, /*with_norm*/ false, /*delayed_softmax*/ true));
    test_cases.emplace_back(new test_topk_moe({ 32, 22, 1, 1 }, 8, /*with_norm*/ false, /*delayed_softmax*/ true));

#if 0
    // these tests are disabled to save execution time, sbut they can be handy for debugging
    test_cases.emplace_back(new test_llama(2, true));
    test_cases.emplace_back(new test_llama(1));
    test_cases.emplace_back(new test_llama(2));
    test_cases.emplace_back(new test_falcon(1));
    test_cases.emplace_back(new test_falcon(2));
#endif

    return test_cases;
}

// Test cases for performance evaluation: should be representative of real-world use cases
static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
    std::vector<std::unique_ptr<test_case>> test_cases;

    // Conv2d: K=CRS=NPQ=4096 matmul performance
    uint32_t                        iwh_idx  = 0;
    uint32_t                        kwh_idx  = 1;
    uint32_t                        Cout_idx = 2;
    uint32_t                        Cin_idx  = 3;
    uint32_t                        B_idx    = 4;
    std::vector<std::array<int, 5>> cases    = {
  //{IWH, KWH, Cout, Cin, B}
  // K=CRS=NPQ=4096 conv2d matmul performance
        {19,   4, 4096, 256, 16},
 // K=128, CRS=128, NPQ=4096
        { 19,  4, 128,  8,   16},
 // K=130, CRS=128, NPQ=4096
        { 19,  4, 130,  8,   16},
 // Edge case: K x CRS is small
        { 19,  2, 4,    4,   16},
 // A ConvNet's first layer
        { 224, 3, 8,    3,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel
        { 224, 2, 8,    1,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel, several images in the batch
        { 224, 2, 8,    1,   8 },
 // A middle layer of a ConvNet
        { 58,  3, 64,   32,  1 },
 // A middle layer of a ConvNet, several images in the batch
        { 58,  3, 64,   32,  8 },
 // A deep layer of a ConvNet, several images in the batch
        { 16,  3, 512,  128, 8 },
    };

    for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (auto act_case : cases) {
            // Direct CONV_2D
            test_cases.emplace_back(new test_conv_2d(
                { act_case[iwh_idx], act_case[iwh_idx], act_case[Cin_idx], act_case[B_idx] },
                { act_case[kwh_idx], act_case[kwh_idx], act_case[Cin_idx], act_case[Cout_idx] },
                kernel_type, 1, 1, 0, 0, 1, 1, false));
        }
    }

    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {4096, 1, 1, 1}, {1,   1, 1, 1}));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {4096, 1, 1, 1}, {1, 512, 1, 1}));

    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F16,  {512, 3072, 1, 1}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F32,  {8192, 512, 2, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F32,  {3072, 512, 2, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_Q4_0, {8192, 512, 2, 1}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_Q4_0, GGML_TYPE_F32,  {8192, 512, 2, 1}));

    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {4096, 4096, 5, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {12888, 256, 5, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {77, 4096, 5, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {1024, 1024, 10, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {77, 1024, 10, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {256, 256, 20, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {64, 64, 20, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {77, 64, 20, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {1024, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32000, 512, 1, 1}));

    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {512, 34, 2, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 80, 1, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 80, 4, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 384, 1, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 384, 4, 1}));

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 16416, 1, 128, {8,  1}, {4, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 1, 16416, {8,  1}, {4, 1}, {0, 1, 2, 3}, true));

    for (int bs : {1, 2, 3, 4, 5, 8, 512}) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 4096, bs, 14336, {1,  1}, {1, 1}));
            }
        }
    }

    // qwen3-30b-a3b
    for (int bs : {1, 4, 8, 32, 64, 128, 256, 512}) {
        for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, 128, 8, false, 768, bs, 2048, 1));
            }
        }
    }

    for (int bs : {1, 4, 8, 32, 64, 128, 256, 512}) {
        for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, 32, 4, false, 1792, bs, 2048, 1));
            }
        }
    }


    // gpt-oss-20b
    for (int bs : {1, 4, 8, 512}) {
        for (ggml_type type_a : {GGML_TYPE_MXFP4}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, 32, 4, false, 2880, bs, 2880, 1));
            }
        }
    }

    for (int K : {3, 5}) {
        for (int IC : {256, 2560}) {
            for (int IW_IH : {32, 64, 256}) {
                if (IC == 2560 && IW_IH == 256) {
                    // too big
                    continue;
                }
                test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {IW_IH, IW_IH, IC, 1}, {K, K, IC, 1}, 1, 1, 1, 1, 1, 1, true));
            }
        }
    }

    for (int kv : { 4096, 8192, 16384, }) {
        for (int hs : { 64, 128, }) {
            for (int nr : { 1, 4, }) {
                test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 8, {nr, 1}, kv, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16));
            }
        }
    }

    test_cases.emplace_back(new test_conv_2d_dw({512, 512, 256, 1}, {3, 3, 1, 256}, 1, 1, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({512, 512, 256, 1}, {3, 3, 1, 256}, 1, 1, 1, true));

    test_cases.emplace_back(new test_conv_transpose_2d({256, 256, 256, 1}, {3, 3, 16, 256}, 1));
    test_cases.emplace_back(new test_conv_transpose_2d({16, 16, 16, 1}, {3, 3, 8, 16}, 1));
    test_cases.emplace_back(new test_conv_transpose_2d({10, 10, 9, 1}, {3, 3, 1, 9}, 2));

    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, {256, 256, 3, 1}));


    for (int n_token : {1, 512}) {
        test_cases.emplace_back(new test_add_id(GGML_TYPE_F32, GGML_TYPE_F32, 2880, 128, 4, n_token));
        test_cases.emplace_back(new test_add_id(GGML_TYPE_F32, GGML_TYPE_F32, 2880, 32, 4, n_token));
    }

    std::vector<std::array<int64_t, 4>> reduce_rows_cases = {
        { 8192, 1,    1, 1 },
        { 8192, 8192, 1, 1 },
        { 128,  8192, 1, 1 },
    };

    for (auto it: reduce_rows_cases){
        test_cases.emplace_back(new test_mean(GGML_TYPE_F32, it));
        test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, it));
        test_cases.emplace_back(new test_sum(GGML_TYPE_F32, it));
    }

    return test_cases;
}

static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_names_filter, const char * params_filter,
                         printer * output_printer) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases, const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        if (backend_cpu == NULL) {
            test_operation_info info("", "", "CPU");
            info.set_error("backend", "Failed to initialize CPU backend");
            output_printer->print_operation(info);
            return false;
        }

        size_t n_ok = 0;
        size_t                   tests_run = 0;
        std::vector<std::string> failed_tests;
        for (auto & test : test_cases) {
            test_status_t status = test->eval(backend, backend_cpu, op_names_filter, output_printer);
            if (status == test_status_t::SKIPPED || status == test_status_t::NOT_SUPPORTED) {
                continue;
            }
            tests_run++;
            if (status == test_status_t::OK) {
                n_ok++;
            } else if (status == test_status_t::FAIL) {
                failed_tests.push_back(test->current_op_name + "(" + test->vars() + ")");
            }
        }
        output_printer->print_summary(test_summary_info(n_ok, tests_run, false));
        output_printer->print_failed_tests(failed_tests);

        ggml_backend_free(backend_cpu);

        return n_ok == tests_run;
    }

    if (mode == MODE_GRAD) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval_grad(backend, op_names_filter, output_printer)) {
                n_ok++;
            }
        }
        output_printer->print_summary(test_summary_info(n_ok, test_cases.size(), false));

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        auto test_cases = make_test_cases_perf();
        filter_test_cases(test_cases, params_filter);
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_names_filter, output_printer);
        }
        return true;
    }

    if (mode == MODE_SUPPORT) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        for (auto & test : test_cases) {
            test->eval_support(backend, op_names_filter, output_printer);
        }
        return true;
    }

    GGML_ABORT("fatal error");
}

static void list_all_ops() {
    printf("GGML operations:\n");
    std::set<std::string> all_ops;

    for (int i = 1; i < GGML_OP_COUNT; i++) {
        all_ops.insert(ggml_op_name((enum ggml_op)i));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        all_ops.insert(ggml_unary_op_name((enum ggml_unary_op)i));
    }
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        all_ops.insert(ggml_glu_op_name((enum ggml_glu_op)i));
    }
    for (const auto & op : all_ops) {
        printf("  %s\n", op.c_str());
    }
    printf("\nTotal: %zu operations\n", all_ops.size());
}

static void show_test_coverage() {
    std::set<std::string> all_ops;
    for (int i = 1; i < GGML_OP_COUNT; i++) {
        auto op = (enum ggml_op)i;
        if (op == GGML_OP_VIEW      ||
            op == GGML_OP_RESHAPE   ||
            op == GGML_OP_PERMUTE   ||
            op == GGML_OP_TRANSPOSE ||
            op == GGML_OP_CONT      ||
            op == GGML_OP_GLU       ||
            op == GGML_OP_UNARY) {
            continue;
        }
        all_ops.insert(ggml_op_name(op));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        all_ops.insert(ggml_unary_op_name((enum ggml_unary_op)i));
    }
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        all_ops.insert(ggml_glu_op_name((enum ggml_glu_op)i));
    }
    auto test_cases = make_test_cases_eval();
    std::set<std::string> tested_ops;

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };

    for (auto & test_case : test_cases) {
        ggml_context * ctx = ggml_init(params);
        if (ctx) {
            test_case->mode = MODE_TEST;
            ggml_tensor * out = test_case->build_graph(ctx);
            if (out && out->op != GGML_OP_NONE) {
                if (out->op == GGML_OP_UNARY) {
                    tested_ops.insert(ggml_unary_op_name(ggml_get_unary_op(out)));
                } else if (out->op == GGML_OP_GLU) {
                    tested_ops.insert(ggml_glu_op_name(ggml_get_glu_op(out)));
                } else {
                    tested_ops.insert(ggml_op_name(out->op));
                }
            }
            ggml_free(ctx);
        }
    }
    std::set<std::string> covered_ops;
    std::set<std::string> uncovered_ops;
    for (const auto & op : all_ops) {
        if (tested_ops.count(op) > 0) {
            covered_ops.insert(op);
        } else {
            uncovered_ops.insert(op);
        }
    }

    printf("Operations covered by tests (%zu):\n", covered_ops.size());
    for (const auto & op : covered_ops) {
        printf("  ✓ %s\n", op.c_str());
    }
    printf("\nOperations without tests (%zu):\n", uncovered_ops.size());
    for (const auto & op : uncovered_ops) {
        printf("  ✗ %s\n", op.c_str());
    }

    printf("\nCoverage Summary:\n");
    printf("  Total operations: %zu\n", all_ops.size());
    printf("  Tested operations: %zu\n", covered_ops.size());
    printf("  Untested operations: %zu\n", uncovered_ops.size());
    printf("  Coverage: %.1f%%\n", (double)covered_ops.size() / all_ops.size() * 100.0);
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output <console|sql|csv>] [--list-ops] [--show-coverage]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("      - grad (compare gradients from backpropagation with method of finite differences)\n");
    printf("      - perf (performance evaluation)\n");
    printf("      - support (probe backend operation support)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc),\n");
    printf("        optionally including the full test case string (e.g. \"ADD(type=f16,ne=[1,1,8,1],nr=[1,1,1,1],nf=1)\")\n");
    printf("    --output specifies output format (default: console, options: console, sql, csv)\n");
    printf("    --list-ops lists all available GGML operations\n");
    printf("    --show-coverage shows test coverage\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    output_formats output_format = CONSOLE;
    const char * op_names_filter = nullptr;
    const char * backend_filter = nullptr;
    const char * params_filter = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "grad") == 0) {
            mode = MODE_GRAD;
        } else if (strcmp(argv[i], "support") == 0) {
            mode = MODE_SUPPORT;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_names_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                params_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                if (!output_format_from_str(argv[++i], output_format)) {
                    usage(argv);
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--list-ops") == 0) {
            list_all_ops();
            return 0;
        } else if (strcmp(argv[i], "--show-coverage") == 0) {
            show_test_coverage();
            return 0;
        } else {
            usage(argv);
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    // Create printer for output format
    std::unique_ptr<printer> output_printer = create_printer(output_format);
    if (output_printer) {
        output_printer->print_header();
    }

    output_printer->print_testing_start(testing_start_info(ggml_backend_dev_count()));

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            output_printer->print_backend_init(
                backend_init_info(i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping"));
            n_ok++;
            continue;
        }

        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && mode != MODE_GRAD) {
            output_printer->print_backend_init(backend_init_info(
                i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping CPU backend"));
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            // TODO: better value for n_threads
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency());
        }

        size_t free, total;  // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        output_printer->print_backend_init(backend_init_info(i, ggml_backend_dev_count(), ggml_backend_dev_name(dev),
                                                             false, "", ggml_backend_dev_description(dev),
                                                             total / 1024 / 1024, free / 1024 / 1024, true));

        bool ok = test_backend(backend, mode, op_names_filter, params_filter, output_printer.get());

        if (ok) {
            n_ok++;
        }
        output_printer->print_backend_status(
            backend_status_info(ggml_backend_name(backend), ok ? test_status_t::OK : test_status_t::FAIL));

        ggml_backend_free(backend);
    }

    ggml_quantize_free();

    if (output_printer) {
        output_printer->print_footer();
    }

    output_printer->print_overall_summary(
        overall_summary_info(n_ok, ggml_backend_dev_count(), n_ok == ggml_backend_dev_count()));

    if (n_ok != ggml_backend_dev_count()) {
        return 1;
    }

    return 0;
}
