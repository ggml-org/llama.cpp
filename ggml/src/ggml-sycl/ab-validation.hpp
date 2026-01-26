#pragma once
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>

namespace ggml_sycl {

// Tolerance settings for numerical comparison
struct ValidationTolerance {
    float absolute_fp32 = 1e-6f;
    float absolute_fp16 = 1e-4f;
    float relative_pct = 0.1f;  // 0.1% relative tolerance
};

// Validation result
struct ValidationResult {
    bool passed = true;
    int total_elements = 0;
    int mismatches = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int worst_index = -1;
    float unified_at_worst = 0.0f;
    float reference_at_worst = 0.0f;
};

// Check if A/B validation is enabled
inline bool is_ab_validation_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_VALIDATE");
        enabled = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

// Get validation tolerance based on data type
inline ValidationTolerance get_tolerance(bool is_fp16) {
    ValidationTolerance tol;
    if (is_fp16) {
        tol.absolute_fp32 = 1e-4f;
        tol.relative_pct = 0.5f;  // Higher tolerance for FP16
    }
    return tol;
}

// Compare two values within tolerance
inline bool values_match(float unified, float reference, const ValidationTolerance& tol) {
    float abs_diff = std::abs(unified - reference);

    // Absolute tolerance
    if (abs_diff <= tol.absolute_fp32) return true;

    // Relative tolerance
    float magnitude = std::max(std::abs(reference), std::abs(unified));
    if (magnitude > 0) {
        float rel_diff = abs_diff / magnitude;
        if (rel_diff <= tol.relative_pct / 100.0f) return true;
    }

    return false;
}

// Compare output arrays
inline ValidationResult compare_outputs(
    const float* unified,
    const float* reference,
    int count,
    const ValidationTolerance& tol = {})
{
    ValidationResult result;
    result.total_elements = count;

    for (int i = 0; i < count; i++) {
        float abs_diff = std::abs(unified[i] - reference[i]);
        float magnitude = std::max(std::abs(reference[i]), std::abs(unified[i]));
        float rel_diff = (magnitude > 0) ? abs_diff / magnitude : 0.0f;

        // Track maximum differences
        if (abs_diff > result.max_abs_diff) {
            result.max_abs_diff = abs_diff;
            result.worst_index = i;
            result.unified_at_worst = unified[i];
            result.reference_at_worst = reference[i];
        }
        if (rel_diff > result.max_rel_diff) {
            result.max_rel_diff = rel_diff;
        }

        // Check if values match
        if (!values_match(unified[i], reference[i], tol)) {
            result.mismatches++;
        }
    }

    result.passed = (result.mismatches == 0);
    return result;
}

// Log validation result
inline void log_validation_result(
    const ValidationResult& result,
    int64_t M, int64_t N, int64_t K,
    const char* quant_type)
{
    if (result.passed) {
        fprintf(stderr, "[VALIDATE] PASS: M=%lld N=%lld K=%lld type=%s max_diff=%.2e\n",
                (long long)M, (long long)N, (long long)K, quant_type, result.max_abs_diff);
    } else {
        fprintf(stderr, "[VALIDATE] FAIL: M=%lld N=%lld K=%lld type=%s\n",
                (long long)M, (long long)N, (long long)K, quant_type);
        fprintf(stderr, "  mismatches: %d / %d (%.2f%%)\n",
                result.mismatches, result.total_elements,
                100.0f * result.mismatches / result.total_elements);
        fprintf(stderr, "  max_abs_diff: %.6e at index %d\n",
                result.max_abs_diff, result.worst_index);
        fprintf(stderr, "  max_rel_diff: %.2f%%\n", result.max_rel_diff * 100.0f);
        fprintf(stderr, "  unified[%d]=%.6f, reference[%d]=%.6f\n",
                result.worst_index, result.unified_at_worst,
                result.worst_index, result.reference_at_worst);
    }
}

// A/B validation context for tracking results
class ABValidationContext {
public:
    void record_result(const ValidationResult& result) {
        total_validations_++;
        if (result.passed) {
            passed_++;
        } else {
            failed_++;
        }
        max_abs_diff_seen_ = std::max(max_abs_diff_seen_, result.max_abs_diff);
        total_mismatches_ += result.mismatches;
        total_elements_ += result.total_elements;
    }

    int get_passed() const { return passed_; }
    int get_failed() const { return failed_; }
    int get_total() const { return total_validations_; }
    float get_pass_rate() const {
        return total_validations_ > 0
            ? 100.0f * passed_ / total_validations_ : 0.0f;
    }
    float get_max_diff() const { return max_abs_diff_seen_; }
    float get_mismatch_rate() const {
        return total_elements_ > 0
            ? 100.0f * total_mismatches_ / total_elements_ : 0.0f;
    }

    void log_summary() const {
        fprintf(stderr, "\n=== A/B Validation Summary ===\n");
        fprintf(stderr, "Total validations: %d\n", total_validations_);
        fprintf(stderr, "Passed: %d (%.1f%%)\n", passed_, get_pass_rate());
        fprintf(stderr, "Failed: %d\n", failed_);
        fprintf(stderr, "Max absolute diff: %.6e\n", max_abs_diff_seen_);
        fprintf(stderr, "Element mismatch rate: %.4f%%\n", get_mismatch_rate());
    }

    void reset() {
        total_validations_ = 0;
        passed_ = 0;
        failed_ = 0;
        max_abs_diff_seen_ = 0.0f;
        total_mismatches_ = 0;
        total_elements_ = 0;
    }

private:
    int total_validations_ = 0;
    int passed_ = 0;
    int failed_ = 0;
    float max_abs_diff_seen_ = 0.0f;
    int64_t total_mismatches_ = 0;
    int64_t total_elements_ = 0;
};

} // namespace ggml_sycl
