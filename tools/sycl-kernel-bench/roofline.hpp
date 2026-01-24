#pragma once

#include <cstdint>

namespace sycl_bench {

struct RooflineEstimate {
    double flops = 0.0;
    double bytes = 0.0;
    double arithmetic_intensity = 0.0;
};

inline RooflineEstimate estimate_mmvq_roofline(int64_t m,
                                               int64_t k,
                                               int64_t batch,
                                               size_t weight_bytes,
                                               size_t activation_bytes,
                                               size_t output_bytes) {
    RooflineEstimate out;
    out.flops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(batch);
    out.bytes = static_cast<double>(weight_bytes + activation_bytes + output_bytes);
    if (out.bytes > 0.0) {
        out.arithmetic_intensity = out.flops / out.bytes;
    }
    return out;
}

}  // namespace sycl_bench
