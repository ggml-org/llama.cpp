#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Parametric Tile format spec.
//
// Single source of truth shared by four consumers, which must never drift:
//   - the evolutionary genome (per-family format search)
//   - the per-tensor GGUF metadata record
//   - the parametric CPU-reference quant/dequant
//   - the parametric Metal kernel (within the layout envelope below)
//
// Layout params change the on-wire storage and the kernel memory layout, so
// they are bounded to the kernel envelope and encoded in metadata. Scalar
// params are pure numbers the kernel takes as runtime args; they are unbounded
// and the evolutionary search ranges over them freely.
typedef struct ts_format_spec {
    // layout params - bounded by the kernel envelope, encoded in metadata
    int32_t page_size;        // shared-scale granularity (elements per page scale)
    int32_t lane_size;        // sub-page scale group (elements per lane scale)
    int32_t lane_scale_bits;  // per-lane scale width in bits
    // scalar params - unbounded runtime params
    float threshold_mult;     // ternary threshold = mean(|W|) x threshold_mult
    float outlier_frac;       // fraction of |W| carved out as fp16 outliers
    float awq_alpha;          // AWQ activation-aware scaling exponent (0 = off)
} ts_format_spec;

// The current production format (T640). The default spec MUST reproduce the
// existing quant/dequant output bit-identically; this is the tripwire for the
// bit-equivalence test.
static inline ts_format_spec ts_format_spec_default(void) {
    ts_format_spec f;
    f.page_size       = 640;
    f.lane_size       = 20;
    f.lane_scale_bits = 8;
    f.threshold_mult  = 1.0f;
    f.outlier_frac    = 0.005f;
    f.awq_alpha       = 0.0f;
    return f;
}

// Layout envelope the parametric Metal kernel supports. The evolutionary
// genome's layout genes are bounded to this set; scalar genes are unbounded.
// Widen here only when a matching kernel parametrization lands (see
// docs/parametric-kernel-design.md).
static inline bool ts_format_spec_in_envelope(const ts_format_spec * f) {
    const bool page_ok = (f->page_size == 320 || f->page_size == 640 || f->page_size == 1280);
    const bool lane_ok = (f->lane_size == 16 || f->lane_size == 20 || f->lane_size == 32);
    const bool bits_ok = (f->lane_scale_bits == 4 || f->lane_scale_bits == 8);
    return page_ok && lane_ok && bits_ok;
}

#ifdef __cplusplus
}
#endif
