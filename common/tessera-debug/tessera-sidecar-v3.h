#pragma once

//
// tessera-sidecar-v3.h
//
// Reader for the Tessera v3 dequant sidecar format (TDQT magic,
// version 3). The writer already lives in tessera-debug.cpp; this
// adds the read path for quantize-time L1.5 reference consumption.
//
// Header `dtype`:
//   - 0 (DEQUANT_DTYPE_F32)  data is F32, rows*cols*4 bytes
//   - 1 (DEQUANT_DTYPE_F16)  data is FP16, rows*cols*2 bytes
// Both the L1 `.dequant.f32` and the L1.5 `.act.dequant.f32` /
// `.act.dequant.f16` sidecars use this reader. FP16 is upcast to F32
// in `data` on read so downstream metrics can stay dtype-agnostic;
// use `header.dtype` to detect the on-disk dtype if the caller
// needs the original.
//

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

struct ts_sidecar_v3_header {
    uint32_t magic;             // "TDQT"
    uint32_t version;           // 3
    int64_t  rows;
    int64_t  cols;
    uint32_t dtype;             // 0 = F32 (DEQUANT_DTYPE_F32), 1 = FP16 (DEQUANT_DTYPE_F16)
    float    outlier_threshold;
    int64_t  outlier_count_total;
};

struct ts_sidecar_v3_row_meta {
    uint64_t timing_ns;
    uint32_t kernel_id;
    uint32_t dispatch_count;
    uint64_t reserved;
};

struct ts_sidecar_v3 {
    ts_sidecar_v3_header header;
    std::vector<int32_t> row_outlier_counts;    // (rows,)
    std::vector<ts_sidecar_v3_row_meta> row_meta;  // (rows,)
    std::vector<float> data;                    // (rows x cols,) row-major F32
                                               // (FP16 is upcast to F32 on read)
};

// Read a v3 sidecar file. Returns 0 on success, -1 on error.
// On error, err_msg (if non-null) receives a description. F32 and FP16
// data are both supported; FP16 is decoded into `out->data` as F32.
int ts_sidecar_v3_read(const char * path, ts_sidecar_v3 * out,
                       std::string * err_msg);

// Read only the header (cheap probe for format validation).
int ts_sidecar_v3_read_header(const char * path, ts_sidecar_v3_header * hdr);

// Compute SHA-256 of the sidecar file (for receipt provenance).
// out must be 32 bytes.
void ts_sidecar_v3_sha256(const char * path, uint8_t * out);
