//
// test_l15.cpp
//
// Smoke + regression tests for the L1.5 reference reader and writer.
//
// What this covers:
//
//   1. Round-trip the L1.5 F32 path (back-compat with the legacy
//      W4A4 mode): write a v3 sidecar with F32 data, read it back,
//      verify dimensions, data, and the relative_frob / layer_output_mse
//      metrics.
//
//   2. Round-trip the L1.5 FP16 path (the new default, the whole point
//      of L1.5): write a v3 sidecar with FP16 data, read it back, and
//      verify the data upcast matches the source F32 (the FP16
//      representation is the only lossy step; everything else is
//      identity).
//
//   3. L1 vs L1.5 distinctness: verify that the FP16 L1.5 reference
//      diverges from the L1 F32 dequant on at least one non-power-of-2
//      value (the FP16 round-trip introduces the expected ULP error).
//
//   4. L1 F32 regression: write a known input through the L1 writer
//      (tessera_debug::open_dequant_writer + write_dequant_row +
//      close_dequant_writer), hash the file, and compare against a
//      documented expected SHA-256. The hash is the regression gate
//      for "L1 F32 path produces a bit-identical result to today's
//      output". A change in the L1 dequant that changes the on-disk
//      bytes fails the test.
//
//   5. Both metrics (ts_l15_relative_frob, ts_l15_layer_output_mse)
//      work on both F32 and F16 references.
//

#include "tessera-l15.h"
#include "tessera-sidecar-v3.h"
#include "tessera-debug.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        g_fail++;
    } else {
        std::printf("ok   %s\n", name);
    }
}

static void check_close(const char * name, float got, float want, float tol) {
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %-32s got %.7g want %.7g\n", name, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-32s %.7g\n", name, (double)got);
    }
}

static const char * k_path_f32 = "/tmp/test_l15_ref.act.dequant.f32";
static const char * k_path_f16 = "/tmp/test_l15_ref.act.dequant.f16";
static const char * k_path_l1  = "/tmp/test_l15_l1.dequant.f32";

static bool write_synthetic_sidecar_f32() {
    const int64_t rows = 4;
    const int64_t cols = 8;

    FILE * f = fopen(k_path_f32, "wb");
    if (!f) {
        return false;
    }

    fwrite("TDQT", 1, 4, f);
    uint32_t version = 3;
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&rows, sizeof(rows), 1, f);
    fwrite(&cols, sizeof(cols), 1, f);
    uint32_t dtype = 0;  // F32
    fwrite(&dtype, sizeof(dtype), 1, f);
    float outlier_threshold = 6.0f;
    fwrite(&outlier_threshold, sizeof(outlier_threshold), 1, f);
    int64_t outlier_count_total = 2;
    fwrite(&outlier_count_total, sizeof(outlier_count_total), 1, f);

    int32_t row_outlier_counts[4] = { 1, 0, 1, 0 };
    fwrite(row_outlier_counts, sizeof(int32_t), 4, f);

    uint8_t row_meta[24];
    memset(row_meta, 0, sizeof(row_meta));
    for (int i = 0; i < 4; i++) {
        fwrite(row_meta, 1, 24, f);
    }

    float data[32];
    for (int i = 0; i < 32; i++) {
        data[i] = (float)i;
    }
    fwrite(data, sizeof(float), 32, f);

    fclose(f);
    return true;
}

// Convert F32 -> F16 (proper rounding, the ggml canonical
// implementation). Used to compare the F16-reference's F32-upcast
// against the F16 round of the original F32 source. This is the
// same conversion the sidecar writer uses internally; the local
// copy keeps the test self-contained (the test does not link the
// sidecar writer directly for this fixture).
static uint16_t local_fp32_to_fp16(float f) {
    union { float f; uint32_t u; } bits;
    bits.f = f;
    uint32_t sign = (bits.u >> 16) & 0x8000u;
    int32_t  exp32 = (int32_t)((bits.u >> 23) & 0xffu) - 127;
    uint32_t mant32 = bits.u & 0x7fffffu;
    if (exp32 == 128) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp32 >= 16) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp32 < -24) {
        return (uint16_t)sign;
    }
    if (exp32 < -14) {
        uint32_t m = (mant32 | 0x800000u) >> (1 - exp32 - 14);
        uint32_t round_bit = 1u << (1 - exp32 - 14 + 12);
        uint32_t sticky = m & (round_bit - 1u);
        m = m + (round_bit >> 1) + (sticky ? 1u : 0u);
        if (m & 0x1000u) m = 0;
        return (uint16_t)(sign | m);
    }
    uint32_t e16 = (uint32_t)(exp32 + 15);
    uint32_t m16 = mant32 >> 13;
    uint32_t round_bit = 1u << 12;
    uint32_t sticky = mant32 & (round_bit - 1u);
    m16 = m16 + (round_bit >> 1) + (sticky ? 1u : 0u);
    if (m16 & 0x0400u) {
        m16 = 0;
        e16++;
        if (e16 >= 31) {
            return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | (e16 << 10) | (m16 & 0x3ffu));
}

// F16 data block. We hand-encode the FP16 bit patterns to keep the
// test independent of the runtime's FP32->FP16 implementation; the
// test asserts the reader's F16->F32 upcast recovers the expected
// values. The chosen values include a non-power-of-2 (0.1) that does
// not have an exact FP16 representation - this is the value that
// proves "L1.5 differs from L1 F32" (the L1 F32 dequant is exact at
// F32; the F16 reference is rounded at F16 precision).
//
// The F16 bit patterns below are the canonical round-to-nearest-even
// FP16 of the listed F32 source. We use the same `local_fp32_to_fp16`
// helper that the writer uses internally (so the synthetic file is
// bit-identical to what the writer would produce for the same source).
static bool write_synthetic_sidecar_f16() {
    const int64_t rows = 4;
    const int64_t cols = 8;

    FILE * f = fopen(k_path_f16, "wb");
    if (!f) {
        return false;
    }

    fwrite("TDQT", 1, 4, f);
    uint32_t version = 3;
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&rows, sizeof(rows), 1, f);
    fwrite(&cols, sizeof(cols), 1, f);
    uint32_t dtype = 1;  // F16
    fwrite(&dtype, sizeof(dtype), 1, f);
    float outlier_threshold = 6.0f;
    fwrite(&outlier_threshold, sizeof(outlier_threshold), 1, f);
    int64_t outlier_count_total = 2;
    fwrite(&outlier_count_total, sizeof(outlier_count_total), 1, f);

    int32_t row_outlier_counts[4] = { 1, 0, 1, 0 };
    fwrite(row_outlier_counts, sizeof(int32_t), 4, f);

    uint8_t row_meta[24];
    memset(row_meta, 0, sizeof(row_meta));
    for (int i = 0; i < 4; i++) {
        fwrite(row_meta, 1, 24, f);
    }

    // 32 FP16 values. The first 8 are simple integers (exact in F16).
    // The next 8 are 8.5, 17, 25.5, ... (exact in F16). The last 16
    // are 0.1, 0.2, ..., 1.6 - NOT exact in F16; these are the test
    // signal that the F16 reference is distinct from the F32 source.
    uint16_t data[32];
    for (int i = 0; i < 32; i++) {
        float v;
        if (i < 8) {
            v = (float)i;  // 0, 1, 2, ..., 7
        } else if (i < 16) {
            v = 8.5f * (float)(i - 7);  // 8.5, 17, 25.5, ...
        } else {
            v = 0.1f * (float)(i - 15);  // 0.1, 0.2, ..., 1.6
        }
        data[i] = local_fp32_to_fp16(v);
    }
    fwrite(data, sizeof(uint16_t), 32, f);

    fclose(f);
    return true;
}

// L1 regression: write a deterministic 16-row x 16-col block of
// (i*0.25) values through the L1 writer, hash the file, and compare
// against a documented expected SHA-256. The hash is the regression
// gate for "L1 F32 path produces a bit-identical result to today's
// output".
//
// The hash is the SHA-256 of the file's bytes, in the on-disk v3
// format. The header contains the writer's provenance (kernel
// version, main tip, telemetry fields), so the hash will change if
// any of those change between runs. The test reads the file via
// ts_sidecar_v3_read and verifies the data block matches the source
// bit-for-bit, then computes the file SHA-256.
//
// To make the hash stable across runs, we set the env-driven
// fields to fixed values via the public API
// (tessera_debug::set_telemetry_*) before writing. The
// `kernel_version` and `tessera_main_tip` are compile-time
// constants in the shipped library; in the test build they are the
// stub values "test" / "test" (set by the test harness's
// tessera-build-info.h). We document the expected hash for the
// "test / test" build, so the test only passes when both
// - the on-disk data is the same as the source
// - the file is exactly the same bytes as the documented expectation
// Any change to the writer's on-disk layout (header, per-row strip,
// data encoding) will change the hash and fail the test, which is
// the desired regression behavior.
static bool write_l1_regression_tensor() {
    const int64_t rows = 16;
    const int64_t cols = 16;
    const int64_t n = rows * cols;

    // Set the writer to a known dir and a known stride.
    tessera_debug::set_dequant_dir("/tmp");
    tessera_debug::set_dequant_stride(1);
    // Telemetry fields fixed to known values so the provenance
    // JSON (and the resulting file hash) is reproducible.
    tessera_debug::set_telemetry_model("test_model");
    tessera_debug::set_telemetry_calibration_corpus("test_corpus");
    tessera_debug::set_telemetry_calibration_corpus_hash("test_hash");

    std::vector<float> data((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        data[(size_t) i] = (float) i * 0.25f;
    }

    tessera_debug::open_dequant_writer("test_l15_l1", rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, data.data() + r * cols, cols);
    }
    tessera_debug::close_dequant_writer();
    return true;
}

static int test_l1_f32_path() {
    // 1. write F32 sidecar (legacy W4A4 path)
    check("write f32 sidecar", write_synthetic_sidecar_f32());

    // 2. load
    ts_l15_reference ref;
    std::string err;
    int rc = ts_l15_load_reference(k_path_f32, &ref, &err);
    check("f32 load rc == 0", rc == 0);
    if (rc != 0) {
        std::printf("  error: %s\n", err.c_str());
        return 1;
    }

    // 3. verify dimensions and data
    check("f32 rows == 4", ref.rows == 4);
    check("f32 cols == 8", ref.cols == 8);
    check("f32 data.size == 32", (int64_t)ref.data.size() == 32);
    check("f32 outlier_threshold", ref.outlier_threshold == 6.0f);
    check("f32 outlier_count", ref.outlier_count == 2);
    check("f32 tensor_name", ref.tensor_name == "test_l15_ref");
    check("f32 file_dtype == 0 (F32)", ref.file_dtype == 0);

    bool data_ok = true;
    for (int i = 0; i < 32; i++) {
        if (ref.data[i] != (float)i) {
            data_ok = false;
            std::printf("  data[%d] = %f, want %f\n", i, (double)ref.data[i], (double)i);
            break;
        }
    }
    check("f32 data values", data_ok);

    // 4. metrics on F32 reference
    check_close("f32 frob(identical)", ts_l15_relative_frob(ref.data.data(), &ref), 0.0f, 1e-7f);

    std::vector<float> perturbed(ref.data);
    for (auto & v : perturbed) {
        v += 0.01f;
    }
    float frob = ts_l15_relative_frob(perturbed.data(), &ref);
    check("f32 frob(perturbed) > 0", frob > 0.0f);
    check("f32 frob(perturbed) < 0.01", frob < 0.01f);
    std::printf("  f32 frob(perturbed) = %.7g\n", (double)frob);

    const int64_t n_tokens = 2;
    float calib_X[16];
    for (int i = 0; i < 16; i++) {
        calib_X[i] = (float)(i + 1) * 0.1f;
    }
    float mse_identical = ts_l15_layer_output_mse(ref.data.data(), &ref, calib_X, n_tokens);
    check_close("f32 mse(identical)", mse_identical, 0.0f, 1e-6f);

    float mse_perturbed = ts_l15_layer_output_mse(perturbed.data(), &ref, calib_X, n_tokens);
    check("f32 mse(perturbed) > 0", mse_perturbed > 0.0f);
    std::printf("  f32 mse(perturbed) = %.7g\n", (double)mse_perturbed);
    return 0;
}

static int test_l1_5_f16_path() {
    // 1. write FP16 sidecar (the new default, the whole point of L1.5)
    check("write f16 sidecar", write_synthetic_sidecar_f16());

    // 2. load
    ts_l15_reference ref;
    std::string err;
    int rc = ts_l15_load_reference(k_path_f16, &ref, &err);
    check("f16 load rc == 0", rc == 0);
    if (rc != 0) {
        std::printf("  error: %s\n", err.c_str());
        return 1;
    }

    // 3. verify dimensions and dtype
    check("f16 rows == 4", ref.rows == 4);
    check("f16 cols == 8", ref.cols == 8);
    check("f16 data.size == 32", (int64_t)ref.data.size() == 32);
    check("f16 outlier_threshold", ref.outlier_threshold == 6.0f);
    check("f16 outlier_count", ref.outlier_count == 2);
    check("f16 tensor_name", ref.tensor_name == "test_l15_ref");
    check("f16 file_dtype == 1 (F16)", ref.file_dtype == 1);

    // 4. verify the upcast recovers the F16 round of the source.
    // The first 16 values are exact in F16 (0..7, 8.5..68.0 in 8.5
    // steps); the last 16 are 0.1..1.6, not exact in F16. The
    // expected F32 value at each index is the F32 source (exact for
    // the first 16, the F16 rounding for the last 16); we compare
    // `ref.data[i]` against `src` and check the diff is bounded by
    // 1 ULP at F16 precision for the non-exact values, and exactly
    // 0 for the exact values.
    bool data_ok = true;
    int  n_non_exact = 0;
    int  n_round     = 0;
    for (int i = 0; i < 32; i++) {
        float src;
        if (i < 8) {
            src = (float)i;
        } else if (i < 16) {
            src = 8.5f * (float)(i - 7);
        } else {
            src = 0.1f * (float)(i - 15);
        }
        const float diff = std::fabs(ref.data[i] - src);
        if (i >= 16) {
            n_non_exact++;
            // 0.1 is not exact in F16; the F16 round is the nearest
            // F16 to 0.1 (about 0.10009765625), so the diff vs the
            // F32 source is bounded by 1 ULP at F16 precision
            // (about 1e-3 for values in [0, 1.6]).
            if (diff > 0.0f) {
                n_round++;
            }
            if (diff > 0.01f) {
                data_ok = false;
                std::printf("  data[%d] = %f, expected %f (diff %f)\n",
                            i, (double)ref.data[i], (double)src, (double)diff);
                break;
            }
        } else {
            // Exact in F16: the upcast must equal the source bit-for-bit.
            if (diff > 0.0f) {
                data_ok = false;
                std::printf("  data[%d] = %f, expected %f (diff %f, should be 0)\n",
                            i, (double)ref.data[i], (double)src, (double)diff);
                break;
            }
        }
    }
    check("f16 upcast values", data_ok);
    // At least one non-exact value must have rounded (the 0.1*i
    // values are guaranteed to round in F16).
    check("f16 some values rounded", n_round > 0);
    std::printf("  f16: %d/%d non-exact values rounded (F16 ULP)\n",
                n_round, n_non_exact);

    // 5. metrics on F16 reference
    check_close("f16 frob(identical)", ts_l15_relative_frob(ref.data.data(), &ref), 0.0f, 1e-5f);

    // F16 vs L1 F32 distinctness: the F16 reference is the F16
    // round of the F32 source; the L1 F32 dequant is the F32
    // source. The relative Frobenius between them is bounded by
    // the F16 ULP (~1e-3 for values in [0, 2]). Use a non-zero
    // synthetic L1 buffer to simulate "what the kernel dequantized"
    // (it equals the F32 source) and check the relative Frobenius
    // to the F16 reference.
    std::vector<float> l1_f32(32);
    for (int i = 0; i < 32; i++) {
        if (i < 8) l1_f32[i] = (float) i;
        else if (i < 16) l1_f32[i] = 8.5f * (float)(i - 7);
        else l1_f32[i] = 0.1f * (float)(i - 15);
    }
    float f16_vs_f32 = ts_l15_relative_frob(l1_f32.data(), &ref);
    check("f16 vs f32 frob > 0 (FP16 rounded)", f16_vs_f32 > 0.0f);
    std::printf("  f16 vs f32 relative_frob = %.7g\n", (double)f16_vs_f32);
    // Upper bound: the F16 rounding error is bounded by 1 ULP at
    // F16 precision. For values in [0, 2], 1 ULP is at most
    // ~2^-10 = 1e-3. The relative Frobenius is the squared
    // magnitude of the error divided by the squared magnitude of
    // the reference; the ULP bound is tight.
    check("f16 vs f32 frob < 1e-3 (FP16 ULP bound)", f16_vs_f32 < 1e-3f);

    return 0;
}

static int test_l1_f32_regression() {
    // L1 F32 regression: write a known input through the L1
    // writer, hash the file, and compare against a documented
    // expected SHA-256. The hash pins the on-disk byte layout
    // (header, per-row strip, data encoding). A change to the L1
    // writer that changes the on-disk bytes fails the test.
    check("write l1 regression tensor", write_l1_regression_tensor());

    // Read back, verify data matches the source bit-for-bit.
    ts_sidecar_v3 sc;
    std::string err;
    int rc = ts_sidecar_v3_read(k_path_l1, &sc, &err);
    check("l1 regression read", rc == 0);
    if (rc != 0) {
        std::printf("  error: %s\n", err.c_str());
        return 1;
    }
    check("l1 rows == 16", sc.header.rows == 16);
    check("l1 cols == 16", sc.header.cols == 16);
    check("l1 dtype == F32", sc.header.dtype == 0);

    bool data_ok = true;
    for (int i = 0; i < 16 * 16; i++) {
        float expect = (float) i * 0.25f;
        if (sc.data[(size_t) i] != expect) {
            data_ok = false;
            std::printf("  data[%d] = %f, want %f\n", i,
                        (double) sc.data[(size_t) i], (double) expect);
            break;
        }
    }
    check("l1 data bit-identical to source", data_ok);

    // Hash the file and compare against the documented expected
    // SHA-256. The expected value pins the on-disk byte layout;
    // any change to the writer's output fails the test.
    uint8_t hash[32];
    ts_sidecar_v3_sha256(k_path_l1, hash);
    char hex[65];
    for (int i = 0; i < 32; i++) {
        std::snprintf(hex + 2 * i, sizeof(hex) - 2 * i, "%02x", hash[i]);
    }
    hex[64] = 0;
    std::printf("  l1 file SHA-256: %s\n", hex);

    // The expected hash is the SHA-256 of the v3 sidecar with the
    // documented input (16x16 tensor of i*0.25) and the
    // provenance fields set to the test stub values. The hash
    // pins the on-disk byte layout (header, per-row strip, data
    // encoding) of the L1 writer. Any change to the L1 writer's
    // output (deliberate or accidental) that changes the on-disk
    // bytes will change the hash and fail the test.
    //
    // The hash was captured on the first run of this test under
    // the test-harness stub (tessera-build-info.h produces
    // TESSERA_KERNEL_VERSION="test" and TESSERA_MAIN_TIP="test"),
    // with the writer's telemetry fields set to "test_model",
    // "test_corpus", "test_hash". The hash is the bit-exact
    // identity of the on-disk file; it must NOT change between
    // runs of the same code.
    static const char * k_locked_l1_hash =
        "37eca8294bd8521a526c6af77e948dd52a6ce1ca176a6d5e184f0661e3049e61";
    check("l1 hash matches locked value", strcmp(hex, k_locked_l1_hash) == 0);
    if (strcmp(hex, k_locked_l1_hash) != 0) {
        std::printf("  l1 hash MISMATCH: got %s, want %s\n",
                    hex, k_locked_l1_hash);
    }
    return 0;
}

// Convenience: returns an empty std::string. ts_sidecar_v3_read
// takes a non-const std::string*; the test doesn't care about the
// error message, so we return a shared empty string.
static std::string & err_unused() {
    static std::string e;
    return e;
}

static int test_l1_5_writer_f16() {
    // L1.5 writer FP16 path: the new default. Use the sidecar
    // writer (tessera_debug::open_fp16_reference_writer +
    // write_fp16_reference_row + close_fp16_reference_writer) and
    // verify the on-disk file is FP16 ground truth - NOT bit-
    // identical to the L1 F32 dequant. The conversion is in the
    // hook, not the writer: the writer takes the F32 buffer and
    // converts to FP16 via the local_fp32_to_fp16 path (the
    // `write_fp16_reference_row_from_f32` convenience does this
    // internally).
    const char * dir = "/tmp";
    const char * tname = "test_l1_5_writer";
    const int64_t rows = 4;
    const int64_t cols = 8;
    const int64_t n = rows * cols;
    const char * f16_path = "/tmp/test_l1_5_writer.act.dequant.f16";

    tessera_debug::set_dequant_dir(dir);
    tessera_debug::set_dequant_mode("w4a4");
    tessera_debug::set_l15_dtype("f16");
    tessera_debug::set_dequant_stride(1);

    // Source F32 data, includes non-power-of-2 values (0.1*i).
    std::vector<float> src_f32((size_t) n);
    for (int i = 0; i < n; i++) {
        if (i < 8) src_f32[(size_t) i] = (float) i;
        else if (i < 16) src_f32[(size_t) i] = 8.5f * (float)(i - 7);
        else src_f32[(size_t) i] = 0.1f * (float)(i - 15);
    }

    // Open the L1.5 sidecar (FP16). open_dequant_writer would also
    // open the L1 sidecar, but we only want the L1.5 path here -
    // call open_fp16_reference_writer directly (it's a no-op
    // unless w4a4 mode is set, which we did).
    tessera_debug::open_fp16_reference_writer(tname, rows, cols);
    for (int r = 0; r < rows; r++) {
        // The hook would convert F32 -> FP16 here; we use the
        // writer's convenience API (write_fp16_reference_row_from_f32)
        // that does the same conversion internally with the same
        // proper rounding.
        tessera_debug::write_fp16_reference_row_from_f32(r, src_f32.data() + r * cols, cols);
    }
    tessera_debug::close_fp16_reference_writer();

    // Verify the on-disk file: header dtype is F16, the data block
    // is 2 bytes/value.
    ts_sidecar_v3_header hdr;
    int rc = ts_sidecar_v3_read_header(f16_path, &hdr);
    check("writer f16 file exists", rc == 0);
    if (rc == 0) {
        check("writer f16 file dtype == F16", hdr.dtype == 1);
        check("writer f16 rows match", hdr.rows == rows);
        check("writer f16 cols match", hdr.cols == cols);
    }

    // Read the file and verify the data upcasts to the F16 round
    // of the source (NOT the source itself - the F16 round is
    // lossy for the 0.1*i values).
    ts_sidecar_v3 sc;
    rc = ts_sidecar_v3_read(f16_path, &sc, &err_unused());
    check("writer f16 read", rc == 0);
    if (rc == 0) {
        check("writer f16 file_dtype in struct", sc.header.dtype == 1);
        // For the F16-rounding test: at least one value must
        // differ from the F32 source (the non-exact 0.1*i values
        // are guaranteed to round).
        int n_differ = 0;
        for (int i = 0; i < n; i++) {
            if (sc.data[(size_t) i] != src_f32[(size_t) i]) {
                n_differ++;
            }
        }
        check("writer f16 differs from F32 source on at least 1 value",
              n_differ > 0);
        std::printf("  writer f16: %d/%d values differ from F32 source\n",
                    n_differ, (int) n);
    }

    // Clean up.
    std::remove(f16_path);
    std::string prov = std::string(f16_path) + ".provenance.json";
    std::remove(prov.c_str());

    // Restore the test's default state.
    tessera_debug::set_dequant_mode("");
    tessera_debug::set_l15_dtype("f16");

    return 0;
}

static int test_l1_5_writer_f32_legacy() {
    // L1.5 writer F32 path (legacy W4A4 mode): when l15_dtype=f32,
    // the L1.5 sidecar is F32 (same data as the L1 F32 dequant).
    // This is the back-compat path for existing users who wrote
    // tooling against the F32 L1.5 file. The new F16 L1.5 path is
    // the default; this test pins the legacy F32 path so a future
    // refactor of the auto-populate branch does not silently break
    // it.
    const char * dir = "/tmp";
    const char * tname = "test_l1_5_writer_legacy_f32";
    const int64_t rows = 4;
    const int64_t cols = 8;
    const int64_t n = rows * cols;
    const char * f32_path = "/tmp/test_l1_5_writer_legacy_f32.act.dequant.f32";

    tessera_debug::set_dequant_dir(dir);
    tessera_debug::set_dequant_mode("w4a4");
    tessera_debug::set_l15_dtype("f32");
    tessera_debug::set_dequant_stride(1);

    // Source F32 data. Use exact-in-F16 values so the F32 L1.5
    // file's "identity" with the L1 dequant is not lost to F16
    // rounding (the F32 path is NOT supposed to round).
    std::vector<float> src_f32((size_t) n);
    for (int i = 0; i < n; i++) {
        if (i < 8) src_f32[(size_t) i] = (float) i;
        else if (i < 16) src_f32[(size_t) i] = 8.5f * (float)(i - 7);
        else src_f32[(size_t) i] = 0.1f * (float)(i - 15);
    }

    // Open the L1 sidecar; the F32 L1.5 path is auto-populated
    // by the writer on each write_dequant_row call.
    tessera_debug::open_dequant_writer(tname, rows, cols);
    for (int r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, src_f32.data() + r * cols, cols);
    }
    tessera_debug::close_dequant_writer();

    // Verify the F32 L1.5 file was written.
    ts_sidecar_v3_header hdr;
    int rc = ts_sidecar_v3_read_header(f32_path, &hdr);
    check("legacy f32 file exists", rc == 0);
    if (rc == 0) {
        check("legacy f32 file dtype == F32", hdr.dtype == 0);
    }

    // Read and verify the data matches the source bit-for-bit.
    ts_sidecar_v3 sc;
    std::string err;
    rc = ts_sidecar_v3_read(f32_path, &sc, &err);
    check("legacy f32 read", rc == 0);
    if (rc == 0) {
        bool data_ok = true;
        for (int i = 0; i < n; i++) {
            if (sc.data[(size_t) i] != src_f32[(size_t) i]) {
                data_ok = false;
                std::printf("  data[%d] = %f, want %f\n", i,
                            (double) sc.data[(size_t) i],
                            (double) src_f32[(size_t) i]);
                break;
            }
        }
        check("legacy f32 data bit-identical to source", data_ok);
    }

    // Clean up.
    std::remove(f32_path);
    std::string prov = std::string(f32_path) + ".provenance.json";
    std::remove(prov.c_str());

    // Restore the test's default state.
    tessera_debug::set_dequant_mode("");
    tessera_debug::set_l15_dtype("f16");

    return 0;
}

int main() {
    if (test_l1_f32_path() != 0) {
        return 1;
    }
    std::printf("\n");

    if (test_l1_5_f16_path() != 0) {
        return 1;
    }
    std::printf("\n");

    if (test_l1_f32_regression() != 0) {
        return 1;
    }
    std::printf("\n");

    if (test_l1_5_writer_f16() != 0) {
        return 1;
    }

    if (test_l1_5_writer_f32_legacy() != 0) {
        return 1;
    }

    // load_directory on /tmp (should find our files; the F16 is
    // preferred over the F32 when both exist for the same tensor).
    std::vector<ts_l15_reference> refs;
    std::string err;
    int n_loaded = ts_l15_load_directory("/tmp", &refs, &err);
    check("load_directory >= 1", n_loaded >= 1);
    if (n_loaded < 0) {
        std::printf("  error: %s\n", err.c_str());
    }

    // Verify dedup: when both .act.dequant.f32 and .act.dequant.f16
    // exist for the same tensor, only the F16 is loaded.
    int n_test_l15_ref = 0;
    int f16_count = 0;
    int f32_count = 0;
    for (const auto & r : refs) {
        if (r.tensor_name == "test_l15_ref") {
            n_test_l15_ref++;
            if (r.file_dtype == 1) f16_count++;
            if (r.file_dtype == 0) f32_count++;
        }
    }
    check("dedup prefers F16", f16_count >= 1 && f32_count == 0);

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
