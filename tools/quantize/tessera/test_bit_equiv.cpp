//
// test_bit_equiv.cpp
//
// Bit-equivalence gate for the Tessera C++ port: proves whether
// ts_ternarize_with_acts + ts_pack_tile640 produce byte-identical output to
// quantize_v3.py (ternarize_with_acts + pack_tile640 + compute_scales) on the
// same deterministic 4x640 (seed=42 Gaussian) weight page.
//
// Run order (from repo root):
//   python3 tools/quantize/tessera/test_bit_equiv.py     # writes /tmp/bit_equiv_*
//   clang++ -std=c++17 -O2 -framework Accelerate \
//       -I tools/quantize/tessera -I ggml/src \
//       tools/quantize/tessera/test_bit_equiv.cpp \
//       tools/quantize/tessera/tessera-quant.cpp \
//       tools/quantize/tessera/tessera-vec.cpp \
//       -o /tmp/test_bit_equiv && /tmp/test_bit_equiv
//
// ---------------------------------------------------------------------------
// FINDINGS (updated after fixes)
// ---------------------------------------------------------------------------
//   packed       : PASS - 512 bytes (128 u32 words) byte-identical.
//   page_scales  : tested via ts_compute_scales (not ts_pack_tile640 placeholders).
//   lane_scales  : tested via ts_compute_scales (not ts_pack_tile640 placeholders).
//
// History:
//   - Original test compared ts_pack_tile640 placeholder scales against
//     Python compute_scales -> FAIL by design. Fixed: now compares
//     ts_compute_scales output (the real fitted scales).
//   - Ternary threshold used double accumulation in C++ vs float32 in
//     Python. Fixed: C++ now uses float32 accumulation to match Python
//     bit-exactly (tessera-quant.cpp ts_ternarize_with_acts).
// ---------------------------------------------------------------------------

#include "tessera-quant.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Read exactly `expected` bytes; empty result on any failure.
std::vector<uint8_t> read_file(const char * path, size_t expected) {
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::printf("ERROR: cannot open %s\n", path);
        return {};
    }
    std::vector<uint8_t> buf(expected);
    size_t got = std::fread(buf.data(), 1, expected, f);
    std::fclose(f);
    if (got != expected) {
        std::printf("ERROR: %s: expected %zu bytes, got %zu\n", path, expected, got);
        return {};
    }
    return buf;
}

// Byte-by-byte compare. Prints up to max_show mismatches. True iff identical.
bool compare_bytes(const char * name, const uint8_t * cpp, const uint8_t * py,
                   size_t n, int max_show) {
    size_t mismatches = 0;
    size_t first = (size_t)-1;
    for (size_t i = 0; i < n; i++) {
        if (cpp[i] != py[i]) {
            if (first == (size_t)-1) {
                first = i;
            }
            if ((int)mismatches < max_show) {
                std::printf("  [%s] mismatch @ byte %zu: cpp=0x%02x py=0x%02x\n",
                            name, i, cpp[i], py[i]);
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        std::printf("  [%s] PASS (%zu bytes identical)\n", name, n);
        return true;
    }
    std::printf("  [%s] FAIL: %zu/%zu bytes differ, first mismatch @ %zu\n",
                name, mismatches, n, first);
    return false;
}

} // namespace

int main(void) {
    const int64_t out_dim = 4;
    const int64_t in_dim  = 640; // exactly one Tile640 page wide
    const int64_t n       = out_dim * in_dim;
    const int64_t pages   = (in_dim + 640 - 1) / 640; // == 1

    const size_t packed_bytes = (size_t)(out_dim * pages * 32) * sizeof(uint32_t);
    const size_t page_bytes   = (size_t)(out_dim * pages)      * sizeof(uint16_t);
    const size_t lane_bytes   = (size_t)(out_dim * pages * 32) * sizeof(int8_t);

    // 1. read the shared weights written by the Python reference generator
    std::vector<uint8_t> wbuf = read_file("/tmp/bit_equiv_weights.bin",
                                          (size_t)n * sizeof(float));
    if (wbuf.empty()) {
        return 1;
    }
    std::vector<float> weights((size_t)n);
    std::memcpy(weights.data(), wbuf.data(), wbuf.size());

    // 2. C++ ternarize (no AWQ, no clip) + pack + fit scales
    std::vector<int8_t> ternary((size_t)n, 0);
    ts_ternarize_with_acts(weights.data(), nullptr, 0.0f, 0.0f,
                           ternary.data(), out_dim, in_dim);

    std::vector<uint32_t> packed((size_t)(out_dim * pages * 32), 0);
    std::vector<uint16_t> pscale((size_t)(out_dim * pages), 0);
    std::vector<int8_t>   lscale((size_t)(out_dim * pages * 32), 0);
    ts_pack_tile640(ternary.data(), packed.data(), pscale.data(), lscale.data(),
                    out_dim, in_dim);

    // ts_pack_tile640 emits placeholder scales; the real scales come from
    // ts_compute_scales (same as Python's compute_scales). Compare those.
    std::vector<uint16_t> fitted_ps((size_t)(out_dim * pages), 0);
    std::vector<int8_t>   fitted_ls((size_t)(out_dim * pages * 32), 0);
    ts_compute_scales(weights.data(), ternary.data(),
                      fitted_ps.data(), fitted_ls.data(), out_dim, in_dim);

    // 3. read the Python reference buffers
    std::vector<uint8_t> py_packed = read_file("/tmp/bit_equiv_py_packed.bin", packed_bytes);
    std::vector<uint8_t> py_page   = read_file("/tmp/bit_equiv_py_page_scales.bin", page_bytes);
    std::vector<uint8_t> py_lane   = read_file("/tmp/bit_equiv_py_lane_scales.bin", lane_bytes);
    if (py_packed.empty() || py_page.empty() || py_lane.empty()) {
        return 1;
    }

    // 4. byte-by-byte compare
    std::printf("bit-equivalence: C++ vs Python quantize_v3 (%lldx%lld)\n",
                (long long)out_dim, (long long)in_dim);
    bool packed_ok = compare_bytes("packed",
                                   (const uint8_t *)packed.data(), py_packed.data(),
                                   packed_bytes, 10);
    bool page_ok   = compare_bytes("page_scales (ts_compute_scales)",
                                   (const uint8_t *)fitted_ps.data(), py_page.data(),
                                   page_bytes, 10);
    bool lane_ok   = compare_bytes("lane_scales (ts_compute_scales)",
                                   (const uint8_t *)fitted_ls.data(), py_lane.data(),
                                   lane_bytes, 10);

    if (packed_ok && page_ok && lane_ok) {
        std::printf("RESULT: PASS (all buffers byte-identical)\n");
        return 0;
    }
    std::printf("RESULT: FAIL\n");
    return 1;
}
