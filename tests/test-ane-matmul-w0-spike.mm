// W0 ANE matmul spike
//
// End-to-end test: load a 256x256 fp16 matmul .mlmodelc, dispatch on the
// ANE, compare to a CPU fp16 reference within 1e-3 tolerance, time 100
// iterations. This is the W0 spike from
// docs/tessera-ane-matmul-research.md.
//
// The .mlmodelc fixture is built by `tools/ane-mtp/make-w0-matmul.py` and
// lives in `tools/ane-mtp/fixtures/w0-matmul/w0-256x256.mlmodelc/`. The
// build also writes a sidecar `w0-256x256.weight.bin` (fp32 row-major)
// so the test can read back the exact weight that was baked into the
// .mlmodelc and compute a CPU reference without sharing an RNG with the
// Python builder.
//
// This test uses Core ML directly (MLModel / MLMultiArray / IOSurface)
// rather than going through the ggml-ane backend, because the ggml-ane
// backend is not currently wired into the default build
// (ggml/src/CMakeLists.txt does not call `ggml_add_backend(ANE)`). W0
// validates the ANE fp16 matmul path on real hardware; W1+ is the
// ggml-ane backend integration work, which is gated on the
// dispatch_op / supports_op rewiring at ggml-ane.mm:1141-1207.
//
// What this validates (per the W0 spec):
//   - Core ML fp16 matmul runs on the ANE end-to-end on this hardware
//   - The IOSurface zero-copy wrap pattern works (input wrapped as
//     MLMultiArray with nil deallocator)
//   - The 256x256 fp16 matmul fits the ANE's legal shape contract
//     (iOS 18 ios18.conv 1x1 path)
//
// What this does NOT validate (W1+ work):
//   - Per-op dispatch through ggml_ane_program_dispatch_op (MUL_MAT path
//     in graph_compute is still stubbed at ggml-ane.mm:940-942)
//   - Multi-matmul bundle amortization
//   - TILE640 ternary->fp16 prologue

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kN = 256;
constexpr uint32_t kIterations = 100;
constexpr float    kTolerance = 1.0e-3f;
constexpr uint32_t kSeed = 0xA11Eu;

fs::path resolve_fixture_path() {
    if (const char * env = std::getenv("TESSERA_ANE_W0_FIXTURE"); env != nullptr && env[0] != '\0') {
        return fs::path(env);
    }
    fs::path candidate = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        fs::path try_path = candidate / "tools/ane-mtp/fixtures/w0-matmul/w0-256x256.mlmodelc";
        if (fs::is_directory(try_path)) {
            return try_path;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        candidate = candidate.parent_path();
    }
    std::fprintf(stderr,
        "W0 ANE matmul fixture not found. Set TESSERA_ANE_W0_FIXTURE or build it via:\n"
        "  python3 tools/ane-mtp/make-w0-matmul.py --n 256 "
        "--output tools/ane-mtp/fixtures/w0-matmul/\n");
    return {};
}

std::vector<float> make_input(uint32_t n) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

// Load the weight from the sidecar .bin file that the Python builder
// also writes. This is the row-major fp32 view of the same fp16 bytes
// the ANE loaded from the .mlmodelc.
std::vector<float> load_weight(const fs::path & mlmodelc_dir, uint32_t n) {
    fs::path weight_path = mlmodelc_dir.parent_path() /
                           (mlmodelc_dir.stem().string() + ".weight.bin");
    if (!fs::is_regular_file(weight_path)) {
        std::fprintf(stderr, "weight sidecar not found: %s\n",
                     weight_path.string().c_str());
        return {};
    }
    std::ifstream f(weight_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "could not open weight file %s\n",
                     weight_path.string().c_str());
        return {};
    }
    const size_t expected = n * n * sizeof(float);
    f.seekg(0, std::ios::end);
    if ((size_t) f.tellg() != expected) {
        std::fprintf(stderr, "weight file size mismatch (expected %zu, got %lld)\n",
                     expected, (long long) f.tellg());
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<float> w(n * n);
    f.read(reinterpret_cast<char *>(w.data()), expected);
    return w;
}

std::vector<float> cpu_reference_matmul(const std::vector<float> & x,
                                       const std::vector<float> & w,
                                       uint32_t n) {
    // CoreML innerProduct: y[i] = sum_j W[i, j] * x[j]. W is row-major.
    std::vector<float> y(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < n; ++j) {
            acc += x[j] * w[i * n + j];
        }
        y[i] = acc;
    }
    return y;
}

bool fp16_close_enough(const std::vector<float> & expected,
                       const std::vector<float> & actual,
                       uint32_t n) {
    float max_abs_err = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float err = std::fabs(expected[i] - actual[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }
    std::printf("max |err| (ANE vs CPU fp32 reference): %.4e\n",
                static_cast<double>(max_abs_err));
    return max_abs_err <= kTolerance;
}

// Allocate an IOSurface-backed buffer of `bytes` size, locked for CPU
// access. Mirrors the ggml-ane.mm:65-104 allocation pattern (16 KB
// page rounding, 64 KB ANE floor).
struct IOSurfaceBuffer {
    IOSurfaceRef surface = nullptr;
    void *        base = nullptr;
    size_t        bytes = 0;
    size_t        bytes_per_row = 0;

    ~IOSurfaceBuffer() {
        if (surface) {
            IOSurfaceUnlock(surface, 0, nullptr);
            CFRelease(surface);
        }
    }
};

std::unique_ptr<IOSurfaceBuffer> make_iosurface_buffer(size_t bytes) {
    auto buf = std::make_unique<IOSurfaceBuffer>();
    constexpr size_t kPage = 16 * 1024;
    constexpr size_t kMin  = 64 * 1024;
    bytes = ((bytes + kPage - 1) / kPage) * kPage;
    if (bytes < kMin) {
        bytes = kMin;
    }
    buf->bytes = bytes;
    buf->bytes_per_row = bytes;

    NSDictionary * props = @{
        (NSString *) kIOSurfaceWidth:        @1,
        (NSString *) kIOSurfaceHeight:       @1,
        (NSString *) kIOSurfaceBytesPerRow:  @(buf->bytes_per_row),
        (NSString *) kIOSurfaceAllocSize:    @(buf->bytes),
        (NSString *) kIOSurfacePixelFormat:  @0,
    };
    buf->surface = IOSurfaceCreate((__bridge CFDictionaryRef) props);
    if (!buf->surface) {
        std::fprintf(stderr, "IOSurfaceCreate failed\n");
        return nullptr;
    }
    IOReturn rc = IOSurfaceLock(buf->surface, 0, nullptr);
    if (rc != kIOReturnSuccess) {
        std::fprintf(stderr, "IOSurfaceLock failed: 0x%x\n", rc);
        return nullptr;
    }
    buf->base = IOSurfaceGetBaseAddress(buf->surface);
    if (!buf->base) {
        std::fprintf(stderr, "IOSurfaceGetBaseAddress returned null\n");
        return nullptr;
    }
    return buf;
}

} // namespace

int main(int /* argc */, char ** /* argv */) {
    const fs::path fixture = resolve_fixture_path();
    if (fixture.empty()) {
        return 2;
    }
    std::printf("W0 ANE matmul spike: loading %s\n", fixture.string().c_str());

    NSError * error = nil;
    NSString * path = [NSString stringWithUTF8String:fixture.string().c_str()];
    NSURL * url = [NSURL fileURLWithPath:path];
    MLModel * model = [MLModel modelWithContentsOfURL:url error:&error];
    if (!model) {
        std::fprintf(stderr, "MLModel load failed: %s\n",
                     [[error localizedDescription] UTF8String]);
        return 1;
    }
    std::printf("loaded .mlmodelc OK\n");

    const std::vector<float> input = make_input(kN);
    const std::vector<float> weight = load_weight(fixture, kN);
    if (weight.empty()) {
        std::fprintf(stderr, "could not load reference weight\n");
        return 1;
    }
    const std::vector<float> expected = cpu_reference_matmul(input, weight, kN);

    auto in_buf = make_iosurface_buffer(kN * sizeof(float));
    if (!in_buf) {
        return 1;
    }
    std::memcpy(in_buf->base, input.data(), kN * sizeof(float));

    // Wrap the input as MLMultiArray with nil deallocator (zero-copy,
    // see common/ane-mtp.mm:535-569 for the canonical pattern).
    NSArray<NSNumber *> * shape = @[@(kN)];
    NSArray<NSNumber *> * strides = @[@(1)];  // contiguous, row-major
    MLMultiArray * x = [[MLMultiArray alloc]
        initWithDataPointer:in_buf->base
                      shape:shape
                   dataType:MLMultiArrayDataTypeFloat32
                    strides:strides
                deallocator:nil
                      error:&error];
    if (!x) {
        std::fprintf(stderr, "MLMultiArray x init failed: %s\n",
                     [[error localizedDescription] UTF8String]);
        return 1;
    }

    auto run_once = [&]() -> std::pair<bool, std::vector<float>> {
        @autoreleasepool {
            NSDictionary<NSString *, MLFeatureValue *> * feature_dict = @{
                @"x": [MLFeatureValue featureValueWithMultiArray:x],
            };
            MLDictionaryFeatureProvider * input_features =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:feature_dict error:&error];
            if (!input_features) {
                std::fprintf(stderr, "MLDictionaryFeatureProvider init failed: %s\n",
                             [[error localizedDescription] UTF8String]);
                return {false, {}};
            }
            id<MLFeatureProvider> output_features =
                [model predictionFromFeatures:input_features error:&error];
            if (!output_features) {
                std::fprintf(stderr, "prediction failed: %s\n",
                             [[error localizedDescription] UTF8String]);
                return {false, {}};
            }
            MLMultiArray * y_out = [output_features featureValueForName:@"y"].multiArrayValue;
            const float * y_data = static_cast<const float *>(y_out.dataPointer);
            return {true, std::vector<float>(y_data, y_data + kN)};
        }
    };

    auto [ok, output] = run_once();
    if (!ok) {
        return 1;
    }
    if (!fp16_close_enough(expected, output, kN)) {
        std::fprintf(stderr, "ANE output disagrees with CPU fp32 reference\n");
        return 1;
    }

    // Timing
    std::vector<double> durations_us;
    durations_us.reserve(kIterations);
    for (uint32_t i = 0; i < kIterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run_once().first) {
            std::fprintf(stderr, "ANE matmul run failed at iteration %u\n", i);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        durations_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(durations_us.begin(), durations_us.end());
    const double median = durations_us[kIterations / 2];
    const double p99    = durations_us[static_cast<size_t>(kIterations * 0.99)];
    std::printf("ANE matmul %ux%u fp16: median=%.2fus p99=%.2fus (over %u iters)\n",
                kN, kN, median, p99, kIterations);

    return 0;
}
