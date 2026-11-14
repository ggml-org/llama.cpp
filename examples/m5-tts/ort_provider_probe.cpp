// Time a single ONNX model under a chosen execution provider on Android.
//
// usage: ort-prov-probe model.onnx [CPU|XNNPACK|NNAPI] [threads] [input.f32] [ref.f32]
//
// The default CPU EP is compared against XNNPACK, which is compiled into the
// shipped libonnxruntime.so but never selected by the M5 host engine.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "onnxruntime_c_api.h"

static const OrtApi* g = NULL;

#define CHECK(expr) do { OrtStatus* st = (expr); if (st) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, g->GetErrorMessage(st)); g->ReleaseStatus(st); exit(1);} } while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static float* read_f32(const char* path, size_t count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float* buf = static_cast<float*>(malloc(count * sizeof(float)));
    size_t got = fread(buf, sizeof(float), count, f);
    if (got != count) { fprintf(stderr, "%s: got %zu of %zu floats\n", path, got, count); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model [provider] [threads] [input.f32] [ref.f32]\n", argv[0]); return 2; }
    const char* model = argv[1];
    const char* provider = argc > 2 ? argv[2] : "CPU";
    int threads = argc > 3 ? atoi(argv[3]) : 4;
    const char* input_path = argc > 4 ? argv[4] : NULL;
    const char* ref_path = argc > 5 ? argv[5] : NULL;

    g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g) { fprintf(stderr, "OrtGetApiBase failed\n"); return 1; }

    OrtEnv* env = NULL;
    CHECK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "probe", &env));
    OrtSessionOptions* so = NULL;
    CHECK(g->CreateSessionOptions(&so));
    CHECK(g->SetIntraOpNumThreads(so, threads));
    CHECK(g->SetInterOpNumThreads(so, 1));
    CHECK(g->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
    if (strcmp(provider, "CPU") != 0) {
        CHECK(g->SessionOptionsAppendExecutionProvider(so, provider, NULL, NULL, 0));
    }

    OrtSession* session = NULL;
    double t0 = now_ms();
    CHECK(g->CreateSession(env, model, so, &session));
    printf("provider=%s threads=%d session_ms=%.1f\n", provider, threads, now_ms() - t0);

    size_t n_in = 0;
    CHECK(g->SessionGetInputCount(session, &n_in));
    size_t n_out = 0;
    CHECK(g->SessionGetOutputCount(session, &n_out));

    OrtAllocator* alloc = NULL;
    CHECK(g->GetAllocatorWithDefaultOptions(&alloc));

    OrtValue** inputs = static_cast<OrtValue**>(calloc(n_in, sizeof(OrtValue*)));
    char** input_names = static_cast<char**>(calloc(n_in, sizeof(char*)));
    void** buffers = static_cast<void**>(calloc(n_in, sizeof(void*)));
    size_t total_elems = 0;
    for (size_t i = 0; i < n_in; i++) {
        char* name = NULL;
        CHECK(g->SessionGetInputName(session, i, alloc, &name));
        input_names[i] = name;
        OrtTypeInfo* info = NULL;
        CHECK(g->SessionGetInputTypeInfo(session, i, &info));
        const OrtTensorTypeAndShapeInfo* tsi = NULL;
        CHECK(g->CastTypeInfoToTensorInfo(info, &tsi));
        ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        CHECK(g->GetTensorElementType(tsi, &elem_type));
        if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            fprintf(stderr, "only float32 inputs are supported (%s)\n", name);
            return 1;
        }
        size_t rank = 0;
        CHECK(g->GetDimensionsCount(tsi, &rank));
        int64_t dims[8] = {0};
        CHECK(g->GetDimensions(tsi, dims, rank));
        size_t elems = 1;
        for (size_t d = 0; d < rank; d++) {
            if (dims[d] <= 0) { fprintf(stderr, "dynamic dim in %s\n", name); return 1; }
            elems *= (size_t)dims[d];
        }
        printf("  in[%zu] %-20s rank=%zu elems=%zu\n", i, name, rank, elems);
        total_elems += elems;
        buffers[i] = calloc(elems, sizeof(float));
        if (i == 0 && input_path) {
            free(buffers[i]);
            buffers[i] = read_f32(input_path, elems);
        }
        OrtMemoryInfo* mi = NULL;
        CHECK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
        CHECK(g->CreateTensorWithDataAsOrtValue(mi, buffers[i], elems * sizeof(float),
                                                dims, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[i]));
        g->ReleaseMemoryInfo(mi);
        g->ReleaseTypeInfo(info);
    }

    OrtValue* output = NULL;
    char** output_names = static_cast<char**>(calloc(n_out, sizeof(char*)));
    for (size_t i = 0; i < n_out; i++) {
        CHECK(g->SessionGetOutputName(session, i, alloc, &output_names[i]));
    }
    const double kWarmup = 1;
    double times[16];
    const int runs = (int)(sizeof(times) / sizeof(times[0]));
    float max_error = 0;
    float* ref = NULL;
    size_t ref_elems = 0;
    OrtValue** outputs = static_cast<OrtValue**>(calloc(n_out, sizeof(OrtValue*)));
    for (int r = 0; r <= runs; r++) {
        double start = now_ms();
        CHECK(g->Run(session, NULL, (const char* const*)input_names,
                     (const OrtValue* const*)inputs, n_in,
                     (const char* const*)output_names, n_out, outputs));
        double elapsed = now_ms() - start;
        if (r == 0) {
            OrtTensorTypeAndShapeInfo* oi = NULL;
            CHECK(g->GetTensorTypeAndShape(outputs[0], &oi));
            size_t oe = 0;
            CHECK(g->GetTensorShapeElementCount(oi, &oe));
            int64_t od[8] = {0};
            size_t orank = 0;
            CHECK(g->GetDimensionsCount(oi, &orank));
            CHECK(g->GetDimensions(oi, od, orank));
            printf("  out shape=[");
            for (size_t d = 0; d < orank; d++) printf("%lld%s", (long long)od[d], d + 1 < orank ? "," : "");
            printf("] elems=%zu\n", oe);
            if (ref_path) {
                ref = read_f32(ref_path, oe);
                ref_elems = oe;
                float* data = NULL;
                CHECK(g->GetTensorMutableData(outputs[0], (void**)&data));
                for (size_t k = 0; k < oe && k < ref_elems; k++) {
                    float e = fabsf(data[k] - ref[k]);
                    if (e > max_error) max_error = e;
                }
            }
            g->ReleaseTensorTypeAndShapeInfo(oi);
        } else {
            times[r - 1] = elapsed;
        }
        for (size_t o = 0; o < n_out; o++) { if (outputs[o]) g->ReleaseValue(outputs[o]); outputs[o] = NULL; }
        (void)kWarmup;
    }
    qsort(times, runs, sizeof(double), cmp_double);
    printf("RESULT provider=%s threads=%d median_ms=%.3f min_ms=%.3f max_abs_error=%.6g\n",
           provider, threads, times[runs / 2], times[0], max_error);

    for (size_t i = 0; i < n_in; i++) {
        if (inputs[i]) g->ReleaseValue(inputs[i]);
    }
    return 0;
}
