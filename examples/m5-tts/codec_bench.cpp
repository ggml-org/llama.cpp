// C++ codec runner for the M5-TTS waveform decoder.
//
// The host engine drives the codec through a Java ORT wrapper; this program
// does the same recurrence natively so the decoder can be profiled and A/B
// tested without a JVM: it feeds one chunk of codes at a time, carries the
// recurrent conv states between calls, and writes the PCM to a WAV file.
//
// usage: codec-bench model.onnx provider threads chunks chunk_len codes.bin out.wav [rate]
//
// codes.bin holds int64 frame-major codes, with the codebook count from the model.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include <string>
#include <vector>

#include "onnxruntime_c_api.h"
#include "m5_snake_op.h"

static const OrtApi* g = nullptr;

#define CHECK(expr) do { OrtStatus* st = (expr); if (st) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, g->GetErrorMessage(st)); \
    g->ReleaseStatus(st); exit(1);} } while (0)

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

struct Input {
    std::string name;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    std::vector<int64_t> dims;
    std::vector<int64_t> initial_dims;
    size_t elems = 0;
    void* buffer = nullptr;
    OrtValue* value = nullptr;
};

static size_t element_count(const std::vector<int64_t>& dims) {
    size_t n = 1;
    for (int64_t d : dims) n *= (size_t)(d < 0 ? 1 : d);
    return n;
}

static void write_wav(const char* path, const std::vector<float>& pcm, int rate) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    const uint32_t samples = (uint32_t)pcm.size();
    const uint32_t data_bytes = samples * 2;
    uint32_t riff = 36 + data_bytes, fmt = 16, byte_rate = rate * 2;
    uint16_t one = 1, bits = 16, block = 2, pcm_tag = 1;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt, 4, 1, f); fwrite(&pcm_tag, 2, 1, f);
    fwrite(&one, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&byte_rate, 4, 1, f);
    fwrite(&block, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_bytes, 4, 1, f);
    std::vector<int16_t> out(samples);
    for (uint32_t i = 0; i < samples; i++) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        out[i] = (int16_t)lrintf(v * 32768.0f);
    }
    fwrite(out.data(), 2, samples, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr,
                "usage: %s model.onnx provider threads chunks chunk_len codes.bin out.wav [rate]\n",
                argv[0]);
        return 2;
    }
    const char* model = argv[1];
    const char* provider = argv[2];
    const int threads = atoi(argv[3]);
    const int chunks = atoi(argv[4]);
    const int chunk_len = atoi(argv[5]);
    const char* codes_path = argv[6];
    const char* wav_path = argv[7];
    const int rate = argc > 8 ? atoi(argv[8]) : 24000;
    const int reps = argc > 9 ? atoi(argv[9]) : 1;
    if (threads <= 0 || chunks <= 0 || chunk_len <= 0 || rate <= 0 || reps <= 0) {
        fprintf(stderr, "threads, chunks, chunk length, rate and repetitions must be positive\n");
        return 2;
    }

    g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g) { fprintf(stderr, "OrtGetApiBase failed\n"); return 1; }

    OrtEnv* env = nullptr;
    CHECK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "codec-bench", &env));
    OrtSessionOptions* so = nullptr;
    CHECK(g->CreateSessionOptions(&so));
    CHECK(g->SetIntraOpNumThreads(so, threads));
    CHECK(g->SetInterOpNumThreads(so, 1));
    CHECK(g->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
    if (!m5::snake::register_ops(g, so)) {
        fprintf(stderr, "cannot register m5::Snake\n");
        return 1;
    }
    const char* profile_prefix = getenv("M5_PROFILE");
    if (strcmp(provider, "CPU") != 0) {
        CHECK(g->SessionOptionsAppendExecutionProvider(so, provider, nullptr, nullptr, 0));
    }
    if (profile_prefix) CHECK(g->EnableProfiling(so, profile_prefix));
    OrtSession* session = nullptr;
    const double t_session = now_ms();
    CHECK(g->CreateSession(env, model, so, &session));
    const double session_ms = now_ms() - t_session;

    OrtAllocator* alloc = nullptr;
    CHECK(g->GetAllocatorWithDefaultOptions(&alloc));
    OrtMemoryInfo* cpu_mem = nullptr;
    CHECK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_mem));

    size_t n_in = 0, n_out = 0;
    CHECK(g->SessionGetInputCount(session, &n_in));
    CHECK(g->SessionGetOutputCount(session, &n_out));

    std::vector<Input> inputs(n_in);
    int codes_index = -1, offset_index = -1, codebooks = 16;
    for (size_t i = 0; i < n_in; i++) {
        Input& in = inputs[i];
        char* name = nullptr;
        CHECK(g->SessionGetInputName(session, i, alloc, &name));
        in.name = name;
        OrtTypeInfo* info = nullptr;
        CHECK(g->SessionGetInputTypeInfo(session, i, &info));
        const OrtTensorTypeAndShapeInfo* tsi = nullptr;
        CHECK(g->CastTypeInfoToTensorInfo(info, &tsi));
        CHECK(g->GetTensorElementType(tsi, &in.type));
        size_t rank = 0;
        CHECK(g->GetDimensionsCount(tsi, &rank));
        in.dims.resize(rank);
        CHECK(g->GetDimensions(tsi, in.dims.data(), rank));
        g->ReleaseTypeInfo(info);

        if (in.name == "codes") {
            codes_index = (int)i;
            codebooks = (int)in.dims[1];
            in.dims[2] = chunk_len;   // dynamic frame axis
        }
        if (in.name == "offset") offset_index = (int)i;
        for (int64_t& d : in.dims) {
            if (d < 0) d = 0;   // dynamic axis, empty at the start of the stream
        }
        in.elems = element_count(in.dims);
        in.initial_dims = in.dims;
        const size_t width = in.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4;
        if (in.elems > 0) {
            in.buffer = calloc(in.elems, width);
            if (!in.buffer) { fprintf(stderr, "oom for %s\n", in.name.c_str()); return 1; }
        }
        fprintf(stderr, "in[%2zu] %-16s elems=%zu dims=[", i, in.name.c_str(), in.elems);
        for (size_t d = 0; d < in.dims.size(); d++) {
            fprintf(stderr, "%lld%s", (long long)in.dims[d], d + 1 < in.dims.size() ? "," : "");
        }
        fprintf(stderr, "]\n");
    }
    if (codes_index < 0 || offset_index < 0) {
        fprintf(stderr, "model has no codes/offset inputs\n");
        return 1;
    }

    std::vector<std::string> output_names(n_out);
    for (size_t i = 0; i < n_out; i++) {
        char* name = nullptr;
        CHECK(g->SessionGetOutputName(session, i, alloc, &name));
        output_names[i] = name;
        fprintf(stderr, "out[%2zu] %s\n", i, name);
    }
    // Recurrent outputs are named for the input they replace: "present" feeds
    // "past" and "next_conv_N" feeds "conv_N".
    std::vector<int> out_to_in(n_out, -1);
    for (size_t o = 1; o < n_out; o++) {
        std::string want = output_names[o];
        if (want == "present") want = "past";
        else if (want.rfind("next_", 0) == 0) want = want.substr(5);
        for (size_t i = 0; i < n_in; i++) {
            if (inputs[i].name == want) { out_to_in[o] = (int)i; break; }
        }
        if (out_to_in[o] < 0) fprintf(stderr, "warning: output %s has no matching input\n", output_names[o].c_str());
    }
    std::vector<const char*> output_cstr(n_out);
    for (size_t i = 0; i < n_out; i++) output_cstr[i] = output_names[i].c_str();

    // codes.bin is frame-major; each chunk consumes chunk_len frames.
    const size_t need_frames = (size_t)chunks * (size_t)chunk_len;
    std::vector<int64_t> codes(need_frames * codebooks, 0);
    {
        FILE* f = fopen(codes_path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", codes_path); return 1; }
        size_t got = fread(codes.data(), sizeof(int64_t), codes.size(), f);
        fclose(f);
        if (got < codes.size()) {
            fprintf(stderr, "codes file has %zu of %zu required codes\n", got, codes.size());
            return 1;
        }
    }

    std::vector<const char*> input_cstr(n_in);
    for (size_t i = 0; i < n_in; i++) input_cstr[i] = inputs[i].name.c_str();

    std::vector<float> pcm;
    std::vector<OrtValue*> outputs(n_out, nullptr);
    double total_ms = 0, first_ms = 0;
    double best_ms = 0;
    const int64_t* code_frames = codes.data();

    for (int rep = 0; rep < reps; rep++) {
    if (rep > 0) {
        // Restart the stream: every recurrent state goes back to zeros.
        for (size_t i = 0; i < n_in; i++) {
            if ((int)i == codes_index || (int)i == offset_index) continue;
            Input& in = inputs[i];
            free(in.buffer);
            in.dims = in.initial_dims;
            in.elems = element_count(in.dims);
            in.buffer = in.elems ? calloc(in.elems, sizeof(float)) : nullptr;
            if (in.elems && !in.buffer) return 1;
        }
        pcm.clear();
    }
    double rep_ms = 0;
    for (int c = 0; c < chunks; c++) {
        memset(inputs[codes_index].buffer, 0,
               inputs[codes_index].elems * sizeof(int64_t));
        // layout: [1, 8, chunk_len]
        int64_t* dst = static_cast<int64_t*>(inputs[codes_index].buffer);
        const int64_t* src = code_frames + (size_t)c * chunk_len * codebooks;
        for (int t = 0; t < chunk_len; t++) {
            for (int k = 0; k < codebooks; k++) dst[k * chunk_len + t] = src[t * codebooks + k];
        }
        static_cast<int64_t*>(inputs[offset_index].buffer)[0] = (int64_t)c * chunk_len;

        std::vector<OrtValue*> in_values(n_in, nullptr);
        for (size_t i = 0; i < n_in; i++) {
            const size_t width = inputs[i].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : 4;
            CHECK(g->CreateTensorWithDataAsOrtValue(
                cpu_mem, inputs[i].buffer, inputs[i].elems * width,
                inputs[i].dims.data(), inputs[i].dims.size(), inputs[i].type, &in_values[i]));
        }

        const double start = now_ms();
        CHECK(g->Run(session, nullptr, input_cstr.data(),
                     (const OrtValue* const*)in_values.data(), n_in,
                     output_cstr.data(), n_out, outputs.data()));
        const double ms = now_ms() - start;
        rep_ms += ms;
        if (c == 0 && rep == 0) first_ms = ms;

        for (size_t i = 0; i < n_in; i++) g->ReleaseValue(in_values[i]);

        // output[0] is PCM, output[1 + k] mirrors input[1 + k] (recurrent state).
        OrtTensorTypeAndShapeInfo* oi = nullptr;
        CHECK(g->GetTensorTypeAndShape(outputs[0], &oi));
        size_t pcm_elems = 0;
        CHECK(g->GetTensorShapeElementCount(oi, &pcm_elems));
        g->ReleaseTensorTypeAndShapeInfo(oi);
        float* data = nullptr;
        CHECK(g->GetTensorMutableData(outputs[0], (void**)&data));
        pcm.insert(pcm.end(), data, data + pcm_elems);

        for (size_t o = 1; o < n_out; o++) {
            if (out_to_in[o] < 0) continue;
            Input& target = inputs[out_to_in[o]];
            OrtTensorTypeAndShapeInfo* si = nullptr;
            CHECK(g->GetTensorTypeAndShape(outputs[o], &si));
            size_t state_elems = 0;
            CHECK(g->GetTensorShapeElementCount(si, &state_elems));
            size_t rank = 0;
            CHECK(g->GetDimensionsCount(si, &rank));
            std::vector<int64_t> shape(rank, 0);
            CHECK(g->GetDimensions(si, shape.data(), rank));
            g->ReleaseTensorTypeAndShapeInfo(si);
            if (state_elems == 0) continue;
            if (state_elems != target.elems) {
                free(target.buffer);
                target.buffer = calloc(state_elems, sizeof(float));
                if (!target.buffer) { fprintf(stderr, "oom growing %s\n", target.name.c_str()); return 1; }
                target.elems = state_elems;
                target.dims = shape;
            }
            float* state = nullptr;
            CHECK(g->GetTensorMutableData(outputs[o], (void**)&state));
            memcpy(target.buffer, state, target.elems * sizeof(float));
        }
        for (size_t i = 0; i < n_out; i++) {
            if (outputs[i]) g->ReleaseValue(outputs[i]);
            outputs[i] = nullptr;
        }
    }

    total_ms = rep_ms;
    if (rep == 0 || rep_ms < best_ms) best_ms = rep_ms;
    fprintf(stderr, "rep %d: %.1f ms\n", rep, rep_ms);
    }

    // The final LSTM tail is not audible; drop the same tail the host harness does.
    const size_t tail = 2400;
    if (pcm.size() > tail) pcm.resize(pcm.size() - tail);
    write_wav(wav_path, pcm, rate);

    if (profile_prefix) {
        char* path = nullptr;
        CHECK(g->SessionEndProfiling(session, alloc, &path));
        printf("profile=%s\n", path);
    }
    const double audio_s = (double)pcm.size() / rate;
    printf("{\"provider\":\"%s\",\"threads\":%d,\"chunks\":%d,\"chunk_len\":%d,"
           "\"session_ms\":%.1f,\"total_ms\":%.1f,\"first_chunk_ms\":%.1f,"
           "\"audio_s\":%.4f,\"codec_rtf\":%.4f}\n",
           provider, threads, chunks, chunk_len, session_ms, best_ms, first_ms,
           audio_s, audio_s > 0 ? best_ms / 1000.0 / audio_s : 0.0);
    return 0;
}
