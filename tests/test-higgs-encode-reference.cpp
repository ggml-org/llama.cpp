#include "../tools/tts/higgs-codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

static uint16_t u16le(const unsigned char * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static uint32_t u32le(const unsigned char * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static std::vector<float> read_wav_mono(const std::string & path, int & sample_rate) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open WAV: " + path);

    std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()
    };
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        throw std::runtime_error("not a WAV");

    const uint16_t channels = u16le(bytes.data() + 22);
    sample_rate = (int) u32le(bytes.data() + 24);
    const uint16_t bps = u16le(bytes.data() + 34);
    const uint16_t block_align = u16le(bytes.data() + 32);
    const uint16_t audio_format = u16le(bytes.data() + 20);
    if (channels == 0 || sample_rate == 0 || block_align == 0) throw std::runtime_error("bad fmt");

    size_t data_off = 12;
    while (data_off + 8 <= bytes.size()) {
        if (std::memcmp(bytes.data() + data_off, "data", 4) == 0) { data_off += 8; break; }
        data_off += 8 + u32le(bytes.data() + data_off + 4) + (u32le(bytes.data() + data_off + 4) & 1u);
    }
    if (data_off >= bytes.size()) throw std::runtime_error("no data chunk");

    const size_t data_size = bytes.size() - data_off;
    const size_t frames = data_size / block_align;
    std::vector<float> mono(frames);

    for (size_t t = 0; t < frames; ++t) {
        const unsigned char * frame = bytes.data() + data_off + t * block_align;
        double sum = 0.0;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            if (audio_format == 1 && bps == 16) {
                int16_t v;
                std::memcpy(&v, frame + ch * 2, sizeof(v));
                sum += (double) v / 32768.0;
            } else if (audio_format == 3 && bps == 32) {
                float v;
                std::memcpy(&v, frame + ch * 4, sizeof(v));
                sum += v;
            }
        }
        mono[t] = (float) std::clamp(sum / (double) channels, -1.0, 1.0);
    }
    return mono;
}

static std::vector<float> resample_linear(const std::vector<float> & input, int src_rate, int dst_rate) {
    if (src_rate == dst_rate) return input;
    if (input.empty()) return {};
    return higgs_audio::resample_kaiser(input, (double)src_rate, (double)dst_rate);
}

static void dump_codes(const std::vector<std::vector<int>> & frames, const std::string & path) {
    std::ofstream out(path);
    out << "{\n  \"source\": \"test-higgs-encode-reference\",\n";
    out << "  \"codebook_size\": 1026,\n";
    out << "  \"num_codebooks\": " << (frames.empty() ? 0 : frames[0].size()) << ",\n";
    out << "  \"codec_frames\": [\n";
    for (size_t f = 0; f < frames.size(); ++f) {
        out << "    [";
        for (size_t c = 0; c < frames[f].size(); ++c) {
            if (c) out << ", ";
            out << frames[f][c];
        }
        out << "]";
        if (f + 1 < frames.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <higgs-audio.gguf> <reference.wav> [max_seconds] [codes_out.json]\n", argv[0]);
        return 1;
    }

    const std::string companion_path = argv[1];
    const std::string wav_path = argv[2];
    const float max_sec = argc > 3 ? (float) std::atof(argv[3]) : 30.0f;
    const std::string out_path = argc > 4 ? argv[4] : "";

    try {
        higgs_audio::companion_file file(companion_path);
        const int dst_rate = file.metadata().sample_rate;
        const auto & meta = file.metadata();
        std::fprintf(stderr, "companion: codebooks=%d codebook_size=%d n_embd=%d sample_rate=%d\n",
                meta.layout.num_codebooks,
                meta.layout.codebook_size,
                meta.layout.n_embd,
                meta.sample_rate);

        int src_rate = 0;
        auto audio = read_wav_mono(wav_path, src_rate);
        std::fprintf(stderr, "wav: samples=%zu rate=%d duration=%.2fs\n",
                audio.size(), src_rate, (double) audio.size() / (double) src_rate);

        audio = resample_linear(audio, src_rate, dst_rate);
        const size_t max_samples = (size_t) (dst_rate * max_sec);
        if (audio.size() > max_samples) audio.resize(max_samples);
        std::fprintf(stderr, "resampled: samples=%zu rate=%d\n", audio.size(), dst_rate);

        const auto codes = higgs_audio::encode_reference_codes(file, audio);
        std::fprintf(stderr, "encoded: frames=%zu codebooks=%zu\n",
                codes.size(), codes.empty() ? 0 : codes[0].size());

        if (codes.empty()) {
            std::fprintf(stderr, "WARNING: no codes produced\n");
            return 1;
        }

        if (!out_path.empty()) {
            dump_codes(codes, out_path);
            std::fprintf(stderr, "wrote: %s\n", out_path.c_str());
        }

        std::fprintf(stderr, "first frame: [");
        for (size_t c = 0; c < codes[0].size(); ++c) {
            if (c) std::fprintf(stderr, ", ");
            std::fprintf(stderr, "%d", codes[0][c]);
        }
        std::fprintf(stderr, "]\n");

    } catch (const std::exception & e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }

    return 0;
}
