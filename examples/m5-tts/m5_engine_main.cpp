// Command line front-end for the native engine.
//
//   m5-engine model --text "今天天气不错" --out out.wav
//   m5-engine model --tokens ids.bin --out out.wav
//   m5-engine model --encode "今天天气不错"        # print token ids
#include "m5_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s model_dir [--text STR | --tokens FILE | --encode STR] --out out.wav\n"
                "  --voice N|NAME     voice from voices.json, by index or name (default 0)\n"
                "  --voices FILE      clone cache to use (default model_dir/voices.json)\n"
                "  --caption TEXT     style/emotion description for the caption slot\n"
                "  --seed S           java.util.Random seed (default 1234)\n"
                "  --backbone CMD     backbone stage command\n"
                "  --depth CMD        depth stage command\n"
                "  --codec FILE       codec model (default codec_decoder_cached.onnx)\n"
                "  --threads N        ONNX Runtime intra-op threads (default 3)\n"
                "  --cwd DIR          working directory for the stage processes\n"
                "  --first N          frames in the first codec chunk (default 2)\n"
                "  --chunk N          frames per codec chunk afterwards (default 4)\n"
                "  --no-overlap       run the codec on the calling thread\n"
                "  --ring             use the shared-memory stage transport\n"
                "  --dump-codes FILE  append generated codes (debug)\n",
                argv[0]);
        return 2;
    }

    m5::Options options;
    options.model_dir = argv[1];
    std::string text, tokens_path, encode_text, out_path = "out.wav";

    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--text") text = next("--text");
        else if (a == "--tokens") tokens_path = next("--tokens");
        else if (a == "--encode") encode_text = next("--encode");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--voice") {
            const std::string v = next("--voice");
            if (v.find_first_not_of("0123456789") == std::string::npos) options.voice = atoi(v.c_str());
            else options.voice_name = v;
        }
        else if (a == "--voices") options.voices_path = next("--voices");
        else if (a == "--caption") options.caption = next("--caption");
        else if (a == "--seed") options.seed = atol(next("--seed").c_str());
        else if (a == "--backbone") options.backbone_cmd = next("--backbone");
        else if (a == "--depth") options.depth_cmd = next("--depth");
        else if (a == "--codec") options.codec_model = next("--codec");
        else if (a == "--threads") options.threads = atoi(next("--threads").c_str());
        else if (a == "--cwd") options.workdir = next("--cwd");
        else if (a == "--first") options.first_frames = atoi(next("--first").c_str());
        else if (a == "--chunk") options.chunk_frames = atoi(next("--chunk").c_str());
        else if (a == "--dump-codes") options.dump_codes = next("--dump-codes");
        else if (a == "--no-overlap") options.overlap = false;
        else if (a == "--ring") options.ring = true;
        else if (a == "--pipe") options.ring = false;   // kept for old scripts
        else if (a == "--verbose") options.verbose = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    setvbuf(stdout, nullptr, _IONBF, 0);
    options.stages = encode_text.empty();   // encoding alone needs no stage process
    std::string error;
    m5::Engine engine;
    if (!engine.open(options, error)) {
        fprintf(stderr, "open failed: %s\n", error.c_str());
        return 1;
    }

    if (!encode_text.empty()) {
        for (const auto & segment : engine.encode_segments(encode_text)) {
            std::string line;
            for (long id : segment) { line += std::to_string(id); line += ' '; }
            if (!line.empty()) line.pop_back();
            printf("%s\n", line.c_str());
        }
        return 0;
    }

    m5::Stats stats;
    bool ok;
    if (!tokens_path.empty()) {
        FILE * f = fopen(tokens_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", tokens_path.c_str()); return 1; }
        std::vector<long> tokens;
        long value;
        while (fread(&value, sizeof(long), 1, f) == 1) tokens.push_back(value);
        fclose(f);
        if (tokens.empty()) { fprintf(stderr, "no tokens\n"); return 1; }
        ok = engine.render_tokens(tokens, out_path, nullptr, stats, error);
    } else {
        if (text.empty()) { fprintf(stderr, "nothing to synthesize\n"); return 2; }
        ok = engine.render_text(text, out_path, nullptr, stats, error);
    }

    if (!ok) { fprintf(stderr, "render failed: %s\n", error.c_str()); return 1; }
    printf("{\"audio_s\":%.4f,\"elapsed_s\":%.4f,\"rtf\":%.4f,\"backbone_s\":%.3f,"
           "\"depth_s\":%.3f,\"codec_s\":%.3f,\"first_pcm_s\":%.3f,\"frames\":%d,\"segments\":%d}\n",
           stats.audio, stats.elapsed, stats.rtf, stats.backbone, stats.depth, stats.codec,
           stats.first_pcm, stats.frames, stats.segments);
    return 0;
}
