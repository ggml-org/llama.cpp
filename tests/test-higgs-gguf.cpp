#include "../tools/tts/higgs-gguf.h"

#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <algorithm>
#include <string>
#include <vector>

static std::string get_optional_path(int argc, char ** argv) {
    if (argc > 1 && argv[1] && argv[1][0] != '\0') {
        return argv[1];
    }

    const char * env = std::getenv("HIGGS_AUDIO_GGUF");
    if (env && env[0] != '\0') {
        return env;
    }

    return {};
}

int main(int argc, char ** argv) {
    const std::string path = get_optional_path(argc, argv);
    if (path.empty()) {
        std::fprintf(stderr, "HIGGS_AUDIO_GGUF not set; skipping local Higgs companion GGUF read-back\n");
        return 0;
    }

    try {
        const higgs_audio::companion_file file(path);
        const auto & meta = file.metadata();

        const auto check = [](bool ok, const char * message) {
            if (!ok) {
                std::fprintf(stderr, "check failed: %s\n", message);
                std::exit(1);
            }
        };

        const size_t expected_codebook_bytes = meta.layout.n_weights() * ggml_type_size(GGML_TYPE_F16);

        std::fprintf(stderr,
                "loaded Higgs companion: tensors=%lld codebooks=%d codebook_size=%d hidden=%d codebook_bytes=%zu\n",
                (long long) meta.n_tensors,
                meta.layout.num_codebooks,
                meta.layout.codebook_size,
                meta.layout.n_embd,
                expected_codebook_bytes);

        check(meta.format == "higgs-audio-v3-tts", "format");
        check(meta.backbone_arch == "qwen3", "backbone_arch");
        check(meta.layout.num_codebooks == 8, "num_codebooks");
        check(meta.layout.codebook_size == 1026, "codebook_size");
        check(meta.layout.n_embd == 2560, "hidden_size");
        check(meta.boc_id == higgs_audio::BOC_ID, "boc_id");
        check(meta.eoc_id == higgs_audio::EOC_ID, "eoc_id");
        check(meta.sample_rate == 24000, "sample_rate");
        check(meta.frame_rate == 25, "frame_rate");
        check(meta.use_delay_pattern, "use_delay_pattern");
        check(meta.n_tensors >= 2, "n_tensors");
        check(meta.codec_tensor_count == 528, "codec_tensor_count");
        check(meta.codec_tensor_names.size() == (size_t) meta.codec_tensor_count, "codec_tensor_names size");
        check(meta.codec_original_tensor_names.size() == (size_t) meta.codec_tensor_count, "codec_original_tensor_names size");
        check(meta.codec_tensor_names.front() == "higgs.codec.0000", "first codec tensor native name");
        check(meta.codec_tensor_names.back() == "higgs.codec.0527", "last codec tensor native name");
        check(meta.codec_original_tensor_names.front().find("acoustic_decoder") != std::string::npos, "first original codec tensor name");
        check(std::any_of(meta.codec_original_tensor_names.begin(), meta.codec_original_tensor_names.end(), [](const std::string & name) {
            return name.find("quantizer.quantizers.0.codebook.embed") != std::string::npos;
        }), "quantizer codebook tensor present");
        check(meta.codebook_embedding.index >= 0, "codebook_embedding.index");
        check(meta.codebook_head.index >= 0, "codebook_head.index");
        check(meta.codebook_embedding.size == expected_codebook_bytes, "codebook_embedding.size");
        check(meta.codebook_head.size == expected_codebook_bytes, "codebook_head.size");

        const auto weights = file.load_codebook_weights();
        check(weights.layout.n_weights() == weights.codebook_embedding.size(), "loaded embedding size");
        check(weights.layout.n_weights() == weights.codebook_head.size(), "loaded head size");

        const std::vector<int> boc_frame((size_t) weights.layout.num_codebooks, higgs_audio::BOC_ID);
        const auto embd = higgs_audio::embed_codebook_frame(weights.layout, weights.codebook_embedding, boc_frame);
        check(embd.size() == (size_t) weights.layout.n_embd, "embedded frame size");

        const auto logits = higgs_audio::project_codebook_logits(weights.layout, weights.codebook_head, embd);
        check(logits.size() == (size_t) weights.layout.num_codebooks, "projected logits codebook count");
        check(logits[0].size() == (size_t) weights.layout.codebook_size, "projected logits vocab size");
        check(std::isfinite(logits[0][0]), "projected logits finite");

        higgs_audio::delay_sampler sampler(weights.layout.num_codebooks);
        const auto codes = higgs_audio::step_greedy_from_hidden(sampler, weights.layout, weights.codebook_head, embd);
        check(codes.size() == (size_t) weights.layout.num_codebooks, "generated codebook frame size");
        check(codes[1] == higgs_audio::BOC_ID, "delay sampler applies BOC to delayed codebooks");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }

    return 0;
}
