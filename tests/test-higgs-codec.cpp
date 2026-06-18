#include "../tools/tts/higgs-codec.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
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
        std::fprintf(stderr, "HIGGS_AUDIO_GGUF not set; skipping local Higgs codec read-back\n");
        return 0;
    }

    try {
        const higgs_audio::companion_file file(path);
        const auto weights = higgs_audio::load_rvq_decoder_weights(file);
        const auto dac_encoder_weights = higgs_audio::load_dac_encoder_weights(file);
        const auto dac_weights = higgs_audio::load_dac_decoder_weights(file);

        const auto check = [](bool ok, const char * message) {
            if (!ok) {
                std::fprintf(stderr, "check failed: %s\n", message);
                std::exit(1);
            }
        };

        check((int) weights.quantizers.size() == weights.num_quantizers, "quantizer count");
        check(weights.quantizers[0].codebook_embed.size() == (size_t) weights.codebook_size * (size_t) weights.codebook_dim, "codebook size");
        check(weights.quantizers[0].project_in_weight.size() == (size_t) weights.codebook_dim * (size_t) weights.hidden_size, "project_in weight size");
        check(weights.quantizers[0].project_in_bias.size() == (size_t) weights.codebook_dim, "project_in bias size");
        check(weights.fc2_weight.size() == (size_t) weights.acoustic_size * (size_t) weights.hidden_size, "fc2 weight size");
        check(weights.fc2_bias.size() == (size_t) weights.acoustic_size, "fc2 bias size");
        check(dac_encoder_weights.conv1.weight.size() == 64 * 1 * 7, "DAC encoder conv1 size");
        check(dac_encoder_weights.conv2.weight.size() == 256 * 2048 * 3, "DAC encoder conv2 size");

        const std::vector<std::vector<int>> codec_frames {
            { 990, 873, 553, 202, 916, 552,  53, 151 },
            { 990, 873, 553, 202, 916,  75, 798, 544 },
            { 990, 873, 553, 202, 916,  75, 798, 606 },
        };

        const auto acoustic = higgs_audio::decode_rvq_acoustic_latents(weights, codec_frames);
        check(acoustic.size() == codec_frames.size() * (size_t) weights.acoustic_size, "acoustic latent size");
        check(higgs_audio::all_finite(acoustic), "acoustic latents finite");

        bool any_nonzero = false;
        for (const float v : acoustic) {
            if (std::fabs(v) > 0.0f) {
                any_nonzero = true;
                break;
            }
        }
        check(any_nonzero, "acoustic latents are nonzero");

        const auto pcm = higgs_audio::dac_decode_pcm(dac_weights, acoustic, (int) codec_frames.size());
        check(pcm.size() == codec_frames.size() * 960, "DAC PCM size");
        check(higgs_audio::all_finite(pcm), "DAC PCM finite");

        std::vector<float> synthetic_pcm(2400);
        for (size_t i = 0; i < synthetic_pcm.size(); ++i) {
            synthetic_pcm[i] = 0.1f * std::sin((float) i * 0.013f);
        }
        const auto encoded_acoustic = higgs_audio::dac_encode_acoustic_latents(dac_encoder_weights, synthetic_pcm);
        check(!encoded_acoustic.empty(), "DAC encoder produced latents");
        check(encoded_acoustic.size() % (size_t) dac_encoder_weights.acoustic_size == 0, "DAC encoder latent alignment");
        check(higgs_audio::all_finite(encoded_acoustic), "DAC encoder latents finite");

        std::vector<float> hidden_frames((size_t) 2 * (size_t) weights.hidden_size);
        for (size_t i = 0; i < hidden_frames.size(); ++i) {
            hidden_frames[i] = 0.01f * std::sin((float) i * 0.017f);
        }
        const auto encoded_codes = higgs_audio::encode_rvq_codes(weights, hidden_frames);
        check(encoded_codes.size() == 2, "RVQ encoder frame count");
        check(encoded_codes[0].size() == (size_t) weights.num_quantizers, "RVQ encoder quantizer count");
        for (const auto & frame : encoded_codes) {
            for (const int code : frame) {
                check(code >= 0 && code < weights.codebook_size, "RVQ encoder code range");
            }
        }

        std::fprintf(stderr,
                "decoded Higgs RVQ/DAC and encoded native DAC/RVQ: frames=%zu channels=%d latent_values=%zu pcm_samples=%zu encoded_latents=%zu encoded_codes=%zu first=%f\n",
                codec_frames.size(),
                weights.acoustic_size,
                acoustic.size(),
                pcm.size(),
                encoded_acoustic.size(),
                encoded_codes.size(),
                acoustic.front());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }

    return 0;
}
