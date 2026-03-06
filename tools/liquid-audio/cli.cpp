#include "mtmd-helper.h"
#include "mtmd.h"
#include "runner.h"

//
#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>

namespace {
std::vector<std::byte> load_file(const char * fname) {
    std::vector<std::byte> buf;
    FILE *                 f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t) file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        exit(1);
    }

    return buf;
}
}  // namespace

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#    include <signal.h>
#    include <unistd.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <signal.h>
#    include <windows.h>
#endif

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG("CLI for LFM2.5-Audio-1.5B\n\n"
        "Usage: %s [options] -m <model.gguf> --mmproj <mmproj.gguf> -mv <vocoder.gguf> --tts-speaker-file "
        "<tokenizer.gguf> "
        "-sys <system_prompt> [--audio "
        "<audio>] [-p <user_prompt>]\n"
        "  --audio, -p, --output can be required depending on <system_prompt>\n",
        argv[0]);
}

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        // TODO(tarek): make this more graceful
        LOG("Force exiting...\n");
        exit(1);
    }
}
#endif

static std::vector<mtmd_output_modality> get_modalities_from_system_prompt(std::string const & system_prompt) {
    if (system_prompt.empty()) {
        LOG_ERR("ERR: -sys is required\n");
        exit(1);
    }

    // modalities depend on system prompt
    static constexpr const char *         asr_system_prompt         = "Perform ASR.";
    static constexpr const char *         interleaved_system_prompt = "Respond with interleaved text and audio.";
    static const std::vector<std::string> tts_system_prompts        = {
        "Perform TTS. Use the US male voice.",
        "Perform TTS. Use the UK male voice.",
        "Perform TTS. Use the US female voice.",
        "Perform TTS. Use the UK female voice.",
    };
    if (system_prompt == asr_system_prompt) {
        return { MTMD_OUTPUT_MODALITY_TEXT };
    }
    if (system_prompt == interleaved_system_prompt) {
        return { MTMD_OUTPUT_MODALITY_AUDIO, MTMD_OUTPUT_MODALITY_TEXT };
    }
    if (std::find(begin(tts_system_prompts), end(tts_system_prompts), system_prompt) != end(tts_system_prompts)) {
        return { MTMD_OUTPUT_MODALITY_AUDIO };
    }

    // print error and exit
    std::vector<std::string> prompts = tts_system_prompts;
    prompts.push_back(asr_system_prompt);
    prompts.push_back(interleaved_system_prompt);
    std::string err = "Unsupported system prompt. Supported prompts are:\n";
    for (const auto & p : prompts) {
        err += " - " + p + "\n";
    }

    LOG_ERR("%s", err.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    // Ctrl+C handling
    {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset(&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    ggml_time_init();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LIQUID_AUDIO, show_additional_info)) {
        show_additional_info(argc, argv);
        return 1;
    }

    common_init();

    // set default context size if not specified
    if (params.n_ctx == 0) {
        params.n_ctx = 4096;
    }

    liquid::audio::Runner runner;
    if (0 != runner.init(params)) {
        exit(1);
    }

    auto modalities = get_modalities_from_system_prompt(params.system_prompt);

    // prepare inputs
    std::vector<liquid::audio::Runner::Message> messages;
    messages.push_back({ "system", params.system_prompt, {} });
    if (!params.prompt.empty()) {
        messages.push_back({ "user", params.prompt, {} });
    }
    if (!params.image.empty()) {
        messages.push_back({ "user", mtmd_default_marker(), load_file(params.image[0].c_str()) });
    }

    std::string                      generated_text;
    liquid::audio::generated_audio_t generated_audio;

    auto text_cb = [&generated_text](const std::string & text) {
        generated_text += text;
    };
    auto audio_cb = [&generated_audio](const std::vector<int16_t> & audio) {
        generated_audio.insert(generated_audio.end(), audio.begin(), audio.end());
    };

    if (0 != runner.generate(messages, params.n_predict, text_cb, audio_cb, modalities)) {
        exit(1);
    }

    LOG("\n");

    // write output
    if (not generated_audio.empty()) {
        if (params.out_file.empty()) {
            LOG_ERR("ERR: --output is required for audio generation\n");
            return 1;
        }
        if (!mtmd_helper_save_wav(params.out_file.c_str(), generated_audio.data(), generated_audio.size(),
                                  runner.get_output_sample_rate())) {
            exit(1);
        }
        LOG("=== GENERATED AUDIO ===\nSaved to %s\n\n", params.out_file.c_str());
    }

    if (not generated_text.empty()) {
        LOG("=== GENERATED TEXT ===\n%s\n\n", generated_text.c_str());
    }

    return 0;
}
