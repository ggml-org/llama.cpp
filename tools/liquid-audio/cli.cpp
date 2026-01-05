#include "mtmd.h"
#include "runner.h"

//
#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <optional>

// for wav
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

namespace {
bool save_wav(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, sample_rate);
    ma_encoder        encoder;

    ma_result res = ma_encoder_init_file(fname.c_str(), &config, &encoder);
    if (res != MA_SUCCESS) {
        LOG_ERR("%s: Failed to open file '%s' for writing (error %d).\n", __func__, fname.c_str(), res);
        return false;
    }

    ma_uint64 frames_written;
    ma_result result = ma_encoder_write_pcm_frames(&encoder, data.data(), data.size(), &frames_written);
    ma_encoder_uninit(&encoder);

    return result == MA_SUCCESS && frames_written == data.size();
}

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
        "Usage: %s [options] -m <model.gguf> --mmproj <mmproj.gguf> -mv <mv.gguf> -sys <system_prompt> [--audio "
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

    // prepare inputs
    std::vector<liquid::audio::Runner::Message> messages;
    if (params.system_prompt.empty()) {
        LOG_ERR("ERR: -sys is required\n");
        return 1;
    }
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
    auto audio_cb = [&generated_audio](const std::vector<float> & audio) {
        generated_audio.insert(generated_audio.end(), audio.begin(), audio.end());
    };

    if (0 != runner.generate(messages, params.n_predict, text_cb, audio_cb)) {
        exit(1);
    }

    LOG("\n");

    // write output
    if (not generated_audio.empty()) {
        if (params.out_file.empty()) {
            LOG_ERR("ERR: --output is required for audio generation\n");
            return 1;
        }
        if (!save_wav(params.out_file, generated_audio, runner.get_output_sample_rate())) {
            exit(1);
        }
        LOG("=== GENERATED AUDIO ===\nSaved to %s\n\n", params.out_file.c_str());
    }

    if (not generated_text.empty()) {
        LOG("=== GENERATED TEXT ===\n%s\n\n", generated_text.c_str());
    }

    return 0;
}
