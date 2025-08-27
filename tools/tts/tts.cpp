#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

enum outetts_version {
    OUTETTS_V0_2,
    OUTETTS_V0_3,
};

//
// Terminal utils
//

#define SQR(X)    ((X) * (X))
#define UNCUBE(x) x < 48 ? 0 : x < 115 ? 1 : (x - 35) / 40

/**
 * Quantizes 24-bit RGB to xterm256 code range [16,256).
 */
static int rgb2xterm256(int r, int g, int b) {
    unsigned char cube[] = {0, 0137, 0207, 0257, 0327, 0377};
    int av, ir, ig, ib, il, qr, qg, qb, ql;
    av = r * .299 + g * .587 + b * .114 + .5;
    ql = (il = av > 238 ? 23 : (av - 3) / 10) * 10 + 8;
    qr = cube[(ir = UNCUBE(r))];
    qg = cube[(ig = UNCUBE(g))];
    qb = cube[(ib = UNCUBE(b))];
    if (SQR(qr - r) + SQR(qg - g) + SQR(qb - b) <=
        SQR(ql - r) + SQR(ql - g) + SQR(ql - b))
        return ir * 36 + ig * 6 + ib + 020;
    return il + 0350;
}

static std::string set_xterm256_foreground(int r, int g, int b) {
    int x = rgb2xterm256(r, g, b);
    std::ostringstream oss;
    oss << "\033[38;5;" << x << "m";
    return oss.str();
}

const std::vector<std::string> k_colors = {
    set_xterm256_foreground(220,   5,  12),
    set_xterm256_foreground(232,  96,  28),
    set_xterm256_foreground(241, 147,  45),
    set_xterm256_foreground(246, 193,  65),
    set_xterm256_foreground(247, 240,  86),
    set_xterm256_foreground(144, 201, 135),
    set_xterm256_foreground( 78, 178, 101),
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello!\"\n", argv[0]);
    LOG("\n");
}

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for writing.\n", __func__, fname.c_str());
        return false;
    }

    wav_header header;
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto & sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    return file.good();
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// very poor-man fft
static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

//
//  y = torch.nn.functional.fold(
//       data, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
//  )[:, 0, 0, pad:-pad]
//
// data.shape =  torch.Size([1, 1280, 261])
// output_size =  84480
// win_length =  1280
// hop_length =  320
// pad =  480
//
static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

// TODO: not optimized at all
static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

// Based on: https://github.com/edwko/OuteTTS/blob/a613e79c489d8256dd657ea9168d78de75895d82/outetts/version/v1/prompt_processor.py#L39
static std::string process_text(const std::string & text, const outetts_version tts_version = OUTETTS_V0_2) {

    // For now I skipped text romanization as I am unsure how to handle
    // uroman and MeCab implementations in C++
    // maybe something like https://github.com/anyascii/anyascii/ could work.
    // currently only English would be supported in this function

    std::string processed_text = replace_numbers_with_words(text);

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    /*
        Replace spaces with the separator token same as in line 365

        for (auto & c : prompt_user) {
        if (c == ' ') {
            prompt_clean += "<|text_sep|>";
    */
    std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

static void prompt_add(llama_tokens & prompt, llama_token token) {
    prompt.push_back(token);
}

static void prompt_add(llama_tokens & prompt, const llama_tokens & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}

static void prompt_add(llama_tokens & prompt, const llama_vocab * vocab, const std::string & txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}

static void prompt_init(llama_tokens & prompt, const llama_vocab * vocab) {
    prompt.clear();

    prompt_add(prompt, vocab, "<|im_start|>\n", true, true);
}

static std::vector<llama_token> prepare_guide_tokens(const llama_vocab * vocab, const std::string & str, const outetts_version tts_version = OUTETTS_V0_2) {
    const std::string& delimiter = (tts_version == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = str.find(delimiter);

    //first token is always a newline, as it was not previously added
    result.push_back(common_tokenize(vocab, "\n", false, true)[0]);

    while (end != std::string::npos) {
        std::string current_word = str.substr(start, end - start);
        auto tmp = common_tokenize(vocab, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = str.substr(start);
    auto tmp = common_tokenize(vocab, current_word, false, true);
    if (tmp.size() > 0) {
        result.push_back(tmp[0]);
    }
    return result;
}

static json speaker_from_file(const std::string & speaker_file) {
    std::ifstream file(speaker_file);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for reading\n", __func__, speaker_file.c_str());
        return json();
    }

    json speaker = json::parse(file);
    return speaker;
}

static outetts_version get_tts_version(llama_model *model, json speaker = json::object()) {
    if (speaker.contains("version")) {
        std::string version = speaker["version"].get<std::string>();
        if (version == "0.2") {
            return OUTETTS_V0_2;
        } else if (version == "0.3") {
            return OUTETTS_V0_3;
        } else {
            LOG_ERR("%s: Unsupported speaker version '%s'\n", __func__, version.c_str());
        }
    }

    // Also could get version from model itself
    const char *chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    }

    // Use 0.2 as the default version
    return OUTETTS_V0_2;
}

static std::string audio_text_from_speaker(json speaker, const outetts_version tts_version = OUTETTS_V0_2) {
    std::string audio_text = "<|text_start|>";

    if (tts_version == OUTETTS_V0_2 || tts_version == OUTETTS_V0_3) {
        std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        for (const auto &word : speaker["words"]) {
            audio_text += word["word"].get<std::string>() + separator;
        }
    }

    return audio_text;
}

static std::string audio_data_from_speaker(json speaker, const outetts_version tts_version = OUTETTS_V0_2) {
    std::string audio_data = "<|audio_start|>\n";

    if (tts_version == OUTETTS_V0_2 || tts_version == OUTETTS_V0_3) {
        std::string code_start = (tts_version == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        for (const auto &word : speaker["words"]) {
            std::string word_text = word["word"].get<std::string>();
            double duration = word["duration"].get<double>();
            std::vector<int> codes = word["codes"].get<std::vector<int>>();

            // Create the audio output entry
            std::ostringstream word_entry;
            word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                       << duration << "|>" + code_start;
            for (const auto &Code : codes) {
                word_entry << "<|" << Code << "|>";
            }
            word_entry << code_end << "\n";
            audio_data += word_entry.str();
        }
    }

    return audio_data;
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "output.wav";
    params.prompt = "";

    params.n_predict = 4096;
    params.n_batch   = 8192;
    params.n_ctx     = 8192;

    params.sampling.top_k = 4;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    const int n_parallel = params.n_parallel;
    const int n_predict  = params.n_predict;

    common_init();

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_ttc = NULL; // text-to-codes
    llama_model * model_cts = NULL; // codes-to-speech

    llama_context * ctx_ttc = NULL;
    llama_context * ctx_cts = NULL;

    common_init_result llama_init_ttc = common_init_from_params(params);

    model_ttc = llama_init_ttc.model.get();
    ctx_ttc   = llama_init_ttc.context.get();

    if (model_ttc == nullptr || ctx_ttc == nullptr) {
        return ENOENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_ttc);

    params.model = params.vocoder.model;
    params.embedding = true;
    params.n_ubatch = params.n_batch;

    common_init_result llama_init_cts = common_init_from_params(params);

    model_cts = llama_init_cts.model.get();
    ctx_cts   = llama_init_cts.context.get();

    if (model_cts == nullptr || ctx_cts == nullptr) {
        return ENOENT;
    }

    std::vector<common_sampler *> smpl(n_parallel);
    for (int i = 0; i < n_parallel; ++i) {
        params.sampling.no_perf = (i != 0);
        params.sampling.seed = params.sampling.seed + 1;

        smpl[i] = common_sampler_init(model_ttc, params.sampling);
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl[0]));
    LOG_INF("sampler params: \n%s\n", params.sampling.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl[0]).c_str());

    LOG_INF("%s: loading done\n", __func__);

    const auto t_main_start = ggml_time_us();

    std::vector<llama_token> codes;
    std::vector<llama_token> guide_tokens;

    // the default speaker profile is from: https://github.com/edwko/OuteTTS/blob/main/outetts/version/v1/default_speakers/en_male_1.json
    std::string audio_text = "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>";
    std::string audio_data = R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>
two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>
people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>
is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>
pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>
remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>
sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>
i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>
have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>
some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>
critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>
about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>
some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>
of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>
the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>
gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>
aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>
but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>
its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>
still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>
really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>
enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>
and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>
it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>
looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>
lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>)";

    // audio data for 0.3 version
    outetts_version tts_version = get_tts_version(model_ttc);
    if (tts_version == OUTETTS_V0_3) {
        audio_text = std::regex_replace(audio_text, std::regex(R"(<\|text_sep\|>)"), "<|space|>");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_start\|>)"), "");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_end\|>)"), "<|space|>");
    }

    // load speaker if given
    if (!params.vocoder.speaker_file.empty()) {
        LOG_INF("%s: loading speaker ..\n", __func__);
        json speaker = speaker_from_file(params.vocoder.speaker_file);
        if (speaker.empty()) {
            LOG_ERR("%s: Failed to load speaker file '%s'\n", __func__, params.vocoder.speaker_file.c_str());
            return 1;
        }
        audio_text = audio_text_from_speaker(speaker, tts_version);
        audio_data = audio_data_from_speaker(speaker, tts_version);
    }

    // process prompt and generate voice codes
    {
        LOG_INF("%s: constructing prompt ..\n", __func__);

        std::vector<llama_token> prompt_inp;

        prompt_init(prompt_inp, vocab);

        prompt_add(prompt_inp, vocab, audio_text, false, true);

        // convert the input text into the necessary format expected by OuteTTS
        {
            std::string prompt_clean = process_text(params.prompt, tts_version);
            if (params.vocoder.use_guide_tokens) {
                guide_tokens = prepare_guide_tokens(vocab, prompt_clean, tts_version);
            }

            LOG_INF("%s: prompt: '%s'\n", __func__, prompt_clean.c_str());

            prompt_add(prompt_inp, vocab, prompt_clean, false, true);
        }

        prompt_add(prompt_inp, vocab, "<|text_end|>\n", false, true);

        if (!params.vocoder.speaker_file.empty()) {
            prompt_add(prompt_inp, vocab, audio_data, false, true);
        } else {
            // disabled to save time on tokenizing each time
#if 1
            const std::string voice_data = audio_data;

            auto tmp = common_tokenize(vocab, voice_data, false, true);

            std::ostringstream tokens_oss;
            for (size_t i = 0; i < tmp.size(); ++i) {
                tokens_oss << tmp[i] << ", ";
            }
            LOG_INF("\n\n%s: llama tokens: %s\n\n", __func__, tokens_oss.str().c_str());

            prompt_add(prompt_inp, tmp);
#else
            prompt_add(prompt_inp, llama_tokens {
                151667, 198, 1782, 155780, 151669, 151929, 152412, 152308, 152585,
                152460, 153375, 151670, 198, 74455, 155808, 151669, 151799,
                151873, 151863, 152446, 152372, 152204, 152728, 152229, 152470,
                151970, 153413, 152419, 153334, 153289, 153374, 153199, 152040,
                153260, 152721, 152680, 153297, 152419, 153248, 152400, 152691,
                153368, 153437, 151670, 198, 1722, 155828, 151669, 152607,
                152256, 152991, 152299, 152688, 153163, 153016, 152789, 153198,
                152712, 151911, 153107, 152623, 152170, 152395, 152852, 152207,
                152461, 153321, 153309, 151750, 152137, 153340, 152573, 152267,
                153347, 151789, 152681, 153339, 151992, 152512, 151751, 152179,
                153434, 153180, 152900, 153440, 152474, 153122, 153129, 151904,
                152311, 151670, 198, 1499, 155791, 151669, 152276, 152454,
                153354, 152544, 153204, 153272, 152708, 153433, 152319, 153226,
                153043, 152325, 153267, 152622, 151670, 198, 4250, 155797,
                151669, 153454, 153342, 151989, 152458, 153420, 152303, 152271,
                152827, 153036, 153196, 151708, 153263, 152561, 153207, 152213,
                152112, 153204, 151722, 152542, 151670, 198, 19789, 155796,
                151669, 153353, 153182, 152345, 152471, 152477, 153014, 152002,
                152191, 151734, 152312, 152810, 152237, 153224, 153169, 153224,
                152244, 153387, 153404, 151670, 198, 16069, 155811, 151669,
                152265, 151946, 151808, 152412, 152363, 152305, 153156, 152733,
                152810, 153157, 152016, 152100, 152069, 153234, 152317, 152589,
                152707, 153121, 153341, 152159, 152114, 153156, 153001, 153504,
                153376, 152272, 152433, 152325, 151941, 151670, 198, 285,
                155788, 151669, 152238, 152255, 153427, 152318, 153009, 152381,
                152474, 152680, 152157, 153255, 152324, 151682, 151670, 198,
                32955, 155804, 151669, 153490, 153419, 152364, 152405, 152682,
                152206, 152078, 153369, 152725, 153193, 153027, 152946, 152488,
                153070, 151883, 152890, 152489, 153144, 153375, 152358, 151685,
                152494, 152117, 152740, 151670, 198, 37448, 480, 155840, 151669,
                151902, 152720, 153377, 152027, 152378, 152821, 153207, 153459,
                153028, 153068, 152507, 153255, 152158, 152921, 151958, 152609,
                152748, 152822, 152286, 151714, 152730, 152377, 152353, 152470,
                152606, 152162, 152186, 153071, 152244, 153118, 153375, 153018,
                152712, 153098, 152976, 152336, 151843, 153202, 152297, 151736,
                153380, 153502, 152702, 152115, 153181, 152735, 153277, 153457,
                152393, 153112, 152595, 151670, 198, 19098, 155808, 151669,
                152464, 153452, 152595, 153312, 151937, 151933, 153197, 152239,
                153163, 152922, 153402, 152034, 152591, 153438, 152215, 151673,
                152005, 151785, 152642, 151924, 153278, 151805, 151974, 153482,
                152718, 152862, 153347, 151670, 198, 72, 155780, 151669, 151795,
                152111, 152746, 152377, 153471, 152309, 151670, 198, 19016,
                155788, 151669, 153181, 152271, 152190, 152842, 152224, 152701,
                152939, 152536, 152091, 151815, 152733, 151672, 151670, 198,
                14689, 155788, 151669, 152291, 152072, 152942, 151734, 153042,
                153504, 152589, 153333, 151839, 151941, 153038, 153180, 151670,
                198, 36996, 8303, 155832, 151669, 152231, 152256, 152835,
                152801, 152985, 153400, 152393, 152818, 152765, 152249, 152600,
                151699, 152302, 152752, 153018, 153009, 151992, 153054, 152847,
                153354, 153228, 152662, 153355, 152532, 153393, 151782, 152458,
                152048, 152757, 152428, 153195, 151906, 153006, 153178, 153250,
                152331, 152284, 152780, 153138, 153319, 151980, 153142, 152418,
                152228, 152733, 151670, 198, 9096, 155801, 151669, 151698,
                153321, 152217, 153039, 152935, 153400, 152122, 152531, 153106,
                152169, 152892, 152957, 151851, 152427, 152826, 152451, 151851,
                152901, 152885, 152594, 153446, 153080, 151670, 198, 14689,
                155795, 151669, 152658, 151700, 153321, 152450, 152530, 153191,
                151673, 151690, 151698, 152714, 152846, 152981, 153171, 153384,
                153364, 153188, 153246, 151670, 198, 1055, 155779, 151669,
                151869, 152388, 152711, 153334, 151736, 151670, 198, 1782,
                155780, 151669, 153483, 153240, 152241, 152558, 152697, 153046,
                151670, 198, 5804, 1363, 155820, 151669, 152941, 152764, 152605,
                153034, 153434, 153372, 153347, 151887, 152453, 152758, 152133,
                152510, 152694, 152431, 152321, 153088, 152676, 152223, 152581,
                152459, 152015, 152502, 153063, 152712, 153294, 153451, 153032,
                152903, 152859, 152989, 151748, 152669, 152661, 152650, 152409,
                151861, 151670, 198, 300, 7973, 155828, 151669, 153095, 152469,
                152988, 152894, 151819, 152391, 153019, 152058, 153062, 153230,
                151826, 152112, 152306, 152264, 152769, 153390, 152384, 152435,
                152790, 153393, 152983, 152540, 152252, 152034, 153107, 152540,
                151919, 151893, 152558, 152817, 152946, 152956, 152129, 152715,
                153131, 153490, 151734, 152271, 152707, 151734, 153321, 152450,
                151670, 198, 8088, 155792, 151669, 152452, 153497, 153353,
                152679, 152533, 152382, 152374, 152611, 153341, 153163, 152285,
                153411, 152495, 153141, 152320, 151670, 198, 1199, 155781,
                151669, 151764, 152360, 153295, 152634, 153342, 152199, 152271,
                151670, 198, 43366, 155799, 151669, 152308, 151682, 152889,
                152016, 152385, 152629, 152495, 151826, 153321, 152958, 152180,
                151886, 153432, 152922, 152128, 153024, 153040, 152593, 152287,
                151677, 151670, 198, 53660, 155808, 151669, 151727, 152092,
                152680, 153331, 151699, 152316, 152938, 152289, 152433, 153384,
                151781, 153137, 153259, 152175, 153213, 152291, 151869, 152691,
                152489, 151941, 152049, 152034, 153053, 152179, 153160, 151676,
                153367, 151670, 198, 268, 4123, 480, 155821, 151669, 152350,
                152173, 152536, 151991, 151960, 153144, 153013, 152358, 152234,
                153135, 152291, 153235, 152143, 152583, 152402, 153483, 152678,
                152192, 152533, 152946, 151797, 153103, 152310, 152293, 151825,
                152548, 153442, 152109, 152659, 153325, 152781, 152570, 152957,
                151752, 152265, 153381, 152515, 151670, 198, 437, 155787,
                151669, 152957, 152659, 151975, 152709, 152402, 152836, 152174,
                151792, 153409, 153327, 152990, 151670, 198, 275, 155781,
                151669, 152520, 153038, 152067, 153273, 153185, 152265, 152974,
                151670, 198, 94273, 155799, 151669, 152953, 152938, 153427,
                152244, 151920, 153423, 152929, 152367, 153052, 152129, 152331,
                152257, 152987, 152777, 153448, 152408, 151696, 152408, 152326,
                152699, 151670, 198, 385, 16239, 155828, 151669, 152306, 152268,
                153438, 153228, 152978, 152957, 153153, 153393, 152795, 152110,
                152918, 152923, 152467, 152331, 153053, 153330, 151889, 153444,
                152234, 152624, 151779, 152801, 152784, 152139, 152222, 152751,
                152512, 153287, 153141, 153052, 151840, 152589, 152508, 153499,
                152109, 152255, 151739, 152267, 152759, 153318, 153165, 153349,
                151670,});
#endif
        }

        // print the prompt token-by-token

        LOG("\n");

        for (auto id : prompt_inp) {
            LOG("%s", common_token_to_piece(ctx_ttc, id).c_str());
        }

        LOG_INF("%s: prompt size: %d\n", __func__, (int) prompt_inp.size());

        LOG("\n");

        // create a llama_batch
        // we use this object to submit token data for decoding
        llama_batch batch = llama_batch_init(std::max(prompt_inp.size(), (size_t) n_parallel), 0, n_parallel);

        std::vector<llama_seq_id> seq_ids(n_parallel, 0);
        for (int32_t i = 0; i < n_parallel; ++i) {
            seq_ids[i] = i;
        }

        // evaluate the initial prompt
        for (size_t i = 0; i < prompt_inp.size(); ++i) {
            common_batch_add(batch, prompt_inp[i], i, seq_ids, false);
        }
        GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx_ttc, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        if (n_parallel > 1) {
            LOG_INF("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
        }

        llama_synchronize(ctx_ttc);

        LOG_INF("%s: time for prompt: %.3f ms\n\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

        const auto t_dec_start = ggml_time_us();

        // main loop

        // remember the batch index of the last token for each parallel sequence
        // we need this to determine which logits to sample from
        std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

        int n_past   = batch.n_tokens;
        int n_decode = 0;

        bool next_token_uses_guide_token = true;

        while (n_decode <= n_predict) {
            // prepare the next batch
            common_batch_clear(batch);

            // sample the next token for each parallel sequence / stream
            for (int32_t i = 0; i < n_parallel; ++i) {
                if (i_batch[i] < 0) {
                    // the stream has already finished
                    continue;
                }

                llama_token new_token_id = common_sampler_sample(smpl[i], ctx_ttc, i_batch[i]);

                //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
                if (!guide_tokens.empty() && next_token_uses_guide_token && !llama_vocab_is_control(vocab, new_token_id) && !llama_vocab_is_eog(vocab, new_token_id)) {
                    llama_token guide_token = guide_tokens[0];
                    guide_tokens.erase(guide_tokens.begin());
                    new_token_id = guide_token; //ensure correct word fragment is used
                }

                //this is the token id that always precedes a new word
                next_token_uses_guide_token = (new_token_id == 198);

                common_sampler_accept(smpl[i], new_token_id, true);

                codes.push_back(new_token_id);

                const auto * cands = common_sampler_get_candidates(smpl[i]);

                // is it an end of generation? -> mark the stream as finished
                if (llama_vocab_is_eog(vocab, new_token_id) || n_decode == n_predict) {
                    std::string reason;
                    if (llama_vocab_is_eog(vocab, new_token_id)) {
                        reason = "eos";
                    } else {
                        reason = "n_predict";
                    }

                    i_batch[i] = -1;

                    LOG("\n");
                    if (n_parallel > 1) {
                        LOG_CNT("\n");
                        LOG_INF("%s: stream %d finished at n_past = %d, reason = '%s'\n", __func__, i, n_past, reason.c_str());
                    }

                    continue;
                }

                {
                    const float p = cands->data[cands->selected].p;

                    const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) ((3*p)*float(k_colors.size()))));

                    LOG_CNT("%s%d%s", k_colors[col].c_str(), i, "\033[0m");
                    //LOG_CNT("%d", i);
                }

                i_batch[i] = batch.n_tokens;

                // push this new token for next evaluation
                common_batch_add(batch, new_token_id, n_past, { i }, true);
            }

            // all streams are finished
            if (batch.n_tokens == 0) {
                break;
            }

            n_decode += 1;
            n_past += 1;

            // evaluate the current batch with the transformer model
            if (llama_decode(ctx_ttc, batch)) {
                LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }
        }

        llama_batch_free(batch);

        LOG("\n");
        LOG_INF("%s: time for decoder:       %.3f ms\n", __func__, (ggml_time_us() - t_dec_start) / 1000.0f);
    }

    common_perf_print(ctx_ttc, smpl[0]);

    //std::vector<llama_token> codes = {198, 88225, 155856, 151669, 152205,
    //    153064, 152537, 153421, 153209, 152524, 151689, 152993, 152438, 152695,
    //    153091, 152945, 152829, 152534, 152934, 153020, 151997, 152263, 153010,
    //    153146, 152399, 153208, 152496, 151793, 152848, 152263, 152571, 153286,
    //    152227, 153300, 152934, 152263, 153208, 152263, 152965, 152430, 152296,
    //    153146, 152920, 152376, 152556, 153363, 151775, 152044, 152972, 152690,
    //    153379, 152368, 152233, 153422, 152490, 151996, 152022, 151694, 152061,
    //    153238, 152539, 153356, 152640, 153021, 153123, 151962, 153094, 151670,
    //    198, 20339, 13189, 155824, 151669, 152070, 152007, 152910, 151683,
    //    152000, 152373, 152760, 152046, 151735, 152334, 152394, 153073, 152908,
    //    151856, 151953, 153247, 153293, 151903, 153480, 153168, 152478, 153359,
    //    153429, 151905, 151678, 152567, 152411, 152165, 152556, 153075, 153424,
    //    151993, 152999, 153078, 152151, 152088, 153389, 152484, 151874, 151670,
    //    198, 285, 155784, 151669, 152226, 152126, 152638, 153215, 151729,
    //    152959, 153479, 153059, 151838, 151670, 198, 1782, 155783, 151669,
    //    153288, 153055, 153314, 152497, 152962, 152741, 152076, 153253, 151670,
    //    198, 471, 16488, 155825, 151669, 152060, 152916, 151893, 153469, 152501,
    //    152080, 152743, 151932, 153161, 152096, 152761, 152698, 153401, 153242,
    //    153336, 152441, 152838, 153467, 152706, 153496, 153310, 152422, 153360,
    //    153115, 152763, 151998, 152373, 153450, 152554, 151968, 153323, 152055,
    //    152468, 153111, 153358, 152813, 152010, 151770, 152823, 152960, 151670,
    //    198, 22627, 155823, 151669, 152814, 152366, 153484, 152931, 153441,
    //    152164, 152877, 152915, 153463, 151692, 152911, 152747, 152776, 151831,
    //    153449, 151882, 152975, 152031, 152513, 153150, 152448, 152667, 153133,
    //    153189, 152619, 153466, 152054, 152106, 153119, 152277, 152439, 153109,
    //    152997, 152141, 153154, 153256, 153311, 151922, 151670, 198, 1055,
    //    155781, 151669, 152633, 151850, 153060, 153270, 152560, 153348, 152729,
    //    151670, 198, 25312, 155803, 151669, 152521, 153403, 152561, 153337,
    //    153383, 152199, 153493, 153326, 151830, 152254, 152248, 152349, 152153,
    //    153007, 151823, 153037, 152575, 152457, 152406, 152592, 153116, 153365,
    //    153456, 151670, 198, 88225, 155817, 151669, 153271, 151925, 152218,
    //    152418, 152253, 153140, 151903, 153151, 152626, 152338, 152647, 153464,
    //    152785, 152768, 151711, 152037, 152033, 151804, 152216, 151701, 151855,
    //    152348, 152995, 152955, 152905, 152342, 152340, 153391, 153453, 152418,
    //    153415, 151990, 153083, 152884, 151670, 198, 151668, 198, 151645};

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);

        LOG("\n");
        LOG_INF("codes: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: codes size: %d\n", __func__, (int) codes.size());
    }

    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

    {
        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);
        LOG_INF("codes audio: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: codes audio size: %d\n", __func__, (int) codes.size());
    }

    for (auto & token : codes) {
        token -= 151672;
    }

    const auto t_voc_start = ggml_time_us();

    const int n_codes = codes.size();

    llama_batch batch = llama_batch_init(n_codes, 0, 1);

    for (size_t i = 0; i < codes.size(); ++i) {
        common_batch_add(batch, codes[i], i, { 0 }, true); // TODO: all logits?
    }
    GGML_ASSERT(batch.n_tokens == n_codes);

    if (llama_encode(ctx_cts, batch) != 0) {
        LOG_ERR("%s: llama_encode() failed\n", __func__);
        return 1;
    }

    llama_synchronize(ctx_cts);

    LOG_INF("%s: time for vocoder:      %.3f ms\n", __func__, (ggml_time_us() - t_voc_start) / 1000.0f);

    const auto t_spec_start = ggml_time_us();

#if 1
    // spectral operations
    const int n_embd = llama_model_n_embd(model_cts);
    const float * embd = llama_get_embeddings(ctx_cts);

    auto audio = embd_to_audio(embd, n_codes, n_embd, params.cpuparams.n_threads);

#else
    // read the spectrogram from a file for debugging purposes
    std::vector<float> audio;
    {
        std::ifstream fin("out.bin", std::ios::binary);
        if (!fin) {
            LOG_ERR("%s: failed to open file '%s'\n", __func__, "out.bin");
            return 1;
        }

        std::vector<float> embd;

        int n_codes;
        int n_embd;

        fin.read(reinterpret_cast<char *>(&n_codes), sizeof(int));
        fin.read(reinterpret_cast<char *>(&n_embd), sizeof(int));

        embd.resize(n_codes * n_embd);
        fin.read(reinterpret_cast<char *>(embd.data()), n_codes * n_embd * sizeof(float));
        fin.close();

        LOG_INF("%s: n_codes: %d, n_embd: %d\n", __func__, n_codes, n_embd);

        audio = embd_to_audio(embd.data(), n_codes, n_embd, params.cpuparams.n_threads);
    }
#endif

    const int n_sr = 24000; // sampling rate

    // zero out first 0.25 seconds
    for (int i = 0; i < 24000/4; ++i) {
        audio[i] = 0.0f;
    }

    LOG_INF("%s: time for spectral ops: %.3f ms\n", __func__, (ggml_time_us() - t_spec_start) / 1000.0f);
    LOG_INF("%s: total time:            %.3f ms\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

    int retval = 0;

    if (save_wav16(params.out_file, audio, n_sr)) {
        LOG_INF("%s: audio written to file '%s'\n", __func__, params.out_file.c_str());
    } else {
        retval = ENOENT;
    }

    llama_backend_free();

    return retval;
}
