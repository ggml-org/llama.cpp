#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
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

static void save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        printf("%s: Failed to open file '%s' for writing", __func__, fname.c_str());
        return;
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

    file.close();
}

static outetts_version get_tts_version(llama_model *model, json speaker = json::object()) {
    if (speaker.contains("version")) {
        std::string version = speaker["version"].get<std::string>();
        if (version == "0.2") {
            return OUTETTS_V0_2;
        } else if (version == "0.3") {
            return OUTETTS_V0_3;
        } else {
            printf("%s: Unsupported speaker version '%s'\n", __func__, version.c_str());
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

static void prompt_add(std::vector<llama_token> & prompt, const llama_vocab * vocab, const std::string & txt, bool add_special, bool parse_special) {
    std::vector<llama_token> tmp(txt.size());
    auto n_tmp = llama_tokenize(vocab, txt.c_str(), txt.size(), tmp.data(), tmp.size(), add_special, parse_special);
    tmp.resize(n_tmp);
    prompt.insert(prompt.end(), tmp.begin(), tmp.end());
}

static void prompt_init(std::vector<llama_token> & prompt, const llama_vocab * vocab) {
    prompt.clear();

    prompt_add(prompt, vocab, "<|im_start|>\n", true, true);
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

static std::vector<llama_token> prepare_guide_tokens(const llama_vocab * vocab, const std::string & str, const outetts_version tts_version = OUTETTS_V0_2) {
    const std::string& delimiter = (tts_version == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = str.find(delimiter);

    //first token is always a newline, as it was not previously added
    result.push_back(llama_vocab_nl(vocab));

    while (end != std::string::npos) {
        std::string current_word = str.substr(start, end - start);
        std::vector<llama_token> tmp(current_word.size());
        auto n_tmp = llama_tokenize(vocab, current_word.c_str(), current_word.size(), tmp.data(), tmp.size(), false, true);
        tmp.resize(n_tmp);
        result.insert(result.end(), tmp.begin(), tmp.end());
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = str.substr(start);
    std::vector<llama_token> tmp(current_word.size());
    auto n_tmp = llama_tokenize(vocab, current_word.c_str(), current_word.size(), tmp.data(), tmp.size(), false, true);
    tmp.resize(n_tmp);
    if (tmp.size() > 0) {
        result.insert(result.end(), tmp.begin(), tmp.end());
    }
    return result;
}

static void batch_add(struct llama_batch & batch, llama_token id,llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
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

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -mv vocoder.gguf -v en_male_1.json -p \"Hello!\"\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // NOTE: parallel processing is incomplete. We can either remove it to make the example simpler,
    // or we can extend the example to support it. The latter is relatively easy to do - just store
    // multiple sets of codes - one for each parallel sequence. And after that, generate multiple audio
    // files - one for each set of codes.
    const int n_parallel = 1;
    const int n_predict  = 4096;

    std::string prompt;
    std::string model_path;
    std::string vocoder_path;
    json speaker;

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-mv") == 0) {
                if (i + 1 < argc) {
                    vocoder_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                if (i + 1 < argc) {
                    std::ifstream file(argv[++i]);
                    if (!file) {
                        fprintf(stderr, "%s: Failed to open file '%s' for reading\n", __func__, argv[i]);
                        return 1;
                    }
                    speaker = json::parse(file);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    prompt = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty() || vocoder_path.empty() || speaker.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    llama_model_params model_params = llama_model_default_params();

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: error: failed to load the model\n", __func__);
        return 1;
    }

    llama_model * vocoder = llama_model_load_from_file(vocoder_path.c_str(), model_params);
    if (!vocoder) {
        fprintf(stderr, "%s: error: failed to load the vocoder\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 8192;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    ctx_params.embeddings = true;

    llama_context * ctx_vocoder = llama_init_from_model(vocoder, ctx_params);
    if (!ctx_vocoder) {
        fprintf(stderr, "%s: error: failed to create the vocoder llama_context\n", __func__);
        return 1;
    }

    std::vector<llama_sampler *> smpl(n_parallel);
    for (int i = 0; i < n_parallel; ++i) {
        smpl[i] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl[i], llama_sampler_init_greedy());
        llama_sampler_chain_add(smpl[i], llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    outetts_version tts_version = get_tts_version(model);

    std::string audio_text = audio_text_from_speaker(speaker, tts_version);
    std::string audio_data = audio_data_from_speaker(speaker, tts_version);

    std::vector<llama_token> prompt_inp;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    prompt_init(prompt_inp, vocab);

    prompt_add(prompt_inp, vocab, audio_text, false, true);

    std::string prompt_clean = process_text(prompt, tts_version);

    std::vector<llama_token> guide_tokens = prepare_guide_tokens(vocab, prompt_clean, tts_version);

    prompt_add(prompt_inp, vocab, prompt_clean, false, true);

    prompt_add(prompt_inp, vocab, "<|text_end|>\n", false, true);

    prompt_add(prompt_inp, vocab, audio_data, false, true);

    // create a llama_batch
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(std::max(prompt_inp.size(), (size_t) n_parallel), 0, n_parallel);

    std::vector<llama_seq_id> seq_ids(n_parallel, 0);
    for (int32_t i = 0; i < n_parallel; ++i) {
        seq_ids[i] = i;
    }

    // evaluate the initial prompt
    for (size_t i = 0; i < prompt_inp.size(); ++i) {
        batch_add(batch, prompt_inp[i], i, seq_ids, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed\n", __func__);
        return 1;
    }

    llama_synchronize(ctx);

    // main loop

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_past   = batch.n_tokens;
    int n_decode = 0;

    bool next_token_uses_guide_token = true;

    std::vector<llama_token> codes;

    while (n_decode <= n_predict) {
        batch.n_tokens = 0;

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            llama_token new_token_id = llama_sampler_sample(smpl[i], ctx, i_batch[i]);

            //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
            if (!guide_tokens.empty() && next_token_uses_guide_token && !llama_vocab_is_control(vocab, new_token_id) && !llama_vocab_is_eog(vocab, new_token_id)) {
                llama_token guide_token = guide_tokens[0];
                guide_tokens.erase(guide_tokens.begin());
                new_token_id = guide_token; //ensure correct word fragment is used
            }

            //this is the token id that always precedes a new word
            next_token_uses_guide_token = (new_token_id == 198);

            llama_sampler_accept(smpl[i], new_token_id);

            codes.push_back(new_token_id);

            if (llama_vocab_is_eog(vocab, new_token_id) || n_decode == n_predict) {
                // Mark the stream as finished
                i_batch[i] = -1;
                continue;
            }

            i_batch[i] = batch.n_tokens;

            batch_add(batch, new_token_id, n_past, { i }, true);
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_decode += 1;
        n_past += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    llama_batch_free(batch);

    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

    for (auto & token : codes) {
        token -= 151672;
    }

    const int n_codes = codes.size();

    batch = llama_batch_init(n_codes, 0, 1);

    for (size_t i = 0; i < codes.size(); ++i) {
        batch_add(batch, codes[i], i, { 0 }, true); // TODO: all logits?
    }

    // evaluate the current batch with the transformer model
    if (llama_decode(ctx_vocoder, batch)) {
        fprintf(stderr, "%s: llama_decode() failed\n", __func__);
        return 1;
    }

    llama_synchronize(ctx_vocoder);

    // spectral operations
    const int n_embd = llama_model_n_embd(vocoder);
    const float * embd = llama_get_embeddings(ctx_vocoder);

    auto audio = embd_to_audio(embd, n_codes, n_embd, ctx_params.n_threads);

    const std::string fname = "output.wav";

    const int n_sr = 24000; // sampling rate

    // zero out first 0.25 seconds
    for (int i = 0; i < 24000/4; ++i) {
        audio[i] = 0.0f;
    }

    save_wav16(fname, audio, n_sr);

    llama_backend_free();

    return 0;
}
