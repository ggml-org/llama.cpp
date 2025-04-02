#include "common.h"
#include "llama.h"
#include "llama-impl.h"
#include "log.h"
#include "arg.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <fstream>
#include <string>
#include <cstdarg>

std::vector<llama_token> redistribute_codes(const std::vector<llama_token>& raw_codes) {  
    std::vector<llama_token> snac_codes;  
    for (size_t i = 0; i < raw_codes.size(); i += 7) {  
        // Ensure we have a full frame (7 codes)  
        if (i + 6 >= raw_codes.size()) break;  

        // Frame offsets (per notebook)  
        snac_codes.push_back(raw_codes[i]);      // Codebook 0 (no offset)  
        snac_codes.push_back(raw_codes[i+1] - 4096);  // Codebook 1  
        snac_codes.push_back(raw_codes[i+2] - 8192);   // Codebook 2  
        snac_codes.push_back(raw_codes[i+3] - 12288);  // Codebook 2  
        snac_codes.push_back(raw_codes[i+4] - 16384);  // Codebook 1  
        snac_codes.push_back(raw_codes[i+5] - 20480);   // Codebook 2  
        snac_codes.push_back(raw_codes[i+6] - 24576);   // Codebook 2  
    }  
    return snac_codes;  
}

static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread);
static bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate);
static void fill_hann_window(int length, bool periodic, float * output);
static void irfft(int n, const float * inp_cplx, float * out_real);
static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output);

static void print_usage(int /*argc*/, char **argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -mv vocoder.gguf -p \"Hello world\"\n", argv[0]);
    LOG("\n");
}

static void prompt_add(std::vector<llama_token> &prompt, const llama_vocab *vocab, const std::string &txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt.insert(prompt.end(), tmp.begin(), tmp.end());
}


// // Include embd_to_audio and save_wav16 from tts.cpp (for now)
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
    fold(hann2, n_out, n_win, n_hop, n_pad, env);

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

static bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for writing.\n", __func__, fname.c_str());
        return false;
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
    } header;

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

// Supporting functions from tts.cpp (for embd_to_audio)
static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

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

int main(int argc, char **argv) {
    common_params params;
    
    params.model = "models/orpheus-3b-0.1-ft-q4_k_m.gguf";
    params.vocoder.model = "models/snac-vocab.gguf";
    params.out_file = "output.wav";

    params.n_predict = 1200;
    params.sampling.top_k = 4;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.n_batch = 4096;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    
    common_init_result orpheus_init_ttc = common_init_from_params(params);

    llama_model * model_ttc = NULL;
    llama_context * ctx_ttc = NULL;

    model_ttc = orpheus_init_ttc.model.get();
    ctx_ttc = orpheus_init_ttc.context.get();

    const llama_vocab *vocab = llama_model_get_vocab(model_ttc);

    common_sampler *sampler = common_sampler_init(model_ttc, params.sampling);
    if (!sampler) {
        LOG_ERR("Failed to initialize sampler\n");
        return 1;
    }

    // Construct prompt: <|startofhuman|> tara: [prompt] <normal> <|eot_id|> <|endofhuman|>
    std::vector<llama_token> tokens;
    tokens.push_back(128259); // <|startofhuman|>
    prompt_add(tokens, vocab, "tara: ", false, true); // Voice prefix
    prompt_add(tokens, vocab, params.prompt, false, true); // User prompt
    prompt_add(tokens, vocab, "<normal>", false, true); // Emotion tag
    tokens.push_back(128009); // <|eot_id|>
    tokens.push_back(128260); // <|endofhuman|>
    

    llama_model * model_cts = NULL;
    llama_context * ctx_cts = NULL;

    params.model = params.vocoder.model;
    params.n_batch = 2;

    params.embedding = true;
    // disable warmup, SNAC doesn't care about BOS or EOS tokens;
    params.warmup = false; 

    common_init_result snac_init_cts = common_init_from_params(params);
    LOG_INF("SNAC model loaded: %s\n", params.model.c_str());

    model_cts = snac_init_cts.model.get();
    ctx_cts   = snac_init_cts.context.get();

    std::vector<llama_token> speech_codes = {100, 4200, 8500, 12500, 16500, 21000, 25000,
                                             200, 4300, 8600, 12600, 16600, 21111, 25100};

    std::vector<llama_token> snac_codes = redistribute_codes(speech_codes);

    const int n_codes = speech_codes.size();
    const int batch_size = n_codes;
    
    llama_batch batch = llama_batch_init(batch_size, 0, 1);    

    for (size_t i = 0; i < n_codes; ++i) {
        common_batch_add(batch, snac_codes[i], i, {0}, true);
    }

    LOG_INF("Batch before decode: n_tokens = %d\n", batch.n_tokens);
    if (llama_decode(ctx_cts, batch) != 0) { /* error */ }

    if (llama_decode(ctx_cts, batch) != 0) { /* error */ }
    GGML_ASSERT(batch.n_tokens == n_codes);

    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(ctx_cts, batch) != 0) {
        LOG_ERR("Failed to decode SNAC batch\n");
        return 1;
    }
    llama_synchronize(ctx_cts);
    
    LOG_INF("SNAC decode completed\n");

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
