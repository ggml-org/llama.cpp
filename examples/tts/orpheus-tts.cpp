#include "common.h"
#include "llama.h"
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

static bool save_wav16(const std::string &fname, const std::vector<float> &data, int sample_rate) {
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

    for (const auto &sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    return file.good();
}

std::vector<llama_token> redistribute_codes(const std::vector<llama_token>& raw_codes) {
    std::vector<llama_token> snac_codes;
    for (size_t i = 0; i < raw_codes.size(); i += 7) {
        if (i + 6 >= raw_codes.size()) break;

        // Subtract 128266 base and layer-specific offsets
        snac_codes.push_back(raw_codes[i] - 128266);                  // Layer 1: offset 0
        snac_codes.push_back(raw_codes[i + 1] - 128266 - 4096);      // Layer 2: offset 4096
        snac_codes.push_back(raw_codes[i + 2] - 128266 - 8192);      // Layer 3: offset 8192
        snac_codes.push_back(raw_codes[i + 3] - 128266 - 12288);     // Layer 3: offset 12288
        snac_codes.push_back(raw_codes[i + 4] - 128266 - 16384);     // Layer 2: offset 16384
        snac_codes.push_back(raw_codes[i + 5] - 128266 - 20480);     // Layer 3: offset 20480
        snac_codes.push_back(raw_codes[i + 6] - 128266 - 24576);     // Layer 3: offset 24576
    }
    return snac_codes;
}

static void print_usage(int /*argc*/, char **argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -mv vocoder.gguf -p \"Hello world\"\n", argv[0]);
    LOG("\n");
}

static void prompt_add(std::vector<llama_token> &prompt, const llama_vocab *vocab, const std::string &txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt.insert(prompt.end(), tmp.begin(), tmp.end());
}

int main(int argc, char **argv) {
    common_params params;

    params.model = "models/orpheus-3b-0.1-ft-q4_k_m.gguf";
    params.vocoder.model = "models/snac-fwd-pass-devel.gguf";
    params.out_file = "output.wav";

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

    params.embedding = true;
    params.warmup = false; // SNAC doesn't care about BOS or EOS tokens

    common_init_result snac_init_cts = common_init_from_params(params);
    LOG_INF("SNAC model loaded: %s\n", params.model.c_str());

    model_cts = snac_init_cts.model.get();
    ctx_cts   = snac_init_cts.context.get();

    // TODO: Use real orpheus codes
    // Just some random numbers for testing
    std::vector<llama_token> orpheus_codes = {
        // Frame 1, 7 codes per frame
        128266 + 100,        // L1: 100
        128266 + 4096 + 200, // L2: 200
        128266 + 8192 + 300, // L3: 300
        128266 + 12288 + 400,// L3: 400
        128266 + 16384 + 500,// L2: 500
        128266 + 20480 + 600,// L3: 600
        128266 + 24576 + 700,// L3: 700
        // Frame 2
        128266 + 150, 128266 + 4096 + 250, 128266 + 8192 + 350, 128266 + 12288 + 450,
        128266 + 16384 + 550, 128266 + 20480 + 650, 128266 + 24576 + 750,
        // Frame 3
        128266 + 110, 128266 + 4096 + 210, 128266 + 8192 + 310, 128266 + 12288 + 410,
        128266 + 16384 + 510, 128266 + 20480 + 610, 128266 + 24576 + 710,
        // Frame 4
        128266 + 120, 128266 + 4096 + 220, 128266 + 8192 + 320, 128266 + 12288 + 420,
        128266 + 16384 + 520, 128266 + 20480 + 620, 128266 + 24576 + 720,
        // Frame 5
        128266 + 130, 128266 + 4096 + 230, 128266 + 8192 + 330, 128266 + 12288 + 430,
        128266 + 16384 + 530, 128266 + 20480 + 630, 128266 + 24576 + 730,
        // Frame 6
        128266 + 140, 128266 + 4096 + 240, 128266 + 8192 + 340, 128266 + 12288 + 440,
        128266 + 16384 + 540, 128266 + 20480 + 640, 128266 + 24576 + 740,
        // Frame 7
        128266 + 160, 128266 + 4096 + 260, 128266 + 8192 + 360, 128266 + 12288 + 460,
        128266 + 16384 + 560, 128266 + 20480 + 660, 128266 + 24576 + 760,
        // Frame 8
        128266 + 170, 128266 + 4096 + 270, 128266 + 8192 + 370, 128266 + 12288 + 470,
        128266 + 16384 + 570, 128266 + 20480 + 670, 128266 + 24576 + 770,
        // Frame 9
        128266 + 180, 128266 + 4096 + 280, 128266 + 8192 + 380, 128266 + 12288 + 480,
        128266 + 16384 + 580, 128266 + 20480 + 680, 128266 + 24576 + 780,
        // Frame 10
        128266 + 190, 128266 + 4096 + 290, 128266 + 8192 + 390, 128266 + 12288 + 490,
        128266 + 16384 + 590, 128266 + 20480 + 690, 128266 + 24576 + 790
    };

    std::vector<llama_token> snac_codes = redistribute_codes(orpheus_codes);

    const int batch_size = snac_codes.size();

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    for (size_t i = 0; i < batch_size; ++i) {
        common_batch_add(batch, snac_codes[i], i, {0}, true);
    }

    LOG_INF("Batch before decode: n_tokens = %d\n", batch.n_tokens);
    GGML_ASSERT(batch.n_tokens == batch_size);

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx_cts, batch) != 0) {
        LOG_ERR("Failed to decode SNAC batch\n");
        return 1;
    }
    LOG_INF("SNAC decode completed\n");
    llama_synchronize(ctx_cts);

    float* embd = llama_get_embeddings(ctx_cts);
    if (!embd) {
        LOG_ERR("No embeddings available\n");
        return 1;
    }

    int n_samples = llama_get_n_outputs(ctx_cts);
    std::vector<float> audio(n_samples);
    LOG_INF("n_samples: %i\n", n_samples);
    memcpy(audio.data(), embd, n_samples * sizeof(float));

    save_wav16(params.out_file, audio, 24000);

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
