#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// forward declaration
struct common_params;
struct ggml_threadpool;

namespace liquid {
namespace audio {

using audio_token_t = std::array<int32_t, 8>;

struct Decoder {
    explicit Decoder(const common_params & params);
    ~Decoder();

    audio_token_t      sample_audio_frame(const std::vector<float> & embedding, float temperature, int top_k);
    std::vector<float> embed(const audio_token_t & token);
    std::vector<float> embed_for_detokenizer(const audio_token_t & token);

    void set_threadpool(ggml_threadpool *, int);

  private:
    class DecoderImpl;
    std::unique_ptr<DecoderImpl> pimpl;
};
}  // namespace audio
}  // namespace liquid
