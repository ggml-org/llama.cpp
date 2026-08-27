#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace hailo_mtmd {

constexpr uint64_t NO_REQUEST_ID = 0;

class HailoVisionEncoder {
public:
    explicit HailoVisionEncoder(const std::string & hef_path);
    ~HailoVisionEncoder();

    HailoVisionEncoder(const HailoVisionEncoder &) = delete;
    HailoVisionEncoder & operator=(const HailoVisionEncoder &) = delete;

    enum class SubmitStatus { started, busy, unsupported };

    // start encoding a frame; `req_id` receives the id that later collects the result.
    // input_img must hold exactly one HEF input frame and is copied, so it need not outlive the call
    // try_submit() is speculative: never blocks, and declines rather than disturb a pending result
    // submit() is for a result needed now: may recycle a finished slot, or wait for one
    SubmitStatus try_submit(uint64_t & req_id, const uint8_t * input_img, size_t n_bytes);
    bool         submit    (uint64_t & req_id, const uint8_t * input_img, size_t n_bytes);

    // wait for the encode, transpose into output_buffer, free the slot and clear `req_id`
    // only valid for an id the encoder still owns - check owns() first, the slot may have been recycled
    bool wait(uint64_t & req_id, float * output_buffer);

    bool owns(uint64_t req_id) const;

    uint32_t input_w() const;
    uint32_t input_h() const;
    uint32_t n_image_tokens() const;
    uint32_t n_embd_per_stream() const;
    uint32_t n_streams() const;
    uint32_t n_embd_inp() const;
    uint32_t patch_size() const;
    uint32_t spatial_merge_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace hailo_mtmd
