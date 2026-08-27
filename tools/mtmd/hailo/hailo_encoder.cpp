#include "hailo_encoder.hpp"
#include "clip-impl.h"

#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/buffer.hpp>
#include <hailo/expected.hpp>
#include <hailo/hef.hpp>
#include <hailo/hailort.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define HAILO_CONFIG_RESOURCE        "hailo-config.json"
#define HAILO_OUTPUT_LAYERS_KEY      "encoder_output_layers_names_suffixes"
#define HAILO_PATCH_SIZE_KEY         "patch_size"
#define HAILO_SPATIAL_MERGE_SIZE_KEY "spatial_merge_size"
// inner-dict keys under HAILO_OUTPUT_LAYERS_KEY
#define HAILO_LAYER_IMAGE_EMBEDDINGS "image_embeddings"
#define HAILO_LAYER_DEEPSTACK_PREFIX "deepstack_layer_"

namespace hailo_mtmd {

static constexpr auto HAILO_ENCODE_TIMEOUT = std::chrono::seconds(10);
// one slot per prefetch plus one kept clear for a needed-now encode. fewer slots lowers the
// prefetch depth too, and at depth 1 nothing overlaps
static constexpr uint32_t HAILO_SLOTS = 3;
static constexpr uint32_t HAILO_PREFETCH_DEPTH = 2;

// a queued job only starts once those ahead of it finish, so a wait covers the whole queue
static constexpr auto HAILO_WAIT_TIMEOUT = HAILO_ENCODE_TIMEOUT * (HAILO_PREFETCH_DEPTH + 1);

static std::atomic<uint64_t> g_next_request_id{NO_REQUEST_ID + 1};

// Async inference callback result
struct SlotSync {
    std::atomic<bool> done{false};
    hailo_status      status = HAILO_SUCCESS;   // only meaningful once done is set
};

struct EncodeSlot {
    hailort::ConfiguredInferModel::Bindings bindings;
    hailort::Buffer                         input;
    std::vector<hailort::Buffer>            outputs;   // by stream slot [main, ds_1 .. ds_3]
    hailort::AsyncInferJob                  job;

    uint64_t                   id = NO_REQUEST_ID;
    std::shared_ptr<SlotSync>  sync;

    // only a device that stops signalling entirely gets here; ordinary errors complete with a bad
    // status instead. the device may still write these buffers, so never reused
    bool stuck = false;

    bool occupied() const { return stuck || id != NO_REQUEST_ID; }
    bool running()  const { return !stuck && id != NO_REQUEST_ID && !sync->done.load(std::memory_order_acquire); }
    bool finished() const { return !stuck && id != NO_REQUEST_ID &&  sync->done.load(std::memory_order_acquire); }
};

struct HailoVisionEncoder::Impl {
    // declared first so it is destroyed last: the device has to be gone before these DMA buffers are
    // freed, or a transfer still in flight would land in memory the allocator has handed on
    std::vector<EncodeSlot> slots;

    std::unique_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::InferModel> infer_model;
    hailort::ConfiguredInferModel configured;

    // only for terminal failure: run_async shuts the model down internally, leaving nothing to retry against
    bool failed_already = false;

    uint32_t input_w = 0;
    uint32_t input_h = 0;
    size_t input_frame_size = 0;
    uint32_t n_image_tokens = 0;
    uint32_t n_embd_per_stream = 0;
    uint32_t n_streams = 0;
    uint32_t patch_size = 0;
    uint32_t spatial_merge_size = 0;

    uint32_t prefetch_depth = 0;
    size_t   n_stuck = 0;

    bool model_configured = false;

    Impl() = default;

    // submitted and not yet complete, i.e. how much the device already has to chew on
    uint32_t n_running() const {
        uint32_t n = 0;
        for (const auto & s : slots) {
            n += s.running() ? 1 : 0;
        }
        return n;
    }

    EncodeSlot * pick_free() {
        for (auto & s : slots) {
            if (!s.occupied()) {
                return &s;
            }
        }
        return nullptr;
    }

    EncodeSlot * pick_oldest(bool (EncodeSlot::*state)() const) {
        EncodeSlot * oldest = nullptr;
        for (auto & s : slots) {
            // ids are handed out in submission order
            if ((s.*state)() && (oldest == nullptr || s.id < oldest->id)) {
                oldest = &s;
            }
        }
        return oldest;
    }

    EncodeSlot * pick_newest(bool (EncodeSlot::*state)() const) {
        EncodeSlot * newest = nullptr;
        for (auto & s : slots) {
            if ((s.*state)() && (newest == nullptr || s.id > newest->id)) {
                newest = &s;
            }
        }
        return newest;
    }

    void mark_stuck(EncodeSlot & slot) {
        slot.stuck = true;
        slot.id    = NO_REQUEST_ID;
        if (++n_stuck == slots.size()) {
            LOG_ERR("HailoVisionEncoder: all %zu encode slots are stuck; no further encodes are possible\n",
                    slots.size());
        }
    }

    EncodeSlot * find_by_request_id(uint64_t id) {
        if (id == NO_REQUEST_ID) {
            return nullptr;
        }
        for (auto & s : slots) {
            if (s.id == id) {
                return &s;
            }
        }
        return nullptr;
    }

    // `slot` already hold its frame
    bool start_encode(EncodeSlot & slot, std::chrono::milliseconds ready_timeout) {
        if (configured.wait_for_async_ready(ready_timeout, 1) != HAILO_SUCCESS) {
            return false;
        }
        slot.sync->done.store(false, std::memory_order_relaxed);
        slot.id = g_next_request_id.fetch_add(1, std::memory_order_relaxed);
        auto job = configured.run_async(slot.bindings,
            [sync = slot.sync](const hailort::AsyncInferCompletionInfo & info) {
                sync->status = info.status;
                sync->done.store(true, std::memory_order_release);
            });
        if (!job) {
            slot.id = NO_REQUEST_ID;
            failed_already = true;
            LOG_ERR("HailoVisionEncoder: run_async failed; encoder is now unusable\n");
            return false;
        }
        slot.job = job.release();
        return true;
    }

    ~Impl() {
        // detach() is non-blocking; shutdown() below is what stops the device touching our buffers
        for (auto & s : slots) {
            if (s.occupied()) {
                s.job.detach();
            }
        }
        if (!model_configured) {
            return;
        }
        const auto status = configured.shutdown();
        if (status != HAILO_SUCCESS) {
            // the slots outlive every member below, so the buffers survive until the device is gone
            LOG_ERR("HailoVisionEncoder: shutdown failed (%d)\n", static_cast<int>(status));
        }
    }
};

// strips the "<network_name>/" prefix from a HEF output stream name and looks
// up its slot in the config-derived map. throws if the suffix isn't present.
static int slot_for_output_name(const std::string & name,
                                const std::unordered_map<std::string, int> & suffix_to_slot)
{
    const auto slash = name.find('/');
    const std::string suffix = (slash == std::string::npos) ? name : name.substr(slash + 1);
    auto found = suffix_to_slot.find(suffix);
    if (found == suffix_to_slot.end()) {
        throw std::runtime_error("HailoVisionEncoder: HEF output stream '" + name
            + "' has no entry in " HAILO_CONFIG_RESOURCE);
    }
    return found->second;
}

static void validate_output_shapes(const std::shared_ptr<hailort::InferModel> & model)
{
    const auto names = model->get_output_names();
    const auto first_shape = model->output(names[0]).expect("HailoVisionEncoder: output() failed").shape();
    for (const auto & name : names) {
        const auto shape = model->output(name).expect("HailoVisionEncoder: output() failed").shape();
        if (shape.height   != first_shape.height
         || shape.width    != first_shape.width
         || shape.features != first_shape.features) {
            throw std::runtime_error("HailoVisionEncoder: output shape mismatch at '" + name + "'");
        }
    }
}

// transposes per-stream output buffers into the token-major [main | ds0 | ds1 | ...]
// layout the decoder expects. `streams` is indexed by slot (assignment happens at init).
static void transpose_to_token_major(const std::vector<hailort::Buffer> & streams,
                                     float * output_buffer,
                                     uint32_t n_tokens,
                                     uint32_t n_embd_per_stream)
{
    const uint32_t n_embd_inp = n_embd_per_stream * static_cast<uint32_t>(streams.size());
    for (size_t slot = 0; slot < streams.size(); ++slot) {
        const float * src = reinterpret_cast<const float *>(streams[slot].data());   // [n_tokens * n_embd]
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float * dst = output_buffer + t * n_embd_inp + slot * n_embd_per_stream;
            std::memcpy(dst, src + t * n_embd_per_stream, n_embd_per_stream * sizeof(float));
        }
    }
}

static hailort::Buffer load_hef_to_buffer(const std::string & hef_path)
{
    std::ifstream hef_file(hef_path, std::ios::binary | std::ios::ate);
    if (!hef_file) {
        throw std::runtime_error("HailoVisionEncoder: cannot open HEF '" + hef_path + "'");
    }
    const std::streamsize hef_size = hef_file.tellg();
    hef_file.seekg(0, std::ios::beg);
    auto blob = hailort::Buffer::create(static_cast<size_t>(hef_size))
        .expect("HailoVisionEncoder: hef Buffer::create failed");
    if (!hef_file.read(reinterpret_cast<char *>(blob.data()), hef_size)) {
        throw std::runtime_error("HailoVisionEncoder: failed to read HEF '" + hef_path + "'");
    }
    return blob;
}

// reads the hailo-config.json blob embedded in the HEF and parses it.
static nlohmann::json read_hailo_config(const std::shared_ptr<hailort::InferModel> & model)
{
    auto view = model->hef().get_external_resources(HAILO_CONFIG_RESOURCE)
        .expect("HailoVisionEncoder: HEF is missing required external resource '" HAILO_CONFIG_RESOURCE "'");
    try {
        return nlohmann::json::parse(reinterpret_cast<const char *>(view.data()),
                                     reinterpret_cast<const char *>(view.data()) + view.size());
    } catch (const std::exception & e) {
        throw std::runtime_error(
            std::string("HailoVisionEncoder: failed to parse ") + HAILO_CONFIG_RESOURCE + ": " + e.what());
    }
}

HailoVisionEncoder::HailoVisionEncoder(const std::string & hef_path)
    : pimpl(new Impl())
{
    pimpl->vdevice = hailort::VDevice::create().expect("HailoVisionEncoder: VDevice::create failed");

    // reads the HEF file into a buffer in order to read the external resources
    const auto hef_blob = load_hef_to_buffer(hef_path);
    pimpl->infer_model = pimpl->vdevice->create_infer_model(
        hailort::MemoryView::create_const(hef_blob.data(), hef_blob.size())
    ).expect("HailoVisionEncoder: create_infer_model failed for '" + hef_path + "'");

    const auto output_names = pimpl->infer_model->get_output_names();
    if (output_names.size() != 4) {
        throw std::runtime_error("HailoVisionEncoder: HEF must have exactly 4 outputs, but got " + std::to_string(output_names.size()));
    }

    auto input = pimpl->infer_model->input().expect("HailoVisionEncoder: input() failed");
    const auto in_shape = input.shape();
    pimpl->input_h = in_shape.height;
    pimpl->input_w = in_shape.width;
    pimpl->input_frame_size = input.get_frame_size();

    validate_output_shapes(pimpl->infer_model);
    const auto output_shape = pimpl->infer_model->output(output_names[0]).expect("HailoVisionEncoder: output(" + output_names[0] + ") failed").shape();
    pimpl->n_image_tokens = output_shape.height * output_shape.width;
    pimpl->n_embd_per_stream = output_shape.features;
    pimpl->n_streams = static_cast<uint32_t>(output_names.size());

    const auto hailo_config = read_hailo_config(pimpl->infer_model);
    pimpl->patch_size = hailo_config.at(HAILO_PATCH_SIZE_KEY).get<uint32_t>();
    pimpl->spatial_merge_size = hailo_config.at(HAILO_SPATIAL_MERGE_SIZE_KEY).get<uint32_t>();

    // slot order is fixed by the decoder: [image_embeddings, deepstack_1, deepstack_2, deepstack_3].
    const auto & layers = hailo_config.at(HAILO_OUTPUT_LAYERS_KEY);
    const std::pair<const char *, int> layer_slots[] = {
        { HAILO_LAYER_IMAGE_EMBEDDINGS,     0 },
        { HAILO_LAYER_DEEPSTACK_PREFIX "1", 1 },
        { HAILO_LAYER_DEEPSTACK_PREFIX "2", 2 },
        { HAILO_LAYER_DEEPSTACK_PREFIX "3", 3 },
    };
    std::unordered_map<std::string, int> suffix_to_slot;
    for (const auto & [key, slot] : layer_slots) {
        const auto suffix = layers.at(key).get<std::string>();
        // a repeat would collapse two layers onto one slot, leaving slot.outputs short of its indices
        if (!suffix_to_slot.emplace(suffix, slot).second) {
            throw std::runtime_error(std::string("HailoVisionEncoder: ") + key + " suffix '" + suffix
                + "' is already used by another layer in " HAILO_CONFIG_RESOURCE);
        }
    }

    for (const auto & name : output_names) {
        pimpl->infer_model->output(name).expect("HailoVisionEncoder: output(" + name + ") failed")
            .set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }

    pimpl->configured = pimpl->infer_model->configure().expect("HailoVisionEncoder: configure failed");
    pimpl->model_configured = true;

    // the device queue is the hard ceiling: run_async past it shuts the model down for good
    const auto queue_size = pimpl->configured.get_async_queue_size();
    const uint32_t n_slots = queue_size
        ? std::max<uint32_t>(1, std::min<uint32_t>(HAILO_SLOTS, static_cast<uint32_t>(queue_size.value())))
        : 1;
    pimpl->prefetch_depth = std::min(HAILO_PREFETCH_DEPTH, n_slots - 1);
    LOG_INF("HailoVisionEncoder: %u encode slot(s), prefetch depth %u, device async queue %s\n",
            n_slots, pimpl->prefetch_depth,
            queue_size ? std::to_string(queue_size.value()).c_str() : "unknown");

    // bound once, so no buffer is ever rebound while the device may be reading it
    pimpl->slots.resize(n_slots);
    for (auto & slot : pimpl->slots) {
        slot.sync = std::make_shared<SlotSync>();
        slot.bindings = pimpl->configured.create_bindings().expect("HailoVisionEncoder: create_bindings failed");

        slot.input = hailort::Buffer::create(pimpl->input_frame_size, hailort::BufferStorageParams::create_dma())
            .expect("HailoVisionEncoder: input Buffer::create failed");
        auto input_bind = slot.bindings.input().expect("HailoVisionEncoder: bindings.input failed");
        const auto in_st = input_bind.set_buffer(hailort::MemoryView(slot.input.data(), slot.input.size()));
        if (HAILO_SUCCESS != in_st) {
            throw std::runtime_error("HailoVisionEncoder: input set_buffer failed (" + std::to_string(in_st) + ")");
        }

        // per-slot outputs
        slot.outputs.resize(suffix_to_slot.size());
        for (const auto & output : pimpl->infer_model->outputs()) {
            const std::string name = output.name();
            const int out_slot = slot_for_output_name(name, suffix_to_slot);

            auto buffer = hailort::Buffer::create(output.get_frame_size(), hailort::BufferStorageParams::create_dma()).expect("HailoVisionEncoder: Buffer::create failed");
            auto output_bind = slot.bindings.output(name).expect("HailoVisionEncoder: bindings.output failed");
            auto out_st = output_bind.set_buffer(hailort::MemoryView(buffer.data(), buffer.size()));
            if (HAILO_SUCCESS != out_st) {
                throw std::runtime_error("HailoVisionEncoder: output set_buffer failed (" + std::to_string(out_st) + ")");
            }
            slot.outputs[out_slot] = std::move(buffer);
        }
    }
}

HailoVisionEncoder::SubmitStatus HailoVisionEncoder::try_submit(uint64_t & req_id, const uint8_t * input_img,
                                                                size_t n_bytes)
{
    req_id = NO_REQUEST_ID;
    // checked before failed_already so a bad frame is reported as such even on a dead encoder
    if (input_img == nullptr || n_bytes != pimpl->input_frame_size) {
        return SubmitStatus::unsupported;
    }
    if (pimpl->failed_already) {
        return SubmitStatus::busy;
    }
    if (pimpl->n_running() >= pimpl->prefetch_depth) {
        return SubmitStatus::busy;   // more depth than a one-frame-at-a-time device can use
    }

    // free slots only - recycling a finished result would discard work its owner is about to collect
    EncodeSlot * slot = pimpl->pick_free();
    if (slot == nullptr) {
        return SubmitStatus::busy;
    }

    // the frame must outlive this call and the chunk it came from need not - a released server slot frees
    // it while the device is still reading, so the encode gets its own slot-owned copy
    std::memcpy(slot->input.data(), input_img, pimpl->input_frame_size);
    // zero timeout keeps this speculative call non-blocking; a busy pipeline just declines
    if (!pimpl->start_encode(*slot, std::chrono::milliseconds(0))) {
        return SubmitStatus::busy;
    }
    req_id = slot->id;
    return SubmitStatus::started;
}

bool HailoVisionEncoder::submit(uint64_t & req_id, const uint8_t * input_img, size_t n_bytes)
{
    req_id = NO_REQUEST_ID;
    if (pimpl->failed_already || input_img == nullptr || n_bytes != pimpl->input_frame_size) {
        return false;
    }

    EncodeSlot * slot = pimpl->pick_free();
    if (slot == nullptr) {
        // taking over a slot costs its owner a re-encode either way, so pick the cheapest: a finished one
        // needs no wait, newest first since it is wanted last. else wait on the job nearest the queue head
        slot = pimpl->pick_newest(&EncodeSlot::finished);
        if (slot == nullptr) {
            slot = pimpl->pick_oldest(&EncodeSlot::running);
            if (slot == nullptr) {
                return false;   // every slot is stuck
            }
            if (slot->job.wait(HAILO_WAIT_TIMEOUT) != HAILO_SUCCESS) {
                LOG_ERR("HailoVisionEncoder: encode slot did not free up\n");
                pimpl->mark_stuck(*slot);
                return false;
            }
        }
    }

    // the frame must outlive this call and the chunk it came from need not - a released server slot frees
    // it while the device is still reading, so the encode gets its own slot-owned copy
    std::memcpy(slot->input.data(), input_img, pimpl->input_frame_size);
    if (!pimpl->start_encode(*slot, HAILO_ENCODE_TIMEOUT)) {
        LOG_ERR("HailoVisionEncoder: pipeline did not free up for a new encode\n");
        return false;
    }
    req_id = slot->id;
    return true;
}

bool HailoVisionEncoder::wait(uint64_t & req_id, float * output_buffer)
{
    if (output_buffer == nullptr) {
        LOG_ERR("HailoVisionEncoder: wait() called with no output buffer\n");
        return false;   // the request is untouched and still collectable
    }
    EncodeSlot * slot = pimpl->find_by_request_id(req_id);
    if (slot == nullptr) {
        req_id = NO_REQUEST_ID;   // no slot honours this id, so its owner has to submit again
        return false;
    }
    if (slot->job.wait(HAILO_WAIT_TIMEOUT) != HAILO_SUCCESS) {
        // poison the slot, not the encoder: killing it here would turn a retry into endless failures
        LOG_ERR("HailoVisionEncoder: encode did not complete within %lld s; slot taken out of service\n",
                static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::seconds>(HAILO_WAIT_TIMEOUT).count()));
        pimpl->mark_stuck(*slot);
        req_id = NO_REQUEST_ID;
        return false;
    }
    const auto status = slot->sync->status;
    slot->id = NO_REQUEST_ID;
    req_id = NO_REQUEST_ID;
    if (status != HAILO_SUCCESS) {
        LOG_ERR("HailoVisionEncoder: encode ended with status %d\n", static_cast<int>(status));
        return false;
    }
    transpose_to_token_major(slot->outputs, output_buffer,
                             pimpl->n_image_tokens, pimpl->n_embd_per_stream);
    return true;
}

bool HailoVisionEncoder::owns(uint64_t req_id) const
{
    return pimpl->find_by_request_id(req_id) != nullptr;
}


uint32_t HailoVisionEncoder::input_w() const { return pimpl->input_w; }
uint32_t HailoVisionEncoder::input_h() const { return pimpl->input_h; }
uint32_t HailoVisionEncoder::n_image_tokens() const { return pimpl->n_image_tokens; }
uint32_t HailoVisionEncoder::n_embd_per_stream() const { return pimpl->n_embd_per_stream; }
uint32_t HailoVisionEncoder::n_streams() const { return pimpl->n_streams; }
uint32_t HailoVisionEncoder::n_embd_inp() const { return pimpl->n_embd_per_stream * pimpl->n_streams; }
uint32_t HailoVisionEncoder::patch_size() const { return pimpl->patch_size; }
uint32_t HailoVisionEncoder::spatial_merge_size() const { return pimpl->spatial_merge_size; }
HailoVisionEncoder::~HailoVisionEncoder() = default;

} // namespace hailo_mtmd
