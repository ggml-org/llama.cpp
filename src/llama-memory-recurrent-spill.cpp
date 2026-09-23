#include "llama-memory-recurrent-spill.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

static bool multiply(size_t a, size_t b, size_t & value) {
    if (b && a > std::numeric_limits<size_t>::max()/b) return false;
    value=a*b;
    return true;
}

static bool add(size_t a, size_t b, size_t & value) {
    if (b > std::numeric_limits<size_t>::max()-a) return false;
    value=a+b;
    return true;
}

static bool row_bytes(ggml_type type, uint64_t width, size_t & value) {
    if (type < 0 || type >= GGML_TYPE_COUNT || !width ||
            width > uint64_t(std::numeric_limits<int64_t>::max())) return false;
    const int block=ggml_blck_size(type);
    if (block <= 0 || width%uint64_t(block)) return false;
    return multiply(size_t(width/uint64_t(block)),ggml_type_size(type),value);
}

bool llama_recurrent_spill_layout_make(const std::vector<llama_recurrent_spill_layer> & layers,
        uint32_t cells, uint32_t depth, llama_recurrent_spill_layout & output) {
    if (layers.empty() || !cells || !depth ||
            uint64_t(cells)*(uint64_t(depth)+1) > uint64_t(std::numeric_limits<int64_t>::max())) return false;
    try {
        llama_recurrent_spill_layout next;
        next.cells=cells; next.depth=depth;
        next.layers.resize(layers.size());
        bool any=false;
        for (size_t i=0;i<layers.size();++i) {
            if (!layers[i].enabled) continue;
            any=true;
            auto & item=next.layers[i];
            item.enabled=true;
            item.type_r=layers[i].type_r; item.type_s=layers[i].type_s;
            if (!row_bytes(layers[i].type_r,layers[i].width_r,item.r_row_bytes) ||
                    !row_bytes(layers[i].type_s,layers[i].width_s,item.s_row_bytes)) return false;
            size_t one_r=0,one_s=0,one_layer=0,host_r=0,host_s=0;
            if (!multiply(item.r_row_bytes,cells,one_r) ||
                    !multiply(item.s_row_bytes,cells,one_s) ||
                    !add(one_r,one_s,one_layer) ||
                    !multiply(one_r,depth,host_r) ||
                    !multiply(one_s,depth,host_s) ||
                    !add(next.device_current_bytes,one_layer,next.device_current_bytes) ||
                    !add(next.host_snapshot_bytes,host_r,item.s_offset)) return false;
            item.r_offset=next.host_snapshot_bytes;
            if (!add(item.s_offset,host_s,next.host_snapshot_bytes)) return false;
            next.device_stage_bytes=std::max(next.device_stage_bytes,one_layer);
        }
        if (!any || !add(next.device_current_bytes,next.host_snapshot_bytes,next.stock_device_bytes)) return false;
        output=std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
}

enum class spill_transfer_state { idle, capturing, capture_pending, restoring, restore_pending };

struct llama_recurrent_spill_bank::implementation {
    llama_recurrent_spill_layout layout;
    ggml_backend_buffer_t host = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_event_t event = nullptr;
    uint64_t cache = 0, sequence = 0, last_generation = 0;
    llama_recurrent_spill_identity active;
    spill_transfer_state state = spill_transfer_state::idle;
    std::vector<uint64_t> generations;
    std::vector<uint8_t> invalidated;
    std::vector<uint8_t> touched;
    uint32_t selected_slot = 0;
    size_t submitted = 0, active_planes = 0;
    bool poisoned = false;

    ~implementation() {
        if (backend && state != spill_transfer_state::idle) ggml_backend_synchronize(backend);
        if (event) ggml_backend_event_free(event);
        if (host) ggml_backend_buffer_free(host);
    }

    size_t index(size_t layer, bool value, uint32_t slot) const {
        return (layer*layout.depth+slot-1)*2+size_t(value);
    }
    bool identity(llama_recurrent_spill_identity candidate) const {
        return candidate.cache == cache && candidate.sequence == sequence && candidate.generation != 0;
    }
    bool bind(ggml_backend_t next) {
        if (!next) return false;
        auto * next_device=ggml_backend_get_device(next);
        if (device && device != next_device) return false;
        if (!device) device=next_device;
        if (!event) event=ggml_backend_event_new(device);
        backend=next;
        return true;
    }
    bool region(size_t layer, bool value, uint32_t slot, size_t & offset, size_t & bytes) const {
        if (layer >= layout.layers.size() || !layout.layers[layer].enabled ||
                slot == 0 || slot > layout.depth) return false;
        const auto & part=layout.layers[layer];
        if (!multiply(value ? part.s_row_bytes : part.r_row_bytes,layout.cells,bytes)) return false;
        size_t displacement=0;
        if (!multiply(bytes,slot-1,displacement) ||
                !add(value ? part.s_offset : part.r_offset,displacement,offset) ||
                offset > layout.host_snapshot_bytes || bytes > layout.host_snapshot_bytes-offset) return false;
        return true;
    }
};

llama_recurrent_spill_bank::llama_recurrent_spill_bank(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
llama_recurrent_spill_bank::~llama_recurrent_spill_bank() = default;

std::unique_ptr<llama_recurrent_spill_bank> llama_recurrent_spill_bank::create(
        const llama_recurrent_spill_layout & layout, ggml_backend_buffer_type_t host_type,
        uint64_t cache, uint64_t sequence) {
    if (!cache || !host_type || !ggml_backend_buft_is_host(host_type) ||
            !layout.cells || !layout.depth || !layout.host_snapshot_bytes || layout.layers.empty() ||
            layout.layers.size() > SIZE_MAX/size_t(layout.depth)/2) return {};
    try {
        auto state=std::make_unique<implementation>();
        state->layout=layout;
        state->cache=cache; state->sequence=sequence;
        state->generations.resize(layout.layers.size()*size_t(layout.depth)*2);
        state->touched.resize(state->generations.size());
        state->invalidated.resize(state->generations.size());
        for (size_t i=0;i<layout.layers.size();++i) if (layout.layers[i].enabled) {
            ++state->active_planes;
            for (bool side : {false,true}) for (uint32_t slot=1;slot<=layout.depth;++slot) {
                size_t offset=0,bytes=0;
                if (!state->region(i,side,slot,offset,bytes)) return {};
            }
        }
        if (!state->active_planes) return {};
        state->active_planes*=2;
        state->host=ggml_backend_buft_alloc_buffer(host_type,layout.host_snapshot_bytes);
        if (!state->host || !ggml_backend_buffer_is_host(state->host) ||
                ggml_backend_buffer_get_size(state->host) < layout.host_snapshot_bytes ||
                !ggml_backend_buffer_get_base(state->host)) return {};
        ggml_backend_buffer_clear(state->host,0);
        return std::unique_ptr<llama_recurrent_spill_bank>(new llama_recurrent_spill_bank(std::move(state)));
    } catch (const std::bad_alloc &) {
        return {};
    } catch (const std::length_error &) {
        return {};
    }
}

bool llama_recurrent_spill_bank::begin_capture(llama_recurrent_spill_identity identity) {
    auto & s=*impl;
    if (s.poisoned || s.state != spill_transfer_state::idle || !s.identity(identity) ||
            identity.generation <= s.last_generation) return false;
    s.active=identity; s.state=spill_transfer_state::capturing;
    s.submitted=0; std::fill(s.touched.begin(),s.touched.end(),0);
    return true;
}

bool llama_recurrent_spill_bank::capture(ggml_backend_t backend, size_t layer, bool value, uint32_t slot,
        const ggml_tensor * source, size_t source_offset) {
    auto & s=*impl;
    size_t offset=0,bytes=0;
    if (s.state != spill_transfer_state::capturing || !s.region(layer,value,slot,offset,bytes) ||
            !source || !source->buffer || source->type !=
                (value ? s.layout.layers[layer].type_s : s.layout.layers[layer].type_r) ||
            source_offset > ggml_nbytes(source) || bytes > ggml_nbytes(source)-source_offset ||
            s.touched[s.index(layer,value,slot)] || !s.bind(backend)) return false;
    auto * host=static_cast<uint8_t *>(ggml_backend_buffer_get_base(s.host));
    ggml_backend_tensor_get_async(backend,source,host+offset,source_offset,bytes);
    s.touched[s.index(layer,value,slot)]=1;
    ++s.submitted;
    return true;
}

bool llama_recurrent_spill_bank::publish_capture() {
    auto & s=*impl;
    if (s.state != spill_transfer_state::capturing || !s.submitted || !s.backend) return false;
    if (s.event) ggml_backend_event_record(s.event,s.backend);
    s.state=spill_transfer_state::capture_pending;
    return true;
}

bool llama_recurrent_spill_bank::complete_capture() {
    auto & s=*impl;
    if (s.state != spill_transfer_state::capture_pending) return false;
    if (s.event) ggml_backend_event_synchronize(s.event);
    else ggml_backend_synchronize(s.backend);
    for (size_t i=0;i<s.touched.size();++i) if (s.touched[i]) {
        s.generations[i]=s.active.generation;
        s.invalidated[i]=0;
    }
    s.last_generation=s.active.generation;
    s.state=spill_transfer_state::idle;
    s.submitted=0;
    return true;
}

bool llama_recurrent_spill_bank::begin_restore(llama_recurrent_spill_identity identity, uint32_t slot) {
    auto & s=*impl;
    if (s.poisoned || s.state != spill_transfer_state::idle || !s.identity(identity) ||
            slot == 0 || slot > s.layout.depth) return false;
    for (size_t layer=0;layer<s.layout.layers.size();++layer) if (s.layout.layers[layer].enabled) {
        for (bool value : {false,true}) {
            const auto index=s.index(layer,value,slot);
            if (s.invalidated[index] || s.generations[index] > identity.generation) return false;
        }
    }
    s.active=identity; s.selected_slot=slot; s.submitted=0;
    std::fill(s.touched.begin(),s.touched.end(),0);
    s.state=spill_transfer_state::restoring;
    return true;
}

bool llama_recurrent_spill_bank::restore(ggml_backend_t backend, size_t layer, bool value,
        ggml_tensor * destination, size_t destination_offset) {
    auto & s=*impl;
    size_t offset=0,bytes=0;
    if (s.state != spill_transfer_state::restoring ||
            !s.region(layer,value,s.selected_slot,offset,bytes) ||
            !destination || !destination->buffer || destination->type !=
                (value ? s.layout.layers[layer].type_s : s.layout.layers[layer].type_r) ||
            destination_offset > ggml_nbytes(destination) ||
            bytes > ggml_nbytes(destination)-destination_offset ||
            s.touched[s.index(layer,value,s.selected_slot)] || !s.bind(backend)) return false;
    const auto * host=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(s.host));
    ggml_backend_tensor_set_async(backend,destination,host+offset,destination_offset,bytes);
    s.touched[s.index(layer,value,s.selected_slot)]=1;
    ++s.submitted;
    return true;
}

bool llama_recurrent_spill_bank::publish_restore() {
    auto & s=*impl;
    if (s.state != spill_transfer_state::restoring || s.submitted != s.active_planes || !s.backend) return false;
    if (s.event) ggml_backend_event_record(s.event,s.backend);
    s.state=spill_transfer_state::restore_pending;
    return true;
}

bool llama_recurrent_spill_bank::complete_restore() {
    auto & s=*impl;
    if (s.state != spill_transfer_state::restore_pending) return false;
    if (s.event) ggml_backend_event_synchronize(s.event);
    else ggml_backend_synchronize(s.backend);
    s.state=spill_transfer_state::idle;
    s.submitted=0;
    return true;
}

void llama_recurrent_spill_bank::cancel() noexcept {
    auto & s=*impl;
    if (s.state == spill_transfer_state::idle) return;
    if (s.backend) ggml_backend_synchronize(s.backend);
    if (s.state == spill_transfer_state::capturing || s.state == spill_transfer_state::capture_pending) {
        for (size_t i=0;i<s.touched.size();++i) if (s.touched[i]) {
            s.generations[i]=0;
            s.invalidated[i]=1;
        }
    } else if (s.submitted) {
        s.poisoned=true;
    }
    s.state=spill_transfer_state::idle;
    s.submitted=0;
}

void llama_recurrent_spill_bank::reset(bool pristine) noexcept {
    auto & s=*impl;
    if (s.state != spill_transfer_state::idle && s.backend) ggml_backend_synchronize(s.backend);
    s.state=spill_transfer_state::idle;
    s.submitted=0;
    s.poisoned=false;
    std::fill(s.generations.begin(),s.generations.end(),0);
    std::fill(s.invalidated.begin(),s.invalidated.end(),pristine ? 0 : 1);
    std::fill(s.touched.begin(),s.touched.end(),0);
    if (pristine) ggml_backend_buffer_clear(s.host,0);
    s.active={};
}


size_t llama_recurrent_spill_bank::host_bytes() const noexcept { return impl->layout.host_snapshot_bytes; }

uint64_t llama_recurrent_spill_bank::snapshot_generation(size_t layer, bool value, uint32_t slot) const noexcept {
    const auto & s=*impl;
    if (layer >= s.layout.layers.size() || !s.layout.layers[layer].enabled || !slot || slot > s.layout.depth) return 0;
    return s.generations[s.index(layer,value,slot)];
}
bool llama_recurrent_spill_bank::snapshot_pristine(size_t layer, bool value, uint32_t slot) const noexcept {
    const auto & s=*impl;
    if (layer >= s.layout.layers.size() || !s.layout.layers[layer].enabled || !slot || slot > s.layout.depth) return false;
    const auto index=s.index(layer,value,slot);
    return s.generations[index] == 0 && s.invalidated[index] == 0;
}


const void * llama_recurrent_spill_bank::snapshot_data(size_t layer, bool value, uint32_t slot,
        uint64_t generation) const noexcept {
    const auto & s=*impl;
    if (s.state != spill_transfer_state::idle || !generation ||
            snapshot_generation(layer,value,slot) != generation) return nullptr;
    size_t offset=0,bytes=0;
    if (!s.region(layer,value,slot,offset,bytes)) return nullptr;
    return static_cast<const uint8_t *>(ggml_backend_buffer_get_base(s.host))+offset;
}
