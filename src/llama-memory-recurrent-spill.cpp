#include "llama-memory-recurrent-spill.h"
#include "../ggml/src/ggml-backend-execution.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

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
            item.width_r=layers[i].width_r; item.width_s=layers[i].width_s;
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

static bool stage_align(size_t value, size_t alignment, size_t & output) {
    if (!alignment || (alignment & (alignment-1)) ||
            value > std::numeric_limits<size_t>::max()-(alignment-1)) return false;
    output=(value+alignment-1)&~(alignment-1);
    return true;
}

bool llama_recurrent_stage_layout_make(const llama_recurrent_spill_layout & snapshots,
        uint32_t slots, size_t alignment, llama_recurrent_stage_layout & output) {
    if (!slots || !snapshots.cells || !snapshots.depth || snapshots.layers.empty() ||
            !alignment || (alignment & (alignment-1))) return false;
    try {
        llama_recurrent_stage_layout next;
        next.slots=slots;
        next.depth=snapshots.depth;
        next.alignment=alignment;
        next.layers.resize(snapshots.layers.size());
        bool any=false;
        for (size_t layer=0;layer<snapshots.layers.size();++layer) {
            const auto & source=snapshots.layers[layer];
            if (!source.enabled) continue;
            any=true;
            if (!source.r_row_bytes || !source.s_row_bytes) return false;
            auto & item=next.layers[layer];
            item.enabled=true;
            size_t r_total=0,s_total=0,used=0,rounded=0;
            if (!multiply(source.r_row_bytes,snapshots.cells,item.r_snapshot_bytes) ||
                    !multiply(source.s_row_bytes,snapshots.cells,item.s_snapshot_bytes) ||
                    !stage_align(item.r_snapshot_bytes,alignment,item.r_stride) ||
                    !stage_align(item.s_snapshot_bytes,alignment,item.s_stride) ||
                    !multiply(item.r_stride,snapshots.depth,r_total) ||
                    !multiply(item.s_stride,snapshots.depth,s_total) ||
                    !stage_align(r_total,alignment,item.s_offset) ||
                    !add(item.s_offset,s_total,used) ||
                    !stage_align(used,alignment,rounded)) return false;
            next.bytes_per_slot=std::max(next.bytes_per_slot,rounded);
        }
        if (!any || !multiply(next.bytes_per_slot,slots,next.total_bytes)) return false;
        output=std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
}

bool llama_recurrent_stage_region(const llama_recurrent_stage_layout & layout,
        size_t layer, bool value, uint32_t snapshot, uint32_t slot, size_t & offset, size_t & bytes) {
    if (layer >= layout.layers.size() || !layout.layers[layer].enabled ||
            !snapshot || snapshot > layout.depth || slot >= layout.slots) return false;
    const auto & item=layout.layers[layer];
    const size_t plane_bytes=value ? item.s_snapshot_bytes : item.r_snapshot_bytes;
    size_t base=0,snapshot_offset=0,local=0,local_end=0,next=0,end=0;
    if (!multiply(layout.bytes_per_slot,slot,base) ||
            !multiply(value ? item.s_stride : item.r_stride,snapshot-1,snapshot_offset) ||
            !add(value ? item.s_offset : 0,snapshot_offset,local) ||
            !add(local,plane_bytes,local_end) ||
            local_end > layout.bytes_per_slot ||
            !add(base,local,next) ||
            !add(next,plane_bytes,end) || end > layout.total_bytes) return false;
    offset=next;
    bytes=plane_bytes;
    return true;
}

struct llama_recurrent_stage_buffer::implementation {
    llama_recurrent_stage_layout planned;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t backing = nullptr;
    std::vector<ggml_tensor *> tensors;

    ~implementation() {
        if (backing) ggml_backend_buffer_free(backing);
        if (ctx) ggml_free(ctx);
    }

    size_t index(size_t layer, bool value, uint32_t snapshot, uint32_t slot) const {
        return (((size_t(slot)*planned.layers.size()+layer)*planned.depth+snapshot-1)*2)+size_t(value);
    }
};

llama_recurrent_stage_buffer::llama_recurrent_stage_buffer(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
llama_recurrent_stage_buffer::~llama_recurrent_stage_buffer() = default;

std::unique_ptr<llama_recurrent_stage_buffer> llama_recurrent_stage_buffer::create(
        const llama_recurrent_spill_layout & snapshots, uint32_t slots,
        ggml_backend_buffer_type_t device_type) {
    if (!device_type) return {};
    try {
        auto state=std::make_unique<implementation>();
        if (!llama_recurrent_stage_layout_make(
                snapshots,slots,ggml_backend_buft_get_alignment(device_type),state->planned)) return {};
        size_t count=0,metadata_bytes=0;
        if (!multiply(state->planned.layers.size(),state->planned.depth,count) ||
                !multiply(count,slots,count) ||
                !multiply(count,2,count) ||
                !multiply(count,ggml_tensor_overhead(),metadata_bytes) ||
                !add(metadata_bytes,4096,metadata_bytes)) return {};
        state->ctx=ggml_init({metadata_bytes,nullptr,true});
        if (!state->ctx) return {};
        state->tensors.resize(count,nullptr);
        for (uint32_t slot=0;slot<slots;++slot) {
            for (size_t layer=0;layer<snapshots.layers.size();++layer) {
                if (!snapshots.layers[layer].enabled) continue;
                const auto & item=snapshots.layers[layer];
                for (uint32_t snapshot=1;snapshot<=snapshots.depth;++snapshot) {
                    for (bool value : {false,true}) {
                        const auto type=value ? item.type_s : item.type_r;
                        const auto width=value ? item.width_s : item.width_r;
                        auto * tensor=ggml_new_tensor_2d(state->ctx,type,int64_t(width),snapshots.cells);
                        if (!tensor) return {};
                        const auto & stage_layer=state->planned.layers[layer];
                        if (ggml_backend_buft_get_alloc_size(device_type,tensor) >
                                (value ? stage_layer.s_stride : stage_layer.r_stride)) return {};
                        state->tensors[state->index(layer,value,snapshot,slot)]=tensor;
                    }
                }
            }
        }
        state->backing=ggml_backend_buft_alloc_buffer(device_type,state->planned.total_bytes);
        if (!state->backing || ggml_backend_buffer_get_size(state->backing) < state->planned.total_bytes ||
                !ggml_backend_buffer_get_base(state->backing)) return {};
        auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(state->backing));
        for (uint32_t slot=0;slot<slots;++slot) {
            for (size_t layer=0;layer<snapshots.layers.size();++layer) {
                if (!snapshots.layers[layer].enabled) continue;
                for (uint32_t snapshot=1;snapshot<=snapshots.depth;++snapshot) {
                    for (bool value : {false,true}) {
                        size_t offset=0,bytes=0;
                        auto * tensor=state->tensors[state->index(layer,value,snapshot,slot)];
                        if (!llama_recurrent_stage_region(state->planned,layer,value,snapshot,slot,offset,bytes) ||
                                ggml_nbytes(tensor) != bytes ||
                                ggml_backend_tensor_alloc(state->backing,tensor,base+offset) != GGML_STATUS_SUCCESS) return {};
                    }
                }
            }
        }
        return std::unique_ptr<llama_recurrent_stage_buffer>(new llama_recurrent_stage_buffer(std::move(state)));
    } catch (const std::bad_alloc &) {
        return {};
    } catch (const std::length_error &) {
        return {};
    }
}

ggml_tensor * llama_recurrent_stage_buffer::tensor(
        size_t layer, bool value, uint32_t snapshot, uint32_t slot) const noexcept {
    const auto & s=*impl;
    if (layer >= s.planned.layers.size() || !s.planned.layers[layer].enabled ||
            !snapshot || snapshot > s.planned.depth || slot >= s.planned.slots) return nullptr;
    return s.tensors[s.index(layer,value,snapshot,slot)];
}
ggml_backend_buffer_t llama_recurrent_stage_buffer::buffer() const noexcept { return impl->backing; }
size_t llama_recurrent_stage_buffer::bytes() const noexcept { return impl->planned.total_bytes; }
const llama_recurrent_stage_layout & llama_recurrent_stage_buffer::layout() const noexcept { return impl->planned; }

enum class spill_transfer_state { idle, capturing, capture_pending, restoring, restore_pending };

struct spill_publication_state {
    std::unique_ptr<llama_recurrent_stage_buffer> stage;
    ggml_backend_t copy_backend = nullptr;
    ggml_backend_buffer_t execution = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_event_t stage_event = nullptr;
    std::vector<ggml_backend_event_t> slot_events;
    std::vector<ggml_backend_event_t> ready_events;
    std::vector<uint8_t> slot_busy;
    std::vector<ggml_tensor *> destinations;
    ggml_backend_t producer = nullptr;
    uint32_t expected_s = 0;
    uint32_t expected_per_layer = 0;
    uint32_t remaining = 0;
    uint32_t next_slot = 0;
    uint32_t current_slot = 0;
    size_t current_layer = SIZE_MAX;
    std::vector<uint32_t> layer_slots;
    size_t completed_layers = 0;
    bool active = false;

    ~spill_publication_state() {
        if (copy_backend) ggml_backend_synchronize(copy_backend);
        if (execution) ggml_backend_buffer_free(execution);
        if (ctx) ggml_free(ctx);
        if (stage_event) ggml_backend_event_free(stage_event);
        for (auto * event : slot_events) if (event) ggml_backend_event_free(event);
        stage.reset();
        for (auto * event : ready_events) if (event) ggml_backend_event_free(event);
        if (copy_backend) ggml_backend_free(copy_backend);
    }

    size_t index(size_t layer, bool value, uint32_t snapshot, uint32_t depth) const {
        return (layer*depth+snapshot-1)*2+size_t(value);
    }
};

struct llama_recurrent_spill_bank::implementation {
    std::unique_ptr<spill_publication_state> publication;
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
        if (state != spill_transfer_state::idle && publication && publication->producer)
            ggml_backend_synchronize(publication->producer);
        if (backend && state != spill_transfer_state::idle) ggml_backend_synchronize(backend);
        if (event) ggml_backend_event_free(event);
        publication.reset();
        if (host) ggml_backend_buffer_free(host);
    }

    size_t index(size_t layer, bool value, uint32_t slot) const {
        return (layer*layout.depth+slot-1)*2+size_t(value);
    }
    bool publish_node(ggml_backend_t producer, ggml_tensor * op);
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

bool llama_recurrent_spill_bank::implementation::publish_node(
        ggml_backend_t producer, ggml_tensor * op) {
    if (!publication || !publication->active || state != spill_transfer_state::capturing ||
            !producer || !op || op->op != GGML_OP_CPY || !op->src[0] || !op->src[1]) return false;
    auto & p=*publication;
    const auto found=std::find(p.destinations.begin(),p.destinations.end(),op->src[1]);
    if (found == p.destinations.end()) return false;
    const size_t index_value=size_t(found-p.destinations.begin());
    const size_t layer=index_value/(size_t(layout.depth)*2);
    const size_t part=index_value%(size_t(layout.depth)*2);
    const uint32_t snapshot=uint32_t(part/2)+1;
    const bool value=(part%2) != 0;
    if (layer >= layout.layers.size() || !layout.layers[layer].enabled ||
            (value && snapshot > p.expected_s) ||
            touched[index(layer,value,snapshot)] ||
            (p.producer && p.producer != producer) ||
            ggml_backend_get_device(producer) != ggml_backend_get_device(p.copy_backend)) return false;
    auto * stage_tensor=p.stage->tensor(layer,value,snapshot,p.layer_slots[layer]);
    if (!stage_tensor || (op->src[0] != stage_tensor && op->src[0]->view_src != stage_tensor) ||
            ggml_nbytes(op->src[0]) != ggml_nbytes(stage_tensor)) return false;
    if (p.current_layer != layer) {
        if (p.remaining || (p.current_layer != SIZE_MAX && layer <= p.current_layer)) return false;
        p.current_slot=p.layer_slots[layer];
        if (p.slot_busy[p.current_slot]) {
            auto * event=p.slot_events[p.current_slot];
            if (event) ggml_backend_event_synchronize(event);
            else ggml_backend_synchronize(p.copy_backend);
            p.slot_busy[p.current_slot]=0;
        }
        p.current_layer=layer;
        p.remaining=p.expected_per_layer;
        ++p.next_slot;
        p.producer=producer;
    }
    size_t host_offset=0,bytes=0;
    if (!region(layer,value,snapshot,host_offset,bytes) || !bind(p.copy_backend)) return false;
    const size_t ready_index=(size_t(p.current_slot)*layout.depth+snapshot-1)*2+size_t(value);
    auto * ready=p.ready_events[ready_index];
    if (ready) {
        ggml_backend_event_record(ready,producer);
        ggml_backend_event_wait(p.copy_backend,ready);
    } else {
        ggml_backend_synchronize(producer);
    }
    auto * host_base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(host));
    ggml_backend_tensor_get_async(p.copy_backend,op->src[0],host_base+host_offset,0,bytes);
    touched[index(layer,value,snapshot)]=1;
    ++submitted;
    if (--p.remaining == 0) {
        auto * consumed=p.slot_events[p.current_slot];
        if (consumed) ggml_backend_event_record(consumed,p.copy_backend);
        p.slot_busy[p.current_slot]=1;
        ++p.completed_layers;
        if (p.completed_layers < active_planes/2) {
            const uint32_t next=p.next_slot%p.stage->layout().slots;
            if (p.slot_busy[next]) {
                auto * event=p.slot_events[next];
                if (event) ggml_backend_event_synchronize(event);
                else ggml_backend_synchronize(p.copy_backend);
                p.slot_busy[next]=0;
            }
        }
    }
    return true;
}


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

bool llama_recurrent_spill_bank::enable_publication(
        ggml_backend_dev_t device, ggml_backend_buffer_type_t stage_type, uint32_t slots) {
    auto & s=*impl;
    if (s.publication || s.state != spill_transfer_state::idle || !device || !stage_type ||
            !ggml_backend_dev_supports_buft(device,stage_type)) return false;
    try {
        auto next=std::make_unique<spill_publication_state>();
        next->stage=llama_recurrent_stage_buffer::create(s.layout,slots,stage_type);
        if (!next->stage) return false;
        next->copy_backend=ggml_backend_dev_init(device,nullptr);
        if (!next->copy_backend) return false;
        const ggml_backend_execution_ops ops{
            [](void * context,const ggml_tensor * op) {
                auto & state=*static_cast<implementation *>(context);
                const auto * pub=state.publication.get();
                return pub && op && op->op == GGML_OP_CPY && op->src[0] && op->src[1] &&
                    std::find(pub->destinations.begin(),pub->destinations.end(),op->src[1]) !=
                        pub->destinations.end();
            },
            [](void * context,ggml_backend_t backend,ggml_tensor * op) {
                return static_cast<implementation *>(context)->publish_node(backend,op)
                    ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
            },
            [](void *) { return false; },
            [](void *) {},
            [](void *) {},
        };
        next->execution=ggml_backend_execution_buffer_new(device,s.host,ops,&s);
        if (!next->execution) return false;
        size_t count=0,metadata_bytes=0;
        if (!multiply(s.layout.layers.size(),s.layout.depth,count) ||
                !multiply(count,2,count) ||
                !multiply(count,ggml_tensor_overhead(),metadata_bytes) ||
                !add(metadata_bytes,4096,metadata_bytes)) return false;
        next->ctx=ggml_init({metadata_bytes,nullptr,true});
        if (!next->ctx) return false;
        next->destinations.resize(count,nullptr);
        auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(next->execution));
        for (size_t layer=0;layer<s.layout.layers.size();++layer) {
            if (!s.layout.layers[layer].enabled) continue;
            const auto & item=s.layout.layers[layer];
            for (uint32_t snapshot=1;snapshot<=s.layout.depth;++snapshot) {
                for (bool value : {false,true}) {
                    const auto type=value ? item.type_s : item.type_r;
                    const auto width=value ? item.width_s : item.width_r;
                    auto * tensor=ggml_new_tensor_2d(next->ctx,type,int64_t(width),s.layout.cells);
                    size_t offset=0,bytes=0;
                    if (!tensor || !s.region(layer,value,snapshot,offset,bytes) ||
                            ggml_nbytes(tensor) != bytes ||
                            ggml_backend_tensor_alloc(next->execution,tensor,base+offset) != GGML_STATUS_SUCCESS) {
                        return false;
                    }
                    next->destinations[next->index(layer,value,snapshot,s.layout.depth)]=tensor;
                }
            }
        }
        next->slot_events.resize(slots,nullptr);
        next->slot_busy.resize(slots,0);
        next->layer_slots.resize(s.layout.layers.size(),UINT32_MAX);
        uint32_t ordinal=0;
        for (size_t layer=0;layer<s.layout.layers.size();++layer) {
            if (s.layout.layers[layer].enabled) next->layer_slots[layer]=ordinal++%slots;
        }

        for (auto & event : next->slot_events) event=ggml_backend_event_new(device);
        size_t ready_count=0;
        if (!multiply(slots,s.layout.depth,ready_count) || !multiply(ready_count,2,ready_count)) return false;
        next->ready_events.resize(ready_count,nullptr);
        for (auto & event : next->ready_events) event=ggml_backend_event_new(device);


        next->stage_event=ggml_backend_event_new(device);
        s.publication=std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
}

bool llama_recurrent_spill_bank::staged_publication_enabled() const noexcept {
    return bool(impl->publication);
}
size_t llama_recurrent_spill_bank::staged_device_bytes() const noexcept {
    return impl->publication ? impl->publication->stage->bytes() : 0;
}


ggml_tensor * llama_recurrent_spill_bank::publication_tensor(
        size_t layer, bool value, uint32_t snapshot) const noexcept {
    const auto & s=*impl;
    if (!s.publication || layer >= s.layout.layers.size() || !s.layout.layers[layer].enabled ||
            !snapshot || snapshot > s.layout.depth) return nullptr;
    return s.publication->destinations[s.publication->index(layer,value,snapshot,s.layout.depth)];
}
ggml_tensor * llama_recurrent_spill_bank::staging_tensor(
        size_t layer, bool value, uint32_t snapshot) const noexcept {
    const auto & s=*impl;
    if (!s.publication || layer >= s.layout.layers.size() || !s.layout.layers[layer].enabled ||
            !snapshot || snapshot > s.layout.depth) return nullptr;
    return s.publication->stage->tensor(layer,value,snapshot,s.publication->layer_slots[layer]);
}




bool llama_recurrent_spill_bank::begin_capture(llama_recurrent_spill_identity identity) {
    auto & s=*impl;
    if (s.poisoned || s.state != spill_transfer_state::idle || !s.identity(identity) ||
            identity.generation <= s.last_generation) return false;
    s.active=identity; s.state=spill_transfer_state::capturing;
    s.submitted=0; std::fill(s.touched.begin(),s.touched.end(),0);
    return true;
}

bool llama_recurrent_spill_bank::begin_staged_capture(
        llama_recurrent_spill_identity identity, uint32_t tokens) {
    auto & s=*impl;
    if (!s.publication || !tokens || !begin_capture(identity)) return false;
    auto & p=*s.publication;
    p.expected_s=std::min(tokens-1,s.layout.depth);
    p.expected_per_layer=s.layout.depth+p.expected_s;
    p.remaining=0;
    p.next_slot=0;
    p.current_layer=SIZE_MAX;
    p.completed_layers=0;
    p.producer=nullptr;
    p.active=true;
    std::fill(p.slot_busy.begin(),p.slot_busy.end(),0);
    return true;
}

bool llama_recurrent_spill_bank::stage_ready() {
    const auto & s=*impl;
    if (!s.publication || !s.publication->producer ||
            s.state != spill_transfer_state::capture_pending) return false;
    if (s.publication->stage_event) ggml_backend_event_synchronize(s.publication->stage_event);
    else ggml_backend_synchronize(s.publication->producer);
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
    if (s.publication && s.publication->active) {
        auto & p=*s.publication;
        if (p.remaining || p.completed_layers != s.active_planes/2 || !p.producer) return false;
        if (p.stage_event) ggml_backend_event_record(p.stage_event,p.producer);
        p.active=false;
    }
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
    if (s.publication) s.publication->producer=nullptr;
    if (s.publication) std::fill(s.publication->slot_busy.begin(),s.publication->slot_busy.end(),0);
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
    if (s.publication && s.publication->producer) ggml_backend_synchronize(s.publication->producer);
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
    if (s.publication) {
        s.publication->active=false;
        std::fill(s.publication->slot_busy.begin(),s.publication->slot_busy.end(),0);
        s.publication->producer=nullptr;
    }
}

void llama_recurrent_spill_bank::reset(bool pristine) noexcept {
    auto & s=*impl;
    if (s.state != spill_transfer_state::idle && s.publication && s.publication->producer)
        ggml_backend_synchronize(s.publication->producer);
    if (s.state != spill_transfer_state::idle && s.backend) ggml_backend_synchronize(s.backend);
    s.state=spill_transfer_state::idle;
    s.submitted=0;
    s.poisoned=false;
    std::fill(s.generations.begin(),s.generations.end(),0);
    std::fill(s.invalidated.begin(),s.invalidated.end(),pristine ? 0 : 1);
    std::fill(s.touched.begin(),s.touched.end(),0);
    if (pristine) ggml_backend_buffer_clear(s.host,0);
    s.active={};
    if (s.publication) {
        s.publication->active=false;
        s.publication->producer=nullptr;
        std::fill(s.publication->slot_busy.begin(),s.publication->slot_busy.end(),0);
    }
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
