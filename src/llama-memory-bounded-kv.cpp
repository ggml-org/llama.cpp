#include "llama-memory-bounded-kv.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <climits>
#include <limits>

static bool aligned_size(size_t bytes,size_t alignment,size_t & output) {
    if (!alignment || (alignment&(alignment-1)) || bytes > SIZE_MAX-(alignment-1)) return false;
    output=(bytes+alignment-1)&~(alignment-1);
    return true;
}

bool llama_memory_bounded_kv_layout_make(const llama_memory_cache_requirement & cache,
        const std::vector<uint32_t> & ids,ggml_backend_buffer_type_t buft,
        llama_memory_bounded_kv_layout & output) noexcept {
    llama_memory_cache_bound bound;
    if (!buft || cache.role != llama_memory_cache_role::bounded_resident || ids.size() != cache.layers ||
            ids.size() > size_t(INT_MAX)/2 || !llama_memory_cache_bound_make(cache,nullptr,bound)) return false;
    try {
        auto sorted=ids;
        std::sort(sorted.begin(),sorted.end());
        if (std::adjacent_find(sorted.begin(),sorted.end()) != sorted.end()) return false;
        const auto & format=cache.format;
        const uint64_t kw=uint64_t(format.key_head_dim)*format.kv_heads, vw=uint64_t(format.value_head_dim)*format.kv_heads;
        const int kb=ggml_blck_size(format.key_type), vb=ggml_blck_size(format.value_type);
        if (kb <= 0 || vb <= 0 || format.key_head_dim%kb || format.value_head_dim%vb ||
                kw > uint64_t(INT64_MAX)/bound.capacity_rows || vw > uint64_t(INT64_MAX)/bound.capacity_rows) return false;
        const size_t kt=ggml_type_size(format.key_type), vt=ggml_type_size(format.value_type);
        if (!kt || !vt || kw > SIZE_MAX/kt || vw > SIZE_MAX/vt) return false;
        const size_t kr=ggml_row_size(format.key_type,int64_t(kw)), vr=ggml_row_size(format.value_type,int64_t(vw));
        if (kr > SIZE_MAX-vr || kr+vr != cache.row_bytes || kr > SIZE_MAX/bound.capacity_rows || vr > SIZE_MAX/bound.capacity_rows) return false;
        const size_t alignment=ggml_backend_buft_get_alignment(buft);
        size_t dummy=0;
        if (!aligned_size(0,alignment,dummy)) return false;
        ggml_context_ptr ctx(ggml_init({2*ggml_tensor_overhead()+4096,nullptr,true}));
        if (!ctx) return false;
        auto * k=ggml_new_tensor_3d(ctx.get(),format.key_type,int64_t(kw),bound.capacity_rows,1);
        auto * v=ggml_new_tensor_3d(ctx.get(),format.value_type,int64_t(vw),bound.capacity_rows,1);
        const size_t ka=ggml_backend_buft_get_alloc_size(buft,k), va=ggml_backend_buft_get_alloc_size(buft,v);
        if (ka < ggml_nbytes(k) || va < ggml_nbytes(v)) return false;
        llama_memory_bounded_kv_layout next;
        next.cache=cache; next.buft=buft;
        next.allocation={cache.identity,0,alignment,bound.capacity_rows,format};
        size_t cursor=0;
        for (auto id : ids) {
            llama_memory_bounded_kv_layer layer;
            layer.id=id; layer.k_bytes=ggml_nbytes(k); layer.v_bytes=ggml_nbytes(v);
            if (!aligned_size(cursor,alignment,layer.k_offset) || ka > SIZE_MAX-layer.k_offset) return false;
            cursor=layer.k_offset+ka;
            if (!aligned_size(cursor,alignment,layer.v_offset) || va > SIZE_MAX-layer.v_offset) return false;
            cursor=layer.v_offset+va;
            next.layers.push_back(layer);
        }
        if (!aligned_size(cursor,alignment,next.allocation.bytes) ||
                next.allocation.bytes > ggml_backend_buft_get_max_size(buft)) return false;
        // The quote must also match the backend's complete tensor-list allocation.
        std::vector<ggml_tensor *> roots;
        roots.reserve(2*ids.size());
        for (size_t i=0;i<ids.size();++i) { roots.push_back(k); roots.push_back(v); }
        if (ggml_backend_buft_get_alloc_size_n(buft,roots.data(),int(roots.size())) != next.allocation.bytes) return false;
        output=std::move(next);
        return true;
    } catch (...) { return false; }
}

struct llama_memory_bounded_kv_binding {
    std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease{nullptr,ggml_backend_memory_lease_free};
    ggml_backend_memory_region region{};
    llama_memory_bounded_kv_layout layout;
};

llama_memory_bounded_kv_read::operator bool() const noexcept { return bool(binding); }
ggml_backend_buffer_t llama_memory_bounded_kv_read::buffer() const noexcept {
    return binding ? ggml_backend_memory_lease_buffer(binding->lease.get()) : nullptr;
}
const llama_memory_bounded_kv_layout * llama_memory_bounded_kv_read::layout() const noexcept {
    return binding ? &binding->layout : nullptr;
}
void llama_memory_bounded_kv_read::reset() noexcept { binding.reset(); }

llama_memory_bounded_kv::llama_memory_bounded_kv(llama_memory_bounded_kv_layout layout, llama_memory_resource resource) :
    planned(std::move(layout)), resource(resource) {}

// All selected phases retain the stock cache, even when their graph does not access it.
bool llama_memory_bounded_kv::register_resources(llama_memory_execution_plan & plan,
        const std::vector<llama_memory_stage_id> & stages) const {
    if (!active || !valid || pending_ticket || preparing || !resource.id || !resource.domain ||
            resource.content != llama_memory_content::preserve || active->region.id != resource.id || stages.empty()) return false;
    try {
        auto next=plan;
        for (const auto & existing : next.resources) if (existing.id == resource.id) return false;
        next.resources.push_back(resource);
        next.inputs.push_back(resource.id);
        next.outputs.push_back(resource.id);
        for (size_t i=0;i<stages.size();++i) {
            if (std::find(stages.begin(),stages.begin()+i,stages[i]) != stages.begin()+i) return false;
            const auto stage=std::find_if(next.stages.begin(),next.stages.end(),
                [&](const llama_memory_stage & item) { return item.id == stages[i]; });
            if (stage == next.stages.end()) return false;
            stage->requirements.push_back({resource.id,planned.allocation.bytes,planned.allocation.bytes,
                planned.allocation.alignment,LLAMA_MEMORY_ACCESS_READ_WRITE,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
        }
        if (llama_memory_plan_validate(next).status != llama_memory_plan_status::success) return false;
        plan=std::move(next);
        return true;
    } catch (...) { return false; }
}

// No cache address or content changes during this transition, so no cache-specific drain or graph retirement is needed.
struct llama_memory_bounded_kv::proposal : llama_memory_preparation {
    llama_memory_bounded_kv & owner;
    size_t arena;
    proposal(llama_memory_bounded_kv & owner,size_t arena) : owner(owner),arena(arena) {}
    ~proposal() override { owner.preparing=false; }
    bool quiesce(const std::vector<llama_memory_resource_id> &) override { return !owner.pending_ticket; }
    bool drain() override { return !owner.pending_ticket; }
    bool invalidate() override { return true; }
    bool release() override { return true; }
    bool bind(const std::vector<llama_memory_region_binding> & bindings) override {
        bool found=false;
        for (const auto & binding : bindings) {
            if (binding.region.id != owner.resource.id) continue;
            if (found || binding.arena != arena || !owner.bind(binding.lease)) return false;
            found=true;
        }
        return found;
    }
    bool activate() override { return owner.valid && !owner.pending_ticket; }
    // Cache state never changed, including when another participant failed after this one activated.
    bool prepare_recovery(std::unique_ptr<llama_memory_preparation> &) override { return true; }
};

bool llama_memory_bounded_kv::prepare(const llama_memory_transition_target & target,const llama_memory_layout & layout,
        std::unique_ptr<llama_memory_preparation> & output) {
    if (!active || !valid || pending_ticket || preparing || output || !resource.id ||
            resource.content != llama_memory_content::preserve || active->region.id != resource.id ||
            llama_memory_plan_validate(target.plan).status != llama_memory_plan_status::success) return false;
    const auto declared=std::find_if(target.plan.resources.begin(),target.plan.resources.end(),
        [&](const llama_memory_resource & item) { return item.id == resource.id; });
    const auto stage=std::find_if(target.plan.stages.begin(),target.plan.stages.end(),
        [&](const llama_memory_stage & item) { return item.id == target.stage; });
    if (declared == target.plan.resources.end() || declared->domain != resource.domain ||
            declared->allocation_class != resource.allocation_class || declared->content != resource.content ||
            stage == target.plan.stages.end()) return false;
    const auto requirement=std::find_if(stage->requirements.begin(),stage->requirements.end(),
        [&](const llama_memory_requirement & item) { return item.resource == resource.id; });
    if (requirement == stage->requirements.end() || requirement->size_min != planned.allocation.bytes ||
            requirement->size_preferred != planned.allocation.bytes || requirement->alignment != planned.allocation.alignment ||
            !(requirement->capabilities&LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS)) return false;
    size_t selected=SIZE_MAX;
    for (size_t a=0;a<layout.arenas.size();++a) {
        const auto & candidate=layout.arenas[a];
        for (const auto & region : candidate.regions) {
            if (region.id != resource.id) continue;
            const auto & previous=active->region;
            if (selected != SIZE_MAX || candidate.budget.domain != resource.domain ||
                    candidate.budget.allocation_class != resource.allocation_class || region.id != previous.id ||
                    region.offset != previous.offset || region.size != previous.size ||
                    region.alignment != previous.alignment || region.flags != previous.flags) return false;
            selected=a;
        }
    }
    if (selected == SIZE_MAX) return false;
    output=std::make_unique<proposal>(*this,selected);
    preparing=true;
    return true;
}

bool llama_memory_bounded_kv::bind(ggml_backend_memory_lease_t lease) noexcept {
    if (pending_ticket || !lease) return false;
    try {
        std::vector<uint32_t> ids;
        for (const auto & layer : planned.layers) ids.push_back(layer.id);
        llama_memory_bounded_kv_layout checked;
        if (!llama_memory_bounded_kv_layout_make(planned.cache,ids,planned.buft,checked) ||
                checked.allocation.bytes != planned.allocation.bytes || checked.allocation.alignment != planned.allocation.alignment ||
                checked.allocation.capacity_rows != planned.allocation.capacity_rows ||
                !llama_memory_cache_format_equal(planned.allocation.format,checked.allocation.format) ||
                planned.allocation.cache.id != planned.cache.identity.id || planned.allocation.cache.generation != planned.cache.identity.generation) return false;
        for (size_t i=0;i<ids.size();++i) {
            const auto & a=planned.layers[i], & b=checked.layers[i];
            if (a.k_offset != b.k_offset || a.v_offset != b.v_offset || a.k_bytes != b.k_bytes || a.v_bytes != b.v_bytes) return false;
        }
        ggml_backend_memory_region region;
        auto * buffer=ggml_backend_memory_lease_buffer(lease);
        if (!ggml_backend_memory_lease_get_region(lease,&region) || !(region.flags&GGML_BACKEND_MEMORY_REGION_PERSISTENT) ||
                !buffer || ggml_backend_buffer_get_type(buffer) != planned.buft || region.size < planned.allocation.bytes ||
                region.alignment < planned.allocation.alignment || ggml_backend_buffer_get_size(buffer) < region.size) return false;
        const auto base=uintptr_t(ggml_backend_buffer_get_base(buffer));
        if (!base || base%planned.allocation.alignment || region.size > UINTPTR_MAX-base) return false;
        if (active) {
            const auto & old=active->region;
            return ggml_backend_memory_lease_buffer(active->lease.get()) == buffer && old.id == region.id &&
                old.offset == region.offset && old.size == region.size && old.flags == region.flags && old.alignment == region.alignment;
        }
        auto next=std::make_shared<llama_memory_bounded_kv_binding>();
        next->layout=planned; next->region=region;
        next->lease.reset(ggml_backend_memory_lease_retain(lease));
        if (!next->lease) return false;
        active=std::move(next); valid=true;
        return true;
    } catch (...) { return false; }
}

llama_memory_bounded_kv_read llama_memory_bounded_kv::read(llama_memory_phase_cache_identity expected) const noexcept {
    llama_memory_bounded_kv_read pin;
    if (active && valid && !pending_ticket && expected.id == planned.cache.identity.id && expected.generation == planned.cache.identity.generation)
        pin.binding=active;
    return pin;
}

uint64_t llama_memory_bounded_kv::begin_write(llama_memory_phase_cache_identity expected) noexcept {
    if (!active || !valid || pending_ticket || preparing || active.use_count() != 1 || next_ticket == UINT64_MAX ||
            planned.cache.identity.generation == UINT64_MAX || expected.id != planned.cache.identity.id ||
            expected.generation != planned.cache.identity.generation) return 0;
    return pending_ticket=++next_ticket;
}

bool llama_memory_bounded_kv::finish_write(uint64_t ticket,bool success) noexcept {
    if (!ticket || ticket != pending_ticket) return false;
    pending_ticket=0;
    if (!success) { valid=false; return true; }
    ++planned.cache.identity.generation;
    planned.allocation.cache=planned.cache.identity;
    active->layout.cache.identity=planned.cache.identity;
    active->layout.allocation.cache=planned.cache.identity;
    return true;
}

bool llama_memory_bounded_kv::reconstruct(llama_memory_phase_cache_identity expected) noexcept {
    if (!active || pending_ticket || preparing || active.use_count() != 1 || planned.cache.identity.generation == UINT64_MAX ||
            expected.id != planned.cache.identity.id || expected.generation != planned.cache.identity.generation) return false;
    ++planned.cache.identity.generation; planned.allocation.cache=planned.cache.identity;
    active->layout.cache.identity=planned.cache.identity; active->layout.allocation.cache=planned.cache.identity;
    valid=true; return true;
}

llama_memory_phase_cache_identity llama_memory_bounded_kv::identity() const noexcept { return planned.cache.identity; }
const llama_memory_bounded_kv_layout & llama_memory_bounded_kv::layout() const noexcept { return planned; }
