#include "ane-mtp.h"

#include "ane-state-layout.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct common_ane_mtp_arena_buffer {
    IOSurfaceRef surface = nullptr;
    void * heap = nullptr;
    void * data = nullptr;
    size_t capacity = 0;

    ~common_ane_mtp_arena_buffer() {
        if (surface) {
            IOSurfaceUnlock(surface, 0, nullptr);
            CFRelease(surface);
            surface = nullptr;
        }
        std::free(heap);
        heap = nullptr;
        data = nullptr;
    }

    bool reserve(size_t size, bool use_iosurface) {
        if (capacity >= size) {
            return true;
        }
        const size_t page = 16 * 1024;
        const size_t rounded = ((size + page - 1) / page) * page;
        if (!use_iosurface) {
            void * replacement = nullptr;
            if (posix_memalign(&replacement, page, rounded) != 0) {
                return false;
            }
            if (surface) {
                IOSurfaceUnlock(surface, 0, nullptr);
                CFRelease(surface);
                surface = nullptr;
            }
            std::free(heap);
            heap = replacement;
            data = heap;
            capacity = rounded;
            return true;
        }
        NSDictionary * properties = @{
            (id) kIOSurfaceWidth: @(rounded),
            (id) kIOSurfaceHeight: @1,
            (id) kIOSurfaceBytesPerElement: @1,
            (id) kIOSurfaceBytesPerRow: @(rounded),
            (id) kIOSurfaceAllocSize: @(rounded),
        };
        IOSurfaceRef replacement = IOSurfaceCreate((CFDictionaryRef) properties);
        if (!replacement || IOSurfaceLock(replacement, 0, nullptr) != kIOReturnSuccess) {
            if (replacement) {
                CFRelease(replacement);
            }
            return false;
        }
        void * replacement_data = IOSurfaceGetBaseAddress(replacement);
        if (!replacement_data) {
            IOSurfaceUnlock(replacement, 0, nullptr);
            CFRelease(replacement);
            return false;
        }
        if (surface) {
            IOSurfaceUnlock(surface, 0, nullptr);
            CFRelease(surface);
        }
        std::free(heap);
        heap = nullptr;
        surface = replacement;
        data = replacement_data;
        capacity = rounded;
        return true;
    }
};

struct common_ane_compute_instance {
    MLModel * model = nil;
    // Single MLState per function. The 5s re-warm timer used to
    // require a separate keepalive_state so the live state wasn't
    // mutated by periodic warm-up predictions. With the timer gone
    // (the .mlmodelc stays loaded; IOSurface-backed state never
    // goes cold) we only need the live execution_state. Future
    // commits will drop this too once the dispatch path uses
    // pinned-state slots + outputBackings instead of MLState.
    MLState * execution_state = nil;
    std::string name;
    std::string role;
    uint32_t bucket = 0;
    std::atomic<bool> warm = false;
};

struct common_ane_prefill_request {
    common_ane_prefill_program_ptr program;
    dispatch_semaphore_t completion = nullptr;
    uint64_t arena_epoch = 0;
    std::atomic<bool> complete = false;
    std::atomic<bool> success = false;

    ~common_ane_prefill_request() {
        completion = nullptr;
    }
};

struct common_ane_mtp_program {
    MLModel * model = nil;
    // sync_model and reset_model are kept for this commit. The
    // architecture call is to drop them (sync/reset become
    // memcpy/memset on the E-core pump), but the dispatch path
    // in this file still calls into these CPU-only MLModel
    // functions. Replacing them is a follow-on commit that
    // re-implements common_ane_mtp_program_sync_kv and
    // common_ane_mtp_program_reset as direct memcpy/memset on
    // a state_iosurface. The 5s re-warm timer (below) and
    // per-instance keepalive_state (which only existed for the
    // timer) are dropped now because they were pure
    // performance-only machinery; the sync_model/reset_model
    // stay until the memcpy/memset refactor lands.
    MLModel * sync_model = nil;
    MLModel * reset_model = nil;
    MLState * execution_state = nil;
    dispatch_queue_t queue = nullptr;
    // keepalive (5s re-warm timer) is dropped. The .mlmodelc is
    // loaded once and stays loaded; state is in IOSurface (not in
    // an opaque MLState) and never "goes cold" in a way that
    // requires re-warming. Removing the timer also removes the
    // dispatch_source event handler and the per-instance
    // keepalive_state that existed only to keep the live state
    // untouched during re-warm.
    std::string cache_path;
    uint32_t batch_bucket = 1;
    uint32_t context_length = 0;
    uint32_t sync_chunk = 0;
    std::atomic<bool> warm = false;
    std::atomic<uint64_t> direct_input_views = 0;
    std::atomic<uint64_t> direct_output_backings = 0;
    std::atomic<uint64_t> arena_input_bytes = 0;
    std::atomic<uint64_t> iosurface_arena_bytes = 0;
    std::atomic<uint64_t> copied_output_bytes = 0;
    std::atomic<uint64_t> async_prefill_submissions = 0;
    std::atomic<uint64_t> async_prefill_completions = 0;
    std::atomic<uint64_t> async_prefill_failures = 0;
    // A prefill prediction may outlive the queue turn which submitted it.
    // Give every asynchronous call distinct arena slots so a later request
    // cannot overwrite Core ML input/output storage still owned by ANE.
    std::atomic<uint64_t> next_prefill_arena_epoch = 0;
    std::unordered_map<std::string, std::unique_ptr<common_ane_mtp_arena_buffer>> arena;
    std::unordered_map<std::string, std::unique_ptr<common_ane_compute_instance>> functions;
    // A multifunction bundle whose layer weights live in the source GGUF (rather
    // than in a per-bundle weight file) holds one MLMultiArray per declared
    // weight input here.  The C++ side memory-maps the relevant blk.0.* tensors
    // and the embedding table at bundle-load time and passes the cached arrays
    // on every prefill call.  This removes the duplicated 2.3 GB weight table
    // from the embedded bundle and keeps a single source of truth in the GGUF.
    std::unordered_map<std::string, MLMultiArray *> weight_inputs;
    std::vector<uint16_t> embedding;  // token_embd.weight, fp16, [vocab_size, hidden]
    uint32_t embedding_dim = 0;
    std::string gguf_path;

    // Optional: the ane_state_layout.v1 manifest next to the
    // materialized .mlmodelc. Read at load time if present; the
    // dispatch path uses the manifest's slot/function tables to
    // wire inputs to the pinned IOSurface slots and outputs to
    // MLPredictionOptions.outputBackings. When the manifest is
    // present, the multifunction refactor has the layout it needs;
    // when absent, the legacy GGUF-metadata path keeps working.
    ane_state_layout_v1_t manifest;
    bool                   manifest_loaded = false;

    // The shared state IOSurface: one buffer of manifest.state_size_bytes
    // holding every declared slot. All slots share this surface so ANE,
    // Metal, and the host CPU all see the same bytes (zero-copy). The
    // state_iosurface is allocated at load when manifest_loaded is true;
    // it is locked for the program's lifetime. Released in the destructor.
    IOSurfaceRef state_iosurface = nullptr;
    void *       state_base      = nullptr;
    size_t       state_size      = 0;

    // Pinned MLMultiArray wrappers, one per manifest slot, each pointing
    // into the state_iosurface at the slot's manifest offset with
    // deallocator:nil. Core ML reads/writes through these arrays; the
    // host reads/writes the same physical pages. Index = slot_id from
    // the manifest. Released in the destructor.
    MLMultiArray * pinned_slots[ANE_STATE_SLOTS_MAX] = {};

    ~common_ane_mtp_program() {
        // Drop the pinned MLMultiArray strong references first; they
        // wrap the IOSurface so the order matters (the IOSurface
        // unlock + release below must come after). Under ARC the
        // strong references are released by setting to nil; no
        // explicit [obj release] is needed (or allowed).
        for (uint32_t i = 0; i < ANE_STATE_SLOTS_MAX; ++i) {
            pinned_slots[i] = nil;
        }
        // Unlock + release the state IOSurface. CFRelease is still
        // required under ARC (CFType is not ARC-managed).
        if (state_iosurface != nullptr) {
            IOSurfaceUnlock(state_iosurface, 0, nullptr);
            CFRelease(state_iosurface);
            state_iosurface = nullptr;
        }
        state_base = nullptr;
        state_size = 0;
        execution_state = nil;
        model = nil;
        sync_model = nil;
        reset_model = nil;
        // The MLMultiArray weight inputs are autoreleased in their creating
        // scope; clearing the map drops the strong references and the next
        // drain releases the underlying objects.
        weight_inputs.clear();
        functions.clear();
        queue = nullptr;
    }
};

static size_t multi_array_element_size(MLMultiArrayDataType type) {
    switch (type) {
        case MLMultiArrayDataTypeFloat16: return sizeof(ggml_fp16_t);
        case MLMultiArrayDataTypeFloat32: return sizeof(float);
        case MLMultiArrayDataTypeInt32:   return sizeof(int32_t);
        default:                         return 0;
    }
}

static NSArray<NSNumber *> * contiguous_strides(NSArray<NSNumber *> * shape) {
    NSMutableArray<NSNumber *> * result =
        [NSMutableArray arrayWithCapacity:shape.count];
    NSUInteger stride = 1;
    for (NSInteger i = (NSInteger) shape.count - 1; i >= 0; --i) {
        [result insertObject:@(stride) atIndex:0];
        stride *= shape[(NSUInteger) i].unsignedIntegerValue;
    }
    return result;
}

static size_t shape_count(NSArray<NSNumber *> * shape) {
    size_t count = 1;
    for (NSNumber * dimension in shape) {
        count *= dimension.unsignedIntegerValue;
    }
    return count;
}

static MLMultiArray * wrap_multi_array(
        void * data,
        NSArray<NSNumber *> * shape,
        MLMultiArrayDataType type,
        NSError ** error) {
    return [[MLMultiArray alloc]
        initWithDataPointer:data
                      shape:shape
                   dataType:type
                    strides:contiguous_strides(shape)
                deallocator:nil
                      error:error];
}

static MLMultiArray * arena_multi_array(
        common_ane_mtp_program & program,
        const std::string & name,
        NSArray<NSNumber *> * shape,
        MLMultiArrayDataType type,
        NSError ** error) {
    const size_t element_size = multi_array_element_size(type);
    if (element_size == 0) {
        return nil;
    }
    auto & slot = program.arena[name];
    if (!slot) {
        slot = std::make_unique<common_ane_mtp_arena_buffer>();
    }
    // Core ML's CPU-only state mutation functions cannot reliably consume an
    // IOSurface-backed MLMultiArray.  They are off the latency-critical path;
    // use the aligned host fallback while stateless ANE functions retain the
    // IOSurface arena contract.
    const bool use_iosurface = name.rfind("sync.", 0) != 0 &&
        name.rfind("reset.", 0) != 0;
    if (!slot->reserve(shape_count(shape) * element_size, use_iosurface)) {
        return nil;
    }
    if (use_iosurface) {
        program.iosurface_arena_bytes += shape_count(shape) * element_size;
    }
    return wrap_multi_array(slot->data, shape, type, error);
}

static void write_float_data(MLMultiArray * array, const float * source, size_t count) {
    if (array.dataType == MLMultiArrayDataTypeFloat32) {
        std::memcpy(array.dataPointer, source, count * sizeof(float));
    } else {
        ggml_fp32_to_fp16_row(source, (ggml_fp16_t *) array.dataPointer, count);
    }
}

// Allocate the shared state IOSurface for a manifest. The size is
// rounded up to the ANE 16 KB page boundary and clamped to the 64 KB
// IOSurface minimum (Orion #4). Returns the locked base address on
// success; returns nullptr on failure. The IOSurfaceRef is stored in
// `out_iosurface` for the caller to release.
static void * allocate_state_iosurface(size_t requested_bytes,
                                        IOSurfaceRef * out_iosurface) {
    const size_t page = 16 * 1024;
    const size_t min_size = 64 * 1024;
    const size_t rounded = std::max(min_size,
            ((requested_bytes + page - 1) / page) * page);
    // Match the ggml-ane backend's IOSurface allocation: width/height/
    // bytesPerElement/bytesPerRow/allocSize all set so the kernel can
    // use the surface for both ANE pages and CPU writes.
    NSDictionary * properties = @{
        (id) kIOSurfaceWidth:          @(rounded),
        (id) kIOSurfaceHeight:         @1,
        (id) kIOSurfaceBytesPerElement:@1,
        (id) kIOSurfaceBytesPerRow:    @(rounded),
        (id) kIOSurfaceAllocSize:      @(rounded),
    };
    IOSurfaceRef surface = IOSurfaceCreate((CFDictionaryRef) properties);
    if (surface == nullptr) {
        return nullptr;
    }
    if (IOSurfaceLock(surface, 0, nullptr) != kIOReturnSuccess) {
        CFRelease(surface);
        return nullptr;
    }
    void * base = IOSurfaceGetBaseAddress(surface);
    if (base == nullptr) {
        IOSurfaceUnlock(surface, 0, nullptr);
        CFRelease(surface);
        return nullptr;
    }
    // Zero the state at load so the first real dispatch isn't polluted
    // by IOSurface allocator garbage. The K/V slots start as zero;
    // INPUT slots are written before each dispatch.
    std::memset(base, 0, rounded);
    *out_iosurface = surface;
    return base;
}

// Wrap one IOSurface-backed slot as an MLMultiArray with deallocator:nil
// (zero-copy). Mirrors the canonical pattern in ggml/src/ggml-ane/ggml-ane.mm
// (ggml_ane_pin_slot). The MLMultiArray's data pointer is
// state_base + slot.offset; its shape and dtype come from the manifest.
// Returns a strong-reference MLMultiArray that the caller stores in a
// long-lived slot (e.g. program->pinned_slots[]). Under ARC the strong
// reference is tracked automatically; setting the slot to nil later
// drops the reference.
static MLMultiArray * pin_iosurface_slot(void * state_base,
                                          const ane_slot_v1_t * slot) {
    NSError * error = nil;
    NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:slot->n_dim];
    for (uint32_t i = 0; i < slot->n_dim; ++i) {
        [shape addObject:@(slot->shape[i])];
    }
    NSArray<NSNumber *> * strides = contiguous_strides(shape);
    MLMultiArrayDataType dtype = MLMultiArrayDataTypeFloat32;
    switch (slot->dtype) {
        case ANE_DTYPE_F32: dtype = MLMultiArrayDataTypeFloat32; break;
        case ANE_DTYPE_F16: dtype = MLMultiArrayDataTypeFloat16; break;
        case ANE_DTYPE_I32: dtype = MLMultiArrayDataTypeInt32;   break;
    }
    void * slot_base = (char *) state_base + slot->offset;
    return [[MLMultiArray alloc]
        initWithDataPointer:slot_base
                      shape:shape
                   dataType:dtype
                    strides:strides
                deallocator:nil
                      error:&error];
}

static void read_float_data(const MLMultiArray * array, float * destination, size_t count) {
    GGML_ASSERT(array && destination && count <= (size_t) array.count);
    const NSArray<NSNumber *> * shape = array.shape;
    const NSArray<NSNumber *> * strides = array.strides;
    bool contiguous = true;
    size_t expected_stride = 1;
    for (NSInteger dim = (NSInteger) shape.count - 1; dim >= 0; --dim) {
        if (strides[(NSUInteger) dim].unsignedIntegerValue != expected_stride) {
            contiguous = false;
            break;
        }
        expected_stride *= shape[(NSUInteger) dim].unsignedIntegerValue;
    }
    if (contiguous) {
        if (array.dataType == MLMultiArrayDataTypeFloat32) {
            std::memcpy(destination, array.dataPointer, count * sizeof(float));
        } else {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) array.dataPointer, destination, count);
        }
        return;
    }
    for (size_t linear = 0; linear < count; ++linear) {
        size_t remaining = linear;
        size_t offset = 0;
        for (NSInteger dim = (NSInteger) shape.count - 1; dim >= 0; --dim) {
            const size_t extent = shape[(NSUInteger) dim].unsignedIntegerValue;
            const size_t index = remaining % extent;
            remaining /= extent;
            offset += index * strides[(NSUInteger) dim].unsignedIntegerValue;
        }
        destination[linear] = array.dataType == MLMultiArrayDataTypeFloat32
            ? ((const float *) array.dataPointer)[offset]
            : ggml_fp16_to_fp32(((const ggml_fp16_t *) array.dataPointer)[offset]);
    }
}

static bool safe_relative_path(const std::string & value) {
    const fs::path path(value);
    if (value.empty() || path.is_absolute()) {
        return false;
    }
    return std::none_of(path.begin(), path.end(), [](const fs::path & part) {
        return part == "..";
    });
}

static std::string gguf_string(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    return gguf_get_val_str(ctx, id);
}

static uint32_t gguf_u32(
        const gguf_context * ctx,
        const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        return 0;
    }
    return gguf_get_val_u32(ctx, id);
}

static std::vector<std::string> gguf_string_array(
        const gguf_context * ctx,
        const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    std::vector<std::string> result;
    const size_t count = gguf_get_arr_n(ctx, id);
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return result;
}

static bool parse_compute_function(
        const std::string & name,
        std::string & role,
        uint32_t & bucket) {
    const auto parse_suffix = [&](const char * prefix) {
        const size_t length = std::strlen(prefix);
        if (name.compare(0, length, prefix) != 0 || name.size() == length) {
            return false;
        }
        char * end = nullptr;
        const unsigned long value = std::strtoul(name.c_str() + length, &end, 10);
        if (!end || *end != '\0' || value == 0 || value > UINT32_MAX) {
            return false;
        }
        bucket = (uint32_t) value;
        return true;
    };
    if (parse_suffix("prefill_s")) {
        role = "prefill";
        return true;
    }
    if (parse_suffix("dflash_b")) {
        role = "dflash";
        return true;
    }
    if (parse_suffix("hybrid_b")) {
        role = "hybrid";
        return true;
    }
    return false;
}

static bool materialize_bundle(
        const std::string & gguf_path,
        const gguf_context * ctx,
        const std::string & key_prefix,
        uint32_t file_count,
        const std::string & digest,
        fs::path & output) {
    @autoreleasepool {
        NSArray<NSURL *> * urls = [[NSFileManager defaultManager]
            URLsForDirectory:NSCachesDirectory
                   inDomains:NSUserDomainMask];
        if (urls.count == 0) {
            return false;
        }
        fs::path cache_root([urls.firstObject.path UTF8String]);
        output = cache_root / "llama.cpp" / "ane-mtp" / (digest + ".mlmodelc");
    }

    if (fs::exists(output / "model.mil") || fs::exists(output / "metadata.json")) {
        return true;
    }

    const fs::path staging = output.string() + ".staging";
    std::error_code ec;
    fs::remove_all(staging, ec);
    fs::create_directories(staging, ec);
    if (ec) {
        return false;
    }

    std::ifstream input(gguf_path, std::ios::binary);
    if (!input) {
        return false;
    }
    const size_t data_offset = gguf_get_data_offset(ctx);

    for (uint32_t i = 0; i < file_count; ++i) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "%04u", i);
        const std::string tensor_name = key_prefix + ".file." + suffix;
        const std::string relative = gguf_string(ctx, tensor_name + ".path");
        if (!safe_relative_path(relative)) {
            fs::remove_all(staging, ec);
            return false;
        }

        const int64_t tensor_id = gguf_find_tensor(ctx, tensor_name.c_str());
        if (tensor_id < 0 || gguf_get_tensor_type(ctx, tensor_id) != GGML_TYPE_I8) {
            fs::remove_all(staging, ec);
            return false;
        }
        const size_t size = gguf_get_tensor_size(ctx, tensor_id);
        const size_t offset = data_offset + gguf_get_tensor_offset(ctx, tensor_id);
        std::vector<char> bytes(size);
        input.seekg((std::streamoff) offset);
        input.read(bytes.data(), (std::streamsize) size);
        if ((size_t) input.gcount() != size) {
            fs::remove_all(staging, ec);
            return false;
        }

        const fs::path destination = staging / fs::path(relative);
        fs::create_directories(destination.parent_path(), ec);
        if (ec) {
            fs::remove_all(staging, ec);
            return false;
        }
        std::ofstream file(destination, std::ios::binary);
        file.write(bytes.data(), (std::streamsize) bytes.size());
        if (!file) {
            fs::remove_all(staging, ec);
            return false;
        }
    }

    fs::create_directories(output.parent_path(), ec);
    fs::rename(staging, output, ec);
    if (ec && !fs::exists(output)) {
        fs::remove_all(staging, ec);
        return false;
    }
    return true;
}

// Map a bundle weight input name to the matching GGUF tensor name in the
// Gemma 4 unified model.  The bundle declares inputs like `q_weight` and
// `attn_norm`; the corresponding tensors live at `blk.0.attn_q.weight` and
// `blk.0.attn_norm.weight` in the source GGUF.  The "embedded" input is
// special: it is computed at runtime from `token_embd.weight` by gathering
// the rows for the active token IDs.
static const std::unordered_map<std::string, std::string> &
weight_input_to_gguf_tensor() {
    static const std::unordered_map<std::string, std::string> map = {
        {"attn_norm", "blk.0.attn_norm.weight"},
        {"q_weight", "blk.0.attn_q.weight"},
        {"k_weight", "blk.0.attn_k.weight"},
        {"v_weight", "blk.0.attn_v.weight"},
        {"q_norm", "blk.0.attn_q_norm.weight"},
        {"k_norm", "blk.0.attn_k_norm.weight"},
        {"o_weight", "blk.0.attn_output.weight"},
        {"post_attn", "blk.0.post_attention_norm.weight"},
        {"ffn_norm", "blk.0.ffn_norm.weight"},
        {"gate_weight", "blk.0.ffn_gate.weight"},
        {"up_weight", "blk.0.ffn_up.weight"},
        {"down_weight", "blk.0.ffn_down.weight"},
        {"post_ffn", "blk.0.post_ffw_norm.weight"},
        {"scale", "blk.0.layer_output_scale.weight"},
    };
    return map;
}

// Memory-map the source GGUF and populate `program->weight_inputs` with one
// MLMultiArray per declared bundle weight.  The arrays are pinned to the
// file mapping for the lifetime of the program, so the OS pages the data
// in on demand and no extra copy is made.  ANE accepts MLMultiArray
// backed by a memmap when the page is read-only and 16 KB aligned, which
// our file mapping satisfies.
static bool populate_weight_arrays(
        common_ane_mtp_program & program,
        const std::string & gguf_path,
        const gguf_context * ctx) {
    const auto & inputs = program.model.modelDescription.inputDescriptionsByName;
    if (!inputs) {
        return true;
    }
    const auto & mapping = weight_input_to_gguf_tensor();
    const size_t data_offset = gguf_get_data_offset(ctx);

    int fd = ::open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_WRN("failed to open GGUF for weight mapping: %s\n", gguf_path.c_str());
        return false;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return false;
    }
    void * mapped = ::mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        LOG_WRN("failed to mmap GGUF for weight inputs\n");
        return false;
    }
    const uint8_t * base = (const uint8_t *) mapped;

    for (NSString * name in inputs.allKeys) {
        const std::string input_name = [name UTF8String];
        if (input_name == "token_ids" || input_name == "positions" || input_name == "embedded") {
            continue;
        }
        const auto found = mapping.find(input_name);
        if (found == mapping.end()) {
            continue;
        }
        const int64_t tid = gguf_find_tensor(ctx, found->second.c_str());
        if (tid < 0) {
            LOG_WRN("ANE weight input %s expects GGUF tensor %s which is missing\n",
                    input_name.c_str(), found->second.c_str());
            ::munmap(mapped, (size_t) st.st_size);
            return false;
        }
        if (gguf_get_tensor_type(ctx, tid) != GGML_TYPE_F16) {
            LOG_WRN("ANE weight input %s expects tensor %s in fp16\n",
                    input_name.c_str(), found->second.c_str());
            ::munmap(mapped, (size_t) st.st_size);
            return false;
        }
        const size_t offset = data_offset + gguf_get_tensor_offset(ctx, tid);
        const size_t size = gguf_get_tensor_size(ctx, tid);
        MLFeatureDescription * desc = inputs[name];
        if (!desc || desc.type != MLFeatureTypeMultiArray) {
            continue;
        }
        NSArray<NSNumber *> * shape = desc.multiArrayConstraint.shape;
        NSMutableArray<NSNumber *> * ns_shape = [shape mutableCopy];
        // MLMultiArray is row-major; GGUF stores tensors in column-major for
        // some llama.cpp tensors.  The bundle expects GGUF column-major
        // ordering on disk, so we hand the bytes through as-is.  The shape
        // declared in the bundle is the row-major view used by Core ML.
        // The deallocator is nil because the backing pages are owned by the
        // mmap and outlive this MLMultiArray instance.
        NSError * err = nil;
        MLMultiArray * array = [[MLMultiArray alloc]
            initWithDataPointer:(void *) (base + offset)
                          shape:ns_shape
                       dataType:MLMultiArrayDataTypeFloat16
                        strides:contiguous_strides(ns_shape)
                    deallocator:nil
                            error:&err];
        if (!array) {
            LOG_WRN("failed to wrap weight %s from GGUF: %s\n",
                    input_name.c_str(),
                    err.localizedDescription.UTF8String ?: "unknown error");
            ::munmap(mapped, (size_t) st.st_size);
            return false;
        }
        program.weight_inputs[input_name] = array;
    }
    // Load the embedding table for runtime token gather.  Hold the bytes
    // in a host buffer rather than via the mmap so the per-call gather has
    // predictable page residency; the table is read once and reused.
    const int64_t emb_tid = gguf_find_tensor(ctx, "token_embd.weight");
    if (emb_tid >= 0 && gguf_get_tensor_type(ctx, emb_tid) == GGML_TYPE_F16) {
        const size_t offset = data_offset + gguf_get_tensor_offset(ctx, emb_tid);
        const size_t size = gguf_get_tensor_size(ctx, emb_tid);
        program.embedding.assign((const uint16_t *) (base + offset),
                (const uint16_t *) (base + offset + size / sizeof(uint16_t)));
        program.embedding_dim = (uint32_t) (size / sizeof(uint16_t) /
                (program.embedding.empty() ? 1 : program.embedding.size()));
    }
    ::munmap(mapped, (size_t) st.st_size);
    return true;
}

static MLDictionaryFeatureProvider * make_zero_inputs(MLModel * model, uint32_t batch_hint, NSError ** error) {
    NSMutableDictionary<NSString *, MLFeatureValue *> * values = [NSMutableDictionary dictionary];
    for (NSString * name in model.modelDescription.inputDescriptionsByName) {
        MLFeatureDescription * desc = model.modelDescription.inputDescriptionsByName[name];
        if (desc.type != MLFeatureTypeMultiArray) {
            return nil;
        }
        MLMultiArrayConstraint * constraint = desc.multiArrayConstraint;
        NSMutableArray<NSNumber *> * shape = [constraint.shape mutableCopy];
        for (NSUInteger i = 0; i < shape.count; ++i) {
            if (shape[i].integerValue <= 0) {
                shape[i] = @((i == 0) ? std::max(1u, batch_hint) : 1u);
            }
        }
        MLMultiArray * array = [[MLMultiArray alloc] initWithShape:shape
                                                         dataType:constraint.dataType
                                                            error:error];
        if (!array) {
            return nil;
        }
        size_t element_size = 0;
        switch (constraint.dataType) {
            case MLMultiArrayDataTypeDouble:  element_size = sizeof(double);  break;
            case MLMultiArrayDataTypeFloat32: element_size = sizeof(float);   break;
            case MLMultiArrayDataTypeInt32:   element_size = sizeof(int32_t); break;
            case MLMultiArrayDataTypeFloat16: element_size = sizeof(uint16_t); break;
            default:
                return nil;
        }
        std::memset(array.dataPointer, 0, array.count * element_size);
        values[name] = [MLFeatureValue featureValueWithMultiArray:array];
    }
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:error];
}

static bool warm_model(
        MLModel * model,
        MLState * state,
        uint32_t batch_hint,
        const char * label) {
    @autoreleasepool {
        NSError * error = nil;
        MLDictionaryFeatureProvider * inputs = make_zero_inputs(model, batch_hint, &error);
        if (!inputs) {
            LOG_WRN("ANE %s warmup input creation failed: %s\n", label,
                    error.localizedDescription.UTF8String ?: "unsupported input");
            return false;
        }
        id<MLFeatureProvider> output;
        if (state) {
            output = [model predictionFromFeatures:inputs
                                        usingState:state
                                           options:[[MLPredictionOptions alloc] init]
                                             error:&error];
        } else {
            output = [model predictionFromFeatures:inputs error:&error];
        }
        if (!output) {
            LOG_WRN("ANE %s warmup prediction failed: %s\n", label,
                    error.localizedDescription.UTF8String ?: "unknown error");
            return false;
        }
        return true;
    }
}

static bool warm_program(common_ane_mtp_program & program, uint32_t batch_hint) {
    // Use the live execution_state (no separate keepalive_state;
    // the 5s re-warm timer is gone, so the warm path can use the
    // live state directly).
    return warm_model(
        program.model, program.execution_state, batch_hint, "MTP");
}

static common_ane_mtp_program_ptr common_ane_program_load(
        const std::string & gguf_path,
        uint32_t batch_hint,
        const std::string & root_prefix,
        uint32_t lane_bucket_override = 0) {
    struct gguf_init_params init = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
        gguf_init_from_file(gguf_path.c_str(), init), gguf_free);
    if (!ctx) {
        return nullptr;
    }
    uint32_t selected_batch = 1;
    std::string key_prefix = root_prefix;
    // A multifunction bundle exposes a single `bundle.file_count` and a
    // `bundle.functions` array.  The Core ML converter shares one weight table
    // across every published function, so a 128/256/512 prefill bundle stays
    // at one weight file's worth of on-disk size.  Prefer it when present and
    // fall through to the per-bucket layout only when it is not.
    const std::string bundle_file_count_key = root_prefix + ".bundle.file_count";
    if (gguf_find_key(ctx.get(), bundle_file_count_key.c_str()) >= 0) {
        key_prefix = root_prefix + ".bundle";
    } else {
        const std::string bucket_key = root_prefix == "tessera.ane.prefill"
            ? root_prefix + ".sequence_buckets"
            : root_prefix + ".batch_buckets";
        const int64_t buckets_key = gguf_find_key(ctx.get(), bucket_key.c_str());
        if (buckets_key >= 0 &&
                gguf_get_kv_type(ctx.get(), buckets_key) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(ctx.get(), buckets_key) == GGUF_TYPE_INT32) {
            const size_t count = gguf_get_arr_n(ctx.get(), buckets_key);
            const int32_t * buckets = (const int32_t *) gguf_get_arr_data(ctx.get(), buckets_key);
            if (count > 0) {
                selected_batch = (uint32_t) buckets[count - 1];
                for (size_t i = 0; i < count; ++i) {
                    if (buckets[i] >= (int32_t) std::max(1u, batch_hint)) {
                        selected_batch = (uint32_t) buckets[i];
                        break;
                    }
                }
                key_prefix = root_prefix + ".bucket." + std::to_string(selected_batch);
            }
        }
    }

    int64_t count_key = gguf_find_key(ctx.get(), (key_prefix + ".file_count").c_str());
    if (count_key < 0 && key_prefix != root_prefix) {
        // Compatibility with the original single-bundle prototype.
        key_prefix = root_prefix;
        selected_batch = 1;
        count_key = gguf_find_key(ctx.get(), (root_prefix + ".file_count").c_str());
    }
    if (count_key < 0 || gguf_get_kv_type(ctx.get(), count_key) != GGUF_TYPE_UINT32) {
        return nullptr;
    }
    const uint32_t file_count = gguf_get_val_u32(ctx.get(), count_key);
    const std::string digest = gguf_string(ctx.get(), key_prefix + ".bundle_sha256");
    if (file_count == 0 || digest.size() != 64) {
        return nullptr;
    }
    const std::vector<std::string> compute_functions =
        gguf_string_array(ctx.get(), key_prefix + ".functions");

    fs::path bundle_path;
    if (!materialize_bundle(gguf_path, ctx.get(), key_prefix, file_count, digest, bundle_path)) {
        LOG_WRN("failed to materialize embedded ANE bundle in namespace %s\n", root_prefix.c_str());
        return nullptr;
    }

    @autoreleasepool {
        NSError * error = nil;
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel * model = [MLModel modelWithContentsOfURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:bundle_path.c_str()]]
                                           configuration:config
                                                   error:&error];
        if (!model) {
            LOG_WRN("failed to load embedded ANE program in namespace %s: %s\n", root_prefix.c_str(),
                    error.localizedDescription.UTF8String ?: "unknown error");
            return nullptr;
        }

        auto program = std::make_shared<common_ane_mtp_program>();
        program->model = model;
        // sync_model and reset_model are kept for this commit.
        // The architecture call is to drop them (sync/reset
        // become memcpy/memset on the E-core pump) but the
        // dispatch path in this file still calls them. The
        // follow-on refactor will replace the public
        // common_ane_mtp_program_sync_kv and
        // common_ane_mtp_program_reset with direct memcpy/memset
        // against a state_iosurface; once that lands, the
        // CPU-only MLModel objects can be dropped along with the
        // .mlmodelc's "sync" and "reset" functions.
        if (@available(macOS 15.0, *)) {
            MLModelConfiguration * sync_config = [[MLModelConfiguration alloc] init];
            // Stateful scatter/reset operations are latency-insensitive and
            // compile poorly on ANE at the larger fixed buckets. Keep them on
            // CPU while sharing the same MLState with the ANE prediction
            // function; this leaves ANE capacity and residency for the MTP
            // transformer itself.
            sync_config.computeUnits = MLComputeUnitsCPUOnly;
            sync_config.functionName = @"sync";
            NSError * sync_error = nil;
            program->sync_model = [MLModel modelWithContentsOfURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:bundle_path.c_str()]]
                                                   configuration:sync_config
                                                           error:&sync_error];
            MLModelConfiguration * reset_config = [[MLModelConfiguration alloc] init];
            reset_config.computeUnits = MLComputeUnitsCPUOnly;
            reset_config.functionName = @"reset";
            NSError * reset_error = nil;
            program->reset_model = [MLModel modelWithContentsOfURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:bundle_path.c_str()]]
                                                   configuration:reset_config
                                                           error:&reset_error];
        }
        // Keep the live execution_state for the dispatch path. The
        // 5s re-warm timer and the keepalive_state it protected
        // are gone; with IOSurface-backed state and the .mlmodelc
        // loaded once at startup, neither the state nor the
        // compiled model goes cold.
        program->execution_state = [model newState];

        // Optional: read the ane_state_layout.v1 manifest next to
        // the .mlmodelc. The manifest is the source of truth for
        // the multifunction state layout (slot kinds, offsets,
        // function inputs/outputs, dependency graph). The
        // dispatch path doesn't use it yet (still on the
        // GGUF-metadata + arena + MLState path) but the load
        // path now reserves the space and validates the manifest
        // so the converter can start emitting it without the
        // runtime being unprepared. When the manifest is
        // present, the multifunction refactor has the layout it
        // needs; when absent, the legacy GGUF-metadata path
        // keeps working.
        {
            const std::string manifest_path_str =
                ane_layout::manifest_path_for_mlmodelc_dir(
                    bundle_path.c_str());
            if (!manifest_path_str.empty()) {
                std::string merror;
                if (ane_layout::read_state_layout(manifest_path_str.c_str(),
                                                  &program->manifest, &merror)) {
                    program->manifest_loaded = true;
                    LOG_INF("loaded ane_state_layout.v1 manifest for %s: "
                            "%u slots, %u functions, %u deps\n",
                            program->manifest.bundle_name,
                            program->manifest.n_slots,
                            program->manifest.n_functions,
                            program->manifest.n_deps);
                } else if (fs::exists(manifest_path_str)) {
                    // Manifest exists but is malformed. The load
                    // is still allowed (the legacy path covers it)
                    // but we warn loudly so the converter knows.
                    LOG_WRN("ignoring malformed ane_state_layout.v1 "
                            "manifest at %s: %s\n",
                            manifest_path_str.c_str(), merror.c_str());
                }
            }
        }
        // Allocate the shared state IOSurface and pin every declared
        // slot to a subregion. Done when the manifest is present; the
        // dispatch refactor (a follow-on commit) will use these pinned
        // arrays for input writes and outputBackings. Without the
        // manifest we keep the legacy per-arena allocation path.
        if (program->manifest_loaded) {
            program->state_base = allocate_state_iosurface(
                program->manifest.state_size_bytes,
                &program->state_iosurface);
            if (program->state_base == nullptr) {
                LOG_WRN("failed to allocate %zu byte state IOSurface for %s\n",
                        program->manifest.state_size_bytes,
                        program->manifest.bundle_name);
                return nullptr;
            }
            program->state_size = program->manifest.state_size_bytes;
            for (uint32_t i = 0; i < program->manifest.n_slots; ++i) {
                program->pinned_slots[i] = pin_iosurface_slot(
                    program->state_base, &program->manifest.slots[i]);
                if (program->pinned_slots[i] == nil) {
                    LOG_WRN("failed to pin slot %s for %s\n",
                            program->manifest.slots[i].name,
                            program->manifest.bundle_name);
                    return nullptr;
                }
            }
            LOG_INF("ANE state IOSurface: %zu bytes, %u pinned slots for %s\n",
                    program->state_size, program->manifest.n_slots,
                    program->manifest.bundle_name);
        }
        program->queue = dispatch_queue_create("org.ggml.llama.ane", DISPATCH_QUEUE_SERIAL);
        program->cache_path = bundle_path.string();
        program->batch_bucket = lane_bucket_override ? lane_bucket_override : selected_batch;
        program->context_length =
            gguf_u32(ctx.get(), key_prefix + ".context_length");
        program->sync_chunk =
            gguf_u32(ctx.get(), key_prefix + ".sync_chunk");

        dispatch_sync(program->queue, ^{
            program->warm.store(warm_program(*program, program->batch_bucket));
        });
        if (!program->warm.load()) {
            return nullptr;
        }
        if (root_prefix == "tessera.ane.prefill") {
            // Multifunction prefill bundles expect their layer-0 weights to
            // come from the source GGUF rather than from a duplicated table
            // in the bundle itself.  Populate the weight cache once at
            // load time; the per-call compute path reads from it directly.
            if (!populate_weight_arrays(*program, gguf_path, ctx.get())) {
                LOG_WRN("failed to populate weight cache from %s\n", gguf_path.c_str());
                return nullptr;
            }
        }

        if (@available(macOS 15.0, *)) {
            for (const std::string & function_name : compute_functions) {
                std::string role;
                uint32_t bucket = 0;
                if (!parse_compute_function(function_name, role, bucket)) {
                    continue;
                }
                MLModelConfiguration * function_config = [[MLModelConfiguration alloc] init];
                // Diagnostic override used by the parity harness to separate
                // a Core ML graph-conversion error from ANE execution error.
                // Production defaults to CPUAndNeuralEngine.
                const bool prefill_cpu_only = root_prefix == "tessera.ane.prefill" &&
                    std::getenv("LLAMA_ANE_PREFILL_CPU_ONLY") != nullptr;
                function_config.computeUnits = prefill_cpu_only
                    ? MLComputeUnitsCPUOnly
                    : MLComputeUnitsCPUAndNeuralEngine;
                function_config.functionName =
                    [NSString stringWithUTF8String:function_name.c_str()];
                NSError * function_error = nil;
                MLModel * function_model = [MLModel modelWithContentsOfURL:
                    [NSURL fileURLWithPath:[NSString stringWithUTF8String:bundle_path.c_str()]]
                                                       configuration:function_config
                                                           error:&function_error];
                // A stateless prefill artifact may contain one Core ML
                // function only.  Core ML calls that default function "main"
                // even though the GGUF ABI names its entry point prefill_sN.
                // Keep this compatibility bridge deliberately narrow: it is
                // legal only for a single declared prefill function in the
                // Tessera prefill namespace. Multifunction assets always use
                // their explicit function name above.
                if (!function_model && root_prefix == "tessera.ane.prefill" &&
                        compute_functions.size() == 1 && role == "prefill") {
                    function_model = program->model;
                    LOG_WRN("using default Core ML function for single-function ANE prefill %s\n",
                            function_name.c_str());
                }
                if (!function_model) {
                    LOG_WRN("failed to load embedded ANE function %s: %s\n",
                            function_name.c_str(),
                            function_error.localizedDescription.UTF8String ?: "unknown error");
                    continue;
                }
                auto instance = std::make_unique<common_ane_compute_instance>();
                instance->model = function_model;
                // Single execution_state (no keepalive_state). The
                // 5s re-warm timer is gone; the live state is the
                // only state. The dispatch path is unchanged in this
                // commit; future commits will migrate to pinned-state
                // slots + outputBackings.
                instance->execution_state = [function_model newState];
                instance->name = function_name;
                instance->role = role;
                instance->bucket = bucket;
                instance->warm.store(warm_model(
                    instance->model,
                    instance->execution_state,
                    program->batch_bucket,
                    function_name.c_str()));
                if (!instance->warm.load()) {
                    continue;
                }
                LOG_INF("loaded and warmed embedded ANE %s function %s\n",
                        role.c_str(), function_name.c_str());
                program->functions.emplace(function_name, std::move(instance));
            }
        }

        // No 5s re-warm timer. The .mlmodelc is loaded once and
        // stays loaded; state is in IOSurface (not in an opaque
        // MLState) and never "goes cold" in a way that requires
        // re-warming. The dispatch path uses execution_state for
        // now; future commits will replace it with pinned-state
        // slots + outputBackings (zero-copy across ANE/Metal/CPU).
        LOG_INF("loaded and warmed embedded ANE program in namespace %s at %s (requested batch=%u, bucket=%u)\n",
                root_prefix.c_str(),
                program->cache_path.c_str(), batch_hint, selected_batch);
        return program;
    }
}

common_ane_mtp_program_ptr common_ane_mtp_program_load(
        const std::string & gguf_path,
        uint32_t batch_hint) {
    return common_ane_program_load(gguf_path, batch_hint, "mtp.ane");
}

common_ane_prefill_program_ptr common_ane_prefill_program_load(
        const std::string & gguf_path,
        uint32_t sequence_hint) {
    common_ane_prefill_manifest manifest;
    if (!common_ane_prefill_manifest_load(gguf_path, &manifest) ||
            (sequence_hint != 0 &&
             std::find(manifest.sequence_buckets.begin(), manifest.sequence_buckets.end(), sequence_hint) ==
                 manifest.sequence_buckets.end())) {
        return nullptr;
    }
    auto program = common_ane_program_load(
        gguf_path, sequence_hint, "tessera.ane.prefill", manifest.batch_size);
    if (!program) {
        return nullptr;
    }
    const auto required_buckets = sequence_hint == 0
        ? manifest.sequence_buckets
        : std::vector<uint32_t>{ sequence_hint };
    for (const uint32_t bucket : required_buckets) {
        const std::string expected_function = "prefill_s" + std::to_string(bucket);
        const auto found = program->functions.find(expected_function);
        if (found == program->functions.end() || !found->second->warm.load()) {
            LOG_WRN("Tessera ANE prefill bundle lacks warm function %s\n", expected_function.c_str());
            return nullptr;
        }
    }
    return program;
}

bool common_ane_prefill_manifest_load(
        const std::string & gguf_path,
        common_ane_prefill_manifest * manifest) {
    if (!manifest) {
        return false;
    }
    struct gguf_init_params init = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
        gguf_init_from_file(gguf_path.c_str(), init), gguf_free);
    if (!ctx || gguf_string(ctx.get(), "tessera.ane.prefill.format") != "tessera-ane-prefill-v1") {
        return false;
    }
    const uint32_t abi_version = gguf_u32(ctx.get(), "tessera.ane.prefill.abi_version");
    const uint32_t hidden_size = gguf_u32(ctx.get(), "tessera.ane.prefill.hidden_size");
    const uint32_t layer_first = gguf_u32(ctx.get(), "tessera.ane.prefill.layer_first");
    const uint32_t layer_last = gguf_u32(ctx.get(), "tessera.ane.prefill.layer_last");
    const std::string architecture = gguf_string(ctx.get(), "tessera.ane.prefill.architecture");
    const std::string execution_stage = gguf_string(ctx.get(), "tessera.ane.prefill.execution_stage");
    const std::string hidden_layout = gguf_string(ctx.get(), "tessera.ane.prefill.hidden_layout");
    const std::string kv_layout = gguf_string(ctx.get(), "tessera.ane.prefill.kv_layout");
    const std::string cache_requirement = gguf_string(ctx.get(), "tessera.ane.prefill.cache_requirement");
    const uint32_t kv_heads = gguf_u32(ctx.get(), "tessera.ane.prefill.kv_heads");
    const uint32_t head_dim = gguf_u32(ctx.get(), "tessera.ane.prefill.head_dim");
    const uint32_t batch_size = gguf_u32(ctx.get(), "tessera.ane.prefill.batch_size");
    const int64_t padding_key = gguf_find_key(ctx.get(), "tessera.ane.prefill.causal_right_padding");
    const int64_t buckets_key = gguf_find_key(ctx.get(), "tessera.ane.prefill.sequence_buckets");
    if (abi_version != 1 || hidden_size == 0 || architecture.empty() ||
            execution_stage != "layer_slab" ||
            hidden_layout != "token_major.f32.v1" || kv_layout.empty() ||
            cache_requirement.empty() || kv_heads == 0 || head_dim == 0 || batch_size == 0 ||
            layer_last < layer_first ||
            buckets_key < 0 || gguf_get_kv_type(ctx.get(), buckets_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx.get(), buckets_key) != GGUF_TYPE_INT32) {
        return false;
    }
    const size_t count = gguf_get_arr_n(ctx.get(), buckets_key);
    const int32_t * buckets = (const int32_t *) gguf_get_arr_data(ctx.get(), buckets_key);
    if (count == 0 || !buckets) {
        return false;
    }
    manifest->abi_version = abi_version;
    manifest->hidden_size = hidden_size;
    manifest->layer_first = layer_first;
    manifest->layer_last = layer_last;
    manifest->architecture = architecture;
    manifest->execution_stage = execution_stage;
    manifest->hidden_layout = hidden_layout;
    manifest->kv_layout = kv_layout;
    manifest->cache_requirement = cache_requirement;
    manifest->kv_heads = kv_heads;
    manifest->head_dim = head_dim;
    manifest->batch_size = batch_size;
    manifest->causal_right_padding = padding_key >= 0 &&
        gguf_get_kv_type(ctx.get(), padding_key) == GGUF_TYPE_BOOL &&
        gguf_get_val_bool(ctx.get(), padding_key);
    manifest->sequence_buckets.clear();
    manifest->sequence_buckets.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (buckets[i] <= 0 || (i > 0 && buckets[i] <= buckets[i - 1])) {
            return false;
        }
        manifest->sequence_buckets.push_back((uint32_t) buckets[i]);
    }
    return true;
}

uint32_t common_ane_prefill_select_bucket(
        const common_ane_prefill_manifest & manifest,
        uint32_t n_tokens) {
    if (n_tokens == 0) {
        return 0;
    }
    const auto found = std::lower_bound(
        manifest.sequence_buckets.begin(), manifest.sequence_buckets.end(), n_tokens);
    if (found == manifest.sequence_buckets.end()) {
        return 0;
    }
    if (*found == n_tokens || manifest.causal_right_padding) {
        return *found;
    }
    return 0;
}

bool common_ane_mtp_program_is_warm(const common_ane_mtp_program_ptr & program) {
    return program && program->warm.load();
}

const char * common_ane_mtp_program_cache_path(const common_ane_mtp_program_ptr & program) {
    return program ? program->cache_path.c_str() : "";
}

common_ane_mtp_boundary_stats common_ane_mtp_program_boundary_stats(
        const common_ane_mtp_program_ptr & program) {
    if (!program) {
        return {};
    }
    return {
        program->direct_input_views.load(),
        program->direct_output_backings.load(),
        program->arena_input_bytes.load(),
        program->iosurface_arena_bytes.load(),
        program->copied_output_bytes.load(),
        program->async_prefill_submissions.load(),
        program->async_prefill_completions.load(),
        program->async_prefill_failures.load(),
    };
}

std::vector<common_ane_compute_function> common_ane_compute_functions(
        const common_ane_mtp_program_ptr & program) {
    std::vector<common_ane_compute_function> result;
    if (!program) {
        return result;
    }
    result.reserve(program->functions.size());
    for (const auto & entry : program->functions) {
        const auto & instance = *entry.second;
        result.push_back({
            instance.name,
            instance.role,
            instance.bucket,
            instance.warm.load(),
        });
    }
    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) {
        if (a.role != b.role) {
            return a.role < b.role;
        }
        return a.bucket < b.bucket;
    });
    return result;
}

static common_ane_compute_instance * find_compute_function(
        const common_ane_mtp_program_ptr & program,
        const char * role,
        uint32_t bucket) {
    if (!program) {
        return nullptr;
    }
    for (const auto & entry : program->functions) {
        auto * instance = entry.second.get();
        if (instance->role == role && instance->bucket == bucket &&
                instance->warm.load()) {
            return instance;
        }
    }
    return nullptr;
}

static MLMultiArray * make_i32_input(
        common_ane_mtp_program & program,
        const std::string & arena_name,
        NSArray<NSNumber *> * shape,
        const int32_t * source,
        size_t active_count,
        bool direct,
        NSError ** error) {
    if (direct) {
        return wrap_multi_array(
            (void *) source, shape, MLMultiArrayDataTypeInt32, error);
    }
    MLMultiArray * result = arena_multi_array(
        program, arena_name, shape, MLMultiArrayDataTypeInt32, error);
    if (!result) {
        return nil;
    }
    std::memcpy(result.dataPointer, source, active_count * sizeof(int32_t));
    std::memset(
        (int32_t *) result.dataPointer + active_count,
        0,
        (result.count - active_count) * sizeof(int32_t));
    program.arena_input_bytes += result.count * sizeof(int32_t);
    return result;
}

static bool valid_multi_array(
        MLFeatureDescription * description,
        MLMultiArrayDataType type,
        size_t count) {
    return description &&
        description.type == MLFeatureTypeMultiArray &&
        description.multiArrayConstraint.dataType == type &&
        shape_count(description.multiArrayConstraint.shape) == count;
}

bool common_ane_compute_prefill(
        const common_ane_mtp_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        float * hidden_states) {
    common_ane_compute_instance * instance =
        find_compute_function(program, "prefill", sequence_length);
    if (!instance || !token_ids || !positions || !hidden_states ||
            n_active == 0 || n_active > program->batch_bucket ||
            hidden_size == 0) {
        return false;
    }
    if (program->context_length > 0) {
        const size_t count = (size_t) n_active * sequence_length;
        for (size_t i = 0; i < count; ++i) {
            if (positions[i] < 0 ||
                    (uint32_t) positions[i] >= program->context_length) {
                return false;
            }
        }
    }

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSDictionary<NSString *, MLFeatureDescription *> * inputs_desc =
                instance->model.modelDescription.inputDescriptionsByName;
            NSDictionary<NSString *, MLFeatureDescription *> * outputs_desc =
                instance->model.modelDescription.outputDescriptionsByName;
            const size_t bucket_tokens =
                (size_t) program->batch_bucket * sequence_length;
            MLFeatureDescription * token_desc = inputs_desc[@"token_ids"];
            MLFeatureDescription * position_desc = inputs_desc[@"positions"];
            MLFeatureDescription * hidden_desc = outputs_desc[@"hidden_states"];
            if (!valid_multi_array(
                    token_desc, MLMultiArrayDataTypeInt32, bucket_tokens) ||
                    !valid_multi_array(
                        position_desc, MLMultiArrayDataTypeInt32, bucket_tokens) ||
                    !hidden_desc || hidden_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(hidden_desc.multiArrayConstraint.shape) !=
                        bucket_tokens * hidden_size) {
                return;
            }
            const MLMultiArrayDataType hidden_type =
                hidden_desc.multiArrayConstraint.dataType;
            if (hidden_type != MLMultiArrayDataTypeFloat16 &&
                    hidden_type != MLMultiArrayDataTypeFloat32) {
                return;
            }

            NSError * error = nil;
            const bool direct = n_active == program->batch_bucket;
            const size_t active_tokens = (size_t) n_active * sequence_length;
            MLMultiArray * tokens = make_i32_input(
                *program, instance->name + ".token_ids",
                token_desc.multiArrayConstraint.shape,
                token_ids, active_tokens, direct, &error);
            MLMultiArray * positions_array = make_i32_input(
                *program, instance->name + ".positions",
                position_desc.multiArrayConstraint.shape,
                positions, active_tokens, direct, &error);
            if (!tokens || !positions_array) {
                return;
            }
            if (direct) {
                program->direct_input_views += 2;
            }

            const bool direct_output =
                direct && hidden_type == MLMultiArrayDataTypeFloat32;
            MLMultiArray * output_backing = direct_output
                ? wrap_multi_array(
                    hidden_states,
                    hidden_desc.multiArrayConstraint.shape,
                    hidden_type,
                    &error)
                : arena_multi_array(
                    *program,
                    instance->name + ".hidden_states",
                    hidden_desc.multiArrayConstraint.shape,
                    hidden_type,
                    &error);
            if (!output_backing) {
                return;
            }
            MLDictionaryFeatureProvider * inputs =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{
                        @"token_ids": [MLFeatureValue featureValueWithMultiArray:tokens],
                        @"positions": [MLFeatureValue featureValueWithMultiArray:positions_array],
                    }
                    error:&error];
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            options.outputBackings = @{@"hidden_states": output_backing};
            id<MLFeatureProvider> output = instance->execution_state
                ? [instance->model predictionFromFeatures:inputs
                                               usingState:instance->execution_state
                                                  options:options
                                                    error:&error]
                : [instance->model predictionFromFeatures:inputs
                                                  options:options
                                                    error:&error];
            MLMultiArray * result =
                [output featureValueForName:@"hidden_states"].multiArrayValue;
            if (!output || !result ||
                    (size_t) result.count < active_tokens * hidden_size ||
                    result.dataType != hidden_type) {
                LOG_WRN("ANE prefill %s failed: %s\n",
                        instance->name.c_str(),
                        error.localizedDescription.UTF8String ?: "invalid output");
                return;
            }
            if (result.dataPointer == output_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            if (!direct_output) {
                read_float_data(
                    result, hidden_states, active_tokens * hidden_size);
                program->copied_output_bytes +=
                    active_tokens * hidden_size * sizeof(float);
            }
            ok = true;
        }
    });
    return ok;
}

bool common_ane_compute_prefill_slab(
        const common_ane_prefill_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        uint32_t kv_heads,
        uint32_t head_dim,
        float * hidden_states,
        float * key_states,
        float * value_states) {
    common_ane_compute_instance * instance =
        find_compute_function(program, "prefill", sequence_length);
    if (!instance || !token_ids || !positions || !hidden_states || !key_states ||
            !value_states || n_active == 0 || n_active > program->batch_bucket ||
            hidden_size == 0 || kv_heads == 0 || head_dim == 0) {
        return false;
    }
    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            const size_t bucket_tokens = (size_t) program->batch_bucket * sequence_length;
            const size_t active_tokens = (size_t) n_active * sequence_length;
            const size_t kv_width = (size_t) kv_heads * head_dim;
            const auto & inputs_desc = instance->model.modelDescription.inputDescriptionsByName;
            const auto & outputs_desc = instance->model.modelDescription.outputDescriptionsByName;
            MLFeatureDescription * token_desc = inputs_desc[@"token_ids"];
            MLFeatureDescription * pos_desc = inputs_desc[@"positions"];
            MLFeatureDescription * hidden_desc = outputs_desc[@"hidden_states"];
            MLFeatureDescription * key_desc = outputs_desc[@"key_states"];
            MLFeatureDescription * value_desc = outputs_desc[@"value_states"];
            if (!valid_multi_array(token_desc, MLMultiArrayDataTypeInt32, bucket_tokens) ||
                    !valid_multi_array(pos_desc, MLMultiArrayDataTypeInt32, bucket_tokens) ||
                    !hidden_desc || !key_desc || !value_desc ||
                    hidden_desc.type != MLFeatureTypeMultiArray ||
                    key_desc.type != MLFeatureTypeMultiArray ||
                    value_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(hidden_desc.multiArrayConstraint.shape) != bucket_tokens * hidden_size ||
                    shape_count(key_desc.multiArrayConstraint.shape) != bucket_tokens * kv_width ||
                    shape_count(value_desc.multiArrayConstraint.shape) != bucket_tokens * kv_width) {
                return;
            }
            const MLMultiArrayDataType hidden_type = hidden_desc.multiArrayConstraint.dataType;
            const MLMultiArrayDataType key_type = key_desc.multiArrayConstraint.dataType;
            const MLMultiArrayDataType value_type = value_desc.multiArrayConstraint.dataType;
            if ((hidden_type != MLMultiArrayDataTypeFloat16 && hidden_type != MLMultiArrayDataTypeFloat32) ||
                    (key_type != MLMultiArrayDataTypeFloat16 && key_type != MLMultiArrayDataTypeFloat32) ||
                    (value_type != MLMultiArrayDataTypeFloat16 && value_type != MLMultiArrayDataTypeFloat32)) {
                return;
            }
            NSError * error = nil;
            const std::string prefix = instance->name + ".slab";
            MLMultiArray * tokens = make_i32_input(*program, prefix + ".token_ids",
                    token_desc.multiArrayConstraint.shape, token_ids, active_tokens, false, &error);
            MLMultiArray * pos = make_i32_input(*program, prefix + ".positions",
                    pos_desc.multiArrayConstraint.shape, positions, active_tokens, false, &error);
            if (!tokens || !pos) {
                return;
            }
            NSMutableDictionary<NSString *, MLFeatureValue *> * features =
                [NSMutableDictionary dictionaryWithDictionary:@{
                    @"token_ids": [MLFeatureValue featureValueWithMultiArray:tokens],
                    @"positions": [MLFeatureValue featureValueWithMultiArray:pos],
                }];

            // Layer weights + per-token embedded values arrive as MLMultiArray
            // inputs.  The weight arrays were created at bundle load time from
            // the source GGUF.  The embedded array is a one-shot gather of
            // token rows from the cached embedding table; it lives in the
            // arena for the duration of this prediction.
            MLMultiArray * embedded = nil;
            if (inputs_desc[@"embedded"] != nil) {
                if (program->embedding.empty() || program->embedding_dim != hidden_size) {
                    return;
                }
                MLFeatureDescription * emb_desc = inputs_desc[@"embedded"];
                if (emb_desc.type != MLFeatureTypeMultiArray ||
                        emb_desc.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) {
                    return;
                }
                NSArray<NSNumber *> * emb_shape = emb_desc.multiArrayConstraint.shape;
                NSMutableArray<NSNumber *> * ns_emb_shape = [emb_shape mutableCopy];
                const std::string emb_name = prefix + ".embedded";
                embedded = arena_multi_array(*program, emb_name, ns_emb_shape,
                        MLMultiArrayDataTypeFloat16, &error);
                if (!embedded) {
                    return;
                }
                uint16_t * dst = (uint16_t *) embedded.dataPointer;
                const uint16_t * vocab = program->embedding.data();
                for (size_t i = 0; i < active_tokens; ++i) {
                    const int32_t tok = token_ids[i];
                    if ((size_t) tok * hidden_size + hidden_size > program->embedding.size()) {
                        return;
                    }
                    std::memcpy(dst + i * hidden_size,
                            vocab + (size_t) tok * hidden_size,
                            hidden_size * sizeof(uint16_t));
                }
                features[@"embedded"] = [MLFeatureValue featureValueWithMultiArray:embedded];
            }
            for (auto & entry : program->weight_inputs) {
                NSString * weight_name = [NSString stringWithUTF8String:entry.first.c_str()];
                features[weight_name] = [MLFeatureValue featureValueWithMultiArray:entry.second];
            }

            MLDictionaryFeatureProvider * inputs = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:features error:&error];
            if (!inputs) {
                return;
            }
            // Core ML 15 accepts multi-output backings, but some ANE drivers
            // return silently sparse data for rank-four output backings.  Use
            // materialized outputs until the backing contract is qualified by
            // the parity test; inputs remain in the IOSurface arena.
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            id<MLFeatureProvider> output = [instance->model predictionFromFeatures:inputs
                                                                            options:options
                                                                              error:&error];
            MLMultiArray * out_hidden = [output featureValueForName:@"hidden_states"].multiArrayValue;
            MLMultiArray * out_keys = [output featureValueForName:@"key_states"].multiArrayValue;
            MLMultiArray * out_values = [output featureValueForName:@"value_states"].multiArrayValue;
            if (!output || !out_hidden || !out_keys || !out_values ||
                    (size_t) out_hidden.count < active_tokens * hidden_size ||
                    (size_t) out_keys.count < active_tokens * kv_width ||
                    (size_t) out_values.count < active_tokens * kv_width) {
                LOG_WRN("ANE prefill slab %s failed: %s\n", instance->name.c_str(),
                        error.localizedDescription.UTF8String ?: "invalid output");
                return;
            }
            read_float_data(out_hidden, hidden_states, active_tokens * hidden_size);
            read_float_data(out_keys, key_states, active_tokens * kv_width);
            read_float_data(out_values, value_states, active_tokens * kv_width);
            program->copied_output_bytes += active_tokens *
                (hidden_size + 2 * kv_width) * sizeof(float);
            ok = true;
        }
    });
    return ok;
}

bool common_ane_prefill_decode(
        const common_ane_prefill_program_ptr & program,
        const common_ane_prefill_manifest & manifest,
        llama_context * ctx,
        const llama_batch & batch,
        int32_t * result) {
    if (result) {
        *result = -1;
    }
    const uint32_t n_tokens = batch.n_tokens > 0 ? (uint32_t) batch.n_tokens : 0;
    const uint32_t sequence = common_ane_prefill_select_bucket(manifest, n_tokens);
    if (!program || !ctx || !batch.token || !batch.pos || sequence == 0 ||
            manifest.execution_stage != "layer_slab" || manifest.layer_first != 0 ||
            manifest.layer_last != 0 || manifest.cache_requirement != "empty_contiguous_prompt" ||
            manifest.hidden_layout != "token_major.f32.v1" || manifest.batch_size != 1 ||
            llama_n_ubatch(ctx) < n_tokens) {
        return false;
    }
    std::vector<int32_t> tokens(sequence);
    std::vector<int32_t> positions(sequence);
    const llama_seq_id seq_id = batch.n_seq_id ? batch.seq_id[0][0] : 0;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        if (batch.pos[i] != (llama_pos) i ||
                (batch.n_seq_id && (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq_id))) {
            return false;
        }
        tokens[i] = batch.token[i];
        positions[i] = batch.pos[i];
    }
    // A declared causal-right-padding bucket may contain harmless future rows.
    // They cannot affect earlier causal attention rows and are intentionally
    // excluded from the imported KV prefix and continuation batch.
    for (uint32_t i = n_tokens; i < sequence; ++i) {
        tokens[i] = 0;
        positions[i] = (int32_t) i;
    }
    const size_t hidden_count = (size_t) sequence * manifest.hidden_size;
    const size_t kv_count = (size_t) sequence * manifest.kv_heads * manifest.head_dim;
    std::vector<float> hidden(hidden_count);
    std::vector<float> keys(kv_count);
    std::vector<float> values(kv_count);
    if (!common_ane_compute_prefill_slab(program, sequence, tokens.data(), positions.data(), 1,
            manifest.hidden_size, manifest.kv_heads, manifest.head_dim,
            hidden.data(), keys.data(), values.data())) {
        return false;
    }
    if (!llama_set_ane_prefill_result(ctx, 1, n_tokens, manifest.kv_heads,
            manifest.head_dim, keys.data(), values.data())) {
        return false;
    }
    llama_batch continuation = batch;
    continuation.embd = hidden.data();
    const int32_t decode_result = llama_decode(ctx, continuation);
    if (result) {
        *result = decode_result;
    }
    return true;
}

// Fast-start path for prompts longer than any declared bucket.  The ANE slab
// can only attend within its own bucket, so cross-bucket chunking would
// silently drop attention to earlier rows.  Instead we run a single ANE slab
// over the first `max_bucket` tokens, import its K/V into the cache, and
// then continue with an ordinary llama_decode over the remaining tail.  The
// tail still benefits from the imported K/V for layer 0 and runs the rest
// of the model normally, so the total work is one ANE call plus a standard
// prefill of the tail instead of a full standard prefill.
//
// The function returns true when the ANE handoff succeeded and the
// continuation is in flight.  A false return means no context mutation
// occurred; callers must use the ordinary Metal decode path.  When true,
// the first `max_bucket` tokens of `batch` are processed by the ANE path,
// the remaining tokens (if any) are processed by the second llama_decode,
// and `result` (if non-null) holds the final llama_decode status.  The
// caller must NOT retry the same batch through Metal.
bool common_ane_prefill_decode_chunked(
        const common_ane_prefill_program_ptr & program,
        const common_ane_prefill_manifest & manifest,
        llama_context * ctx,
        const llama_batch & batch,
        int32_t * result) {
    if (result) {
        *result = -1;
    }
    const uint32_t n_tokens = batch.n_tokens > 0 ? (uint32_t) batch.n_tokens : 0;
    if (n_tokens == 0 || !program || !ctx || !batch.token ||
            manifest.sequence_buckets.empty() ||
            manifest.execution_stage != "layer_slab" || manifest.layer_first != 0 ||
            manifest.layer_last != 0 || manifest.cache_requirement != "empty_contiguous_prompt" ||
            manifest.hidden_layout != "token_major.f32.v1" || manifest.batch_size != 1) {
        return false;
    }
    const uint32_t max_bucket = manifest.sequence_buckets.back();
    if (n_tokens <= max_bucket) {
        // A short prompt still fits in a single bucket; defer to the
        // strict prefill path so the K/V import contract is honored exactly.
        return common_ane_prefill_decode(program, manifest, ctx, batch, result);
    }
    if (llama_n_ubatch(ctx) < max_bucket) {
        return false;
    }
    const llama_seq_id seq_id = batch.n_seq_id ? batch.seq_id[0][0] : 0;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_pos p = batch.pos ? batch.pos[i] : (llama_pos) i;
        if (p != (llama_pos) i) {
            return false;
        }
        if (batch.n_seq_id && (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq_id)) {
            return false;
        }
    }
    // Phase 1: ANE prefill over the first max_bucket tokens with causal
    // right-padding.  We invoke the same compute path as the single-bucket
    // helper so the K/V import contract is identical.
    const uint32_t sequence = max_bucket;
    std::vector<int32_t> tokens(sequence);
    std::vector<int32_t> positions(sequence);
    for (uint32_t i = 0; i < sequence; ++i) {
        tokens[i] = i < n_tokens ? batch.token[i] : 0;
        positions[i] = (int32_t) i;
    }
    const size_t hidden_count = (size_t) sequence * manifest.hidden_size;
    const size_t kv_count = (size_t) sequence * manifest.kv_heads * manifest.head_dim;
    std::vector<float> hidden(hidden_count);
    std::vector<float> keys(kv_count);
    std::vector<float> values(kv_count);
    if (!common_ane_compute_prefill_slab(program, sequence, tokens.data(), positions.data(), 1,
            manifest.hidden_size, manifest.kv_heads, manifest.head_dim,
            hidden.data(), keys.data(), values.data())) {
        return false;
    }
    if (!llama_set_ane_prefill_result(ctx, 1, sequence, manifest.kv_heads,
            manifest.head_dim, keys.data(), values.data())) {
        return false;
    }
    // Build the head batch view.  We cannot trivially slice the source
    // llama_batch because its array pointers are owned by the caller, so
    // we build a fresh batch that aliases the same per-row arrays.
    llama_batch head_batch = {};
    head_batch.n_tokens = (int32_t) sequence;
    head_batch.token = batch.token;
    head_batch.pos = batch.pos;
    head_batch.n_seq_id = batch.n_seq_id;
    head_batch.seq_id = batch.seq_id;
    head_batch.logits = batch.logits;
    head_batch.embd = hidden.data();
    const int32_t head_result = llama_decode(ctx, head_batch);
    if (head_result != 0) {
        if (result) {
            *result = head_result;
        }
        return true;
    }
    // Phase 2: ordinary prefill for the tail.  The K/V cache already holds
    // the first max_bucket rows for layer 0 (imported by the head call);
    // the tail call extends them with the remaining positions.  No ANE
    // involvement and no embeddings override — llama_decode will use the
    // normal embedding table for these tokens.
    if (n_tokens > sequence) {
        const uint32_t tail = n_tokens - sequence;
        if (llama_n_ubatch(ctx) < (int32_t) tail) {
            // Cannot fit the tail in a single ubatch; bail so the caller
            // falls back to the all-Metal path which can split across
            // multiple llama_decode calls.  The first head_result already
            // mutated context state, so this is a soft failure and the
            // caller should NOT treat it as a clean return.
            if (result) {
                *result = -1;
            }
            return true;
        }
        llama_batch tail_batch = {};
        tail_batch.n_tokens = (int32_t) tail;
        tail_batch.token = batch.token ? batch.token + sequence : nullptr;
        tail_batch.pos = batch.pos ? batch.pos + sequence : nullptr;
        tail_batch.n_seq_id = batch.n_seq_id ? batch.n_seq_id + sequence : nullptr;
        tail_batch.seq_id = batch.seq_id ? batch.seq_id + sequence : nullptr;
        tail_batch.logits = batch.logits ? batch.logits + sequence : nullptr;
        const int32_t tail_result = llama_decode(ctx, tail_batch);
        if (result) {
            *result = tail_result;
        }
    } else if (result) {
        *result = head_result;
    }
    return true;
}

common_ane_prefill_request_ptr common_ane_compute_prefill_async(
        const common_ane_prefill_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        float * hidden_states) {
    if (!program || !token_ids || !positions || !hidden_states ||
            n_active == 0 || sequence_length == 0 || hidden_size == 0) {
        return nullptr;
    }
    auto request = std::make_shared<common_ane_prefill_request>();
    request->program = program;
    request->completion = dispatch_semaphore_create(0);
    request->arena_epoch = program->next_prefill_arena_epoch.fetch_add(1) + 1;
    ++program->async_prefill_submissions;
    dispatch_async(program->queue, ^{
        @autoreleasepool {
            common_ane_compute_instance * instance =
                find_compute_function(request->program, "prefill", sequence_length);
            const size_t active_tokens = (size_t) n_active * sequence_length;
            const size_t bucket_tokens =
                (size_t) request->program->batch_bucket * sequence_length;
            const auto finish = ^(bool success) {
                ++request->program->async_prefill_completions;
                if (!success) {
                    ++request->program->async_prefill_failures;
                }
                request->success.store(success);
                request->complete.store(true);
                dispatch_semaphore_signal(request->completion);
            };
            if (!instance || request->program->context_length > 0) {
                for (size_t i = 0; instance && i < active_tokens; ++i) {
                    if (positions[i] < 0 ||
                            (uint32_t) positions[i] >= request->program->context_length) {
                        finish(false);
                        return;
                    }
                }
            }
            if (!instance) {
                finish(false);
                return;
            }
            NSDictionary<NSString *, MLFeatureDescription *> * input_desc =
                instance->model.modelDescription.inputDescriptionsByName;
            NSDictionary<NSString *, MLFeatureDescription *> * output_desc =
                instance->model.modelDescription.outputDescriptionsByName;
            MLFeatureDescription * token_desc = input_desc[@"token_ids"];
            MLFeatureDescription * position_desc = input_desc[@"positions"];
            MLFeatureDescription * hidden_desc = output_desc[@"hidden_states"];
            if (!valid_multi_array(token_desc, MLMultiArrayDataTypeInt32, bucket_tokens) ||
                    !valid_multi_array(position_desc, MLMultiArrayDataTypeInt32, bucket_tokens) ||
                    !hidden_desc || hidden_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(hidden_desc.multiArrayConstraint.shape) !=
                        bucket_tokens * hidden_size) {
                finish(false);
                return;
            }
            const MLMultiArrayDataType hidden_type = hidden_desc.multiArrayConstraint.dataType;
            if (hidden_type != MLMultiArrayDataTypeFloat16 &&
                    hidden_type != MLMultiArrayDataTypeFloat32) {
                finish(false);
                return;
            }
            NSError * error = nil;
            // Inputs are copied into per-request IOSurface slots.  This makes
            // completion-handler execution safe even if the scheduler recycles
            // its prompt buffers immediately after submitting the request.
            const std::string epoch = ".async." + std::to_string(request->arena_epoch);
            MLMultiArray * tokens = make_i32_input(
                *request->program, instance->name + epoch + ".token_ids",
                token_desc.multiArrayConstraint.shape, token_ids, active_tokens,
                false, &error);
            MLMultiArray * positions_array = make_i32_input(
                *request->program, instance->name + epoch + ".positions",
                position_desc.multiArrayConstraint.shape, positions, active_tokens,
                false, &error);
            MLMultiArray * output_backing = arena_multi_array(
                *request->program, instance->name + epoch + ".hidden_states",
                hidden_desc.multiArrayConstraint.shape, hidden_type, &error);
            if (!tokens || !positions_array || !output_backing) {
                finish(false);
                return;
            }
            MLDictionaryFeatureProvider * inputs = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{
                    @"token_ids": [MLFeatureValue featureValueWithMultiArray:tokens],
                    @"positions": [MLFeatureValue featureValueWithMultiArray:positions_array],
                } error:&error];
            if (!inputs) {
                finish(false);
                return;
            }
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            options.outputBackings = @{@"hidden_states": output_backing};
            [instance->model predictionFromFeatures:inputs
                                            options:options
                                  completionHandler:^(id<MLFeatureProvider> output, NSError * completion_error) {
                @autoreleasepool {
                    MLMultiArray * result =
                        [output featureValueForName:@"hidden_states"].multiArrayValue;
                    const bool valid = output && !completion_error && result &&
                        result.dataType == hidden_type &&
                        (size_t) result.count >= active_tokens * hidden_size;
                    if (valid) {
                        if (result.dataPointer == output_backing.dataPointer) {
                            ++request->program->direct_output_backings;
                        }
                        read_float_data(result, hidden_states, active_tokens * hidden_size);
                        request->program->copied_output_bytes +=
                            active_tokens * hidden_size * sizeof(float);
                    } else {
                        LOG_WRN("ANE async prefill %s failed: %s\n", instance->name.c_str(),
                                completion_error.localizedDescription.UTF8String ?: "invalid output");
                    }
                    finish(valid);
                }
            }];
        }
    });
    return request;
}

bool common_ane_prefill_request_is_complete(
        const common_ane_prefill_request_ptr & request) {
    return request && request->complete.load();
}

bool common_ane_prefill_request_wait(
        const common_ane_prefill_request_ptr & request) {
    if (!request || !request->completion) {
        return false;
    }
    if (!request->complete.load()) {
        dispatch_semaphore_wait(request->completion, DISPATCH_TIME_FOREVER);
    }
    return request->success.load();
}

bool common_ane_compute_dflash(
        const common_ane_mtp_program_ptr & program,
        uint32_t block_size,
        const float * target_features,
        uint32_t n_active,
        uint32_t feature_width,
        const int32_t * token_ids,
        const int32_t * positions,
        int32_t * draft_tokens,
        float * confidence) {
    common_ane_compute_instance * instance =
        find_compute_function(program, "dflash", block_size);
    if (!instance || !target_features || !token_ids || !positions ||
            !draft_tokens || !confidence || n_active == 0 ||
            n_active > program->batch_bucket || feature_width == 0) {
        return false;
    }
    if (program->context_length > 0) {
        for (uint32_t lane = 0; lane < n_active; ++lane) {
            if (positions[lane] < 0 ||
                    (uint32_t) positions[lane] >= program->context_length) {
                return false;
            }
        }
    }

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSDictionary<NSString *, MLFeatureDescription *> * inputs_desc =
                instance->model.modelDescription.inputDescriptionsByName;
            NSDictionary<NSString *, MLFeatureDescription *> * outputs_desc =
                instance->model.modelDescription.outputDescriptionsByName;
            const size_t lane_bucket = program->batch_bucket;
            MLFeatureDescription * features_desc = inputs_desc[@"target_features"];
            MLFeatureDescription * token_desc = inputs_desc[@"token_ids"];
            MLFeatureDescription * position_desc = inputs_desc[@"positions"];
            MLFeatureDescription * drafts_desc = outputs_desc[@"draft_tokens"];
            MLFeatureDescription * confidence_desc = outputs_desc[@"confidence"];
            if (!features_desc || features_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(features_desc.multiArrayConstraint.shape) !=
                        lane_bucket * feature_width ||
                    !valid_multi_array(
                        token_desc, MLMultiArrayDataTypeInt32, lane_bucket) ||
                    !valid_multi_array(
                        position_desc, MLMultiArrayDataTypeInt32, lane_bucket) ||
                    !valid_multi_array(
                        drafts_desc, MLMultiArrayDataTypeInt32,
                        lane_bucket * block_size) ||
                    !confidence_desc ||
                    confidence_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(confidence_desc.multiArrayConstraint.shape) !=
                        lane_bucket * block_size) {
                return;
            }
            const MLMultiArrayDataType features_type =
                features_desc.multiArrayConstraint.dataType;
            const MLMultiArrayDataType confidence_type =
                confidence_desc.multiArrayConstraint.dataType;
            if ((features_type != MLMultiArrayDataTypeFloat16 &&
                        features_type != MLMultiArrayDataTypeFloat32) ||
                    (confidence_type != MLMultiArrayDataTypeFloat16 &&
                        confidence_type != MLMultiArrayDataTypeFloat32)) {
                return;
            }

            NSError * error = nil;
            const bool direct = n_active == lane_bucket;
            MLMultiArray * features =
                direct && features_type == MLMultiArrayDataTypeFloat32
                ? wrap_multi_array(
                    (void *) target_features,
                    features_desc.multiArrayConstraint.shape,
                    features_type,
                    &error)
                : arena_multi_array(
                    *program,
                    instance->name + ".target_features",
                    features_desc.multiArrayConstraint.shape,
                    features_type,
                    &error);
            MLMultiArray * tokens = make_i32_input(
                *program, instance->name + ".token_ids",
                token_desc.multiArrayConstraint.shape,
                token_ids, n_active, direct, &error);
            MLMultiArray * positions_array = make_i32_input(
                *program, instance->name + ".positions",
                position_desc.multiArrayConstraint.shape,
                positions, n_active, direct, &error);
            if (!features || !tokens || !positions_array) {
                return;
            }
            const size_t active_features = (size_t) n_active * feature_width;
            if (!(direct && features_type == MLMultiArrayDataTypeFloat32)) {
                write_float_data(features, target_features, active_features);
                std::memset(
                    (char *) features.dataPointer +
                        active_features * multi_array_element_size(features_type),
                    0,
                    (lane_bucket * feature_width - active_features) *
                        multi_array_element_size(features_type));
                program->arena_input_bytes +=
                    lane_bucket * feature_width *
                    multi_array_element_size(features_type);
            } else {
                ++program->direct_input_views;
            }
            if (direct) {
                program->direct_input_views += 2;
            }

            const bool direct_confidence =
                direct && confidence_type == MLMultiArrayDataTypeFloat32;
            MLMultiArray * draft_backing = direct
                ? wrap_multi_array(
                    draft_tokens,
                    drafts_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32,
                    &error)
                : arena_multi_array(
                    *program,
                    instance->name + ".draft_tokens",
                    drafts_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32,
                    &error);
            MLMultiArray * confidence_backing = direct_confidence
                ? wrap_multi_array(
                    confidence,
                    confidence_desc.multiArrayConstraint.shape,
                    confidence_type,
                    &error)
                : arena_multi_array(
                    *program,
                    instance->name + ".confidence",
                    confidence_desc.multiArrayConstraint.shape,
                    confidence_type,
                    &error);
            if (!draft_backing || !confidence_backing) {
                return;
            }
            MLDictionaryFeatureProvider * inputs =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{
                        @"target_features": [MLFeatureValue featureValueWithMultiArray:features],
                        @"token_ids": [MLFeatureValue featureValueWithMultiArray:tokens],
                        @"positions": [MLFeatureValue featureValueWithMultiArray:positions_array],
                    }
                    error:&error];
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            options.outputBackings = @{
                @"draft_tokens": draft_backing,
                @"confidence": confidence_backing,
            };
            id<MLFeatureProvider> output = instance->execution_state
                ? [instance->model predictionFromFeatures:inputs
                                               usingState:instance->execution_state
                                                  options:options
                                                    error:&error]
                : [instance->model predictionFromFeatures:inputs
                                                  options:options
                                                    error:&error];
            MLMultiArray * drafts =
                [output featureValueForName:@"draft_tokens"].multiArrayValue;
            MLMultiArray * probabilities =
                [output featureValueForName:@"confidence"].multiArrayValue;
            const size_t active_drafts = (size_t) n_active * block_size;
            if (!output || !drafts || !probabilities ||
                    drafts.dataType != MLMultiArrayDataTypeInt32 ||
                    probabilities.dataType != confidence_type ||
                    (size_t) drafts.count < active_drafts ||
                    (size_t) probabilities.count < active_drafts) {
                LOG_WRN("ANE DFlash %s failed: %s\n",
                        instance->name.c_str(),
                        error.localizedDescription.UTF8String ?: "invalid output");
                return;
            }
            if (!direct) {
                std::memcpy(
                    draft_tokens, drafts.dataPointer,
                    active_drafts * sizeof(int32_t));
                program->copied_output_bytes +=
                    active_drafts * sizeof(int32_t);
            }
            if (!direct_confidence) {
                read_float_data(probabilities, confidence, active_drafts);
                program->copied_output_bytes +=
                    active_drafts * sizeof(float);
            }
            if (drafts.dataPointer == draft_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            if (probabilities.dataPointer == confidence_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            ok = true;
        }
    });
    return ok;
}

bool common_ane_compute_hybrid(
        const common_ane_mtp_program_ptr & program,
        uint32_t block_size,
        const int32_t * dflash_tokens,
        const float * dflash_confidence,
        const int32_t * dflash_counts,
        const int32_t * mtp_tokens,
        const float * mtp_confidence,
        const int32_t * mtp_counts,
        uint32_t n_active,
        float dflash_cutoff,
        int32_t * selected_source,
        int32_t * agreement) {
    common_ane_compute_instance * instance =
        find_compute_function(program, "hybrid", block_size);
    if (!instance || !dflash_tokens || !dflash_confidence || !dflash_counts ||
            !mtp_tokens || !mtp_confidence || !mtp_counts ||
            !selected_source || !agreement || n_active == 0 ||
            n_active > program->batch_bucket) {
        return false;
    }

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSDictionary<NSString *, MLFeatureDescription *> * inputs_desc =
                instance->model.modelDescription.inputDescriptionsByName;
            NSDictionary<NSString *, MLFeatureDescription *> * outputs_desc =
                instance->model.modelDescription.outputDescriptionsByName;
            const size_t lane_bucket = program->batch_bucket;
            const size_t block_count = lane_bucket * block_size;

            MLFeatureDescription * d_tokens_desc = inputs_desc[@"dflash_tokens"];
            MLFeatureDescription * d_conf_desc = inputs_desc[@"dflash_confidence"];
            MLFeatureDescription * d_counts_desc = inputs_desc[@"dflash_counts"];
            MLFeatureDescription * m_tokens_desc = inputs_desc[@"mtp_tokens"];
            MLFeatureDescription * m_conf_desc = inputs_desc[@"mtp_confidence"];
            MLFeatureDescription * m_counts_desc = inputs_desc[@"mtp_counts"];
            MLFeatureDescription * cutoff_desc = inputs_desc[@"dflash_cutoff"];
            MLFeatureDescription * source_desc = outputs_desc[@"selected_source"];
            MLFeatureDescription * agreement_desc = outputs_desc[@"agreement"];
            if (!valid_multi_array(d_tokens_desc, MLMultiArrayDataTypeInt32, block_count) ||
                    !valid_multi_array(d_counts_desc, MLMultiArrayDataTypeInt32, lane_bucket) ||
                    !valid_multi_array(m_tokens_desc, MLMultiArrayDataTypeInt32, block_count) ||
                    !valid_multi_array(m_counts_desc, MLMultiArrayDataTypeInt32, lane_bucket) ||
                    !valid_multi_array(source_desc, MLMultiArrayDataTypeInt32, lane_bucket) ||
                    !valid_multi_array(agreement_desc, MLMultiArrayDataTypeInt32, lane_bucket) ||
                    !cutoff_desc || cutoff_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(cutoff_desc.multiArrayConstraint.shape) != lane_bucket ||
                    !d_conf_desc || d_conf_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(d_conf_desc.multiArrayConstraint.shape) != block_count ||
                    !m_conf_desc || m_conf_desc.type != MLFeatureTypeMultiArray ||
                    shape_count(m_conf_desc.multiArrayConstraint.shape) != block_count) {
                return;
            }
            const MLMultiArrayDataType d_conf_type =
                d_conf_desc.multiArrayConstraint.dataType;
            const MLMultiArrayDataType m_conf_type =
                m_conf_desc.multiArrayConstraint.dataType;
            const MLMultiArrayDataType cutoff_type =
                cutoff_desc.multiArrayConstraint.dataType;
            if ((d_conf_type != MLMultiArrayDataTypeFloat16 &&
                        d_conf_type != MLMultiArrayDataTypeFloat32) ||
                    (m_conf_type != MLMultiArrayDataTypeFloat16 &&
                        m_conf_type != MLMultiArrayDataTypeFloat32) ||
                    (cutoff_type != MLMultiArrayDataTypeFloat16 &&
                        cutoff_type != MLMultiArrayDataTypeFloat32)) {
                return;
            }

            NSError * error = nil;
            const bool direct = n_active == lane_bucket;
            const size_t active_blocks = (size_t) n_active * block_size;
            MLMultiArray * d_tokens = make_i32_input(
                *program, instance->name + ".dflash_tokens",
                d_tokens_desc.multiArrayConstraint.shape,
                dflash_tokens, active_blocks, direct, &error);
            MLMultiArray * d_counts = make_i32_input(
                *program, instance->name + ".dflash_counts",
                d_counts_desc.multiArrayConstraint.shape,
                dflash_counts, n_active, direct, &error);
            MLMultiArray * m_tokens = make_i32_input(
                *program, instance->name + ".mtp_tokens",
                m_tokens_desc.multiArrayConstraint.shape,
                mtp_tokens, active_blocks, direct, &error);
            MLMultiArray * m_counts = make_i32_input(
                *program, instance->name + ".mtp_counts",
                m_counts_desc.multiArrayConstraint.shape,
                mtp_counts, n_active, direct, &error);
            auto make_confidence = [&](const std::string & name,
                                       MLFeatureDescription * desc,
                                       MLMultiArrayDataType type,
                                       const float * values) -> MLMultiArray * {
                MLMultiArray * array = arena_multi_array(
                    *program, name, desc.multiArrayConstraint.shape, type, &error);
                if (array) {
                    write_float_data(array, values, active_blocks);
                    if (active_blocks < block_count) {
                        std::memset(
                            (char *) array.dataPointer +
                                active_blocks * multi_array_element_size(type),
                            0,
                            (block_count - active_blocks) *
                                multi_array_element_size(type));
                    }
                    program->arena_input_bytes +=
                        block_count * multi_array_element_size(type);
                }
                return array;
            };
            MLMultiArray * d_conf = make_confidence(
                instance->name + ".dflash_confidence",
                d_conf_desc, d_conf_type, dflash_confidence);
            MLMultiArray * m_conf = make_confidence(
                instance->name + ".mtp_confidence",
                m_conf_desc, m_conf_type, mtp_confidence);
            MLMultiArray * cutoff = arena_multi_array(
                *program, instance->name + ".dflash_cutoff",
                cutoff_desc.multiArrayConstraint.shape,
                cutoff_type, &error);
            if (!d_tokens || !d_counts || !m_tokens || !m_counts ||
                    !d_conf || !m_conf || !cutoff) {
                return;
            }
            std::vector<float> cutoff_values(lane_bucket, dflash_cutoff);
            write_float_data(cutoff, cutoff_values.data(), lane_bucket);
            program->arena_input_bytes +=
                lane_bucket * multi_array_element_size(cutoff_type);
            if (direct) {
                program->direct_input_views += 4;
            }

            MLMultiArray * source_backing = direct
                ? wrap_multi_array(
                    selected_source,
                    source_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32, &error)
                : arena_multi_array(
                    *program, instance->name + ".selected_source",
                    source_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32, &error);
            MLMultiArray * agreement_backing = direct
                ? wrap_multi_array(
                    agreement,
                    agreement_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32, &error)
                : arena_multi_array(
                    *program, instance->name + ".agreement",
                    agreement_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32, &error);
            if (!source_backing || !agreement_backing) {
                return;
            }

            MLDictionaryFeatureProvider * inputs =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{
                        @"dflash_tokens": [MLFeatureValue featureValueWithMultiArray:d_tokens],
                        @"dflash_confidence": [MLFeatureValue featureValueWithMultiArray:d_conf],
                        @"dflash_counts": [MLFeatureValue featureValueWithMultiArray:d_counts],
                        @"mtp_tokens": [MLFeatureValue featureValueWithMultiArray:m_tokens],
                        @"mtp_confidence": [MLFeatureValue featureValueWithMultiArray:m_conf],
                        @"mtp_counts": [MLFeatureValue featureValueWithMultiArray:m_counts],
                        @"dflash_cutoff": [MLFeatureValue featureValueWithMultiArray:cutoff],
                    }
                    error:&error];
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            options.outputBackings = @{
                @"selected_source": source_backing,
                @"agreement": agreement_backing,
            };
            id<MLFeatureProvider> output = instance->execution_state
                ? [instance->model predictionFromFeatures:inputs
                                               usingState:instance->execution_state
                                                  options:options
                                                    error:&error]
                : [instance->model predictionFromFeatures:inputs
                                                  options:options
                                                    error:&error];
            MLMultiArray * source =
                [output featureValueForName:@"selected_source"].multiArrayValue;
            MLMultiArray * agreed =
                [output featureValueForName:@"agreement"].multiArrayValue;
            if (!output || !source || !agreed ||
                    source.dataType != MLMultiArrayDataTypeInt32 ||
                    agreed.dataType != MLMultiArrayDataTypeInt32 ||
                    (size_t) source.count < n_active ||
                    (size_t) agreed.count < n_active) {
                LOG_WRN("ANE hybrid %s failed: %s\n",
                        instance->name.c_str(),
                        error.localizedDescription.UTF8String ?: "invalid output");
                return;
            }
            if (!direct) {
                std::memcpy(selected_source, source.dataPointer,
                            n_active * sizeof(int32_t));
                std::memcpy(agreement, agreed.dataPointer,
                            n_active * sizeof(int32_t));
                program->copied_output_bytes +=
                    n_active * 2 * sizeof(int32_t);
            }
            if (source.dataPointer == source_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            if (agreed.dataPointer == agreement_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            ok = true;
        }
    });
    return ok;
}

bool common_ane_mtp_program_predict(
        const common_ane_mtp_program_ptr & program,
        const int32_t * token_ids,
        const float * h_nextn,
        uint32_t n_active,
        uint32_t hidden_size,
        const int32_t * positions,
        int32_t * top_token,
        float * confidence,
        float * next_hidden) {
    if (!program || !token_ids || !h_nextn || !top_token || !confidence || !next_hidden ||
            n_active == 0 || n_active > program->batch_bucket) {
        return false;
    }
    if (program->context_length > 0 && positions) {
        for (uint32_t lane = 0; lane < n_active; ++lane) {
            if (positions[lane] < 0 ||
                    (uint32_t) positions[lane] >= program->context_length) {
                return false;
            }
        }
    }

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSError * error = nil;
            MLFeatureDescription * hidden_desc =
                program->model.modelDescription.inputDescriptionsByName[@"h_nextn"];
            const MLMultiArrayDataType hidden_type = hidden_desc.multiArrayConstraint.dataType;
            NSArray<NSNumber *> * token_shape = @[@(program->batch_bucket)];
            NSArray<NSNumber *> * hidden_shape =
                @[@(program->batch_bucket), @(hidden_size)];
            const bool full_bucket = n_active == program->batch_bucket;
            MLMultiArray * tokens = full_bucket
                ? wrap_multi_array((void *) token_ids, token_shape,
                    MLMultiArrayDataTypeInt32, &error)
                : arena_multi_array(*program, "predict.token_ids", token_shape,
                    MLMultiArrayDataTypeInt32, &error);
            MLMultiArray * hidden =
                full_bucket && hidden_type == MLMultiArrayDataTypeFloat32
                ? wrap_multi_array((void *) h_nextn, hidden_shape, hidden_type, &error)
                : arena_multi_array(*program, "predict.h_nextn", hidden_shape,
                    hidden_type, &error);
            if (!tokens || !hidden) {
                return;
            }

            const size_t active_hidden = (size_t) n_active * hidden_size;
            if (hidden_type != MLMultiArrayDataTypeFloat32 &&
                    hidden_type != MLMultiArrayDataTypeFloat16) {
                return;
            }
            if (!full_bucket) {
                int32_t * token_data = (int32_t *) tokens.dataPointer;
                std::memcpy(token_data, token_ids, n_active * sizeof(int32_t));
                std::memset(token_data + n_active, 0,
                        (program->batch_bucket - n_active) * sizeof(int32_t));
                program->arena_input_bytes +=
                    program->batch_bucket * sizeof(int32_t);
            } else {
                ++program->direct_input_views;
            }
            if (!(full_bucket && hidden_type == MLMultiArrayDataTypeFloat32)) {
                write_float_data(hidden, h_nextn, active_hidden);
                const size_t padded_hidden =
                    (size_t) program->batch_bucket * hidden_size - active_hidden;
                std::memset((char *) hidden.dataPointer +
                            active_hidden * multi_array_element_size(hidden_type),
                        0, padded_hidden * multi_array_element_size(hidden_type));
                program->arena_input_bytes +=
                    (size_t) program->batch_bucket * hidden_size *
                    multi_array_element_size(hidden_type);
            } else {
                ++program->direct_input_views;
            }

            NSMutableDictionary<NSString *, MLFeatureValue *> * input_values =
                [@{
                    @"token_ids": [MLFeatureValue featureValueWithMultiArray:tokens],
                    @"h_nextn": [MLFeatureValue featureValueWithMultiArray:hidden],
                } mutableCopy];
            if (program->model.modelDescription.inputDescriptionsByName[@"positions"]) {
                if (!positions) {
                    return;
                }
                MLMultiArray * position_values = full_bucket
                    ? wrap_multi_array((void *) positions, token_shape,
                        MLMultiArrayDataTypeInt32, &error)
                    : arena_multi_array(*program, "predict.positions", token_shape,
                        MLMultiArrayDataTypeInt32, &error);
                if (!position_values) {
                    return;
                }
                if (!full_bucket) {
                    int32_t * position_data = (int32_t *) position_values.dataPointer;
                    std::memcpy(position_data, positions, n_active * sizeof(int32_t));
                    std::memset(position_data + n_active, 0,
                            (program->batch_bucket - n_active) * sizeof(int32_t));
                    program->arena_input_bytes +=
                        program->batch_bucket * sizeof(int32_t);
                } else {
                    ++program->direct_input_views;
                }
                input_values[@"positions"] =
                    [MLFeatureValue featureValueWithMultiArray:position_values];
            }
            MLDictionaryFeatureProvider * inputs = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:input_values error:&error];
            if (!inputs) {
                return;
            }

            NSDictionary<NSString *, MLFeatureDescription *> * output_descriptions =
                program->model.modelDescription.outputDescriptionsByName;
            MLFeatureDescription * token_out_desc = output_descriptions[@"top_token"];
            MLFeatureDescription * confidence_out_desc = output_descriptions[@"confidence"];
            MLFeatureDescription * hidden_out_desc = output_descriptions[@"next_hidden"];
            if (!token_out_desc || !confidence_out_desc || !hidden_out_desc) {
                return;
            }
            const MLMultiArrayDataType confidence_type =
                confidence_out_desc.multiArrayConstraint.dataType;
            const MLMultiArrayDataType output_hidden_type =
                hidden_out_desc.multiArrayConstraint.dataType;
            MLMultiArray * token_backing = full_bucket
                ? wrap_multi_array(top_token, token_out_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32, &error)
                : arena_multi_array(*program, "predict.out_token",
                    token_out_desc.multiArrayConstraint.shape,
                    MLMultiArrayDataTypeInt32, &error);
            MLMultiArray * confidence_backing =
                full_bucket && confidence_type == MLMultiArrayDataTypeFloat32
                ? wrap_multi_array(confidence,
                    confidence_out_desc.multiArrayConstraint.shape,
                    confidence_type, &error)
                : arena_multi_array(*program, "predict.out_confidence",
                    confidence_out_desc.multiArrayConstraint.shape,
                    confidence_type, &error);
            MLMultiArray * hidden_backing =
                full_bucket && output_hidden_type == MLMultiArrayDataTypeFloat32
                ? wrap_multi_array(next_hidden,
                    hidden_out_desc.multiArrayConstraint.shape,
                    output_hidden_type, &error)
                : arena_multi_array(*program, "predict.out_hidden",
                    hidden_out_desc.multiArrayConstraint.shape,
                    output_hidden_type, &error);
            if (!token_backing || !confidence_backing || !hidden_backing) {
                return;
            }
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            options.outputBackings = @{
                @"top_token": token_backing,
                @"confidence": confidence_backing,
                @"next_hidden": hidden_backing,
            };
            id<MLFeatureProvider> output = program->execution_state
                ? [program->model predictionFromFeatures:inputs
                                              usingState:program->execution_state
                                                 options:options
                                                   error:&error]
                : [program->model predictionFromFeatures:inputs error:&error];
            if (!output) {
                LOG_WRN("ANE MTP prediction failed: %s\n",
                        error.localizedDescription.UTF8String ?: "unknown error");
                return;
            }

            MLMultiArray * out_token = [output featureValueForName:@"top_token"].multiArrayValue;
            MLMultiArray * out_confidence = [output featureValueForName:@"confidence"].multiArrayValue;
            MLMultiArray * out_hidden = [output featureValueForName:@"next_hidden"].multiArrayValue;
            if (!out_token || !out_confidence || !out_hidden ||
                    out_token.dataType != MLMultiArrayDataTypeInt32 ||
                    out_token.count < n_active ||
                    out_confidence.count < n_active ||
                    (NSUInteger) out_hidden.count < (NSUInteger) n_active * hidden_size) {
                return;
            }

            if ((out_confidence.dataType != MLMultiArrayDataTypeFloat32 &&
                        out_confidence.dataType != MLMultiArrayDataTypeFloat16) ||
                    (out_hidden.dataType != MLMultiArrayDataTypeFloat32 &&
                        out_hidden.dataType != MLMultiArrayDataTypeFloat16)) {
                return;
            }
            if (!full_bucket) {
                std::memcpy(top_token, out_token.dataPointer,
                        n_active * sizeof(int32_t));
                program->copied_output_bytes +=
                    n_active * sizeof(int32_t);
            }
            if (out_token.dataPointer == token_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            if (!(full_bucket && confidence_type == MLMultiArrayDataTypeFloat32)) {
                read_float_data(out_confidence, confidence, n_active);
                program->copied_output_bytes += n_active * sizeof(float);
            }
            if (out_confidence.dataPointer == confidence_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            if (!(full_bucket && output_hidden_type == MLMultiArrayDataTypeFloat32)) {
                read_float_data(out_hidden, next_hidden, active_hidden);
                program->copied_output_bytes += active_hidden * sizeof(float);
            }
            if (out_hidden.dataPointer == hidden_backing.dataPointer) {
                ++program->direct_output_backings;
            }
            ok = true;
        }
    });
    return ok;
}

bool common_ane_mtp_program_reset(
        const common_ane_mtp_program_ptr & program,
        uint32_t n_lanes,
        const int32_t * active) {
    if (!program || !program->reset_model || !active ||
            n_lanes == 0 || n_lanes > program->batch_bucket) {
        return false;
    }

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSError * error = nil;
            NSArray<NSNumber *> * shape = @[@(program->batch_bucket)];
            const bool full_bucket = n_lanes == program->batch_bucket;
            MLMultiArray * lanes = full_bucket
                ? wrap_multi_array((void *) active, shape,
                    MLMultiArrayDataTypeInt32, &error)
                : arena_multi_array(*program, "reset.active", shape,
                    MLMultiArrayDataTypeInt32, &error);
            if (!lanes) {
                return;
            }
            if (!full_bucket) {
                int32_t * lane_data = (int32_t *) lanes.dataPointer;
                std::memcpy(lane_data, active, n_lanes * sizeof(int32_t));
                std::memset(lane_data + n_lanes, 0,
                        (program->batch_bucket - n_lanes) * sizeof(int32_t));
            }
            MLDictionaryFeatureProvider * input = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{
                    @"active": [MLFeatureValue featureValueWithMultiArray:lanes],
                }
                error:&error];
            if (!input) {
                return;
            }
            id<MLFeatureProvider> output = [program->reset_model
                predictionFromFeatures:input
                            usingState:program->execution_state
                               options:[[MLPredictionOptions alloc] init]
                                 error:&error];
            if (!output) {
                LOG_WRN("ANE MTP state reset failed: %s\n",
                        error.localizedDescription.UTF8String ?: "unknown error");
                return;
            }
            ok = true;
        }
    });
    return ok;
}

bool common_ane_mtp_program_sync_kv(
        const common_ane_mtp_program_ptr & program,
        uint32_t n_active,
        uint32_t row_stride,
        const uint32_t * row_counts,
        const int32_t * positions,
        const float * base_keys,
        const float * base_values,
        uint32_t base_width,
        const float * swa_keys,
        const float * swa_values,
        uint32_t swa_width) {
    if (!program || !program->sync_model || !row_counts || !positions ||
            !base_keys || !base_values || !swa_keys || !swa_values ||
            n_active == 0 || n_active > program->batch_bucket || row_stride == 0) {
        return false;
    }

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
                program->sync_model.modelDescription.inputDescriptionsByName;
            MLFeatureDescription * pos_desc = descriptions[@"positions"];
            if (!pos_desc || pos_desc.type != MLFeatureTypeMultiArray ||
                    pos_desc.multiArrayConstraint.shape.count != 2) {
                return;
            }
            const uint32_t chunk = pos_desc.multiArrayConstraint.shape[1].unsignedIntValue;
            if (chunk == 0 ||
                    (program->sync_chunk > 0 &&
                     chunk != program->sync_chunk)) {
                return;
            }
            for (uint32_t lane = 0; lane < n_active; ++lane) {
                if (row_counts[lane] > chunk ||
                        row_counts[lane] > row_stride) {
                    return;
                }
                if (program->context_length > 0) {
                    for (uint32_t row = 0; row < row_counts[lane]; ++row) {
                        const int32_t position =
                            positions[(size_t) lane * row_stride + row];
                        if (position < 0 ||
                                (uint32_t) position >= program->context_length) {
                            return;
                        }
                    }
                }
            }

            NSError * error = nil;
            NSArray<NSNumber *> * active_shape = @[@(program->batch_bucket)];
            NSArray<NSNumber *> * position_shape =
                @[@(program->batch_bucket), @(chunk)];
            bool exact_layout = n_active == program->batch_bucket &&
                row_stride == chunk;
            for (uint32_t lane = 0; lane < n_active; ++lane) {
                exact_layout = exact_layout && row_counts[lane] == chunk;
            }
            MLMultiArray * out_positions = exact_layout
                ? wrap_multi_array((void *) positions, position_shape,
                    MLMultiArrayDataTypeInt32, &error)
                : arena_multi_array(*program, "sync.positions", position_shape,
                    MLMultiArrayDataTypeInt32, &error);
            MLMultiArray * out_active = arena_multi_array(
                *program, "sync.active", active_shape,
                MLMultiArrayDataTypeInt32, &error);
            auto make_values = [&](NSString * name, const char * arena_name,
                                   uint32_t width, const float * source) -> MLMultiArray * {
                MLFeatureDescription * desc = descriptions[name];
                if (!desc || desc.type != MLFeatureTypeMultiArray) {
                    return nil;
                }
                NSArray<NSNumber *> * shape =
                    @[@(program->batch_bucket), @(chunk), @(width)];
                if (exact_layout &&
                        desc.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat32) {
                    return wrap_multi_array((void *) source, shape,
                        MLMultiArrayDataTypeFloat32, &error);
                }
                return arena_multi_array(*program, arena_name, shape,
                    desc.multiArrayConstraint.dataType, &error);
            };
            MLMultiArray * out_base_k = make_values(
                @"base_keys", "sync.base_k", base_width, base_keys);
            MLMultiArray * out_base_v = make_values(
                @"base_values", "sync.base_v", base_width, base_values);
            MLMultiArray * out_swa_k = make_values(
                @"swa_keys", "sync.swa_k", swa_width, swa_keys);
            MLMultiArray * out_swa_v = make_values(
                @"swa_values", "sync.swa_v", swa_width, swa_values);
            if (!out_active || !out_positions || !out_base_k || !out_base_v || !out_swa_k || !out_swa_v) {
                return;
            }

            int32_t * active_data = (int32_t *) out_active.dataPointer;
            for (uint32_t lane = 0; lane < program->batch_bucket; ++lane) {
                active_data[lane] =
                    lane < n_active && row_counts[lane] > 0 ? 1 : 0;
            }
            auto copy_row = [](MLMultiArray * destination, size_t dst_row,
                               const float * source, size_t src_row,
                               uint32_t width) {
                const size_t element_size =
                    multi_array_element_size(destination.dataType);
                void * dst = (char *) destination.dataPointer +
                    dst_row * width * element_size;
                if (destination.dataType == MLMultiArrayDataTypeFloat32) {
                    std::memcpy(dst, source + src_row * width,
                            width * sizeof(float));
                } else {
                    ggml_fp32_to_fp16_row(source + src_row * width,
                            (ggml_fp16_t *) dst, width);
                }
            };

            int32_t * position_data = (int32_t *) out_positions.dataPointer;
            for (uint32_t lane = 0; lane < program->batch_bucket; ++lane) {
                const uint32_t source_lane = std::min(lane, n_active - 1);
                const uint32_t count = row_counts[source_lane];
                for (uint32_t row = 0; row < chunk; ++row) {
                    const uint32_t source_row = count > 0 ? std::min(row, count - 1) : 0;
                    const size_t src_row = (size_t) source_lane * row_stride + source_row;
                    const size_t dst_row = (size_t) lane * chunk + row;
                    if (!exact_layout) {
                        position_data[dst_row] = positions[src_row];
                    }
                    if (!exact_layout ||
                            out_base_k.dataType != MLMultiArrayDataTypeFloat32) {
                        copy_row(out_base_k, dst_row, base_keys, src_row, base_width);
                        copy_row(out_base_v, dst_row, base_values, src_row, base_width);
                        copy_row(out_swa_k, dst_row, swa_keys, src_row, swa_width);
                        copy_row(out_swa_v, dst_row, swa_values, src_row, swa_width);
                    }
                }
            }

            MLDictionaryFeatureProvider * inputs = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{
                    @"active": [MLFeatureValue featureValueWithMultiArray:out_active],
                    @"positions": [MLFeatureValue featureValueWithMultiArray:out_positions],
                    @"base_keys": [MLFeatureValue featureValueWithMultiArray:out_base_k],
                    @"base_values": [MLFeatureValue featureValueWithMultiArray:out_base_v],
                    @"swa_keys": [MLFeatureValue featureValueWithMultiArray:out_swa_k],
                    @"swa_values": [MLFeatureValue featureValueWithMultiArray:out_swa_v],
                } error:&error];
            id<MLFeatureProvider> output = [program->sync_model
                predictionFromFeatures:inputs
                            usingState:program->execution_state
                               options:[[MLPredictionOptions alloc] init]
                                 error:&error];
            if (!output) {
                LOG_WRN("ANE MTP K/V synchronization failed: %s\n",
                        error.localizedDescription.UTF8String ?: "unknown error");
                return;
            }
            ok = true;
        }
    });
    return ok;
}
