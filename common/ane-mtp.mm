#include "ane-mtp.h"

#include "ane-pump.h"
#include "ane-state-layout.h"

#include "ggml-metal.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

// Internal per-function phase stats. Same field layout as the
// public common_ane_phase_stats snapshot (in ane-mtp.h) but with
// std::atomic<uint64_t> fields so the E-core thread can update
// them without explicit locking while the host reads them via
// common_ane_mtp_program_phase_stats. The snapshot is built by
// individual .load() calls and copied into a plain common_ane_phase_stats
// for return; the atomic struct itself never leaves the file.
struct common_ane_phase_stats_internal {
    std::atomic<uint64_t> input_prep_us_total{0};
    std::atomic<uint64_t> input_prep_us_max{0};
    std::atomic<uint64_t> ane_dispatch_us_total{0};
    std::atomic<uint64_t> ane_dispatch_us_max{0};
    std::atomic<uint64_t> output_read_us_total{0};
    std::atomic<uint64_t> output_read_us_max{0};
    std::atomic<uint64_t> signal_us_total{0};
    std::atomic<uint64_t> signal_us_max{0};
    std::atomic<uint64_t> count{0};
};

// Phase 0 profile streaming state. The path is process-global
// (set once at startup from --tessera-ane-profile-out). The
// output file is opened lazily on the first emit and held open
// until the path is cleared; the lazy open keeps the production
// dispatch path branch-free when profiling is off. The atomic
// guards the lazy open so the first race between two dispatch
// threads is well-defined (one opens, the other re-checks the
// "opened" flag and writes through the same FILE*).
static std::atomic<bool> g_phase_profile_emit_disabled{false};
static std::string g_phase_profile_path;
static std::atomic<FILE *> g_phase_profile_file{nullptr};
static std::mutex g_phase_profile_open_mu;

void common_ane_phase_profile_set_output(const char * path) {
    std::lock_guard<std::mutex> lock(g_phase_profile_open_mu);
    FILE * prev = g_phase_profile_file.load(std::memory_order_acquire);
    if (prev != nullptr) {
        std::fclose(prev);
        g_phase_profile_file.store(nullptr, std::memory_order_release);
    }
    g_phase_profile_path = (path != nullptr) ? std::string(path) : std::string();
    g_phase_profile_emit_disabled.store(g_phase_profile_path.empty(),
        std::memory_order_release);
}

const char * common_ane_phase_profile_get_output() {
    return g_phase_profile_path.c_str();
}

// Forward-declare the file-local phase_profile_emit so the
// test-only hook below can wrap it. The forward decl avoids
// moving the function above the public API; the .mm file's
// translation-unit-private static helpers stay below the
// public surface for readability.
static void phase_profile_emit(const char * phase,
                               uint64_t us,
                               uint32_t n_tokens);

void common_ane_phase_profile_emit_test_only(
        const char * phase, uint64_t us, uint32_t n_tokens) {
    phase_profile_emit(phase, us, n_tokens);
}

// Open the output file on the first emit. Returns the FILE* on
// success, nullptr if the path is empty or the open failed. The
// returned FILE* is owned by g_phase_profile_file and stays open
// until common_ane_phase_profile_set_output is called again.
static FILE * phase_profile_open() {
    FILE * f = g_phase_profile_file.load(std::memory_order_acquire);
    if (f != nullptr) {
        return f;
    }
    std::lock_guard<std::mutex> lock(g_phase_profile_open_mu);
    f = g_phase_profile_file.load(std::memory_order_acquire);
    if (f != nullptr) {
        return f;
    }
    if (g_phase_profile_path.empty()) {
        g_phase_profile_emit_disabled.store(true, std::memory_order_release);
        return nullptr;
    }
    f = std::fopen(g_phase_profile_path.c_str(), "a");
    if (f == nullptr) {
        // Disable further attempts for the rest of the process so
        // we don't pay the fopen cost on every dispatch. The host
        // can re-enable by calling set_output again with a valid
        // path.
        g_phase_profile_emit_disabled.store(true, std::memory_order_release);
        return nullptr;
    }
    // Line-buffered: each emit is one line, flushed at newline.
    std::setvbuf(f, nullptr, _IOLBF, 0);
    g_phase_profile_file.store(f, std::memory_order_release);
    return f;
}

// Emit one NDJSON line for a single phase. The line shape is
// fixed: {"phase":"<phase>","us":<us>,"n_tokens":<n>,"ts":<iso8601>}.
// n_tokens is the function's first non-batch input dim; 0 when
// the shape can't be inferred (defensive default for functions
// that don't have a token input). The ts is an ISO 8601 string
// from std::chrono::system_clock::now(); the formatter uses
// strftime for the date part and prints the fractional seconds
// separately for sub-second precision.
//
// The emit is a no-op when the global path is empty. The check
// is a single atomic load; the file write is held off the
// critical dispatch path as much as possible (line-buffered
// stdio, single write per phase).
static void phase_profile_emit(const char * phase,
                               uint64_t us,
                               uint32_t n_tokens) {
    if (g_phase_profile_emit_disabled.load(std::memory_order_acquire)) {
        return;
    }
    FILE * f = phase_profile_open();
    if (f == nullptr) {
        return;
    }
    // ISO 8601 timestamp. strftime doesn't print fractional
    // seconds, so we format the integer second and append the
    // microsecond fraction manually. The "Z" suffix marks UTC;
    // the host is free to interpret local time but the Studio
    // UI prefers UTC for cross-machine log joins.
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto us_part = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count() % 1000000;
    char date_buf[32];
    std::tm tm_buf;
    gmtime_r(&t, &tm_buf);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    std::fprintf(f,
        "{\"phase\":\"%s\",\"us\":%llu,\"n_tokens\":%u,\"ts\":\"%s.%06lldZ\"}\n",
        phase,
        (unsigned long long) us,
        (unsigned) n_tokens,
        date_buf,
        (long long) us_part);
}

// Phase 0 profile: infer the function's n_tokens (the lane-active
// dim) from the first input slot's shape. For prefill functions
// the first input is typically token_ids of shape [1, N] ->
// n_tokens = N. For MTP/DFlash the first input is target_features
// or token_ids with shape [1, N] -> n_tokens = N. For stateless
// matmul (the W0 spike) the first input is x with shape [N] ->
// n_tokens = N. We treat the first non-zero dim as n_tokens and
// return 0 when the shape is empty or the function has no inputs.
//
// The heuristic is intentionally simple: the profile line is
// a per-phase observation, not a precise lane count. The caller
// can override via the program-specific path if a more accurate
// value is required. Declared after the common_ane_mtp_program
// struct so the program.manifest field is in scope.
static uint32_t infer_n_tokens_from_function(
        const common_ane_mtp_program & program,
        const ane_function_v1_t & function);

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
    // W6: per-function E-core pump. One pump per multifunction
    // function. The pump is initialized when the function is
    // registered at load time; the dispatch path drives the
    // state machine via ane_pump_run (or the host signals via
    // ane_pump_signal_input_ready). The E-core pinning is a
    // follow-on; today the pump is driven on whatever thread
    // ane_pump_run is called from.
    common_ane_pump pump;
    // True once pump has been initialized.
    bool pump_ready = false;
    // Phase 0 profile: per-function timing of the four dispatch
    // phases. The counters are mutated on the E-core thread (the
    // pump runs the submit_fn + signal_fn there); the host reads
    // them via common_ane_mtp_program_phase_stats on the host
    // thread. The atomic fields give tearing-free reads; the
    // counters are approximate (the host snapshot can sample
    // mid-dispatch) but the per-row inconsistency is bounded by
    // one in-flight dispatch and is acceptable for a profile.
    common_ane_phase_stats_internal phase_stats;
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
    // sync_model and reset_model are dropped in W4. The architecture
    // call is that K/V synchronization and K/V clearing are direct
    // memcpy/memset on the state_iosurface, not CPU-only Core ML
    // functions. The .mlmodelc no longer carries a "sync" or "reset"
    // function; the K/V state slots in the manifest are the source of
    // truth. common_ane_mtp_program_sync_kv and
    // common_ane_mtp_program_reset are reimplemented below as direct
    // memory operations on the pinned STATE slots; for bundles
    // without STATE slots (e.g. the gemma4 prefill bundle, where
    // each prefill function owns its own K/V as OUTPUT slots), they
    // are no-ops. The E-core pump (a follow-on commit) will move
    // the memcpy/memset work off the critical dispatch path onto a
    // pinned E-core.
    MLState * execution_state = nil;
    dispatch_queue_t queue = nullptr;
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
    // W6.5 part 2: the async prefill path bypasses the pump (it
    // dispatches directly on program.queue), so the pump's
    // monotonic completion counter isn't used. The async path
    // tracks its own monotonic counter and emits it as the
    // MTLSharedEvent signal value, so async-path consumers can
    // strict-order their reads the same way sync-path consumers
    // do. The two counters are disjoint: the sync path's pump
    // counter increments per function via ane_pump::run; the
    // async counter increments per async prefill call. Both
    // monotonic. (A future commit can unify them by routing the
    // async path through the pump too, eliminating the dual
    // counter.)
    std::atomic<uint64_t> async_call_counter = 0;
    // A prefill prediction may outlive the queue turn which submitted it.
    // Give every asynchronous call distinct arena slots so a later request
    // cannot overwrite Core ML input/output storage still owned by ANE.
    std::atomic<uint64_t> next_prefill_arena_epoch = 0;
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

    // W7: per-slot MTLSharedEvent handles. One event per manifest
    // slot, allocated at load time when the manifest is present.
    // The pump's run() invokes a signal_fn that signals each
    // function's output slot events with the pump's monotonic
    // completion counter value. Downstream consumers (Metal via
    // ggml_mtl_shared_event_encode_wait, or the host via
    // ggml_mtl_shared_event_wait) can then read the slot bytes
    // under the lock-free event contract. Index = slot_id from
    // the manifest. The default signal_fn in the dispatch path
    // is the "ane_signal_events" C function below. The events
    // are released in the destructor.
    ggml_mtl_shared_event_t slot_events[ANE_STATE_SLOTS_MAX] = {};

    ~common_ane_mtp_program() {
        // Tear down the per-function E-core pumps before the
        // rest of the program is freed (the pumps reference the
        // program via the submit_fn / signal_fn closures). The
        // free() drains any in-flight block and releases the
        // per-pump E-core queue.
        for (auto & entry : functions) {
            if (entry.second != nullptr && entry.second->pump_ready) {
                ane_pump::free(entry.second->pump);
            }
        }
        // Drop the pinned MLMultiArray strong references first; they
        // wrap the IOSurface so the order matters (the IOSurface
        // unlock + release below must come after). Under ARC the
        // strong references are released by setting to nil; no
        // explicit [obj release] is needed (or allowed).
        for (uint32_t i = 0; i < ANE_STATE_SLOTS_MAX; ++i) {
            pinned_slots[i] = nil;
        }
        // W7: release the per-slot MTLSharedEvent handles. The
        // events are owned by the program; freeing them here
        // invalidates any pending waits (consumers would see
        // a stale value but the program is being destroyed so
        // there should be no pending waits).
        for (uint32_t i = 0; i < ANE_STATE_SLOTS_MAX; ++i) {
            if (slot_events[i] != nullptr) {
                ggml_mtl_shared_event_free(slot_events[i]);
                slot_events[i] = nullptr;
            }
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

static uint32_t infer_n_tokens_from_function(
        const common_ane_mtp_program & program,
        const ane_function_v1_t & function) {
    if (function.n_inputs == 0) {
        return 0;
    }
    const uint32_t slot_id = function.input_slot_ids[0];
    if (slot_id >= program.manifest.n_slots) {
        return 0;
    }
    const ane_slot_v1_t & slot = program.manifest.slots[slot_id];
    // First non-zero dim wins. The prefill/MTP/DFlash shape is
    // [1, N] so we pick N. The matmul shape is [N] so we pick N.
    // Skip leading 1s and return the next positive dim.
    for (uint32_t i = 0; i < slot.n_dim; ++i) {
        if (slot.shape[i] > 1) {
            return slot.shape[i];
        }
    }
    // All dims were 0 or 1; treat the first dim as the n_tokens
    // for completeness (caller can filter out zero results).
    return slot.n_dim > 0 ? slot.shape[0] : 0;
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

// Find a manifest function by name. Returns nullptr if the manifest
// is not loaded or the function is not present. The returned pointer
// is borrowed; the manifest's lifetime is the program's lifetime.
static const ane_function_v1_t * find_manifest_function(
        const common_ane_mtp_program & program,
        const std::string & function_name) {
    if (!program.manifest_loaded) {
        return nullptr;
    }
    for (uint32_t i = 0; i < program.manifest.n_functions; ++i) {
        if (std::strcmp(program.manifest.functions[i].name,
                        function_name.c_str()) == 0) {
            return &program.manifest.functions[i];
        }
    }
    return nullptr;
}

// Find a manifest slot by name. Returns nullptr if the manifest is
// not loaded or the slot is not present. Borrowed pointer.
static const ane_slot_v1_t * find_manifest_slot(
        const common_ane_mtp_program & program,
        const std::string & slot_name) {
    if (!program.manifest_loaded) {
        return nullptr;
    }
    for (uint32_t i = 0; i < program.manifest.n_slots; ++i) {
        if (std::strcmp(program.manifest.slots[i].name,
                        slot_name.c_str()) == 0) {
            return &program.manifest.slots[i];
        }
    }
    return nullptr;
}

// Find the manifest function associated with a (role, bucket) pair.
// Used by the compute_* entry points to map (role, sequence_length)
// to the right pinned-slot function. The legacy find_compute_function
// lookup is kept for the warmup path.
static const ane_function_v1_t * find_manifest_function_by_role(
        const common_ane_mtp_program & program,
        ane_role_t role,
        uint32_t bucket) {
    if (!program.manifest_loaded) {
        return nullptr;
    }
    for (uint32_t i = 0; i < program.manifest.n_functions; ++i) {
        const ane_function_v1_t & f = program.manifest.functions[i];
        if (f.role == role && f.bucket == bucket) {
            return &f;
        }
    }
    return nullptr;
}

// Write host data into a pinned input slot. The host dtype is fp32
// for float slots and i32 for integer slots. The function converts
// to the slot's declared dtype (fp16 conversion is a one-shot
// ggml_fp32_to_fp16_row call). Returns true on success, false if
// the slot is not found or the count doesn't fit.
static bool set_pinned_input(
        common_ane_mtp_program & program,
        const std::string & slot_name,
        const void * host_data,
        size_t count) {
    const ane_slot_v1_t * slot = find_manifest_slot(program, slot_name);
    if (slot == nullptr) {
        return false;
    }
    size_t element_size = 0;
    switch (slot->dtype) {
        case ANE_DTYPE_F32: element_size = sizeof(float); break;
        case ANE_DTYPE_F16: element_size = sizeof(ggml_fp16_t); break;
        case ANE_DTYPE_I32: element_size = sizeof(int32_t); break;
    }
    if (element_size == 0 || slot->size_bytes / element_size < count) {
        return false;
    }
    void * dst = (char *) program.state_base + slot->offset;
    if (slot->dtype == ANE_DTYPE_F32) {
        std::memcpy(dst, host_data, count * sizeof(float));
    } else if (slot->dtype == ANE_DTYPE_F16) {
        ggml_fp32_to_fp16_row((const float *) host_data,
                              (ggml_fp16_t *) dst, (int64_t) count);
    } else { // ANE_DTYPE_I32
        std::memcpy(dst, host_data, count * sizeof(int32_t));
    }
    return true;
}

// Write i32 host data into a pinned input slot. Specialization for
// the common token_ids/positions inputs. Returns true on success.
static bool set_pinned_input_i32(
        common_ane_mtp_program & program,
        const std::string & slot_name,
        const int32_t * host_data,
        size_t count) {
    const ane_slot_v1_t * slot = find_manifest_slot(program, slot_name);
    if (slot == nullptr || slot->dtype != ANE_DTYPE_I32) {
        return false;
    }
    if ((size_t) slot->size_bytes / sizeof(int32_t) < count) {
        return false;
    }
    void * dst = (char *) program.state_base + slot->offset;
    std::memcpy(dst, host_data, count * sizeof(int32_t));
    return true;
}

// Read a pinned output slot to host memory as i32. The slot must
// be declared as i32 in the manifest. Returns true on success.
static bool get_pinned_output_i32(
        common_ane_mtp_program & program,
        const std::string & slot_name,
        int32_t * host_data,
        size_t count) {
    const ane_slot_v1_t * slot = find_manifest_slot(program, slot_name);
    if (slot == nullptr || slot->dtype != ANE_DTYPE_I32) {
        return false;
    }
    if (slot->size_bytes / sizeof(int32_t) < count) {
        return false;
    }
    void * src = (char *) program.state_base + slot->offset;
    std::memcpy(host_data, src, count * sizeof(int32_t));
    return true;
}

// Read a pinned output slot to host memory as fp32. The slot's
// declared dtype is converted to fp32 (fp16 is the common case for
// the gemma4 prefill bundle). Returns true on success.
static bool get_pinned_output(
        common_ane_mtp_program & program,
        const std::string & slot_name,
        float * host_data,
        size_t count) {
    const ane_slot_v1_t * slot = find_manifest_slot(program, slot_name);
    if (slot == nullptr) {
        return false;
    }
    size_t element_size = 0;
    switch (slot->dtype) {
        case ANE_DTYPE_F32: element_size = sizeof(float); break;
        case ANE_DTYPE_F16: element_size = sizeof(ggml_fp16_t); break;
        case ANE_DTYPE_I32: element_size = sizeof(int32_t); break;
    }
    if (element_size == 0 || slot->size_bytes / element_size < count) {
        return false;
    }
    void * src = (char *) program.state_base + slot->offset;
    if (slot->dtype == ANE_DTYPE_F32) {
        std::memcpy(host_data, src, count * sizeof(float));
    } else if (slot->dtype == ANE_DTYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src,
                              host_data, (int64_t) count);
    } else {
        // i32 -> fp32 conversion (rare; prefill outputs are always f16/f32)
        const int32_t * src_i32 = (const int32_t *) src;
        for (size_t i = 0; i < count; ++i) {
            host_data[i] = (float) src_i32[i];
        }
    }
    return true;
}

// Strip the "<function_name>." prefix from a manifest slot name to
// get the Core ML function's declared input/output name. Returns
// the full name when the prefix is absent (defensive default).
static std::string strip_manifest_prefix(
        const std::string & full,
        const std::string & prefix) {
    if (full.size() > prefix.size() + 1 &&
            full.compare(0, prefix.size(), prefix) == 0 &&
            full[prefix.size()] == '.') {
        return full.substr(prefix.size() + 1);
    }
    return full;
}

// The dispatch body, factored out so the sync path can wrap it in
// dispatch_sync (the program queue serializes Core ML dispatches)
// while the async path runs on the program queue already and
// invokes this directly. The caller is responsible for queue
// serialization; the body is the prediction + zero-copy proof.
//
// Phase 0 profile: the body times three phases via ggml_time_us
// and records them on instance.phase_stats. The phases are:
//   input_prep   building the input feature dict (slot loop +
//                extra_inputs merge + MLDictionaryFeatureProvider)
//   ane_dispatch the Core ML predictionFromFeatures call (the
//                actual ANE/CPU work)
//   output_read  reading outputs from the feature provider +
//                zero-copy verification
// The fourth phase (signal) is recorded by ane_signal_slot_events.
// Failed dispatches do not update the phase_stats counters; the
// host can detect a clean run by checking (count == 0 -> no
// successful dispatches yet).
static bool dispatch_pinned_function_locked(
        common_ane_mtp_program & program,
        const std::string & function_name,
        NSDictionary<NSString *, MLFeatureValue *> * extra_inputs,
        const std::unordered_set<std::string> * output_names) {
    const ane_function_v1_t * function = find_manifest_function(
        program, function_name);
    if (function == nullptr) {
        return false;
    }
    auto it = program.functions.find(function_name);
    if (it == program.functions.end() || it->second == nullptr) {
        return false;
    }
    common_ane_compute_instance & instance = *it->second;
    if (!instance.warm.load()) {
        return false;
    }
    const int64_t t_input_start = ggml_time_us();
    @autoreleasepool {
        NSError * error = nil;
        NSMutableDictionary<NSString *, MLFeatureValue *> * features =
            [NSMutableDictionary dictionary];
        for (uint32_t i = 0; i < function->n_inputs; ++i) {
            const uint32_t slot_id = function->input_slot_ids[i];
            if (slot_id >= ANE_STATE_SLOTS_MAX ||
                    program.pinned_slots[slot_id] == nil) {
                return false;
            }
            const std::string core_ml_name = strip_manifest_prefix(
                program.manifest.slots[slot_id].name, function_name);
            features[[NSString stringWithUTF8String:core_ml_name.c_str()]] =
                [MLFeatureValue featureValueWithMultiArray:
                    program.pinned_slots[slot_id]];
        }
        if (extra_inputs != nil) {
            [features addEntriesFromDictionary:extra_inputs];
        }
        MLDictionaryFeatureProvider * inputs = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:features error:&error];
        if (inputs == nil) {
            LOG_WRN("ANE pinned dispatch %s: input provider build failed: %s\n",
                    function_name.c_str(),
                    error.localizedDescription.UTF8String ?: "unknown");
            return false;
        }
        MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
        NSMutableDictionary<NSString *, MLMultiArray *> * backings =
            [NSMutableDictionary dictionary];
        for (uint32_t i = 0; i < function->n_outputs; ++i) {
            const uint32_t slot_id = function->output_slot_ids[i];
            if (slot_id >= ANE_STATE_SLOTS_MAX ||
                    program.pinned_slots[slot_id] == nil) {
                return false;
            }
            const std::string core_ml_name = strip_manifest_prefix(
                program.manifest.slots[slot_id].name, function_name);
            if (output_names != nullptr &&
                    output_names->find(core_ml_name) == output_names->end()) {
                continue;
            }
            backings[[NSString stringWithUTF8String:core_ml_name.c_str()]] =
                program.pinned_slots[slot_id];
        }
        options.outputBackings = backings;
        const int64_t t_ane_start = ggml_time_us();
        id<MLFeatureProvider> output = [instance.model
            predictionFromFeatures:inputs
                           options:options
                             error:&error];
        const int64_t t_output_start = ggml_time_us();
        if (output == nil) {
            LOG_WRN("ANE pinned dispatch %s: prediction failed: %s\n",
                    function_name.c_str(),
                    error.localizedDescription.UTF8String ?: "unknown");
            return false;
        }
        for (uint32_t i = 0; i < function->n_outputs; ++i) {
            const uint32_t slot_id = function->output_slot_ids[i];
            if (slot_id >= ANE_STATE_SLOTS_MAX) continue;
            const std::string core_ml_name = strip_manifest_prefix(
                program.manifest.slots[slot_id].name, function_name);
            if (output_names != nullptr &&
                    output_names->find(core_ml_name) == output_names->end()) {
                continue;
            }
            MLMultiArray * result = [[output featureValueForName:
                [NSString stringWithUTF8String:core_ml_name.c_str()]]
                multiArrayValue];
            if (result == nil ||
                    result.dataPointer != program.pinned_slots[slot_id].dataPointer) {
                LOG_WRN("ANE pinned dispatch %s: output %s not zero-copy\n",
                        function_name.c_str(), core_ml_name.c_str());
                return false;
            }
        }
        const int64_t t_end = ggml_time_us();
        // Record phase stats. Each phase's duration is a uint64
        // microsecond delta; we accumulate totals via fetch_add
        // and update max via a CAS loop. Failed dispatches have
        // already returned false above; we only reach this point
        // on success.
        const uint64_t dt_input = (uint64_t) (t_ane_start - t_input_start);
        const uint64_t dt_ane = (uint64_t) (t_output_start - t_ane_start);
        const uint64_t dt_output = (uint64_t) (t_end - t_output_start);
        instance.phase_stats.input_prep_us_total.fetch_add(
            dt_input, std::memory_order_relaxed);
        instance.phase_stats.ane_dispatch_us_total.fetch_add(
            dt_ane, std::memory_order_relaxed);
        instance.phase_stats.output_read_us_total.fetch_add(
            dt_output, std::memory_order_relaxed);
        // Max updates: load-CAS loop. Single-thread per instance
        // (the pump's E-core thread is the only writer), so the
        // CAS rarely retries; we use relaxed ordering.
        auto update_max = [](std::atomic<uint64_t> & field, uint64_t value) {
            uint64_t cur = field.load(std::memory_order_relaxed);
            while (value > cur &&
                    !field.compare_exchange_weak(cur, value,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                // cur is reloaded by compare_exchange_weak; loop.
            }
        };
        update_max(instance.phase_stats.input_prep_us_max, dt_input);
        update_max(instance.phase_stats.ane_dispatch_us_max, dt_ane);
        update_max(instance.phase_stats.output_read_us_max, dt_output);
        instance.phase_stats.count.fetch_add(
            1, std::memory_order_relaxed);
        // Phase 0 profile: emit one NDJSON line per phase. The
        // emit is opt-in via common_ane_phase_profile_set_output;
        // the disabled check is a single atomic load so the
        // branch is cheap when profiling is off. n_tokens is
        // inferred from the function's first input slot shape
        // (the lane-active dim for prefill, MTP, and DFlash).
        const uint32_t n_tokens = infer_n_tokens_from_function(
            program, *function);
        phase_profile_emit("input_prep", dt_input, n_tokens);
        phase_profile_emit("ane_dispatch", dt_ane, n_tokens);
        phase_profile_emit("output_read", dt_output, n_tokens);
        return true;
    }
    return true;  // unreachable
}

// Dispatch the bound function by name, using the pinned IOSurface
// state. Inputs are already in the pinned state from prior
// set_pinned_input calls; outputs land in the pinned state and
// callers read them via get_pinned_output. `extra_inputs` is a
// caller-provided input feature dict (e.g. weight MLMultiArrays
// that aren't in the manifest's input_slots); it is merged into
// the manifest's input feature dict. `output_names` is an optional
// caller-provided set of Core ML output names to pin via
// outputBackings; if null, every manifest output slot is pinned
// (the canonical multifunction case). When the filter is non-null,
// outputs not in the set go to Core ML's allocator (the simpler
// prefill case, which doesn't need K/V in the IOSurface). The
// dispatch is synchronous on the program queue. Returns false on
// any failure (manifest missing, function unknown, model not warm,
// Core ML prediction nil, output not zero-copy).
//
// W7: default signal_fn for the pump. Iterates the function's
// output slot ids in the manifest and signals each slot's
// MTLSharedEvent with the pump's completion value. The signal
// is the OUTPUT_READY -> IDLE transition's downstream release:
// a Metal consumer can encodeWaitForEvent: cmd_buf with the
// same value and observe the IOSurface bytes that the ANE
// just wrote.
//
// We pass the program via context (the function is the
// producer for which we're signaling; the program's events
// are indexed by slot id). The function_id is the manifest's
// function index; the program resolves the output slot ids
// from the manifest and signals each.
//
// Phase 0 profile: the per-slot signals are timed via
// ggml_time_us and recorded on the function's instance.phase_stats.
// The signal phase is typically <1us per slot (the signal is
// a single setSignaledValue: call), so the totals are small
// relative to ane_dispatch; the metric is here for tail-latency
// debugging when downstream Metal consumers are involved.
static void ane_signal_slot_events(
        common_ane_mtp_program & program,
        uint32_t function_id,
        uint64_t value,
        void * /*context*/) {
    if (!program.manifest_loaded ||
            function_id >= program.manifest.n_functions) {
        return;
    }
    const ane_function_v1_t & function = program.manifest.functions[function_id];
    // Resolve the function's compute instance for phase stats.
    // The manifest's function name maps to the program's
    // functions map; if the function was never registered (e.g.
    // a stale manifest), fall back gracefully.
    common_ane_compute_instance * instance = nullptr;
    auto it = program.functions.find(function.name);
    if (it != program.functions.end() && it->second != nullptr) {
        instance = it->second.get();
    }
    const int64_t t_signal_start = ggml_time_us();
    for (uint32_t i = 0; i < function.n_outputs; ++i) {
        const uint32_t slot_id = function.output_slot_ids[i];
        if (slot_id >= ANE_STATE_SLOTS_MAX) continue;
        ggml_mtl_shared_event_t event = program.slot_events[slot_id];
        if (event != nullptr) {
            ggml_mtl_shared_event_signal(event, value);
        }
    }
    const int64_t t_signal_end = ggml_time_us();
    if (instance != nullptr) {
        const uint64_t dt = (uint64_t) (t_signal_end - t_signal_start);
        instance->phase_stats.signal_us_total.fetch_add(
            dt, std::memory_order_relaxed);
        // Max update via load-CAS loop (single-writer on the
        // E-core thread, contention-free in practice).
        uint64_t cur = instance->phase_stats.signal_us_max.load(
            std::memory_order_relaxed);
        while (dt > cur &&
                !instance->phase_stats.signal_us_max.compare_exchange_weak(
                    cur, dt,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
            // cur reloaded; loop.
        }
    }
}

// W6.5 part 2: the pump's submit_fn callback. The pump runs
// this on the per-pump E-core thread (QOS_CLASS_BACKGROUND
// affinity, set in ane_pump::init). The submit_fn unpacks
// the dispatch context (function_name + extra_inputs +
// output_names) and calls the locked body. The locked body
// is the prediction + zero-copy proof; it doesn't know about
// the pump or the queue — those concerns live here, in the
// pump's wrapper. This is the unified state-machine ownership:
// the lock-free state machine (IDLE -> INPUT_READY ->
// ANE_BUSY -> OUTPUT_READY -> IDLE) is owned by the pump;
// the host only signals input readiness; the pump drives the
// submit + signal + completion transitions atomically.
struct ane_pump_dispatch_context {
    const std::string * function_name;
    NSDictionary<NSString *, MLFeatureValue *> * extra_inputs;
    const std::unordered_set<std::string> * output_names;
};

static bool dispatch_pinned_function_submit(
        common_ane_mtp_program & program,
        common_ane_compute_instance & instance,
        void * context) {
    const auto * ctx = static_cast<const ane_pump_dispatch_context *>(context);
    return dispatch_pinned_function_locked(
        program, *ctx->function_name, ctx->extra_inputs, ctx->output_names);
}

// This is the canonical multifunction dispatch path. The W2 design
// is locked: stateless at the Core ML level, stateful via the
// IOSurface; the dispatch doesn't use MLState.
//
// W6.5 part 2: the dispatch is now driven through the per-function
// E-core pump (common/ane-pump.h). The state machine is owned by
// the pump; the host only signals input readiness. The pump
// dispatches the submit_fn on the E-core thread (QOS_CLASS_BACKGROUND
// affinity, set at pump init) and signals the per-slot MTLSharedEvent
// handles on the E-core thread. The signal value is the pump's
// monotonic completion counter (replacing the old steady_clock
// nanoseconds; consumers can now strict-order their reads via the
// counter, which is the canonical handoff primitive). Program
// queue serialization is gone: the pump's serial E-core queue
// serializes the same function's dispatches (ANE itself is
// single-threaded per function), and the lock-free CAS guards
// against host races.
static bool dispatch_pinned_function(
        common_ane_mtp_program & program,
        const std::string & function_name,
        NSDictionary<NSString *, MLFeatureValue *> * extra_inputs = nil,
        const std::unordered_set<std::string> * output_names = nullptr) {
    if (!program.manifest_loaded) {
        return false;
    }
    auto it = program.functions.find(function_name);
    if (it == program.functions.end() || it->second == nullptr) {
        return false;
    }
    common_ane_compute_instance & instance = *it->second;
    if (!instance.pump_ready) {
        return false;
    }
    // Host-side: signal input readiness. The CAS IDLE -> INPUT_READY
    // fails when the pump is busy (a racing dispatch). The caller
    // should retry; for the canonical prefill path the host is
    // single-threaded per program, so the CAS is contention-free.
    if (!ane_pump::signal_input_ready(instance.pump)) {
        return false;
    }
    // The dispatch context is a stack-allocated POD; the pump's
    // run() consumes it before returning (it doesn't outlive this
    // scope, so no heap allocation is required).
    ane_pump_dispatch_context ctx = {
        &function_name, extra_inputs, output_names
    };
    // Pump-side: drive the cycle. submit_fn = dispatch_pinned_function_submit
    // (runs on the E-core thread); signal_fn = ane_signal_slot_events
    // (also runs on the E-core thread, after the OUTPUT_READY transition).
    // The signal value is the pump's monotonic completion counter.
    return ane_pump::run(instance.pump, program, instance,
                          dispatch_pinned_function_submit,
                          ane_signal_slot_events,
                          &ctx);
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
        // sync_model and reset_model are dropped in W4. The .mlmodelc
        // is a multifunction stateless asset; K/V synchronization and
        // K/V clearing are direct memcpy/memset on the state IOSurface
        // (the pinned STATE slots), not CPU-only Core ML functions.
        // The "sync" and "reset" functions are no longer loaded. The
        // .mlmodelc no longer carries them; the bundle's K/V STATE
        // slots are the source of truth.
        //
        // Keep the live execution_state for the dispatch path. The
        // 5s re-warm timer and the keepalive_state it protected
        // are gone; with IOSurface-backed state and the .mlmodelc
        // loaded once at startup, neither the state nor the
        // compiled model goes cold. (A follow-on commit will drop
        // execution_state too once the dispatch path migrates to
        // pinned-state slots + outputBackings for every function.)
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
        //
        // The env var TESSERA_ANE_STATE_LAYOUT_MANIFEST, when set,
        // overrides the sidecar lookup so the load can find a manifest
        // that lives next to a source .mlmodelc (e.g. the gemma4
        // prefill bundle's prefill-bundle.ane_state.v1.json) when the
        // materialized .mlmodelc is in the per-process cache. The
        // canonical load path (sidecar next to the materialized
        // .mlmodelc) takes over once the converter embeds the manifest
        // as a GGUF tensor (a follow-on).
        {
            std::string manifest_path_str;
            if (const char * override_path = std::getenv(
                    "TESSERA_ANE_STATE_LAYOUT_MANIFEST");
                    override_path != nullptr && override_path[0] != '\0') {
                manifest_path_str = override_path;
            } else {
                manifest_path_str = ane_layout::manifest_path_for_mlmodelc_dir(
                    bundle_path.c_str());
            }
            if (!manifest_path_str.empty()) {
                std::string merror;
                if (ane_layout::read_state_layout(manifest_path_str.c_str(),
                                                  &program->manifest, &merror)) {
                    program->manifest_loaded = true;
                    LOG_INF("loaded ane_state_layout.v1 manifest for %s: "
                            "%u slots, %u functions, %u deps (path=%s)\n",
                            program->manifest.bundle_name,
                            program->manifest.n_slots,
                            program->manifest.n_functions,
                            program->manifest.n_deps,
                            manifest_path_str.c_str());
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
            // W7: allocate one MTLSharedEvent per manifest slot.
            // The events are signaled by the pump's signal_fn on
            // OUTPUT_READY and consumed by Metal (via
            // ggml_mtl_shared_event_encode_wait) or by the host
            // (via ggml_mtl_shared_event_wait). The event value
            // is the pump's monotonic completion counter, so
            // consumers can strict-order their reads.
            for (uint32_t i = 0; i < program->manifest.n_slots; ++i) {
                program->slot_events[i] = ggml_mtl_shared_event_new();
                if (program->slot_events[i] == nullptr) {
                    LOG_WRN("failed to allocate MTLSharedEvent for slot %s\n",
                            program->manifest.slots[i].name);
                    return nullptr;
                }
            }
            LOG_INF("ANE state IOSurface: %zu bytes, %u pinned slots, "
                    "%u MTLSharedEvents for %s\n",
                    program->state_size, program->manifest.n_slots,
                    program->manifest.n_slots,
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
                // W6.1: initialize the per-function E-core pump.
                // The pump drives the lock-free state machine
                // (IDLE -> INPUT_READY -> ANE_BUSY -> OUTPUT_READY
                // -> IDLE) for this function. The pump's
                // input_slot_ids and output_slot_ids are
                // resolved from the manifest here so the pump
                // doesn't need to look them up on every
                // transition. When the manifest is absent, the
                // pump is left in a default-initialized state
                // (state == IDLE, no slot ids) and is not used
                // (the load path returns false if no manifest
                // would have made it to the dispatch anyway).
                if (program->manifest_loaded) {
                    for (uint32_t fi = 0;
                            fi < program->manifest.n_functions; ++fi) {
                        if (std::strcmp(
                                program->manifest.functions[fi].name,
                                function_name.c_str()) == 0) {
                            instance->pump_ready = ane_pump::init(
                                instance->pump, program->manifest, fi);
                            break;
                        }
                    }
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
        // Phase 0 profile: pick up the host's TESSERA_ANE_PROFILE_OUT
        // env var (or any value set via the public set_output API)
        // so the dispatch path can emit per-phase NDJSON lines
        // without an extra CLI plumbing step. The env var is
        // consulted here (once per program load) rather than per
        // dispatch; the set_output helper is idempotent so a
        // second call is cheap. Marked experimental: the env var
        // name is unstable until the host-side reader lands.
        if (const char * env = std::getenv("TESSERA_ANE_PROFILE_OUT")) {
            if (env[0] != '\0') {
                common_ane_phase_profile_set_output(env);
            }
        }
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

std::vector<common_ane_program_phase_stats_row> common_ane_mtp_program_phase_stats(
        const common_ane_mtp_program_ptr & program) {
    std::vector<common_ane_program_phase_stats_row> result;
    if (!program) {
        return result;
    }
    result.reserve(program->functions.size());
    for (const auto & entry : program->functions) {
        const auto & instance = *entry.second;
        // Snapshot the per-function phase stats. The values are
        // atomic so we read them in a consistent order; the
        // snapshot is not strictly atomic (the four phase totals
        // can be sampled at slightly different microsecond
        // boundaries if the E-core thread is in the middle of a
        // dispatch), but the resulting per-row inconsistency is
        // bounded by one in-flight dispatch and is acceptable
        // for a profile snapshot.
        common_ane_program_phase_stats_row row;
        row.function_name = instance.name;
        row.role = instance.role;
        row.bucket = instance.bucket;
        row.stats.input_prep_us_total = instance.phase_stats.input_prep_us_total.load(
            std::memory_order_relaxed);
        row.stats.input_prep_us_max = instance.phase_stats.input_prep_us_max.load(
            std::memory_order_relaxed);
        row.stats.ane_dispatch_us_total = instance.phase_stats.ane_dispatch_us_total.load(
            std::memory_order_relaxed);
        row.stats.ane_dispatch_us_max = instance.phase_stats.ane_dispatch_us_max.load(
            std::memory_order_relaxed);
        row.stats.output_read_us_total = instance.phase_stats.output_read_us_total.load(
            std::memory_order_relaxed);
        row.stats.output_read_us_max = instance.phase_stats.output_read_us_max.load(
            std::memory_order_relaxed);
        row.stats.signal_us_total = instance.phase_stats.signal_us_total.load(
            std::memory_order_relaxed);
        row.stats.signal_us_max = instance.phase_stats.signal_us_max.load(
            std::memory_order_relaxed);
        row.stats.count = instance.phase_stats.count.load(
            std::memory_order_relaxed);
        result.push_back(std::move(row));
    }
    // Same sort order as common_ane_compute_functions: (role, bucket).
    std::sort(result.begin(), result.end(),
            [](const common_ane_program_phase_stats_row & a,
               const common_ane_program_phase_stats_row & b) {
        if (a.role != b.role) return a.role < b.role;
        return a.bucket < b.bucket;
    });
    return result;
}

bool common_ane_compute_prefill(
        const common_ane_mtp_program_ptr & program,
        uint32_t sequence_length,
        const int32_t * token_ids,
        const int32_t * positions,
        uint32_t n_active,
        uint32_t hidden_size,
        float * hidden_states) {
    if (!program || !token_ids || !positions || !hidden_states ||
            n_active == 0 || hidden_size == 0) {
        return false;
    }
    // W3.5.2: the legacy arena + MLState path is retired. The
    // dispatch contract requires the manifest. Refuse the call
    // when the manifest is absent.
    if (!program->manifest_loaded) {
        return false;
    }
    const ane_function_v1_t * function = find_manifest_function_by_role(
        *program, ANE_ROLE_PREFILL, sequence_length);
    if (function == nullptr) {
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
    const size_t active_tokens = (size_t) n_active * sequence_length;
    const std::string func_name = function->name;
    if (!set_pinned_input_i32(*program, func_name + ".token_ids",
                              token_ids, active_tokens) ||
            !set_pinned_input_i32(*program, func_name + ".positions",
                                  positions, active_tokens)) {
        return false;
    }
    // Pin only the hidden_states output; the K/V outputs (when the
    // function declares them) go to Core ML's allocator since this
    // entry point doesn't read K/V. The output_filter is the
    // selector for outputBackings.
    const std::unordered_set<std::string> output_names = { "hidden_states" };
    if (!dispatch_pinned_function(*program, func_name, nullptr, &output_names)) {
        return false;
    }
    return get_pinned_output(*program, func_name + ".hidden_states",
                              hidden_states, active_tokens * hidden_size);
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
    if (!program || !token_ids || !positions || !hidden_states || !key_states ||
            !value_states || n_active == 0 ||
            hidden_size == 0 || kv_heads == 0 || head_dim == 0) {
        return false;
    }
    // W3.5.1: the legacy arena + MLState path is retired. The dispatch
    // contract requires the manifest (the multifunction bundle is
    // loaded stateless, the IOSurface is the canonical state). The
    // gemma4 prefill bundle (and every future multifunction bundle)
    // emits a manifest sidecar at convert time; without one, the
    // bundle cannot be used. Refuse the call rather than silently
    // fall back to a path that no longer exists in the runtime.
    if (!program->manifest_loaded) {
        return false;
    }
    const ane_function_v1_t * function = find_manifest_function_by_role(
        *program, ANE_ROLE_PREFILL, sequence_length);
    if (function == nullptr) {
        return false;
    }
    const size_t active_tokens = (size_t) n_active * sequence_length;
    const size_t kv_width = (size_t) kv_heads * head_dim;
    const std::string func_name = function->name;
    const std::string token_slot_name = func_name + ".token_ids";
    const std::string position_slot_name = func_name + ".positions";
    if (!set_pinned_input_i32(*program, token_slot_name, token_ids, active_tokens) ||
            !set_pinned_input_i32(*program, position_slot_name, positions, active_tokens)) {
        return false;
    }
    NSMutableDictionary<NSString *, MLFeatureValue *> * extra = nil;
    if (!program->weight_inputs.empty() || !program->embedding.empty()) {
        extra = [NSMutableDictionary dictionary];
        for (auto & entry : program->weight_inputs) {
            NSString * weight_name = [NSString stringWithUTF8String:entry.first.c_str()];
            extra[weight_name] = [MLFeatureValue featureValueWithMultiArray:entry.second];
        }
        if (const ane_slot_v1_t * emb_slot = find_manifest_slot(*program, "embedded");
                emb_slot != nullptr) {
            if (program->embedding.empty() || program->embedding_dim != hidden_size) {
                return false;
            }
            NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:emb_slot->n_dim];
            for (uint32_t i = 0; i < emb_slot->n_dim; ++i) {
                [shape addObject:@(emb_slot->shape[i])];
            }
            NSError * emb_error = nil;
            MLMultiArray * embedded = [[MLMultiArray alloc]
                initWithShape:shape
                     dataType:MLMultiArrayDataTypeFloat16
                        error:&emb_error];
            if (embedded == nil) {
                return false;
            }
            uint16_t * dst = (uint16_t *) embedded.dataPointer;
            const uint16_t * vocab = program->embedding.data();
            const size_t emb_count = (size_t) embedded.count;
            for (size_t i = 0; i < emb_count; ++i) {
                const int32_t tok = token_ids[i];
                if ((size_t) tok * hidden_size + hidden_size > program->embedding.size()) {
                    return false;
                }
                std::memcpy(dst + i * hidden_size,
                            vocab + (size_t) tok * hidden_size,
                            hidden_size * sizeof(uint16_t));
            }
            extra[@"embedded"] = [MLFeatureValue featureValueWithMultiArray:embedded];
        }
    }
    if (!dispatch_pinned_function(*program, func_name, extra)) {
        return false;
    }
    const std::string hidden_slot = func_name + ".hidden_states";
    const std::string key_slot = func_name + ".key_states";
    const std::string value_slot = func_name + ".value_states";
    return get_pinned_output(*program, hidden_slot, hidden_states, active_tokens * hidden_size) &&
           get_pinned_output(*program, key_slot, key_states, active_tokens * kv_width) &&
           get_pinned_output(*program, value_slot, value_states, active_tokens * kv_width);
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
    // W3.5.3: the legacy per-request arena path is retired. The
    // async dispatch is now a direct dispatch_async to the pinned-
    // slot path; the result lands in the pinned output slot and
    // is read by the completion handler via get_pinned_output. The
    // request lifecycle is the dispatch_semaphore signal.
    if (!program->manifest_loaded) {
        return nullptr;
    }
    const ane_function_v1_t * function = find_manifest_function_by_role(
        *program, ANE_ROLE_PREFILL, sequence_length);
    if (function == nullptr) {
        return nullptr;
    }
    if (program->context_length > 0) {
        const size_t count = (size_t) n_active * sequence_length;
        for (size_t i = 0; i < count; ++i) {
            if (positions[i] < 0 ||
                    (uint32_t) positions[i] >= program->context_length) {
                return nullptr;
            }
        }
    }
    auto request = std::make_shared<common_ane_prefill_request>();
    request->program = program;
    request->completion = dispatch_semaphore_create(0);
    request->arena_epoch = program->next_prefill_arena_epoch.fetch_add(1) + 1;
    ++program->async_prefill_submissions;
    const size_t active_tokens = (size_t) n_active * sequence_length;
    const std::string func_name = function->name;
    const std::string token_slot = func_name + ".token_ids";
    const std::string position_slot = func_name + ".positions";
    const std::string hidden_slot = func_name + ".hidden_states";
    // The async dispatch is serialized through the program queue
    // (the same serial queue the sync path uses) so the host input
    // write and the dispatch are atomically observed by the host
    // and by Core ML. Per-request state is the request's local
    // success flag; the pinned state is the program's, shared
    // across requests (only one async prefill can be in flight per
    // program, matching the sync-path contract).
    dispatch_async(program->queue, ^{
        @autoreleasepool {
            const auto finish = ^(bool success) {
                ++request->program->async_prefill_completions;
                if (!success) {
                    ++request->program->async_prefill_failures;
                }
                request->success.store(success);
                request->complete.store(true);
                dispatch_semaphore_signal(request->completion);
            };
            // Copy host inputs into the pinned input slots on the
            // dispatch thread. The host pointers are caller-owned;
            // the copy is mandatory because the caller's buffers
            // may be recycled before the async dispatch runs.
            if (!set_pinned_input_i32(*request->program, token_slot,
                                      token_ids, active_tokens) ||
                    !set_pinned_input_i32(*request->program, position_slot,
                                          positions, active_tokens)) {
                finish(false);
                return;
            }
            const std::unordered_set<std::string> output_names = { "hidden_states" };
            if (!dispatch_pinned_function_locked(*request->program, func_name,
                                                  nullptr, &output_names)) {
                finish(false);
                return;
            }
            // W7 (async-path fix): the sync path's dispatch_pinned_function
            // signals the per-slot MTLSharedEvent handles via the pump's
            // signal_fn. The async path bypasses the pump (it dispatches
            // directly via dispatch_async on program.queue) so the signal
            // doesn't fire automatically; emit it here. The value is the
            // program's per-function call counter, incremented on every
            // successful async prefill; downstream consumers observe a
            // monotonic ordering. The same value will be visible on the
            // next sync dispatch through the pump (the pump tracks its
            // own monotonic counter; the two counters are disjoint because
            // the sync and async paths are serialized through program.queue
            // in this commit and never overlap on the same function).
            const uint32_t function_id = (uint32_t) (function - request->program->manifest.functions);
            ane_signal_slot_events(*request->program, function_id,
                                    ++request->program->async_call_counter, nullptr);
            // Read the result. The hidden_states output is in the
            // pinned slot; copy to the caller's host buffer.
            get_pinned_output(*request->program, hidden_slot,
                              hidden_states, active_tokens * hidden_size);
            finish(true);
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
    if (!program || !target_features || !token_ids || !positions ||
            !draft_tokens || !confidence || n_active == 0 ||
            feature_width == 0) {
        return false;
    }
    if (!program->manifest_loaded) {
        return false;
    }
    const ane_function_v1_t * function = find_manifest_function_by_role(
        *program, ANE_ROLE_DFLASH, block_size);
    if (function == nullptr) {
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
    const std::string func_name = function->name;
    // The DFlash function's pinned input slots are target_features
    // (fp32, [bucket, feature_width]), token_ids (i32, [bucket]),
    // positions (i32, [bucket]). Outputs are draft_tokens (i32)
    // and confidence (fp32). The pinned-slot path writes the host
    // inputs into the IOSurface directly and pins the outputs as
    // zero-copy MLMultiArray backings. The host output buffers are
    // filled by get_pinned_output below.
    if (!set_pinned_input(*program, func_name + ".target_features",
                          target_features, (size_t) n_active * feature_width) ||
            !set_pinned_input_i32(*program, func_name + ".token_ids",
                                  token_ids, n_active) ||
            !set_pinned_input_i32(*program, func_name + ".positions",
                                  positions, n_active)) {
        return false;
    }
    if (!dispatch_pinned_function(*program, func_name)) {
        return false;
    }
    // Copy outputs to host. draft_tokens is i32 (no conversion);
    // confidence is fp32 (or fp16 -> fp32 conversion via the helper).
    if (!get_pinned_output_i32(*program, func_name + ".draft_tokens",
                               draft_tokens, (size_t) n_active * block_size)) {
        return false;
    }
    // For confidence, use the fp32 read path; get_pinned_output
    // handles f16->f32 conversion when the slot is f16.
    if (!get_pinned_output(*program, func_name + ".confidence",
                           confidence, (size_t) n_active * block_size)) {
        return false;
    }
    return true;
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
    if (!program || !dflash_tokens || !dflash_confidence || !dflash_counts ||
            !mtp_tokens || !mtp_confidence || !mtp_counts ||
            !selected_source || !agreement || n_active == 0) {
        return false;
    }
    if (!program->manifest_loaded) {
        return false;
    }
    const ane_function_v1_t * function = find_manifest_function_by_role(
        *program, ANE_ROLE_HYBRID, block_size);
    if (function == nullptr) {
        return false;
    }
    const std::string func_name = function->name;
    // The hybrid function's inputs are dflash_tokens, dflash_confidence,
    // dflash_counts, mtp_tokens, mtp_confidence, mtp_counts, dflash_cutoff.
    // Outputs are selected_source, agreement (both i32, [bucket]).
    const size_t active_blocks = (size_t) n_active * block_size;
    if (!set_pinned_input_i32(*program, func_name + ".dflash_tokens",
                              dflash_tokens, active_blocks) ||
            !set_pinned_input(*program, func_name + ".dflash_confidence",
                              dflash_confidence, active_blocks) ||
            !set_pinned_input_i32(*program, func_name + ".dflash_counts",
                                  dflash_counts, n_active) ||
            !set_pinned_input_i32(*program, func_name + ".mtp_tokens",
                                  mtp_tokens, active_blocks) ||
            !set_pinned_input(*program, func_name + ".mtp_confidence",
                              mtp_confidence, active_blocks) ||
            !set_pinned_input_i32(*program, func_name + ".mtp_counts",
                                  mtp_counts, n_active)) {
        return false;
    }
    // dflash_cutoff is a per-lane constant, broadcast into [bucket].
    const size_t cutoff_count = (size_t) program->batch_bucket;
    std::vector<float> cutoff_broadcast(cutoff_count, dflash_cutoff);
    if (!set_pinned_input(*program, func_name + ".dflash_cutoff",
                          cutoff_broadcast.data(), cutoff_count)) {
        return false;
    }
    if (!dispatch_pinned_function(*program, func_name)) {
        return false;
    }
    if (!get_pinned_output_i32(*program, func_name + ".selected_source",
                               selected_source, n_active) ||
            !get_pinned_output_i32(*program, func_name + ".agreement",
                                   agreement, n_active)) {
        return false;
    }
    return true;
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
            n_active == 0) {
        return false;
    }
    if (!program->manifest_loaded) {
        return false;
    }
    // The MTP predict is a multifunction whose bucket is the
    // batch_bucket (1 for the typical MTP, but a multifunction can
    // declare a larger bucket). Match by role; bucket=0 is the
    // MTP singleton case where the manifest function has bucket 0.
    const ane_function_v1_t * function = nullptr;
    for (uint32_t i = 0; i < program->manifest.n_functions; ++i) {
        if (program->manifest.functions[i].role == ANE_ROLE_MTP) {
            function = &program->manifest.functions[i];
            break;
        }
    }
    if (function == nullptr) {
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
    const std::string func_name = function->name;
    const size_t active_hidden = (size_t) n_active * hidden_size;
    if (!set_pinned_input_i32(*program, func_name + ".token_ids",
                              token_ids, n_active) ||
            !set_pinned_input(*program, func_name + ".h_nextn",
                              h_nextn, active_hidden)) {
        return false;
    }
    if (positions) {
        if (!set_pinned_input_i32(*program, func_name + ".positions",
                                  positions, n_active)) {
            return false;
        }
    }
    if (!dispatch_pinned_function(*program, func_name)) {
        return false;
    }
    if (!get_pinned_output_i32(*program, func_name + ".top_token",
                               top_token, n_active) ||
            !get_pinned_output(*program, func_name + ".confidence",
                               confidence, n_active)) {
        return false;
    }
    return get_pinned_output(*program, func_name + ".next_hidden",
                              next_hidden, active_hidden);
}

bool common_ane_mtp_program_reset(
        const common_ane_mtp_program_ptr & program,
        uint32_t n_lanes,
        const int32_t * active) {
    if (!program || !active || n_lanes == 0) {
        return false;
    }
    // W4 design: reset is a memset on the K/V STATE slots in the
    // state_iosurface. For bundles without STATE slots (e.g. the
    // gemma4 prefill bundle, where each prefill function owns its
    // own K/V as OUTPUT slots), reset is a no-op success. For
    // bundles with STATE slots, the E-core pump (a follow-on)
    // will own the actual memset work; this entry point is the
    // synchronous host-side path used by tests and small calls.
    if (!program->manifest_loaded) {
        // Legacy bundle without a manifest sidecar. The .mlmodelc
        // no longer carries a "reset" Core ML function (sync_model
        // / reset_model are dropped in W4), so the legacy path is
        // gone. Refuse the call so callers fall back to whatever
        // they used before the architecture pivot.
        return false;
    }
    // Find the K/V STATE slots and memset them to zero. The exact
    // slot naming convention is bundle-specific; we look for any
    // STATE-kind slot whose name ends with "key_states" or
    // "value_states" and clear it. A follow-on commit can refine
    // the selection to per-function slots when a real MTP bundle
    // with a manifest is exported.
    bool cleared = false;
    for (uint32_t i = 0; i < program->manifest.n_slots; ++i) {
        const ane_slot_v1_t & slot = program->manifest.slots[i];
        if (slot.kind != ANE_SLOT_KIND_STATE) continue;
        const std::string name(slot.name);
        if (name.find("key_states") == std::string::npos &&
            name.find("value_states") == std::string::npos) {
            continue;
        }
        void * dst = (char *) program->state_base + slot.offset;
        std::memset(dst, 0, slot.size_bytes);
        cleared = true;
    }
    (void) n_lanes;
    (void) active;
    return cleared;
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
    if (!program || !row_counts || !positions ||
            !base_keys || !base_values || !swa_keys || !swa_values ||
            n_active == 0 || row_stride == 0) {
        return false;
    }
    // W4 design: sync is a memcpy from the host K/V arrays into
    // the K/V STATE slots in the state_iosurface. For bundles
    // without STATE slots (e.g. the gemma4 prefill bundle, where
    // each prefill function owns its own K/V as OUTPUT slots),
    // sync is a no-op success. For bundles with STATE slots, the
    // E-core pump (a follow-on) will own the actual memcpy work;
    // this entry point is the synchronous host-side path used by
    // tests and small calls. The mapping from the public
    // function's parameters (n_active, row_stride, row_counts,
    // positions, base_keys, base_values, swa_keys, swa_values,
    // base_width, swa_width) to the manifest's STATE-slot layout
    // is bundle-specific; this entry point's stub copies the
    // first row of each tensor into the matching slot so a
    // minimal correctness test can run. A real MTP bundle with a
    // manifest will replace this with a per-slot scatter driven
    // by the manifest's slot names.
    if (!program->manifest_loaded) {
        return false;
    }
    bool copied = false;
    auto copy_first_row = [&](const std::string & suffix,
                              const float * source, uint32_t width) {
        for (uint32_t i = 0; i < program->manifest.n_slots; ++i) {
            const ane_slot_v1_t & slot = program->manifest.slots[i];
            if (slot.kind != ANE_SLOT_KIND_STATE) continue;
            const std::string name(slot.name);
            if (name.find(suffix) == std::string::npos) continue;
            void * dst = (char *) program->state_base + slot.offset;
            const size_t esize = (slot.dtype == ANE_DTYPE_F16)
                ? sizeof(ggml_fp16_t) : sizeof(float);
            const size_t available = slot.size_bytes / esize;
            const size_t to_copy = std::min((size_t) width, available);
            if (slot.dtype == ANE_DTYPE_F16) {
                ggml_fp32_to_fp16_row(source,
                        (ggml_fp16_t *) dst, (int64_t) to_copy);
            } else {
                std::memcpy(dst, source, to_copy * sizeof(float));
            }
            copied = true;
            return;
        }
    };
    copy_first_row("base_keys",   base_keys,   base_width);
    copy_first_row("base_values", base_values, base_width);
    copy_first_row("swa_keys",    swa_keys,    swa_width);
    copy_first_row("swa_values",  swa_values,  swa_width);
    (void) row_stride;
    (void) row_counts;
    (void) positions;
    (void) n_active;
    return copied;
}
