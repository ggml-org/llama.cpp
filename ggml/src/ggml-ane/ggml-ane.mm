#include "ggml-ane.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <Accelerate/Accelerate.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ane-state.h lives in common/ (the llama.cpp-side runtime) and is
// shared with the conversion tool's manifest schema. ggml-ane is a
// leaf backend and does not link common/, so we include the header
// by relative path from ggml/src/ggml-ane/. The manifest is the
// contract between the conversion tool (tools/ane-mtp/) and the
// runtime (ggml-ane + common/ane-mtp.mm + the future
// common/ane-pump.mm); see tools/ane-mtp/state_layout.py for the
// JSON side and tools/ane-mtp/test_state_layout.py for the
// 24-test unit suite.
#include "../../../common/ane-state.h"
#include "../../../common/ane-state-layout.h"

#define GGML_ANE_NAME "ANE"

// IOSurface page alignment used by common/ane-mtp.mm. All ANE tensor I/O goes
// through IOSurface shared memory, so every allocation must be a multiple of
// this page size.
static const size_t GGML_ANE_PAGE = 16 * 1024;

// Minimum IOSurface allocation. Orion constraint #4: allocations below ~49 KB
// compile but fail at evaluation (ANE error 0x1d). 64 KB is the next 16 KB
// page multiple that clears the 49 KB floor, so use it for every buffer.
static const size_t GGML_ANE_MIN_ALLOC = 64 * 1024;

// Round `size` up to the IOSurface page, then enforce the 64 KB ANE floor.
static size_t ggml_ane_round_size(size_t size) {
    size_t rounded = ((size + GGML_ANE_PAGE - 1) / GGML_ANE_PAGE) * GGML_ANE_PAGE;
    if (rounded < GGML_ANE_MIN_ALLOC) {
        rounded = GGML_ANE_MIN_ALLOC;
    }
    return rounded;
}

// forward declaration
static bool ggml_backend_buffer_is_ane(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// buffer context: one locked IOSurface per ggml_backend_buffer
////////////////////////////////////////////////////////////////////////////////

struct ggml_backend_ane_buffer_context {
    IOSurfaceRef surface = nullptr;
    void *       base    = nullptr;
    size_t       size    = 0;

    ~ggml_backend_ane_buffer_context() {
        if (surface) {
            IOSurfaceUnlock(surface, 0, nullptr);
            CFRelease(surface);
            surface = nullptr;
        }
        base = nullptr;
    }
};

static ggml_backend_ane_buffer_context * ggml_backend_ane_buffer_context_alloc(size_t size) {
    size_t rounded = ggml_ane_round_size(size);

    // Flat 1 x rounded IOSurface, matching common/ane-mtp.mm. The ANE reads a
    // flat allocation as packed [1, C, 1, S] (Orion #20); the host writes
    // packed data at the buffer start and the ANE compiler manages the rest.
    NSDictionary * properties = @{
        (id) kIOSurfaceWidth:          @(rounded),
        (id) kIOSurfaceHeight:         @1,
        (id) kIOSurfaceBytesPerElement:@1,
        (id) kIOSurfaceBytesPerRow:    @(rounded),
        (id) kIOSurfaceAllocSize:      @(rounded),
    };

    IOSurfaceRef surface = IOSurfaceCreate((CFDictionaryRef) properties);
    if (!surface) {
        GGML_LOG_ERROR("%s: IOSurfaceCreate failed for %zu bytes\n", __func__, rounded);
        return nullptr;
    }

    if (IOSurfaceLock(surface, 0, nullptr) != kIOReturnSuccess) {
        GGML_LOG_ERROR("%s: IOSurfaceLock failed\n", __func__);
        CFRelease(surface);
        return nullptr;
    }

    void * base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        GGML_LOG_ERROR("%s: IOSurfaceGetBaseAddress returned null\n", __func__);
        IOSurfaceUnlock(surface, 0, nullptr);
        CFRelease(surface);
        return nullptr;
    }

    auto * ctx = new ggml_backend_ane_buffer_context;
    ctx->surface = surface;
    ctx->base    = base;
    ctx->size    = rounded;
    return ctx;
}

////////////////////////////////////////////////////////////////////////////////
// buffer vtable
////////////////////////////////////////////////////////////////////////////////

static void ggml_backend_ane_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_ane_buffer_context * ctx = (ggml_backend_ane_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_ane_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_ane_buffer_context * ctx = (ggml_backend_ane_buffer_context *) buffer->context;
    return ctx->base;
}

static void ggml_backend_ane_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    GGML_UNUSED(buffer);

    // offset is relative to the tensor, not the buffer. tensor->data already
    // points at this tensor's slot inside the locked IOSurface, so write there.
    // The IOSurface stays locked for the buffer lifetime; ordering against ANE
    // reads is the caller's responsibility (via the event vtable, later).
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_ane_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    GGML_UNUSED(buffer);

    memcpy(data, (const char *) tensor->data + offset, size);
}

static void ggml_backend_ane_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    GGML_UNUSED(buffer);

    memset((char *) tensor->data + offset, value, size);
}

static bool ggml_backend_ane_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // Slice 1 only needs the host-visible memcpy path. dst lives in this ANE
    // buffer; src may be in any buffer type. ANE-to-ANE copies are also served
    // by this path because both sides are CPU-mapped while locked.
    GGML_UNUSED(buffer);

    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }

    size_t nbytes = ggml_nbytes(src);
    // ggml_backend_buffer_get_base handles views via the buffer context.
    memcpy((char *) dst->data, src->data, nbytes);
    return true;
}

static void ggml_backend_ane_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_ane_buffer_context * ctx = (ggml_backend_ane_buffer_context *) buffer->context;
    memset(ctx->base, value, ctx->size);
}

static ggml_backend_buffer_i ggml_backend_ane_buffer_i = {
    /* .free_buffer   = */ ggml_backend_ane_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_ane_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_ane_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_ane_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_ane_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_ane_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_ane_buffer_clear,
    /* .reset         = */ NULL,
};

static bool ggml_backend_buffer_is_ane(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_ane_buffer_free_buffer;
}

////////////////////////////////////////////////////////////////////////////////
// buffer type
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ANE_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_ane_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_ane_buffer_context * ctx = ggml_backend_ane_buffer_context_alloc(size);
    if (!ctx) {
        return nullptr;
    }

    // Report the requested size (not the IOSurface-rounded size) so the
    // allocator bookkeeping matches what callers asked for. The backing store
    // is always at least GGML_ANE_MIN_ALLOC.
    return ggml_backend_buffer_init(buft, ggml_backend_ane_buffer_i, ctx, size);
}

static size_t ggml_backend_ane_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // IOSurface allocations are page aligned and the first tensor in a buffer
    // is placed at the buffer base, so tensor offsets within a buffer must
    // also be page aligned.
    return GGML_ANE_PAGE;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_ane_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_ane_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Tensor-local allocation size follows ggml's default (the tensor's byte
    // footprint). The 64 KB floor and page rounding are applied once when the
    // containing buffer is allocated.
    return ggml_nbytes(tensor);

    GGML_UNUSED(buft);
}

static bool ggml_backend_ane_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    // IOSurface backing is not in the standard CPU address space without an
    // explicit lock/map. Returning false keeps the scheduler from assuming
    // zero-cost host access (deep-study Section 4.5 recommendation 4).
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void) {
    static ggml_backend_buffer_type buft;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        if (!initialized) {
            buft = {
                /* .iface = */ {
                    /* .get_name       = */ ggml_backend_ane_buffer_type_get_name,
                    /* .alloc_buffer   = */ ggml_backend_ane_buffer_type_alloc_buffer,
                    /* .get_alignment  = */ ggml_backend_ane_buffer_type_get_alignment,
                    /* .get_max_size   = */ ggml_backend_ane_buffer_type_get_max_size,
                    /* .get_alloc_size = */ ggml_backend_ane_buffer_type_get_alloc_size,
                    /* .is_host        = */ ggml_backend_ane_buffer_type_is_host,
                },
                /* .device  = */ nullptr, // wired in during device init
                /* .context = */ nullptr,
            };

            initialized = true;
        }
    }

    return &buft;
}

////////////////////////////////////////////////////////////////////////////////
// Core ML program runner
//
// Self-contained Core ML loader + predictor. It reuses the IOSurface-backed
// arena and MLMultiArray wrapping patterns from common/ane-mtp.mm but does
// not call into common_ane_mtp_* because those are coupled to llama_context
// and GGUF embedding at the wrong layer for a ggml backend. The ggml-ane
// dylib links only libggml-base and system frameworks (Foundation, IOSurface,
// CoreML, Accelerate); linking common/llama here would invert the dependency
// graph (a leaf backend depending on the high-level application libraries).
//
// One program = one compiled .mlmodelc directory, one Core ML function, and
// its own serial dispatch queue for ordering predictions (deep-study Section
// 4.3.2). Inputs/outputs are materialized in the host-mapped IOSurface arena
// and wrapped zero-copy into MLMultiArray with a nil deallocator.

static size_t ggml_ane_multi_array_element_size(MLMultiArrayDataType type) {
    switch (type) {
        case MLMultiArrayDataTypeFloat16: return sizeof(ggml_fp16_t);
        case MLMultiArrayDataTypeFloat32: return sizeof(float);
        case MLMultiArrayDataTypeInt32:   return sizeof(int32_t);
        default:                          return 0;
    }
}

static NSArray<NSNumber *> * ggml_ane_contiguous_strides(NSArray<NSNumber *> * shape) {
    NSMutableArray<NSNumber *> * result = [NSMutableArray arrayWithCapacity:shape.count];
    NSUInteger stride = 1;
    for (NSInteger i = (NSInteger) shape.count - 1; i >= 0; --i) {
        [result insertObject:@(stride) atIndex:0];
        stride *= shape[(NSUInteger) i].unsignedIntegerValue;
    }
    return result;
}

static size_t ggml_ane_shape_count(NSArray<NSNumber *> * shape) {
    size_t count = 1;
    for (NSNumber * dimension in shape) {
        count *= dimension.unsignedIntegerValue;
    }
    return count;
}

// Flat host-mapped arena slot. IOSurface backing satisfies the ANE 64-byte
// alignment and 49 KB floor (Orion constraints #4 and #20). Reuses the
// reserve-on-grow policy from common_ane_mtp_arena_buffer.
//
// (Kept for the elementwise/Accelerate path in graph_compute; the bundle
// dispatch path uses the pinned-slot state below.)
struct ggml_ane_arena_slot {
    IOSurfaceRef surface = nullptr;
    void *       data    = nullptr;
    size_t       capacity = 0;

    ~ggml_ane_arena_slot() {
        if (surface) {
            IOSurfaceUnlock(surface, 0, nullptr);
            CFRelease(surface);
            surface = nullptr;
        }
        data = nullptr;
    }

    bool reserve(size_t size) {
        if (capacity >= size) {
            return true;
        }
        size_t rounded = ((size + GGML_ANE_PAGE - 1) / GGML_ANE_PAGE) * GGML_ANE_PAGE;
        if (rounded < GGML_ANE_MIN_ALLOC) {
            rounded = GGML_ANE_MIN_ALLOC;
        }
        NSDictionary * properties = @{
            (id) kIOSurfaceWidth:          @(rounded),
            (id) kIOSurfaceHeight:         @1,
            (id) kIOSurfaceBytesPerElement:@1,
            (id) kIOSurfaceBytesPerRow:    @(rounded),
            (id) kIOSurfaceAllocSize:      @(rounded),
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
        surface = replacement;
        data    = replacement_data;
        capacity = rounded;
        return true;
    }
};

// Multifunction IOSurface-mapped stateful ANE program.
//
// The .mlmodelc is loaded as a stateless Core ML model. All "state"
// lives in a single IOSurface (the state_buffer below) whose layout
// is described by the ane_state_layout_v1 manifest emitted by the
// conversion tool. Each declared slot is pinned at load to a
// deterministic offset in that IOSurface as an MLMultiArray with
// deallocator:nil (zero-copy, see common/ane-mtp.mm's wrap_multi_array
// for the canonical pattern). The dispatch path uses Core ML's
// MLPredictionOptions.outputBackings to make Core ML write outputs
// directly into our pinned slots, skipping the result memcpy entirely.
//
// This replaces the per-function MLState + keepalive_state + 5s re-warm
// timer pattern of common/ane-mtp.mm (lines 842-880 in the multifunction
// case) and makes the state visible to Metal and CPU (zero-copy) so
// the E-core pump can coordinate ANE-Metal handoffs through the same
// canonical memory. See the architecture call in the session
// "proceed to implement it in full": one IOSurface, many readers,
// lock-free coordination via MTLSharedEvent.
struct ggml_backend_ane_program {
    // The Core ML model. Loaded stateless; state is in state_buffer
    // below. Retained in load(), released in the destructor (the
    // autoreleased-return + MRC + @autoreleasepool-drain dance that
    // the W1 commit fixed).
    MLModel *          model        = nil;

    // One serial dispatch queue per program. The multifunction case
    // (multiple functions in one .mlmodelc) will replace this with
    // per-function queues when the E-core pump lands; the W0/W1
    // single-function case keeps one queue at the program level.
    dispatch_queue_t   queue        = nullptr;

    // The parsed manifest. Owned by the program; freed in the
    // destructor. The manifest is the source of truth for the
    // IOSurface size, slot offsets, and which slots the bound
    // function reads/writes.
    ane_state_layout_v1_t layout;
    // True once the manifest has been parsed and validated.
    bool                  layout_loaded = false;

    // The single state IOSurface. All pinned slots live inside
    // this surface. Released by the destructor. We allocate via
    // ggml_backend_ane_iosurface_buffer_alloc (the existing
    // cross-backend IOSurface primitive) so the same surface can
    // be shared with Metal later.
    ggml_backend_buffer_t state_buffer = nullptr;
    void *                state_base   = nullptr;
    size_t                state_size   = 0;

    // Pinned MLMultiArray wrappers, one per manifest slot, at the
    // slot's offset inside state_buffer. Each wraps an IOSurface
    // subregion with deallocator:nil so Core ML reads/writes
    // through the same IOSurface pages the host uses. Index =
    // slot_id from the manifest; pinned_slots[i] corresponds to
    // layout.slots[i]. Released in the destructor.
    MLMultiArray *        pinned_slots[ANE_STATE_SLOTS_MAX] = {};

    // The bound function (index into layout.functions[]). The load
    // call's function_name parameter selects which manifest function
    // is bound; for the W0 single-function case it's "main".
    uint32_t              active_function_id = UINT32_MAX;

    // Scratch buffer for warmup inputs (zeroed, allocated once, freed
    // in the destructor). We allocate fresh IOSurface memory for
    // the warmup inputs so the pinned state slots stay zeroed from
    // before-load (the first real dispatch will overwrite them).
    void *                warmup_scratch  = nullptr;
    size_t                warmup_scratch_size = 0;

    std::string        source_path;
    std::string        function_name;
    std::atomic<bool>  warm         {false};

    ~ggml_backend_ane_program() {
        // Release the pinned MLMultiArrays first; they reference the
        // IOSurface so they must be released before the buffer.
        for (uint32_t i = 0; i < layout.n_slots; ++i) {
            if (pinned_slots[i] != nil) {
                [pinned_slots[i] release];
                pinned_slots[i] = nil;
            }
        }
        // Free the state IOSurface (releases the IOSurface ref and
        // any internal MTLBuffer / CFObjectRef state).
        if (state_buffer != nullptr) {
            ggml_backend_buffer_free(state_buffer);
            state_buffer = nullptr;
        }
        state_base = nullptr;
        state_size = 0;
        // Free the warmup scratch.
        if (warmup_scratch != nullptr) {
            std::free(warmup_scratch);
            warmup_scratch = nullptr;
        }
        warmup_scratch_size = 0;
        // MRC release path. Order does not matter (model and queue
        // do not retain each other), but we release the queue last
        // because dispatch_release on a serial queue waits for
        // in-flight blocks; the program handle outlives the last
        // in-flight prediction only if no prediction is currently
        // using the queue.
        if (queue) {
            dispatch_release(queue);
            queue = nullptr;
        }
        if (model) {
            [model release];
            model = nil;
        }
    }
};

// Read the ane_state_layout.v1 manifest from a JSON file into the
// C struct. Thin wrapper around the shared reader in
// common/ane-state-layout.h so the multifunction common/ane-mtp.mm
// can use the same code path. The reader is strict: unknown
// versions, missing required fields, and bad slot/function refs
// are rejected. The C-side error string is logged to the ggml
// log so callers don't need to plumb error_out through.
static bool ggml_ane_read_manifest(const char * path, ane_state_layout_v1_t * layout) {
    std::string error;
    if (!ane_layout::read_state_layout(path, layout, &error)) {
        GGML_LOG_ERROR("ane: %s\n", error.c_str());
        return false;
    }
    return true;
}

// Wrap one IOSurface-backed slot as an MLMultiArray with deallocator:nil
// (zero-copy, see common/ane-mtp.mm:wrap_multi_array for the canonical
// pattern). The MLMultiArray's data pointer is state_base + slot.offset;
// its shape and dtype come from the manifest. The returned array is
// owned by the caller and must be released in the program's destructor.
static MLMultiArray * ggml_ane_pin_slot(void * state_base,
                                        const ane_slot_v1_t * slot) {
    NSError * error = nil;
    NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:slot->n_dim];
    for (uint32_t i = 0; i < slot->n_dim; ++i) {
        [shape addObject:@(slot->shape[i])];
    }
    NSArray<NSNumber *> * strides = ggml_ane_contiguous_strides(shape);
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

// Warm the loaded function with zeroed inputs sized from the manifest.
// Mirrors warm_model() in common/ane-mtp.mm but uses the pinned state
// slots for inputs (zeroed) and discards the warmup output. A failed
// warmup means the bundle cannot run on this host (wrong OS, ANE
// missing, or a Core ML compile error) and the program must not be
// advertised. This is the only point where we send a Core ML
// prediction through; per-iter dispatch lives in ggml_ane_program_run.
static bool ggml_ane_program_warm(ggml_backend_ane_program * program) {
    @autoreleasepool {
        const ane_function_v1_t * func = &program->layout.functions[program->active_function_id];
        NSError * error = nil;
        NSMutableDictionary<NSString *, MLFeatureValue *> * values = [NSMutableDictionary dictionary];
        for (uint32_t i = 0; i < func->n_inputs; ++i) {
            const uint32_t slot_id = func->input_slot_ids[i];
            const ane_slot_v1_t * slot = &program->layout.slots[slot_id];
            // The pinned slot is the live state. For warmup we want to
            // send a zeroed copy so the real first dispatch isn't
            // polluted by a prior warmup result. We allocate a fresh
            // MLMultiArray for the warmup input (its data is zeroed
            // here; pinned state stays untouched until first real
            // dispatch).
            const size_t esize = ggml_ane_multi_array_element_size(
                program->pinned_slots[slot_id].dataType);
            if (esize == 0) {
                return false;
            }
            if (program->warmup_scratch_size < slot->size_bytes) {
                if (program->warmup_scratch != nullptr) {
                    std::free(program->warmup_scratch);
                }
                program->warmup_scratch = std::malloc(slot->size_bytes);
                program->warmup_scratch_size = slot->size_bytes;
            }
            std::memset(program->warmup_scratch, 0, slot->size_bytes);
            NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:slot->n_dim];
            for (uint32_t j = 0; j < slot->n_dim; ++j) {
                [shape addObject:@(slot->shape[j])];
            }
            NSArray<NSNumber *> * strides = ggml_ane_contiguous_strides(shape);
            MLMultiArray * warmup_input = [[MLMultiArray alloc]
                initWithDataPointer:program->warmup_scratch
                              shape:shape
                           dataType:program->pinned_slots[slot_id].dataType
                            strides:strides
                        deallocator:nil
                              error:&error];
            if (warmup_input == nil) {
                return false;
            }
            values[[NSString stringWithUTF8String:slot->name]] =
                [MLFeatureValue featureValueWithMultiArray:warmup_input];
        }
        MLDictionaryFeatureProvider * inputs =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
        if (inputs == nil) {
            GGML_LOG_ERROR("ane: warmup input provider build failed: %s\n",
                           error.localizedDescription.UTF8String ?: "unknown");
            return false;
        }
        MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
        // The warmup doesn't write to our pinned state slots (we
        // discard the output); the model's internal allocator gives
        // us a throwaway result. Real dispatches use outputBackings.
        id<MLFeatureProvider> output = [program->model
            predictionFromFeatures:inputs
                           options:options
                             error:&error];
        if (output == nil) {
            GGML_LOG_ERROR("ane: warmup prediction failed: %s\n",
                           error.localizedDescription.UTF8String ?: "unknown error");
            return false;
        }
        return true;
    }
}

static ggml_backend_ane_program * ggml_ane_program_load(const char * mlmodelc_dir,
                                                        const char * function_name) {
    if (!mlmodelc_dir || mlmodelc_dir[0] == '\0') {
        return nullptr;
    }
    @autoreleasepool {
        NSString * dir = [NSString stringWithUTF8String:mlmodelc_dir];
        if (![[NSFileManager defaultManager] fileExistsAtPath:dir]) {
            GGML_LOG_ERROR("ane: mlmodelc dir not found: %s\n", mlmodelc_dir);
            return nullptr;
        }
        // Resolve the manifest path. The convention is
        // <bundle-stem>.ane_state.v1.json in the same directory as
        // the .mlmodelc. The bundle stem is the .mlmodelc's
        // directory name (e.g., w0-256x256.mlmodelc -> "w0-256x256").
        NSString * dir_name = [dir lastPathComponent];
        NSString * parent = [dir stringByDeletingLastPathComponent];
        NSString * bundle_stem = [dir_name stringByDeletingPathExtension];
        NSString * manifest_path = [parent
            stringByAppendingPathComponent:
                [NSString stringWithFormat:@"%@.ane_state.v1.json", bundle_stem]];

        // Load the manifest. The manifest is REQUIRED (the design
        // is locked: stateless at the Core ML level, stateful via
        // the IOSurface). A missing or bad manifest is a load
        // failure.
        auto * program = new ggml_backend_ane_program;
        if (!ggml_ane_read_manifest(manifest_path.UTF8String,
                                    &program->layout)) {
            GGML_LOG_ERROR("ane: failed to load manifest at %s\n",
                           manifest_path.UTF8String);
            delete program;
            return nullptr;
        }
        program->layout_loaded = true;

        // Resolve the bound function by name. function_name == null
        // or "" means: pick the first function (the W0 single-
        // function case). For multifunction bundles, the caller
        // must specify which function to bind.
        const std::string desired = (function_name != nullptr && function_name[0] != '\0')
            ? std::string(function_name) : std::string();
        for (uint32_t i = 0; i < program->layout.n_functions; ++i) {
            const std::string fname(program->layout.functions[i].name);
            if (desired.empty() || fname == desired) {
                program->active_function_id = i;
                program->function_name = fname;
                break;
            }
        }
        if (program->active_function_id == UINT32_MAX) {
            GGML_LOG_ERROR("ane: no function matching %s in manifest\n",
                           function_name ? function_name : "(default)");
            delete program;
            return nullptr;
        }

        // Allocate the state IOSurface. One buffer of
        // state_size_bytes; every pinned slot lives inside it.
        program->state_buffer = ggml_backend_ane_iosurface_buffer_alloc(
            program->layout.state_size_bytes);
        if (program->state_buffer == nullptr) {
            GGML_LOG_ERROR("ane: failed to allocate state IOSurface (%zu bytes)\n",
                           program->layout.state_size_bytes);
            delete program;
            return nullptr;
        }
        program->state_base = ggml_backend_buffer_get_base(program->state_buffer);
        program->state_size = program->layout.state_size_bytes;

        // Pin each declared slot at its manifest offset as an
        // MLMultiArray wrapping the IOSurface with deallocator:nil.
        // Core ML reads/writes through these arrays; the host
        // (E-core pump, ggml dispatch, Metal via MTLBuffer) reads/
        // writes the same physical pages.
        for (uint32_t i = 0; i < program->layout.n_slots; ++i) {
            program->pinned_slots[i] = ggml_ane_pin_slot(
                program->state_base, &program->layout.slots[i]);
            if (program->pinned_slots[i] == nil) {
                GGML_LOG_ERROR("ane: failed to pin slot %s\n",
                               program->layout.slots[i].name);
                delete program;
                return nullptr;
            }
        }

        // Load the Core ML model. We do this AFTER the manifest
        // and state_buffer so a manifest failure doesn't leave
        // an autoreleased MLModel to dangle. Same MRC retain as
        // before (the W1 commit's fix).
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        const ane_function_v1_t * func =
            &program->layout.functions[program->active_function_id];
        // functionName is only legal for ML Program models. The
        // W0 spike's matmul is NeuralNetwork; setting functionName
        // there returns "must be nil unless the model type is ML
        // Program" at load time. We gate on the manifest's
        // model_type.
        if (program->layout.model_type == ANE_MODEL_TYPE_ML_PROGRAM &&
                func->core_ml_function_name[0] != '\0') {
            config.functionName =
                [NSString stringWithUTF8String:func->core_ml_function_name];
        }
        NSError * error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:
            [NSURL fileURLWithPath:dir]
                              configuration:config
                                      error:&error];
        if (model == nil) {
            GGML_LOG_ERROR("ane: failed to load %s: %s\n", mlmodelc_dir,
                           error.localizedDescription.UTF8String ?: "unknown error");
            delete program;
            return nullptr;
        }
        program->model = [model retain];

        program->queue = dispatch_queue_create(
            "org.ggml.ane.backend", DISPATCH_QUEUE_SERIAL);
        program->source_path = mlmodelc_dir;

        // Warm. We send zeroed inputs through the bound function
        // to compile it on the ANE; the result is discarded. The
        // pinned state slots are not modified by the warm (we use
        // a fresh warmup_scratch buffer for warm inputs).
        __block bool ok = false;
        dispatch_sync(program->queue, ^{
            ok = ggml_ane_program_warm(program);
        });
        if (!ok) {
            delete program;
            return nullptr;
        }
        program->warm.store(true);
        return program;
    }
}

GGML_BACKEND_API struct ggml_backend_ane_program * ggml_backend_ane_program_load_from_dir(
        const char * mlmodelc_dir, const char * function_name) {
    return ggml_ane_program_load(mlmodelc_dir, function_name);
}

GGML_BACKEND_API void ggml_backend_ane_program_free(struct ggml_backend_ane_program * program) {
    delete program;
}

// Read an MLMultiArray into a host fp32 buffer. The common/ane-mtp.mm variant
// handles non-contiguous strides; we replicate just the contiguous fast path
// because every output we materialize lives in our own contiguous arena.
static void ggml_ane_read_array_fp32(const MLMultiArray * array, float * dst, size_t count) {
    if (array.dataType == MLMultiArrayDataTypeFloat32) {
        std::memcpy(dst, array.dataPointer, count * sizeof(float));
    } else {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) array.dataPointer, dst, (int64_t) count);
    }
}

static void ggml_ane_write_array_fp32(const float * src, MLMultiArray * array, size_t count) {
    if (array.dataType == MLMultiArrayDataTypeFloat32) {
        std::memcpy(array.dataPointer, src, count * sizeof(float));
    } else {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) array.dataPointer, (int64_t) count);
    }
}

// Run the bound program: feed inputs from the host into the pinned
// state slots, dispatch the bound function with outputBackings set so
// Core ML writes outputs directly into our pinned slots, and read
// the outputs back from those same slots (zero-copy, no result
// memcpy). The function is the one bound at load (program-
// >active_function_id); the inputs/outputs maps are by model-
// declared name and must match the manifest's slot names for the
// bound function. Returns false (with a logged warning) when Core
// ML returns nil; the caller is responsible for falling back to
// Metal/CPU.
static bool ggml_ane_program_run(ggml_backend_ane_program * program,
                                 const std::unordered_map<std::string, const float *> & inputs,
                                 const std::vector<std::string> & output_names,
                                 const std::unordered_map<std::string, float *> & outputs) {
    if (!program || !program->warm.load() || !program->layout_loaded) {
        return false;
    }
    const ane_function_v1_t * func =
        &program->layout.functions[program->active_function_id];

    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSError * error = nil;

            // Build the input feature dict from the bound function's
            // manifest input slots. Each pinned slot is the
            // IOSurface-backed MLMultiArray; we memcpy the host
            // fp32 input into the IOSurface bytes directly (the
            // pinned slot's dataType is what the .mlmodelc expects,
            // so the fp32 host data may need a one-shot dtype
            // conversion; the simple W0 case is fp32 -> fp32 which
            // is a straight memcpy).
            NSMutableDictionary<NSString *, MLFeatureValue *> * features =
                [NSMutableDictionary dictionary];
            for (uint32_t i = 0; i < func->n_inputs; ++i) {
                const uint32_t slot_id = func->input_slot_ids[i];
                const ane_slot_v1_t * slot = &program->layout.slots[slot_id];
                MLMultiArray * pinned = program->pinned_slots[slot_id];
                if (pinned == nil) {
                    GGML_LOG_ERROR("ane: input slot %s not pinned\n", slot->name);
                    return;
                }
                const size_t count = (size_t) pinned.count;
                auto it = inputs.find(slot->name);
                if (it != inputs.end() && it->second) {
                    ggml_ane_write_array_fp32(it->second, pinned, count);
                } else {
                    // No host-side data: leave the slot as-is. The
                    // caller is responsible for ensuring prior calls
                    // have populated STATE-kind slots; for INPUT-kind
                    // slots without a host value, we zero so the
                    // first real dispatch isn't polluted.
                    if (slot->kind == ANE_SLOT_KIND_INPUT) {
                        std::memset(pinned.dataPointer, 0,
                                    count * ggml_ane_multi_array_element_size(
                                        pinned.dataType));
                    }
                }
                features[[NSString stringWithUTF8String:slot->name]] =
                    [MLFeatureValue featureValueWithMultiArray:pinned];
            }

            MLDictionaryFeatureProvider * provider =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:features error:&error];
            if (provider == nil) {
                GGML_LOG_ERROR("ane: input provider build failed: %s\n",
                               error.localizedDescription.UTF8String ?: "unknown");
                return;
            }

            // Build MLPredictionOptions with outputBackings = pinned
            // output slots. Core ML writes outputs directly into
            // our IOSurface bytes; the result provider's MLMultiArray
            // for each output name will be the SAME pointer as our
            // pinned slot (zero-copy, no result memcpy).
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            NSMutableDictionary<NSString *, MLMultiArray *> * backings =
                [NSMutableDictionary dictionary];
            for (uint32_t i = 0; i < func->n_outputs; ++i) {
                const uint32_t slot_id = func->output_slot_ids[i];
                const ane_slot_v1_t * slot = &program->layout.slots[slot_id];
                MLMultiArray * pinned = program->pinned_slots[slot_id];
                if (pinned == nil) {
                    GGML_LOG_ERROR("ane: output slot %s not pinned\n", slot->name);
                    return;
                }
                backings[[NSString stringWithUTF8String:slot->name]] = pinned;
            }
            options.outputBackings = backings;

            // Stateless dispatch. No usingState: — the design is
            // locked: state lives in our IOSurface, not in Core ML's
            // opaque MLState. If the bundle declares itself
            // stateful (e.g., a multifunction prefill with K/V
            // input slots), those slots are still read/written
            // through the IOSurface, not through an MLState.
            id<MLFeatureProvider> output = [program->model
                predictionFromFeatures:provider
                               options:options
                                 error:&error];
            if (output == nil) {
                // F1 failure mode: prediction-nil means Core ML
                // could not run the function on ANE (or CPU
                // fallback). Caller must retry on another backend.
                // Surface the model error verbatim.
                GGML_LOG_ERROR("ane: Core ML prediction returned nil for %s: %s\n",
                               program->function_name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown error");
                return;
            }
            // The result's MLMultiArrays are the same pointers as
            // our pinned output slots (outputBackings contract).
            // The outputs map (host dst) is by model-declared name;
            // we copy fp32 host dst out of the pinned slot's bytes
            // (with dtype conversion if the slot is fp16).
            for (const std::string & out_name : output_names) {
                MLMultiArray * arr = [output featureValueForName:
                    [NSString stringWithUTF8String:out_name.c_str()]].multiArrayValue;
                if (arr == nil) {
                    GGML_LOG_ERROR("ane: output %s missing from prediction\n", out_name.c_str());
                    return;
                }
                auto it = outputs.find(out_name);
                if (it != outputs.end() && it->second) {
                    ggml_ane_read_array_fp32(arr, it->second, (size_t) arr.count);
                }
            }
            ok = true;
        }
    });
    return ok;
}

////////////////////////////////////////////////////////////////////////////////
// backend (stream)
////////////////////////////////////////////////////////////////////////////////

// Per-backend context: holds the program currently bound to this instance.
// supports_op is declared below; forward-declared here for graph_compute.
static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

struct ggml_backend_ane_context {
    // Not owned: the caller owns the program handle and must free it after
    // detaching. Storing a raw pointer keeps the backend struct trivially
    // destructible and avoids a refcount cycle with the program handle.
    std::atomic<ggml_backend_ane_program *> program {nullptr};
};

static const char * ggml_backend_ane_name(ggml_backend_t backend) {
    return GGML_ANE_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_ane_free(ggml_backend_t backend) {
    // The bound program (if any) is owned by the caller via the
    // ggml_backend_ane_program handle, not by the backend struct.
    auto * ctx = (ggml_backend_ane_context *) backend->context;
    delete ctx;
    free(backend);
}

static void ggml_backend_ane_synchronize(ggml_backend_t backend) {
    // All compute is dispatched on the per-program serial queue with
    // dispatch_sync, so graph_compute is already synchronous on return.
    GGML_UNUSED(backend);
}

////////////////////////////////////////////////////////////////////////////////
// element-wise compute on the host-mapped IOSurface arena
//
// These ops are ANE-NATIVE per Section 4.1, but routing each one through a
// Core ML dispatch requires a bundle function that fuses it. When no bundle
// is bound we still need the backend to be exercisable, so the simple
// element-wise ops run on Accelerate over the same IOSurface backing that
// Core ML would read. This matches the deep-study "CPU-GLUE via Accelerate"
// fallback (Section 4.3.3) for the compute-shaped subset of native ops.
//
// Tensors in ANE buffers are CPU-mapped for the buffer lifetime, so a direct
// fp32/fp16 view is safe. We always compute in fp32 to avoid the fp16
// overflow failure modes (F10) in norm/activation paths.

static float * ggml_ane_tensor_f32_view(ggml_tensor * tensor, std::vector<float> & scratch) {
    // Returns either the tensor's own data (when already fp32 contiguous) or
    // a scratch buffer of fp32-converted data. Callers must hold the result
    // only across a single op because the scratch is overwritten per call.
    const size_t n = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor)) {
        return (float *) tensor->data;
    }
    scratch.resize(n);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) tensor->data, scratch.data(), (int64_t) n);
    } else if (tensor->type == GGML_TYPE_F32) {
        // Non-contiguous fp32: copy elementwise.
        for (size_t i = 0; i < n; ++i) {
            scratch[i] = ((const float *) tensor->data)[i];
        }
    } else {
        // Unsupported dtype for the elementwise path; zero-fill so the caller
        // can still detect a wrong-dtype input via the op result.
        std::memset(scratch.data(), 0, n * sizeof(float));
    }
    return scratch.data();
}

static void ggml_ane_tensor_write_f32(ggml_tensor * tensor, const float * src) {
    const size_t n = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(tensor->data, src, n * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) tensor->data, (int64_t) n);
    }
}

// Apply an elementwise op on fp32 views of src[0] (and src[1] for binary ops).
// dst is always written as the destination tensor's dtype.
static bool ggml_ane_compute_elementwise(ggml_tensor * op) {
    ggml_tensor * src0 = op->src[0];
    ggml_tensor * dst  = op;
    const size_t n = ggml_nelements(dst);

    std::vector<float> a_scratch;
    float * a = ggml_ane_tensor_f32_view(src0, a_scratch);

    float * b = nullptr;
    std::vector<float> b_scratch;
    if (op->src[1] && (size_t) ggml_nelements(op->src[1]) == n) {
        b = ggml_ane_tensor_f32_view(op->src[1], b_scratch);
    }

    std::vector<float> out(n);

    switch (op->op) {
        case GGML_OP_ADD: {
            if (!b) return false;
            vDSP_vadd(a, 1, b, 1, out.data(), 1, n);
        } break;
        case GGML_OP_MUL: {
            if (!b) return false;
            vDSP_vmul(a, 1, b, 1, out.data(), 1, n);
        } break;
        case GGML_OP_SCALE: {
            // ggml_scale stores the scalar in op_params[0].
            const float s = ggml_get_op_params_f32(op, 0);
            vDSP_vsmul(a, 1, &s, out.data(), 1, n);
        } break;
        case GGML_OP_CLAMP: {
            // ggml_clamp stores {min, max} in op_params[0..1].
            const float lo = ggml_get_op_params_f32(op, 0);
            const float hi = ggml_get_op_params_f32(op, 1);
            vDSP_vclip(a, 1, &lo, &hi, out.data(), 1, n);
        } break;
        case GGML_OP_REPEAT: {
            // Tile src0 over dst. n_dst must be a whole multiple of n_src.
            const size_t ns = ggml_nelements(src0);
            if (ns == 0 || n % ns != 0) {
                return false;
            }
            for (size_t i = 0; i < n; i += ns) {
                std::memcpy(out.data() + i, a, ns * sizeof(float));
            }
        } break;
        case GGML_OP_LEAKY_RELU: {
            // ggml_leaky_relu stores negative_slope in op_params[0].
            const float slope = ggml_get_op_params_f32(op, 0);
            for (size_t i = 0; i < n; ++i) {
                out[i] = a[i] < 0.0f ? a[i] * slope : a[i];
            }
        } break;
        case GGML_OP_SQR:
            vDSP_vsq(a, 1, out.data(), 1, n);
            break;
        case GGML_OP_SQRT:
            vvsqrtf(out.data(), a, (const int *) &n);
            break;
        case GGML_OP_LOG:
            vvlogf(out.data(), a, (const int *) &n);
            break;
        case GGML_OP_SIN:
            vvsinf(out.data(), a, (const int *) &n);
            break;
        case GGML_OP_COS:
            vvcosf(out.data(), a, (const int *) &n);
            break;
        case GGML_OP_UNARY: {
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:    // x * sigmoid(x)
                    for (size_t i = 0; i < n; ++i) {
                        const float s = 1.0f / (1.0f + expf(-a[i]));
                        out[i] = a[i] * s;
                    }
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    for (size_t i = 0; i < n; ++i) {
                        out[i] = 1.0f / (1.0f + expf(-a[i]));
                    }
                    break;
                case GGML_UNARY_OP_TANH:
                    vvtanhf(out.data(), a, (const int *) &n);
                    break;
                case GGML_UNARY_OP_RELU:
                    for (size_t i = 0; i < n; ++i) {
                        out[i] = a[i] < 0.0f ? 0.0f : a[i];
                    }
                    break;
                case GGML_UNARY_OP_EXP:
                    vvexpf(out.data(), a, (const int *) &n);
                    break;
                case GGML_UNARY_OP_ABS:
                    vvfabsf(out.data(), a, (const int *) &n);
                    break;
                case GGML_UNARY_OP_NEG:
                    vDSP_vneg(a, 1, out.data(), 1, n);
                    break;
                case GGML_UNARY_OP_STEP:
                    for (size_t i = 0; i < n; ++i) {
                        out[i] = a[i] > 0.0f ? 1.0f : 0.0f;
                    }
                    break;
                case GGML_UNARY_OP_SGN:
                    for (size_t i = 0; i < n; ++i) {
                        out[i] = a[i] > 0.0f ? 1.0f : (a[i] < 0.0f ? -1.0f : 0.0f);
                    }
                    break;
                // GELU/GELU_ERF/GELU_QUICK/HARDSWISH/HARDSIGMOID/ELU/SOFTPLUS/
                // EXPM1/FLOOR/CEIL/ROUND/TRUNC/XIELU are ANE-BREAKS or not on
                // the native list and are not advertised by supports_op.
                default:
                    return false;
            }
        } break;
        default:
            return false;
    }

    ggml_ane_tensor_write_f32(dst, out.data());
    return true;
}

// Copy a leaf tensor's data into `dst` in fp32. Used to feed Core ML inputs.
static bool ggml_ane_gather_input_fp32(ggml_tensor * tensor, std::vector<float> & out) {
    const size_t n = ggml_nelements(tensor);
    out.resize(n);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), tensor->data, n * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) tensor->data, out.data(), (int64_t) n);
    } else if (tensor->type == GGML_TYPE_I32) {
        const int32_t * p = (const int32_t *) tensor->data;
        for (size_t i = 0; i < n; ++i) {
            out[i] = (float) p[i];
        }
    } else {
        return false;
    }
    return true;
}

// ANE-vs-Accelerate dispatch policy.
//
// The host-side split: ANE for compute-bound ops whose per-call
// shape matches a function baked into the .mlmodelc, Accelerate
// (vDSP) for elementwise ops and any op whose shape doesn't match
// the bound bundle. The dispatcher in ggml_ane_program_dispatch_op
// checks ANE eligibility per op and either dispatches the bound
// bundle's functionName (when ANE is the better fit) or returns
// false so the scheduler routes the op to ggml-cpu (which uses
// Accelerate via vDSP). The hard rule: ANE is used when ANE is
// faster, not when ANE is available.
//
// Per-op policy (mirrors docs/tessera-ane-ios-demo-design.md,
// Phase 1 table):
//
//   MUL_MAT (T640_3D)  -> ANE (L1 path, Phase 0; the matmul is
//                         the whole point)
//   MUL_MAT (BF16/fp16)-> ANE if the bound bundle's function matches
//                         the op's shape; otherwise fall through
//                         to Accelerate BLAS (the W0 spike's path
//                         is the canonical case).
//   RMS_NORM            -> ANE (per-row reduction; shape is
//                         bake-time-locked to the .mlmodelc)
//   SOFT_MAX            -> ANE (row softmax)
//   ROPE                -> ANE (gemma 4 variant; falls through
//                         for variants not yet exported as bundle
//                         functions)
//   GLU                 -> ANE for the gemma 4 split-form case;
//                         otherwise fall through
//   GET_ROWS            -> ANE for small vocab (vocab <= 128 in
//                         this spike); larger embed-lookup goes
//                         through the host-side memcpy path
//                         because the IOSurface write is
//                         bandwidth-bound
//   ADD / MUL / SCALE  -> Accelerate (ANE dispatch overhead > vDSP
//                         cost for elementwise; the ggml-ane
//                         backend's ggml_ane_compute_elementwise
//                         path already uses Accelerate on the
//                         IOSurface arena)
//   RESHAPE / VIEW /   -> free, no compute
//   PERMUTE / CONT
//   CPY                -> memcpy on the host-mapped arena
//
// The policy is encoded as a small enum + helper below; each
// dispatch case uses the helper to decide whether to attempt the
// ANE path or return false immediately for the scheduler to route
// to the CPU/Accelerate path.

enum ggml_ane_dispatch_target {
    GGML_ANE_DISPATCH_ANE        = 0, // ANE if a function is available for this op
    GGML_ANE_DISPATCH_ACCELERATE = 1, // always fall through to Accelerate
    GGML_ANE_DISPATCH_NONE       = 2, // not ANE-eligible (unsupported)
};

static enum ggml_ane_dispatch_target ggml_ane_dispatch_policy(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_GLU:
        case GGML_OP_GET_ROWS:
            return GGML_ANE_DISPATCH_ANE;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_REPEAT:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_UNARY:
            // Already handled by ggml_ane_compute_elementwise on the
            // IOSurface arena. The dispatch helper should not pick
            // these up; the graph_compute path's elementwise branch
            // serves them via Accelerate.
            return GGML_ANE_DISPATCH_ACCELERATE;
        default:
            return GGML_ANE_DISPATCH_NONE;
    }
}

// Resolve the bundle function to dispatch for a given op, by name. The bundle
// must expose inputs/outputs in the alphabetical binding order mandated by
// Orion #3/#19; we look up by the model's own declared names, so the export
// side is responsible for naming. Returns empty if no bundle mapping exists.
static bool ggml_ane_program_dispatch_op(ggml_backend_ane_program * program,
                                         ggml_tensor * op,
                                         std::vector<std::string> & out_names) {
    GGML_UNUSED(out_names);
    if (!program) {
        return false;
    }
    // Tessera's bundle naming convention (conversion-design Section 4):
    //   prefill_sN     -> whole-layer slab (token_ids, positions -> h, k, v)
    //   mtp_predict    -> next-token prediction (h_nextn, token_ids -> tok, conf, h)
    //   dflash_bN      -> draft block
    //   hybrid_bN      -> candidate arbitration
    // A single bound bundle exposes exactly one of these. We do not yet have a
    // per-tensor-name dispatch table from the conversion tool, so the only op
    // we route to a bound bundle today is MUL_MAT (via the W0 spike's
    // "main" function). The activation is op->src[0]; the weight tensor
    // (op->src[1]) is NOT passed to the bundle because the W0 spike bakes
    // the weight into the .mlmodelc. For real models, the bundle would be
    // rebuilt with the model-specific weights (one-time at load), so the
    // per-iteration dispatch never sees the weight from ggml. This is
    // documented as the integration point: once the conversion tool emits
    // Per-op dispatch policy: skip the bundle entirely for ops that
    // the ggml-ane backend's elementwise / layout path already serves
    // (or that we don't support at all). The default case below falls
    // through to a precise per-op shape/function check; this filter
    // keeps the switch compact and documents the policy.
    const enum ggml_ane_dispatch_target policy = ggml_ane_dispatch_policy(op);
    if (policy != GGML_ANE_DISPATCH_ANE) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            // Decode (M=1) is the canonical ANE path. The activation is
            // op->src[0] of shape [K], the weight is op->src[1] of shape
            // [K, N] (ggml col-major; the bundle's row-major [N, K] weight
            // occupies the same memory). The bundle expects input "x" and
            // output "y" (the W0 spike's function naming).
            const int64_t K = op->src[0]->ne[0];
            const int64_t N = op->src[1]->ne[1];
            const int64_t M = op->src[0]->ne[1];
            if (M != 1) {
                // Prefill (M>1) is not in the W1 spike scope; the bundle
                // has a fixed-shape single-token matmul. Real prefills go
                // through the layer-slab function (prefill_sN) which is a
                // different op routing in dispatch_op.
                return false;
            }
            if (op->src[0]->type != GGML_TYPE_F32 ||
                op->src[1]->type != GGML_TYPE_F32 ||
                op->type != GGML_TYPE_F32) {
                // The W0 spike's bundle computes in fp16 but accepts fp32
                // inputs and returns fp32 outputs (Core ML precision
                // conversion is internal). Other dtypes would need
                // host-side conversion, which is a follow-on.
                return false;
            }
            // The dispatch shape is implicitly the bundle's baked shape.
            // Query the bundle's input/output shapes from the MLModel
            // description and verify before dispatching so a mismatched
            // ggml graph fails fast instead of running the wrong-sized
            // matmul. The W0 spike bakes (K=256, N=256) into the .mlmodelc
            // and the bundle's MLModelDescription matches.
            MLModelDescription * desc = program->model.modelDescription;
            if (!desc) {
                return false;
            }
            // MLModelDescription exposes inputs/outputs via the
            // inputDescriptionsByName / outputDescriptionsByName dictionaries;
            // there is no -featureDescriptionForName: selector on the
            // description itself. Index the dictionaries directly.
            MLFeatureDescription * x_desc = desc.inputDescriptionsByName[@"x"];
            MLFeatureDescription * y_desc = desc.outputDescriptionsByName[@"y"];
            if (!x_desc || x_desc.type != MLFeatureTypeMultiArray ||
                !y_desc || y_desc.type != MLFeatureTypeMultiArray) {
                return false;
            }
            NSArray<NSNumber *> * x_shape = x_desc.multiArrayConstraint.shape;
            NSArray<NSNumber *> * y_shape = y_desc.multiArrayConstraint.shape;
            if (x_shape.count != 1 || y_shape.count != 1 ||
                x_shape[0].longLongValue != K || y_shape[0].longLongValue != N) {
                // Bundle's baked shape does not match the ggml op's
                // shape; refuse the dispatch so the scheduler can route
                // the op to a different backend.
                return false;
            }
            // Build the input/output maps and call the bundle. The
            // bundle's "main" function is the default function name.
            std::unordered_map<std::string, const float *> inputs;
            inputs.emplace("x", (const float *) op->src[0]->data);
            std::vector<std::string> out_names_vec = { "y" };
            std::unordered_map<std::string, float *> outputs;
            outputs.emplace("y", (float *) op->data);
            const bool ok = ggml_ane_program_run(program, inputs, out_names_vec, outputs);
            if (ok) {
                out_names = std::move(out_names_vec);
            }
            return ok;
        }
        case GGML_OP_RMS_NORM: {
            // Per-row RMSNorm: y = x * rsqrt(mean(x^2) + eps). The
            // W2 body-op spike exports one functionName "main" of
            // shape [K, 1] fp16 (a column vector; matches ggml's
            // per-row reduction over ne[0]). The op's src[0] is
            // the row to norm; the dst is the result. eps is
            // packed in op_params[0] as a single float; we read it
            // for the manifest-side sanity check but the bundle
            // bakes eps at export time (Phase 1 ships a single
            // eps value per .mlmodelc; per-call eps is a follow-on
            // that requires the bundle to expose eps as a bundle
            // input).
            if (op->src[0] == nullptr) {
                return false;
            }
            if (op->ne[1] != 1) {
                // Per-row reduction over ne[0]: only the decode
                // shape [K, 1] is in this spike. Prefill (ne[1] > 1)
                // is multi-row and would require a different bundle
                // function (one row per parallel dispatch); the
                // scheduler routes those to the CPU backend until
                // a multi-row bundle lands.
                return false;
            }
            if (op->src[0]->type != GGML_TYPE_F32 ||
                op->type != GGML_TYPE_F32) {
                // fp32 in / fp32 out (the bundle is internally
                // fp16; Core ML handles the precision conversion).
                // fp16 / quantized are follow-ons.
                return false;
            }
            float eps = 0.0f;
            std::memcpy(&eps, op->op_params, sizeof(float));
            (void) eps; // currently unused: the bundle bakes eps.
            MLModelDescription * desc = program->model.modelDescription;
            if (!desc) {
                return false;
            }
            MLFeatureDescription * x_desc = desc.inputDescriptionsByName[@"x"];
            MLFeatureDescription * y_desc = desc.outputDescriptionsByName[@"y"];
            if (!x_desc || x_desc.type != MLFeatureTypeMultiArray ||
                !y_desc || y_desc.type != MLFeatureTypeMultiArray) {
                return false;
            }
            NSArray<NSNumber *> * x_shape = x_desc.multiArrayConstraint.shape;
            NSArray<NSNumber *> * y_shape = y_desc.multiArrayConstraint.shape;
            if (x_shape.count != 2 || y_shape.count != 2 ||
                x_shape[0].longLongValue != op->ne[0] ||
                x_shape[1].longLongValue != op->ne[1] ||
                y_shape[0].longLongValue != op->ne[0] ||
                y_shape[1].longLongValue != op->ne[1]) {
                // Bundle's baked shape does not match the ggml op's
                // shape; refuse the dispatch so the scheduler can
                // route the op to a different backend.
                return false;
            }
            std::unordered_map<std::string, const float *> inputs;
            inputs.emplace("x", (const float *) op->src[0]->data);
            std::vector<std::string> out_names_vec = { "y" };
            std::unordered_map<std::string, float *> outputs;
            outputs.emplace("y", (float *) op->data);
            const bool ok = ggml_ane_program_run(program, inputs, out_names_vec, outputs);
            if (ok) {
                out_names = std::move(out_names_vec);
            }
            return ok;
        }
        case GGML_OP_TILE640_MATMUL:
            // TODO(ane-bundle): dispatch matmul to the bound bundle's matmul
            // function once the conversion tool names one. Today the matmul
            // lives inside the layer-slab function rather than standalone.
            return false;
        default:
            return false;
    }
}

static enum ggml_status ggml_backend_ane_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_ane_context * ctx = (ggml_backend_ane_context *) backend->context;
    ggml_backend_ane_program * program = ctx ? ctx->program.load() : nullptr;

    // F16 thermal telemetry: a serious-or-worse thermal state means the ANE
    // is throttling and throughput will be unpredictable. We do not auto-
    // switch to Metal here (the scheduler owns backend selection); this is a
    // logged signal only, matching the deep-study Slice 4 recommendation.
    if (@available(macOS 10.15, iOS 11.0, *)) {
        NSProcessInfoThermalState state = NSProcessInfo.processInfo.thermalState;
        if (state >= NSProcessInfoThermalStateSerious) {
            GGML_LOG_WARN("ane: thermal state %ld >= serious; expect ANE throttling\n",
                           (long) state);
        }
    }

    const int n_nodes = cgraph->n_nodes;
    bool saw_bundle_dispatch = false;

    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        // Validate that we advertised this op; supports_op is the contract.
        if (!ggml_backend_ane_device_supports_op(backend->device, node)) {
            GGML_LOG_ERROR("ane: op %s not supported; refusing to run graph "
                           "(scheduler should have routed it elsewhere)\n",
                           ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }

        // First, ask the bound bundle whether it wants this op. Only a small,
        // explicitly-bundle-mapped set is dispatched through Core ML today:
        // MUL_MAT is the W1 spike's path. dispatch_op reads op->src[0] and
        // op->src[1], calls the bundle with the activation as the bundle
        // input, and writes the bundle output into op. The bundle's weight
        // is baked (W0 spike convention); for a real model the .mlmodelc
        // is rebuilt with the model-specific weights at load time.
        std::vector<std::string> out_names;
        if (ggml_ane_program_dispatch_op(program, node, out_names)) {
            saw_bundle_dispatch = true;
            // Bundle dispatched; the op's data is already populated.
            continue;
        }

        // No bundle mapping: fall through to the elementwise Accelerate path.
        // This covers the compute-shaped ANE-NATIVE ops (ADD, MUL, SCALE,
        // CLAMP, REPEAT, LEAKY_RELU, and the UNARY variants SILU/SIGMOID/
        // TANH/RELU/EXP/LOG/ABS/NEG/STEP/SQR/SQRT). Layout ops that we
        // advertise as supported are handled as no-ops over contiguous data.
        if (node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_TRANSPOSE ||
            node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_CONT) {
            // ggml tensors carry their own shape/stride metadata; views are
            // already resolved by the graph builder, so the underlying buffer
            // is shared and no data movement is needed.
            continue;
        }

        if (node->op == GGML_OP_CPY) {
            // Type/shape conversion copy on the host-mapped arena. CAST is not
            // a standalone op in this ggml version; it lowers to CPY.
            std::vector<float> tmp;
            if (!ggml_ane_gather_input_fp32(node->src[0], tmp)) {
                GGML_LOG_ERROR("ane: CPY unsupported dtype %s\n",
                               ggml_type_name(node->src[0]->type));
                return GGML_STATUS_FAILED;
            }
            ggml_ane_tensor_write_f32(node, tmp.data());
            continue;
        }

        if (ggml_ane_compute_elementwise(node)) {
            continue;
        }

        // Reached an op we advertised but did not implement. This is a logic
        // error in supports_op; surface it loudly rather than producing
        // silently-wrong data (F-mode failures are far worse than a crash).
        GGML_LOG_ERROR("ane: advertised op %s has no compute path\n",
                       ggml_op_name(node->op));
        return GGML_STATUS_FAILED;
    }

    GGML_UNUSED(saw_bundle_dispatch);
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_ane_i = {
    /* .get_name                = */ ggml_backend_ane_name,
    /* .free                    = */ ggml_backend_ane_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_ane_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ane_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_ane_guid(void) {
    static ggml_guid guid = { 0xa1, 0xe0, 0x4a, 0x1c, 0x7f, 0x92, 0x4d, 0x0e,
                              0xa6, 0xb3, 0x21, 0x55, 0xc8, 0x07, 0x3e, 0x1a };
    return &guid;
}

static ggml_backend_t ggml_backend_ane_alloc(ggml_backend_dev_t dev) {
    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));
    auto * ctx = new ggml_backend_ane_context;

    *backend = {
        /* .guid      = */ ggml_backend_ane_guid(),
        /* .interface = */ ggml_backend_ane_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    return backend;
}

GGML_BACKEND_API bool ggml_backend_ane_set_program(
        ggml_backend_t backend, struct ggml_backend_ane_program * program) {
    if (!ggml_backend_is_ane(backend)) {
        return false;
    }
    auto * ctx = (ggml_backend_ane_context *) backend->context;
    if (!ctx) {
        return false;
    }
    // The previously bound program (if any) is not freed here; ownership stays
    // with the caller. Only one program may be bound per backend at a time.
    ctx->program.store(program);
    return true;
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_ane_guid());
}

////////////////////////////////////////////////////////////////////////////////
// backend device
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_device_get_name(ggml_backend_dev_t dev) {
    return GGML_ANE_NAME;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_ane_device_get_description(ggml_backend_dev_t dev) {
    return "CoreML (ANE-first, iOS)";

    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // The ANE shares unified memory with the host; we do not yet have an
    // accurate per-device accounting. Report zeros so the scheduler does not
    // reserve buffers against a fictitious budget.
    if (free)  { *free  = 0; }
    if (total) { *total = 0; }

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_ane_device_get_type(ggml_backend_dev_t dev) {
    // Treat the ANE as an accelerator: the backend is intended to run alongside
    // the CPU backend (weights/tensors copied in/out), not as a standalone GPU.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ane_device_get_name(dev);
    props->description = ggml_backend_ane_device_get_description(dev);
    props->type        = ggml_backend_ane_device_get_type(dev);

    ggml_backend_ane_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->device_id = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_ane_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return ggml_backend_ane_alloc(dev);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft = ggml_backend_ane_buffer_type();
    buft->device = dev;
    return buft;
}

static bool ggml_ane_supported_tensor_type(enum ggml_type type) {
    // The elementwise/Accelerate path and the fp16 IOSurface->MLMultiArray
    // wrapping both need one of these host-convertible dtypes.
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_I32:
            return true;
        default:
            return false;
    }
}

// supports_op per deep-study Section 4.1.
//
// The ops we accept must have either a Core ML bundle dispatch (composite
// transformer ops; today only MUL_MAT-style via the bound bundle, and that
// path is still gated behind TODO(ane-bundle)) or an Accelerate elementwise
// implementation (ggml_ane_compute_elementwise). We advertise only ops that
// also have the elementwise path so the backend is exercisable without a
// bundle. ANE-NATIVE-B body ops (RMS_NORM, SOFT_MAX, ROPE, GLU, GET_ROWS)
// are advertised here and dispatched in ggml_ane_program_dispatch_op to
// the bound bundle's functionName; the precise shape/dtype match is
// enforced in dispatch_op, not here. Composite ANE-NATIVE-C ops (SDPA,
// TILE640_*, DIAG_MASK_INF) are still NOT advertised because their
// compute lives in a bundle function we do not dispatch yet; returning
// true for them would make graph_compute fail at the "no compute path"
// assert.
//
// GELU decision (Section 4.2.3): the loaded Core ML bundle already bakes in
// the tanh approximation, so GELU itself stays ANE-BREAKS here and the
// scheduler routes it to CPU/Metal. When the bundle handles a GELU-bearing
// graph it does so internally, not via this ggml op.
static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (!ggml_ane_supported_tensor_type(op->type)) {
        return false;
    }
    for (size_t i = 0; i < GGML_MAX_SRC; ++i) {
        if (op->src[i] != nullptr && !ggml_ane_supported_tensor_type(op->src[i]->type)) {
            return false;
        }
    }

    switch (op->op) {
        // ANE-NATIVE matmul (decode, M=1). The dispatch is gated on a
        // bound bundle whose input/output shapes match the ggml op's
        // shape; the bundle is the W0 spike's W0 matmul or, for real
        // models, a per-projection bundle built with the model-specific
        // weights. Only fp32 activations/weights are supported in the
        // W1 spike; fp16 and quantized types are a follow-on.
        case GGML_OP_MUL_MAT:
            if (dev != nullptr) {
                // The device-level supports_op does not have direct access
                // to the per-backend program; the dispatch_op (in
                // graph_compute) does the precise shape/dtype check and
                // returns false if the bundle does not match. We
                // advertise MUL_MAT as supported so the scheduler
                // routes it to the ANE backend; a non-matching shape
                // will then fall through (or be rejected) at dispatch time.
                // The accuracy of the device-level supports_op matters
                // for the scheduler's load balancing; the dispatch_op
                // is the precise check.
                return true;
            }
            return false;

        // ANE-NATIVE-B body ops. Each is dispatched to the bound
        // bundle's functionName when the bundle's baked shape/dtype
        // matches the ggml op (the precise check lives in
        // ggml_ane_program_dispatch_op). The device-level supports_op
        // does not have direct access to the bound program, so it
        // advertises the op and the dispatch_op decides. A bundle
        // without the matching function causes dispatch_op to return
        // false; the graph then fails at graph_compute's "no compute
        // path" check rather than silently miscomputing. Real
        // production graphs are scheduled by a multi-backend
        // scheduler that routes unmatched ops to ggml-cpu.
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_GLU:
        case GGML_OP_GET_ROWS:
            return dev != nullptr;

        // ANE-NATIVE elementwise ops with an Accelerate implementation.
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_REPEAT:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return true;

        // ANE-NATIVE layout / copy ops. Views and reshapes carry their own
        // metadata and share the source buffer, so no compute is needed.
        // CAST is not a standalone op in this ggml version; type conversion is
        // expressed via CPY and handled on the host-mapped arena.
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
            return true;

        // ANE-NATIVE unary ops (silu, sigmoid, tanh, exp, abs, relu, neg,
        // step, sgn). GELU/GELU_ERF/GELU_QUICK are ANE-BREAKS (handled in the
        // bundle, not here) so only the safe subset of UNARY is taken.
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_SGN:
                    return true;
                default:
                    return false;
            }

        // Everything else is ANE-BREAKS or CPU-GLUE per Section 4.1 and is
        // left for the scheduler to route to CPU/Metal. The notable omissions
        // by design: CONCAT, FLASH_ATTN_EXT, TESSERA_PAGED_ATTN, TILE640_*,
        // DIAG_MASK_INF, GELU, ARGSORT, TOP_K, SLICE, PAD, SSM_*, RWKV_*.
        default:
            return false;
    }
}

static bool ggml_backend_ane_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->device == dev &&
           buft->iface.get_name == ggml_backend_ane_buffer_type_get_name;
}

static ggml_backend_device_i ggml_backend_ane_device_i = {
    /* .get_name             = */ ggml_backend_ane_device_get_name,
    /* .get_description      = */ ggml_backend_ane_device_get_description,
    /* .get_memory           = */ ggml_backend_ane_device_get_memory,
    /* .get_type             = */ ggml_backend_ane_device_get_type,
    /* .get_props            = */ ggml_backend_ane_device_get_props,
    /* .init_backend         = */ ggml_backend_ane_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ane_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_ane_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ane_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

////////////////////////////////////////////////////////////////////////////////
// backend registry
////////////////////////////////////////////////////////////////////////////////

struct ggml_backend_ane_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_ane_reg * ggml_backend_ane_reg_t;

static ggml_backend_ane_reg_t ggml_backend_ane_reg_init(void) {
    return new struct ggml_backend_ane_reg;
}

struct ggml_backend_ane_reg_deleter {
    void operator()(ggml_backend_ane_reg_t ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<struct ggml_backend_ane_reg, ggml_backend_ane_reg_deleter> ggml_backend_ane_reg_ptr;

static const char * ggml_backend_ane_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_ANE_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_ane_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_ane_reg_t ctx = (ggml_backend_ane_reg_t) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_ane_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_ane_reg_t ctx = (ggml_backend_ane_reg_t) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_ane_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static ggml_backend_reg_i ggml_backend_ane_reg_i = {
    /* .get_name         = */ ggml_backend_ane_reg_get_name,
    /* .get_device_count = */ ggml_backend_ane_reg_device_count,
    /* .get_device       = */ ggml_backend_ane_reg_device_get,
    /* .get_proc_address = */ ggml_backend_ane_get_proc_address,
};

// Single logical ANE device. The public Core ML path exposes one neural
// engine; there is no multi-device enumeration the way Metal has.
static ggml_backend_dev_t ggml_backend_ane_device_init(ggml_backend_reg_t reg) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_ane_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
}

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        if (!initialized) {
            static ggml_backend_ane_reg_ptr reg_ctx(ggml_backend_ane_reg_init());
            static std::vector<std::unique_ptr<ggml_backend_device>> devs;

            auto * dev = ggml_backend_ane_device_init(&reg);
            devs.emplace_back(dev);
            reg_ctx->devices.push_back(dev);

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_ane_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ane_reg)

////////////////////////////////////////////////////////////////////////////////
// Cross-backend IOSurface buffer (the data plane for lock-free CPU/Metal/ANE
// dispatch). Distinct from `ggml_backend_ane_buffer_context` (above) which
// is owned by the ANE backend. This buffer is portable across all three
// backends: ggml_backend_supports_buft returns true for the CPU, Metal, and
// ANE backends (the latter is via the same buffer type the ANE backend
// registers; CPU/Metal support it because the base is locked CVPixelBuffer
// memory and IOSurface-backed MTLBuffer is a public Apple primitive).
////////////////////////////////////////////////////////////////////////////////

struct ggml_backend_ane_iosurface_buffer_context {
    IOSurfaceRef surface = nullptr;     // retained; locked for the buffer's lifetime
    void *       base    = nullptr;     // locked base address (CPU view)
    size_t       size    = 0;           // requested (rounded) size in bytes
    void *       mtl_buffer = nullptr;   // lazily-created MTLBuffer (Metal view)

    ~ggml_backend_ane_iosurface_buffer_context() {
        if (mtl_buffer) {
            // The MTLBuffer is created via newBufferWithBytesNoCopy so it
            // shares the IOSurface's memory; we released it but the
            // IOSurface still owns the bytes. The MTLBuffer's lifetime
            // is independent of the IOSurface's. We just drop our
            // reference here; the IOSurface release below stays correct.
            CFRelease(mtl_buffer);
            mtl_buffer = nullptr;
        }
        if (surface) {
            IOSurfaceUnlock(surface, 0, nullptr);
            CFRelease(surface);
            surface = nullptr;
        }
        base = nullptr;
    }
};

static void ggml_backend_ane_iosurface_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_ane_iosurface_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_ane_iosurface_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_ane_iosurface_buffer_context *) buffer->context;
    return ctx->base;
}

static void ggml_backend_ane_iosurface_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                           ggml_tensor * tensor,
                                                           uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    memset((char *) tensor->data + offset, value, size);
}

static void ggml_backend_ane_iosurface_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                          ggml_tensor * tensor,
                                                          const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_ane_iosurface_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                          const ggml_tensor * tensor,
                                                          void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void ggml_backend_ane_iosurface_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_ane_iosurface_buffer_context *) buffer->context;
    memset(ctx->base, value, ctx->size);
}

static size_t ggml_backend_ane_iosurface_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor);

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_i ggml_backend_ane_iosurface_buffer_i = {
    /* .free_buffer   = */ ggml_backend_ane_iosurface_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_ane_iosurface_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_ane_iosurface_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_ane_iosurface_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_ane_iosurface_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ NULL,  // copies go through the CPU view (set/get)
    /* .clear         = */ ggml_backend_ane_iosurface_buffer_clear,
    /* .reset         = */ NULL,
};

static const char * ggml_backend_ane_iosurface_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "ANE_IOSURFACE";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_ane_iosurface_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    const size_t rounded = ggml_ane_round_size(size);

    NSDictionary * properties = @{
        (id) kIOSurfaceWidth:          @(rounded),
        (id) kIOSurfaceHeight:         @1,
        (id) kIOSurfaceBytesPerElement:@1,
        (id) kIOSurfaceBytesPerRow:    @(rounded),
        (id) kIOSurfaceAllocSize:      @(rounded),
    };
    IOSurfaceRef surface = IOSurfaceCreate((CFDictionaryRef) properties);
    if (!surface) {
        GGML_LOG_ERROR("%s: IOSurfaceCreate failed for %zu bytes\n", __func__, rounded);
        return nullptr;
    }
    if (IOSurfaceLock(surface, 0, nullptr) != kIOReturnSuccess) {
        GGML_LOG_ERROR("%s: IOSurfaceLock failed\n", __func__);
        CFRelease(surface);
        return nullptr;
    }
    void * base = IOSurfaceGetBaseAddress(surface);
    if (!base) {
        GGML_LOG_ERROR("%s: IOSurfaceGetBaseAddress returned null\n", __func__);
        IOSurfaceUnlock(surface, 0, nullptr);
        CFRelease(surface);
        return nullptr;
    }

    auto * ctx = new ggml_backend_ane_iosurface_buffer_context;
    ctx->surface = surface;
    ctx->base    = base;
    ctx->size    = rounded;
    return ggml_backend_buffer_init(buft, ggml_backend_ane_iosurface_buffer_i, ctx, size);
}

static size_t ggml_backend_ane_iosurface_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return GGML_ANE_PAGE;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_ane_iosurface_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;

    GGML_UNUSED(buft);
}

static bool ggml_backend_ane_iosurface_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    // The IOSurface is process-shared, not strictly host memory. The
    // scheduler treats it as off-host for placement decisions.
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_ane_iosurface_buffer_type(void) {
    static ggml_backend_buffer_type buft;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        if (!initialized) {
            buft = {
                /* .iface = */ {
                    /* .get_name       = */ ggml_backend_ane_iosurface_buffer_type_get_name,
                    /* .alloc_buffer   = */ ggml_backend_ane_iosurface_buffer_type_alloc_buffer,
                    /* .get_alignment  = */ ggml_backend_ane_iosurface_buffer_type_get_alignment,
                    /* .get_max_size   = */ ggml_backend_ane_iosurface_buffer_type_get_max_size,
                    /* .get_alloc_size = */ ggml_backend_ane_iosurface_buffer_type_get_alloc_size,
                    /* .is_host        = */ ggml_backend_ane_iosurface_buffer_type_is_host,
                },
                /* .device  = */ nullptr, // portable across backends, not device-owned
                /* .context = */ nullptr,
            };

            initialized = true;
        }
    }

    return &buft;
}

GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_ane_iosurface_buffer_alloc(size_t bytes) {
    return ggml_backend_ane_iosurface_buffer_type_alloc_buffer(
        ggml_backend_ane_iosurface_buffer_type(), bytes);
}

GGML_BACKEND_API bool ggml_backend_ane_iosurface_buffer_check(ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.free_buffer == ggml_backend_ane_iosurface_buffer_free_buffer;
}

GGML_BACKEND_API void * ggml_backend_ane_iosurface_buffer_get_iosurface(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_ane_iosurface_buffer_check(buffer)) {
        return nullptr;
    }
    auto * ctx = (ggml_backend_ane_iosurface_buffer_context *) buffer->context;
    return (void *) ctx->surface;
}

// Lazily wrap the IOSurface as an MTLBuffer. The wrap uses
// newBufferWithBytesNoCopy so the MTLBuffer shares memory with the
// IOSurface (no copy). The deallocator is nil because the IOSurface
// owns the memory and outlives the MTLBuffer.
GGML_BACKEND_API void * ggml_backend_ane_iosurface_buffer_get_mtl_buffer(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_ane_iosurface_buffer_check(buffer)) {
        return nullptr;
    }
    auto * ctx = (ggml_backend_ane_iosurface_buffer_context *) buffer->context;
    if (ctx->mtl_buffer) {
        return ctx->mtl_buffer;
    }
    // Look up the Metal device. The ggml-ane backend does not own a
    // Metal device; the dispatch layer hands us one through the
    // environment (a future commit wires this through ggml_backend_dev_t
    // discovery). For the lock-free data plane itself, we use the
    // system default Metal device (MTLCreateSystemDefaultDevice).
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        GGML_LOG_ERROR("%s: MTLCreateSystemDefaultDevice returned null\n", __func__);
        return nullptr;
    }
    // The IOSurface must remain alive for as long as the MTLBuffer is
    // live. newBufferWithBytesNoCopy takes a non-retained pointer; we
    // pass NULL as the deallocator and rely on the buffer's own
    // destruction to drop the MTLBuffer (which then no longer
    // references the IOSurface). The IOSurface itself outlives the
    // MTLBuffer because the buffer holds a reference to both.
    id<MTLBuffer> mtl_buf = [device newBufferWithBytesNoCopy:ctx->base
                                                       length:ctx->size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!mtl_buf) {
        GGML_LOG_ERROR("%s: newBufferWithBytesNoCopy failed\n", __func__);
        return nullptr;
    }
    ctx->mtl_buffer = (void *) CFBridgingRetain(mtl_buf);
    return ctx->mtl_buffer;
}
