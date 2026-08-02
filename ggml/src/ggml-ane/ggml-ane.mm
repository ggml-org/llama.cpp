#include "ggml-ane.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <Accelerate/Accelerate.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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

struct ggml_backend_ane_program {
    MLModel *          model        = nil;
    MLState *          state        = nil;
    dispatch_queue_t   queue        = nullptr;
    std::string        source_path;
    std::string        function_name;
    std::atomic<bool>  warm         {false};
    std::mutex         arena_mutex;
    std::unordered_map<std::string, std::unique_ptr<ggml_ane_arena_slot>> arena;

    ~ggml_backend_ane_program() {
        @autoreleasepool {
            state = nil;
            model = nil;
            queue = nullptr;
        }
    }

    // Get (creating if needed) an arena slot of the right byte size, wrapped
    // as a zero-copy MLMultiArray. The arena is reused across calls; only the
    // first call for a given (name, shape, dtype) pays the IOSurface cost.
    MLMultiArray * array_for(const std::string & name,
                             NSArray<NSNumber *> * shape,
                             MLMultiArrayDataType type,
                             NSError ** error) {
        const size_t esize = ggml_ane_multi_array_element_size(type);
        if (esize == 0) {
            return nil;
        }
        const size_t bytes = ggml_ane_shape_count(shape) * esize;
        std::lock_guard<std::mutex> lock(arena_mutex);
        auto & slot = arena[name];
        if (!slot) {
            slot = std::make_unique<ggml_ane_arena_slot>();
        }
        if (!slot->reserve(bytes)) {
            return nil;
        }
        return [[MLMultiArray alloc]
            initWithDataPointer:slot->data
                          shape:shape
                       dataType:type
                        strides:ggml_ane_contiguous_strides(shape)
                    deallocator:nil
                          error:error];
    }
};

// Warm the loaded function with zeroed inputs sized from the model's own
// input description. Mirrors warm_model() in common/ane-mtp.mm. A failed
// warmup means the bundle cannot run on this host (wrong OS, ANE missing,
// or a Core ML compile error) and the program must not be advertised.
static bool ggml_ane_program_warm(ggml_backend_ane_program * program) {
    @autoreleasepool {
        NSError * error = nil;
        NSMutableDictionary<NSString *, MLFeatureValue *> * values = [NSMutableDictionary dictionary];
        for (NSString * name in program->model.modelDescription.inputDescriptionsByName) {
            MLFeatureDescription * desc = program->model.modelDescription.inputDescriptionsByName[name];
            if (desc.type != MLFeatureTypeMultiArray) {
                return false;
            }
            MLMultiArrayConstraint * c = desc.multiArrayConstraint;
            MLMultiArray * array = [[MLMultiArray alloc] initWithShape:c.shape
                                                              dataType:c.dataType
                                                                 error:&error];
            if (!array) {
                return false;
            }
            size_t esize = ggml_ane_multi_array_element_size(c.dataType);
            if (esize == 0) {
                return false;
            }
            std::memset(array.dataPointer, 0, (size_t) array.count * esize);
            values[name] = [MLFeatureValue featureValueWithMultiArray:array];
        }
        MLDictionaryFeatureProvider * inputs =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
        if (!inputs) {
            return false;
        }
        MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
        id<MLFeatureProvider> output;
        if (program->state) {
            output = [program->model predictionFromFeatures:inputs
                                                  usingState:program->state
                                                     options:options
                                                       error:&error];
        } else {
            output = [program->model predictionFromFeatures:inputs
                                                    options:options
                                                      error:&error];
        }
        if (!output) {
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
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        // CPUAndNeuralEngine is a preference, not a command (deep-study Section
        // 1.1). The Core ML scheduler may still fall a specific op to GPU/CPU.
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        if (function_name && function_name[0] != '\0') {
            config.functionName = [NSString stringWithUTF8String:function_name];
        }
        NSError * error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:dir]
                                            configuration:config
                                                    error:&error];
        if (!model) {
            GGML_LOG_ERROR("ane: failed to load %s: %s\n", mlmodelc_dir,
                           error.localizedDescription.UTF8String ?: "unknown error");
            return nullptr;
        }
        auto * program = new ggml_backend_ane_program;
        program->model  = model;
        program->state  = [model newState];
        program->queue  = dispatch_queue_create("org.ggml.ane.backend", DISPATCH_QUEUE_SERIAL);
        program->source_path    = mlmodelc_dir;
        program->function_name  = function_name ? function_name : "";

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

// Run the bound program: feed `inputs` (by model-declared name) and read back
// `outputs` (by model-declared name). Inputs/outputs are fp32 host buffers of
// the model-declared element count; conversions to/from fp16 are handled here.
// Returns false (with a logged warning) when Core ML returns nil; the caller
// is responsible for falling back to Metal/CPU.
static bool ggml_ane_program_run(ggml_backend_ane_program * program,
                                 const std::unordered_map<std::string, const float *> & inputs,
                                 const std::vector<std::string> & output_names,
                                 const std::unordered_map<std::string, float *> & outputs) {
    if (!program || !program->warm.load()) {
        return false;
    }
    __block bool ok = false;
    dispatch_sync(program->queue, ^{
        @autoreleasepool {
            NSError * error = nil;
            NSMutableDictionary<NSString *, MLFeatureValue *> * features = [NSMutableDictionary dictionary];
            NSDictionary<NSString *, MLFeatureDescription *> * inputs_desc =
                program->model.modelDescription.inputDescriptionsByName;
            for (NSString * name in inputs_desc) {
                MLFeatureDescription * desc = inputs_desc[name];
                if (desc.type != MLFeatureTypeMultiArray) {
                    GGML_LOG_ERROR("ane: non-multiarray input %s unsupported\n", name.UTF8String);
                    return;
                }
                MLMultiArrayConstraint * c = desc.multiArrayConstraint;
                MLMultiArrayDataType type = c.dataType;
                if (type != MLMultiArrayDataTypeFloat16 && type != MLMultiArrayDataTypeFloat32) {
                    GGML_LOG_ERROR("ane: input %s dtype %ld unsupported\n", name.UTF8String, (long) type);
                    return;
                }
                NSArray<NSNumber *> * shape = c.shape;
                // Resolve any dynamic (<=0) dimension to 1 to get a concrete size.
                NSMutableArray<NSNumber *> * fixed = [shape mutableCopy];
                for (NSUInteger i = 0; i < fixed.count; ++i) {
                    if (fixed[i].integerValue <= 0) {
                        fixed[i] = @1;
                    }
                }
                const size_t count = ggml_ane_shape_count(fixed);
                MLMultiArray * array = program->array_for(name.UTF8String, fixed, type, &error);
                if (!array) {
                    GGML_LOG_ERROR("ane: arena wrap for input %s failed: %s\n", name.UTF8String,
                                   error.localizedDescription.UTF8String ?: "unknown");
                    return;
                }
                auto it = inputs.find(name.UTF8String);
                if (it != inputs.end() && it->second) {
                    ggml_ane_write_array_fp32(it->second, array, count);
                } else {
                    // Input not provided: leave the arena zeroed from the prior
                    // call's memset so the prediction still runs.
                    std::memset(array.dataPointer, 0, count * ggml_ane_multi_array_element_size(type));
                }
                features[name] = [MLFeatureValue featureValueWithMultiArray:array];
            }

            MLDictionaryFeatureProvider * provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
            if (!provider) {
                GGML_LOG_ERROR("ane: input provider build failed: %s\n",
                               error.localizedDescription.UTF8String ?: "unknown");
                return;
            }
            MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
            id<MLFeatureProvider> output;
            if (program->state) {
                output = [program->model predictionFromFeatures:provider
                                                      usingState:program->state
                                                         options:options
                                                           error:&error];
            } else {
                output = [program->model predictionFromFeatures:provider
                                                        options:options
                                                          error:&error];
            }
            if (!output) {
                // F1 failure mode: prediction-nil means Core ML could not run
                // the function on ANE (or CPU fallback). Caller must retry on
                // another backend. Surface the model error verbatim.
                GGML_LOG_ERROR("ane: Core ML prediction returned nil for %s: %s\n",
                               program->function_name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown error");
                return;
            }
            for (const std::string & out_name : output_names) {
                MLMultiArray * arr = [output featureValueForName:
                    [NSString stringWithUTF8String:out_name.c_str()]].multiArrayValue;
                if (!arr) {
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
    // we route to a bound bundle today is MUL_MAT (via a "matmul"/"linear"
    // function name). This is documented as the integration point: once the
    // conversion tool emits a function-name table, the dispatch below grows.
    switch (op->op) {
        case GGML_OP_MUL_MAT:
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
        // explicitly-bundle-mapped set is dispatched through Core ML today.
        std::vector<std::string> out_names;
        if (ggml_ane_program_dispatch_op(program, node, out_names)) {
            saw_bundle_dispatch = true;
            // TODO(ane-bundle): populate program_run inputs/outputs from the
            // node's sources and destination once a standalone matmul bundle
            // function name is exposed by the conversion tool.
            GGML_LOG_ERROR("ane: bundle dispatch for op %s is stubbed\n",
                           ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
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
// bundle. Composite ANE-NATIVE-C ops (RMS_NORM, ROPE, SOFT_MAX, SDPA,
// TILE640_*, DIAG_MASK_INF) are NOT advertised yet because their compute
// lives in a bundle function we do not dispatch today; returning true for
// them would make graph_compute fail at the "no compute path" assert.
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
        // by design: MUL_MAT, RMS_NORM, ROPE, SOFT_MAX, CONCAT, GET_ROWS,
        // FLASH_ATTN_EXT, TESSERA_PAGED_ATTN, TILE640_*, DIAG_MASK_INF,
        // GELU, ARGSORT, TOP_K, SLICE, PAD, SSM_*, RWKV_*.
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
