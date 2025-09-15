#import "ggml-metal-device.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetal3 (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

struct ggml_metal_device {
    id<MTLDevice>  mtl_device;
    id<MTLLibrary> mtl_library;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    struct ggml_metal_device_props props;
};

ggml_metal_device_t ggml_metal_device_init(void) {
    ggml_metal_device_t ctx = calloc(1, sizeof(struct ggml_metal_device));

    assert(ctx != NULL);

    if (ctx->mtl_device == nil) {
        ctx->mtl_device = MTLCreateSystemDefaultDevice();

        if (ctx->mtl_device) {
            ctx->mtl_queue = [ctx->mtl_device newCommandQueue];
            if (ctx->mtl_queue == nil) {
                GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            }

            ctx->props.has_simdgroup_reduction  = [ctx->mtl_device supportsFamily:MTLGPUFamilyApple7];
            ctx->props.has_simdgroup_reduction |= [ctx->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

            ctx->props.has_simdgroup_mm = [ctx->mtl_device supportsFamily:MTLGPUFamilyApple7];
            ctx->props.has_unified_memory = ctx->mtl_device.hasUnifiedMemory;

            ctx->props.has_bfloat  = [ctx->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            ctx->props.has_bfloat |= [ctx->mtl_device supportsFamily:MTLGPUFamilyApple6];

            ctx->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            ctx->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            ctx->props.use_shared_buffers = ctx->props.has_unified_memory;

            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                ctx->props.use_shared_buffers = false;
            }

            ctx->props.supports_gpu_family_apple7 = [ctx->mtl_device supportsFamily:MTLGPUFamilyApple7];

            ctx->props.max_buffer_size            = ctx->mtl_device.maxBufferLength;
            ctx->props.max_working_set_size       = ctx->mtl_device.recommendedMaxWorkingSetSize;
            ctx->props.max_theadgroup_memory_size = ctx->mtl_device.maxThreadgroupMemoryLength;

            strncpy(ctx->props.name, [[ctx->mtl_device name] UTF8String], sizeof(ctx->props.name) - 1);

            // load library
            //
            // - first check if the library is embedded
            // - then check if the library is in the bundle
            // - if not found, load the source and compile it
            // - if that fails, return NULL
            //
            // TODO: move to a function
            {
                const int64_t t_start = ggml_time_us();

                NSError * error = nil;
                NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
                GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

                extern const char ggml_metallib_start[];
                extern const char ggml_metallib_end[];

                src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];

#else

#ifdef SWIFT_PACKAGE
                NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
                NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

                NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
                if (path_lib == nil) {
                    // Try to find the resource in the directory where the current binary located.
                    NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
                    NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

                    NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
                    if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                       GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                       NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                       if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                           // Optionally, if this is a symlink, try to resolve it.
                           path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                           if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                               // It is a relative path, adding the binary directory as directory prefix.
                               path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                           }
                           if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                               // Link to the resource could not be resolved.
                               path_lib_default = nil;
                           } else {
                               GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                           }
                       }
                    } else {
                        // The resource couldn't be found in the binary's directory.
                        path_lib_default = nil;
                    }

                    path_lib = path_lib_default;
                }

                if (path_lib != nil) {
                    // pre-compiled library found
                    NSURL * libURL = [NSURL fileURLWithPath:path_lib];
                    GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

                    ctx->mtl_library = [ctx->mtl_device newLibraryWithURL:libURL error:&error];
                    if (error) {
                        GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    }
                } else {
                    GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

                    NSString * path_source;
                    NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

                    GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

                    if (path_resource) {
                        path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
                    } else {
                        path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
                    }

                    if (path_source == nil) {
                        GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                        path_source = @"ggml-metal.metal";
                    }

                    GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

                    src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
                    if (error) {
                        GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    }
                }
#endif

                if (!ctx->mtl_library) {
                    @autoreleasepool {
                        // dictionary of preprocessor macros
                        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                        if (ctx->props.has_bfloat) {
                            [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                        }

#if GGML_METAL_EMBED_LIBRARY
                        [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

                        MTLCompileOptions * options = [MTLCompileOptions new];
                        options.preprocessorMacros = prep;

                        //[options setFastMathEnabled:false];

                        ctx->mtl_library = [ctx->mtl_device newLibraryWithSource:src options:options error:&error];
                        if (error) {
                            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                        }

#if !__has_feature(objc_arc)
                        [options release];
#endif
                    }
                }

#if GGML_METAL_EMBED_LIBRARY
                [src release];
#endif // GGML_METAL_EMBED_LIBRARY

                GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
            }

            // --------------------------------------------------

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, ctx->props.name);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([ctx->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([ctx->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([ctx->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, ctx->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, ctx->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, ctx->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, ctx->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, ctx->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, ctx->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, ctx->props.max_working_set_size / 1e6);
            }
#endif
        }
    }

    return ctx;
}

void ggml_metal_device_free(ggml_metal_device_t ctx) {
    assert(ctx != NULL);

    if (ctx->mtl_library) {
        [ctx->mtl_library release];
        ctx->mtl_library = nil;
    }

    if (ctx->mtl_queue) {
        [ctx->mtl_queue release];
        ctx->mtl_queue = nil;
    }

    if (ctx->mtl_device) {
        [ctx->mtl_device release];
        ctx->mtl_device = nil;
    }

    free(ctx);
}

void * ggml_metal_device_get_device(ggml_metal_device_t ctx) {
    return ctx->mtl_device;
}

void * ggml_metal_device_get_library(ggml_metal_device_t ctx) {
    return ctx->mtl_library;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t ctx) {
    return ctx->mtl_queue;
}

void ggml_metal_device_get_memory(ggml_metal_device_t ctx, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = ctx->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - ctx->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

bool ggml_metal_device_supports_op(ggml_metal_device_t ctx, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = ctx->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = ctx->props.has_simdgroup_reduction;
    const bool has_bfloat              = ctx->props.has_bfloat;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                    return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ACC:
        case GGML_OP_REPEAT:
        case GGML_OP_SCALE:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LOG:
            return false; // TODO: implement
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && (op->ne[0] % 4 == 0 && ggml_is_contiguous_1(op->src[0]));
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
            return has_simdgroup_reduction && (op->ne[0] % 4 == 0 && ggml_is_contiguous_1(op->src[0]));
        case GGML_OP_ROPE:
            return true;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_POOL_1D:
            return false;
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256) {
                return false;
            }
            if (op->src[0]->ne[0] == 576) {
                // DeepSeek sizes
                // TODO: disabled for now, until optmized
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return has_simdgroup_reduction &&
                (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F32);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_GET_ROWS:
            {
                return op->ne[3] == 1;
            }
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                };
            }
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t ctx) {
    return &ctx->props;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data; // TODO: https://github.com/ggml-org/llama.cpp/pull/15985
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicity "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device objects
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t ctx) {
    ctx->rset = nil;

    if (!ctx->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = ctx->n_buffers;

        NSError * error;
        ctx->rset = [ctx->device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < ctx->n_buffers; i++) {
            [ctx->rset addAllocation:ctx->buffers[i].metal];
        }

        [ctx->rset commit];
        [ctx->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t ctx) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (ctx->rset) {
            [ctx->rset endResidency];
            [ctx->rset removeAllAllocations];
            [ctx->rset release];
        }
    }
#else
    GGML_UNUSED(ctx);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t ctx, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // dummy, non-NULL value - we'll populate this after creating the Metal buffer below
        res->all_data = (void *) 0x000000400ULL;
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->device = ggml_metal_device_get_device(ctx);
    res->queue  = ggml_metal_device_get_queue (ctx);

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers &&shared) {
                res->buffers[0].metal = [res->device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];

                res->all_data = (void *) (res->buffers[0].metal.gpuAddress);
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t ctx, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    res->device = ggml_metal_device_get_device(ctx);
    res->queue  = ggml_metal_device_get_queue (ctx);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t ctx) {
    for (int i = 0; i < ctx->n_buffers; i++) {
        [ctx->buffers[i].metal release];
    }

    ggml_metal_buffer_rset_free(ctx);

    if (ctx->is_shared) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ctx->all_data, ctx->all_size);
#else
        free(ctx->all_data);
#endif
    }

    free(ctx);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t ctx) {
    return ctx->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t ctx) {
    return ctx->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t ctx, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (ctx->is_shared) {
        memset((char *)tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id buf_dst = ggml_metal_buffer_get_id(ctx, tensor);
        buf_dst.offs += offset;

        id<MTLCommandQueue>  queue   = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf_dst.metal
                          range:NSMakeRange(buf_dst.offs, buf_dst.offs + size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (ctx->is_shared) {
        memcpy((char *)tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        id<MTLBuffer> buf_src = [ctx->device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        // dst
        struct ggml_metal_buffer_id buf_dst = ggml_metal_buffer_get_id(ctx, tensor);
        buf_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandQueue>  queue   = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:buf_dst.metal
                  destinationOffset:buf_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        //[cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (ctx->is_shared) {
        memcpy(data, (const char *)tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id buf_src = ggml_metal_buffer_get_id(ctx, tensor);
        buf_src.offs += offset;

        // dst
        id<MTLBuffer> buf_dst = [ctx->device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        id<MTLCommandQueue>  queue   = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src.metal
                       sourceOffset:buf_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t ctx, uint8_t value) {
    if (ctx->is_shared) {
        memset(ctx->all_data, value, ctx->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandQueue>  queue   = ctx->queue;
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:ctx->buffers[0].metal
                          range:NSMakeRange(0, ctx->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t ctx, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, ctx->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, ctx->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            res.metal = ctx->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}
