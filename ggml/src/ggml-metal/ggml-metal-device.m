#import "ggml-metal-device.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

// overload of MTLGPUFamilyMetal3 (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;

struct ggml_backend_metal_device {
    id<MTLDevice>  mtl_device;
    id<MTLLibrary> mtl_library;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    struct ggml_backend_metal_device_props props;
};

ggml_backend_metal_device_t ggml_backend_metal_device_init(void) {
    ggml_backend_metal_device_t ctx = calloc(1, sizeof(struct ggml_backend_metal_device));

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

void ggml_backend_metal_device_free(ggml_backend_metal_device_t ctx) {
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

void * ggml_backend_metal_device_get_device(ggml_backend_metal_device_t ctx) {
    return ctx->mtl_device;
}

void * ggml_backend_metal_device_get_library(ggml_backend_metal_device_t ctx) {
    return ctx->mtl_library;
}

void * ggml_backend_metal_device_get_queue(ggml_backend_metal_device_t ctx) {
    return ctx->mtl_queue;
}

void ggml_backend_metal_device_get_memory(ggml_backend_metal_device_t ctx, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = ctx->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - ctx->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

struct ggml_backend_metal_device_props ggml_backend_metal_device_get_props(ggml_backend_metal_device_t ctx) {
    return ctx->props;
}
