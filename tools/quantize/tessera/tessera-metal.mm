//
// tessera-metal.mm
//
// Host-side Metal dispatch for the Tessera quantize pipeline. Creates the
// MTLDevice / MTLCommandQueue, loads the kernel library (either the
// build-time-compiled tessera-metal.metallib, or by compiling the embedded
// source string at runtime as a fallback), and exposes the four entry points
// declared in tessera-metal.h.
//
// Thread-safety: ts_metal_init and ts_metal_shutdown take a global mutex; the
// per-candidate eval entry points each build their own MTLCommandBuffer and
// are safe to call concurrently from multiple GA threads (MTLCommandQueue is
// documented thread-safe for encoding).
//

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tessera-metal.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

// Embedded Metal source, compiled at build time into tessera-metal.metallib
// and also embedded here as the runtime-compile fallback (used when the
// .metallib cannot be found on disk, e.g. when running straight from the
// build dir without install). Keep in sync with tessera-metal.metal.
#include "tessera-metal-source.h"

// ---------------------------------------------------------------------------
// global context
//
// These struct definitions live in the global namespace so they match the
// `typedef struct ts_metal_context ts_metal_context_t;` declarations in the
// public header (which forward-declares them). The helper functions that do
// not need to be public live in the anonymous namespace further below.
// ---------------------------------------------------------------------------

struct ts_metal_context {
    id<MTLDevice>       device  = nil;
    id<MTLCommandQueue> queue   = nil;
    id<MTLLibrary>      library = nil;

    // cached pipelines (created once at init). Pipeline creation is expensive,
    // so we hoist them out of the per-candidate hot path.
    id<MTLComputePipelineState> sct_reduce     = nil;
    id<MTLComputePipelineState> sct_ternarize  = nil;
    id<MTLComputePipelineState> dmr            = nil;
    id<MTLComputePipelineState> awq_threshold  = nil;
    id<MTLComputePipelineState> awq_grid       = nil;

    int max_threadgroup_mem = 0;  // device.threadgroupMemoryLength
};

struct ts_metal_weights {
    int64_t out_dim = 0;
    int64_t in_dim  = 0;

    // GPU-resident weight + activation buffers. The weight buffer is created
    // once (per layer) and reused across all candidate evals of that layer.
    id<MTLBuffer> W          = nil;  // [out_dim * in_dim] float
    id<MTLBuffer> act        = nil;  // [in_dim] act_scales (may be nil)
    id<MTLBuffer> act2       = nil;  // [in_dim] act_scales^2 (may be nil)
    float         inv_median = 0.0f; // 1/median(|act|), precomputed
};

namespace {

ts_metal_context * g_ctx = nullptr;
std::mutex         g_init_mutex;
std::atomic<int>   g_avail{0};   // -1 = untried, 0 = unavailable, 1 = available

// once-init the atomic to "untried"
struct once_init { once_init() { g_avail.store(-1); } };
once_init g_once;

// Mirror of the metal-side partial structs (keep layout identical).
struct ts_sct_partial_metal { float abs_sum; float maxabs; };
struct ts_dmr_partial_metal { float mse_sum; float n_count; };
struct ts_awq_partial_metal { float err_sum; float n_count; };

#define TS_PAGE_SIZE      640
#define TS_LANE_SIZE      20
#define TS_LANES_PER_PAGE 32
#define TS_SCT_THREADS    256
#define TS_DMR_MAX_ROW    8192

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// Load the kernel library, trying in order:
//   1. tessera-metal.metallib next to the host executable (install layout)
//   2. tessera-metal.metallib in the build output dir (developer layout)
//   3. compile the embedded source string at runtime
// Returns nil and sets err on failure.
static id<MTLLibrary> load_library(id<MTLDevice> device, NSError ** err) {
    NSBundle * main = [NSBundle mainBundle];

    // 1. installed next to the executable
    NSURL * lib_url = [main URLForResource:@"tessera-metal" withExtension:@"metallib"];
    if (lib_url != nil) {
        id<MTLLibrary> lib = [device newLibraryWithURL:lib_url error:err];
        if (lib != nil) return lib;
    }

    // 2. developer / build-output layout: look in the same dir as the
    //    executable binary (CMAKE_RUNTIME_OUTPUT_DIRECTORY).
    const char * argv0 = [[[NSProcessInfo processInfo] arguments] firstObject]
                          .fileSystemRepresentation;
    if (argv0 != nullptr) {
        NSString * bindir = [@(argv0) stringByDeletingLastPathComponent];
        NSString * path   = [bindir stringByAppendingPathComponent:@"tessera-metal.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            NSURL * url = [NSURL fileURLWithPath:path];
            id<MTLLibrary> lib = [device newLibraryWithURL:url error:err];
            if (lib != nil) return lib;
        }
    }

    // 3. fall back to compiling the embedded source string. This is slow
    //    (sub-second) but means the path works with no .metallib on disk.
    NSString * src = [NSString stringWithUTF8String:ts_metal_kernel_source()];
    if (src == nil) {
        if (err != nil) {
            *err = [NSError errorWithDomain:@"tessera-metal"
                                       code:2
                                   userInfo:@{NSLocalizedDescriptionKey: @"embedded source is empty"}];
        }
        return nil;
    }
    MTLCompileOptions * opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    opts.fastMathEnabled = YES;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:err];
    return lib;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                                 id<MTLLibrary> lib,
                                                 NSString * name,
                                                 NSError ** err) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) {
        if (err != nil) {
            *err = [NSError errorWithDomain:@"tessera-metal"
                                       code:3
                                   userInfo:@{NSLocalizedDescriptionKey:
                                              [NSString stringWithFormat:@"kernel '%@' not found in library", name]}];
        }
        return nil;
    }
    return [device newComputePipelineStateWithFunction:fn error:err];
}

// Create a private (GPU-resident) buffer initialized from host bytes.
// MTLResourceStorageModePrivate is incompatible with newBufferWithBytes, so
// stage through a shared buffer and blit. Used for the per-layer weight
// upload so subsequent candidate evals read over GPU bandwidth.
static id<MTLBuffer> new_private_from_bytes(id<MTLDevice> device,
                                            id<MTLCommandQueue> queue,
                                            const void * bytes,
                                            NSUInteger length) {
    id<MTLBuffer> shared = [device newBufferWithBytes:bytes
                                               length:length
                                              options:MTLResourceStorageModeShared];
    if (shared == nil) return nil;
    id<MTLBuffer> priv = [device newBufferWithLength:length
                                            options:MTLResourceStorageModePrivate];
    if (priv == nil) return nil;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:shared sourceOffset:0
                      toBuffer:priv destinationOffset:0
                          size:length];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return priv;
}

// dispatch a compute encoder and commit+wait, shared by the entry points.
static bool dispatch_and_wait(id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> pipeline,
                              NSArray<id<MTLBuffer>> * buffers,
                              MTLSize tg_per_grid,
                              MTLSize threads_per_tg,
                              NSUInteger tg_mem_bytes = 0) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) return false;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (enc == nil) return false;
        [enc setComputePipelineState:pipeline];
        NSUInteger idx = 0;
        for (id<MTLBuffer> b in buffers) {
            // bind nil too so buffer indices stay aligned with the kernel
            // signature (Metal accepts a nil buffer binding).
            [enc setBuffer:b offset:0 atIndex:idx];
            idx++;
        }
        if (tg_mem_bytes > 0) {
            [enc setThreadgroupMemoryLength:tg_mem_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:tg_per_grid threadsPerThreadgroup:threads_per_tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return true;
    }
}

// median of the finite, strictly-positive entries; mirrors the CPU
// ts_median_finite_positive so AWQ normalization matches exactly.
static float median_finite_positive(const float * x, int64_t n) {
    std::vector<float> v;
    v.reserve((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        float xv = x[i];
        if (std::isfinite(xv) && xv > 0.0f) v.push_back(xv);
    }
    if (v.empty()) return 0.0f;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if (v.size() % 2 == 0) {
        float lo = *std::max_element(v.begin(), v.begin() + mid);
        m = 0.5f * (lo + m);
    }
    return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

extern "C" {

int ts_metal_available(void) {
    // Env-var kill switch for isolating Metal-related crashes. Set
    // TS_METAL_DISABLE=1 to force the CPU fallback path.
    if (g_avail.load() == 1) {
        const char * dis = getenv("TS_METAL_DISABLE");
        if (dis && dis[0] == '1') return 0;
        return 1;
    }
    return 0;
}

int ts_metal_init(void) {
    // fast path
    if (g_avail.load() == 1) return 0;

    std::lock_guard<std::mutex> lk(g_init_mutex);
    if (g_avail.load() == 1) return 0;
    if (g_ctx != nullptr) return 1;  // previously failed in a weird state

    ts_metal_context * ctx = new ts_metal_context();

    @autoreleasepool {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            delete ctx;
            g_avail.store(0);
            return 1;
        }
        ctx->queue  = [ctx->device newCommandQueue];
        ctx->max_threadgroup_mem = (int)ctx->device.maxThreadgroupMemoryLength;

        NSError * err = nil;
        ctx->library = load_library(ctx->device, &err);
        if (ctx->library == nil) {
            delete ctx;
            g_avail.store(0);
            return 2;
        }

        ctx->sct_reduce    = make_pipeline(ctx->device, ctx->library,
                                           @"ts_metal_sct_reduce",    &err);
        ctx->sct_ternarize = make_pipeline(ctx->device, ctx->library,
                                           @"ts_metal_sct_ternarize", &err);
        ctx->dmr           = make_pipeline(ctx->device, ctx->library,
                                           @"ts_metal_dmr",           &err);
        ctx->awq_threshold = make_pipeline(ctx->device, ctx->library,
                                           @"ts_metal_awq_threshold", &err);
        ctx->awq_grid      = make_pipeline(ctx->device, ctx->library,
                                           @"ts_metal_awq_grid",      &err);
        if (ctx->sct_reduce == nil || ctx->sct_ternarize == nil ||
            ctx->dmr == nil || ctx->awq_threshold == nil || ctx->awq_grid == nil) {
            delete ctx;
            g_avail.store(0);
            return 3;
        }
    }

    g_ctx = ctx;
    g_avail.store(1);
    return 0;
}

void ts_metal_shutdown(void) {
    std::lock_guard<std::mutex> lk(g_init_mutex);
    delete g_ctx;
    g_ctx = nullptr;
    g_avail.store(-1);
}

ts_metal_weights_t * ts_metal_upload_weights(const float * weights,
                                             const float * act_scales,
                                             int64_t out_dim,
                                             int64_t in_dim) {
    if (!ts_metal_available() || g_ctx == nullptr ||
        weights == nullptr || out_dim <= 0 || in_dim <= 0) {
        return nullptr;
    }

    ts_metal_weights_t * w = new ts_metal_weights_t;
    w->out_dim = out_dim;
    w->in_dim  = in_dim;

    @autoreleasepool {
        const NSUInteger wbytes = (NSUInteger)(out_dim * in_dim) * sizeof(float);
        // GPU-resident weight buffer: stage through shared then blit to
        // private so candidate evals read it over GPU bandwidth, not the
        // shared CPU bus.
        w->W = new_private_from_bytes(g_ctx->device, g_ctx->queue, weights, wbytes);
        if (w->W == nil) {
            delete w;
            return nullptr;
        }

        if (act_scales != nullptr) {
            const NSUInteger sbytes = (NSUInteger)in_dim * sizeof(float);
            w->act = new_private_from_bytes(g_ctx->device, g_ctx->queue,
                                            act_scales, sbytes);
            std::vector<float> act2((size_t)in_dim);
            for (int64_t c = 0; c < in_dim; c++) {
                float a = act_scales[c];
                act2[(size_t)c] = a * a;
            }
            w->act2 = new_private_from_bytes(g_ctx->device, g_ctx->queue,
                                             act2.data(), sbytes);

            float med = median_finite_positive(act_scales, in_dim);
            w->inv_median = (med > 0.0f) ? (1.0f / med) : 0.0f;
        }
    }
    return w;
}

void ts_metal_release_weights(ts_metal_weights_t * w) {
    delete w;
}

// -----------------------------------------------------------------------
// Kernel 1: scale + clip + ternarize
// -----------------------------------------------------------------------

int ts_metal_scale_clip_ternarize(ts_metal_weights_t * w,
                                  const float * wscale,
                                  float clip,
                                  float * ws_out,
                                  float * core_out,
                                  int8_t * ternary_out,
                                  float * global_amp_out) {
    if (!ts_metal_available() || g_ctx == nullptr || w == nullptr ||
        wscale == nullptr || ws_out == nullptr || core_out == nullptr ||
        ternary_out == nullptr || global_amp_out == nullptr) {
        return 1;
    }
    // threadgroup scratch bounds: kernel uses TS_SCT_THREADS per row and
    // strides the row; rows can be arbitrarily wide. ws/core are staged in
    // device buffers and copied back at the end.
    const int64_t out_dim = w->out_dim;
    const int64_t in_dim  = w->in_dim;
    const NSUInteger n    = (NSUInteger)(out_dim * in_dim);

    int rc = 0;
    @autoreleasepool {
        id<MTLDevice> dev = g_ctx->device;

        // wscale -> device buffer (private via blit; small but read by GPU)
        id<MTLBuffer> wscale_b = new_private_from_bytes(
            dev, g_ctx->queue, wscale, (NSUInteger)in_dim * sizeof(float));
        // ws/core outputs live on the GPU during the dispatch
        id<MTLBuffer> ws_b   = [dev newBufferWithLength:n * sizeof(float)
                                               options:MTLResourceStorageModePrivate];
        id<MTLBuffer> core_b = [dev newBufferWithLength:n * sizeof(float)
                                               options:MTLResourceStorageModePrivate];
        // partials + clip_limits + threshold are shared (readable from CPU)
        id<MTLBuffer> partials_b = [dev newBufferWithLength:(NSUInteger)out_dim *
                                                       sizeof(ts_sct_partial_metal)
                                                   options:MTLResourceStorageModeShared];
        if (wscale_b == nil || ws_b == nil || core_b == nil || partials_b == nil) {
            return 1;
        }

        // ---- phase 0: scale + reduce ----
        const bool do_clip = (clip > 0.0f && clip < 1.0f);
        // struct ts_sct_args { uint32 out_dim; uint32 in_dim; float clip; uint32 do_clip; }
        struct { uint32_t out_dim; uint32_t in_dim; float clip; uint32_t do_clip; } args;
        args.out_dim = (uint32_t)out_dim;
        args.in_dim  = (uint32_t)in_dim;
        args.clip    = clip;
        args.do_clip = do_clip ? 1u : 0u;
        id<MTLBuffer> args_b = [dev newBufferWithBytes:&args
                                                length:sizeof(args)
                                               options:MTLResourceStorageModeShared];

        MTLSize tg_grid   = MTLSizeMake((NSUInteger)out_dim, 1, 1);
        MTLSize tg_threads= MTLSizeMake(TS_SCT_THREADS, 1, 1);
        NSArray * bufs0 = @[args_b, w->W, wscale_b, ws_b, core_b, partials_b];
        if (!dispatch_and_wait(g_ctx->queue, g_ctx->sct_reduce,
                               bufs0, tg_grid, tg_threads)) {
            return 1;
        }

        // host: reduce partials -> global threshold + per-row clip limits
        const ts_sct_partial_metal * part =
            (const ts_sct_partial_metal *)partials_b.contents;
        double total_abs = 0.0;
        std::vector<float> clip_limits((size_t)out_dim);
        for (int64_t r = 0; r < out_dim; r++) {
            total_abs += (double)part[r].abs_sum;
            clip_limits[(size_t)r] = part[r].maxabs * clip;
        }
        const double total_n = (double)(out_dim * in_dim);
        const float threshold = (total_n > 0.0) ? (float)(total_abs / total_n) : 0.0f;
        *global_amp_out = threshold;

        // ---- phase 1: clip core in place + ternarize ----
        id<MTLBuffer> clip_b = [dev newBufferWithBytes:clip_limits.data()
                                                length:(NSUInteger)out_dim * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        // constant buffer binding for the threshold scalar
        id<MTLBuffer> thr_b = [dev newBufferWithBytes:&threshold
                                               length:sizeof(float)
                                              options:MTLResourceStorageModeShared];

        // ternary_out is int8 -> needs a device buffer
        id<MTLBuffer> tern_b = [dev newBufferWithLength:n * sizeof(int8_t)
                                                options:MTLResourceStorageModeShared];

        // buffers layout (per kernel signature):
        // 0 args, 1 core, 2 ternary, 3 clip_limits, 4 threshold
        NSArray * bufs1 = @[args_b, core_b, tern_b, clip_b, thr_b];
        if (!dispatch_and_wait(g_ctx->queue, g_ctx->sct_ternarize,
                               bufs1, tg_grid, tg_threads)) {
            return 1;
        }

        // ---- copy results back ----
        // core is private; we need it readable. Re-stage through a shared
        // buffer would double the cost; instead create the per-dispatch
        // buffers as shared from the start. (See note below; we keep ws/core
        // private during the two phases because the second phase reads core,
        // then do a single blit back.) For simplicity and correctness here we
        // use a blit encoder to copy private -> shared.
        id<MTLBuffer> ws_shared = [dev newBufferWithLength:n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> core_shared = [dev newBufferWithLength:n * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:ws_b   sourceOffset:0
                          toBuffer:ws_shared destinationOffset:0
                              size:n * sizeof(float)];
            [blit copyFromBuffer:core_b sourceOffset:0
                          toBuffer:core_shared destinationOffset:0
                              size:n * sizeof(float)];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        std::memcpy(ws_out,   ws_shared.contents,   n * sizeof(float));
        std::memcpy(core_out, core_shared.contents, n * sizeof(float));
        std::memcpy(ternary_out, tern_b.contents, n * sizeof(int8_t));
    }
    return rc;
}

// -----------------------------------------------------------------------
// Kernel 2: dequant + MSE + recon
// -----------------------------------------------------------------------

int ts_metal_dequant_mse_recon(ts_metal_weights_t * w,
                               const int8_t * ternary,
                               const uint16_t * page_scales,
                               const int8_t * lane_scales,
                               const int32_t * outlier_idx,
                               int64_t n_outliers,
                               const float * ws,
                               const float * input_scale,
                               float * recon_out,
                               float * mse_out) {
    if (!ts_metal_available() || g_ctx == nullptr || w == nullptr ||
        ternary == nullptr || page_scales == nullptr || lane_scales == nullptr ||
        ws == nullptr || input_scale == nullptr || recon_out == nullptr ||
        mse_out == nullptr) {
        return 1;
    }
    // The kernel uses a threadgroup scratch of in_dim floats (static-sized
    // TS_DMR_MAX_ROW). Fall back to CPU if too wide.
    if (w->in_dim > TS_DMR_MAX_ROW) return 2;

    const int64_t out_dim = w->out_dim;
    const int64_t in_dim  = w->in_dim;
    const NSUInteger n    = (NSUInteger)(out_dim * in_dim);
    const NSUInteger pages_per_row = (NSUInteger)((in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE);

    @autoreleasepool {
        id<MTLDevice> dev = g_ctx->device;

        // stage inputs into shared device buffers
        id<MTLBuffer> tern_b = [dev newBufferWithBytes:ternary
                                                length:n * sizeof(int8_t)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ps_b   = [dev newBufferWithBytes:page_scales
                                                length:(NSUInteger)out_dim * pages_per_row * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ls_b   = [dev newBufferWithBytes:lane_scales
                                                length:(NSUInteger)out_dim * pages_per_row * TS_LANES_PER_PAGE * sizeof(int8_t)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ws_b   = [dev newBufferWithBytes:ws
                                                length:n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> iscale_b = [dev newBufferWithBytes:input_scale
                                                  length:(NSUInteger)in_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> recon_b = [dev newBufferWithLength:n * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> partials_b = [dev newBufferWithLength:(NSUInteger)out_dim *
                                                       sizeof(ts_dmr_partial_metal)
                                                   options:MTLResourceStorageModeShared];
        if (tern_b == nil || ps_b == nil || ls_b == nil || ws_b == nil ||
            iscale_b == nil || recon_b == nil || partials_b == nil) {
            return 1;
        }

        // outlier CSR: build row_starts on the CPU (outlier_idx is already
        // sorted by row, matching the quantize_2d output). The kernel reads
        // outlier_idx[r*in_dim + col] absolute indices and needs row ranges.
        std::vector<uint32_t> row_starts((size_t)out_dim + 1, 0);
        for (int64_t i = 0; i < n_outliers; i++) {
            int64_t row = outlier_idx[i] / in_dim;
            if (row >= 0 && row < out_dim) row_starts[(size_t)row + 1]++;
        }
        for (int64_t r = 0; r < out_dim; r++) {
            row_starts[(size_t)r + 1] += row_starts[(size_t)r];
        }
        id<MTLBuffer> oidx_b = [dev newBufferWithBytes:outlier_idx
                                                length:(NSUInteger)n_outliers * sizeof(int32_t)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> rs_b   = [dev newBufferWithBytes:row_starts.data()
                                                length:(NSUInteger)(out_dim + 1) * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
        if (oidx_b == nil || rs_b == nil) return 1;

        struct { uint32_t out_dim; uint32_t in_dim; uint32_t n_outliers; uint32_t _pad; } args;
        args.out_dim = (uint32_t)out_dim;
        args.in_dim  = (uint32_t)in_dim;
        args.n_outliers = (uint32_t)n_outliers;
        args._pad = 0;
        id<MTLBuffer> args_b = [dev newBufferWithBytes:&args
                                                length:sizeof(args)
                                               options:MTLResourceStorageModeShared];

        // kernel signature buffers:
        // 0 args, 1 ternary, 2 page_scales, 3 lane_scales, 4 outlier_idx,
        // 5 row_starts, 6 ws, 7 input_scale, 8 recon, 9 partials
        NSArray * bufs = @[args_b, tern_b, ps_b, ls_b, oidx_b, rs_b,
                           ws_b, iscale_b, recon_b, partials_b];

        // threadgroup scratch for the dequant row: in_dim floats + up to 256
        // bytes (32 floats) for the cross-simdgroup MSE reduction tail.
        const NSUInteger tg_mem = (NSUInteger)in_dim * sizeof(float) + 256u;

        MTLSize tg_grid    = MTLSizeMake((NSUInteger)out_dim, 1, 1);
        MTLSize tg_threads = MTLSizeMake(TS_SCT_THREADS, 1, 1);
        if (!dispatch_and_wait(g_ctx->queue, g_ctx->dmr,
                               bufs, tg_grid, tg_threads, tg_mem)) {
            return 1;
        }

        // reduce partials -> mse
        const ts_dmr_partial_metal * part =
            (const ts_dmr_partial_metal *)partials_b.contents;
        double total_mse = 0.0;
        double total_n   = 0.0;
        for (int64_t r = 0; r < out_dim; r++) {
            total_mse += (double)part[r].mse_sum;
            total_n   += (double)part[r].n_count;
        }
        *mse_out = (total_n > 0.0) ? (float)(total_mse / total_n) : 0.0f;

        std::memcpy(recon_out, recon_b.contents, n * sizeof(float));
    }
    return 0;
}

// -----------------------------------------------------------------------
// Kernel 3: AWQ grid search
// -----------------------------------------------------------------------

int ts_metal_awq_grid_search(ts_metal_weights_t * w,
                             const float * grid,
                             int64_t n_grid,
                             float * mse_out) {
    if (!ts_metal_available() || g_ctx == nullptr || w == nullptr ||
        w->act == nullptr || w->act2 == nullptr ||
        grid == nullptr || mse_out == nullptr || n_grid <= 0) {
        return 1;
    }
    if (w->in_dim > TS_DMR_MAX_ROW) return 2;

    const int64_t out_dim = w->out_dim;
    const int64_t in_dim  = w->in_dim;

    @autoreleasepool {
        id<MTLDevice> dev = g_ctx->device;

        id<MTLBuffer> grid_b = [dev newBufferWithBytes:grid
                                                length:(NSUInteger)n_grid * sizeof(float)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> invmed_b = [dev newBufferWithBytes:&w->inv_median
                                                  length:sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> abs_partials_b = [dev newBufferWithLength:
                                    (NSUInteger)(out_dim * n_grid) * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> partials_b = [dev newBufferWithLength:
                                    (NSUInteger)(out_dim * n_grid) * sizeof(ts_awq_partial_metal)
                                                options:MTLResourceStorageModeShared];
        if (grid_b == nil || abs_partials_b == nil || partials_b == nil) return 1;

        struct { uint32_t out_dim; uint32_t in_dim; uint32_t n_grid; uint32_t _pad; } args;
        args.out_dim = (uint32_t)out_dim;
        args.in_dim  = (uint32_t)in_dim;
        args.n_grid  = (uint32_t)n_grid;
        args._pad    = 0;
        id<MTLBuffer> args_b = [dev newBufferWithBytes:&args
                                                length:sizeof(args)
                                               options:MTLResourceStorageModeShared];

        MTLSize tg_grid    = MTLSizeMake((NSUInteger)out_dim, (NSUInteger)n_grid, 1);
        MTLSize tg_threads = MTLSizeMake(TS_SCT_THREADS, 1, 1);

        // ---- phase 1: per-(row, alpha) abs_sum partials ----
        // buffers: 0 args, 1 W, 2 act, 3 grid, 4 inv_median, 5 abs_partials
        NSArray * bufs0 = @[args_b, w->W, w->act, grid_b, invmed_b, abs_partials_b];
        if (!dispatch_and_wait(g_ctx->queue, g_ctx->awq_threshold,
                               bufs0, tg_grid, tg_threads)) {
            return 1;
        }

        // host: reduce abs_partials across rows -> per-alpha threshold =
        // sum / (out_dim * in_dim) (matches the CPU GLOBAL mean(|ws|)).
        const float * abs_part = (const float *)abs_partials_b.contents;
        std::vector<float> threshold((size_t)n_grid, 0.0f);
        for (int64_t g = 0; g < n_grid; g++) {
            double s = 0.0;
            for (int64_t r = 0; r < out_dim; r++) {
                s += (double)abs_part[r * n_grid + g];
            }
            threshold[(size_t)g] = (float)(s / (double)(out_dim * in_dim));
        }
        id<MTLBuffer> thr_b = [dev newBufferWithBytes:threshold.data()
                                               length:(NSUInteger)n_grid * sizeof(float)
                                              options:MTLResourceStorageModeShared];

        // ---- phase 2: ternarize + dequant + err using the thresholds ----
        // buffers: 0 args, 1 W, 2 act, 3 act2, 4 grid, 5 thresholds,
        //          6 inv_median, 7 partials
        NSArray * bufs1 = @[args_b, w->W, w->act, w->act2, grid_b,
                            thr_b, invmed_b, partials_b];
        if (!dispatch_and_wait(g_ctx->queue, g_ctx->awq_grid,
                               bufs1, tg_grid, tg_threads)) {
            return 1;
        }

        // reduce per-(row, alpha) partials across rows -> one MSE per alpha.
        const ts_awq_partial_metal * part =
            (const ts_awq_partial_metal *)partials_b.contents;
        std::vector<double> sum_err((size_t)n_grid, 0.0);
        std::vector<double> sum_n((size_t)n_grid, 0.0);
        for (int64_t r = 0; r < out_dim; r++) {
            for (int64_t g = 0; g < n_grid; g++) {
                const ts_awq_partial_metal & p = part[r * n_grid + g];
                sum_err[(size_t)g] += (double)p.err_sum;
                sum_n[(size_t)g]   += (double)p.n_count;
            }
        }
        for (int64_t g = 0; g < n_grid; g++) {
            mse_out[g] = (sum_n[(size_t)g] > 0.0)
                         ? (float)(sum_err[(size_t)g] / sum_n[(size_t)g])
                         : 0.0f;
        }
    }
    return 0;
}

}  // extern "C"
