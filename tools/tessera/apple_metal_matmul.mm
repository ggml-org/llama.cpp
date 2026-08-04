// Apple Metal Performance Shaders (MPS) matmul bridge for the
// per-chunk calibration GEMM (Phase 16.5, memopt-metal-dispatch).
//
// The legacy per-chunk LRQ / FLRQ matmul in
// ``tools/tessera/calibration_memory.py`` is pure numpy. On
// Apple Silicon (M1/M2/M3/M4) the unified-memory architecture
// lets the chunked numpy arrays be GPU-visible without an
// explicit copy, and MPS's MPSMatrixMultiplication is the
// fastest path for the (4096 x 4096) per-chunk GEMM that
// drives LRQ / FLRQ.
//
// This file is Objective-C++ because the MPS API is
// Objective-C.  We expose a single extern "C" entry point
// (tessera_metal_sgemm_f32) so ctypes can call it without an
// Objective-C ABI dependency.  The Python wrapper in
// ``tools/tessera/calibration_metal.py`` invokes this.
//
// Synchronisation: MPSMatrixMultiplication encodes onto a
// command buffer; the host waits for completion via
// -[MTLCommandBuffer waitUntilCompleted].  The chunked
// calibration loop is double-buffered at the pipeline level
// (CalibPipeline overlaps the mmap of the next layer with
// the compute of the current layer), so the per-chunk
// wait-until-completed latency is hidden by the next
// layer's mmap.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" int tessera_metal_sgemm_f32(
        const float * a,
        const float * b,
        float * c,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        int transpose_a,
        int transpose_b) {
    if (!a || !b || !c || m == 0 || n == 0 || k == 0) {
        return -1;
    }
    if (transpose_a || transpose_b) {
        // MPSMatrixMultiplication takes the transpose flags
        // at init time; supporting them is a follow-up if
        // any caller needs the transposed shapes.  For now
        // the per-chunk LRQ / FLRQ matmuls are all
        // non-transposed (or use the V.T / U.T view via a
        // .copy() that the Python side already pays for).
        return -2;
    }

    // The MTLDevice and MTLCommandQueue are process-wide
    // resources.  Allocating a new queue per call is the
    // textbook mistake: Metal keeps a reference to every
    // committed command buffer until it completes, and the
    // queue's internal pool grows until the device runs
    // out of memory.  In a tight 200-call loop the device
    // starts returning garbage (NaN in the result buffer)
    // around the 135th iteration.  We pin one queue on
    // the device and reuse it for every call; the
    // command buffers are still deallocated as soon as
    // waitUntilCompleted returns.
    static id<MTLDevice> g_device = nil;
    static id<MTLCommandQueue> g_queue = nil;
    static dispatch_once_t g_once = 0;
    dispatch_once(&g_once, ^{
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device) {
            g_queue = [g_device newCommandQueue];
        }
    });
    if (!g_device || !g_queue) {
        return -3;  // no Metal device / queue
    }
    id<MTLDevice> device = g_device;
    id<MTLCommandQueue> queue = g_queue;

    @autoreleasepool {
        const std::size_t a_bytes = m * k * sizeof(float);
        const std::size_t b_bytes = k * n * sizeof(float);
        const std::size_t c_bytes = m * n * sizeof(float);
        // Storage mode: shared.  On Apple Silicon the
        // unified memory aliasing means the CPU and the
        // GPU see the same pointer; no copy is needed for
        // either the input (the bridge copies the bytes
        // once at allocation) or the output (the memcpy
        // below reads the unified-memory pages the GPU
        // wrote).  The trade-off vs. managed memory: the
        // GPU may sync the cache after waitUntilCompleted,
        // which is what we want.
        const MTLResourceOptions options = MTLResourceStorageModeShared;
        id<MTLBuffer> aBuf = [device newBufferWithBytes:a
                                                  length:a_bytes
                                                 options:options];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:b
                                                  length:b_bytes
                                                 options:options];
        id<MTLBuffer> cBuf = [device newBufferWithLength:c_bytes
                                                 options:options];
        if (!aBuf || !bBuf || !cBuf) {
            return -5;  // buffer alloc failed
        }
        // MPSMatrixMultiplication.  The
        // MPSMatrixDescriptor rowBytes is the row stride in
        // bytes; for a contiguous (M, K) F32 array the row
        // stride is K * sizeof(Float).  The result is C(M, N).
        MPSMatrixDescriptor *aDesc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:m
                             columns:k
                            rowBytes:k * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *bDesc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:k
                             columns:n
                            rowBytes:n * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *cDesc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:m
                             columns:n
                            rowBytes:n * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *aMat = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix *bMat = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        MPSMatrix *cMat = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
              transposeLeft:NO
             transposeRight:NO
                 resultRows:m
              resultColumns:n
            interiorColumns:k
                    alpha:1.0
                     beta:0.0];
        if (!mm) {
            return -6;  // matmul init failed
        }
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            return -7;  // command buffer alloc failed
        }
        [mm encodeToCommandBuffer:cmd
                       leftMatrix:aMat
                      rightMatrix:bMat
                    resultMatrix:cMat];
        [cmd commit];
        [cmd waitUntilCompleted];
        // Copy the GPU result back into the caller's buffer.
        // On Apple Silicon the unified memory aliasing
        // means the GPU wrote to the same physical pages
        // the caller owns, so the memcpy is the right
        // shape (the caller's c pointer might not be the
        // MPS buffer's pointer).
        memcpy(c, [cBuf contents], c_bytes);
        return 0;
    }
}
