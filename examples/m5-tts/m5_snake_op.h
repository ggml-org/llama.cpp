// Fused Snake activation as an ORT custom op.
//
// The M5 codec applies snake(x) = x + c * sin(alpha * x)^2 with per-channel
// alpha/c at ~29 sites. As graph nodes that is Mul -> Sin -> Pow -> Mul -> Add:
// five full passes over sometimes multi-megabyte tensors, and the Sin kernel
// alone costs ~14% of codec time on the SM6650. This op computes the whole
// activation in one pass with a range-reduced degree-9 sine polynomial
// (error ~2e-6, far below audibility) so a future NPU port can reuse the same
// maths as an HVX kernel where sin() does not exist at all.
//
// Inputs:  x [1,C,T] f32, alpha [1,C,1] f32, c [1,C,1] f32
// Output:  y [1,C,T] f32
#pragma once

#include "onnxruntime_c_api.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace m5 {
namespace snake {

inline void snake_f32(const float * x, const float * alpha, const float * beta,
                      float * out, int channels, int length) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kInvPi = 0.31830988618379067154f;
    for (int ch = 0; ch < channels; ch++) {
        const float a = alpha[ch];
        const float b = beta[ch];
        const float * xr = x + (size_t) ch * length;
        float * o = out + (size_t) ch * length;
        int i = 0;
#if defined(__aarch64__)
        const float32x4_t vpi = vdupq_n_f32(kPi);
        const float32x4_t vinvpi = vdupq_n_f32(kInvPi);
        const float32x4_t vc1 = vdupq_n_f32(-1.0f / 6.0f);
        const float32x4_t vc2 = vdupq_n_f32(1.0f / 120.0f);
        const float32x4_t vc3 = vdupq_n_f32(-1.0f / 5040.0f);
        const float32x4_t vc4 = vdupq_n_f32(1.0f / 362880.0f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t va = vdupq_n_f32(a);
        const float32x4_t vb = vdupq_n_f32(b);
        for (; i + 4 <= length; i += 4) {
            const float32x4_t xv = vld1q_f32(xr + i);
            const float32x4_t t = vmulq_f32(xv, va);
            // u = t - pi*round(t/pi) puts every t into [-pi/2, pi/2]
            const float32x4_t n = vrndnq_f32(vmulq_f32(t, vinvpi));
            const float32x4_t u = vmlsq_f32(t, n, vpi);
            const float32x4_t w = vmulq_f32(u, u);
            // sin(u) = u * (1 + w*(-1/6 + w*(1/120 + w*(-1/5040 + w/362880))))
            float32x4_t p = vmlaq_f32(vc3, w, vc4);
            p = vmlaq_f32(vc2, w, p);
            p = vmlaq_f32(vc1, w, p);
            p = vmlaq_f32(one, w, p);
            const float32x4_t s = vmulq_f32(u, p);
            const float32x4_t s2 = vmulq_f32(s, s);
            vst1q_f32(o + i, vmlaq_f32(xv, vb, s2));
        }
#endif
        for (; i < length; i++) {
            const float t = a * xr[i];
            const float u = t - kPi * std::nearbyintf(t * kInvPi);
            const float w = u * u;
            const float s = u * (1.0f + w * (-1.0f / 6.0f +
                    w * (1.0f / 120.0f + w * (-1.0f / 5040.0f + w * (1.0f / 362880.0f)))));
            o[i] = xr[i] + b * (s * s);
        }
    }
}

// ---------------------------------------------------------------- ORT plumbing

struct SnakeKernel {
    const OrtApi * api;
};

inline const char * ORT_API_CALL GetName(const OrtCustomOp *) { return "Snake"; }
inline const char * ORT_API_CALL GetEP(const OrtCustomOp *) { return nullptr; }  // CPU
inline ONNXTensorElementDataType ORT_API_CALL GetInputType(const OrtCustomOp *, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
inline size_t ORT_API_CALL GetInputCount(const OrtCustomOp *) { return 3; }
inline ONNXTensorElementDataType ORT_API_CALL GetOutputType(const OrtCustomOp *, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
inline size_t ORT_API_CALL GetOutputCount(const OrtCustomOp *) { return 1; }

// ORT queries these during session creation; leaving them null segfaults.
inline OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetInputCharacteristic(const OrtCustomOp *, size_t) {
    return INPUT_OUTPUT_REQUIRED;
}
inline OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetOutputCharacteristic(const OrtCustomOp *, size_t) {
    return INPUT_OUTPUT_REQUIRED;
}
inline OrtMemType ORT_API_CALL GetInputMemoryType(const OrtCustomOp *, size_t) {
    return OrtMemTypeDefault;
}

inline void * ORT_API_CALL CreateKernel(const OrtCustomOp * op, const OrtApi * api,
                                        const OrtKernelInfo * info) {
    (void) op; (void) info;
    return new SnakeKernel{api};
}

inline void ORT_API_CALL KernelDestroy(void * k) { delete static_cast<SnakeKernel *>(k); }

inline void ORT_API_CALL KernelCompute(void * k, OrtKernelContext * context) {
    auto * kernel = static_cast<SnakeKernel *>(k);
    const OrtApi * api = kernel->api;

    const OrtValue * in_x = nullptr;
    const OrtValue * in_a = nullptr;
    const OrtValue * in_c = nullptr;
    api->KernelContext_GetInput(context, 0, &in_x);
    api->KernelContext_GetInput(context, 1, &in_a);
    api->KernelContext_GetInput(context, 2, &in_c);

    OrtTensorTypeAndShapeInfo * info = nullptr;
    api->GetTensorTypeAndShape(in_x, &info);
    size_t rank = 0;
    api->GetDimensionsCount(info, &rank);
    int64_t dims[4] = {1, 1, 1, 1};
    if (rank <= 4) api->GetDimensions(info, dims, rank);
    api->ReleaseTensorTypeAndShapeInfo(info);

    // x is [1, C, T]; alpha/c are [1, C, 1]
    const int channels = (int) (rank >= 2 ? dims[rank - 2] : 1);
    const int length = (int) (rank >= 1 ? dims[rank - 1] : 1);

    const float * xp = nullptr;
    const float * ap = nullptr;
    const float * cp = nullptr;
    api->GetTensorData(in_x, (const void **) &xp);
    api->GetTensorData(in_a, (const void **) &ap);
    api->GetTensorData(in_c, (const void **) &cp);

    int64_t out_dims[3] = {1, channels, length};
    OrtValue * out = nullptr;
    api->KernelContext_GetOutput(context, 0, out_dims, rank <= 3 ? rank : 3, &out);
    float * op_ = nullptr;
    api->GetTensorMutableData(out, (void **) &op_);

    snake_f32(xp, ap, cp, op_, channels, length);
}

inline const OrtCustomOp * snake_op() {
    static OrtCustomOp op = {};
    static bool init = false;
    if (!init) {
        op.version = ORT_API_VERSION;
        op.CreateKernel = CreateKernel;
        op.GetName = GetName;
        op.GetExecutionProviderType = GetEP;
        op.GetInputType = GetInputType;
        op.GetInputTypeCount = GetInputCount;
        op.GetOutputType = GetOutputType;
        op.GetOutputTypeCount = GetOutputCount;
        op.KernelCompute = KernelCompute;
        op.KernelDestroy = KernelDestroy;
        op.GetInputCharacteristic = GetInputCharacteristic;
        op.GetOutputCharacteristic = GetOutputCharacteristic;
        op.GetInputMemoryType = GetInputMemoryType;
        init = true;
    }
    return &op;
}

// Registers the "m5" domain with the Snake op on a session options object.
// Must run before CreateSession.
inline bool register_ops(const OrtApi * api, OrtSessionOptions * so) {
    static OrtCustomOpDomain * domain = nullptr;
    if (!domain) {
        if (api->CreateCustomOpDomain("m5", &domain)) return false;
        if (api->CustomOpDomain_Add(domain, snake_op())) return false;
    }
    return api->AddCustomOpDomain(so, domain) == nullptr;
}

}  // namespace snake
}  // namespace m5
