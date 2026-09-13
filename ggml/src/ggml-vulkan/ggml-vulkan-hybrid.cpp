#include "ggml-vulkan-hybrid.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#define NOMINMAX
#include <windows.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>
#include <chrono>

using sdpa_fn = int (*)(int, int, const uint16_t *, const int8_t *, const uint16_t *, const int8_t *, const uint16_t *, const uint16_t *, float, float *);

bool ggml_vk_hybrid_supported(const ggml_tensor * node) {
    const char * enabled = std::getenv("GGML_VULKAN_HYBRID");
    if (!enabled || std::strcmp(enabled, "1") != 0 || node->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const auto * q = node->src[0];
    const auto * k = node->src[1];
    const auto * v = node->src[2];
    const auto * mask = node->src[3];
    const auto * params = reinterpret_cast<const float *>(node->op_params);
    return q && k && v && mask && !node->src[4] &&
        q->ne[0] == 256 && q->ne[1] >= 256 && q->ne[1] <= INT32_MAX && q->ne[2] == 16 && q->ne[3] == 1 &&
        k->ne[0] == 256 && k->ne[1] > 0 && k->ne[1] <= INT32_MAX && k->ne[2] == 4 && k->ne[3] == 1 &&
        v->ne[0] == 256 && v->ne[1] == k->ne[1] && v->ne[2] == 4 && v->ne[3] == 1 &&
        (q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
        k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0 && k->nb[0] == 34 && v->nb[0] == 34 &&
        mask->type == GGML_TYPE_F16 && mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        params[0] > 0 && std::isfinite(params[0]) && params[1] == 0 && params[2] == 0 &&
        node->type == GGML_TYPE_F32 && ggml_is_contiguous(node);
}

static std::vector<uint8_t> read_tensor(const ggml_tensor * tensor) {
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    return data;
}

bool ggml_vk_hybrid_try(ggml_tensor * node) {
    try {
        if (std::getenv("GGML_VULKAN_HYBRID_FORCE_FAIL")) {
            GGML_LOG_WARN("vulkan-hybrid: forced failure -> Vulkan fallback\n");
            return false;
        }
        static sdpa_fn run = []() -> sdpa_fn {
            const char * path = std::getenv("GGML_VULKAN_HYBRID_DLL");
            HMODULE module = LoadLibraryA(path ? path : "ggml-vulkan-onednn.dll");
            return module ? reinterpret_cast<sdpa_fn>(GetProcAddress(module, "ggml_vulkan_onednn_sdpa")) : nullptr;
        }();
        if (!run) {
            GGML_LOG_WARN("vulkan-hybrid: bridge DLL unavailable -> Vulkan fallback\n");
            return false;
        }
        const auto * q = node->src[0];
        const auto * k = node->src[1];
        const auto * v = node->src[2];
        const auto * mask = node->src[3];
        const int nq = (int) q->ne[1];
        const int nk = (int) k->ne[1];
        // TODO: replace staging with Vulkan/L0 shared memory
        const auto t_get = std::chrono::steady_clock::now();
        const auto q_raw = read_tensor(q);
        const auto k_raw = read_tensor(k);
        const auto v_raw = read_tensor(v);
        const auto m_raw = read_tensor(mask);
        const double tensor_get_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_get).count();
        const auto t_split = std::chrono::steady_clock::now();
        std::vector<uint16_t> q_half(size_t(16)*nq*256);
        std::vector<int8_t> kp(size_t(4)*256*nk), vp(size_t(4)*nk*256);
        std::vector<uint16_t> ks(size_t(4)*8*nk), vs(size_t(4)*nk*8), mh(size_t(nq)*nk);
        for (int h = 0; h < 16; ++h) {
            for (int t = 0; t < nq; ++t) {
                for (int d = 0; d < 256; ++d) {
                    const uint8_t * src = q_raw.data() + h*q->nb[2] + t*q->nb[1] + d*q->nb[0];
                    uint16_t & dst = q_half[(size_t(h)*nq + t)*256 + d];
                    if (q->type == GGML_TYPE_F32) {
                        float value;
                        std::memcpy(&value, src, sizeof(value));
                        dst = ggml_fp32_to_fp16(value);
                    } else {
                        std::memcpy(&dst, src, sizeof(dst));
                    }
                }
            }
        }
        for (int h = 0; h < 4; ++h) {
            for (int t = 0; t < nk; ++t) {
                for (int g = 0; g < 8; ++g) {
                    const uint8_t * kb = k_raw.data() + h*k->nb[2] + t*k->nb[1] + g*k->nb[0];
                    const uint8_t * vb = v_raw.data() + h*v->nb[2] + t*v->nb[1] + g*v->nb[0];
                    std::memcpy(&ks[(size_t(h)*8 + g)*nk + t], kb, 2);
                    std::memcpy(&vs[(size_t(h)*nk + t)*8 + g], vb, 2);
                    for (int j = 0; j < 32; ++j) {
                        kp[(size_t(h)*256 + g*32 + j)*nk + t] = static_cast<int8_t>(kb[2 + j]);
                        vp[(size_t(h)*nk + t)*256 + g*32 + j] = static_cast<int8_t>(vb[2 + j]);
                    }
                }
            }
        }
        for (int t = 0; t < nq; ++t) {
            std::memcpy(mh.data() + size_t(t)*nk, m_raw.data() + t*mask->nb[1], size_t(nk)*2);
        }
        const double host_split_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_split).count();
        std::vector<float> out(size_t(16)*nq*256), dst(out.size());
        const float scale = reinterpret_cast<const float *>(node->op_params)[0];
        GGML_LOG_INFO("vulkan-hybrid: staging q=%d kv=%d tensor_get_ms=%.3f host_split_ms=%.3f\n", nq, nk, tensor_get_ms, host_split_ms);
        if (!run(nq, nk, q_half.data(), kp.data(), ks.data(), vp.data(), vs.data(), mh.data(), 1.0f/scale, out.data())) {
            GGML_LOG_WARN("vulkan-hybrid: oneDNN failed -> Vulkan fallback\n");
            return false;
        }
        for (int h = 0; h < 16; ++h) {
            for (int t = 0; t < nq; ++t) {
                for (int d = 0; d < 256; ++d) {
                    const float value = out[(size_t(h)*nq + t)*256 + d];
                    if (!std::isfinite(value)) {
                        GGML_LOG_WARN("vulkan-hybrid: non-finite output -> Vulkan fallback\n");
                        return false;
                    }
                    dst[(size_t(t)*16 + h)*256 + d] = value;
                }
            }
        }
        ggml_backend_tensor_set(node, dst.data(), 0, dst.size()*sizeof(float));
        GGML_LOG_INFO("vulkan-hybrid: complete %s output_finite=yes\n", node->name);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_WARN("vulkan-hybrid: %s -> Vulkan fallback\n", e.what());
        return false;
    }
}
