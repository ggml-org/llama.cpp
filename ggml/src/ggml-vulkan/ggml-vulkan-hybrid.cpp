#include "ggml-vulkan-hybrid.h"
#include "ggml-vulkan-hybrid-gpu.h"
#include "ggml-vulkan-onednn.h"
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

using sdpa_fn = int (*)(int, int, int, const uint16_t *, const int8_t *, const uint16_t *, const int8_t *, const uint16_t *, const uint16_t *, float, float *);
using sdpa_win32_fn = int (*)(int, int, int, const ggml_vulkan_onednn_win32_allocation *, float);
using release_win32_fn = int (*)(uint64_t);

void ggml_vk_hybrid_gpu_release_external_allocation(uint64_t allocation_id) {
    if (!allocation_id) return;
    static release_win32_fn release_zc = []() -> release_win32_fn {
        const char * path = std::getenv("GGML_VULKAN_HYBRID_DLL");
        HMODULE module = LoadLibraryA(path ? path : "ggml-vulkan-onednn.dll");
        return module ? reinterpret_cast<release_win32_fn>(GetProcAddress(module, "ggml_vulkan_onednn_release_win32")) : nullptr;
    }();
    if (!release_zc || !release_zc(allocation_id)) {
        GGML_LOG_WARN("vulkan-hybrid-zc: external allocation release failed allocation_id=%llu\n", (unsigned long long) allocation_id);
    }
}

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
        (q->ne[0] == 128 || q->ne[0] == 256) && q->ne[1] >= 256 && q->ne[1] <= INT32_MAX && q->ne[2] == 16 && q->ne[3] == 1 &&
        k->ne[0] == q->ne[0] && k->ne[1] > 0 && k->ne[1] <= INT32_MAX && k->ne[2] == 4 && k->ne[3] == 1 &&
        v->ne[0] == q->ne[0] && v->ne[1] == k->ne[1] && v->ne[2] == 4 && v->ne[3] == 1 &&
        (q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
        k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0 && k->nb[0] == 34 && v->nb[0] == 34 &&
        mask->type == GGML_TYPE_F16 && mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        params[0] > 0 && std::isfinite(params[0]) && params[1] == 0 && params[2] == 0 &&
        node->type == GGML_TYPE_F32 && ggml_is_contiguous(node);
}

bool ggml_vk_hybrid_try(void * backend_ctx, ggml_tensor * node) {
    try {
        if (std::getenv("GGML_VULKAN_HYBRID_FORCE_FAIL")) {
            GGML_LOG_WARN("vulkan-hybrid: forced failure -> Vulkan fallback\n");
            return false;
        }
        if (std::getenv("GGML_VULKAN_HYBRID_ZC")) {
            static sdpa_win32_fn run_zc = []() -> sdpa_win32_fn {
                const char * path = std::getenv("GGML_VULKAN_HYBRID_DLL");
                HMODULE module = LoadLibraryA(path ? path : "ggml-vulkan-onednn.dll");
                return module ? reinterpret_cast<sdpa_win32_fn>(GetProcAddress(module, "ggml_vulkan_onednn_sdpa_win32")) : nullptr;
            }();
            if (!run_zc) {
                GGML_LOG_WARN("vulkan-hybrid-zc: bridge entry unavailable -> Vulkan fallback\n");
                return false;
            }
            const auto * q = node->src[0];
            const auto * k = node->src[1];
            const auto * v = node->src[2];
            const auto * mask = node->src[3];
            ggml_vk_hybrid_gpu_planes planes;
            const float scale = reinterpret_cast<const float *>(node->op_params)[0];
            if (!ggml_vk_hybrid_gpu_pack(backend_ctx, q, k, v, mask, 1.0f / scale, &planes)) {
                GGML_LOG_WARN("vulkan-hybrid-zc: GPU pack failed -> Vulkan fallback\n");
                return false;
            }
            const ggml_vulkan_onednn_win32_allocation shared = {
                planes.handle, planes.allocation_size,
                planes.q_off, planes.q_size,
                planes.k_off, planes.k_size,
                planes.ks_off, planes.ks_size,
                planes.v_off, planes.v_size,
                planes.vs_off, planes.vs_size,
                planes.mask_off, planes.mask_size,
                planes.divisor_off, planes.divisor_size,
                planes.out_off, planes.out_size,
                planes.allocation_id,
            };
            const int nq = (int) q->ne[1];
            const int nk = (int) k->ne[1];
            const int d = (int) q->ne[0];
            ggml_vk_hybrid_gpu_mark_imported(backend_ctx);
            const bool executed = run_zc(nq, nk, d, &shared, 1.0f / scale) != 0;
            const bool unpacked = executed && ggml_vk_hybrid_gpu_unpack(backend_ctx, node, planes.workspace);
            ggml_vk_hybrid_gpu_release(planes.workspace);
            if (!unpacked) {
                GGML_LOG_WARN("vulkan-hybrid-zc: execution failed -> Vulkan fallback\n");
                return false;
            }
            GGML_LOG_INFO("vulkan-hybrid-zc: complete %s q=%d kv=%d host_traffic=0\n", node->name, nq, nk);
            return true;
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
        const int d = (int) q->ne[0];
        const auto t_readback = std::chrono::steady_clock::now();
        std::vector<uint8_t> q_raw(ggml_nbytes(q)), k_raw(ggml_nbytes(k)), v_raw(ggml_nbytes(v)), m_raw(ggml_nbytes(mask));
        const ggml_tensor * tensors[] = { q, k, v, mask };
        void * data[] = { q_raw.data(), k_raw.data(), v_raw.data(), m_raw.data() };
        const size_t sizes[] = { q_raw.size(), k_raw.size(), v_raw.size(), m_raw.size() };
        if (!ggml_vk_hybrid_read_tensors(backend_ctx, tensors, data, sizes, 4)) {
            GGML_LOG_WARN("vulkan-hybrid: batched readback failed -> Vulkan fallback\n");
            return false;
        }
        const double readback_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_readback).count();
        const auto t_split = std::chrono::steady_clock::now();
        std::vector<uint16_t> q_half(size_t(16)*nq*d);
        std::vector<int8_t> kp(size_t(4)*d*nk), vp(size_t(4)*nk*d);
        std::vector<uint16_t> ks(size_t(4)*(d/32)*nk), vs(size_t(4)*nk*(d/32)), mh(size_t(nq)*nk);
        for (int h = 0; h < 16; ++h) {
            for (int t = 0; t < nq; ++t) {
                for (int di = 0; di < d; ++di) {
                    const uint8_t * src = q_raw.data() + h*q->nb[2] + t*q->nb[1] + di*q->nb[0];
                    uint16_t & dst = q_half[(size_t(h)*nq + t)*d + di];
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
                for (int g = 0; g < d/32; ++g) {
                    const uint8_t * kb = k_raw.data() + h*k->nb[2] + t*k->nb[1] + g*k->nb[0];
                    const uint8_t * vb = v_raw.data() + h*v->nb[2] + t*v->nb[1] + g*v->nb[0];
                    std::memcpy(&ks[(size_t(h)*(d/32) + g)*nk + t], kb, 2);
                    std::memcpy(&vs[(size_t(h)*nk + t)*(d/32) + g], vb, 2);
                    for (int j = 0; j < 32; ++j) {
                        kp[(size_t(h)*d + g*32 + j)*nk + t] = static_cast<int8_t>(kb[2 + j]);
                        vp[(size_t(h)*nk + t)*d + g*32 + j] = static_cast<int8_t>(vb[2 + j]);
                    }
                }
            }
        }
        for (int t = 0; t < nq; ++t) {
            std::memcpy(mh.data() + size_t(t)*nk, m_raw.data() + t*mask->nb[1], size_t(nk)*2);
        }
        const double host_split_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_split).count();
        std::vector<float> out(size_t(16)*nq*d), dst(out.size());
        const float scale = reinterpret_cast<const float *>(node->op_params)[0];
        GGML_LOG_INFO("vulkan-hybrid: staging q=%d kv=%d batched_readback_ms=%.3f host_split_ms=%.3f\n", nq, nk, readback_ms, host_split_ms);
        if (!run(nq, nk, d, q_half.data(), kp.data(), ks.data(), vp.data(), vs.data(), mh.data(), 1.0f/scale, out.data())) {
            GGML_LOG_WARN("vulkan-hybrid: oneDNN failed -> Vulkan fallback\n");
            return false;
        }
        for (int h = 0; h < 16; ++h) {
            for (int t = 0; t < nq; ++t) {
                for (int di = 0; di < d; ++di) {
                    const float value = out[(size_t(h)*nq + t)*d + di];
                    if (!std::isfinite(value)) {
                        GGML_LOG_WARN("vulkan-hybrid: non-finite output -> Vulkan fallback\n");
                        return false;
                    }
                    dst[(size_t(t)*16 + h)*d + di] = value;
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
