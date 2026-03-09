// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#include <arm_neon.h>
#include <assert.h>
#include <stdio.h>
#include <atomic>
#include <cfloat>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <set>
#include <iostream>
#include <climits>
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <string_view>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <windows.h>
#include <excpt.h>
#endif

#include "kleidiai.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-threading.h"
#include "traits.h"

#include "kernels.h"

#include "kai_common.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

static constexpr int      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
static constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
static constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
static constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;

struct ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q8;
    int sme_thread_cap; // <= 0 means “SME disabled/unknown”;
    int thread_hint;    // <= 0 means “no hint”
} static ctx = { CPU_FEATURE_NONE, nullptr, nullptr, 0, -1 };

static const char* cpu_feature_to_string(cpu_feature f) {
    if (f == CPU_FEATURE_NONE) {
        return "NONE";
    } else if ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        return "SME";
    } else if ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        return "SVE";
    }
    else if ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        return "I8MM";
    } else if ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        return "DOTPROD";
    }
    else {
        return "UNKNOWN";
    }
}

static size_t detect_num_smcus() {
    if (!ggml_cpu_has_sme()) {
        return 0;
    }

#if defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t num_private = 0;
    std::set<uint32_t> shared_ids;

    for (size_t cpu = 0;; ++cpu) {
        const std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        if (!file.is_open()) {
            break;
        }

        uint64_t smidr = 0;
        if (!(file >> std::hex >> smidr)) {
            continue;
        }

        // Arm ARM: SMIDR_EL1
        const uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
        // Build an "affinity-like" identifier for shared SMCUs.
        // Keep the original packing logic, but isolate it here.
        const uint32_t id = (uint32_t)((smidr & 0xFFFu) | ((smidr >> 20) & 0xFFFFF000u));

        switch (sh) {
            case 0b10: // private SMCU
                ++num_private;
                break;
            case 0b11: // shared SMCU
                shared_ids.emplace(id);
                break;
            case 0b00:
                // Ambiguous / implementation-defined. Be conservative:
                // treat id==0 as private, otherwise as shared.
                if (id == 0) ++num_private;
                else shared_ids.emplace(id);
                break;
            default:
                break;
        }
    }

    return num_private + shared_ids.size();

#elif defined(__APPLE__) && defined(__aarch64__)
    // table for known M4 variants. Users can override via GGML_KLEIDIAI_SME=<n>.
    char chip_name[256] = {};
    size_t size = sizeof(chip_name);

    if (sysctlbyname("machdep.cpu.brand_string", chip_name, &size, nullptr, 0) == 0) {
        const std::string brand(chip_name);

        struct ModelSMCU { const char *match; size_t smcus; };
        static const ModelSMCU table[] = {
            { "M4 Ultra", 2 },
            { "M4 Max",   2 },
            { "M4 Pro",   2 },
            { "M4",       1 },
        };

        for (const auto &e : table) {
            if (brand.find(e.match) != std::string::npos) {
                return e.smcus;
            }
        }
    }
    return 1;

#else
    return 1;
#endif
}

static int parse_uint_env(const char *s, const char *name, bool *ok) {
    if (!s) { *ok = false; return 0; }
    char *end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = false;
        return 0;
    }
    if (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = false;
        return 0;
    }
    *ok = true;
    return (int)v;
}

static void init_kleidiai_context(void) {
    ggml_critical_section_start();
    static bool initialized = false;

    if (!initialized) {
        initialized = true;

        const char *env_sme     = getenv("GGML_KLEIDIAI_SME");
        const char *env_threads = getenv("GGML_TOTAL_THREADS");

        const bool cpu_has_sme = ggml_cpu_has_sme();
        size_t detected_smcus = 0;

        ctx.features  = (ggml_cpu_has_dotprod()     ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (ggml_cpu_has_matmul_int8() ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        ((ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0) ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        if (env_threads) {
            bool ok = false;
            int hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            if (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        // SME policy:
        // - If CPU doesn't support SME: SME always off.
        // - Else:
        //   - env unset => auto-detect cores; enable if detected > 0.
        //   - env=0     => force off.
        //   - env>0     => force N cores (skip detection).
        int sme_cores = 0;
        bool sme_env_ok = false;
        bool sme_env_set = (env_sme != nullptr);

        if (!cpu_has_sme) {
            if (sme_env_set) {
                bool ok = false;
                int req = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
                if (ok && req > 0) {
                    GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME\n", req);
                }
            }
            sme_cores = 0;
        } else {
            if (sme_env_set) {
                bool ok = false;
                int v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
                sme_env_ok = ok;

                if (!ok) {
                    GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; falling back to runtime SME-core detection\n");
                    detected_smcus = detect_num_smcus();
                    sme_cores = detected_smcus > 0 ? (int)detected_smcus : 0;
                } else if (v == 0) {
                    sme_cores = 0;
                } else {
                    sme_cores = v;
                }
            } else {
                detected_smcus = detect_num_smcus();
                sme_cores = detected_smcus > 0 ? (int)detected_smcus : 0;
            }

            if (!sme_env_set && sme_cores == 0) {
                GGML_LOG_WARN("kleidiai: SME supported but runtime SME-core detection returned 0; falling back to NEON\n");
            }

            if (sme_cores > 0) {
                ctx.features |= CPU_FEATURE_SME;
            }
        }

        // Kernel selection
        ctx.kernels_q4 = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q8 = ggml_kleidiai_select_kernels_q8_0(ctx.features);

        if (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (int)ctx.features);
        } else {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        if (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (int)ctx.features);
        } else {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        ctx.sme_thread_cap = (ctx.features & CPU_FEATURE_SME) ? sme_cores : 0;

        if (ctx.features & CPU_FEATURE_SME) {
            if (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d override)\n", sme_cores);
            } else {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME cores=%d)\n", sme_cores);
            }
        } else {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

static inline int kleidiai_sme_thread_cap() {
    return ctx.sme_thread_cap;
}

static inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static inline bool kleidiai_pack_fallback_allowed() {
    if (ctx.sme_thread_cap <= 0) {
        return false;
    }
    if (ctx.thread_hint <= 0) {
        return true;
    }
    return ctx.thread_hint > ctx.sme_thread_cap;
}

struct kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_count;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

static inline kleidiai_weight_header * kleidiai_weight_header_from_ptr(void * data) {
    return reinterpret_cast<kleidiai_weight_header *>(data);
}

static inline const kleidiai_weight_header * kleidiai_weight_header_from_ptr(const void * data) {
    return reinterpret_cast<const kleidiai_weight_header *>(data);
}

static inline bool kleidiai_is_weight_header_valid(const kleidiai_weight_header * header) {
    if (!header) {
        return false;
    }
    if (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        return false;
    }
    if (header->slot_count == 0 || header->slot_count > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        return false;
    }
    return true;
}

static inline uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, int slot) {
    if (!kleidiai_is_weight_header_valid(header)) {
        return nullptr;
    }
    if (slot < 0 || slot >= header->slot_count) {
        return nullptr;
    }
    return reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

static inline const uint8_t * kleidiai_weight_slot_ptr(const kleidiai_weight_header * header, int slot) {
    if (!kleidiai_is_weight_header_valid(header)) {
        return nullptr;
    }
    if (slot < 0 || slot >= header->slot_count) {
        return nullptr;
    }
    return reinterpret_cast<const uint8_t *>(header) + header->offsets[slot];
}

static inline ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    return ctx.kernels_q4;
}

static inline ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    return ctx.kernels_q8;
}

template <typename SelectFallback>
static int kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    int count = 0;
    if (!primary) {
        return 0;
    }
    out[count++] = primary;

    if ((primary->required_cpu & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        const cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~CPU_FEATURE_SME);
        if (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            if (fallback && fallback != primary &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    return count;
}

static int kleidiai_collect_kernel_chain(const struct ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    return kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { return ggml_kleidiai_select_kernels(mask, op); });
}

static int kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    return kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { return ggml_kleidiai_select_kernels_q4_0(mask); });
}

static int kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    return kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { return ggml_kleidiai_select_kernels_q8_0(mask); });
}

static inline int64_t ggml_ne(const ggml_tensor * tensor, int dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    return tensor->ne[dim];
}

namespace ggml::cpu::kleidiai {

static size_t round_down(size_t x, size_t y) {
    return y == 0 ? x : x - (x % y);
}

static void transpose_f32kxn_f16nxk(size_t n, size_t k, float * dst, const uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / sizeof(uint16_t);
    size_t dst_stride = n;

    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        const int slot_count = kleidiai_collect_kernel_chain(op, kernel_chain);
        if (slot_count == 0) {
            return false;
        }

        const bool is_gemv = op->src[1]->ne[1] == 1;
        const size_t k = op->src[0]->ne[0];
        const size_t n = op->src[0]->ne[1];
        const size_t m = op->src[1]->ne[1];

        if (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0) {
            const size_t qk = (op->src[0]->type == GGML_TYPE_Q4_0) ? QK4_0 : QK8_0;

            size_t cursor = 0;
            bool any_slot = false;

            for (int slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                if (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    return false;
                }

                const size_t mr = kernel->get_mr();
                const size_t kr = kernel->get_kr();
                const size_t sr = kernel->get_sr();

                const size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = true;
            }

            if (!any_slot) {
                return false;
            }

            size = cursor;
            return true;
        }

        if (op->src[0]->type == GGML_TYPE_F16) {
            const int64_t lhs_batch_size0 = op->src[1]->ne[2];
            const int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            const int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            bool any_slot = false;

            for (int slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                if (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    return false;
                }

                const size_t mr = kernel->get_mr();
                const size_t kr = kernel->get_kr();
                const size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = true;
            }

            for (int slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                if (!kernel || !kernels->rhs_info.packed_size_ex) {
                    return false;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * sizeof(float);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * sizeof(float);

            if (!any_slot) {
                return false;
            }

            size = cursor;
            return true;
        }

        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * dst) override {
        if (dst->op == GGML_OP_MUL_MAT) {
            if (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0) {
                return compute_forward_qx(params, dst);
            } else if (dst->src[0]->type == GGML_TYPE_F16) {
                return compute_forward_fp16(params, dst);
            }
        } else if (dst->op == GGML_OP_GET_ROWS) {
            if (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0) {
                return compute_forward_get_rows(params, dst);
            }
        }
        return false;
    }

    bool compute_forward_fp16(ggml_compute_params * params, struct ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        if (!kernels) {
            return false;
        }

        const bool is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        if (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            return false;
        }

        const int nth = params->nth;
        const int ith = params->ith;

        const int64_t lhs_batch_size0 = ne12;
        const int64_t rhs_batch_size0 = ne02;
        const int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        const int64_t r = lhs_batch_size0 / rhs_batch_size0;

        const int64_t m_group = ne11;
        const int64_t m       = m_group;
        const int64_t n       = ne01;
        const int64_t k       = ne00;

        const size_t lhs_stride = src1->nb[1];
        const size_t rhs_stride = src0->nb[1];
        const size_t dst_stride = dst->nb[1];

        const int64_t mr = (int64_t) kernel->get_mr();
        const int64_t nr = (int64_t) kernel->get_nr();
        const int64_t kr = (int64_t) kernel->get_kr();
        const int64_t sr = (int64_t) kernel->get_sr();

        const size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        const size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        const size_t kxn_size        = k * n * sizeof(float);
        const size_t bias_size       = n * sizeof(float);

        const size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const int64_t rhs_batch_idx = batch_idx / r;
            const uint8_t * rhs_batch_base = static_cast<const uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                const int64_t m_roundup_mr = kai_roundup(m, mr);
                const int64_t num_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                if (ith < num_threads) {
                    const int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / num_threads), (size_t)mr);
                    const int64_t num_m_per_threadN_1 = m - (num_threads - 1) * num_m_per_thread0;

                    const int64_t m_start = ith * num_m_per_thread0;
                    const int64_t m_count = (ith == num_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    const size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    const size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    const size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    while (remaining > 0) {
                        const int64_t row_in_group = cur;
                        const int64_t avail        = m_group - row_in_group;
                        const int64_t take         = std::min(avail, remaining);

                        const uint8_t * lhs_batch_base = static_cast<const uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        const void * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        const size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        void * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            if (ith == 0) {
                memset(bias, 0, (size_t)n * sizeof(float));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<float *>(rhs_kxn),
                                        reinterpret_cast<const uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * sizeof(float),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                const int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t num_threads_n = KAI_MIN(n / n_step, nth);
                if (num_threads_n <= 0) {
                    num_threads_n = 1;
                }

                if (ith < num_threads_n) {
                    const int64_t num_n_per_thread0   = round_down((size_t)(n / num_threads_n), (size_t)n_step);
                    const int64_t num_n_per_threadN_1 = n - (num_threads_n - 1) * num_n_per_thread0;

                    const int64_t n_start      = ith * num_n_per_thread0;
                    const int64_t n_to_process = (ith == num_threads_n - 1) ? num_n_per_threadN_1 : num_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    const size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    const size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    const size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    const void * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    const void * rhs_ptr = rhs_packed + rhs_packed_offset;
                    float * dst_ptr      = reinterpret_cast<float *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, sizeof(float), -FLT_MAX, FLT_MAX);
                }
            }

            if (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        return true;
    }

    bool compute_forward_qx(struct ggml_compute_params * params, struct ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0);

        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        const kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        const bool has_header = kleidiai_is_weight_header_valid(header);
        const bool is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        const int slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](int slot_index, size_t & size_out) -> const uint8_t * {
            if (slot_index < 0 || slot_index >= slot_total) {
                return nullptr;
            }
            if (has_header) {
                if (slot_index < header->slot_count) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    return kleidiai_weight_slot_ptr(header, slot_index);
                }
                return nullptr;
            }
            if (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                return static_cast<const uint8_t *>(src0->data);
            }
            return nullptr;
        };

        struct runtime_slot {
            int slot_index;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_step;
            size_t lhs_packed_size;
            size_t lhs_offset;
            size_t n_offset;
            size_t n_cols;
            int assigned_threads;
            int thread_begin;
            int thread_end;
            const uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        int runtime_count = 0;

        for (int slot = 0; slot < slot_total && runtime_count < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            if (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                continue;
            }

            size_t rhs_size = 0;
            const uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            if (!rhs_ptr || rhs_size == 0) {
                continue;
            }

            runtime[runtime_count] = {
                slot,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_step(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_count;
        }

        if (runtime_count == 0) {
            ggml_kleidiai_kernels * fallback = ggml_kleidiai_select_kernels(ctx.features, dst);
            if (!fallback) {
                return false;
            }
            kernel_info * kinfo      = is_gemv ? &fallback->gemv : &fallback->gemm;
            lhs_packing_info * linfo = is_gemv ? &fallback->gemv_lhs_info : &fallback->gemm_lhs_info;
            rhs_packing_info * rinfo = &fallback->rhs_info;
            if (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset ||
                !rinfo || !rinfo->pack_func_ex || !rinfo->packed_size_ex) {
                return false;
            }
            kernel_chain[0] = fallback;
            runtime[0] = {
                0,
                fallback,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_step(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                nullptr
            };
            size_t rhs_size_fallback = 0;
            const uint8_t * rhs_base = weight_for_slot(0, rhs_size_fallback);
            if (!rhs_base) {
                rhs_base = static_cast<const uint8_t *>(src0->data);
            }
            runtime[0].rhs_base = rhs_base;
            runtime_count = 1;
        }

        const int nth_total = params->nth > 0 ? params->nth : 1;
        const int ith_total = params->ith;

        int sme_slot = -1;
        for (int i = 0; i < runtime_count; ++i) {
            if ((runtime[i].kernels->required_cpu & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
                sme_slot = i;
                break;
            }
        }

        const int sme_cap_limit = ctx.sme_thread_cap;
        const bool use_hybrid = sme_cap_limit > 0 &&
                                 runtime_count > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        if (runtime_count > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        const bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        if (!hybrid_enabled) {
            int chosen_slot = 0;
            if (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = sme_slot;
            } else if (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            if (chosen_slot != 0 && chosen_slot < runtime_count) {
                runtime[0] = runtime[chosen_slot];
            }
            runtime_count = runtime_count > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            if (runtime_count > 0 &&
                (runtime[0].kernels->required_cpu & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
                sme_slot = 0;
            }
        }

        int sme_cap = kleidiai_sme_thread_cap();
        if (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        int threads_remaining = nth_total;
        if (sme_slot != -1) {
            int sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        int fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        int fallback_count = 0;
        for (int i = 0; i < runtime_count; ++i) {
            if (i == sme_slot) {
                continue;
            }
            fallback_indices[fallback_count++] = i;
        }

        for (int fi = 0; fi < fallback_count; ++fi) {
            if (threads_remaining <= 0) {
                break;
            }
            const int slot_index = fallback_indices[fi];
            const int slots_left = fallback_count - fi;
            int share = (threads_remaining + slots_left - 1) / slots_left;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        if (threads_remaining > 0) {
            const int fallback_slot = (sme_slot != -1) ? sme_slot : 0;
            runtime[fallback_slot].assigned_threads += threads_remaining;
            threads_remaining = 0;
        }

        int thread_cursor = 0;
        for (int i = 0; i < runtime_count; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }

        if (thread_cursor < nth_total && runtime_count > 0) {
            runtime[runtime_count - 1].assigned_threads += nth_total - thread_cursor;
            runtime[runtime_count - 1].thread_end = nth_total;
        }

        int local_slot = -1;
        int local_ith  = 0;
        for (int i = 0; i < runtime_count; ++i) {
            if (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                break;
            }
        }
        if (local_slot == -1) {
            return false;
        }

        const size_t k = ne00;
        const size_t m = ne11;
        const size_t n = ne01;

        size_t cursor = 0;
        for (int i = 0; i < runtime_count; ++i) {
            const ggml_type slot_rhs_type = runtime[i].kernels->rhs_type;
            const size_t slot_pack_size_arg = slot_rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                              slot_rhs_type == GGML_TYPE_Q8_0 ? QK8_0 : 0;
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, slot_pack_size_arg, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t assigned_cols = 0;
        uint64_t weighted_total = 0;
        if (runtime_count > 1 && sme_slot != -1) {
            for (int i = 0; i < runtime_count; ++i) {
                const uint64_t weight = (i == sme_slot) ? (sme_cap << 1) : 1;
                weighted_total += (uint64_t)runtime[i].assigned_threads * weight;
            }
        }
        for (int i = 0; i < runtime_count; ++i) {
            runtime[i].n_offset = assigned_cols;
            if (runtime[i].assigned_threads == 0) {
                runtime[i].n_cols = 0;
                continue;
            }
            const size_t remaining_cols = n - assigned_cols;
            if (remaining_cols == 0) {
                runtime[i].n_cols = 0;
                continue;
            }
            const size_t step = runtime[i].n_step ? runtime[i].n_step : 1;
            size_t target      = 0;
            if (weighted_total > 0) {
                const uint64_t weight = (i == sme_slot) ? (sme_cap << 1) : 1;
                target = (size_t)(((uint64_t)n * runtime[i].assigned_threads * weight) / weighted_total);
            } else {
                target = (size_t)(((uint64_t)n * runtime[i].assigned_threads) / nth_total);
            }
            target             = std::min(target, remaining_cols);
            size_t aligned     = round_down(target, step);
            if (aligned == 0 && remaining_cols >= step) {
                aligned = step;
            }
            runtime[i].n_cols = aligned;
            assigned_cols += aligned;
        }

        if (assigned_cols < n) {
            for (int i = runtime_count - 1; i >= 0; --i) {
                if (runtime[i].assigned_threads > 0) {
                    runtime[i].n_cols += n - assigned_cols;
                    break;
                }
            }
        }
        const size_t dst_stride = dst->nb[1];

        for (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            const uint8_t * lhs_batch_base = static_cast<const uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            if (runtime[local_slot].assigned_threads > 0) {
                runtime_slot & slot = runtime[local_slot];
                const ggml_type slot_rhs_type = slot.kernels->rhs_type;
                const size_t slot_lhs_exec_arg = slot_rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                                 slot_rhs_type == GGML_TYPE_Q8_0 ? 0 : 0;
                const int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                const int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                if (local_ith < use_threads) {
                    const int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    const int64_t num_m_per_threadN_1 = (int64_t)m - (use_threads - 1) * num_m_per_thread0;

                    const int64_t m_start = (int64_t)local_ith * num_m_per_thread0;
                    const int64_t m_count = (local_ith == use_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    const size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot_lhs_exec_arg, slot.mr, slot.kr, slot.sr);
                    const size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot_lhs_exec_arg, slot.mr, slot.kr, slot.sr);
                    const size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    while (remaining > 0) {
                        const int64_t row_in_group = cur;
                        const int64_t avail        = (int64_t)m - row_in_group;
                        const int64_t take         = std::min(avail, remaining);

                        const size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        const void * src_ptr = lhs_batch_base + src_off;
                        const size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        void * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot_lhs_exec_arg, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            ggml_barrier(params->threadpool);

            runtime_slot & slot = runtime[local_slot];
            if (slot.n_cols > 0 && slot.assigned_threads > 0) {
                int64_t active_threads = slot.assigned_threads;
                const int64_t max_threads = slot.n_step ? (slot.n_cols / slot.n_step) : slot.assigned_threads;
                if (max_threads > 0) {
                    active_threads = std::min<int64_t>(active_threads, std::max<int64_t>(1, max_threads));
                }
                active_threads = std::max<int64_t>(1, active_threads);

                if (local_ith < active_threads) {
                    const size_t step = slot.n_step ? slot.n_step : 1;
                    const size_t chunk0 = round_down((size_t)(slot.n_cols / active_threads), step);
                    const size_t chunkN = slot.n_cols - (active_threads - 1) * chunk0;
                    const size_t local_start = (size_t)local_ith * chunk0;
                    const size_t cols = (local_ith == active_threads - 1) ? chunkN : chunk0;

                    if (cols > 0) {
                        const ggml_type slot_rhs_type = slot.kernels->rhs_type;
                        const size_t slot_lhs_exec_arg = slot_rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                                         slot_rhs_type == GGML_TYPE_Q8_0 ? 0 : 0;
                        const size_t slot_rhs_block_arg = slot_rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                                          slot_rhs_type == GGML_TYPE_Q8_0 ? 0 : 0;
                        const size_t global_start = slot.n_offset + local_start;
                        const size_t lhs_packed_offset = slot.lhs_info->get_packed_offset_ex(0, k, slot_lhs_exec_arg, slot.mr, slot.kr, slot.sr);
                        const size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot_rhs_block_arg);
                        const size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

                        const uint8_t * lhs_ptr = scratch + slot.lhs_offset + lhs_packed_offset;
                        const uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
                        float * dst_ptr         = reinterpret_cast<float *>(dst_batch_base + dst_offset);

                        slot.kernel->run_kernel_ex(m, cols, k, slot_rhs_block_arg,
                                                   lhs_ptr,
                                                   rhs_ptr,
                                                   dst_ptr,
                                                   dst_stride,
                                                   sizeof(float),
                                                   -FLT_MAX,
                                                   FLT_MAX);
                    }
                }
            }

            if (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        return true;
    }

    bool compute_forward_get_rows(struct ggml_compute_params * params, struct ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0);
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        const kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        const bool has_header = kleidiai_is_weight_header_valid(header);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        const bool want_q8 = src0->type == GGML_TYPE_Q8_0;
        const int chain_count = want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : kleidiai_collect_q4_chain(kernel_chain);

        ggml_kleidiai_kernels * kernels = nullptr;
        const uint8_t * packed_base = static_cast<const uint8_t *>(src0->data);

        if (has_header && chain_count > 0) {
            int select_slot = 0;
            if (select_slot >= header->slot_count) {
                select_slot = header->slot_count - 1;
            }
            if (select_slot >= 0 && select_slot < chain_count) {
                kernels = kernel_chain[select_slot];
                const uint8_t * slot_ptr = kleidiai_weight_slot_ptr(header, select_slot);
                if (slot_ptr) {
                    packed_base = slot_ptr;
                }
            }
        }

        if (!kernels && chain_count > 0) {
            kernels = kernel_chain[0];
            if (has_header) {
                const uint8_t * slot_ptr = kleidiai_weight_slot_ptr(header, 0);
                if (slot_ptr) {
                    packed_base = slot_ptr;
                }
            }
        }

        if (!kernels) {
            return false;
        }

        rhs_packing_info * rhs_info = &kernels->rhs_info;
        kernel_info * kernel        = &kernels->gemm;
        if (!rhs_info->to_float || !kernel->get_nr) {
            return false;
        }

        const int64_t nc     = ne00;
        const int64_t nr     = ggml_nelements(src1);

        const ggml_type rhs_type = kernels->rhs_type;
        size_t block_len = 0;
        size_t num_bytes_multiplier = 0;
        if (rhs_type == GGML_TYPE_Q4_0) {
            block_len = QK4_0;
            num_bytes_multiplier = sizeof(uint16_t);
        } else if (rhs_type == GGML_TYPE_Q8_0) {
            block_len = QK8_0;
            num_bytes_multiplier = sizeof(float);
        } else {
            return false;
        }

        const size_t block_rows = kernel->get_nr();
        const size_t kr         = kernel->get_kr();

        const size_t packed_stride = rhs_info->packed_stride(nc, block_rows, kr, block_len);

        const int ith = params->ith;
        const int nth = params->nth;

        const int dr = (nr + nth - 1) / nth;
        const int ir0 = dr * ith;
        const int ir1 = MIN(ir0 + dr, nr);

        for (int64_t i = ir0; i < ir1; ++i) {
            GGML_ASSERT(src1->type == GGML_TYPE_I32);
            int64_t row_idx = ((const int32_t *)src1->data)[i];
            GGML_ASSERT(row_idx >= 0 && row_idx < src0->ne[1]);

            float *out = (float *)((char *)dst->data + i * nb1);
            rhs_info->to_float(packed_base, row_idx, nc, out, block_rows, packed_stride, kr, block_len, num_bytes_multiplier);
        }

        return true;
    }

public:
    int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q8_0);
        const size_t n = tensor->ne[1];
        const size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        if (!header) {
            return -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_count = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = sizeof(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        const bool want_q8 = tensor->type == GGML_TYPE_Q8_0;
        const int slot_total = want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                       : kleidiai_collect_q4_chain(kernel_chain);
        const bool allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<float>  scales;

        if (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            const size_t row_stride = tensor->nb[1];
            const size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            for (size_t row = 0; row < n; ++row) {
                const auto * row_blocks = reinterpret_cast<const block_q8_0 *>(
                    static_cast<const uint8_t *>(data) + row * row_stride);

                float max_abs = 0.0f;
                for (size_t block = 0; block < k_blocks; ++block) {
                    const block_q8_0 & blk = row_blocks[block];
                    const float d = GGML_FP16_TO_FP32(blk.d);
                    for (size_t l = 0; l < QK8_0; ++l) {
                        const size_t linear_idx = block * QK8_0 + l;
                        if (linear_idx >= k) {
                            break;
                        }
                        const float value = d * static_cast<float>(blk.qs[l]);
                        max_abs = std::max(max_abs, std::fabs(value));
                    }
                }

                float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                for (size_t block = 0; block < k_blocks; ++block) {
                    const block_q8_0 & blk = row_blocks[block];
                    const float d = GGML_FP16_TO_FP32(blk.d);
                    for (size_t l = 0; l < QK8_0; ++l) {
                        const size_t linear_idx = block * QK8_0 + l;
                        if (linear_idx >= k) {
                            break;
                        }
                        const float value = d * static_cast<float>(blk.qs[l]);
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                }
            }
        }

        for (int slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            if (!allow_fallback && slot > 0) {
                break;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            if (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                continue;
            }

            const size_t nr = kernel->get_nr();
            const size_t kr = kernel->get_kr();
            const size_t sr = kernel->get_sr();
            const ggml_type rhs_type = kernels->rhs_type;
            const size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 : 0;
            if (block_len == 0) {
                continue;
            }

            const size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            const size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            if (rhs_type == GGML_TYPE_Q4_0) {
                struct kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<const uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } else if (rhs_type == GGML_TYPE_Q8_0) {
                struct kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } else {
                continue;
            }

            header->offsets[header->slot_count] = aligned_cursor;
            header->sizes[header->slot_count]   = packed_size;
            ++header->slot_count;

            cursor = aligned_cursor + packed_size;
        }

        if (header->slot_count == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        return 0;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::kleidiai

static enum ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "CPU_KLEIDIAI";
}

static ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    if (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q8_0) {
        return ggml_nbytes(tensor);
    }

    const size_t n = tensor->ne[1];
    const size_t k = tensor->ne[0];

    size_t cursor = sizeof(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    const bool want_q8 = tensor->type == GGML_TYPE_Q8_0;
    const int slot_total = want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                   : kleidiai_collect_q4_chain(kernel_chain);
    const bool allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_count = 0;
    for (int slot = 0; slot < slot_total; ++slot) {
        if (!allow_fallback && slot > 0) {
            break;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        if (!kernels) {
            continue;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        if (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            continue;
        }

        const ggml_type rhs_type = kernels->rhs_type;
        const size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 : 0;
        if (block_len == 0) {
            continue;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_count;
    }

    if (slot_count == 0) {
        return ggml_nbytes(tensor);
    }

    return std::max(cursor, ggml_nbytes(tensor));
}

namespace ggml::cpu::kleidiai {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        const int slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_GET_ROWS) &&
            (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0) &&
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type() &&
            slot_total > 0) {
            if (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                return false;
            }
            if (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                return false;
            }
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if ((op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_I32) &&
                ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1) {
                return true;
            }
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_GET_ROWS) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } else {
                if(op->src[0]->type != GGML_TYPE_F16) {
                    return nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                const int slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                if (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    if ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        return nullptr;
                    }
                    return ggml::cpu::kleidiai::get_tensor_traits(NULL, NULL);
                }
            }
        }
        return nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void) {
    static ggml::cpu::kleidiai::extra_buffer_type ctx;
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    return &ggml_backend_cpu_buffer_type_kleidiai;
}
