// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#include <arm_neon.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cfloat>
#include <cctype>
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
#include <map>
#include <iostream>
#include <climits>
#include <charconv>
#include <system_error>
#if defined(__linux__)
#include <asm/hwcap.h>
#include <dirent.h>
#include <sys/auxv.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include "kleidiai.h"

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-feats.h"
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
    ggml_kleidiai_kernels * kernels_f32;
    int sme_thread_cap; // <= 0 means "SME disabled/unknown"
    int thread_hint;    // <= 0 means "no hint"
    int chunk_multiplier;
} static ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, 0, -1, 4 };

static inline bool is_sme_family(cpu_feature f) {
    return (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

static const char* cpu_feature_to_string(cpu_feature f) {
    if (f == CPU_FEATURE_NONE) {
        return "NONE";
    } else if ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        return "SME2";
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

#if defined(__linux__) && defined(__aarch64__)
static bool parse_cpu_dir_name(const char* name, size_t* cpu) {
    if (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        return false;
    }

    const char* first = name + 3;
    const char* last = name + strlen(name);

    size_t value = 0;
    const auto [end, ec] = std::from_chars(first, last, value, 10);

    if (ec != std::errc{} || end != last) {
        return false;
    }

    *cpu = value;
    return true;
}

static std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    if (dir == nullptr) {
        return cpus;
    }

    while (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        if (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}
#endif

#if defined(__APPLE__) && defined(__aarch64__)
static bool apple_sme_counted_perf_level(std::string name) {
    for (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (char) std::tolower((unsigned char) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    return name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

static void add_smcus_from_smidr(uint64_t smidr, size_t & num_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    const uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    const uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    const size_t shared_count = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    const uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    const uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    const uint32_t id = (affinity2 << 12) | affinity;

    if (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    switch (sh) {
        case 2: // private SMCU
            ++num_private;
            break;
        case 3: // shared SMCU
            if (shared_counts[id] < shared_count) {
                shared_counts[id] = shared_count;
            }
            break;
        case 0:
            if (id == 0) {
                ++num_private;
            } else if (shared_counts[id] < shared_count) {
                shared_counts[id] = shared_count;
            }
            break;
        default:
            break;
    }
}

static size_t detect_num_smcus() {
    const auto runtime_feat = ggml_feats_get_arch64_runtime();
    if (!runtime_feat.has_sme) {
        return 0;
    }

#if defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t num_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    const std::vector<size_t> cpus = detect_cpu_ids();
    for (const size_t cpu : cpus) {
        const std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        if (!file.is_open()) {
            continue;
        }

        uint64_t smidr = 0;
        if (!(file >> std::hex >> smidr)) {
            continue;
        }

        add_smcus_from_smidr(smidr, num_private, shared_counts);
    }

    size_t total = num_private;
    for (const auto & entry : shared_counts) {
        total += entry.second;
    }
    return total;

#elif defined(__APPLE__) && defined(__aarch64__)
    int perf_levels = 0;
    size_t size = sizeof(perf_levels);
    if (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != sizeof(perf_levels) || perf_levels <= 0) {
        return 0;
    }

    size_t units = 0;
    for (int i = 0; i < perf_levels; ++i) {
        char key[64] = {};
        int physical_cpus = 0;
        int cpus_per_l2 = 0;

        snprintf(key, sizeof(key), "hw.perflevel%d.physicalcpu", i);
        size = sizeof(physical_cpus);
        if (sysctlbyname(key, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != sizeof(physical_cpus) || physical_cpus <= 0) {
            continue;
        }

        snprintf(key, sizeof(key), "hw.perflevel%d.cpusperl2", i);
        size = sizeof(cpus_per_l2);
        if (sysctlbyname(key, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != sizeof(cpus_per_l2) || cpus_per_l2 <= 0) {
            continue;
        }

        snprintf(key, sizeof(key), "hw.perflevel%d.name", i);
        size = 0;
        if (sysctlbyname(key, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            continue;
        }

        std::string name(size, '\0');
        if (sysctlbyname(key, &name[0], &size, nullptr, 0) != 0) {
            continue;
        }
        name.resize(size);
        while (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        if (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    return units;

#elif defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    return 0;

#else
    return 0;
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

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        const char *env_sme         = getenv("GGML_KLEIDIAI_SME");
        const char *env_threads     = getenv("GGML_TOTAL_THREADS");
        const char *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        const auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        if (env_threads) {
            bool ok = false;
            int hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            if (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        if (env_chunk_mult) {
            bool ok = false;
            int multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            if (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        int sme_cores = 0;
        bool sme_env_ok = false;
        bool sme_env_set = (env_sme != nullptr);

        const bool has_supported_sme_family = runtime_feat.has_sme;
        bool sme_cap_detected = false;

        if (has_supported_sme_family) {
            detected_smcus = detect_num_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (int)detected_smcus : 1;

            if (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        if (sme_env_set) {
            bool ok = false;
            int v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            if (ok) {
                if (has_supported_sme_family) {
                    sme_cores = v;
                } else {
                    if (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } else {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        if (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            if (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

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

        if (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (int)ctx.features);
        } else {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        const bool has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        if (has_selected_sme_family_kernel) {
            if (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } else if (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } else {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
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

static inline size_t gcd_size(size_t a, size_t b) {
    while (b != 0) {
        const size_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static inline bool lcm_size(size_t a, size_t b, size_t & result) {
    if (a == 0 || b == 0) {
        result = 0;
        return false;
    }
    const size_t g = gcd_size(a, b);
    const size_t q = a / g;
    if (q > SIZE_MAX / b) {
        return false;
    }
    result = q * b;
    return true;
}

static inline size_t ceil_div_size(size_t a, size_t b) {
    return b == 0 ? 0 : (a + b - 1) / b;
}

static inline size_t kleidiai_chunk_cols(size_t n, int nth_total, bool disable_chunking, size_t n_step) {
    const size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    const size_t divisor = std::max<size_t>(1, (size_t) nth_total * multiplier);
    const size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, divisor)), n_step);
    return chunk_cols ? chunk_cols : n_step;
}

struct kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

static inline kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    switch (rhs_type) {
        case GGML_TYPE_Q4_0:
            return { QK4_0, QK4_0, QK4_0 };
        case GGML_TYPE_Q8_0:
            return { 0, 0, QK8_0 };
        default:
            return { 0, 0, 0 };
    }
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

static inline ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    return ctx.kernels_f32;
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

    if (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        return count;
    }

    if (is_sme_family(primary->required_cpu)) {
        const cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        if (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            if (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
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

static int kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    return kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { return ggml_kleidiai_select_kernels_f32(mask); });
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

        if (op->src[0]->type == GGML_TYPE_F32) {
            size_t cursor = 0;
            bool any_slot = false;

            for (int slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                if (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    return false;
                }

                const size_t mr = kernel->get_mr();
                const size_t kr = kernel->get_kr();
                const size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
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
            } else if (dst->src[0]->type == GGML_TYPE_F32) {
                return compute_forward_f32(params, dst);
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

    bool compute_forward_f32(ggml_compute_params * params, struct ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            return false;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        if (!kernels) {
            return false;
        }

        kernel_info * kernel        = &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        if (!kernel || !lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
            !lhs_info->packed_size_ex || !lhs_info->pack_func_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            return false;
        }

        const kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        const bool has_header = kleidiai_is_weight_header_valid(header);

        const uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<const uint8_t *>(src0->data);
        if (!rhs_base) {
            return false;
        }

        const int nth = params->nth > 0 ? params->nth : 1;
        const int ith = params->ith;

        const size_t k = ne00;
        const size_t m = ne11;
        const size_t n = ne01;

        const size_t mr = kernel->get_mr();
        const size_t kr = kernel->get_kr();
        const size_t sr = kernel->get_sr();

        const size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        GGML_ASSERT(lhs_packed_size <= params->wsize);

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        const size_t dst_stride = dst->nb[1];
        const size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        const bool disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        for (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            const uint8_t * lhs_batch_base = static_cast<const uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            {
                const int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                const int64_t use_threads = std::min<int64_t>(nth, max_threads);

                if (ith < use_threads) {
                    const int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    const int64_t num_m_per_threadN_1 = (int64_t)m - (use_threads - 1) * num_m_per_thread0;

                    const int64_t m_start = (int64_t)ith * num_m_per_thread0;
                    const int64_t m_count = (ith == use_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    const size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    const size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    const size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    while (remaining > 0) {
                        const int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        const size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        const void * src_ptr = lhs_batch_base + src_off;
                        const size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        void * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            if (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            const size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            int current_col = ggml_threadpool_chunk_add(params->threadpool, (int) chunk_cols);
            while ((size_t) current_col < n) {
                const size_t n_start = (size_t) current_col;
                const size_t n_to_process = std::min(chunk_cols, n - n_start);

                if (n_to_process > 0) {
                    const size_t lhs_packed_offset = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    const size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    const size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    const void * lhs_ptr = lhs_packed + lhs_packed_offset;
                    const void * rhs_ptr = rhs_base + rhs_packed_offset;
                    float * dst_ptr      = reinterpret_cast<float *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          sizeof(float),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (int) chunk_cols);
            }

            if (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        return true;
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
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
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

            const kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

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
                block_args.lhs_bl,
                block_args.rhs_bl,
                block_args.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_count;
        }

        if (runtime_count == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            return false;
        }

        const int nth_total = params->nth > 0 ? params->nth : 1;
        const int ith_total = params->ith;

        int sme_slot = -1;
        int non_sme_slot = -1;
        for (int i = 0; i < runtime_count; ++i) {
            if (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                break;
            }
        }

        for (int i = 0; i < runtime_count; ++i) {
            if (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
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
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } else if (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            if (chosen_slot != 0 && chosen_slot < runtime_count) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_count = runtime_count > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            if (runtime_count > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
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
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
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
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        for (int i = 0; i < runtime_count; ++i) {
            if (runtime[i].assigned_threads == 0) {
                continue;
            }
            size_t next_step = 0;
            if (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                return false;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        const bool disable_chunking = ggml_is_numa();
        const size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        const size_t chunk_divisor = (nth_total == 1 || disable_chunking) ? (size_t)nth_total : (size_t)nth_total * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_divisor)), common_step);
        if (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        const size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        const int nchunk = (int)nchunk_size;
        const size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            const size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            const size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            const uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            const uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            float * dst_ptr         = reinterpret_cast<float *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       sizeof(float),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        for (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            const uint8_t * lhs_batch_base = static_cast<const uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            if (runtime[local_slot].assigned_threads > 0) {
                runtime_slot & slot = runtime[local_slot];
                const int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                const int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                if (local_ith < use_threads) {
                    const int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    const int64_t num_m_per_threadN_1 = (int64_t)m - (use_threads - 1) * num_m_per_thread0;

                    const int64_t m_start = (int64_t)local_ith * num_m_per_thread0;
                    const int64_t m_count = (local_ith == use_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    const size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    const size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
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

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            if (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, nth_total);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            runtime_slot & slot = runtime[local_slot];
            int current_chunk = ith_total;
            while (current_chunk < nchunk) {
                const size_t global_start = (size_t)current_chunk * chunk_cols;
                if (global_start >= n) {
                    break;
                }

                const size_t cols = std::min(chunk_cols, n - global_start);
                if (cols > 0) {
                    // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                    // only non-tail chunks are guaranteed to be n_step-aligned.
                    run_chunk(slot, global_start, cols, dst_batch_base);
                }

                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
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
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_F32);
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
        const bool want_f32 = tensor->type == GGML_TYPE_F32;
        const int slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                                  : kleidiai_collect_q4_chain(kernel_chain);
        const bool allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<float>  scales;
        std::vector<float>  bias;

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

        if (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
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
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            if (block_len == SIZE_MAX) {
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
            } else if (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
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

    if (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_F32) {
        return ggml_nbytes(tensor);
    }

    const size_t n = tensor->ne[1];
    const size_t k = tensor->ne[0];

    size_t cursor = sizeof(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    const bool want_q8 = tensor->type == GGML_TYPE_Q8_0;
    const bool want_f32 = tensor->type == GGML_TYPE_F32;
    const int slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
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
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        if (block_len == SIZE_MAX) {
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
        const bool src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type() &&
            slot_total > 0;

        if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_GET_ROWS) &&
            (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            if (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                return false;
            }
            if (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                return false;
            }
            if (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                return false;
            }
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }

            if (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0) {
                if ((op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_I32) &&
                    ggml_ne(op->src[1], 3) == 1) {
                    return true;
                }
                return false;
            }

            if (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }

            return true;
        }

        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_GET_ROWS) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } else {
                // KleidiAI only has kernels for Q4_0 and Q8_0. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                if (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q8_0) {
                    static std::atomic<bool> warned(false);
                    if (!warned.exchange(true)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0 and Q8_0)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                if (op->src[0]->type != GGML_TYPE_F16) {
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
