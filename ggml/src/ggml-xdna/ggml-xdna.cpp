// AMD XDNA (Ryzen AI NPU) backend
//
// Design: a narrow accelerator backend in the spirit of ggml-blas. It lives on host
// memory (CPU buffer type) so the scheduler can hand it individual ops out of a
// CPU graph with no tensor copies, and it only claims MUL_MATs that map onto the
// precompiled block GEMM kernels it finds at startup. Everything else stays on CPU.
//
// Kernel contract (see kernels/README.md; built with mlir-aie whole_array.py using
// --b-col-maj 1 --c-col-maj 1 --dtype_in bf16 --dtype_out f32):
//   A : [M x K] bf16 row-major   -> a block of weight rows (ggml src0, ne01 x ne00)
//   B : [N x K] bf16 row-major   -> a block of activation rows (ggml src1, ne11 x ne10)
//   C : [N x M] f32  row-major   -> C[t][r] = sum_k A[r][k] * B[t][k], i.e. exactly the
//                                   ggml dst layout (ne0 = ne01 features, ne1 = tokens)
// so no transposes are needed on either side; larger matmuls are tiled over (M, K, N)
// blocks and partial K products are accumulated on the host.
//
// CPU + GPU + NPU: a first-generation XDNA part is slower than eight Zen 4 cores at dense GEMM,
// and far slower than the integrated Radeon, so each admitted MUL_MAT is additionally split by
// output features between the NPU, a private Vulkan backend on the integrated GPU (zero-copy
// activations/dst on UMA, cached device weights) and a private CPU backend, all computing
// their rows concurrently; the split can adapt to the measured times of the workers
// (see ggml_backend_xdna_mul_mat).

#include "ggml-xdna.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#ifdef GGML_XDNA_HAVE_CPU
#include "ggml-cpu.h"
#endif
#ifdef GGML_XDNA_HAVE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

#define GGML_XDNA_KERNEL_PREFIX "MLIR_AIE"   // kernel name prefix inside IRON-built xclbins
#define GGML_XDNA_OPCODE_TXN    3u           // opcode for a transaction-binary instruction stream

// a precompiled block GEMM for one fixed (M, K, N)
struct ggml_xdna_kernel {
    int64_t M = 0;   // A rows  (weight features per block)
    int64_t K = 0;   // reduction length per block
    int64_t N = 0;   // B rows  (tokens per block)

    std::string xclbin_path;
    std::string insts_path;

    bool loaded = false;
    bool broken = false;

    xrt::xclbin     xclbin;
    xrt::hw_context hwctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    xrt::bo         bo_a;
    xrt::bo         bo_b;
    xrt::bo         bo_c;
    size_t          n_instr = 0;

    ggml_bf16_t * a_map = nullptr;
    ggml_bf16_t * b_map = nullptr;
    float       * c_map = nullptr;
};

// process-wide device + kernel table, shared by the device (supports_op) and the backend
struct ggml_xdna_state {
    std::mutex mutex;

    bool probed    = false;
    bool available = false;

    std::string description = "AMD XDNA NPU";
    std::string kernel_dir;

    std::unique_ptr<xrt::device>  device;
    std::vector<ggml_xdna_kernel> kernels;

    int64_t min_batch = 32;   // smallest ne11 (tokens) worth sending to the NPU
};

static ggml_xdna_state & ggml_xdna_get_state() {
    static ggml_xdna_state state;
    return state;
}

// getenv that treats an empty value as unset (shells clear variables that way)
static const char * ggml_xdna_getenv(const char * name) {
    const char * v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

static bool ggml_xdna_debug() {
    static const bool debug = ggml_xdna_getenv("GGML_XDNA_DEBUG") != nullptr;
    return debug;
}

static void ggml_xdna_scan_kernels(ggml_xdna_state & st, const std::string & dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return;
    }

    static const std::regex re("^mm_bf16_f32_M([0-9]+)_K([0-9]+)_N([0-9]+)\\.xclbin$");

    for (const auto & entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, re)) {
            continue;
        }

        ggml_xdna_kernel k;
        k.M = std::stoll(m[1]);
        k.K = std::stoll(m[2]);
        k.N = std::stoll(m[3]);
        k.xclbin_path = entry.path().string();
        k.insts_path  = (entry.path().parent_path() / (name.substr(0, name.size() - 7) + ".insts.bin")).string();

        if (!fs::is_regular_file(k.insts_path, ec)) {
            GGML_LOG_WARN("%s: %s has no matching .insts.bin, skipping\n", __func__, name.c_str());
            continue;
        }
        st.kernels.push_back(std::move(k));
    }

    std::sort(st.kernels.begin(), st.kernels.end(), [](const ggml_xdna_kernel & a, const ggml_xdna_kernel & b) {
        return a.M*a.K*a.N < b.M*b.K*b.N;
    });
}

// open the NPU and enumerate kernels; cheap, done once
static void ggml_xdna_probe(ggml_xdna_state & st) {
    std::lock_guard<std::mutex> lock(st.mutex);
    if (st.probed) {
        return;
    }
    st.probed = true;

    try {
        st.device = std::make_unique<xrt::device>(0u);
        st.description = "AMD XDNA NPU (" + st.device->get_info<xrt::info::device::name>() + ")";
        st.available = true;
    } catch (const std::exception & e) {
        GGML_LOG_DEBUG("%s: no XRT device: %s\n", __func__, e.what());
        st.device.reset();
        return;
    }

    if (const char * env = ggml_xdna_getenv("GGML_XDNA_MIN_BATCH")) {
        st.min_batch = std::max<int64_t>(1, std::atoll(env));
    }

    if (const char * env = ggml_xdna_getenv("GGML_XDNA_KERNEL_DIR")) {
        st.kernel_dir = env;
    } else {
#ifdef GGML_XDNA_DEFAULT_KERNEL_DIR
        st.kernel_dir = GGML_XDNA_DEFAULT_KERNEL_DIR;
#else
        st.kernel_dir = "xdna-kernels";
#endif
    }
    ggml_xdna_scan_kernels(st, st.kernel_dir);

    if (st.kernels.empty()) {
        GGML_LOG_WARN("%s: no kernels found in %s - the XDNA backend will not claim any ops\n",
                      __func__, st.kernel_dir.c_str());
    } else {
        for (const auto & k : st.kernels) {
            GGML_LOG_INFO("%s: kernel M=%" PRId64 " K=%" PRId64 " N=%" PRId64 " (%s)\n",
                          __func__, k.M, k.K, k.N, fs::path(k.xclbin_path).filename().string().c_str());
        }
    }
}

static bool ggml_xdna_kernel_load(ggml_xdna_state & st, ggml_xdna_kernel & k) {
    if (k.loaded) {
        return true;
    }
    if (k.broken) {
        return false;
    }

    try {
        std::ifstream f(k.insts_path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("cannot open " + k.insts_path);
        }
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bytes.empty() || bytes.size() % 4 != 0) {
            throw std::runtime_error("malformed instruction stream " + k.insts_path);
        }
        k.n_instr = bytes.size() / sizeof(uint32_t);

        k.xclbin = xrt::xclbin(k.xclbin_path);

        std::string kname;
        for (const auto & xk : k.xclbin.get_kernels()) {
            const std::string name = xk.get_name();
            if (name.rfind(GGML_XDNA_KERNEL_PREFIX, 0) == 0) {
                kname = name;
                break;
            }
        }
        if (kname.empty()) {
            throw std::runtime_error("no " GGML_XDNA_KERNEL_PREFIX "* kernel in " + k.xclbin_path);
        }

        st.device->register_xclbin(k.xclbin);
        k.hwctx  = xrt::hw_context(*st.device, k.xclbin.get_uuid());
        k.kernel = xrt::kernel(k.hwctx, kname);

        // argument layout of an IRON runtime sequence: (opcode, instr, n_instr, A, B, C)
        k.bo_instr = xrt::bo(*st.device, bytes.size(),     xrt::bo::flags::cacheable, k.kernel.group_id(1));
        k.bo_a     = xrt::bo(*st.device, k.M*k.K*sizeof(ggml_bf16_t), xrt::bo::flags::host_only, k.kernel.group_id(3));
        k.bo_b     = xrt::bo(*st.device, k.N*k.K*sizeof(ggml_bf16_t), xrt::bo::flags::host_only, k.kernel.group_id(4));
        k.bo_c     = xrt::bo(*st.device, k.N*k.M*sizeof(float),       xrt::bo::flags::host_only, k.kernel.group_id(5));

        std::memcpy(k.bo_instr.map<void *>(), bytes.data(), bytes.size());
        k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        k.a_map = k.bo_a.map<ggml_bf16_t *>();
        k.b_map = k.bo_b.map<ggml_bf16_t *>();
        k.c_map = k.bo_c.map<float *>();

        k.loaded = true;
        GGML_LOG_INFO("%s: loaded %s (kernel %s, %zu instruction words)\n",
                      __func__, fs::path(k.xclbin_path).filename().string().c_str(), kname.c_str(), k.n_instr);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, k.xclbin_path.c_str(), e.what());
        k.broken = true;
        return false;
    }
}

// run one block GEMM on the NPU; a_map/b_map must be filled by the caller
static bool ggml_xdna_kernel_run(ggml_xdna_kernel & k) {
    try {
        k.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k.bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(k.kernel);
        run.set_arg(0, GGML_XDNA_OPCODE_TXN);
        run.set_arg(1, k.bo_instr);
        run.set_arg(2, k.n_instr);
        run.set_arg(3, k.bo_a);
        run.set_arg(4, k.bo_b);
        run.set_arg(5, k.bo_c);
        run.start();

        const ert_cmd_state state = run.wait();
        if (state != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("%s: kernel did not complete (ert state %d)\n", __func__, (int) state);
            return false;
        }
        k.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: XRT error: %s\n", __func__, e.what());
        return false;
    }
}

// pick the kernel that wastes the least padded work for this problem
static ggml_xdna_kernel * ggml_xdna_select_kernel(ggml_xdna_state & st, int64_t n_feat, int64_t n_k, int64_t n_tok) {
    ggml_xdna_kernel * best = nullptr;
    double best_cost = 0.0;
    for (auto & k : st.kernels) {
        if (k.broken) {
            continue;
        }
        const double launches = (double) ((n_feat + k.M - 1)/k.M) * ((n_k + k.K - 1)/k.K) * ((n_tok + k.N - 1)/k.N);
        const double cost = launches * ((double) k.M*k.K*k.N + 8.0e6);   // ~fixed launch overhead in "MACs"
        if (best == nullptr || cost < best_cost) {
            best = &k;
            best_cost = cost;
        }
    }
    return best;
}

// -------------------------------------------------------------------------------------------------
// host-side helpers

struct ggml_backend_xdna_context {
    int n_threads = GGML_DEFAULT_N_THREADS;

    std::vector<ggml_bf16_t> w_bf16;   // staged weight rows for the current M block, [M x K]
    std::vector<ggml_bf16_t> x_bf16;   // staged activations for the current plane,   [tokens x K]
    std::vector<std::future<void>> tasks;

    bool warned_fallback = false;

    // Work split: the output features of each MUL_MAT are divided between the NPU, a private
    // Vulkan backend (the integrated GPU) and a private CPU backend, all running concurrently.
    // share[] holds the fraction per worker: fixed (defaults below, or GGML_XDNA_NPU_SHARE /
    // GGML_XDNA_VK_SHARE, CPU takes the rest), or rebalanced from the measured times of the
    // workers with GGML_XDNA_NPU_SHARE=auto.
    ggml_backend_t cpu = nullptr;
    ggml_backend_t vk  = nullptr;
    bool  vk_tried = false;
    float share[3] = { 0.4f, 0.0f, 0.6f };   // NPU, VK, CPU
    bool  share_fixed = true;
    bool  shares_ready = false;
    double npu_min_gflop = 0.0;               // ops below this skip the NPU (set when a GPU exists)

    // Vulkan zero-copy: on a UMA GPU the worker imports the host ranges of the operands as
    // device buffers (VK_EXT_external_memory_host) instead of copying them. Imports are cached
    // by page-aligned range since weights and scheduler buffers keep their addresses.
    bool      vk_zero_copy = true;
    uintptr_t vk_import_align = 4096;
    std::map<std::pair<uintptr_t, uintptr_t>, ggml_backend_buffer_t> vk_imports;

    // Weight slices are copied once into device memory and kept (mmap'd model files cannot be
    // imported), keyed by (data pointer, row range) and guarded by a sampled content signature.
    struct vk_weight {
        ggml_context *        wctx = nullptr;
        ggml_backend_buffer_t buf  = nullptr;
        ggml_tensor *         a    = nullptr;
        uint64_t              sig  = 0;
        size_t                bytes = 0;
    };
    std::map<std::tuple<const void *, int64_t, int64_t>, vk_weight> vk_weights;
    size_t vk_weight_bytes = 0;
    size_t vk_weight_limit = (size_t) 8 << 30;
};

enum ggml_xdna_worker {
    GGML_XDNA_WORKER_NPU = 0,
    GGML_XDNA_WORKER_VK  = 1,
    GGML_XDNA_WORKER_CPU = 2,
    GGML_XDNA_N_WORKERS  = 3,
};

static ggml_backend_t ggml_xdna_cpu_backend(ggml_backend_xdna_context * ctx);

static ggml_backend_t ggml_xdna_vk_backend(ggml_backend_xdna_context * ctx) {
#ifdef GGML_XDNA_HAVE_VULKAN
    if (ctx->vk == nullptr && !ctx->vk_tried) {
        ctx->vk_tried = true;
        const int n_dev = ggml_backend_vk_get_device_count();
        int dev = -1;
        if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_DEVICE")) {
            dev = std::atoi(env);
        } else {
            // prefer the integrated AMD GPU that shares the die with the NPU
            for (int i = 0; i < n_dev; i++) {
                char desc[256];
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                if (std::strstr(desc, "AMD") || std::strstr(desc, "Radeon")) {
                    dev = i;
                    break;
                }
            }
            if (dev < 0 && n_dev > 0) {
                dev = 0;
            }
        }
        if (dev >= 0 && dev < n_dev) {
            ctx->vk = ggml_backend_vk_init(dev);
            char desc[256];
            ggml_backend_vk_get_device_description(dev, desc, sizeof(desc));
            GGML_LOG_INFO("%s: Vulkan worker on device %d (%s)\n", __func__, dev, desc);
        }
    }
    return ctx->vk;
#else
    GGML_UNUSED(ctx);
    return nullptr;
#endif
}

// compute dst rows [feat0, feat1) of a MUL_MAT on the Vulkan backend: the weight slice and the
// activations are copied into a device buffer, the product is read back into dst
static bool ggml_xdna_mul_mat_vk_copy(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    ggml_backend_t vk = ggml_xdna_vk_backend(ctx);
    if (vk == nullptr) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t n_rows = feat1 - feat0;

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_4d(gctx, src0->type, src0->ne[0], n_rows, src0->ne[2], src0->ne[3]);
    ggml_tensor * b = ggml_new_tensor_4d(gctx, src1->type, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    ggml_tensor * c = ggml_mul_mat(gctx, a, b);

    const auto t0 = std::chrono::steady_clock::now();
    auto ms_since = [](std::chrono::steady_clock::time_point s) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
    };

    bool ok = ggml_backend_supports_op(vk, c);
    ggml_backend_buffer_t buf = ok ? ggml_backend_alloc_ctx_tensors(gctx, vk) : nullptr;
    if (buf == nullptr) {
        ggml_free(gctx);
        return false;
    }
    const double t_alloc = ms_since(t0);

    // weight rows [feat0, feat1) are contiguous within each plane of the (contiguous) src0
    const size_t plane_bytes = n_rows*src0->nb[1];
    for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
            const char * src = (const char *) src0->data + feat0*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
            ggml_backend_tensor_set(a, src, (i2 + i3*src0->ne[2])*plane_bytes, plane_bytes);
        }
    }
    ggml_backend_tensor_set(b, src1->data, 0, ggml_nbytes(src1));
    const double t_upload = ms_since(t0);

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, c);
    ok = ggml_backend_graph_compute(vk, graph) == GGML_STATUS_SUCCESS;
    const double t_compute = ms_since(t0);

    if (ok) {
        // c is [n_rows x tokens x ne2 x ne3] contiguous; scatter each token row into dst
        std::vector<float> tmp(ggml_nelements(c));
        ggml_backend_tensor_get(c, tmp.data(), 0, ggml_nbytes(c));
        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                for (int64_t t = 0; t < dst->ne[1]; t++) {
                    float * drow = (float *) ((char *) dst->data + t*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]) + feat0;
                    std::memcpy(drow, tmp.data() + ((i3*dst->ne[2] + i2)*dst->ne[1] + t)*n_rows, n_rows*sizeof(float));
                }
            }
        }
    }
    const double t_download = ms_since(t0);

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);

    if (ggml_xdna_debug()) {
        static int n_logged = 0;
        if (n_logged++ < 12) {
            GGML_LOG_INFO("%s: %s[%" PRId64 " x %" PRId64 "] x [%" PRId64 " tok]: alloc %.2f ms, upload %.2f ms, compute %.2f ms, download %.2f ms, free %.2f ms\n",
                          __func__, ggml_type_name(src0->type), src0->ne[0], n_rows, src1->ne[1],
                          t_alloc, t_upload - t_alloc, t_compute - t_upload, t_download - t_compute, ms_since(t0) - t_download);
        }
    }
    return ok;
}

// import the host range [ptr, ptr + size) into the Vulkan device (page-rounded) and return the
// buffer plus the address the tensor must be placed at; nullptr if the device cannot import
static ggml_backend_buffer_t ggml_xdna_vk_import(ggml_backend_xdna_context * ctx, const void * ptr, size_t size, void ** tensor_addr) {
    const uintptr_t align = ctx->vk_import_align;
    const uintptr_t base  = (uintptr_t) ptr & ~(align - 1);
    const uintptr_t end   = ((uintptr_t) ptr + size + align - 1) & ~(align - 1);

    ggml_backend_buffer_t buf = nullptr;
    auto it = ctx->vk_imports.find({base, end});
    if (it != ctx->vk_imports.end()) {
        buf = it->second;
    } else {
        if (ctx->vk_imports.size() >= 4096) {
            for (auto & kv : ctx->vk_imports) {
                ggml_backend_buffer_free(kv.second);
            }
            ctx->vk_imports.clear();
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(ctx->vk);
        buf = ggml_backend_dev_buffer_from_host_ptr(dev, (void *) base, end - base, end - base);
        if (buf == nullptr) {
            return nullptr;
        }
        ctx->vk_imports[{base, end}] = buf;
    }
    *tensor_addr = (char *) ggml_backend_buffer_get_base(buf) + ((uintptr_t) ptr - base);
    return buf;
}

// cheap content signature of a weight slice (16 samples), to catch a reused address
static uint64_t ggml_xdna_weight_sig(const void * data, size_t bytes, int type, int64_t ne0) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t) type ^ ((uint64_t) ne0 << 8) ^ ((uint64_t) bytes << 24);
    const size_t n = 16;
    for (size_t i = 0; i < n && bytes >= sizeof(uint64_t); i++) {
        uint64_t v;
        std::memcpy(&v, (const char *) data + (bytes - sizeof(uint64_t))*i/(n - 1), sizeof(v));
        h = (h ^ v)*1099511628211ull;
    }
    return h;
}

// device-resident copy of weight rows [feat0, feat1): cached across calls when within budget
static ggml_tensor * ggml_xdna_vk_weight_slice(ggml_backend_xdna_context * ctx, const ggml_tensor * src0, int64_t feat0, int64_t feat1,
                                               ggml_backend_xdna_context::vk_weight & temp, bool & cached) {
    const int64_t n_rows = feat1 - feat0;
    const char *  data   = (const char *) src0->data + feat0*src0->nb[1];
    const size_t  bytes  = n_rows*src0->nb[1];
    const uint64_t sig   = ggml_xdna_weight_sig(data, bytes, src0->type, src0->ne[0]);

    const auto key = std::make_tuple((const void *) src0->data, feat0, feat1);
    auto it = ctx->vk_weights.find(key);
    if (it != ctx->vk_weights.end()) {
        if (it->second.sig == sig) {
            cached = true;
            return it->second.a;
        }
        // the address was reused by different weights: drop the stale copy
        ggml_backend_buffer_free(it->second.buf);
        ggml_free(it->second.wctx);
        ctx->vk_weight_bytes -= it->second.bytes;
        ctx->vk_weights.erase(it);
    }

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*2,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_backend_xdna_context::vk_weight w;
    w.wctx  = ggml_init(ip);
    w.a     = ggml_new_tensor_2d(w.wctx, src0->type, src0->ne[0], n_rows);
    w.buf   = ggml_backend_alloc_ctx_tensors(w.wctx, ctx->vk);
    w.sig   = sig;
    w.bytes = bytes;
    if (w.buf == nullptr) {
        ggml_free(w.wctx);
        return nullptr;
    }
    ggml_backend_tensor_set(w.a, data, 0, bytes);

    if (ctx->vk_weight_bytes + bytes <= ctx->vk_weight_limit) {
        ctx->vk_weights[key] = w;
        ctx->vk_weight_bytes += bytes;
        cached = true;
    } else {
        temp   = w;   // caller frees after the compute
        cached = false;
    }
    return w.a;
}

// zero-copy variant: the activations and the dst rows are the caller's host memory imported
// into the device (the GPU writes its rows straight into dst, strided); the weight slice is a
// cached device copy. 2-D operands only.
static bool ggml_xdna_mul_mat_vk_zero_copy(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    ggml_backend_t vk = ggml_xdna_vk_backend(ctx);
    if (vk == nullptr || !ctx->vk_zero_copy) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t n_rows = feat1 - feat0;

    if (src0->ne[2]*src0->ne[3] != 1 || src1->ne[2]*src1->ne[3] != 1) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto ms_since = [](std::chrono::steady_clock::time_point s) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
    };

    // activations and destination rows: imported in place
    void * addr_b = nullptr;
    void * addr_c = nullptr;
    ggml_backend_buffer_t buf_b = ggml_xdna_vk_import(ctx, src1->data, ggml_nbytes(src1), &addr_b);
    ggml_backend_buffer_t buf_c = buf_b ? ggml_xdna_vk_import(ctx, dst->data, ggml_nbytes(dst), &addr_c) : nullptr;
    if (buf_c == nullptr) {
        return false;
    }
    const double t_import = ms_since(t0);

    ggml_backend_xdna_context::vk_weight temp;
    bool cached = false;
    ggml_tensor * a = ggml_xdna_vk_weight_slice(ctx, src0, feat0, feat1, temp, cached);
    if (a == nullptr) {
        return false;
    }
    const double t_weights = ms_since(t0);

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * b = ggml_new_tensor_2d(gctx, src1->type, src1->ne[0], src1->ne[1]);
    ggml_tensor * c = ggml_mul_mat(gctx, a, b);
    c->nb[1] = dst->nb[1];   // write into the caller's dst rows in place
    c->nb[2] = dst->nb[2];
    c->nb[3] = dst->nb[3];

    bool ok = ggml_backend_supports_op(vk, c) &&
              ggml_backend_tensor_alloc(buf_b, b, addr_b) == GGML_STATUS_SUCCESS &&
              ggml_backend_tensor_alloc(buf_c, c, (char *) addr_c + feat0*sizeof(float)) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_cgraph * graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(graph, c);
        ok = ggml_backend_graph_compute(vk, graph) == GGML_STATUS_SUCCESS;
    }
    ggml_free(gctx);

    if (!cached) {
        ggml_backend_buffer_free(temp.buf);
        ggml_free(temp.wctx);
    }

    if (ggml_xdna_debug()) {
        static int n_logged = 0;
        if (n_logged++ < 12) {
            GGML_LOG_INFO("%s: %s[%" PRId64 " x %" PRId64 "] x [%" PRId64 " tok]: import %.2f ms, weights %.2f ms (%s), compute %.2f ms%s (%zu imports, %.0f MB weights cached)\n",
                          __func__, ggml_type_name(src0->type), src0->ne[0], n_rows, src1->ne[1],
                          t_import, t_weights - t_import, cached ? "cached" : "copied", ms_since(t0) - t_weights, ok ? "" : " FAILED",
                          ctx->vk_imports.size(), ctx->vk_weight_bytes/1048576.0);
        }
    }
    return ok;
}

static bool ggml_xdna_mul_mat_vk(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    if (ggml_xdna_mul_mat_vk_zero_copy(ctx, dst, feat0, feat1)) {
        return true;
    }
    if (ctx->vk_zero_copy && ggml_xdna_debug()) {
        GGML_LOG_INFO("%s: zero-copy path unavailable for this op, using the copy path\n", __func__);
    }
    return ggml_xdna_mul_mat_vk_copy(ctx, dst, feat0, feat1);
}

// pick the default split once we know which workers exist; env overrides
static void ggml_xdna_init_shares(ggml_backend_xdna_context * ctx) {
    if (ctx->shares_ready) {
        return;
    }
    ctx->shares_ready = true;

    const bool have_vk  = ggml_xdna_vk_backend(ctx)  != nullptr;
    const bool have_cpu = ggml_xdna_cpu_backend(ctx) != nullptr;

    // measured on a Ryzen 7 8700G: the 780M (Vulkan) is ~10x the 4-column XDNA1 at batched
    // GEMM and the 8 Zen 4 cores sit in between, so next to the GPU worker the NPU only ever
    // lengthened the critical path (Qwen3-0.6B and gemma-4-E2B, 476 tokens). It therefore stays
    // out by default when a GPU worker exists; GGML_XDNA_NPU_SHARE opts it back in, and
    // GGML_XDNA_NPU_MIN_GFLOP keeps it to the large products where it has a chance.
    float npu = have_vk ? 0.0f : 0.4f;
    float vkf = have_vk ? 0.8f : 0.0f;
    ctx->npu_min_gflop = have_vk ? 4.0 : 0.0;
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_MIN_GFLOP")) {
        ctx->npu_min_gflop = std::max(0.0, std::atof(env));
    }

    if (const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_SHARE")) {
        if (std::strcmp(env, "auto") == 0) {
            ctx->share_fixed = false;
        } else {
            npu = (float) std::min(1.0, std::max(0.0, std::atof(env)));
        }
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_SHARE")) {
        vkf = have_vk ? (float) std::min(1.0, std::max(0.0, std::atof(env))) : 0.0f;
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_ZERO_COPY")) {
        ctx->vk_zero_copy = std::atoi(env) != 0;
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_IMPORT_ALIGN")) {
        ctx->vk_import_align = (uintptr_t) std::max(4096LL, std::atoll(env));
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_CACHE_MB")) {
        ctx->vk_weight_limit = (size_t) std::max(0LL, std::atoll(env)) << 20;
    }
    if (npu + vkf > 1.0f) {
        vkf = 1.0f - npu;
    }
    float cpu = have_cpu ? 1.0f - npu - vkf : 0.0f;
    if (!have_cpu) {
        // no CPU worker: its share goes to the GPU if present, else to the NPU
        if (have_vk) { vkf = 1.0f - npu; } else { npu = 1.0f; }
    }

    ctx->share[GGML_XDNA_WORKER_NPU] = npu;
    ctx->share[GGML_XDNA_WORKER_VK]  = vkf;
    ctx->share[GGML_XDNA_WORKER_CPU] = cpu;
    GGML_LOG_INFO("%s: matmul split NPU %.2f / Vulkan %.2f / CPU %.2f%s\n", __func__, npu, vkf, cpu,
                  ctx->share_fixed ? "" : " (auto)");
}

static ggml_backend_t ggml_xdna_cpu_backend(ggml_backend_xdna_context * ctx) {
#ifdef GGML_XDNA_HAVE_CPU
    if (ctx->cpu == nullptr) {
        ctx->cpu = ggml_backend_cpu_init();
        if (ctx->cpu == nullptr) {
            return nullptr;
        }
    }
    ggml_backend_cpu_set_n_threads(ctx->cpu, ctx->n_threads);
    return ctx->cpu;
#else
    GGML_UNUSED(ctx);
    return nullptr;
#endif
}

// compute dst rows [feat0, feat1) of a MUL_MAT on the CPU backend, as a tiny standalone graph
// built over views of the operands (no allocation, no copies)
static bool ggml_xdna_mul_mat_cpu(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    ggml_backend_t cpu = ggml_xdna_cpu_backend(ctx);
    if (cpu == nullptr) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_4d(gctx, src0->type, src0->ne[0], feat1 - feat0, src0->ne[2], src0->ne[3]);
    a->data = (char *) src0->data + feat0*src0->nb[1];
    std::memcpy(a->nb, src0->nb, sizeof(a->nb));

    ggml_tensor * b = ggml_new_tensor_4d(gctx, src1->type, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    b->data = src1->data;
    std::memcpy(b->nb, src1->nb, sizeof(b->nb));

    ggml_tensor * c = ggml_mul_mat(gctx, a, b);
    c->data  = (char *) dst->data + feat0*dst->nb[0];
    c->nb[1] = dst->nb[1];
    c->nb[2] = dst->nb[2];
    c->nb[3] = dst->nb[3];

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, c);
    const enum ggml_status status = ggml_backend_graph_compute(cpu, graph);

    ggml_free(gctx);
    return status == GGML_STATUS_SUCCESS;
}

template <typename F>
static void ggml_xdna_parallel_for(ggml_backend_xdna_context * ctx, int64_t n, int64_t min_per_thread, F && fn) {
    int n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(ctx->n_threads, n/min_per_thread));
    if (n_threads <= 1) {
        fn(0, n);
        return;
    }
    for (int i = 1; i < n_threads; i++) {
        const int64_t start = i*n/n_threads;
        const int64_t end   = (i + 1)*n/n_threads;
        if (start < end) {
            ctx->tasks.push_back(std::async(std::launch::async, [=, &fn]() { fn(start, end); }));
        }
    }
    fn(0, n/n_threads);
    for (auto & t : ctx->tasks) {
        t.get();
    }
    ctx->tasks.clear();
}

// convert rows [row0, row0 + n_rows) of plane (i2, i3) of t to bf16, one row per dst_stride elements
static void ggml_xdna_rows_to_bf16(ggml_backend_xdna_context * ctx, const ggml_tensor * t,
                                   int64_t row0, int64_t n_rows, int64_t i2, int64_t i3,
                                   ggml_bf16_t * dst, int64_t dst_stride) {
    const int64_t ne0 = t->ne[0];
    const enum ggml_type type = t->type;
    const ggml_to_float_t to_float = ggml_get_type_traits(type)->to_float;

    ggml_xdna_parallel_for(ctx, n_rows, 8, [&](int64_t r0, int64_t r1) {
        std::vector<float> tmp;
        for (int64_t r = r0; r < r1; r++) {
            const char  * src = (const char *) t->data + (row0 + r)*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
            ggml_bf16_t * out = dst + r*dst_stride;
            switch (type) {
                case GGML_TYPE_BF16:
                    std::memcpy(out, src, ne0*sizeof(ggml_bf16_t));
                    break;
                case GGML_TYPE_F32:
                    ggml_fp32_to_bf16_row((const float *) src, out, ne0);
                    break;
                default:
                    tmp.resize(ne0);
                    if (type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, tmp.data(), ne0);
                    } else {
                        to_float(src, tmp.data(), ne0);
                    }
                    ggml_fp32_to_bf16_row(tmp.data(), out, ne0);
                    break;
            }
        }
    });
}

// copy an [n_rows x n_cols] window of a bf16 row-major matrix into a zero-padded [rows x cols] block
static void ggml_xdna_fill_block(ggml_bf16_t * blk, int64_t rows, int64_t cols,
                                 const ggml_bf16_t * src, int64_t src_stride, int64_t n_rows, int64_t n_cols) {
    if (n_rows < rows || n_cols < cols) {
        std::memset(blk, 0, rows*cols*sizeof(ggml_bf16_t));
    }
    for (int64_t r = 0; r < n_rows; r++) {
        std::memcpy(blk + r*cols, src + r*src_stride, n_cols*sizeof(ggml_bf16_t));
    }
}

// last-resort host GEMM on the staged bf16 blocks, so a runtime NPU failure never corrupts results
static void ggml_xdna_block_gemm_host(const ggml_xdna_kernel & k, int64_t n_rows_a, int64_t n_rows_b) {
    for (int64_t t = 0; t < n_rows_b; t++) {
        for (int64_t r = 0; r < n_rows_a; r++) {
            float sum = 0.0f;
            const ggml_bf16_t * a = k.a_map + r*k.K;
            const ggml_bf16_t * b = k.b_map + t*k.K;
            for (int64_t i = 0; i < k.K; i++) {
                sum += ggml_bf16_to_fp32(a[i]) * ggml_bf16_to_fp32(b[i]);
            }
            k.c_map[t*k.M + r] = sum;
        }
    }
}

// compute dst rows [feat0, feat1) of a MUL_MAT on the NPU (host fallback if it fails)
static void ggml_xdna_mul_mat_npu(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    ggml_xdna_state & st = ggml_xdna_get_state();

    const int64_t n_k    = ne00;           // reduction length
    const int64_t n_feat = feat1 - feat0;  // output features handled here (weight rows)
    const int64_t n_tok  = ne11;           // tokens (activation rows)

    ggml_xdna_kernel * kp = ggml_xdna_select_kernel(st, n_feat, n_k, n_tok);
    GGML_ASSERT(kp != nullptr && "supports_op admitted a MUL_MAT but no kernel is available");
    ggml_xdna_kernel & k = *kp;

    bool use_npu = false;
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        use_npu = ggml_xdna_kernel_load(st, k);
    }
    if (!use_npu && !ctx->warned_fallback) {
        GGML_LOG_WARN("%s: NPU kernel unavailable, computing on the host instead\n", __func__);
        ctx->warned_fallback = true;
    }

    const int64_t kM = k.M, kK = k.K, kN = k.N;

    if ((int64_t) ctx->w_bf16.size() < kM*n_k)    { ctx->w_bf16.resize(kM*n_k); }
    if ((int64_t) ctx->x_bf16.size() < n_tok*n_k) { ctx->x_bf16.resize(n_tok*n_k); }

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            ggml_xdna_rows_to_bf16(ctx, src1, 0, n_tok, i12, i13, ctx->x_bf16.data(), n_k);

            char * d_plane = (char *) dst->data + i12*nb2 + i13*nb3 + feat0*nb0;

            for (int64_t i0 = 0; i0 < n_feat; i0 += kM) {
                const int64_t m_rows = std::min(kM, n_feat - i0);
                ggml_xdna_rows_to_bf16(ctx, src0, feat0 + i0, m_rows, i02, i03, ctx->w_bf16.data(), n_k);

                for (int64_t j0 = 0; j0 < n_tok; j0 += kN) {
                    const int64_t n_rows = std::min(kN, n_tok - j0);

                    for (int64_t k0 = 0; k0 < n_k; k0 += kK) {
                        const int64_t kk = std::min(kK, n_k - k0);

                        ggml_xdna_fill_block(k.a_map, kM, kK, ctx->w_bf16.data() + k0,          n_k, m_rows, kk);
                        ggml_xdna_fill_block(k.b_map, kN, kK, ctx->x_bf16.data() + j0*n_k + k0, n_k, n_rows, kk);

                        // the NPU serializes work per hw_context; take the state lock for the dispatch
                        {
                            std::lock_guard<std::mutex> lock(st.mutex);
                            if (!use_npu || !ggml_xdna_kernel_run(k)) {
                                if (use_npu) {
                                    GGML_LOG_WARN("%s: NPU dispatch failed, finishing this op on the host\n", __func__);
                                    use_npu = false;
                                }
                                ggml_xdna_block_gemm_host(k, m_rows, n_rows);
                            }
                        }

                        // C block is [kN x kM]: row t = token, col r = feature
                        for (int64_t t = 0; t < n_rows; t++) {
                            float       * drow = (float *) (d_plane + (j0 + t)*nb1) + i0;
                            const float * crow = k.c_map + t*kM;
                            if (k0 == 0) {
                                std::memcpy(drow, crow, m_rows*sizeof(float));
                            } else {
                                for (int64_t r = 0; r < m_rows; r++) {
                                    drow[r] += crow[r];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void ggml_backend_xdna_mul_mat(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int64_t n_feat = ne01;

    ggml_xdna_init_shares(ctx);

    // split the output features into consecutive ranges: NPU, Vulkan, CPU. the NPU part is
    // rounded to whole kernel blocks so no block is padded needlessly.
    ggml_xdna_state & st = ggml_xdna_get_state();

    // the NPU part is rounded to the smallest kernel block so small shares stay possible, and
    // an op too small to amortize the NPU's per-launch cost goes to the other workers only
    int64_t kR = 256;
    for (const auto & k : st.kernels) {
        if (!k.broken) {
            kR = std::min(kR, k.M);
        }
    }
    const double gflop = 2.0*n_feat*ne00*ne11/1e9;
    const bool npu_worthwhile = gflop >= ctx->npu_min_gflop || (ctx->vk == nullptr && ctx->cpu == nullptr);

    int64_t n[GGML_XDNA_N_WORKERS] = { 0, 0, 0 };

    if (ctx->share[GGML_XDNA_WORKER_NPU] >= 1.0f || (ctx->vk == nullptr && ctx->cpu == nullptr)) {
        n[GGML_XDNA_WORKER_NPU] = n_feat;
    } else if (ctx->share[GGML_XDNA_WORKER_NPU] > 0.0f && npu_worthwhile) {
        n[GGML_XDNA_WORKER_NPU] = (int64_t) (n_feat*ctx->share[GGML_XDNA_WORKER_NPU] + kR/2) / kR * kR;
        n[GGML_XDNA_WORKER_NPU] = std::min(n[GGML_XDNA_WORKER_NPU], n_feat);
        if (n_feat - n[GGML_XDNA_WORKER_NPU] < 64) {
            n[GGML_XDNA_WORKER_NPU] = n_feat;
        }
    }
    if (ctx->vk && ctx->share[GGML_XDNA_WORKER_VK] > 0.0f) {
        n[GGML_XDNA_WORKER_VK] = (int64_t) (n_feat*ctx->share[GGML_XDNA_WORKER_VK] + 32) / 64 * 64;
        n[GGML_XDNA_WORKER_VK] = std::min(n[GGML_XDNA_WORKER_VK], n_feat - n[GGML_XDNA_WORKER_NPU]);
    }
    n[GGML_XDNA_WORKER_CPU] = n_feat - n[GGML_XDNA_WORKER_NPU] - n[GGML_XDNA_WORKER_VK];
    if (n[GGML_XDNA_WORKER_CPU] > 0 && (ctx->cpu == nullptr || n[GGML_XDNA_WORKER_CPU] < 16)) {
        // no CPU worker, or a sliver not worth a thread pool: hand it to the GPU, else the NPU
        n[ctx->vk && n[GGML_XDNA_WORKER_VK] > 0 ? GGML_XDNA_WORKER_VK : GGML_XDNA_WORKER_NPU] += n[GGML_XDNA_WORKER_CPU];
        n[GGML_XDNA_WORKER_CPU] = 0;
    }

    const int64_t f_npu0 = 0,                            f_npu1 = f_npu0 + n[GGML_XDNA_WORKER_NPU];
    const int64_t f_vk0  = f_npu1,                       f_vk1  = f_vk0  + n[GGML_XDNA_WORKER_VK];
    const int64_t f_cpu0 = f_vk1,                        f_cpu1 = n_feat;

    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    double t[GGML_XDNA_N_WORKERS] = { 0.0, 0.0, 0.0 };
    bool   vk_ok = true, cpu_ok = true;

    std::thread npu_thread, vk_thread;
    if (n[GGML_XDNA_WORKER_NPU] > 0) {
        npu_thread = std::thread([&]() {
            ggml_xdna_mul_mat_npu(ctx, dst, f_npu0, f_npu1);
            t[GGML_XDNA_WORKER_NPU] = elapsed();
        });
    }
    if (n[GGML_XDNA_WORKER_VK] > 0) {
        vk_thread = std::thread([&]() {
            vk_ok = ggml_xdna_mul_mat_vk(ctx, dst, f_vk0, f_vk1);
            t[GGML_XDNA_WORKER_VK] = elapsed();
        });
    }
    if (n[GGML_XDNA_WORKER_CPU] > 0) {
        cpu_ok = ggml_xdna_mul_mat_cpu(ctx, dst, f_cpu0, f_cpu1);
        t[GGML_XDNA_WORKER_CPU] = elapsed();
    }
    if (npu_thread.joinable()) { npu_thread.join(); }
    if (vk_thread.joinable())  { vk_thread.join();  }

    // a worker that failed leaves its rows to the NPU (always available once we got here)
    if (!vk_ok) {
        GGML_LOG_WARN("%s: Vulkan worker failed, finishing its rows on the NPU\n", __func__);
        ggml_xdna_mul_mat_npu(ctx, dst, f_vk0, f_vk1);
        ctx->share[GGML_XDNA_WORKER_NPU] += ctx->share[GGML_XDNA_WORKER_VK];
        ctx->share[GGML_XDNA_WORKER_VK]   = 0.0f;
    }
    if (!cpu_ok) {
        ggml_xdna_mul_mat_npu(ctx, dst, f_cpu0, f_cpu1);
        ctx->share[GGML_XDNA_WORKER_NPU] += ctx->share[GGML_XDNA_WORKER_CPU];
        ctx->share[GGML_XDNA_WORKER_CPU]  = 0.0f;
    }

    if (!ctx->share_fixed && vk_ok && cpu_ok) {
        // rebalance towards equal finishing times: each worker's rate is rows/time
        double rate[GGML_XDNA_N_WORKERS] = { 0.0, 0.0, 0.0 };
        double sum = 0.0;
        for (int i = 0; i < GGML_XDNA_N_WORKERS; i++) {
            if (n[i] > 0 && t[i] > 0.0) {
                rate[i] = (double) n[i]/t[i];
                sum += rate[i];
            }
        }
        if (sum > 0.0) {
            float total = 0.0f;
            for (int i = 0; i < GGML_XDNA_N_WORKERS; i++) {
                if (n[i] > 0) {
                    ctx->share[i] = (float) std::max(0.05, 0.85*ctx->share[i] + 0.15*rate[i]/sum);
                }
                total += ctx->share[i];
            }
            for (int i = 0; i < GGML_XDNA_N_WORKERS; i++) {
                ctx->share[i] /= total;
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return "XDNA";

    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;
    if (ctx->cpu) {
        ggml_backend_free(ctx->cpu);
    }
    for (auto & kv : ctx->vk_imports) {
        ggml_backend_buffer_free(kv.second);
    }
    for (auto & kv : ctx->vk_weights) {
        ggml_backend_buffer_free(kv.second.buf);
        ggml_free(kv.second.wctx);
    }
    if (ctx->vk) {
        ggml_backend_free(ctx->vk);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_xdna_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i xdna_backend_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x7c, 0x0f, 0x3a, 0x91, 0x5e, 0xd2, 0x4b, 0x86, 0xa4, 0x19, 0xc7, 0x62, 0x0e, 0xb5, 0x88, 0x31 };
    return &guid;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_xdna_probe(ggml_xdna_get_state());

    ggml_backend_xdna_context * ctx = new ggml_backend_xdna_context;   // shares resolved on first use

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

void ggml_backend_xdna_set_n_threads(ggml_backend_t backend_xdna, int n_threads) {
    GGML_ASSERT(ggml_backend_is_xdna(backend_xdna));

    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend_xdna->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return "XDNA";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    ggml_xdna_state & st = ggml_xdna_get_state();
    ggml_xdna_probe(st);
    return st.description.c_str();

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // operands live in host memory
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xdna_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            ggml_xdna_state & st = ggml_xdna_get_state();
            ggml_xdna_probe(st);
            if (!st.available || st.kernels.empty()) {
                return false;
            }

            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            const int64_t n_k    = src0->ne[0];
            const int64_t n_feat = src0->ne[1];
            const int64_t n_tok  = src1->ne[1];

            // per-launch overhead makes small products a loss; decode-sized (n_tok == 1) never qualifies
            const int64_t min_dim = 32;

            const bool ok = ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   op->type   == GGML_TYPE_F32 &&
                   (n_tok >= st.min_batch && n_k >= min_dim && n_feat >= min_dim) &&
                   (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                    ggml_get_type_traits(src0->type)->to_float != NULL);

            const bool debug = ggml_xdna_debug();
            if (debug) {
                GGML_LOG_INFO("%s: MUL_MAT %s[%" PRId64 ",%" PRId64 "] x %s[%" PRId64 ",%" PRId64 "] -> %s cont=%d/%d min_batch=%" PRId64 " : %s\n",
                              __func__, ggml_type_name(src0->type), n_k, n_feat, ggml_type_name(src1->type), src1->ne[0], n_tok,
                              ggml_type_name(op->type), ggml_is_contiguous(src0), ggml_is_contiguous(src1), st.min_batch,
                              ok ? "yes" : "no");
            }
            return ok;
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return "XDNA";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_xdna_state & st = ggml_xdna_get_state();
    ggml_xdna_probe(st);
    return st.available ? 1 : 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xdna_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_xdna_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
