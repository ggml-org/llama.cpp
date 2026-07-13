// libppu_moe.so — torch-free C ABI shim over DeepGEMM's bf16 grouped-GEMM (contiguous / m_indices layout).
//
// Goal: the shipped .so must NOT require libtorch at the user's runtime. DeepGEMM's public entry
// (deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous) is torch-native, but its compute path is only shallowly
// torch-bound: the impl unpacks tensors via .data_ptr()/.stride()/.scalar_type(), and LaunchRuntime::launch() grabs
// at::cuda::getCurrentCUDAStream(). Everything else (get_best_config, the cutlass3 codegen in
// SM90BF16GemmRuntime::generate, Compiler::build + cubin load, construct_launch_config, launch_impl) takes raw
// pointers / cudaStream_t / CUfunction and is torch-free.
//
// So instead of calling the torch entry, we RE-IMPLEMENT the ~40-line sm90 contiguous impl body here on raw pointers
// (verified against csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) with:
//   * raw TMA descriptor builders (make_tma_2d_desc takes a torch::Tensor purely for data_ptr/element_size/dtype;
//     ours hardcodes bf16 and takes the pointer), and
//   * a launch that uses the CALLER's stream instead of getCurrentCUDAStream().
// We still #include DeepGEMM's headers (which pull torch headers at COMPILE time) but call no torch function; built
// with -ffunction-sections + --gc-sections so the unreferenced torch-using helpers are stripped.
// Verify on the box:  ldd libppu_moe.so | grep -i torch   -> expect EMPTY.
//
// Arch: DeepGEMM's bf16 GEMM exists only for SM90 (wgmma+TMA) and SM100 (tcgen05). We wire the SM90 path; on any
// other arch we return non-zero and llama.cpp falls back to its inline MoE path.
//
// Runtime prereqs: CUDA driver + nvcc (or DG_JIT_USE_NVRTC=1 + nvrtc) for the first JIT compile of each shape; the
// cubin is then disk-cached under DG_JIT_CACHE_DIR (default $HOME/.deep_gemm), so later runs only cuModuleLoad it.
// Env: CUDA_HOME, DG_LIBRARY_ROOT (=.../DeepGEMM/deep_gemm, holds include/deep_gemm/*), DG_JIT_CACHE_DIR. See README.
//
// ABI mapping: A=lhs[total_rows,K] bf16, B=rhs[n_experts,N,K] bf16, out[total_rows,N] bf16, m_indices[total_rows] i32.

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <stdexcept>

#include "jit_kernels/impls/sm90_bf16_gemm.hpp"   // SM90BF16GemmRuntime, runtime_utils (make_tma_2d_desc, ...)
#include "jit/compiler.hpp"                        // deep_gemm::compiler (LazyInit), Compiler::prepare_init
#include "jit/kernel_runtime.hpp"                  // KernelRuntime::prepare_init, LaunchArgs
#include "jit/handle.hpp"                          // construct_launch_config
#include "jit/include_parser.hpp"                  // IncludeParser::prepare_init
#include "jit/device_runtime.hpp"                  // device_runtime

// The only libtorch symbol left after -DDG_NO_TORCH: an inline TORCH_CHECK in a torch header bottoms out here.
// Define it ourselves (as c10 does: [[noreturn]], throws) so the .so links against no libtorch at all. Verify:
// `nm -D -u libppu_moe.so | c++filt | grep -iE 'c10::|at::|torch'` must come back empty.
namespace c10::detail {
void torchCheckFail(const char * func, const char * file, uint32_t line, const char * msg) {
    char buf[512];
    snprintf(buf, sizeof(buf), "[ppu-moe] TORCH_CHECK failed at %s:%u in %s: %s",
             file ? file : "?", line, func ? func : "?", msg ? msg : "");
    throw std::runtime_error(buf);
}
} // namespace c10::detail

using namespace deep_gemm;

namespace {

const char * env_or(const char * k, const char * dflt) {
    const char * v = getenv(k);
    return (v && v[0]) ? v : dflt;
}

// One-time JIT init. The python package does this at import via deep_gemm::runtime::init (csrc/apis/runtime.hpp:44);
// we're C++-only, so we call the same three prepare_init's ourselves. Missing IncludeParser::prepare_init leaves its
// library_include_path empty and every JIT include hashes as "Failed to open: deep_gemm/impls/...".
void ensure_jit_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        const std::string cuda_home = env_or("CUDA_HOME", "/usr/local/cuda");
        const std::string lib_root  = env_or("DG_LIBRARY_ROOT", "");   // .../DeepGEMM/deep_gemm (has include/deep_gemm)
        Compiler::prepare_init(lib_root, cuda_home);
        KernelRuntime::prepare_init(cuda_home);
        IncludeParser::prepare_init(lib_root);
    });
}

// Raw-pointer twin of runtime_utils.hpp's make_tma_2d_desc, specialised to bf16 (elem_size 2). Same encode call,
// same swizzle handling; only the torch::Tensor accessors (data_ptr / element_size / scalar_type) are replaced.
CUtensorMap make_tma_2d_desc_bf16(void * ptr,
                                  int gmem_inner_dim, int gmem_outer_dim,
                                  int smem_inner_dim, int smem_outer_dim,
                                  int64_t gmem_outer_stride, int swizzle_mode) {
    constexpr int elem_size = 2;
    if (swizzle_mode != 0) {
        smem_inner_dim = swizzle_mode / elem_size;
    }
    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2]    = {(cuuint64_t) gmem_inner_dim, (cuuint64_t) gmem_outer_dim};
    const cuuint32_t smem_dims[2]    = {(cuuint32_t) smem_inner_dim, (cuuint32_t) smem_outer_dim};
    const cuuint64_t gmem_strides[1] = {(cuuint64_t) (gmem_outer_stride * elem_size)};
    const cuuint32_t elem_strides[2] = {1, 1};
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, 0),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

} // namespace

// Required per-expert row alignment of the compact "contiguous" layout. For a grouped-contiguous GEMM DeepGEMM
// pins BLOCK_M to exactly this value (heuristics/sm90.hpp:33) and the kernel reads ONE expert id per BLOCK_M row
// block, from that block's first row. So each expert's row segment must start (and therefore end) on a multiple of
// it -- otherwise a block straddles two experts and is silently computed against the wrong weights. Callers must
// pad; the padding rows' m_indices should repeat their block's expert id.
extern "C" int ppu_moe_row_alignment(void) {
    return heuristics_runtime->get_mk_alignment_for_contiguous_layout();
}

extern "C" int ppu_moe_grouped_gemm_bf16_nopad(
        const void * A, const void * B, void * out, const int * m_indices,
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream) {
    (void) expected_m;
    if (total_rows < 0 || N <= 0 || K <= 0 || n_experts <= 0) return 1;
    if (total_rows == 0) return 0;                  // nothing to do
    if (K % 64 != 0) return 1;                      // DG_HOST_ASSERT(k % 64 == 0) in the impl

    try {
        ensure_jit_init();

        // DeepGEMM's bf16 GEMM is SM90 (wgmma) / SM100 (tcgen05) only. We wire SM90.
        if (device_runtime->get_arch_major() != 9) return 3;

        // Reject an unpadded row count rather than compute silent garbage (see ppu_moe_row_alignment above).
        // This only catches a wrong TOTAL; per-expert segment alignment is the caller's contract.
        if (total_rows % ppu_moe_row_alignment() != 0) return 1;

        const int m = total_rows, n = N, k = K, num_groups = n_experts;
        const auto major_a = cute::UMMA::Major::K;   // A [m,k], k-contiguous
        const auto major_b = cute::UMMA::Major::K;   // B [G,n,k], k-contiguous  (the "nt" in the entry name)
        const std::string compiled_dims = "nk";

        // --- torch-free re-implementation of sm90_m_grouped_bf16_gemm_contiguous (use_psum_layout = false) ---
        const auto desc = GemmDesc {
            .gemm_type = GemmType::MGroupedContiguous,
            .kernel_type = KernelType::KernelNoSF,
            .m = m, .n = n, .k = k, .num_groups = num_groups,
            .a_dtype = torch::kBFloat16, .b_dtype = torch::kBFloat16,
            .cd_dtype = torch::kBFloat16,
            .major_a = major_a, .major_b = major_b,
            .with_accumulation = false,
            .num_sms = device_runtime->get_num_sms(),
            .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
            .expected_m = m,
            .expected_n = n, .expected_k = k,
            .expected_num_groups = 1
        };
        const auto config = get_best_config<SM90ArchSpec>(desc);

        // TMA descriptors. Strides are the "outer" (non-contiguous) stride in elements:
        //   A [m,k]   -> stride(-2) = k        B [G,n,k] -> stride(-2) = k        D [m,n] -> stride(-2) = n
        const auto tensor_map_a = make_tma_2d_desc_bf16(const_cast<void *>(A),
            /*gmem*/ k, m, /*smem*/ config.layout.block_k, config.storage_config.load_block_m,
            /*outer_stride*/ k, config.storage_config.swizzle_a_mode);
        const auto tensor_map_b = make_tma_2d_desc_bf16(const_cast<void *>(B),
            /*gmem*/ k, n * num_groups, /*smem*/ config.layout.block_k, config.storage_config.load_block_n,
            /*outer_stride*/ k, config.storage_config.swizzle_b_mode);
        const auto tensor_map_cd = make_tma_2d_desc_bf16(out,
            /*gmem*/ n, m, /*smem*/ config.storage_config.store_block_n, config.storage_config.store_block_m,
            /*outer_stride*/ n, config.storage_config.swizzle_cd_mode);

        const SM90BF16GemmRuntime::Args args = {
            .gemm_desc = desc,
            .gemm_config = config,
            .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                      config.pipeline_config.smem_size,
                                      config.layout.get_cluster_size()),
            .grouped_layout = const_cast<int *>(m_indices),
            .tensor_map_a = tensor_map_a,
            .tensor_map_b = tensor_map_b,
            .tensor_map_cd = tensor_map_cd,
        };

        const auto code    = SM90BF16GemmRuntime::generate(args);
        const auto runtime = compiler->build("sm90_m_grouped_bf16_gemm_contiguous", code);
        const auto kernel  = runtime->kernel;

        // The one substitution vs upstream LaunchRuntime::launch(): launch on the CALLER's stream (upstream grabs
        // at::cuda::getCurrentCUDAStream()). construct_launch_config + launch_impl are both torch-free.
        LaunchArgs launch_args = args.launch_args;
        launch_args.enable_pdl = device_runtime->get_pdl();
        const dim3 grid_dim  = {(unsigned) launch_args.grid_dim.first, (unsigned) launch_args.grid_dim.second, 1};
        const dim3 block_dim = {(unsigned) launch_args.num_threads, 1, 1};
        const auto launch_cfg = construct_launch_config(kernel, reinterpret_cast<cudaStream_t>(stream),
                                                        launch_args.smem_size, grid_dim, block_dim,
                                                        launch_args.cluster_dim, launch_args.enable_pdl);
        SM90BF16GemmRuntime::launch_impl(kernel, launch_cfg, args);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "[ppu-moe] grouped-gemm (m=%d N=%d K=%d G=%d) failed: %s -> inline fallback\n",
                total_rows, N, K, n_experts, e.what());
        return 2;
    }
}
