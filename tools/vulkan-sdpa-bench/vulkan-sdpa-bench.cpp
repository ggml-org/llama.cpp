#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
    int q        = 512;
    int kv       = 40960;
    int heads    = 16;
    int kv_heads = 4;
    int d        = 256;
    int warmup   = 3;
    int iters    = 50;

    std::string device = "Vulkan0";
    bool list_devices  = false;
};

static void usage() {
    std::cout <<
        "vulkan-sdpa-bench options:\n"
        "  --q N             query tokens       (default 512)\n"
        "  --kv N            KV tokens          (default 40960)\n"
        "  --heads N         query heads        (default 16)\n"
        "  --kv-heads N      KV heads           (default 4)\n"
        "  --head-dim N      head dimension     (default 256)\n"
        "  --warmup N        warmup runs        (default 3)\n"
        "  --iters N         timed runs         (default 50)\n"
        "  --device NAME     backend device     (default Vulkan0)\n"
        "  --list-devices    list devices and exit\n"
        "\n"
        "Fixed benchmark dtypes:\n"
        "  Q    = F32\n"
        "  K/V  = F16\n"
        "  mask = F16\n"
        "  acc  = F32\n"
        "\n"
        "No coopmat override or tuning is applied.\n";
}

static Args parse_args(int argc, char ** argv) {
    Args a;

    auto value = [&](int & i) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(
                std::string("missing value for ") + argv[i]);
        }
        return argv[++i];
    };

    auto integer = [&](int & i) {
        return std::stoi(value(i));
    };

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];

        if      (s == "--q")            a.q        = integer(i);
        else if (s == "--kv")           a.kv       = integer(i);
        else if (s == "--heads")        a.heads    = integer(i);
        else if (s == "--kv-heads")     a.kv_heads = integer(i);
        else if (s == "--head-dim")     a.d        = integer(i);
        else if (s == "--warmup")       a.warmup   = integer(i);
        else if (s == "--iters")        a.iters    = integer(i);
        else if (s == "--device")       a.device   = value(i);
        else if (s == "--list-devices") a.list_devices = true;
        else if (s == "-h" || s == "--help") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + s);
        }
    }

    if (a.q <= 0 ||
        a.kv <= 0 ||
        a.heads <= 0 ||
        a.kv_heads <= 0 ||
        a.d <= 0 ||
        a.warmup < 0 ||
        a.iters <= 0) {
        throw std::runtime_error("invalid numeric argument");
    }

    if (a.heads % a.kv_heads != 0) {
        throw std::runtime_error(
            "heads must be divisible by kv-heads");
    }

    return a;
}

static double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());

    const double pos =
        p * static_cast<double>(values.size() - 1);

    const size_t idx =
        static_cast<size_t>(std::ceil(pos));

    return values[idx];
}

static void list_devices() {
    const size_t n = ggml_backend_dev_count();

    std::cout << "Available devices:\n";

    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev =
            ggml_backend_dev_get(i);

        size_t free_mem  = 0;
        size_t total_mem = 0;

        ggml_backend_dev_memory(
            dev,
            &free_mem,
            &total_mem
        );

        std::cout
            << "  "
            << ggml_backend_dev_name(dev)
            << " : "
            << ggml_backend_dev_description(dev)
            << "  total="
            << std::fixed
            << std::setprecision(1)
            << total_mem / 1048576.0
            << " MiB"
            << "  free="
            << free_mem / 1048576.0
            << " MiB\n";
    }
}

static void zero_tensor(
    ggml_tensor * t,
    std::vector<uint8_t> & zeros) {

    const size_t n = ggml_nbytes(t);

    if (zeros.size() < n) {
        zeros.resize(n, 0);
    }

    ggml_backend_tensor_set(
        t,
        zeros.data(),
        0,
        n
    );
}

int main(int argc, char ** argv) try {
    const Args a = parse_args(argc, argv);

    //
    // Load registered backend DLLs. We explicitly select Vulkan0 below;
    // no SYCL operation is submitted.
    //
    ggml_backend_load_all();

    if (a.list_devices) {
        list_devices();
        return 0;
    }

    ggml_backend_dev_t dev =
        ggml_backend_dev_by_name(
            a.device.c_str()
        );

    if (!dev) {
        std::cerr
            << "ERROR: device not found: "
            << a.device
            << "\n\n";

        list_devices();
        return 2;
    }

    ggml_backend_t backend =
        ggml_backend_dev_init(
            dev,
            nullptr
        );

    if (!backend) {
        throw std::runtime_error(
            "ggml_backend_dev_init failed");
    }

    size_t free_before  = 0;
    size_t total_memory = 0;

    ggml_backend_dev_memory(
        dev,
        &free_before,
        &total_memory
    );

    //
    // Enough metadata space for our tiny graph.
    // Tensor payloads themselves are allocated by the backend.
    //
    constexpr size_t GRAPH_NODES = 64;
    constexpr size_t META_TENSORS = 16;

    ggml_init_params params = {
        /* .mem_size = */
        ggml_tensor_overhead() * META_TENSORS +
        ggml_graph_overhead_custom(
            GRAPH_NODES,
            false
        ),

        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx =
        ggml_init(params);

    if (!ctx) {
        ggml_backend_free(backend);
        throw std::runtime_error(
            "ggml_init failed");
    }

    //
    // llama.cpp FLASH_ATTN_EXT layout:
    //
    // Q    [D, Q,  H,   batch]
    // K/V  [D, KV, Hkv, batch]
    // mask [KV,Q,  1,   batch]
    //
    // This gives H=16 / Hkv=4 => GQA ratio 4.
    //
    ggml_tensor * q =
        ggml_new_tensor_4d(
            ctx,
            GGML_TYPE_F32,
            a.d,
            a.q,
            a.heads,
            1
        );

    ggml_tensor * k =
        ggml_new_tensor_4d(
            ctx,
            GGML_TYPE_F16,
            a.d,
            a.kv,
            a.kv_heads,
            1
        );

    ggml_tensor * v =
        ggml_new_tensor_4d(
            ctx,
            GGML_TYPE_F16,
            a.d,
            a.kv,
            a.kv_heads,
            1
        );

    ggml_tensor * mask =
        ggml_new_tensor_4d(
            ctx,
            GGML_TYPE_F16,
            a.kv,
            a.q,
            1,
            1
        );

    ggml_set_name(q,    "q");
    ggml_set_name(k,    "k");
    ggml_set_name(v,    "v");
    ggml_set_name(mask, "mask");

    const float scale =
        1.0f / std::sqrt(
            static_cast<float>(a.d)
        );

    ggml_tensor * out =
        ggml_flash_attn_ext(
            ctx,
            q,
            k,
            v,
            mask,
            scale,
            0.0f,   // max_bias
            0.0f    // logit_softcap
        );

    ggml_prec_set_acc(
        out,
        GGML_PREC_F32
    );

    ggml_set_name(out, "out");

    if (!ggml_backend_dev_supports_op(
            dev,
            out)) {
        ggml_free(ctx);
        ggml_backend_free(backend);

        throw std::runtime_error(
            "Vulkan device does not support "
            "this FLASH_ATTN_EXT shape");
    }

    //
    // Allocate all Q/K/V/mask/output tensors directly
    // in the selected Vulkan backend buffer.
    //
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(
            ctx,
            backend
        );

    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);

        throw std::runtime_error(
            "backend tensor allocation failed");
    }

    //
    // Dense zero mask deliberately matches the standalone
    // oneDNN benchmark: no causal -INF block skipping.
    //
    std::vector<uint8_t> zeros;

    zero_tensor(q,    zeros);
    zero_tensor(k,    zeros);
    zero_tensor(v,    zeros);
    zero_tensor(mask, zeros);

    ggml_backend_synchronize(backend);

    //
    // Build a graph containing only FLASH_ATTN_EXT.
    //
    ggml_cgraph * graph =
        ggml_new_graph_custom(
            ctx,
            GRAPH_NODES,
            false
        );

    ggml_build_forward_expand(
        graph,
        out
    );

    size_t free_after_alloc = 0;
    size_t total_after      = 0;

    ggml_backend_dev_memory(
        dev,
        &free_after_alloc,
        &total_after
    );

    const size_t explicit_bytes =
        ggml_nbytes(q) +
        ggml_nbytes(k) +
        ggml_nbytes(v) +
        ggml_nbytes(mask) +
        ggml_nbytes(out);

    std::cout
        << "============================================================\n"
        << "Vulkan standalone GQA SDPA benchmark\n";

    std::cout
        << "device           : "
        << ggml_backend_dev_name(dev)
        << "\n";

    std::cout
        << "GPU              : "
        << ggml_backend_dev_description(dev)
        << "\n";

    std::cout
        << "shape            : Q="
        << a.q
        << " KV="
        << a.kv
        << " H="
        << a.heads
        << " Hkv="
        << a.kv_heads
        << " D="
        << a.d
        << "\n";

    std::cout
        << "GQA ratio        : "
        << (a.heads / a.kv_heads)
        << "\n";

    std::cout
        << "Q dtype          : "
        << ggml_type_name(q->type)
        << "\n";

    std::cout
        << "K/V dtype        : "
        << ggml_type_name(k->type)
        << "/"
        << ggml_type_name(v->type)
        << "\n";

    std::cout
        << "mask dtype       : "
        << ggml_type_name(mask->type)
        << "\n";

    std::cout
        << "output dtype     : "
        << ggml_type_name(out->type)
        << "\n";

    std::cout
        << "accumulation     : F32\n";

    std::cout
        << "graph nodes      : "
        << ggml_graph_n_nodes(graph)
        << "\n";

    std::cout
        << "device memory    : "
        << std::fixed
        << std::setprecision(2)
        << total_memory / 1048576.0
        << " MiB\n";

    std::cout
        << "explicit tensors : "
        << explicit_bytes / 1048576.0
        << " MiB\n";

    if (free_before >= free_after_alloc) {
        std::cout
            << "VRAM alloc delta : "
            << (free_before - free_after_alloc)
                   / 1048576.0
            << " MiB\n";
    }

    auto compute_once = [&]() {
        const ggml_status status =
            ggml_backend_graph_compute(
                backend,
                graph
            );

        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "ggml_backend_graph_compute failed, status=" +
                std::to_string(
                    static_cast<int>(status)
                )
            );
        }

        ggml_backend_synchronize(backend);
    };

    //
    // First execution includes Vulkan pipeline/shader startup effects.
    //
    auto first_0 =
        std::chrono::steady_clock::now();

    compute_once();

    auto first_1 =
        std::chrono::steady_clock::now();

    const double first_ms =
        std::chrono::duration<double, std::milli>(
            first_1 - first_0
        ).count();

    //
    // Additional warmup.
    //
    for (int i = 0; i < a.warmup; ++i) {
        compute_once();
    }

    //
    // Steady-state runs.
    //
    std::vector<double> times;
    times.reserve(a.iters);

    for (int i = 0; i < a.iters; ++i) {
        const auto t0 =
            std::chrono::steady_clock::now();

        compute_once();

        const auto t1 =
            std::chrono::steady_clock::now();

        times.push_back(
            std::chrono::duration<double, std::milli>(
                t1 - t0
            ).count()
        );
    }

    double total_ms = 0.0;

    for (double x : times) {
        total_ms += x;
    }

    const double avg_ms =
        total_ms / times.size();

    const double p50_ms =
        percentile(times, 0.50);

    const double p95_ms =
        percentile(times, 0.95);

    const double q_throughput =
        static_cast<double>(a.q) /
        (avg_ms / 1000.0);

    //
    // QK^T:
    //   2 * H * Q * KV * D
    //
    // P*V:
    //   2 * H * Q * KV * D
    //
    const double attention_flops =
        4.0 *
        static_cast<double>(a.heads) *
        static_cast<double>(a.q) *
        static_cast<double>(a.kv) *
        static_cast<double>(a.d);

    const double tflops =
        attention_flops /
        (avg_ms / 1000.0) /
        1.0e12;

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "first execution  : "
        << first_ms
        << " ms\n";

    std::cout
        << "steady avg       : "
        << avg_ms
        << " ms\n";

    std::cout
        << "p50              : "
        << p50_ms
        << " ms\n";

    std::cout
        << "p95              : "
        << p95_ms
        << " ms\n";

    std::cout
        << "Q throughput     : "
        << q_throughput
        << " query-tok/s\n";

    std::cout
        << "attention rate   : "
        << tflops
        << " TFLOP/s-equivalent\n";

    std::cout
        << "============================================================\n";

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}

catch (const std::exception & e) {
    std::cerr
        << "ERROR: "
        << e.what()
        << "\n";

    return 1;
}