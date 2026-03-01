#include <sycl/sycl.hpp>

#include "/opt/intel/oneapi/dnnl/2025.3/include/oneapi/dnnl/dnnl.hpp"
#include "/opt/intel/oneapi/dnnl/2025.3/include/oneapi/dnnl/dnnl_sycl.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t k_qk4_0 = 32;

struct Options {
    int device_id = 0;
    int64_t m = 2048;
    int64_t n = 1;
    int64_t k = 2048;
    int warmup = 1;
    int iters = 1;
    bool list_devices = false;
    bool reorder_only = false;
    bool grouped_scales = true;
    bool sweep = false;
    int64_t sweep_start = 0;
    int64_t sweep_end = 0;
    int64_t sweep_step = 0;
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --device=<id>\n"
                 "  --m=<rows> --n=<cols> --k=<k>\n"
                 "  --warmup=<n> --iters=<n>\n"
                 "  --reorder-only\n"
                 "  --grouped-scales=0|1\n"
                 "  --sweep-m=<start:end:step>\n"
                 "  --list-devices\n",
                 argv0);
}

static bool parse_sweep(const char * value, int64_t & start, int64_t & end, int64_t & step) {
    if (!value) return false;
    const char * c1 = std::strchr(value, ':');
    if (!c1) return false;
    const char * c2 = std::strchr(c1 + 1, ':');
    if (!c2) return false;
    start = std::strtoll(value, nullptr, 10);
    end = std::strtoll(c1 + 1, nullptr, 10);
    step = std::strtoll(c2 + 1, nullptr, 10);
    return start > 0 && end >= start && step > 0;
}

static void list_devices() {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::fprintf(stderr, "No GPU devices found.\n");
        return;
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto & dev = devices[i];
        std::string name = dev.get_info<sycl::info::device::name>();
        std::string vendor = dev.get_info<sycl::info::device::vendor>();
        std::fprintf(stderr, "[%zu] %s (%s)\n", i, name.c_str(), vendor.c_str());
    }
}

static sycl::device pick_device(int device_id) {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices available");
    }
    if (device_id < 0 || static_cast<size_t>(device_id) >= devices.size()) {
        throw std::runtime_error("Invalid device id");
    }
    return devices[static_cast<size_t>(device_id)];
}

static void fill_weights_s4(std::vector<uint8_t> & weights_s4) {
    for (size_t i = 0; i < weights_s4.size(); ++i) {
        weights_s4[i] = 0x11;
    }
}

static void fill_scales(std::vector<float> & scales) {
    for (float & v : scales) {
        v = 1.0f;
    }
}

static void fill_zero_points(std::vector<int8_t> & zps) {
    for (int8_t & v : zps) {
        v = 0;
    }
}

static void fill_src(std::vector<sycl::half> & src) {
    for (auto & v : src) {
        v = sycl::half(1.0f);
    }
}

static bool run_woq_once(const Options & opts, int64_t m, int64_t n, int64_t k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        std::fprintf(stderr, "Invalid dims: M=%lld N=%lld K=%lld\n",
                     static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k));
        return false;
    }
    if ((k % k_qk4_0) != 0) {
        std::fprintf(stderr, "K must be divisible by %lld for s4 weights (K=%lld)\n",
                     static_cast<long long>(k_qk4_0), static_cast<long long>(k));
        return false;
    }

    const int64_t groups = k / k_qk4_0;
    const size_t weights_elems = static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t weights_bytes = (weights_elems + 1) / 2;
    const size_t src_elems = static_cast<size_t>(n) * static_cast<size_t>(k);
    const size_t dst_elems = static_cast<size_t>(n) * static_cast<size_t>(m);

    std::vector<uint8_t> weights_s4(weights_bytes);
    const size_t scale_elems = opts.grouped_scales
        ? static_cast<size_t>(groups) * static_cast<size_t>(m)
        : 1;
    std::vector<float> scales(scale_elems);
    std::vector<int8_t> zero_points(scale_elems);
    std::vector<sycl::half> src(src_elems);

    fill_weights_s4(weights_s4);
    fill_scales(scales);
    fill_zero_points(zero_points);
    fill_src(src);

    sycl::queue queue(pick_device(opts.device_id));

    uint8_t * wei_user_dev = static_cast<uint8_t *>(
        sycl::aligned_alloc_device(64, weights_bytes, queue));
    sycl::half * src_dev = static_cast<sycl::half *>(
        sycl::aligned_alloc_device(64, src_elems * sizeof(sycl::half), queue));
    sycl::half * dst_dev = static_cast<sycl::half *>(
        sycl::aligned_alloc_device(64, dst_elems * sizeof(sycl::half), queue));
    float * scales_dev = static_cast<float *>(
        sycl::aligned_alloc_device(64, scales.size() * sizeof(float), queue));
    int8_t * zp_dev = static_cast<int8_t *>(
        sycl::aligned_alloc_device(64, zero_points.size() * sizeof(int8_t), queue));

    if (!wei_user_dev || !src_dev || !dst_dev || !scales_dev || !zp_dev) {
        if (wei_user_dev) sycl::free(wei_user_dev, queue);
        if (src_dev) sycl::free(src_dev, queue);
        if (dst_dev) sycl::free(dst_dev, queue);
        if (scales_dev) sycl::free(scales_dev, queue);
        if (zp_dev) sycl::free(zp_dev, queue);
        std::fprintf(stderr, "Device allocation failed\n");
        return false;
    }

    queue.memcpy(wei_user_dev, weights_s4.data(), weights_bytes);
    queue.memcpy(src_dev, src.data(), src_elems * sizeof(sycl::half));
    queue.memcpy(scales_dev, scales.data(), scales.size() * sizeof(float));
    queue.memcpy(zp_dev, zero_points.data(), zero_points.size() * sizeof(int8_t));
    queue.wait_and_throw();

    auto eng = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(eng, queue);

    dnnl::memory::desc src_md({n, k}, dnnl::memory::data_type::f16, {k, 1});
    dnnl::memory::desc wei_user_md({k, m}, dnnl::memory::data_type::s4, {m, 1});
    dnnl::memory::desc dst_md({n, m}, dnnl::memory::data_type::f16, {m, 1});
    dnnl::memory::desc wei_any_md({k, m}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::any);

    dnnl::primitive_attr attr;
    if (opts.grouped_scales) {
        const int mask = (1 << 0) | (1 << 1);
        attr.set_scales(DNNL_ARG_WEIGHTS, mask,
                        dnnl::memory::dims{static_cast<dnnl_dim_t>(k_qk4_0), 1},
                        dnnl::memory::data_type::f32);
        attr.set_zero_points(DNNL_ARG_WEIGHTS, mask,
                             dnnl::memory::dims{static_cast<dnnl_dim_t>(k_qk4_0), 1},
                             dnnl::memory::data_type::s8);
    } else {
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
        attr.set_zero_points_mask(DNNL_ARG_WEIGHTS, 0);
    }
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);

    dnnl::matmul::primitive_desc matmul_pd(eng, src_md, wei_any_md, dst_md, attr);

    dnnl::memory src_mem(src_md, eng, src_dev);
    dnnl::memory wei_user_mem(wei_user_md, eng, wei_user_dev);
    dnnl::memory dst_mem(dst_md, eng, dst_dev);
    dnnl::memory scales_mem = opts.grouped_scales
        ? dnnl::memory({{m, groups}, dnnl::memory::data_type::f32, {1, m}}, eng, scales_dev)
        : dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng, scales_dev);
    dnnl::memory zp_mem = opts.grouped_scales
        ? dnnl::memory({{m, groups}, dnnl::memory::data_type::s8, {1, m}}, eng, zp_dev)
        : dnnl::memory({{1}, dnnl::memory::data_type::s8, {1}}, eng, zp_dev);

    dnnl::memory wei_mem = wei_user_mem;
    uint8_t * wei_packed_dev = nullptr;
    bool reordered = false;
    if (matmul_pd.weights_desc() != wei_user_mem.get_desc()) {
        const size_t wei_packed_bytes = matmul_pd.weights_desc().get_size();
        wei_packed_dev = static_cast<uint8_t *>(sycl::aligned_alloc_device(64, wei_packed_bytes, queue));
        if (!wei_packed_dev) {
            sycl::free(wei_user_dev, queue);
            sycl::free(src_dev, queue);
            sycl::free(dst_dev, queue);
            sycl::free(scales_dev, queue);
            sycl::free(zp_dev, queue);
            std::fprintf(stderr, "Packed weight allocation failed\n");
            return false;
        }
        wei_mem = dnnl::memory(matmul_pd.weights_desc(), eng, wei_packed_dev);
        dnnl::reorder(wei_user_mem, wei_mem).execute(stream, wei_user_mem, wei_mem);
        stream.wait();
        reordered = true;
    }

    if (opts.reorder_only) {
        std::fprintf(stderr,
                     "M=%lld N=%lld K=%lld status=OK avg_us=0.00 reorder=%d (reorder-only)\n",
                     static_cast<long long>(m), static_cast<long long>(n),
                     static_cast<long long>(k), reordered ? 1 : 0);
        sycl::free(wei_user_dev, queue);
        sycl::free(src_dev, queue);
        sycl::free(dst_dev, queue);
        sycl::free(scales_dev, queue);
        sycl::free(zp_dev, queue);
        if (wei_packed_dev) {
            sycl::free(wei_packed_dev, queue);
        }
        return true;
    }

    dnnl::matmul matmul_prim(matmul_pd);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_SRC, src_mem});
    args.insert({DNNL_ARG_WEIGHTS, wei_mem});
    args.insert({DNNL_ARG_DST, dst_mem});
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem});
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem});

    for (int i = 0; i < opts.warmup; ++i) {
        matmul_prim.execute(stream, args);
    }
    stream.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < opts.iters; ++i) {
        matmul_prim.execute(stream, args);
    }
    stream.wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double avg_us = (opts.iters > 0) ? (us / opts.iters) : 0.0;

    std::fprintf(stderr,
                 "M=%lld N=%lld K=%lld status=OK avg_us=%.2f reorder=%d\n",
                 static_cast<long long>(m), static_cast<long long>(n),
                 static_cast<long long>(k), avg_us, reordered ? 1 : 0);

    sycl::free(wei_user_dev, queue);
    sycl::free(src_dev, queue);
    sycl::free(dst_dev, queue);
    sycl::free(scales_dev, queue);
    sycl::free(zp_dev, queue);
    if (wei_packed_dev) {
        sycl::free(wei_packed_dev, queue);
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    Options opts;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--list-devices") == 0) {
            opts.list_devices = true;
            continue;
        }
        if (std::strncmp(arg, "--device=", 9) == 0) {
            opts.device_id = std::atoi(arg + 9);
            continue;
        }
        if (std::strncmp(arg, "--m=", 4) == 0) {
            opts.m = std::strtoll(arg + 4, nullptr, 10);
            continue;
        }
        if (std::strncmp(arg, "--n=", 4) == 0) {
            opts.n = std::strtoll(arg + 4, nullptr, 10);
            continue;
        }
        if (std::strncmp(arg, "--k=", 4) == 0) {
            opts.k = std::strtoll(arg + 4, nullptr, 10);
            continue;
        }
        if (std::strncmp(arg, "--warmup=", 9) == 0) {
            opts.warmup = std::atoi(arg + 9);
            continue;
        }
        if (std::strncmp(arg, "--iters=", 8) == 0) {
            opts.iters = std::atoi(arg + 8);
            continue;
        }
        if (std::strcmp(arg, "--reorder-only") == 0) {
            opts.reorder_only = true;
            continue;
        }
        if (std::strncmp(arg, "--grouped-scales=", 16) == 0) {
            opts.grouped_scales = std::atoi(arg + 16) != 0;
            continue;
        }
        if (std::strncmp(arg, "--sweep-m=", 10) == 0) {
            opts.sweep = parse_sweep(arg + 10, opts.sweep_start, opts.sweep_end, opts.sweep_step);
            if (!opts.sweep) {
                std::fprintf(stderr, "Invalid sweep format. Expected start:end:step\n");
                return 1;
            }
            continue;
        }
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", arg);
        print_usage(argv[0]);
        return 1;
    }

    if (opts.list_devices) {
        list_devices();
        return 0;
    }

    try {
        if (opts.sweep) {
            for (int64_t m = opts.sweep_start; m <= opts.sweep_end; m += opts.sweep_step) {
                if (!run_woq_once(opts, m, opts.n, opts.k)) {
                    return 1;
                }
            }
            return 0;
        }
        return run_woq_once(opts, opts.m, opts.n, opts.k) ? 0 : 1;
    } catch (const dnnl::error & e) {
        std::fprintf(stderr, "oneDNN error: %s (%d)\n", e.message, e.status);
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
    }
    return 1;
}
