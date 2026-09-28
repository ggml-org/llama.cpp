// Standalone probe: can a second in-order SYCL queue on the same device/context
// run an H2D memcpy concurrently with a compute kernel on the first queue?
// Discriminator: device profiling intervals (command_start..command_end) intersect.
// Also: host-side submit latency of the memcpy, per host-memory kind, and a
// cross-queue device-side wait via ext_oneapi_submit_barrier(waitlist).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct iv { uint64_t s, e; };
static iv prof(const sycl::event & ev) {
    return { ev.get_profiling_info<sycl::info::event_profiling::command_start>(),
             ev.get_profiling_info<sycl::info::event_profiling::command_end>() };
}
static double dur_ms(iv a) { return (a.e - a.s) / 1e6; }
static double overlap_ms(iv a, iv b) {
    uint64_t lo = std::max(a.s, b.s), hi = std::min(a.e, b.e);
    return hi > lo ? (hi - lo) / 1e6 : 0.0;
}

static const size_t BUSY_N = 1u << 20;

static sycl::event busy(sycl::queue & q, float * buf, int iters) {
    return q.parallel_for(sycl::nd_range<1>(BUSY_N, 256), [=](sycl::nd_item<1> it) {
        size_t i = it.get_global_id(0);
        float x = buf[i];
        for (int k = 0; k < iters; ++k) {
            x = sycl::fma(x, 1.0000001f, 0.0000001f);
        }
        buf[i] = x;
    });
}

static void print_adapters() {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    bool v1 = false, v2 = false;
    while (std::getline(maps, line)) {
        if (line.find("libur_adapter_level_zero_v2") != std::string::npos) v2 = true;
        else if (line.find("libur_adapter_level_zero.so") != std::string::npos) v1 = true;
    }
    printf("adapters mapped: level_zero_v1=%d level_zero_v2=%d\n", v1, v2);
    const char * envs[] = { "SYCL_UR_USE_LEVEL_ZERO_V2", "UR_L0_USE_COPY_ENGINE",
                            "UR_L0_USE_IMMEDIATE_COMMANDLISTS", "UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE",
                            "ONEAPI_DEVICE_SELECTOR" };
    for (const char * e : envs) {
        const char * v = getenv(e);
        printf("env %s=%s\n", e, v ? v : "(unset)");
    }
}

struct src_mem {
    std::string kind;
    void * ptr = nullptr;
    size_t size = 0;
    int fd = -1;
};

static uint32_t pat(size_t i) { return (uint32_t) (i * 2654435761u); }

static src_mem make_src(const std::string & kind, size_t sz, sycl::queue & q, const std::string & file_path) {
    src_mem m;
    m.kind = kind;
    m.size = sz;
    if (kind == "pageable" || kind == "imported") {
        m.ptr = aligned_alloc(4096, sz);
    } else if (kind == "host_usm") {
        m.ptr = sycl::malloc_host(sz, q);
    } else if (kind == "mmap" || kind == "mmap_imported") {
        // Write the pattern to a file once, then map it read-only like llama.cpp's mmap loader.
        {
            std::vector<uint32_t> tmp(sz / 4);
            for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = pat(i);
            FILE * f = fopen(file_path.c_str(), "wb");
            if (!f) { perror("fopen"); exit(2); }
            fwrite(tmp.data(), 1, sz, f);
            fclose(f);
        }
        m.fd = open(file_path.c_str(), O_RDONLY);
        m.ptr = mmap(nullptr, sz, PROT_READ, MAP_SHARED, m.fd, 0);
        if (m.ptr == MAP_FAILED) { perror("mmap"); exit(2); }
        volatile uint32_t acc = 0;
        const uint32_t * p = (const uint32_t *) m.ptr;
        for (size_t i = 0; i < sz / 4; i += 1024) acc += p[i]; // prefault
        (void) acc;
        if (kind == "mmap_imported") {
            auto t0 = clk::now();
            sycl::ext::oneapi::experimental::prepare_for_device_copy(m.ptr, sz, q.get_context());
            printf("prepare_for_device_copy(mmap) took %.2f ms\n", ms_since(t0));
        }
        return m;
    }
    uint32_t * p = (uint32_t *) m.ptr;
    for (size_t i = 0; i < sz / 4; ++i) p[i] = pat(i);
    if (kind == "imported") {
        sycl::ext::oneapi::experimental::prepare_for_device_copy(m.ptr, sz, q.get_context());
    }
    return m;
}

static void free_src(src_mem & m, sycl::queue & q) {
    if (m.kind == "pageable") free(m.ptr);
    else if (m.kind == "imported") {
        sycl::ext::oneapi::experimental::release_from_device_copy(m.ptr, q.get_context());
        free(m.ptr);
    } else if (m.kind == "host_usm") sycl::free(m.ptr, q);
    else if (m.kind == "mmap") { munmap(m.ptr, m.size); close(m.fd); }
    else if (m.kind == "mmap_imported") {
        sycl::ext::oneapi::experimental::release_from_device_copy(m.ptr, q.get_context());
        munmap(m.ptr, m.size);
        close(m.fd);
    }
}

int main(int argc, char ** argv) {
    size_t mb = argc > 1 ? strtoul(argv[1], nullptr, 10) : 512;
    std::string kinds_arg = argc > 2 ? argv[2] : "host_usm,pageable,imported,mmap";
    std::string file_path = argc > 3 ? argv[3] : "/var/tmp/q2probe.bin";
    const size_t SZ = mb << 20;

    sycl::property_list props{ sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() };
    sycl::queue q1(sycl::gpu_selector_v, props);
    sycl::queue q2(q1.get_context(), q1.get_device(), props);
    sycl::queue q3(q1.get_device(), props); // how dpct create_queue_impl builds queues

    auto dev = q1.get_device();
    printf("device: %s | driver %s | backend %d\n", dev.get_info<sycl::info::device::name>().c_str(),
           dev.get_info<sycl::info::device::driver_version>().c_str(), (int) q1.get_backend());
    printf("q2 shares q1 context: %d | device-constructed q3 shares q1 context: %d\n",
           q2.get_context() == q1.get_context(), q3.get_context() == q1.get_context());
    print_adapters();

    float * kbuf = sycl::malloc_device<float>(BUSY_N, q1);
    q1.memset(kbuf, 0, BUSY_N * sizeof(float)).wait();
    uint32_t * dst = (uint32_t *) sycl::malloc_device(SZ, q1);
    uint32_t * err = sycl::malloc_device<uint32_t>(1, q1);

    // calibrate busy kernel to ~60 ms
    int iters = 4096;
    for (int rep = 0; rep < 6; ++rep) {
        auto e = busy(q1, kbuf, iters);
        e.wait();
        double d = dur_ms(prof(e));
        if (rep >= 1 && d > 40 && d < 90) break;
        iters = std::max(1, (int) (iters * (60.0 / std::max(d, 0.01))));
    }
    {
        auto e = busy(q1, kbuf, iters);
        e.wait();
        printf("busy kernel: iters=%d dur=%.2f ms\n", iters, dur_ms(prof(e)));
    }

    size_t pos = 0;
    while (pos < kinds_arg.size()) {
        size_t c = kinds_arg.find(',', pos);
        std::string kind = kinds_arg.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        pos = c == std::string::npos ? kinds_arg.size() : c + 1;

        src_mem src = make_src(kind, SZ, q1, file_path);
        printf("\n=== src=%s size=%zu MiB ===\n", kind.c_str(), mb);

        for (int rep = 0; rep < 3; ++rep) {
            // copy alone on q2
            auto t0 = clk::now();
            auto ec = q2.memcpy(dst, src.ptr, SZ);
            double sub = ms_since(t0);
            ec.wait();
            double wall = ms_since(t0);
            iv ci = prof(ec);
            // kernel first on q1, then copy on q2
            auto t1 = clk::now();
            auto ek = busy(q1, kbuf, iters);
            auto t2 = clk::now();
            auto ec2 = q2.memcpy(dst, src.ptr, SZ);
            double sub2 = ms_since(t2);
            ek.wait(); ec2.wait();
            double wall2 = ms_since(t1);
            iv ki = prof(ek), c2 = prof(ec2);
            // copy first on q2, then kernel on q1
            auto t3 = clk::now();
            auto ec3 = q2.memcpy(dst, src.ptr, SZ);
            double sub3 = ms_since(t3);
            auto ek3 = busy(q1, kbuf, iters);
            ek3.wait(); ec3.wait();
            double wall3 = ms_since(t3);
            iv k3 = prof(ek3), c3 = prof(ec3);
            // control: both on q1 (in-order, must serialize)
            auto t4 = clk::now();
            auto ek4 = busy(q1, kbuf, iters);
            auto ec4 = q1.memcpy(dst, src.ptr, SZ);
            ec4.wait();
            double wall4 = ms_since(t4);
            iv k4 = prof(ek4), c4 = prof(ec4);

            printf("rep%d alone: copy %.2f ms dev (%.1f GB/s), submit %.2f ms, wall %.2f ms\n", rep, dur_ms(ci),
                   SZ / (dur_ms(ci) * 1e6), sub, wall);
            printf("rep%d k-then-c (2q): kernel %.2f copy %.2f overlap %.2f ms | copy submit %.2f ms | wall %.2f ms\n",
                   rep, dur_ms(ki), dur_ms(c2), overlap_ms(ki, c2), sub2, wall2);
            printf("rep%d c-then-k (2q): kernel %.2f copy %.2f overlap %.2f ms | copy submit %.2f ms | wall %.2f ms\n",
                   rep, dur_ms(k3), dur_ms(c3), overlap_ms(k3, c3), sub3, wall3);
            printf("rep%d control (1q): kernel %.2f copy %.2f overlap %.2f ms | wall %.2f ms\n", rep, dur_ms(k4),
                   dur_ms(c4), overlap_ms(k4, c4), wall4);
        }

        // cross-queue ordering: copy on q2, record marker on q2 (like ggml event_record),
        // device-side wait on q1 (proposed event_wait), then a verify kernel on q1.
        for (int rep = 0; rep < 2; ++rep) {
            q1.memset(dst, 0, SZ).wait();
            q1.memset(err, 0, sizeof(uint32_t)).wait();
            auto ec = q2.memcpy(dst, src.ptr, SZ);
            auto rec = q2.ext_oneapi_submit_barrier();
            auto t0 = clk::now();
            q1.ext_oneapi_submit_barrier({ rec });
            double bsub = ms_since(t0);
            auto st = ec.get_info<sycl::info::event::command_execution_status>();
            const size_t n = SZ / 4;
            uint32_t * d = dst;
            uint32_t * er = err;
            auto ev = q1.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                size_t i = id[0];
                if (d[i] != (uint32_t) (i * 2654435761u)) {
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*er);
                    a.fetch_add(1);
                }
            });
            double ksub = ms_since(t0);
            ev.wait();
            uint32_t herr = 0;
            q1.memcpy(&herr, err, sizeof(uint32_t)).wait();
            iv ci = prof(ec), vi = prof(ev);
            printf("xq-wait rep%d: barrier submit %.3f ms, copy status after barrier submit=%s, verify submit %.3f ms, "
                   "verify starts %.3f ms after copy end, mismatches=%u\n",
                   rep, bsub, st == sycl::info::event_command_status::complete ? "complete" : "not-complete", ksub,
                   ((double) vi.s - (double) ci.e) / 1e6, herr);
        }
        free_src(src, q1);
    }
    unlink(file_path.c_str());
    sycl::free(kbuf, q1);
    sycl::free(dst, q1);
    sycl::free(err, q1);
    return 0;
}
