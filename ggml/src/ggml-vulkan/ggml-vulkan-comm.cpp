#include "ggml-vulkan-common.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#if defined(__linux__)
#include <pthread.h>
#endif

static constexpr size_t ggml_vk_comm_f16_min_size = 1024 * 1024;

// follows ops ggml-backend-meta may delay a reduction across
// TODO: the required precision should come from the meta backend instead.
static bool ggml_vk_comm_needs_f32(const ggml_tensor * t, int depth = 0) {
    if (t == nullptr || depth > 64) {
        return false;
    }
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return ggml_get_op_params_i32(t, 0) == GGML_PREC_F32;
        case GGML_OP_ADD:
            return ggml_vk_comm_needs_f32(t->src[0], depth + 1) || ggml_vk_comm_needs_f32(t->src[1], depth + 1);
        case GGML_OP_MUL:
        case GGML_OP_ADD_ID:
            return ggml_vk_comm_needs_f32(t->src[0], depth + 1);
        case GGML_OP_VIEW:
            return ggml_vk_comm_needs_f32(t->src[0], depth + 1);
        default:
            return false;
    }
}

// host_bridge: device i signals done[i], a host thread signals go[i] once all peers of i are done
// sync_fd: a host thread exchanges SYNC_FD semaphores between peers, the wait resolves in the kernel
enum class vk_comm_sync {
    host_bridge,
    sync_fd,
};

struct vk_comm_device {
    ggml_backend_t                     backend  = nullptr;
    ggml_backend_vk_context *          ctx      = nullptr;
    void *                             host_ptr = nullptr;
    std::vector<ggml_backend_buffer_t> host_buf; // host_ptr imported into each peer

    struct {
        PFN_vkGetSemaphoreFdKHR    get_fd    = nullptr;
        PFN_vkImportSemaphoreFdKHR import_fd = nullptr;
        vk::Semaphore              sem;
        std::vector<vk::Semaphore> imp;      // [peer], pooled, reused only after ggml_vk_graph_cleanup
    } sfd;

    struct {
        vk::Semaphore done;
        vk::Semaphore go;
    } bridge;
};

struct ggml_backend_vk_comm_context {
    std::vector<vk_comm_device> devs;
    vk_comm_sync                mode     = vk_comm_sync::host_bridge;
    size_t                      align    = 4096;
    size_t                      cap      = 0;
    uint64_t                    round    = 0;
    ggml_context *              tctx     = nullptr;
    ggml_tensor *               slot     = nullptr;
    bool                        disabled = false;

    std::thread                 thread;
    std::chrono::nanoseconds    spin{ 200'000 };
    std::atomic<uint64_t>       submitted{ 0 };
    std::atomic<uint64_t>       imported{ 0 };
    std::atomic<bool>           sleeping{ false };
    std::atomic<bool>           stop{ false };
    std::mutex                  mutex;
    std::condition_variable     cv;
};

static void ggml_vk_comm_free_buffers(ggml_backend_vk_comm_context * comm) {
    for (vk_comm_device & d : comm->devs) {
        for (ggml_backend_buffer_t & buf : d.host_buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        std::free(d.host_ptr);
        d.host_ptr = nullptr;
    }
    comm->cap = 0;
}

static bool ggml_backend_vk_comm_ensure(ggml_backend_vk_comm_context * comm, size_t nbytes) {
    if (comm->disabled) {
        return false;
    }
    if (nbytes <= comm->cap) {
        return true;
    }
    auto fail = [comm]() {
        GGML_LOG_WARN("ggml_vulkan: shared host buffer allocation failed, using generic all-reduce\n");
        comm->disabled = true;
        return false;
    };
    for (vk_comm_device & d : comm->devs) {
        ggml_backend_synchronize(d.backend);
    }
    ggml_vk_comm_free_buffers(comm);

    const size_t n   = comm->devs.size();
    const size_t cap = GGML_PAD(nbytes, comm->align);
    for (size_t i = 0; i < n; i++) {
        vk_comm_device & d = comm->devs[i];
        if (posix_memalign(&d.host_ptr, comm->align, 2 * cap) != 0) {
            d.host_ptr = nullptr;
            return fail();
        }
        for (size_t j = 0; j < n; j++) {
            d.host_buf[j] = ggml_backend_vk_device_buffer_from_host_ptr(
                ggml_backend_get_device(comm->devs[j].backend), d.host_ptr, 2 * cap, 2 * cap);
            if (!d.host_buf[j]) {
                return fail();
            }
        }
    }
    comm->cap = cap;
    return true;
}

static void ggml_vk_comm_set_slot(ggml_tensor * slot, const ggml_tensor * src, ggml_type type, ggml_backend_buffer_t buf, size_t offs) {
    slot->type  = type;
    slot->nb[0] = ggml_type_size(type);
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        slot->ne[d] = src->ne[d];
        if (d > 0) {
            slot->nb[d] = slot->nb[d - 1] * slot->ne[d - 1];
        }
    }
    slot->buffer = buf;
    slot->data   = (char *) ggml_backend_buffer_get_base(buf) + offs;
}

static void ggml_vk_comm_submit(ggml_backend_vk_context * ctx, vk_context & c, vk::Semaphore signal, uint64_t value) {
    ggml_vk_ctx_end(c);
    for (auto & cpy : c->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }
    for (auto & mset : c->memsets) {
        memset(mset.dst, mset.val, mset.n);
    }
    c->in_memcpys.clear();
    c->memsets.clear();
    if (signal) {
        c->seqs.back().back().signal_semaphores.push_back({ signal, value });
    }
    ggml_vk_submit(c, {});
    ctx->submit_pending = true;
    ctx->compute_ctx.reset();
}

static void ggml_vk_comm_notify_submitted(ggml_backend_vk_comm_context * comm, uint64_t round) {
    comm->submitted.store(round);
    if (comm->sleeping.load()) {
        std::lock_guard<std::mutex> lock(comm->mutex);
        comm->cv.notify_one();
    }
}

// false on stop
static bool ggml_vk_comm_wait_submitted(ggml_backend_vk_comm_context * comm, uint64_t r) {
    using clock = std::chrono::steady_clock;
    for (const auto t0 = clock::now(); comm->submitted.load() < r;) {
        if (comm->stop.load()) {
            return false;
        }
        if (clock::now() - t0 < comm->spin) {
            YIELD();
            continue;
        }
        comm->sleeping.store(true);
        if (comm->submitted.load() < r) {
            std::unique_lock<std::mutex> lock(comm->mutex);
            comm->cv.wait(lock, [&] { return comm->submitted.load() >= r || comm->stop.load(); });
        }
        comm->sleeping.store(false);
    }
    return true;
}

// signals go[i] once all peers of i are done; false on stop
static bool ggml_vk_comm_bridge_release(ggml_backend_vk_comm_context * comm, uint64_t r, std::vector<uint64_t> & seen, std::vector<bool> & sent) {
    using clock = std::chrono::steady_clock;
    const size_t n = comm->devs.size();
    std::fill(sent.begin(), sent.end(), false);
    size_t left = n;
    for (const auto t0 = clock::now();;) {
        size_t pending = n;
        for (size_t i = 0; i < n; i++) {
            if (sent[i]) {
                continue;
            }
            bool ready = true;
            for (size_t j = 0; j < n && ready; j++) {
                if (j != i && seen[j] < r) {
                    const vk_comm_device & p = comm->devs[j];
                    seen[j] = p.ctx->device->device.getSemaphoreCounterValue(p.bridge.done);
                    ready   = seen[j] >= r;
                    pending = ready ? pending : j;
                }
            }
            if (ready) {
                const vk_comm_device & d = comm->devs[i];
                d.ctx->device->device.signalSemaphore({ d.bridge.go, r });
                sent[i] = true;
                left--;
            }
        }
        if (left == 0) {
            return true;
        }
        if (clock::now() - t0 < comm->spin) {
            YIELD();
            continue;
        }
        const vk_comm_device &      p = comm->devs[pending];
        const vk::SemaphoreWaitInfo wi{ {}, 1, &p.bridge.done, &r };
        while (p.ctx->device->device.waitSemaphores(wi, 100'000'000) != vk::Result::eSuccess) {
            if (comm->stop.load()) {
                return false;
            }
        }
    }
}

static void ggml_vk_comm_bridge_thread(ggml_backend_vk_comm_context * comm) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "ggml-vk-comm");
#endif
    std::vector<uint64_t> seen(comm->devs.size(), 0);
    std::vector<bool>     sent(comm->devs.size());
    try {
        for (uint64_t r = 1;; r++) {
            if (!ggml_vk_comm_wait_submitted(comm, r) || !ggml_vk_comm_bridge_release(comm, r, seen, sent)) {
                return;
            }
        }
    } catch (const vk::SystemError & e) {
        GGML_ABORT("ggml_vulkan: all-reduce bridge failed: %s", e.what());
    }
}

static void ggml_vk_comm_bridge_init(ggml_backend_vk_comm_context * comm) {
    vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo     sci{};
    sci.pNext = &stci;
    for (vk_comm_device & d : comm->devs) {
        d.bridge.done = d.ctx->device->device.createSemaphore(sci);
        d.bridge.go   = d.ctx->device->device.createSemaphore(sci);
    }
    comm->thread = std::thread(ggml_vk_comm_bridge_thread, comm);
}

static void ggml_vk_comm_bridge_free(vk_comm_device & d) {
    if (d.bridge.done) {
        d.ctx->device->device.destroySemaphore(d.bridge.done);
    }
    if (d.bridge.go) {
        d.ctx->device->device.destroySemaphore(d.bridge.go);
    }
}

static void ggml_vk_comm_barrier_to_host(vk_context & c) {
    const vk::MemoryBarrier to_host{ vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead };
    c->s->buffer->buf.pipelineBarrier(c->p->q->stage_flags, vk::PipelineStageFlagBits::eHost, {}, { to_host }, {}, {});
}

// the peers' slots were made available to the host before the wait resolved, the submit predates that
static void ggml_vk_comm_barrier_from_host(vk_context & c) {
    const vk::MemoryBarrier from_host{ vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead };
    c->s->buffer->buf.pipelineBarrier(vk::PipelineStageFlagBits::eHost, c->p->q->stage_flags, {}, { from_host }, {}, {});
}

static void ggml_vk_comm_bridge_signal(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
    const vk_comm_device & d = comm->devs[i];
    ggml_vk_comm_barrier_to_host(c);
    ggml_vk_comm_submit(d.ctx, c, d.bridge.done, round);
}

static void ggml_vk_comm_bridge_wait(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
    c->s->wait_semaphores.push_back({ comm->devs[i].bridge.go, round });
    ggml_vk_comm_barrier_from_host(c);
}

static bool ggml_vk_comm_sfd_supported(vk_comm_device & d) {
    const vk_device & dev = d.ctx->device;
    const bool mesa = dev->driver_id == vk::DriverId::eMesaRadv ||
                      dev->driver_id == vk::DriverId::eIntelOpenSourceMESA ||
                      dev->driver_id == vk::DriverId::eMesaNvk;
    if (!mesa || !dev->external_semaphore) {
        return false;
    }
    d.sfd.get_fd    = (PFN_vkGetSemaphoreFdKHR)    dev->device.getProcAddr("vkGetSemaphoreFdKHR");
    d.sfd.import_fd = (PFN_vkImportSemaphoreFdKHR) dev->device.getProcAddr("vkImportSemaphoreFdKHR");
    if (!d.sfd.get_fd || !d.sfd.import_fd) {
        return false;
    }
    vk::PhysicalDeviceExternalSemaphoreInfo esi{ vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd };
    vk::ExternalSemaphoreProperties esp = dev->physical_device.getExternalSemaphoreProperties(esi);
    const vk::ExternalSemaphoreFeatureFlags need =
        vk::ExternalSemaphoreFeatureFlagBits::eExportable | vk::ExternalSemaphoreFeatureFlagBits::eImportable;
    return (esp.externalSemaphoreFeatures & need) == need &&
           (esp.compatibleHandleTypes & vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd);
}

// Mesa moves a queue to its submit thread on the first wait-before-signal, taking the CS ioctls off this thread
static void ggml_vk_comm_sfd_enable_submit_thread(const vk_device & dev) {
    vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo     sci{};
    sci.pNext = &stci;
    const vk::Semaphore sem   = dev->device.createSemaphore(sci);
    const vk::Fence     fence = dev->device.createFence({});
    const uint64_t      one   = 1;
    vk::TimelineSemaphoreSubmitInfo tsi{ 1, &one, 0, nullptr };
    const vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
    vk::SubmitInfo si{ 1, &sem, &stage, 0, nullptr, 0, nullptr };
    si.setPNext(&tsi);
    dev->compute_queue->handle->submit(si, fence);
    dev->device.signalSemaphore({ sem, 1 });
    VK_CHECK(dev->device.waitForFences({ fence }, true, UINT64_MAX), "ggml_vk_comm_sfd_enable_submit_thread", dev);
    dev->device.destroyFence(fence);
    dev->device.destroySemaphore(sem);
}

// Requires a submitted signal on sem; the export resets it.
static int ggml_vk_comm_sfd_export(vk_comm_device & d) {
    int fd = -1;
    VkSemaphoreGetFdInfoKHR gi{};
    gi.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    gi.semaphore  = d.sfd.sem;
    gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    const VkResult result = d.sfd.get_fd((VkDevice) d.ctx->device->device, &gi, &fd);
    if (result != VK_SUCCESS) {
        GGML_ABORT("ggml_vulkan: vkGetSemaphoreFdKHR failed: %d", (int) result);
    }
    return fd;
}

// fd -1 imports an already signalled payload
static void ggml_vk_comm_sfd_import(vk_comm_device & d, vk::Semaphore sem, int fd) {
    VkImportSemaphoreFdInfoKHR isi{};
    isi.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
    isi.semaphore  = sem;
    isi.flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    isi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    isi.fd         = fd;
    const VkResult result = d.sfd.import_fd((VkDevice) d.ctx->device->device, &isi);
    if (result != VK_SUCCESS) {
        GGML_ABORT("ggml_vulkan: vkImportSemaphoreFdKHR failed: %d", (int) result);
    }
}

static void ggml_vk_comm_sfd_exchange(ggml_backend_vk_comm_context * comm, uint64_t r) {
    const size_t n = comm->devs.size();
    for (size_t j = 0; j < n; j++) {
        const int    fd   = ggml_vk_comm_sfd_export(comm->devs[j]);
        const size_t last = j == n - 1 ? n - 2 : n - 1;
        for (size_t i = 0; i < n; i++) {
            if (i == j) {
                continue;
            }
            const int imp_fd = fd < 0 || i == last ? fd : fcntl(fd, F_DUPFD_CLOEXEC, 0);
            GGML_ASSERT(imp_fd >= 0 || fd < 0);
            ggml_vk_comm_sfd_import(comm->devs[i], comm->devs[i].sfd.imp[j], imp_fd);
        }
    }
    comm->imported.store(r, std::memory_order_release);
}

static void ggml_vk_comm_sfd_thread(ggml_backend_vk_comm_context * comm) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "ggml-vk-comm");
#endif
    for (uint64_t r = 1; ggml_vk_comm_wait_submitted(comm, r); r++) {
        ggml_vk_comm_sfd_exchange(comm, r);
    }
}

static void ggml_vk_comm_sfd_init(ggml_backend_vk_comm_context * comm) {
    vk::ExportSemaphoreCreateInfo esci{ vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd };
    vk::SemaphoreCreateInfo       sci{};
    sci.pNext = &esci;
    for (vk_comm_device & d : comm->devs) {
        d.sfd.sem = d.ctx->device->device.createSemaphore(sci);
        d.sfd.imp.assign(comm->devs.size(), {});
        ggml_vk_comm_sfd_enable_submit_thread(d.ctx->device);
    }
    comm->thread = std::thread(ggml_vk_comm_sfd_thread, comm);
}

static void ggml_vk_comm_sfd_free(vk_comm_device & d) {
    if (d.sfd.sem) {
        d.ctx->device->device.destroySemaphore(d.sfd.sem);
    }
}

static void ggml_vk_comm_sfd_signal(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c) {
    vk_comm_device & d = comm->devs[i];
    ggml_vk_comm_barrier_to_host(c);
    ggml_vk_comm_submit(d.ctx, c, d.sfd.sem, 0);
    for (size_t j = 0; j < comm->devs.size(); j++) {
        d.sfd.imp[j] = j == i ? vk::Semaphore{} : ggml_vk_create_binary_semaphore(d.ctx);
    }
}

static void ggml_vk_comm_sfd_wait(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
    for (size_t j = 0; j < comm->devs.size(); j++) {
        if (j != i) {
            c->s->wait_semaphores.push_back({ comm->devs[i].sfd.imp[j], 0 });
        }
    }
    c->s->ready       = &comm->imported;
    c->s->ready_value = round;
    ggml_vk_comm_barrier_from_host(c);
}

void ggml_backend_vk_comm_free(void * comm_ctx) {
    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
    // pending reductions need the thread
    for (vk_comm_device & d : comm->devs) {
        ggml_backend_synchronize(d.backend);
    }
    if (comm->thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(comm->mutex);
            comm->stop.store(true);
        }
        comm->cv.notify_all();
        comm->thread.join();
    }
    for (vk_comm_device & d : comm->devs) {
        ggml_vk_comm_sfd_free(d);
        ggml_vk_comm_bridge_free(d);
    }
    ggml_vk_comm_free_buffers(comm);
    if (comm->tctx) {
        ggml_free(comm->tctx);
    }
    delete comm;
}

void * ggml_backend_vk_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (vk_instance.allreduce_disabled || n_backends < 2) {
        return nullptr;
    }
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_vk(backends[i])) {
            return nullptr;
        }
    }
    ggml_backend_vk_comm_context * comm = new ggml_backend_vk_comm_context;
    comm->devs.resize(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        vk_comm_device & d = comm->devs[i];
        d.backend = backends[i];
        d.ctx     = (ggml_backend_vk_context *) backends[i]->context;
        d.host_buf.assign(n_backends, nullptr);
    }

    for (size_t i = 0; i < n_backends; i++) {
        const vk_device & dev = comm->devs[i].ctx->device;
        comm->align = std::max(comm->align, (size_t) dev->min_imported_host_pointer_alignment);
        if (!dev->external_memory_host) {
            GGML_LOG_WARN("ggml_vulkan: device %zu: VK_EXT_external_memory_host missing, using generic all-reduce\n", i);
            ggml_backend_vk_comm_free(comm);
            return nullptr;
        }
        if (dev->serialize_submissions) {
            GGML_LOG_WARN("ggml_vulkan: device %zu: GGML_VK_SERIALIZE_SUBMISSIONS set, using generic all-reduce\n", i);
            ggml_backend_vk_comm_free(comm);
            return nullptr;
        }
    }

    if (std::all_of(comm->devs.begin(), comm->devs.end(), ggml_vk_comm_sfd_supported)) {
        comm->mode = vk_comm_sync::sync_fd;
    }

    if (const char * spin = getenv("GGML_VK_ALLREDUCE_SPIN_US")) {
        comm->spin = std::chrono::microseconds(std::max(0, atoi(spin)));
    }
    const ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    comm->tctx = ggml_init(ip);
    comm->slot = ggml_new_tensor_1d(comm->tctx, GGML_TYPE_F32, 1);

    try {
        if (comm->mode == vk_comm_sync::sync_fd) {
            ggml_vk_comm_sfd_init(comm);
        } else {
            ggml_vk_comm_bridge_init(comm);
        }
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: semaphore initialization failed: %s, using generic all-reduce\n", e.what());
        ggml_backend_vk_comm_free(comm);
        return nullptr;
    }
    return comm;
}

// A: partial to own host slot (round & 1), signal; B: wait on peers' A, add their slots; all A precede any B
// slot reuse: A(k) follows own B(k-1), which waited on peers' A(k-1), submitted after their B(k-2), the last reader
bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
    const size_t n = comm->devs.size();

    const int64_t ne = ggml_nelements(tensors[0]);
    for (size_t i = 0; i < n; i++) {
        const ggml_tensor * t = tensors[i];
        if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne || !ggml_is_contiguous(t) || !ggml_backend_buffer_is_vk(t->buffer)) {
            return false;
        }
        const ggml_backend_vk_context * ctx = comm->devs[i].ctx;
        if (!ctx->compute_ctx.expired() && !ctx->compute_ctx.lock()->out_memcpys.empty()) {
            return false;
        }
    }
    if (ne == 0) {
        return true;
    }
    const size_t nbytes = ggml_nbytes(tensors[0]);
    if (!ggml_backend_vk_comm_ensure(comm, nbytes)) {
        return false;
    }

    const bool      f32_wire  = nbytes < ggml_vk_comm_f16_min_size || ggml_vk_comm_needs_f32(tensors[0]);
    const ggml_type wire      = f32_wire ? GGML_TYPE_F32 : GGML_TYPE_F16;
    const uint64_t  round     = ++comm->round;
    const size_t    slot_offs = (round & 1) * comm->cap;
    const size_t    slot_size = ne * ggml_type_size(wire);

    for (size_t i = 0; i < n; i++) {
        const vk_comm_device &    d   = comm->devs[i];
        ggml_backend_vk_context * ctx = d.ctx;
        ggml_tensor *             t   = tensors[i];
        vk_buffer & tbuf = ((ggml_backend_vk_buffer_context *) t->buffer->context)->dev_buffer;
        vk_buffer & hbuf = ((ggml_backend_vk_buffer_context *) d.host_buf[i]->context)->dev_buffer;
        const size_t toffs = vk_tensor_offset(t) + t->view_offs;

        vk_context c = ggml_vk_get_compute_ctx(ctx);
        ggml_vk_sync_buffers(ctx, c);
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            c->s->buffer->buf.fillBuffer(hbuf->buffer, slot_offs, GGML_PAD(slot_size, 4), 0);
            c->s->buffer->buf.fillBuffer(tbuf->buffer, toffs, nbytes, 0);
        } else if (wire == GGML_TYPE_F32) {
            ggml_vk_buffer_copy_async(c, hbuf, slot_offs, tbuf, toffs, nbytes);
        } else {
            ggml_vk_comm_set_slot(comm->slot, t, wire, d.host_buf[i], slot_offs);
            ggml_vk_cpy(ctx, c, t, comm->slot);
        }
        if (comm->mode == vk_comm_sync::sync_fd) {
            ggml_vk_comm_sfd_signal(comm, i, c);
        } else {
            ggml_vk_comm_bridge_signal(comm, i, c, round);
        }
    }
    ggml_vk_comm_notify_submitted(comm, round);

    for (size_t i = 0; i < n; i++) {
        ggml_backend_vk_context * ctx = comm->devs[i].ctx;
        vk_context c = ggml_vk_get_compute_ctx(ctx);
        ctx->do_add_rms_partials = false;
        if (comm->mode == vk_comm_sync::sync_fd) {
            ggml_vk_comm_sfd_wait(comm, i, c, round);
        } else {
            ggml_vk_comm_bridge_wait(comm, i, c, round);
        }
        ggml_vk_sync_buffers(ctx, c);
        for (size_t j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            ggml_vk_comm_set_slot(comm->slot, tensors[i], wire, comm->devs[j].host_buf[i], slot_offs);
            ggml_vk_add(ctx, c, tensors[i], comm->slot, tensors[i]);
            ggml_vk_sync_buffers(ctx, c);
        }
        // small reductions go out with the next graph's first submit
        if (wire != GGML_TYPE_F32 || vk_perf_logger_enabled) {
            ggml_vk_comm_submit(ctx, c, {}, 0);
        }
    }
    return true;
}
#endif
