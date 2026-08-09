#include "llama-expert-preload.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstring>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace llama_expert_preload {

namespace {
    // matches an expert weight tensor, e.g. blk.0.ffn_gate_exps.weight
    const std::regex g_re_exps("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

    int g_slots = 0;

    struct ggml_backend_buffer * g_gpu_buf = nullptr;
    struct ggml_context * g_ctx = nullptr; // temp context for the store write tensors
    uint8_t * g_cpu_buf = nullptr;         // host buffer for the cold experts
    size_t g_cpu_size = 0;
    std::vector<entry> g_entries;
    std::unordered_map<const ggml_tensor *, size_t> g_src_idx; // src -> entry index
    std::vector<std::vector<uint64_t>> g_addrs; // [entry][expert] -> host address
    std::vector<std::vector<uint64_t>> g_hashes; // [entry][expert] -> gguf FNV-1a(1024B)
    size_t g_gpu_cursor = 0; // next free offset in the store
    size_t g_cpu_cursor = 0; // next free offset in the host buffer

    static uint64_t fnv1a(const uint8_t * p, size_t n) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < n; i++) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    std::string g_path;
    int g_fd = -1;

    // C callback for the ggml cold op: resolve a cold expert's address from the
    // table (set at write time, so it always matches where the slice landed)
    const uint8_t * preload_slice_cb(const struct ggml_tensor * src0, int expert) {
        if (!g_cpu_buf || !src0 || expert < 0) {
            return nullptr;
        }
        const int i = index_of(src0);
        if (i < 0 || expert >= (int) g_addrs[i].size()) {
            return nullptr;
        }
        return (const uint8_t *) (uintptr_t) g_addrs[i][expert];
    }
}

void set_model_path(const char * path) {
    g_path = path ? path : "";
    if (g_fd >= 0) {
        close(g_fd);
        g_fd = -1;
    }
    g_fd = open(g_path.c_str(), O_RDONLY);
}

void set_slots(int s) {
    g_slots = s;
}

int get_slots() {
    return g_slots;
}

bool g_force = false;
bool g_no_evict = false;

void set_force(bool force) {
    g_force = force;
}

bool get_force() {
    return g_force;
}

void set_no_evict(bool no_evict) {
    g_no_evict = no_evict;
}

bool get_no_evict() {
    return g_no_evict;
}

bool tier_will_engage() {
    if (g_slots <= 0) {
        return false;
    }
    if (g_force || getenv("LLAMA_EXPERT_CACHE_FORCE") != nullptr) {
        return true;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return true; // any GPU backend (CUDA, Vulkan, ROCm, SYCL, Metal, ...)
        }
    }
    return false;
}

size_t align_up256(size_t x) {
    const size_t align = 256;
    return (x + align - 1) & ~(align - 1);
}

size_t store_total_bytes(size_t entries_bytes, int n_layers, int n_experts, int slots) {
    size_t off = align_up256(entries_bytes);
    for (int il = 0; il < n_layers; il++) {
        off = align_up256(off);
        off += align_up256((size_t) n_experts * sizeof(int32_t));
        off = align_up256(off);
        off += align_up256((size_t) (slots + 1) * sizeof(float));
    }
    return off;
}

bool is_exps(const char * name, int & layer_idx) {
    if (!name) {
        return false;
    }
    std::cmatch m;
    if (!std::regex_search(name, m, g_re_exps)) {
        return false;
    }
    layer_idx = std::stoi(m[1].str());
    return true;
}

void begin(struct ggml_backend_buffer_type * buft, size_t gpu_bytes, size_t cpu_bytes, int max_entries) {
    if (g_gpu_buf || g_cpu_buf) {
        return; // load_all_data can run more than once (fit estimate + real load)
    }
    g_gpu_buf = ggml_backend_buft_alloc_buffer(buft, gpu_bytes);
    // zero the whole store up front: the sentinel plane and any unwritten gaps
    // would otherwise carry per-launch garbage into the graph output (Vulkan)
    ggml_backend_buffer_clear(g_gpu_buf, 0);
    g_ctx = ggml_init({ ggml_tensor_overhead() * (max_entries + 8), nullptr, true });
    g_cpu_buf = (uint8_t *) malloc(cpu_bytes);
    g_cpu_size = cpu_bytes;
    g_entries.clear();
    g_src_idx.clear();
    g_addrs.clear();
    g_hashes.clear();
    g_gpu_cursor = 0;
    g_cpu_cursor = 0;
}

size_t register_tensor(const ggml_tensor * src, size_t plane_bytes, int n_experts, int startup,
                       size_t file_off, int fd) {
    for (size_t i = 0; i < g_entries.size(); i++) {
        if (g_entries[i].src == src) {
            g_entries[i].fd = fd; // refresh: the file may be reopened between passes
            return i; // already registered (second load pass)
        }
    }
    const size_t gpu_off = g_gpu_cursor;
    const size_t cpu_off = g_cpu_cursor;
    g_entries.push_back({src, plane_bytes, gpu_off, cpu_off, file_off, fd, n_experts, startup});
    g_src_idx[src] = g_entries.size() - 1;
    g_addrs.emplace_back((size_t) n_experts, 0);
    g_hashes.emplace_back((size_t) n_experts, 0);
    g_gpu_cursor += (size_t) (startup + 1) * plane_bytes;
    g_cpu_cursor += (size_t) n_experts * plane_bytes; // all experts, cold committed at load
    return g_entries.size() - 1;
}

bool write_entry(size_t idx, const uint8_t * data, size_t nbytes) {
    if (idx >= g_entries.size() || !data) {
        return false;
    }
    const entry & e = g_entries[idx];
    if (nbytes < (size_t) e.startup * e.plane_bytes) {
        return false;
    }
    // stream the startup slices into the store buffer at load (fast startup,
    // no token-1 file reads). the store adopts this buffer and re-allocates
    // its dst tensors at the same gpu_offset; the scratch tensor below dies
    // with g_ctx at take_buffer(). the cold op's slice callback ignores the
    // startup experts (they are GPU-counted), so no host copy is needed.
    if (g_gpu_buf && g_ctx) {
        ggml_tensor * t = ggml_new_tensor_1d(g_ctx, GGML_TYPE_I8, e.plane_bytes);
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(g_gpu_buf);
        for (int ex = 0; ex < e.startup; ex++) {
            uint8_t * dst = base + e.gpu_offset + (size_t) ex * e.plane_bytes;
            if (ex == 0) {
                ggml_backend_tensor_alloc(g_gpu_buf, t, dst);
            } else {
                t->data = dst; // manual view into the adopted store buffer
            }
            ggml_backend_tensor_set(t, data + (size_t) ex * e.plane_bytes, 0, e.plane_bytes);
        }
    }
    const size_t chunk = e.plane_bytes < 1024 ? e.plane_bytes : 1024;
    for (int ex = 0; ex < e.startup; ex++) {
        g_hashes[idx][ex] = fnv1a(data + (size_t) ex * e.plane_bytes, chunk);
    }
    return true;
}

bool read_expert(size_t idx, int expert, void * out, size_t n) {
    if (idx >= g_entries.size() || expert < 0 || expert >= g_entries[idx].n_experts) {
        return false;
    }
    const entry & e = g_entries[idx];
    if (n > e.plane_bytes) {
        return false;
    }
    const ssize_t got = pread(g_fd >= 0 ? g_fd : e.fd, out, n,
        (off_t) (e.file_off + (size_t) expert * e.plane_bytes));
    return got == (ssize_t) n;
}

bool write_cold(size_t idx, const uint8_t * data, size_t nbytes) {
    if (idx >= g_entries.size() || !g_cpu_buf || !data) {
        fprintf(stderr, "write_cold: FAIL idx=%zu cpu_buf=%p data=%p nbytes=%zu\n",
            idx, (void *) g_cpu_buf, (void *) data, nbytes);
        return false;
    }
    const entry & e = g_entries[idx];
    const size_t dst = e.cpu_offset + (size_t) e.startup * e.plane_bytes;
    if (dst + nbytes > g_cpu_size) {
        fprintf(stderr, "write_cold: OUT OF BOUNDS\n");
        return false;
    }
    std::memcpy(g_cpu_buf + dst, data, nbytes);
    const size_t chunk = e.plane_bytes < 1024 ? e.plane_bytes : 1024;
    for (int ex = e.startup; ex < e.n_experts; ex++) {
        g_addrs[idx][ex] = (uint64_t) (uintptr_t) (g_cpu_buf + e.cpu_offset + (size_t) ex * e.plane_bytes);
        g_hashes[idx][ex] = fnv1a(data + (size_t) (ex - e.startup) * e.plane_bytes, chunk);
    }
    if (getenv("LLAMA_EXPERT_DEBUG") && idx == 0) {
        fprintf(stderr, "write_cold: idx0 %s first bytes: %02x %02x %02x %02x\n",
            e.src->name, g_cpu_buf[dst], g_cpu_buf[dst+1], g_cpu_buf[dst+2], g_cpu_buf[dst+3]);
    }
    ggml_mmid_cold_set_slice_fn(preload_slice_cb);
    ggml_mmid_cold_set_hash_fn(
        [](const struct ggml_tensor * src0, int expert) -> uint64_t {
            return expected_hash(src0, expert);
        });
    return true;
}

struct ggml_backend_buffer * take_buffer() {
    struct ggml_backend_buffer * b = g_gpu_buf;
    g_gpu_buf = nullptr;
    if (g_ctx) {
        ggml_free(g_ctx);
        g_ctx = nullptr;
    }
    return b;
}

size_t num_entries() {
    return g_entries.size();
}

const entry * entry_at(size_t idx) {
    return idx < g_entries.size() ? &g_entries[idx] : nullptr;
}

int index_of(const ggml_tensor * src) {
    if (!src) {
        return -1;
    }
    for (size_t i = 0; i < g_entries.size(); i++) {
        if (g_entries[i].src == src) {
            return (int) i;
        }
    }
    // the hotstore's entry tensors can be distinct objects with the same name
    // (created in a different context); fall back to matching by name
    for (size_t i = 0; i < g_entries.size(); i++) {
        if (g_entries[i].src && g_entries[i].src->name[0] &&
            strcmp(g_entries[i].src->name, src->name) == 0) {
            return (int) i;
        }
    }
    return -1;
}

uint64_t expected_hash(const ggml_tensor * src, int expert) {
    const int idx = index_of(src);
    if (idx < 0 || expert < 0 || expert >= (int) g_hashes[idx].size()) {
        return 0;
    }
    return g_hashes[idx][expert];
}

size_t entries_size() {
    return g_gpu_cursor;
}

const uint8_t * cpu_slice(size_t idx, int expert) {
    if (idx >= g_entries.size() || !g_cpu_buf) {
        return nullptr;
    }
    const entry & e = g_entries[idx];
    if (expert < 0 || expert >= e.n_experts) {
        return nullptr;
    }
    return g_cpu_buf + e.cpu_offset + (size_t) expert * e.plane_bytes;
}

static void release_pages(void * ptr, size_t len) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize;
#else
    const long page = sysconf(_SC_PAGESIZE);
#endif
    const uintptr_t base   = (uintptr_t) ptr;
    const uintptr_t start  = (base + (uintptr_t) page - 1) & ~((uintptr_t) page - 1);
    const uintptr_t end    = (base + len) & ~((uintptr_t) page - 1);
    if (start < end) {
#ifdef _WIN32
        VirtualFree((LPVOID) start, end - start, MEM_RESET);
#else
        madvise((void *) start, end - start, MADV_DONTNEED);
#endif
    }
}

void free_cpu_slice(size_t idx, int expert) {
    if (idx >= g_entries.size() || !g_cpu_buf) {
        return;
    }
    const entry & e = g_entries[idx];
    if (expert < 0 || expert >= e.n_experts) {
        return;
    }
    release_pages(g_cpu_buf + e.cpu_offset + (size_t) expert * e.plane_bytes, e.plane_bytes);
}

void set_cpu_slice(size_t idx, int expert, const uint8_t * data) {
    if (idx >= g_entries.size() || !g_cpu_buf || !data) {
        return;
    }
    const entry & e = g_entries[idx];
    if (expert < 0 || expert >= e.n_experts) {
        return;
    }
    std::memcpy(g_cpu_buf + e.cpu_offset + (size_t) expert * e.plane_bytes, data, e.plane_bytes);
    g_addrs[idx][expert] = (uint64_t) (uintptr_t) (g_cpu_buf + e.cpu_offset + (size_t) expert * e.plane_bytes);
}

void clear() {
    if (g_gpu_buf) {
        ggml_backend_buffer_free(g_gpu_buf);
        g_gpu_buf = nullptr;
    }
    if (g_ctx) {
        ggml_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_cpu_buf) {
        free(g_cpu_buf);
        g_cpu_buf = nullptr;
    }
    g_cpu_size = 0;
    g_entries.clear();
    g_src_idx.clear();
    g_addrs.clear();
    g_gpu_cursor = 0;
    g_cpu_cursor = 0;
}

} // namespace llama_expert_preload
