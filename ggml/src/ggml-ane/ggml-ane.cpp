// Apple Neural Engine backend (experimental)
//
// Uses the private AppleNeuralEngine framework (_ANEInMemoryModel) on macOS and iOS.
// - Each graph is cut into segments (by weight budget); each segment is translated to one MIL program with the weights
//   baked in (any GGUF type is converted to int8 with one scale per output row; f16/f32 weights stay fp16).
// - Programs are compiled by the OS (aned) and cached on disk by it. All the shapes of a segment that bake the same
//   weights are functions of one multi-function program ("family"), so the weights are stored once.
// - Program inputs/outputs are fp16 IOSurfaces; the KV cache is read in place through surfaces that wrap its memory.
// - KV cache writes (SET_ROWS) are done on the host after the program; attention in the same program uses the new rows.
// - ggml_backend_ane_collect_shapes() lets an app compile each program once with all its shapes (see ggml-ane.h).
//
// Environment:
//   GGML_ANE_WEIGHTS=auto|fp16|int8   weight format (auto: quantized -> int8, f16/f32 -> fp16)
//   GGML_ANE_SEG_WEIGHT_MB=64         weights per program
//   GGML_ANE_PROFILE=1                print where graph_compute spends its time
//   GGML_ANE_DUMP_DIR=dir             write programs that fail to compile to dir
//   GGML_ANE_PURGE_CACHE=1            remove compiled programs from the OS cache when they are freed (development)

#include "ggml-ane.h"
#include "ggml-ane-mil.h"
#include "ggml-ane-runtime.h"

#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <TargetConditionals.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_ANE_PAGE_SIZE    16384
#define GGML_ANE_MAX_PROGRAMS 1024 // cached shapes (cheap: programs are owned by families)
#define GGML_ANE_MAX_FAMILIES 160  // loaded programs: the ANE refuses to load more than ~240 per process
#define GGML_ANE_MAX_DIM      16384 // larger tensor dims fail to compile (the limit is lower on older ANEs)

//
// write tracking
//
// compiled programs embed the weight data, so a write to a baked tensor (or a clear of its buffer) must invalidate them
//

static std::mutex                                g_ane_write_mutex;
static std::unordered_map<const void *, uint64_t> g_ane_write_gen;
static uint64_t                                  g_ane_write_counter = 0;

static void ggml_ane_mark_written(const void * ptr) {
    std::lock_guard<std::mutex> lock(g_ane_write_mutex);
    g_ane_write_gen[ptr] = ++g_ane_write_counter;
}

static uint64_t ggml_ane_write_gen(const void * ptr) {
    std::lock_guard<std::mutex> lock(g_ane_write_mutex);
    auto it = g_ane_write_gen.find(ptr);
    return it == g_ane_write_gen.end() ? 0 : it->second;
}

// identity of the current content of a baked tensor: address + write generations of the tensor and of its buffer
// (whole-buffer writes only bump the buffer's generation)
static std::string ggml_ane_content_key(const ggml_tensor * t) {
    std::string key = std::to_string((uintptr_t) t->data) + ":" + std::to_string(ggml_ane_write_gen(t->data));
    if (t->buffer && t->buffer->iface.get_base) {
        key += ":" + std::to_string(ggml_ane_write_gen(ggml_backend_buffer_get_base(t->buffer)));
    }
    return key;
}

//
// buffer
//
// each buffer is backed by its own IOSurface (the ANE only reads/writes surfaces at offset 0), except buffers that wrap host memory
//

// surfaces over windows of host memory (no copy), cached by (address, size) and dropped when the memory is freed
static std::mutex g_ane_win_mutex;
static std::map<std::pair<uintptr_t, size_t>, IOSurfaceRef> g_ane_windows;

static IOSurfaceRef ggml_ane_window(void * ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_ane_win_mutex);
    auto key = std::make_pair((uintptr_t) ptr, size);
    auto it = g_ane_windows.find(key);
    if (it != g_ane_windows.end()) {
        return it->second;
    }
    IOSurfaceRef s = ggml_ane_surface_wrap(ptr, size);
    if (s) {
        g_ane_windows[key] = s;
    }
    return s;
}

static void ggml_ane_drop_windows(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(g_ane_win_mutex);
    for (auto it = g_ane_windows.lower_bound({ (uintptr_t) base, 0 }); it != g_ane_windows.end() && it->first.first < (uintptr_t) base + size; ) {
        CFRelease(it->second);
        it = g_ane_windows.erase(it);
    }
}

struct ggml_backend_ane_buffer_context {
    IOSurfaceRef surf = nullptr;
    void *       base = nullptr;

    ~ggml_backend_ane_buffer_context() {
        if (surf) {
            CFRelease(surf);
        }
    }
};

static void ggml_backend_ane_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_ane_buffer_context *) buffer->context;
    ggml_ane_drop_windows(ctx->base, buffer->size);
    delete ctx;
}

static void * ggml_backend_ane_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_ane_buffer_context *) buffer->context)->base;
}

static void ggml_backend_ane_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    ggml_ane_mark_written(tensor->data);
    GGML_UNUSED(buffer);
}

static void ggml_backend_ane_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    ggml_ane_mark_written(tensor->data);
    GGML_UNUSED(buffer);
}

static void ggml_backend_ane_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_ane_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        ggml_ane_mark_written(dst->data);
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void ggml_backend_ane_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_ane_buffer_context *) buffer->context;
    memset(ctx->base, value, buffer->size);
    ggml_ane_mark_written(ctx->base);
}

static const ggml_backend_buffer_i ggml_backend_ane_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_ane_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ane_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_ane_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_ane_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ane_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_ane_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_ane_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

//
// buffer type
//

static const char * ggml_backend_ane_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "ANE";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_ane_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t alloc = GGML_PAD(size > 0 ? size : 1, GGML_ANE_PAGE_SIZE);
    IOSurfaceRef surf = ggml_ane_surface_create(alloc);
    if (!surf) {
        GGML_LOG_ERROR("%s: failed to allocate IOSurface of %zu bytes\n", __func__, alloc);
        return nullptr;
    }
    auto * ctx = new ggml_backend_ane_buffer_context;
    ctx->surf = surf;
    ctx->base = IOSurfaceGetBaseAddress(surf);
    return ggml_backend_buffer_init(buft, ggml_backend_ane_buffer_interface, ctx, size);
}

// one buffer (= one IOSurface) per tensor, so that every tensor can be bound directly as a program input/output
static ggml_backend_buffer_t ggml_backend_ane_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    std::vector<ggml_backend_buffer_t> buffers;
    auto free_all = [&]() {
        for (ggml_backend_buffer_t b : buffers) {
            ggml_backend_buffer_free(b);
        }
    };

    for (int i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        if (t->data != nullptr || t->view_src != nullptr) {
            continue;
        }
        const size_t size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), ggml_backend_buft_get_alignment(buft));
        ggml_backend_buffer_t buf = ggml_backend_ane_buffer_type_alloc_buffer(buft, size);
        if (!buf) {
            free_all();
            return nullptr;
        }
        buffers.push_back(buf);
        ggml_tallocr talloc = ggml_tallocr_new(buf);
        if (ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to allocate tensor %s\n", __func__, t->name);
            free_all();
            return nullptr;
        }
    }
    for (int i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        if (t->view_src != nullptr && t->buffer == nullptr && ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to initialize view %s\n", __func__, t->name);
            free_all();
            return nullptr;
        }
    }

    if (buffers.empty()) {
        return nullptr;
    }
    if (buffers.size() == 1) {
        return buffers[0];
    }
    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

static size_t ggml_backend_ane_buffer_type_get_alloc_size_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    size_t total = 0;
    for (int i = 0; i < n_tensors; i++) {
        if (tensors[i]->data == nullptr && tensors[i]->view_src == nullptr) {
            total += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensors[i]), GGML_ANE_PAGE_SIZE);
        }
    }
    return total;
}

static size_t ggml_backend_ane_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

static bool ggml_backend_ane_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(ggml_backend_dev_t dev);

//
// program translation
//
// graph_compute splits the graph at host ops (SET_ROWS); each segment in between becomes one ANE program
// ggml tensors use the MIL shape [ne3, ne2, ne1, ne0], so that ggml reshapes map to MIL reshapes
// tensors not computed inside a segment (leafs, results of earlier segments, views of those) are program inputs
//

// program input "x<i>": copied from a tensor, or synthesized on the host (e.g. RoPE tables)
struct ggml_ane_input {
    ggml_tensor *      tensor = nullptr;
    std::function<void(float *)> gen;      // host-computed input (rope tables, masks): filled at every evaluation
    size_t             n_aux  = 0;
    bool               direct = false; // first rows of a tensor alone on its surface: bound without a copy when the strides match
};

struct ggml_ane_plan {
    ggml_ane_mil                mil;
    ggml_ane_blob               blob;
    std::vector<ggml_ane_input> inputs;
    std::vector<ggml_tensor *>  outputs;  // program output "y<i>"
    std::vector<ggml_tensor *>  baked;    // tensors whose data is embedded in the program
    std::vector<ggml_tensor *>  set_rows; // SET_ROWS nodes, applied on the host after the program
    std::vector<std::pair<const ggml_tensor *, std::vector<int32_t>>> guards; // data baked into the program text (get_rows ids)

    explicit ggml_ane_plan(bool dry) : blob(dry) {}
};

enum ggml_ane_weight_mode {
    GGML_ANE_WEIGHTS_AUTO, // quantized -> int8 per row, others -> fp16
    GGML_ANE_WEIGHTS_FP16,
    GGML_ANE_WEIGHTS_INT8,
};

static ggml_ane_weight_mode ggml_ane_get_weight_mode() {
    static const ggml_ane_weight_mode mode = []() {
        const char * s = getenv("GGML_ANE_WEIGHTS");
        if (s && strcmp(s, "fp16") == 0) return GGML_ANE_WEIGHTS_FP16;
        if (s && strcmp(s, "int8") == 0) return GGML_ANE_WEIGHTS_INT8;
        return GGML_ANE_WEIGHTS_AUTO;
    }();
    return mode;
}

// kv chunk size for flash attention: longer rows are processed in chunks (one fp16 softmax over thousands of entries
// loses precision)
static int64_t ggml_ane_fa_chunk() {
    return 1024;
}

static ggml_ane_shape ggml_ane_mil_shape(const ggml_tensor * t) {
    return { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
}

static void ggml_ane_row_to_f32(const ggml_tensor * t, int64_t row, float * dst) {
    const char * src = (const char *) t->data + row * t->nb[1];
    switch (t->type) {
        case GGML_TYPE_F32: memcpy(dst, src, t->ne[0] * sizeof(float)); break;
        case GGML_TYPE_F16: ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, t->ne[0]); break;
        default:            ggml_get_type_traits(t->type)->to_float(src, dst, t->ne[0]); break;
    }
}

static bool ggml_ane_can_bake(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) {
        return true;
    }
    return ggml_get_type_traits(t->type)->to_float != nullptr;
}

// conv weight [oc, ic, 1, 1] from input columns [k0, k0 + kn) of a 2D ggml weight [K, oc] (kn < 0: all)
static std::string ggml_ane_bake_weight(ggml_ane_plan & plan, ggml_tensor * w, int64_t k0 = 0, int64_t kn = -1) {
    const int64_t K  = w->ne[0];
    const int64_t ic = kn < 0 ? K : kn;
    const int64_t oc = w->ne[1];
    const std::string part = ic == K ? "" : ":" + std::to_string(k0) + "+" + std::to_string(ic);

    ggml_ane_weight_mode mode = ggml_ane_get_weight_mode();
    if (mode == GGML_ANE_WEIGHTS_AUTO) {
        mode = ggml_is_quantized(w->type) ? GGML_ANE_WEIGHTS_INT8 : GGML_ANE_WEIGHTS_FP16;
    }
    if (k0 == 0) {
        plan.baked.push_back(w);
    }

    if (mode == GGML_ANE_WEIGHTS_FP16) {
        std::vector<ggml_fp16_t> data;
        if (!plan.blob.dry) {
            data.resize(ic * oc);
            std::vector<float> row(K);
            for (int64_t r = 0; r < oc; r++) {
                ggml_ane_row_to_f32(w, r, row.data());
                ggml_fp32_to_fp16_row(row.data() + k0, data.data() + r * ic, ic);
            }
        }
        const std::string wkey = "w:" + ggml_ane_content_key(w) + part;
        const uint64_t off = plan.blob.add(data.data(), ic * oc * sizeof(ggml_fp16_t), ggml_ane_blob::DT_FP16, wkey + ":f16");
        return plan.mil.const_blob({ oc, ic, 1, 1 }, off);
    }

    // int8 with one fp16 scale per output row (the only granularity the ANE accepts)
    std::vector<int8_t>      q;
    std::vector<ggml_fp16_t> scale;
    if (!plan.blob.dry) {
        q.resize(ic * oc);
        scale.resize(oc);
        std::vector<float> full(K);
        for (int64_t r = 0; r < oc; r++) {
            ggml_ane_row_to_f32(w, r, full.data());
            const float * row = full.data() + k0;
            float amax = 0.0f;
            for (int64_t c = 0; c < ic; c++) {
                amax = std::max(amax, fabsf(row[c]));
            }
            const ggml_fp16_t s16 = ggml_fp32_to_fp16(amax / 127.0f);
            const float       s   = ggml_fp16_to_fp32(s16);
            const float       id  = s > 0.0f ? 1.0f / s : 0.0f;
            scale[r] = s16;
            for (int64_t c = 0; c < ic; c++) {
                q[r * ic + c] = (int8_t) std::max(-127.0f, std::min(127.0f, roundf(row[c] * id)));
            }
        }
    }
    const std::string wkey = "w:" + ggml_ane_content_key(w) + part;
    const uint64_t off_q = plan.blob.add(q.data(), ic * oc, ggml_ane_blob::DT_INT8, wkey + ":i8");
    const uint64_t off_s = plan.blob.add(scale.data(), oc * sizeof(ggml_fp16_t), ggml_ane_blob::DT_FP16, wkey + ":i8s");
    return plan.mil.const_int8_rows(oc, ic, off_q, off_s);
}

static bool ggml_ane_is_view_op(ggml_op op) {
    return op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool ggml_ane_rows_contiguous(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && t->nb[0] == ggml_type_size(t->type);
}

// KV length buckets: the attention of a program reads the first nb rows of the cache, nb = the smallest bucket >= the
// used length (fewer shapes = fewer program functions; a multi-function program holds ~5 functions at most).
// Default: powers of 2 from 256; ggml_backend_ane_set_kv_buckets() sets a fixed list (the last one should cover the cache).
static std::vector<int64_t> g_ane_kv_buckets;

void ggml_backend_ane_set_kv_buckets(const int * sizes, int n) {
    g_ane_kv_buckets.assign(sizes, sizes + n);
    std::sort(g_ane_kv_buckets.begin(), g_ane_kv_buckets.end());
}

static int64_t ggml_ane_kv_bucket(int64_t nkv) {
    for (int64_t b : g_ane_kv_buckets) {
        if (b >= nkv) {
            return b;
        }
    }
    return std::max<int64_t>(256, (int64_t) 1 << (64 - __builtin_clzll((unsigned long long) std::max<int64_t>(nkv - 1, 1))));
}

// translate nodes [i0, i1) of the graph into one program
static bool ggml_ane_translate(ggml_cgraph * cgraph, int i0, int i1, ggml_ane_plan & plan, std::string & err) {
    ggml_ane_mil & mil = plan.mil;
    std::unordered_map<const ggml_tensor *, ggml_ane_mil::value> vals;
    // KV cache writes (SET_ROWS) seen in this segment: cache tensor -> new rows and their slots
    struct pending_write {
        ggml_tensor * rows;
        ggml_tensor * idx;
    };
    std::unordered_map<const ggml_tensor *, pending_write> pending;
    std::unordered_map<const ggml_tensor *, bool> force_out;

    // uses of each tensor by nodes of this segment; graph-wide counts (cgraph->use_counts) include other splits
    std::unordered_map<const ggml_tensor *, int> seg_uses;
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (cgraph->nodes[i]->src[j]) {
                seg_uses[cgraph->nodes[i]->src[j]]++;
            }
        }
    }

    // value of a tensor as fp16; tensors not computed in this segment become program inputs (read with their strides)
    auto get = [&](ggml_tensor * t) -> ggml_ane_mil::value {
        auto it = vals.find(t);
        if (it != vals.end()) {
            return it->second;
        }
        GGML_ASSERT(ggml_ane_rows_contiguous(t));
        const std::string name = "x" + std::to_string(plan.inputs.size());
        ggml_ane_input in;
        in.tensor = t;
        plan.inputs.push_back(std::move(in));
        // program I/O is always fp16, f32 tensors are converted on the host (a cast to fp32 after concat fails to compile)
        const ggml_ane_mil::value v = mil.input(name, "fp16", ggml_ane_mil_shape(t));
        vals[t] = v;
        return v;
    };

    // host-computed input (fp32 on the host, fp16 in the program), generated at every evaluation from the current data;
    // tables with the same recipe (rope, masks of every layer) are passed once
    std::unordered_map<std::string, ggml_ane_mil::value> aux_vals;
    auto aux = [&](const ggml_ane_shape & shape, const std::string & recipe, std::function<void(float *)> gen) -> ggml_ane_mil::value {
        auto it = aux_vals.find(recipe);
        if (it != aux_vals.end()) {
            return it->second;
        }
        const std::string name = "x" + std::to_string(plan.inputs.size());
        ggml_ane_input in;
        in.gen   = std::move(gen);
        in.n_aux = (size_t) (shape[0] * shape[1] * shape[2] * shape[3]);
        plan.inputs.push_back(std::move(in));
        return aux_vals[recipe] = mil.input(name, "fp16", shape);
    };
    auto f16 = [&](float v) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.9g", v);
        std::string lit = buf;
        if (lit.find_first_of(".einn") == std::string::npos) {
            lit += ".0"; // "0" would not parse as a float literal
        }
        return mil.const_scalar("fp16", lit);
    };
    auto binop = [&](const char * op, const ggml_ane_mil::value & a, const std::string & b, const ggml_ane_shape & shape) {
        return mil.op("fp16", shape, std::string(op) + "(x=" + a.name + ",y=" + b + ")");
    };
    auto slice = [&](const ggml_ane_mil::value & x, const ggml_ane_shape & begin, const ggml_ane_shape & size) {
        return mil.op("fp16", size, "slice_by_size(x=" + x.name + ",begin=" + mil.const_i32(begin) + ",size=" + mil.const_i32(size) + ")");
    };

    for (int i = i0; i < i1; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->op == GGML_OP_NONE) {
            continue;
        }
        if (ggml_ane_is_view_op(node->op) && vals.find(node->src[0]) == vals.end()) {
            continue; // view of an input: becomes an input itself when used
        }
        const ggml_ane_shape shape = ggml_ane_mil_shape(node);
        switch (node->op) {
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE: {
                // ggml dim d of the source goes to dim axis[d]; MIL axis k is ggml dim 3 - k
                int axis[4] = { 1, 0, 2, 3 };
                if (node->op == GGML_OP_PERMUTE) {
                    for (int d = 0; d < 4; d++) {
                        axis[d] = ggml_get_op_params_i32(node, d);
                    }
                }
                std::vector<int64_t> perm(4);
                for (int d = 0; d < 4; d++) {
                    perm[3 - axis[d]] = 3 - d;
                }
                const ggml_ane_mil::value src = get(node->src[0]);
                vals[node] = mil.op("fp16", shape, "transpose(perm=" + mil.const_i32(perm) + ",x=" + src.name + ")");
            } break;
            case GGML_OP_SET_ROWS: {
                // the cache is written on the host after the program; attention later in this program uses the new rows directly
                ggml_tensor * base = node->view_src ? node->view_src : node;
                pending[base] = { node->src[0], node->src[1] };
                plan.set_rows.push_back(node);
                if (vals.find(node->src[0]) != vals.end()) {
                    force_out[node->src[0]] = true;
                }
            } break;
            case GGML_OP_CONT: {
                // cont can also change the shape (ggml_cont_2d etc.)
                const ggml_ane_mil::value src = get(node->src[0]);
                vals[node] = src.shape == shape ? src : mil.op("fp16", shape, "reshape(shape=" + mil.const_i32(shape) + ",x=" + src.name + ")");
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                // q [1, H, nq, D], k [1, Hkv, nkv, D], v [1, Hkv, nkv, Dv] -> out [1, nq, H, Dv]
                // q head h uses kv head h / g: q is regrouped to [1, Hkv, g*nq, D] (no K/V repetition). The kv axis is a
                // list of parts (cache chunks, rows written by SET_ROWS in this segment) so the cache is never copied.
                ggml_tensor * q = node->src[0];
                ggml_tensor * k = node->src[1];
                ggml_tensor * v = node->src[2];
                ggml_tensor * m = node->src[3];
                const float   scale = ggml_get_op_params_f32(node, 0);
                const int64_t H   = q->ne[2];
                const int64_t nq  = q->ne[1];
                const int64_t nkv = k->ne[1];
                const int64_t Hkv = k->ne[2];
                const int64_t g   = H / Hkv;
                const int64_t Dk  = k->ne[0];
                const int64_t Dv  = v->ne[0];
                const int64_t gq  = g * nq;
                if (m && (!m->buffer || !ggml_backend_buffer_is_host(m->buffer))) {
                    err = "flash attention: mask must be in host memory";
                    return false;
                }

                struct part { ggml_ane_mil::value k, v; int64_t n; };
                std::vector<part> parts;

                // KV cache: read the first nb rows of the cache tensor directly (bound without a copy), nb = nkv rounded up
                // to a power of 2 so that a growing context recompiles rarely; the rows past nkv are masked
                int64_t nb = ggml_ane_kv_bucket(nkv);
                auto direct_ok = [&](const ggml_tensor * t, int64_t D) {
                    const ggml_tensor * b = t->view_src;
                    return b && t->view_offs == 0 && b->type == GGML_TYPE_F16 && b->buffer && ggml_backend_buffer_is_host(b->buffer) &&
                           ((uintptr_t) b->data) % (uintptr_t) sysconf(_SC_PAGESIZE) == 0 && ggml_is_contiguous(b) &&
                           b->ne[0] == Hkv * D && t->nb[0] == 2 && t->nb[1] == b->nb[1] && t->nb[2] == (size_t) D * 2 && t->ne[3] == 1;
                };
                const bool direct = direct_ok(k, Dk) && direct_ok(v, Dv) && k->view_src->ne[1] == v->view_src->ne[1];
                if (direct) {
                    nb = std::min(nb, k->view_src->ne[1]);
                    if (nb > GGML_ANE_MAX_DIM) {
                        nb = nkv;
                    }
                }
                auto cache_value = [&](ggml_tensor * t, int64_t D) -> ggml_ane_mil::value {
                    if (!direct || nb > GGML_ANE_MAX_DIM) {
                        return get(t);
                    }
                    const std::string name = "x" + std::to_string(plan.inputs.size());
                    ggml_ane_input in;
                    in.tensor = t->view_src;
                    in.direct = true;
                    plan.inputs.push_back(std::move(in));
                    const auto r0 = mil.input(name, "fp16", { 1, 1, nb, Hkv * D });
                    const auto r4 = mil.op("fp16", { 1, nb, Hkv, D }, "reshape(shape=" + mil.const_i32({ 1, nb, Hkv, D }) + ",x=" + r0.name + ")");
                    return mil.op("fp16", { 1, Hkv, nb, D }, "transpose(perm=" + mil.const_i32({ 0, 2, 1, 3 }) + ",x=" + r4.name + ")");
                };
                const bool use_direct = direct && nb <= GGML_ANE_MAX_DIM;
                const int64_t ncache = use_direct ? nb : nkv;
                const ggml_ane_mil::value kc = cache_value(k, Dk);
                const ggml_ane_mil::value vc = cache_value(v, Dv);
                const int64_t chunk = ggml_ane_fa_chunk() > 0 ? ggml_ane_fa_chunk() : ncache;
                for (int64_t c0 = 0; c0 < ncache; c0 += chunk) {
                    const int64_t n = std::min(chunk, ncache - c0);
                    if (n == ncache) {
                        parts.push_back({ kc, vc, n });
                    } else {
                        parts.push_back({ slice(kc, { 0, 0, c0, 0 }, { 1, Hkv, n, Dk }), slice(vc, { 0, 0, c0, 0 }, { 1, Hkv, n, Dv }), n });
                    }
                }

                // mask [1, 1, g*nq, ntot] built on the host (identical for every layer, so passed once)
                const ggml_tensor * slot_idx = nullptr;
                int64_t n_slots = 0;
                auto pk = pending.find(k->view_src ? k->view_src : k);
                auto pv = pending.find(v->view_src ? v->view_src : v);
                if (pk != pending.end() || pv != pending.end()) {
                    // the cache rows at the written slots are stale here: add the new rows as a part and fix the mask
                    if (pk == pending.end() || pv == pending.end()) {
                        err = "flash attention: only one of K/V written in this segment";
                        return false;
                    }
                    ggml_tensor * rk  = pk->second.rows;
                    ggml_tensor * rv  = pv->second.rows;
                    ggml_tensor * idx = pk->second.idx;
                    const int64_t S = rk->ne[1];
                    if (rk->ne[0] != Hkv * Dk || rv->ne[0] != Hkv * Dv || rv->ne[1] != S || k->view_offs != 0 || v->view_offs != 0 ||
                        idx->ne[0] != S || idx->ne[1] != 1 || ggml_nelements(pv->second.idx) != S || !idx->buffer || !ggml_backend_buffer_is_host(idx->buffer)) {
                        err = "flash attention: unsupported KV write layout";
                        return false;
                    }
                    auto new_rows = [&](ggml_tensor * r, int64_t D) {
                        const ggml_ane_mil::value r0 = get(r);
                        const auto r4 = mil.op("fp16", { 1, S, Hkv, D }, "reshape(shape=" + mil.const_i32({ 1, S, Hkv, D }) + ",x=" + r0.name + ")");
                        return mil.op("fp16", { 1, Hkv, S, D }, "transpose(perm=" + mil.const_i32({ 0, 2, 1, 3 }) + ",x=" + r4.name + ")");
                    };
                    parts.push_back({ new_rows(rk, Dk), new_rows(rv, Dv), S });
                    slot_idx = idx;
                    n_slots  = S;
                }
                const int64_t ntot = ncache + n_slots;
                const bool has_mask = m || n_slots > 0 || ncache > nkv;
                ggml_ane_mil::value mv;
                if (has_mask) {
                    char recipe[160];
                    snprintf(recipe, sizeof(recipe), "mask:%p:%p:%lld:%lld:%lld:%lld:%lld", (const void *) m, (const void *) slot_idx,
                        (long long) nkv, (long long) ncache, (long long) nq, (long long) g, (long long) n_slots);
                    const ggml_tensor * sidx = slot_idx;
                    mv = aux({ 1, 1, gq, ntot }, recipe, [=](float * md) {
                        std::vector<int64_t> slots(n_slots);
                        for (int64_t s2 = 0; s2 < n_slots; s2++) {
                            const char * ip = (const char *) sidx->data + s2 * sidx->nb[0];
                            slots[s2] = sidx->type == GGML_TYPE_I64 ? *(const int64_t *) ip : *(const int32_t *) ip;
                        }
                        std::vector<float> row(nkv, 0.0f);
                        for (int64_t i = 0; i < nq; i++) {
                            if (m) {
                                ggml_fp16_to_fp32_row((const ggml_fp16_t *) ((const char *) m->data + i * m->nb[1]), row.data(), nkv);
                            }
                            float * dst = md + i * ntot;
                            memcpy(dst, row.data(), nkv * sizeof(float));
                            std::fill(dst + nkv, dst + ncache, -INFINITY);
                            for (int64_t s2 = 0; s2 < n_slots; s2++) {
                                dst[ncache + s2] = slots[s2] < nkv ? row[slots[s2]] : -INFINITY;
                            }
                            for (int64_t s2 = 0; s2 < n_slots; s2++) {
                                if (slots[s2] < nkv) {
                                    dst[slots[s2]] = -INFINITY;
                                }
                            }
                        }
                        for (int64_t r = 1; r < g; r++) {
                            memcpy(md + r * nq * ntot, md, nq * ntot * sizeof(float));
                        }
                    });
                }

                const std::string bf = mil.const_scalar("bool", "false");
                const std::string bt = mil.const_scalar("bool", "true");
                const ggml_ane_mil::value qv0 = get(q);
                const auto qv = mil.op("fp16", { 1, Hkv, gq, Dk }, "reshape(shape=" + mil.const_i32({ 1, Hkv, gq, Dk }) + ",x=" + qv0.name + ")");
                const ggml_ane_shape oshape = { 1, Hkv, gq, Dv };
                auto scores = [&](const part & p) {
                    return mil.op("fp16", { 1, Hkv, gq, p.n }, "matmul(transpose_x=" + bf + ",transpose_y=" + bt + ",x=" + qv.name + ",y=" + p.k.name + ")");
                };
                ggml_ane_mil::value o;
                if (ntot <= chunk) {
                    // one softmax over the concatenated scores of all parts
                    const ggml_ane_shape sshape = { 1, Hkv, gq, ntot };
                    ggml_ane_mil::value sc;
                    if (parts.size() == 1) {
                        sc = scores(parts[0]);
                    } else {
                        std::string list;
                        for (const auto & p : parts) {
                            list += (list.empty() ? "" : ",") + scores(p).name;
                        }
                        sc = mil.op("fp16", sshape, "concat(axis=" + mil.const_scalar("int32", "3") + ",interleave=" + bf + ",values=(" + list + "))");
                    }
                    sc = binop("mul", sc, f16(scale), sshape);
                    if (has_mask) {
                        sc = binop("add", sc, mv.name, sshape);
                    }
                    const auto pr = mil.op("fp16", sshape, "softmax(axis=" + mil.const_scalar("int32", "-1") + ",x=" + sc.name + ")");
                    int64_t c0 = 0;
                    for (const auto & p : parts) {
                        const auto pp = parts.size() == 1 ? pr : slice(pr, { 0, 0, 0, c0 }, { 1, Hkv, gq, p.n });
                        const auto op = mil.op("fp16", oshape, "matmul(transpose_x=" + bf + ",transpose_y=" + bf + ",x=" + pp.name + ",y=" + p.v.name + ")");
                        o = c0 == 0 ? op : binop("add", o, op.name, oshape);
                        c0 += p.n;
                    }
                } else {
                    // long kv: each part gets its own max / sum, then they are combined (fp16 sums over long rows lose precision)
                    const std::string ax = mil.const_i32({ 3 });
                    const std::string kd = mil.const_scalar("bool", "true");
                    const ggml_ane_shape rshape = { 1, Hkv, gq, 1 };
                    std::vector<ggml_ane_mil::value> cm, cl, co;
                    int64_t c0 = 0;
                    for (const auto & p : parts) {
                        const ggml_ane_shape sshape = { 1, Hkv, gq, p.n };
                        auto sc = binop("mul", scores(p), f16(scale), sshape);
                        if (has_mask) {
                            sc = binop("add", sc, slice(mv, { 0, 0, 0, c0 }, { 1, 1, gq, p.n }).name, sshape);
                        }
                        const auto mx = mil.op("fp16", rshape, "reduce_max(x=" + sc.name + ",axes=" + ax + ",keep_dims=" + kd + ")");
                        const auto ex = mil.op("fp16", sshape, "exp(x=" + binop("sub", sc, mx.name, sshape).name + ")");
                        std::vector<ggml_fp16_t> ones;
                        if (!plan.blob.dry) {
                            ones.assign(p.n, ggml_fp32_to_fp16(1.0f));
                        }
                        const std::string ov = mil.const_blob({ p.n, 1 }, plan.blob.add(ones.data(), p.n * sizeof(ggml_fp16_t), ggml_ane_blob::DT_FP16, "ones:" + std::to_string(p.n)));
                        cl.push_back(mil.op("fp16", rshape, "matmul(transpose_x=" + bf + ",transpose_y=" + bf + ",x=" + ex.name + ",y=" + ov + ")"));
                        co.push_back(mil.op("fp16", oshape, "matmul(transpose_x=" + bf + ",transpose_y=" + bf + ",x=" + ex.name + ",y=" + p.v.name + ")"));
                        cm.push_back(mx);
                        c0 += p.n;
                    }
                    ggml_ane_mil::value M = cm[0];
                    for (size_t c = 1; c < cm.size(); c++) {
                        M = binop("maximum", M, cm[c].name, rshape);
                    }
                    ggml_ane_mil::value num, den;
                    for (size_t c = 0; c < cm.size(); c++) {
                        const auto w  = mil.op("fp16", rshape, "exp(x=" + binop("sub", cm[c], M.name, rshape).name + ")");
                        const auto wo = binop("mul", co[c], w.name, oshape);
                        const auto wl = binop("mul", cl[c], w.name, rshape);
                        num = c == 0 ? wo : binop("add", num, wo.name, oshape);
                        den = c == 0 ? wl : binop("add", den, wl.name, rshape);
                    }
                    o = binop("real_div", num, den.name, oshape);
                }
                const auto o4 = mil.op("fp16", { 1, H, nq, Dv }, "reshape(shape=" + mil.const_i32({ 1, H, nq, Dv }) + ",x=" + o.name + ")");
                vals[node] = mil.op("fp16", shape, "transpose(perm=" + mil.const_i32({ 0, 2, 1, 3 }) + ",x=" + o4.name + ")");
            } break;
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW: {
                if (!ggml_is_contiguous(node) || node->view_offs != 0) {
                    err = std::string("unsupported view ") + node->name;
                    return false;
                }
                const ggml_ane_mil::value src = get(node->src[0]);
                vals[node] = mil.op("fp16", ggml_ane_mil_shape(node), "reshape(shape=" + mil.const_i32(ggml_ane_mil_shape(node)) + ",x=" + src.name + ")");
            } break;
            case GGML_OP_MUL_MAT: {
                // [1, 1, M, K] -> [1, K, 1, M] -> conv -> [1, N, 1, M] -> [1, 1, M, N]
                ggml_tensor * w = node->src[0];
                const int64_t K = w->ne[0];
                const int64_t N = w->ne[1];
                const int64_t M = node->src[1]->ne[1];
                const ggml_ane_mil::value x = get(node->src[1]);
                // the conv accumulates in reduced precision: partial sums can overflow fp16 even when the result fits
                // (e.g. Qwen3 FFN down projection on special tokens). Scale the input down and the output back up.
                const float pre = 16.0f;
                const ggml_ane_mil::value xs = binop("mul", x, f16(1.0f / pre), x.shape);
                const auto xt = mil.op("fp16", { 1, K, 1, M }, "transpose(perm=" + mil.const_i32({ 0, 3, 1, 2 }) + ",x=" + xs.name + ")");
                // convs with more than 4096 input channels fail to compile for larger batches (e.g. Llama-3.2-1B FFN
                // down, 8192 -> 2048 at 128 tokens): sum convs over input slices of at most 4096 channels
                const int64_t kc = 4096;
                ggml_ane_mil::value y;
                for (int64_t k0 = 0; k0 < K; k0 += kc) {
                    const int64_t kn = std::min(kc, K - k0);
                    const std::string wname = ggml_ane_bake_weight(plan, w, k0, kn == K ? -1 : kn);
                    const ggml_ane_mil::value xk = kn == K ? xt : slice(xt, { 0, k0, 0, 0 }, { 1, kn, 1, M });
                    const auto yk = mil.op("fp16", { 1, N, 1, M },
                        "conv(dilations=" + mil.const_i32({ 1, 1 }) + ",groups=" + mil.const_scalar("int32", "1") +
                        ",pad=" + mil.const_i32({ 0, 0, 0, 0 }) + ",pad_type=" + mil.const_scalar("string", "\"valid\"") +
                        ",strides=" + mil.const_i32({ 1, 1 }) + ",weight=" + wname + ",x=" + xk.name + ")");
                    y = k0 == 0 ? yk : binop("add", y, yk.name, { 1, N, 1, M });
                }
                y = binop("mul", y, f16(pre), y.shape);
                vals[node] = mil.op("fp16", ggml_ane_mil_shape(node), "transpose(perm=" + mil.const_i32({ 0, 2, 3, 1 }) + ",x=" + y.name + ")");
            } break;
            case GGML_OP_ADD:
            case GGML_OP_MUL: {
                const ggml_ane_mil::value a = get(node->src[0]);
                const ggml_ane_mil::value b = get(node->src[1]);
                vals[node] = binop(node->op == GGML_OP_ADD ? "add" : "mul", a, b.name, shape);
            } break;
            case GGML_OP_RMS_NORM: {
                // x / sqrt(mean(x^2) + eps) == xs / sqrt(mean(xs^2) + eps*c^2) with xs = c*x
                // c = 1/max(max|x|, 1e-2) keeps x^2 in fp16 range (activations can reach ~7000)
                const float eps = ggml_get_op_params_f32(node, 0);
                const ggml_ane_mil::value x = get(node->src[0]);
                ggml_ane_shape rshape = shape;
                rshape[3] = 1;
                const std::string axes = mil.const_i32({ 3 });
                const std::string kd   = mil.const_scalar("bool", "true");
                const auto ax  = mil.op("fp16", shape, "abs(x=" + x.name + ")");
                const auto mx  = mil.op("fp16", rshape, "reduce_max(x=" + ax.name + ",axes=" + axes + ",keep_dims=" + kd + ")");
                const auto mxc = binop("maximum", mx, f16(1e-2f), rshape);
                const auto c   = mil.op("fp16", rshape, "real_div(x=" + f16(1.0f) + ",y=" + mxc.name + ")");
                const auto xs  = binop("mul", x, c.name, shape);
                const auto sq  = binop("mul", xs, xs.name, shape);
                // sum of squares as a matmul with ones: reduce ops accumulate in fp16 and lose small terms
                const int64_t n = shape[3];
                std::vector<ggml_fp16_t> ones;
                if (!plan.blob.dry) {
                    ones.assign(n, ggml_fp32_to_fp16(1.0f));
                }
                const std::string ov = mil.const_blob({ n, 1 }, plan.blob.add(ones.data(), n * sizeof(ggml_fp16_t), ggml_ane_blob::DT_FP16, "ones:" + std::to_string(n)));
                const std::string bf = mil.const_scalar("bool", "false");
                const auto ss  = mil.op("fp16", rshape, "matmul(transpose_x=" + bf + ",transpose_y=" + bf + ",x=" + sq.name + ",y=" + ov + ")");
                const auto ms  = binop("mul", ss, f16(1.0f / n), rshape);
                ggml_ane_mil::value den = ms;
                if (eps > 0.0f) {
                    // a mul by a constant 0 gives wrong results on the ANE, so skip the eps term entirely when eps == 0
                    const auto c2  = binop("mul", c, c.name, rshape);
                    const auto ec2 = binop("mul", c2, f16(eps), rshape);
                    den = binop("add", ms, ec2.name, rshape);
                }
                const auto sd  = mil.op("fp16", rshape, "sqrt(x=" + den.name + ")");
                vals[node] = binop("real_div", xs, sd.name, shape);
            } break;
            case GGML_OP_GLU: {
                // swiglu: silu(a) * b, with a/b either two tensors or the two halves of src0
                const bool swapped = ggml_get_op_params_i32(node, 1) != 0;
                ggml_ane_mil::value a, b;
                if (node->src[1]) {
                    a = get(node->src[0]);
                    b = get(node->src[1]);
                } else {
                    const ggml_ane_mil::value x = get(node->src[0]);
                    const int64_t nc = node->ne[0];
                    const ggml_ane_shape hs = { shape[0], shape[1], shape[2], nc };
                    a = slice(x, { 0, 0, 0, swapped ? nc : 0 }, hs);
                    b = slice(x, { 0, 0, 0, swapped ? 0 : nc }, hs);
                }
                const auto act = mil.op("fp16", shape, "silu(x=" + a.name + ")");
                vals[node] = binop("mul", act, b.name, shape);
            } break;
            case GGML_OP_GET_ROWS: {
                // the row ids are known now (host memory): slices of the source, baked into the program
                const ggml_tensor * ids = node->src[1];
                if (!ids->buffer || !ggml_backend_buffer_is_host(ids->buffer)) {
                    err = "get_rows ids must be in host memory";
                    return false;
                }
                const ggml_ane_mil::value x = get(node->src[0]);
                const int64_t n = ids->ne[0];
                plan.guards.push_back({ ids, std::vector<int32_t>((const int32_t *) ids->data, (const int32_t *) ids->data + n) });
                std::vector<std::string> parts;
                for (int64_t r = 0; r < n; ) {
                    const int32_t a = ((const int32_t *) ids->data)[r];
                    if (a < 0 || a >= node->src[0]->ne[1]) {
                        err = "get_rows: id out of range";
                        return false;
                    }
                    int64_t e = r + 1;
                    while (e < n && ((const int32_t *) ids->data)[e] == a + (e - r)) {
                        e++;
                    }
                    parts.push_back(slice(x, { 0, 0, a, 0 }, { 1, 1, e - r, node->ne[0] }).name);
                    r = e;
                }
                if (parts.size() == 1) {
                    vals[node] = mil.op("fp16", shape, "identity(x=" + parts[0] + ")");
                } else {
                    std::string list;
                    for (const auto & p2 : parts) {
                        list += (list.empty() ? "" : ",") + p2;
                    }
                    vals[node] = mil.op("fp16", shape, "concat(axis=" + mil.const_scalar("int32", "2") + ",interleave=" + mil.const_scalar("bool", "false") + ",values=(" + list + "))");
                }
            } break;
            case GGML_OP_ROPE: {
                // NEOX rope: rotate (x[i], x[i + n_dims/2]); cos/sin tables are computed here from the positions.
                // NORMAL rope rotates (x[2i], x[2i+1]): a 0/1 permutation matmul moves even dims to the first half and odd
                // dims to the second, the NEOX rotation is applied, and the transposed permutation restores the order.
                const ggml_tensor * pos = node->src[1];
                const ggml_tensor * ff  = node->src[2]; // frequency factors (e.g. Llama 3), divide the angle of each pair
                const bool normal       = ggml_get_op_params_i32(node, 2) == GGML_ROPE_TYPE_NORMAL;
                const int   n_dims      = ggml_get_op_params_i32(node, 1);
                const float freq_base   = ggml_get_op_params_f32(node, 5);
                const float freq_scale  = ggml_get_op_params_f32(node, 6);
                const float attn_factor = ggml_get_op_params_f32(node, 8);
                const int64_t n_tok = node->ne[2];
                const int64_t half  = n_dims / 2;
                if (!pos->buffer || !ggml_backend_buffer_is_host(pos->buffer) || (ff && (!ff->buffer || !ggml_backend_buffer_is_host(ff->buffer)))) {
                    err = "rope positions and frequency factors must be in host memory";
                    return false;
                }
                const float theta_scale = powf(freq_base, -2.0f / n_dims);
                auto table = [=](bool is_sin) {
                    return [=](float * dst) {
                        for (int64_t t = 0; t < n_tok; t++) {
                            float theta = (float) ((const int32_t *) pos->data)[t];
                            for (int64_t j = 0; j < half; j++) {
                                const float th = ff ? theta / ((const float *) ff->data)[j] : theta;
                                dst[t * half + j] = (is_sin ? sinf(freq_scale * th) : cosf(freq_scale * th)) * attn_factor;
                                theta *= theta_scale;
                            }
                        }
                    };
                };
                char recipe[160];
                snprintf(recipe, sizeof(recipe), "rope:%p:%p:%lld:%d:%a:%a:%a", (const void *) pos, (const void *) ff, (long long) n_tok, n_dims, freq_base, freq_scale, attn_factor);
                ggml_ane_mil::value x = get(node->src[0]);
                const int64_t D = node->ne[0];
                std::string perm;
                if (normal) {
                    std::vector<ggml_fp16_t> pm;
                    if (!plan.blob.dry) {
                        pm.assign(D * D, ggml_fp32_to_fp16(0.0f));
                        for (int64_t r = 0; r < D; r++) {
                            const int64_t c = r >= n_dims ? r : (r % 2 == 0 ? r / 2 : half + r / 2); // row r of x goes to column c
                            pm[r * D + c] = ggml_fp32_to_fp16(1.0f);
                        }
                    }
                    perm = mil.const_blob({ D, D }, plan.blob.add(pm.data(), D * D * sizeof(ggml_fp16_t), ggml_ane_blob::DT_FP16,
                        "perm:" + std::to_string(D) + ":" + std::to_string(n_dims)));
                    x = mil.op("fp16", shape, "matmul(transpose_x=" + mil.const_scalar("bool", "false") + ",transpose_y=" + mil.const_scalar("bool", "false") + ",x=" + x.name + ",y=" + perm + ")");
                }
                const ggml_ane_shape tshape = { 1, n_tok, 1, half };
                const auto c = aux(tshape, std::string(recipe) + ":c", table(false));
                const auto s = aux(tshape, std::string(recipe) + ":s", table(true));
                const ggml_ane_shape hs = { shape[0], shape[1], shape[2], half };
                const auto x1 = slice(x, { 0, 0, 0, 0 },    hs);
                const auto x2 = slice(x, { 0, 0, 0, half }, hs);
                const auto o1 = binop("sub", binop("mul", x1, c.name, hs), binop("mul", x2, s.name, hs).name, hs);
                const auto o2 = binop("add", binop("mul", x2, c.name, hs), binop("mul", x1, s.name, hs).name, hs);
                std::string parts = o1.name + "," + o2.name;
                if (n_dims < node->ne[0]) {
                    parts += "," + slice(x, { 0, 0, 0, n_dims }, { shape[0], shape[1], shape[2], node->ne[0] - n_dims }).name;
                }
                vals[node] = mil.op("fp16", shape, "concat(axis=" + mil.const_scalar("int32", "3") + ",interleave=" + mil.const_scalar("bool", "false") + ",values=(" + parts + "))");
                if (normal) {
                    vals[node] = mil.op("fp16", shape, "matmul(transpose_x=" + mil.const_scalar("bool", "false") + ",transpose_y=" + mil.const_scalar("bool", "true") + ",x=" + vals[node].name + ",y=" + perm + ")");
                }
            } break;
            default:
                err = std::string("unsupported op ") + ggml_op_desc(node);
                return false;
        }
    }

    // outputs: computed tensors that are used anywhere outside this segment (later segments, other splits), or not used at all
    for (int i = i0; i < i1; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        auto it = vals.find(node);
        if (it == vals.end()) {
            continue;
        }
        const int32_t n_uses = ggml_node_get_use_count(cgraph, i);
        if (!(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !force_out[node] && n_uses > 0 && seg_uses[node] >= n_uses) {
            continue;
        }
        const std::string name = "y" + std::to_string(plan.outputs.size());
        plan.outputs.push_back(node);
        mil.output(mil.op("fp16", it->second.shape, "identity(x=" + it->second.name + ")", name));
    }
    return true;
}

//
// backend
//

// all programs (shapes) of a segment that bake the same weights: compiled together as one multi-function program
// with a single weight blob, so the OS keeps one copy of the weights however many shapes are used
struct ggml_ane_family {
    ggml_ane_program *                        prog    = nullptr;
    uint64_t                                  version = 0;  // bumped when recompiled with more functions
    uint64_t                                  last_used = 0;
    size_t                                    n_compiled = 0; // functions in prog (fewer than funcs: recompile needed)
    std::vector<std::string>                  funcs;        // function texts, offsets into the family blob
    std::unordered_map<std::string, uint64_t> chunk_off;    // blob chunk key -> offset
    // the blob lives in an unlinked temporary file mapping (clean pages, not counted in the app footprint)
    int       fd   = -1;
    uint8_t * map  = nullptr;
    size_t    size = 0;

    ~ggml_ane_family() {
        ggml_ane_program_free(prog);
        if (map) {
            munmap(map, size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

struct ggml_ane_program_entry {
    ggml_ane_program *        prog = nullptr;
    int                       proc = 0;           // procedure (function) of prog
    std::shared_ptr<ggml_ane_family> fam;         // owner of prog (null in purge-only mode)
    std::string               fname;              // function name in the family program
    uint64_t                  fam_version = 0;
    uint64_t                  last_used = 0;
    std::vector<int>          in_map;   // program input  -> plan input index
    std::vector<int>          out_map;  // program output -> plan output index
    std::vector<IOSurfaceRef> in_surf;
    std::vector<IOSurfaceRef> out_surf;
    bool                      no_windows = false; // bind copies instead of memory windows

    ~ggml_ane_program_entry() {
        for (IOSurfaceRef s : in_surf)  CFRelease(s);
        for (IOSurfaceRef s : out_surf) CFRelease(s);
    }

    // point at the current version of the family program; build the I/O maps on first use
    bool prepare() {
        refresh();
        if (!prog || proc < 0) {
            return false;
        }
        if (in_map.empty() && out_map.empty()) {
            for (int i = 0; i < ggml_ane_program_n_inputs(prog, proc); i++) {
                const ggml_ane_io_desc * d = ggml_ane_program_input(prog, proc, i);
                in_map.push_back(d->name[0] == 'x' ? atoi(d->name + 1) : -1);
                in_surf.push_back(ggml_ane_surface_create(d->batch_stride * d->n));
            }
            for (int i = 0; i < ggml_ane_program_n_outputs(prog, proc); i++) {
                const ggml_ane_io_desc * d = ggml_ane_program_output(prog, proc, i);
                out_map.push_back(d->name[0] == 'y' ? atoi(d->name + 1) : -1);
                out_surf.push_back(ggml_ane_surface_create(d->batch_stride * d->n));
            }
        }
        return true;
    }

    void refresh() {
        if (fam && fam_version != fam->version) {
            prog        = fam->prog;
            proc        = ggml_ane_program_procedure(prog, fname.c_str());
            fam_version = fam->version;
        }
    }
};

// translated segment, reused while the graph nodes are unchanged (llama.cpp reuses graphs between tokens)
struct ggml_ane_segment_entry {
    std::unique_ptr<ggml_ane_plan> plan;
    struct ggml_ane_program_entry * prog = nullptr; // nullptr: nothing to run on the ANE
    uint64_t                       evictions = 0;  // program cache evictions when this entry was made
};

struct ggml_backend_ane_context {
    std::unordered_map<std::string, std::unique_ptr<ggml_ane_program_entry>> programs; // by plan key (one per shape)
    std::unordered_map<std::string, std::shared_ptr<ggml_ane_family>>        families; // by baked weights
    std::unordered_map<uint64_t, ggml_ane_segment_entry> segments; // by ggml_ane_segment_fingerprint
    uint64_t n_calls     = 0;
    uint64_t n_evictions = 0;
};

static std::string ggml_ane_plan_key(const ggml_ane_plan & plan) {
    std::string key = plan.mil.text();
    for (const ggml_tensor * t : plan.baked) {
        key += "|" + ggml_ane_content_key(t);
    }
    return key;
}

// debug: write a program that failed to compile to $GGML_ANE_DUMP_DIR/<n>/{model.mil,weights/weight.bin}
static void ggml_ane_dump_program(const std::string & mil, const std::vector<uint8_t> & weights) {
    static int n_dumped = 0;
    const char * dir = getenv("GGML_ANE_DUMP_DIR");
    if (!dir) {
        return;
    }
    const std::string base = std::string(dir) + "/" + std::to_string(n_dumped++);
    mkdir(dir, 0755);
    mkdir(base.c_str(), 0755);
    if (mkdir((base + "/weights").c_str(), 0755) != 0) {
        return;
    }
    FILE * f = fopen((base + "/model.mil").c_str(), "wb");
    if (f) { fwrite(mil.data(), 1, mil.size(), f); fclose(f); }
    f = fopen((base + "/weights/weight.bin").c_str(), "wb");
    if (f) { fwrite(weights.data(), 1, weights.size(), f); fclose(f); }
    GGML_LOG_INFO("%s: wrote %s\n", __func__, base.c_str());
}

// add the function of a fully translated plan to a family: blob chunks with known keys are reused, the others are
// appended to the family blob (a new temporary file mapping), and the offsets in the function text are remapped
static bool ggml_ane_family_add(ggml_ane_family & fam, const ggml_ane_plan & full, const std::string & fname, std::string & err) {
    const std::vector<uint8_t> & src = full.blob.data;
    std::unordered_map<uint64_t, uint64_t> remap; // offset in the plan blob -> offset in the family blob
    std::vector<std::pair<uint64_t, uint64_t>> appended; // (plan offset, length incl. header)
    size_t new_size = fam.size ? fam.size : 64;
    for (size_t c = 0; c < full.blob.chunks.size(); c++) {
        const auto & [ckey, pos] = full.blob.chunks[c];
        const uint64_t next = c + 1 < full.blob.chunks.size() ? full.blob.chunks[c + 1].second : src.size();
        auto it = ckey.empty() ? fam.chunk_off.end() : fam.chunk_off.find(ckey);
        if (it != fam.chunk_off.end()) {
            remap[pos] = it->second;
        } else {
            remap[pos] = new_size;
            appended.push_back({ pos, next - pos });
            new_size += next - pos;
        }
    }
    // new blob file: old content + appended chunks, chunk count updated
    char path[1024];
    snprintf(path, sizeof(path), "%s/ggml-ane-fam-XXXXXX", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    const int fd = mkstemp(path);
    if (fd < 0) {
        err = "cannot create a temporary file for the weight blob";
        return false;
    }
    unlink(path);
    std::vector<uint8_t> head(64, 0);
    if (fam.map) {
        memcpy(head.data(), fam.map, 64);
    } else {
        const uint32_t ver = 2;
        memcpy(head.data() + 4, &ver, 4);
    }
    uint32_t n_chunks;
    memcpy(&n_chunks, head.data(), 4);
    n_chunks += (uint32_t) appended.size();
    memcpy(head.data(), &n_chunks, 4);
    bool ok = write(fd, head.data(), 64) == 64;
    if (ok && fam.map && fam.size > 64) {
        ok = write(fd, fam.map + 64, fam.size - 64) == (ssize_t) (fam.size - 64);
    }
    for (const auto & [pos, len] : appended) {
        // chunk header holds the absolute data offset: rewrite it for the new position
        std::vector<uint8_t> chunk(src.begin() + pos, src.begin() + pos + len);
        const uint64_t data_off = remap[pos] + 64;
        memcpy(chunk.data() + 16, &data_off, 8);
        ok = ok && write(fd, chunk.data(), len) == (ssize_t) len;
    }
    void * m = ok ? mmap(nullptr, new_size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    if (m == MAP_FAILED) {
        close(fd);
        err = "cannot write the weight blob";
        return false;
    }
    if (fam.map) {
        munmap(fam.map, fam.size);
        close(fam.fd);
    }
    fam.fd   = fd;
    fam.map  = (uint8_t *) m;
    fam.size = new_size;
    for (size_t c = 0; c < full.blob.chunks.size(); c++) {
        if (!full.blob.chunks[c].first.empty()) {
            fam.chunk_off[full.blob.chunks[c].first] = remap[full.blob.chunks[c].second];
        }
    }
    // function text with remapped BLOBFILE offsets
    const std::string f = full.mil.func_text(fname);
    std::string out;
    out.reserve(f.size());
    const std::string tag = "offset=uint64(";
    size_t p = 0;
    for (;;) {
        const size_t q = f.find(tag, p);
        if (q == std::string::npos) {
            out.append(f, p, std::string::npos);
            break;
        }
        const size_t a = q + tag.size();
        const size_t b = f.find(')', a);
        const uint64_t off = std::stoull(f.substr(a, b - a));
        auto it = remap.find(off);
        if (it == remap.end()) {
            err = "blob offset without a chunk in the function text";
            return false;
        }
        out.append(f, p, a - p);
        out += std::to_string(it->second);
        p = b;
    }
    fam.funcs.push_back(out);
    return true;
}

// collect mode (ggml_backend_ane_collect_shapes): graphs are only translated, adding their shapes to the families
// without compiling or running anything; each family is then compiled once with all its shapes at its first real use
static std::atomic<bool> g_ane_collect{ false };

void ggml_backend_ane_collect_shapes(bool collect) {
    g_ane_collect = collect;
}

// compile all functions of a family into one program (replaces the previous version)
static bool ggml_ane_family_compile(ggml_ane_family & fam, std::string & err) {
    const int64_t t0 = ggml_time_us();
    char msg[512] = { 0 };
    std::string text = ggml_ane_mil::program_header();
    for (const std::string & f : fam.funcs) {
        text += f;
    }
    text += ggml_ane_mil::program_footer();
    ggml_ane_program * prog = ggml_ane_program_create(text.c_str(), text.size(), fam.map, fam.size, msg, sizeof(msg));
    if (!prog) {
        err = msg;
        ggml_ane_dump_program(text, std::vector<uint8_t>(fam.map, fam.map + fam.size));
        return false;
    }
    GGML_LOG_DEBUG("%s: compiled program with %zu functions, %.1f MB of weights in %.1f ms\n", __func__,
        fam.funcs.size(), fam.size / 1e6, (ggml_time_us() - t0) / 1e3);
    if (fam.prog) {
        // the larger version replaces the previous one (on iOS the purge does not give back the space of the compile)
        ggml_ane_program_purge_compiled(fam.prog);
        ggml_ane_program_free(fam.prog);
    }
    fam.prog       = prog;
    fam.n_compiled = fam.funcs.size();
    fam.version++;
    if (fam.funcs.size() > 16) {
        GGML_LOG_WARN("%s: %zu shapes compiled for one segment\n", __func__, fam.funcs.size());
    }
    return true;
}

// plan: dry translation of the segment, done by the caller
static ggml_ane_program_entry * ggml_ane_get_program(ggml_backend_ane_context * ctx, ggml_cgraph * cgraph, int i0, int i1, ggml_ane_plan & plan) {
    std::string err;

    std::string key = ggml_ane_plan_key(plan);
    auto it = ctx->programs.find(key);
    if (it != ctx->programs.end()) {
        it->second->last_used = ++ctx->n_calls;
        if (it->second->fam) {
            it->second->fam->last_used = ctx->n_calls;
        }
        return it->second.get();
    }

    // cache miss: translate again, this time packing the weights
    ggml_ane_plan full(false);
    if (!ggml_ane_translate(cgraph, i0, i1, full, err)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
        return nullptr;
    }
    const std::string mil = full.mil.text();
    // family of this shape: same baked weights (a shape-independent part of the key)
    std::string fam_key;
    for (const ggml_tensor * t : full.baked) {
        fam_key += ggml_ane_content_key(t) + "|";
    }
    if (fam_key.empty()) {
        fam_key = key;
    }
    if (ctx->families.find(fam_key) == ctx->families.end() && ctx->families.size() >= GGML_ANE_MAX_FAMILIES) {
        // unload the least recently used family: drop its shapes so that its program is freed
        auto lru = ctx->families.begin();
        for (auto f = ctx->families.begin(); f != ctx->families.end(); ++f) {
            if (f->second->last_used < lru->second->last_used) {
                lru = f;
            }
        }
        for (auto p2 = ctx->programs.begin(); p2 != ctx->programs.end(); ) {
            p2 = p2->second->fam == lru->second ? ctx->programs.erase(p2) : std::next(p2);
        }
        ctx->families.erase(lru);
        ctx->n_evictions++;
    }
    std::shared_ptr<ggml_ane_family> & fam = ctx->families[fam_key];
    if (!fam) {
        fam = std::make_shared<ggml_ane_family>();
    }
    fam->last_used = ++ctx->n_calls;
    const std::string fname = fam->funcs.empty() ? "main" : "f" + std::to_string(fam->funcs.size());
    if (!ggml_ane_family_add(*fam, full, fname, err)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
        return nullptr;
    }
    auto entry = std::make_unique<ggml_ane_program_entry>();
    entry->fam   = fam;
    entry->fname = fname;
    if (!g_ane_collect) {
        if (!ggml_ane_family_compile(*fam, err)) {
            GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
            fam->funcs.pop_back(); // the previous version stays in use
            return nullptr;
        }
        if (!entry->prepare()) {
            GGML_LOG_ERROR("%s: function %s not found in the compiled program\n", __func__, fname.c_str());
            return nullptr;
        }
    }

    if (ctx->programs.size() >= GGML_ANE_MAX_PROGRAMS) {
        // evict the least recently used program
        auto lru = ctx->programs.begin();
        for (auto p = ctx->programs.begin(); p != ctx->programs.end(); ++p) {
            if (p->second->last_used < lru->second->last_used) {
                lru = p;
            }
        }
        ctx->programs.erase(lru);
        ctx->n_evictions++;
    }
    entry->last_used = ++ctx->n_calls;
    ggml_ane_program_entry * res = entry.get();
    ctx->programs[std::move(key)] = std::move(entry);
    return res;
}

struct ggml_ane_copy_args {
    const ggml_ane_io_desc * d;
    uint8_t *                base;
    uint8_t *                data;
    const size_t *           nb;
    bool                     is_f32;
    bool                     to_surface;
    int64_t                  rows_per_task;
};

// copy rows [task * rows_per_task, ...) of the flattened (n, c, h) row index
static void ggml_ane_copy_rows_task(void * ctx, size_t task) {
    const ggml_ane_copy_args & a = *(const ggml_ane_copy_args *) ctx;
    const ggml_ane_io_desc * d = a.d;
    const int64_t n_rows = d->n * d->c * d->h;
    const size_t  row    = d->w * (a.is_f32 ? sizeof(float) : sizeof(ggml_fp16_t));
    for (int64_t r = (int64_t) task * a.rows_per_task; r < std::min(n_rows, (int64_t) (task + 1) * a.rows_per_task); r++) {
        const int64_t h = r % d->h;
        const int64_t c = (r / d->h) % d->c;
        const int64_t n = r / (d->h * d->c);
        ggml_fp16_t * s = (ggml_fp16_t *) (a.base + n * d->batch_stride + c * d->plane_stride + h * d->row_stride);
        uint8_t     * g = a.data + h * a.nb[1] + c * a.nb[2] + n * a.nb[3];
        if (a.is_f32) {
            if (a.to_surface) {
                ggml_fp32_to_fp16_row((const float *) g, s, d->w);
            } else {
                ggml_fp16_to_fp32_row(s, (float *) g, d->w);
            }
        } else if (a.to_surface) {
            memcpy(s, g, row);
        } else {
            memcpy(g, s, row);
        }
    }
}

// copy rows between memory (row i1, i2, i3 at data + i1*nb1 + i2*nb2 + i3*nb3) and an fp16 ANE surface [n, c, h, w];
// large copies (prompt batches) are split over several threads
static void ggml_ane_copy_rows(const ggml_ane_io_desc * d, IOSurfaceRef surf, uint8_t * data, const size_t nb[4], bool is_f32, bool to_surface) {
    GGML_ASSERT(d->esize == sizeof(ggml_fp16_t));
    const int64_t n_rows = d->n * d->c * d->h;
    const int64_t bytes  = n_rows * d->w * (is_f32 ? 4 : 2);
    const int64_t n_task = bytes < (1 << 20) ? 1 : std::min<int64_t>(8, n_rows);
    ggml_ane_copy_args a = { d, (uint8_t *) IOSurfaceGetBaseAddress(surf), data, nb, is_f32, to_surface, (n_rows + n_task - 1) / n_task };
    if (n_task == 1) {
        ggml_ane_copy_rows_task(&a, 0);
    } else {
        dispatch_apply_f((size_t) n_task, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &a, ggml_ane_copy_rows_task);
    }
}

static void ggml_ane_copy_tensor(const ggml_ane_io_desc * d, IOSurfaceRef surf, ggml_tensor * t, bool to_surface) {
    GGML_ASSERT(d->w == t->ne[0] && d->h == t->ne[1] && d->c == t->ne[2] && d->n == t->ne[3]);
    ggml_ane_copy_rows(d, surf, (uint8_t *) t->data, t->nb, t->type == GGML_TYPE_F32, to_surface);
}

static void ggml_ane_copy_aux(const ggml_ane_io_desc * d, IOSurfaceRef surf, std::vector<float> & data) {
    const size_t row = d->w * sizeof(float);
    const size_t nb[4] = { sizeof(float), row, row * d->h, row * d->h * d->c };
    ggml_ane_copy_rows(d, surf, (uint8_t *) data.data(), nb, true, true);
}

// dst = src rows written at the row indices in idx (same semantics as the CPU backend)
static void ggml_ane_set_rows(ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    const int64_t nc = src->ne[0];
    for (int64_t i03 = 0; i03 < src->ne[3]; i03++) {
        for (int64_t i02 = 0; i02 < src->ne[2]; i02++) {
            for (int64_t i = 0; i < src->ne[1]; i++) {
                const char * ip = (const char *) idx->data + i * idx->nb[0] + (i02 % idx->ne[1]) * idx->nb[1] + (i03 % idx->ne[2]) * idx->nb[2];
                const int64_t r = idx->type == GGML_TYPE_I64 ? *(const int64_t *) ip : *(const int32_t *) ip;
                GGML_ASSERT(r >= 0 && r < dst->ne[1]);
                const char * s = (const char *) src->data + i * src->nb[1] + i02 * src->nb[2] + i03 * src->nb[3];
                char       * d = (char *) dst->data + r * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3];
                if (src->type == dst->type) {
                    memcpy(d, s, nc * ggml_type_size(src->type));
                } else if (src->type == GGML_TYPE_F32) {
                    ggml_fp32_to_fp16_row((const float *) s, (ggml_fp16_t *) d, nc);
                } else {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) s, (float *) d, nc);
                }
            }
        }
    }
}

// bytes of weights a node bakes into its program
static int64_t ggml_ane_node_weight_bytes(const ggml_tensor * n) {
    if (n->op != GGML_OP_MUL_MAT || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return 0;
    }
    const ggml_tensor * w = n->src[0];
    ggml_ane_weight_mode mode = ggml_ane_get_weight_mode();
    if (mode == GGML_ANE_WEIGHTS_AUTO) {
        mode = ggml_is_quantized(w->type) ? GGML_ANE_WEIGHTS_INT8 : GGML_ANE_WEIGHTS_FP16;
    }
    return ggml_nelements(w) * (mode == GGML_ANE_WEIGHTS_FP16 ? 2 : 1);
}

static bool ggml_ane_segment_has_compute(ggml_cgraph * cgraph, int i0, int i1) {
    for (int i = i0; i < i1; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) && n->op != GGML_OP_NONE && !ggml_ane_is_view_op(n->op)) {
            return true;
        }
    }
    return false;
}

// debug: GGML_ANE_PROFILE prints where graph_compute spends its time
struct ggml_ane_prof {
    int64_t t_translate = 0, t_in = 0, t_eval = 0, t_out = 0, t_host = 0, n_eval = 0, n_graphs = 0;
};
static ggml_ane_prof g_ane_prof;

// hash of everything the translation of nodes [i0, i1) depends on, except tensor data (host tables are regenerated
// at every evaluation, data baked into the program text is checked with plan.guards)
static uint64_t ggml_ane_segment_fingerprint(ggml_cgraph * cgraph, int i0, int i1) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void * p, size_t n) { // 8 bytes at a time (called on every graph, per-byte hashing was ~1 ms on the A15)
        const uint8_t * b = (const uint8_t *) p;
        for (; n >= 8; n -= 8, b += 8) {
            uint64_t w;
            memcpy(&w, b, 8);
            h = (h ^ w) * 0x100000001b3ull;
            h ^= h >> 29;
        }
        for (; n > 0; n--, b++) {
            h = (h ^ *b) * 0x100000001b3ull;
        }
    };
    auto mix_tensor = [&](const ggml_tensor * t) {
        mix(&t, sizeof(t));
        mix(&t->type, sizeof(t->type));
        mix(&t->op, sizeof(t->op));
        mix(t->ne, sizeof(t->ne));
        mix(t->nb, sizeof(t->nb));
        mix(&t->data, sizeof(t->data));
        mix(&t->view_src, sizeof(t->view_src));
        mix(&t->view_offs, sizeof(t->view_offs));
        mix(&t->flags, sizeof(t->flags));
    };
    mix(&i0, sizeof(i0));
    mix(&i1, sizeof(i1));
    for (int i = i0; i < i1; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        mix_tensor(n);
        mix(n->op_params, sizeof(n->op_params));
        const int32_t uses = ggml_node_get_use_count(cgraph, i);
        mix(&uses, sizeof(uses));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (n->src[j]) {
                mix_tensor(n->src[j]);
            }
        }
        if (n->op == GGML_OP_MUL_MAT) {
            const uint64_t gen = ggml_ane_write_gen(n->src[0]->data); // baked weights rewritten -> new program
            mix(&gen, sizeof(gen));
        }
    }
    return h;
}

static enum ggml_status ggml_ane_run_segment(ggml_backend_ane_context * ctx, ggml_cgraph * cgraph, int i0, int i1) {
    int64_t t0 = ggml_time_us();
    const uint64_t fp = ggml_ane_segment_fingerprint(cgraph, i0, i1);
    ggml_ane_segment_entry * se = nullptr;
    {
        auto it = ctx->segments.find(fp);
        if (it != ctx->segments.end() && it->second.evictions == ctx->n_evictions) {
            bool same = true;
            for (const auto & g : it->second.plan->guards) {
                same = same && memcmp(g.first->data, g.second.data(), g.second.size() * sizeof(int32_t)) == 0;
            }
            if (same) {
                se = &it->second;
            }
        }
    }
    if (!se) {
        auto plan = std::make_unique<ggml_ane_plan>(true);
        std::string terr;
        if (!ggml_ane_translate(cgraph, i0, i1, *plan, terr)) {
            GGML_LOG_ERROR("%s: %s\n", __func__, terr.c_str());
            return GGML_STATUS_FAILED;
        }
        ggml_ane_program_entry * pe = nullptr;
        if (!plan->outputs.empty()) {
            pe = ggml_ane_get_program(ctx, cgraph, i0, i1, *plan);
            if (!pe) {
                return GGML_STATUS_FAILED;
            }
        }
        if (ctx->segments.size() >= 4096) {
            ctx->segments.clear();
        }
        ggml_ane_segment_entry & ne = ctx->segments[fp];
        ne.plan      = std::move(plan);
        ne.prog      = pe;
        ne.evictions = ctx->n_evictions;
        se = &ne;
    }
    ggml_ane_plan & plan = *se->plan;
    if (!se->prog) {
        // nothing computed on the ANE (e.g. only SET_ROWS of existing rows)
        for (ggml_tensor * n : plan.set_rows) {
            ggml_ane_set_rows(n);
        }
        return GGML_STATUS_SUCCESS;
    }
    ggml_ane_program_entry * e = se->prog;
    if (g_ane_collect) {
        return GGML_STATUS_SUCCESS; // collect mode: shapes only
    }
    if (e->fam && e->fam->n_compiled < e->fam->funcs.size()) {
        // shapes were collected: compile the family once with all of them
        std::string cerr;
        if (!ggml_ane_family_compile(*e->fam, cerr)) {
            GGML_LOG_ERROR("%s: %s\n", __func__, cerr.c_str());
            return GGML_STATUS_FAILED;
        }
    }
    if (e->fam && !e->prepare()) {
        GGML_LOG_ERROR("%s: function %s not found in the compiled program\n", __func__, e->fname.c_str());
        return GGML_STATUS_FAILED;
    }
    if (!e->prog) {
        return GGML_STATUS_SUCCESS; // purge-only mode: nothing was compiled
    }
    int64_t t1 = ggml_time_us();
    g_ane_prof.t_translate += t1 - t0;
    std::vector<IOSurfaceRef> in_surf = e->in_surf;
    int n_windows = 0;
    for (int i = 0; i < ggml_ane_program_n_inputs(e->prog, e->proc); i++) {
        ggml_ane_input & in = plan.inputs[e->in_map[i]];
        if (in.direct) {
            const ggml_ane_io_desc * d = ggml_ane_program_input(e->prog, e->proc, i);
            // a surface over just the rows the program reads (binding large surfaces makes every eval slower)
            IOSurfaceRef s = d->n == 1 && d->c == 1 && d->row_stride == in.tensor->nb[1] && d->batch_stride <= ggml_nbytes(in.tensor) ?
                ggml_ane_window(in.tensor->data, d->batch_stride) : nullptr;
            if (s && !e->no_windows) {
                in_surf[i] = s;
                n_windows++;
            } else {
                ggml_ane_copy_rows(d, e->in_surf[i], (uint8_t *) in.tensor->data, in.tensor->nb, false, true);
            }
        } else if (in.tensor) {
            ggml_ane_copy_tensor(ggml_ane_program_input(e->prog, e->proc, i), e->in_surf[i], in.tensor, true);
        } else {
            std::vector<float> data(in.n_aux);
            in.gen(data.data());
            ggml_ane_copy_aux(ggml_ane_program_input(e->prog, e->proc, i), e->in_surf[i], data);
        }
    }
    char err[512] = { 0 };
    int64_t t2 = ggml_time_us();
    g_ane_prof.t_in += t2 - t1;
    bool ok = ggml_ane_program_eval(e->prog, e->proc, in_surf.data(), e->out_surf.data(), err, sizeof(err));
    if (!ok && n_windows > 0) {
        // windows fail in some large programs: copy those inputs from now on
        GGML_LOG_WARN("%s: evaluation with memory windows failed, using copies for this program\n", __func__);
        e->no_windows = true;
        for (int i = 0; i < ggml_ane_program_n_inputs(e->prog, e->proc); i++) {
            ggml_ane_input & in = plan.inputs[e->in_map[i]];
            if (in.direct) {
                ggml_ane_copy_rows(ggml_ane_program_input(e->prog, e->proc, i), e->in_surf[i], (uint8_t *) in.tensor->data, in.tensor->nb, false, true);
                in_surf[i] = e->in_surf[i];
            }
        }
        ok = ggml_ane_program_eval(e->prog, e->proc, in_surf.data(), e->out_surf.data(), err, sizeof(err));
    }
    if (!ok) {
        GGML_LOG_ERROR("%s: %s\n", __func__, err);
        return GGML_STATUS_FAILED;
    }
    int64_t t3 = ggml_time_us();
    g_ane_prof.t_eval += t3 - t2;
    g_ane_prof.n_eval++;
    // write back outputs and run the deferred SET_ROWS in graph order: the allocator may reuse the memory of a
    // SET_ROWS source for a tensor computed after it, so the source must be read before that output is written
    std::unordered_map<const ggml_tensor *, int> order;
    for (int i = i0; i < i1; i++) {
        order[cgraph->nodes[i]] = i;
    }
    std::vector<std::pair<int, int>> ev; // (graph index, output slot or -1 - set_rows index)
    for (int i = 0; i < ggml_ane_program_n_outputs(e->prog, e->proc); i++) {
        ev.push_back({ order[plan.outputs[e->out_map[i]]], i });
    }
    for (size_t i = 0; i < plan.set_rows.size(); i++) {
        ev.push_back({ order[plan.set_rows[i]], -1 - (int) i });
    }
    std::sort(ev.begin(), ev.end());
    for (const auto & x : ev) {
        if (x.second >= 0) {
            ggml_ane_copy_tensor(ggml_ane_program_output(e->prog, e->proc, x.second), e->out_surf[x.second], plan.outputs[e->out_map[x.second]], false);
        } else {
            ggml_ane_set_rows(plan.set_rows[-1 - x.second]);
        }
    }
    g_ane_prof.t_out += ggml_time_us() - t3;
    return GGML_STATUS_SUCCESS;
}

static const char * ggml_backend_ane_get_name(ggml_backend_t backend) {
    return "ANE";
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_free(ggml_backend_t backend) {
    delete (ggml_backend_ane_context *) backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_ane_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_ane_context *) backend->context;

    // one program per segment; segments are cut when their baked weights exceed the budget
    static const int64_t budget = (getenv("GGML_ANE_SEG_WEIGHT_MB") ? atoll(getenv("GGML_ANE_SEG_WEIGHT_MB")) : 64) << 20; // memory windows fail in larger programs
    for (int i = 0; i < cgraph->n_nodes; ) {
        int     j = i;
        int64_t w = 0;
        while (j < cgraph->n_nodes) {
            const int64_t wj = ggml_ane_node_weight_bytes(cgraph->nodes[j]);
            if (j > i && w + wj > budget) {
                break;
            }
            w += wj;
            j++;
        }
        if (ggml_ane_segment_has_compute(cgraph, i, j)) {
            const enum ggml_status st = ggml_ane_run_segment(ctx, cgraph, i, j);
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
        }
        i = j;
    }
    static const bool prof = getenv("GGML_ANE_PROFILE") != nullptr;
    if (prof && ++g_ane_prof.n_graphs % 16 == 0) {
        const double n = 16.0;
        GGML_LOG_INFO("ggml-ane profile (per graph, avg of 16): translate %.2f ms, copy in %.2f ms, eval %.2f ms (%.1f calls), copy out %.2f ms, host %.2f ms\n",
            g_ane_prof.t_translate / n / 1e3, g_ane_prof.t_in / n / 1e3, g_ane_prof.t_eval / n / 1e3, g_ane_prof.n_eval / n, g_ane_prof.t_out / n / 1e3, g_ane_prof.t_host / n / 1e3);
        g_ane_prof = ggml_ane_prof();
        g_ane_prof.n_graphs = 0;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_ane_i = {
    /* .get_name            = */ ggml_backend_ane_get_name,
    /* .free                = */ ggml_backend_ane_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_ane_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

static ggml_guid_t ggml_backend_ane_guid(void) {
    static ggml_guid guid = { 0x61, 0x6e, 0x65, 0x2d, 0x6b, 0x65, 0x73, 0x74, 0x72, 0x65, 0x6c, 0x2d, 0x76, 0x30, 0x30, 0x31 };
    return &guid;
}

ggml_backend_t ggml_backend_ane_init(void) {
    ggml_backend_reg_t reg = ggml_backend_ane_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    return new ggml_backend {
        /* .guid    = */ ggml_backend_ane_guid(),
        /* .iface   = */ ggml_backend_ane_i,
        /* .device  = */ ggml_backend_reg_dev_get(reg, 0),
        /* .context = */ new ggml_backend_ane_context,
    };
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_ane_guid());
}

//
// device
//

static const char * ggml_backend_ane_device_get_name(ggml_backend_dev_t dev) {
    return "ANE";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_ane_device_get_description(ggml_backend_dev_t dev) {
    static std::string desc = std::string("Apple Neural Engine (") + ggml_ane_runtime_arch() + ")";
    return desc.c_str();
    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    uint64_t memsize = 0;
    size_t   len     = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    *total = memsize;

    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) == KERN_SUCCESS) {
        *free = (size_t) (vm.free_count + vm.inactive_count) * vm_page_size;
    } else {
        *free = 0;
    }
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_ane_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ane_device_get_name(dev);
    props->description = ggml_backend_ane_device_get_description(dev);
    props->type        = ggml_backend_ane_device_get_type(dev);
    ggml_backend_ane_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_ane_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_ane_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_ane_buffer_type(dev);
}

// wraps host memory (e.g. mmapped weights); such tensors can be baked into programs but not bound as surfaces
static ggml_backend_buffer_t ggml_backend_ane_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    auto * ctx = new ggml_backend_ane_buffer_context;
    ctx->base = ptr;
    return ggml_backend_buffer_init(ggml_backend_ane_buffer_type(dev), ggml_backend_ane_buffer_interface, ctx, size);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_ane_dims_ok(const ggml_tensor * t) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] > GGML_ANE_MAX_DIM) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (ggml_is_empty(op)) {
        return false; // e.g. rows selected by an empty output list: nothing to compute
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
            // views of computed values must be plain reshapes; views of inputs only need contiguous rows
            return (ggml_is_contiguous(op) && op->view_offs == 0 && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16)) ||
                   (src0->op == GGML_OP_NONE && ggml_ane_rows_contiguous(op));
        case GGML_OP_PERMUTE:
            return ggml_get_op_params_i32(op, 0) == 0 && ggml_ane_rows_contiguous(op); // dim 0 stays innermost
        case GGML_OP_CONT:
            return ggml_ane_rows_contiguous(src0) && op->type == src0->type && ggml_ane_dims_ok(op);
        case GGML_OP_SET_ROWS:
            return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   (src1->type == GGML_TYPE_I32 || src1->type == GGML_TYPE_I64) && ggml_is_contiguous_rows(op);
        case GGML_OP_FLASH_ATTN_EXT: {
            const ggml_tensor * k = op->src[1];
            const ggml_tensor * v = op->src[2];
            const ggml_tensor * m = op->src[3];
            if (op->src[4] || op->src[5] || ggml_get_op_params_f32(op, 1) != 0.0f || ggml_get_op_params_f32(op, 2) != 0.0f) {
                return false; // sinks, kv_rows, alibi, softcap
            }
#if TARGET_OS_IPHONE
            // very long KV with long batches fails to compile on phone ANEs (A15: 16384 KV x 512 tokens x 32 heads)
            if (k->ne[1] > 8192) {
                return false;
            }
#endif
            return src0->type == GGML_TYPE_F32 && ggml_ane_rows_contiguous(src0) && ggml_ane_rows_contiguous(k) && ggml_ane_rows_contiguous(v) &&
                   (!m || (m->type == GGML_TYPE_F16 && m->ne[2] == 1 && m->ne[3] == 1)) &&
                   src0->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 && src0->ne[2] % k->ne[2] == 0 &&
                   ggml_ane_dims_ok(src0) && ggml_ane_dims_ok(k) && ggml_ane_dims_ok(v) && ggml_ane_dims_ok(op);
        }
        case GGML_OP_MUL_MAT:
            // src0 must be a leaf (constant data, baked into the program)
            return src0->op == GGML_OP_NONE && src0->view_src == nullptr && ggml_ane_can_bake(src0) &&
                   ggml_is_contiguous(src0) && src0->ne[2] == 1 && src0->ne[3] == 1 &&
                   src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[2] == 1 && src1->ne[3] == 1 &&
                   ggml_ane_dims_ok(src1) && ggml_ane_dims_ok(op) && // e.g. the LM head (vocab rows) stays off the ANE
                   op->type == GGML_TYPE_F32;
        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_are_same_shape(src0, op)) {
                return false;
            }
            // MIL broadcasts only dims of size 1
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (src1->ne[i] != src0->ne[i] && src1->ne[i] != 1) {
                    return false;
                }
            }
            return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_ane_dims_ok(op);
        }
        case GGML_OP_GET_ROWS:
            // ids are read at translation time and baked as slices
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
                   ggml_ane_rows_contiguous(src0) && ggml_is_contiguous(src1) && ggml_ane_dims_ok(src0) && ggml_ane_dims_ok(op);
        case GGML_OP_RMS_NORM:
            // very short rows are dominated by fp16 rounding (models use >= 64)
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && src0->ne[0] >= 32 && ggml_ane_dims_ok(op);
        case GGML_OP_GLU: {
            if (ggml_get_op_params_i32(op, 0) != GGML_GLU_OP_SWIGLU || src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (src1 && (src1->type != GGML_TYPE_F32 || !ggml_are_same_shape(src0, src1) || !ggml_is_contiguous(src1))) {
                return false;
            }
            return ggml_is_contiguous(src0) && ggml_ane_dims_ok(op);
        }
        case GGML_OP_ROPE: {
            const int n_dims = ggml_get_op_params_i32(op, 1);
            const int mode   = ggml_get_op_params_i32(op, 2);
            const ggml_tensor * ff = op->src[2];
            // very long batches fail to compile on older ANEs (A15: 8192 tokens); llama.cpp batches are much shorter
            return (mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_NORMAL) && ggml_get_op_params_f32(op, 7) == 0.0f &&
                   src0->ne[2] <= 4096 &&
                   (!ff || (ff->type == GGML_TYPE_F32 && ggml_is_contiguous(ff) && ff->ne[0] >= n_dims / 2)) &&
                   n_dims % 2 == 0 && n_dims <= src0->ne[0] && src0->ne[3] == 1 &&
                   src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
                   src1->type == GGML_TYPE_I32 && ggml_ane_dims_ok(op);
        }
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_ane_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_ane_buffer_type(dev) || ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_ane_device_i = {
    /* .get_name             = */ ggml_backend_ane_device_get_name,
    /* .get_description      = */ ggml_backend_ane_device_get_description,
    /* .get_memory           = */ ggml_backend_ane_device_get_memory,
    /* .get_type             = */ ggml_backend_ane_device_get_type,
    /* .get_props            = */ ggml_backend_ane_device_get_props,
    /* .init_backend         = */ ggml_backend_ane_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ane_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_ane_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_ane_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ane_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_ane_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_ane_buffer_type_alloc_buffer,
            /* .alloc_buffer_n   = */ ggml_backend_ane_buffer_type_alloc_buffer_n,
            /* .get_alignment    = */ ggml_backend_ane_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ ggml_backend_ane_buffer_type_get_alloc_size_n,
            /* .is_host          = */ ggml_backend_ane_buffer_type_is_host,
        },
        /* .device  = */ dev,
        /* .context = */ NULL,
    };
    return &buft;
}

//
// registry
//

static const char * ggml_backend_ane_reg_get_name(ggml_backend_reg_t reg) {
    return "ANE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_ane_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_ane_runtime_available() ? 1 : 0;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_ane_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_ane_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &dev;
}

static void * ggml_backend_ane_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return nullptr;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const ggml_backend_reg_i ggml_backend_ane_reg_i = {
    /* .get_name         = */ ggml_backend_ane_reg_get_name,
    /* .get_device_count = */ ggml_backend_ane_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ane_reg_get_device,
    /* .get_proc_address = */ ggml_backend_ane_get_proc_address,
};

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ane_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ane_reg)
