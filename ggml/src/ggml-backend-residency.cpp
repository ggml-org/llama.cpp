// ggml-backend-residency.cpp
//
// Memory residency tracker (Project C of the heterogeneous-backend
// scout). The companion to ``ggml-backend-auto.cpp`` (Project B).
// See ``ggml/include/ggml-backend.h`` for the API contract.
//
// What this does
// --------------
//
// Tracks per-(backend, tensor) last-use iteration. The scheduler
// calls ``mark_used`` for every tensor that is touched in a
// given iter; before each iter, the scheduler queries
// ``suggest_releases`` to find tensors that have been idle for N
// iters and are safe to evict from the backend's heap.
//
// On Apple Silicon's unified memory, idle backends' heap copies
// are pure waste: the bytes still occupy physical RAM and the
// backend's command-queue / IOSurface pages, but no compute is
// going through them. The residency tracker is the signal the
// Metal/ANE backends use to free those buffers when their
// residency goes stale.
//
// Why this is separate from the scheduler
// ---------------------------------------
//
// The scheduler already knows what tensors are in flight (via
// the per-iter ``ggml_cgraph``). What it does NOT know is which
// tensors are sitting on which backend's heap between iters;
// that's a cross-iter concern that lives outside the graph. The
// residency tracker is the cross-iter state; the scheduler
// queries it, the backends' ``buffer_free`` hooks consume the
// suggestions.
//
// Project C. See ``ggml/include/ggml-backend.h`` for the API.

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// One (backend, tensor) last-use record. The pair is the natural
// key: the same tensor can be resident on multiple backends (e.g.
// a weight in Metal's heap, the K/V cache in CPU RAM, a stage
// in ANE's IOSurface), and each is its own residency entry.
struct residency_entry {
    int64_t last_used_iter;
    // The backend and tensor are stored as raw pointers in the
    // map key; the (const char *) name is captured at mark-time
    // so the suggest API can return stable strings without
    // holding the tensor alive (the user is responsible for
    // freeing the tensor; the tracker does not own it).
    std::string backend_name;
    std::string tensor_name;
};

// The residency table: key = (backend, tensor), value =
// last_used_iter + cached display names.
struct residency_table {
    // Hash/eq for (backend, tensor *) pairs. Two pairs are
    // equal when both pointers are equal. Defined inline so the
    // template specialisation picks them up.
    struct pair_hash {
        size_t operator()(const std::pair<ggml_backend_t, struct ggml_tensor *> & p) const {
            return std::hash<ggml_backend_t>()(p.first) ^
                   std::hash<struct ggml_tensor *>()(p.second) << 1;
        }
    };
    struct pair_eq {
        bool operator()(
            const std::pair<ggml_backend_t, struct ggml_tensor *> & a,
            const std::pair<ggml_backend_t, struct ggml_tensor *> & b) const {
            return a.first == b.first && a.second == b.second;
        }
    };
    std::unordered_map<
        std::pair<ggml_backend_t, struct ggml_tensor *>,
        residency_entry,
        pair_hash,
        pair_eq> entries;
    int64_t current_iter = 0;
};

} // namespace

extern "C" ggml_backend_residency_t ggml_backend_residency_new(void) {
    auto * t = new residency_table;
    return reinterpret_cast<ggml_backend_residency_t>(t);
}

extern "C" void ggml_backend_residency_free(ggml_backend_residency_t res) {
    if (res == nullptr) {
        return;
    }
    auto * t = reinterpret_cast<residency_table *>(res);
    delete t;
}

extern "C" void ggml_backend_residency_mark_used(
        ggml_backend_residency_t res,
        ggml_backend_t backend,
        struct ggml_tensor * tensor,
        int64_t iter) {
    if (res == nullptr || backend == nullptr || tensor == nullptr) {
        return;
    }
    auto * t = reinterpret_cast<residency_table *>(res);
    auto key = std::make_pair(backend, tensor);
    auto it = t->entries.find(key);
    if (it == t->entries.end()) {
        residency_entry e;
        e.last_used_iter = iter;
        // The names are stable strings owned by ggml / the
        // caller; the copy is intentional because the API
        // contract returns ``const char *`` from suggestions, and
        // the user could free the tensor before the suggestions
        // are consumed.
        e.backend_name = std::string(ggml_backend_name(backend));
        e.tensor_name = std::string(tensor->name);
        t->entries.emplace(key, std::move(e));
    } else {
        // Idempotent: a second call with the same iter is a
        // no-op (the stored iter is already equal to or larger
        // than the input). We don't bump to a higher iter than
        // what's stored; the "last used" is monotonic.
        if (it->second.last_used_iter < iter) {
            it->second.last_used_iter = iter;
        }
    }
}

extern "C" ggml_backend_residency_suggestion_t * ggml_backend_residency_suggest_releases(
        ggml_backend_residency_t res,
        int64_t current_iter,
        int64_t idle_threshold,
        size_t * out_count) {
    if (res == nullptr) {
        if (out_count != nullptr) {
            *out_count = 0;
        }
        return nullptr;
    }
    auto * t = reinterpret_cast<residency_table *>(res);
    // Walk the entries. An entry is "release-suggested" if its
    // last_used_iter is older than (current_iter - idle_threshold).
    // The default idle_threshold is 4 iters (a sensible default
    // for a decoder loop where each layer is touched once per
    // iter; 4 iters = the K/V cache window of one prefill chunk).
    std::vector<ggml_backend_residency_suggestion_t> out;
    out.reserve(t->entries.size() / 4);
    for (const auto & kv : t->entries) {
        const residency_entry & e = kv.second;
        if (current_iter - e.last_used_iter >= idle_threshold) {
            ggml_backend_residency_suggestion_t s;
            s.backend_name = e.backend_name.c_str();
            s.tensor_name = e.tensor_name.c_str();
            s.last_used_iter = e.last_used_iter;
            out.push_back(s);
        }
    }
    if (out_count != nullptr) {
        *out_count = out.size();
    }
    if (out.empty()) {
        return nullptr;
    }
    // Caller-owned allocation; the matching free is
    // ggml_backend_residency_free_suggestions. The strings
    // point into the table's owned names; the table outlives
    // the suggestions because the caller is expected to
    // consume the suggestions and free them in the same
    // call frame (e.g. the scheduler queries and consumes
    // before the next iter).
    auto * raw = static_cast<ggml_backend_residency_suggestion_t *>(
        std::malloc(out.size() * sizeof(ggml_backend_residency_suggestion_t)));
    std::memcpy(raw, out.data(),
                out.size() * sizeof(ggml_backend_residency_suggestion_t));
    return raw;
}

extern "C" void ggml_backend_residency_free_suggestions(
        ggml_backend_residency_suggestion_t * suggestions) {
    if (suggestions != nullptr) {
        std::free(suggestions);
    }
}
