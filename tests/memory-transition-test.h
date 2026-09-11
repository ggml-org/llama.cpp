#pragma once

#include "../src/llama-memory-transition.h"
#include "../src/llama-memory-executor.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

using state = llama_memory_transition_state;
using status = llama_memory_transition_status;
using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

// Build two real regions in one CPU arena; only execution is simulated.
static arena_ptr make_arena(ggml_backend_buffer_type_t buft, bool persistent = true) {
    const size_t a = ggml_backend_buft_get_alignment(buft);
    arena_ptr arena(ggml_backend_memory_arena_new(buft, 4*a), ggml_backend_memory_arena_free);
    GGML_ASSERT(arena);
    GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), 0));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 11, 0, a, a, 0, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 12, 2*a, a, a,
        persistent ? GGML_BACKEND_MEMORY_REGION_PERSISTENT : 0, nullptr));
    GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
    return arena;
}

struct component : llama_memory_consumer, llama_memory_executor_backend {
    testing & t;
    ggml_backend_memory_arena_t arena;
    uint64_t id;
    std::vector<std::string> & events;
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    llama_memory_executor executor;
    std::vector<llama_memory_execution> work;
    uint64_t revision = 1;
    int destroyed = 0;
    int activations = 0;
    int drained = 0;
    bool complete = true;
    bool runtime_changed = false;
    bool unchanged = false;
    bool can_recover = true;
    std::string fail_at;
    std::string throw_at;
    std::function<void(const std::string &)> hook;

    struct executable : llama_memory_executable {
        component & owner;
        explicit executable(component & owner) : owner(owner) {}
        ~executable() override {
            owner.t.assert_true(ggml_backend_memory_arena_lease_count(owner.arena) > 0);
            ++owner.destroyed;
        }
    };

    // Install a fake capture that still pins its real arena lease.
    component(testing & t, ggml_backend_memory_arena_t arena, uint64_t id, std::vector<std::string> & events) :
        t(t), arena(arena), id(id), events(events) {
        lease.reset(ggml_backend_memory_arena_acquire(arena, id));
        GGML_ASSERT(lease);
        *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease.get()))) = int(id);
        install();
    }

    // Finish fake work before destroying the backend that owns its execution pins.
    ~component() override {
        complete = true;
        drain();
        executor.retire(*this);
    }

    // Record protocol ordering and inject a failure at a named boundary.
    bool step(const std::string & name) {
        events.push_back(name + ":" + std::to_string(id));
        if (hook) {
            hook(name);
        }
        if (throw_at == name) {
            throw std::runtime_error("activation callback");
        }
        return fail_at != name;
    }

    // Rebuild only a capture that was explicitly retired.
    void install() {
        std::unique_ptr<llama_memory_executable> native = std::make_unique<executable>(*this);
        GGML_ASSERT(executor.capture(native, {lease.get()}, revision));
    }

    // Completion writes make a backup taken before draining observably wrong.
    bool drain() override {
        if (!complete) {
            return false;
        }
        for (auto & pin : work) {
            GGML_ASSERT(pin);
            auto * value = static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease.get())));
            ++*value;
            ++drained;
            pin.reset();
        }
        work.clear();
        return true;
    }

    struct proposal : llama_memory_preparation {
        component & owner;
        bool affected = false;
        int saved = 0;
        bool backup_valid = false;
        bool released = false;
        bool alive = true;
        uint64_t old_revision;
        lease_ptr next{nullptr, ggml_backend_memory_lease_free};

        explicit proposal(component & owner) : owner(owner), old_revision(owner.revision) {}
        ~proposal() override { alive = false; owner.events.push_back("cleanup:" + std::to_string(owner.id)); }

        bool quiesce(const std::vector<llama_memory_resource_id> & changed) override {
            affected = owner.runtime_changed || std::find(changed.begin(), changed.end(), owner.id) != changed.end();
            if (affected) {
                owner.executor.quiesce();
            }
            return owner.step("quiesce");
        }
        bool drain() override {
            if (!owner.step("drain")) {
                return false;
            }
            if (affected) {
                if (!owner.drain()) {
                    return false;
                }
                saved = *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(owner.lease.get())));
                backup_valid = true;
            }
            return true;
        }
        bool invalidate() override {
            if (!owner.step("invalidate")) {
                return false;
            }
            return !affected || owner.executor.retire(owner).status == llama_memory_executor_status::retired;
        }
        bool release() override {
            if (!owner.step("release")) {
                return false;
            }
            if (affected) {
                released = true;
                owner.lease.reset();
            }
            return true;
        }
        bool bind(const std::vector<llama_memory_region_binding> & bindings) override {
            for (const auto & binding : bindings) {
                if (binding.region.id != owner.id) {
                    continue;
                }
                if (affected) {
                    next.reset(ggml_backend_memory_lease_retain(binding.lease));
                    *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(next.get()))) = saved;
                } else if (ggml_backend_memory_lease_buffer(binding.lease) !=
                           ggml_backend_memory_lease_buffer(owner.lease.get())) {
                    return false;
                }
                return owner.step("bind");
            }
            return false;
        }
        bool activate() override {
            if (!owner.step("activate")) {
                return false;
            }
            if (affected) {
                owner.lease = std::move(next);
                ++owner.revision;
                owner.install();
            }
            ++owner.activations;
            return true;
        }

        struct restoration : llama_memory_preparation {
            proposal & original;
            lease_ptr next{nullptr, ggml_backend_memory_lease_free};

            explicit restoration(proposal & original) : original(original) {}
            ~restoration() override {
                original.owner.t.assert_true(original.alive);
                original.owner.events.push_back("restore_cleanup:" + std::to_string(original.owner.id));
            }

            bool quiesce(const std::vector<llama_memory_resource_id> &) override {
                original.owner.executor.quiesce();
                return original.owner.step("restore_quiesce");
            }
            bool drain() override {
                auto & owner = original.owner;
                if (!owner.step("restore_drain") || !owner.drain()) return false;
                if (!original.backup_valid) {
                    if (original.released || !owner.lease) return false;
                    original.saved = *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(owner.lease.get())));
                    original.backup_valid = true;
                }
                return true;
            }
            bool invalidate() override {
                auto & owner = original.owner;
                if (!owner.step("restore_invalidate")) return false;
                const auto result = owner.executor.retire(owner);
                return result.status == llama_memory_executor_status::retired || result.status == llama_memory_executor_status::unchanged;
            }
            bool release() override {
                auto & owner = original.owner;
                if (!owner.step("restore_release")) return false;
                original.next.reset();
                owner.lease.reset();
                return true;
            }
            bool bind(const std::vector<llama_memory_region_binding> & bindings) override {
                auto & owner = original.owner;
                if (!original.backup_valid) return false;
                for (const auto & binding : bindings) {
                    if (binding.region.id == owner.id) {
                        next.reset(ggml_backend_memory_lease_retain(binding.lease));
                        *static_cast<int *>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(next.get()))) = original.saved;
                        return owner.step("restore_bind");
                    }
                }
                return false;
            }
            bool activate() override {
                auto & owner = original.owner;
                if (!owner.step("restore_activate")) return false;
                owner.lease = std::move(next);
                owner.revision = original.old_revision;
                owner.install();
                return true;
            }
        };

        // Explicitly preserve old consumer data; metadata rollback alone is not recovery.
        bool prepare_recovery(std::unique_ptr<llama_memory_preparation> & output) override {
            if (!owner.can_recover || !owner.step("prepare_recovery")) return false;
            if (affected) output = std::make_unique<restoration>(*this);
            return true;
        }
    };

    bool prepare(const llama_memory_transition_target &, const llama_memory_layout &,
            std::unique_ptr<llama_memory_preparation> & output) override {
        if (!unchanged) {
            output = std::make_unique<proposal>(*this);
        }
        return true;
    }
};

struct fixture {
    ggml_backend_ptr backend{ggml_backend_cpu_init()};
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());
    size_t a = ggml_backend_buft_get_alignment(buft);
    arena_ptr arena;
    std::vector<std::string> events;
    component workspace;
    component preserved;
    llama_memory_transition transition{std::vector<llama_memory_consumer *>{&workspace, &preserved}};
    llama_memory_transition_target target;
    std::vector<llama_memory_transition_arena> arenas;

    explicit fixture(testing & t, bool persistent = true) :
        arena(make_arena(buft, persistent)), workspace(t, arena.get(), 11, events), preserved(t, arena.get(), 12, events) {
        events.reserve(256);
        target = {
            {
                {{1, LLAMA_MEMORY_ALLOCATION_HOST, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}},
                {{11, 1, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable},
                 {12, 1, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::preserve}},
                {{10, {}, {{11, 2*a, 2*a, a, LLAMA_MEMORY_ACCESS_WRITE, 0},
                           {12, a, a, a, LLAMA_MEMORY_ACCESS_READ, 0}}}},
                {12}, {12},
            },
            10, {{1, LLAMA_MEMORY_ALLOCATION_HOST, 4*a, a}}, {},
        };
        if (persistent) {
            target.fixed = {{0, {12, 2*a, a, a, GGML_BACKEND_MEMORY_REGION_PERSISTENT}}};
        }
        arenas = {{1, LLAMA_MEMORY_ALLOCATION_HOST, arena.get()}};
    }

    // Assert preparation succeeds before exercising any destructive transition boundary.
    bool prepare(testing & t) {
        return t.assert_true(transition.prepare(target).status == status::prepared);
    }
};
