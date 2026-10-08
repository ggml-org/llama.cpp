#include "dispatch-registry.h"

#include "dispatch-add.h"
#include "dispatch-gather-add.h"
#include "dispatch-llm-matmul.h"
#include "dispatch-moe-router.h"
#include "dispatch-qwen-attention-postprocess.h"
#include "dispatch-qwen-flash-attention.h"
#include "dispatch-qwen-matmul.h"
#include "dispatch-qwen-preamble.h"
#include "dispatch-rmsnorm.h"
#include "dispatch-routed-ffn.h"

#include <algorithm>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr size_t kOpCount = static_cast<size_t>(GGML_OP_COUNT);

static const std::vector<DispatchRegistration> kEmptyRegistrations;

static bool valid_root_op(ggml_op op) {
    return op >= 0 && static_cast<size_t>(op) < kOpCount;
}

static void sort_registrations(std::vector<DispatchRegistration> & registrations) {
    std::stable_sort(
        registrations.begin(), registrations.end(),
        [](const DispatchRegistration & lhs, const DispatchRegistration & rhs) { return lhs.priority > rhs.priority; });
}

static void register_llm_dispatches(DispatchRegistryBuilder & builder) {
    register_qwen_attention_postprocess_dispatches(builder);
    register_qwen_flash_attention_dispatches(builder);
    register_qwen_matmul_dispatches(builder);
    register_llm_matmul_dispatches(builder);
    register_routed_ffn_dispatches(builder);
    register_qwen_preamble_dispatches(builder);
    register_qwen_rmsnorm_dispatches(builder);
    register_moe_router_dispatches(builder);
}

static DispatchRegistry build_llm_registry() {
    DispatchRegistryBuilder builder;
    register_add_dispatch(builder);
    register_gather_add_dispatch(builder);
    register_llm_dispatches(builder);
    return builder.build();
}

}  // namespace

bool DispatchRegistry::match(const DispatchMatchContext & context, DispatchMatch & match) const {
    return this->match(context, match, nullptr);
}

bool DispatchRegistry::match(const DispatchMatchContext & context,
                             DispatchMatch &              match,
                             DispatchMatchDiagnostics *   diagnostics) const {
    if (context.root_node == nullptr || !valid_root_op(context.root_node->op) || registrations_by_root_.empty()) {
        return false;
    }
    if (diagnostics != nullptr) {
        diagnostics->root_op = context.root_node->op;
        diagnostics->attempts.clear();
    }
    const std::vector<DispatchRegistration> & registrations =
        registrations_by_root_[static_cast<size_t>(context.root_node->op)].ordered;
    for (const DispatchRegistration & registration : registrations) {
        DispatchMatch candidate;
        if (registration.matcher != nullptr && registration.matcher(context, candidate)) {
            if (diagnostics != nullptr) {
                diagnostics->attempts.push_back({
                    registration.name != nullptr ? registration.name : "",
                    registration.root_op,
                    registration.kind,
                    registration.priority,
                    registration.source,
                    true,
                    candidate.covered_nodes,
                    candidate.status.errors(),
                });
            }
            match = std::move(candidate);
            return true;
        }
        if (diagnostics != nullptr) {
            diagnostics->attempts.push_back({
                registration.name != nullptr ? registration.name : "",
                registration.root_op,
                registration.kind,
                registration.priority,
                registration.source,
                false,
                candidate.covered_nodes,
                candidate.status.errors(),
            });
        }
        match.status.append(candidate.status);
    }
    return false;
}

const std::vector<DispatchRegistration> & DispatchRegistry::registrations_for_root(ggml_op root_op) const {
    if (!valid_root_op(root_op) || registrations_by_root_.empty()) {
        return kEmptyRegistrations;
    }
    return registrations_by_root_[static_cast<size_t>(root_op)].ordered;
}

void DispatchRegistryBuilder::add(DispatchRegistration registration) {
    if (!valid_root_op(registration.root_op) || registration.matcher == nullptr) {
        return;
    }
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    DispatchRegistry::RegistrationGroup & group =
        registry_.registrations_by_root_[static_cast<size_t>(registration.root_op)];
    if (registration.kind == DispatchMatchKind::Fused) {
        group.fused.push_back(std::move(registration));
    } else {
        group.single_op.push_back(registration);
        registry_.single_op_registrations_.push_back(std::move(registration));
    }
}

DispatchRegistry DispatchRegistryBuilder::build() {
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    for (DispatchRegistry::RegistrationGroup & group : registry_.registrations_by_root_) {
        sort_registrations(group.fused);
        sort_registrations(group.single_op);
        group.ordered = group.fused;
        group.ordered.insert(group.ordered.end(), group.single_op.begin(), group.single_op.end());
    }
    sort_registrations(registry_.single_op_registrations_);
    return std::move(registry_);
}

const DispatchRegistry * find_dispatch_registry(const DispatchTarget & target) {
    static const DispatchRegistry gfx1100_registry = build_llm_registry();
    static const DispatchRegistry gfx1151_registry = build_llm_registry();

    if (target.architecture == "gfx1100") {
        return &gfx1100_registry;
    }
    if (target.architecture == "gfx1151") {
        return &gfx1151_registry;
    }
    return nullptr;
}

}  // namespace ggml::hrx
