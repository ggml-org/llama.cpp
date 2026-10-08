#include "dispatch-registry.h"

#include "common/dispatch-common.h"
#include "llm/dispatch-attention-qkv.h"
#include "llm/dispatch-gated-delta-net.h"
#include "llm/dispatch-ssm-conv.h"
#include "qwen/dispatch-qwen.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr size_t kOpCount = static_cast<size_t>(GGML_OP_COUNT);

static const std::vector<DispatchRegistration> kEmptyRegistrations;
static const std::vector<DispatchActivationConsumerRegistration> kEmptyActivationConsumers;

static bool valid_root_op(ggml_op op) {
    return op >= 0 && static_cast<size_t>(op) < kOpCount;
}

static void sort_registrations(std::vector<DispatchRegistration> & registrations) {
    std::stable_sort(
        registrations.begin(), registrations.end(),
        [](const DispatchRegistration & lhs, const DispatchRegistration & rhs) { return lhs.priority > rhs.priority; });
}

static bool qwen_dispatch_disabled_from_environment() {
    const char * value = std::getenv("GGML_HRX_DISABLE_QWEN_DISPATCH");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

static DispatchRegistry build_registry(bool include_qwen) {
    DispatchRegistryBuilder builder;
    register_common_dispatches(builder);
    register_llm_attention_qkv_dispatches(builder);
    register_llm_gated_delta_net_dispatch(builder);
    register_llm_ssm_conv_dispatch(builder);
    if (include_qwen) {
        register_qwen_dispatches(builder);
    }
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

const std::vector<DispatchActivationConsumerRegistration> &
DispatchRegistry::activation_consumers_for_root(ggml_op root_op) const {
    if (!valid_root_op(root_op) || activation_consumers_by_root_.empty()) {
        return kEmptyActivationConsumers;
    }
    return activation_consumers_by_root_[static_cast<size_t>(root_op)];
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

void DispatchRegistryBuilder::add_activation_consumer(DispatchActivationConsumerRegistration registration) {
    if (!valid_root_op(registration.root_op) || registration.accepts == nullptr ||
        registration.accepted_formats == DispatchActivationInputNone) {
        return;
    }
    if (registry_.activation_consumers_by_root_.empty()) {
        registry_.activation_consumers_by_root_.resize(kOpCount);
    }
    registry_.activation_consumers_by_root_[static_cast<size_t>(registration.root_op)].push_back(
        std::move(registration));
}

DispatchRegistry DispatchRegistryBuilder::build() {
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    if (registry_.activation_consumers_by_root_.empty()) {
        registry_.activation_consumers_by_root_.resize(kOpCount);
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
    static const DispatchRegistry gfx1100_registry              = build_registry(true);
    static const DispatchRegistry gfx1100_generic_only_registry = build_registry(false);
    static const DispatchRegistry gfx1151_registry              = build_registry(true);
    static const DispatchRegistry gfx1151_generic_only_registry = build_registry(false);

    const bool qwen_disabled = qwen_dispatch_disabled_from_environment();

    if (target.architecture == "gfx1100") {
        return qwen_disabled ? &gfx1100_generic_only_registry : &gfx1100_registry;
    }
    if (target.architecture == "gfx1151") {
        return qwen_disabled ? &gfx1151_generic_only_registry : &gfx1151_registry;
    }
    return nullptr;
}

}  // namespace ggml::hrx
