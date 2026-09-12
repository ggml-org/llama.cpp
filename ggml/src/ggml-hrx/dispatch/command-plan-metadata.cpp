#include "command-plan-metadata.h"

#include <cstring>
#include <utility>

namespace ggml::hrx {
namespace {

static bool metadata_matches(const CommandPlanResourceMetadata & lhs, const CommandPlanResourceMetadata & rhs) {
    return lhs.kind == rhs.kind && lhs.size == rhs.size &&
           std::memcmp(lhs.bytes.data(), rhs.bytes.data(), lhs.size) == 0;
}

static bool generated_resource_matches(const CommandPlanGeneratedResource & lhs,
                                       const CommandPlanGeneratedResource & rhs) {
    return lhs.source_value == rhs.source_value && lhs.role == rhs.role && lhs.generated_value == rhs.generated_value &&
           lhs.byte_count == rhs.byte_count && metadata_matches(lhs.metadata, rhs.metadata);
}

static bool alternate_value_matches(const CommandPlanAlternateValue & lhs, const CommandPlanAlternateValue & rhs) {
    return lhs.graph_value == rhs.graph_value && lhs.alternate_value == rhs.alternate_value && lhs.type == rhs.type &&
           lhs.byte_count == rhs.byte_count && lhs.name == rhs.name;
}

static bool moe_routing_bundle_matches(const CommandPlanMoeRoutingBundle & lhs,
                                       const CommandPlanMoeRoutingBundle & rhs) {
    return lhs.route_ids == rhs.route_ids && lhs.route_weights == rhs.route_weights &&
           lhs.expert_table == rhs.expert_table && lhs.partition_table == rhs.partition_table &&
           lhs.expert_table_byte_count == rhs.expert_table_byte_count &&
           lhs.partition_table_byte_count == rhs.partition_table_byte_count && lhs.token_count == rhs.token_count &&
           lhs.route_count == rhs.route_count && lhs.route_stride == rhs.route_stride &&
           lhs.expert_count == rhs.expert_count;
}

}  // namespace

void CommandPlanMetadata::clear() {
    generated_resources_.clear();
    alternate_values_.clear();
    moe_routing_bundles_.clear();
}

bool CommandPlanMetadata::append(CommandPlanMetadata && other, Status & status) {
    for (CommandPlanGeneratedResource & resource : other.generated_resources_) {
        if (!append_generated_resource(std::move(resource), status)) {
            return false;
        }
    }
    for (CommandPlanAlternateValue & alternate : other.alternate_values_) {
        if (!append_alternate_value(std::move(alternate), status)) {
            return false;
        }
    }
    for (CommandPlanMoeRoutingBundle & bundle : other.moe_routing_bundles_) {
        if (!append_moe_routing_bundle(std::move(bundle), status)) {
            return false;
        }
    }
    return true;
}

bool CommandPlanMetadata::append_generated_resource(CommandPlanGeneratedResource resource, Status & status) {
    for (const CommandPlanGeneratedResource & existing : generated_resources_) {
        if (existing.source_value == resource.source_value && existing.role == resource.role) {
            if (generated_resource_matches(existing, resource)) {
                return true;
            }
            status.log("conflicting generated resource for source value %d role %d", resource.source_value.value,
                       static_cast<int>(resource.role));
            return false;
        }
    }
    generated_resources_.push_back(std::move(resource));
    return true;
}

bool CommandPlanMetadata::append_alternate_value(CommandPlanAlternateValue alternate, Status & status) {
    for (const CommandPlanAlternateValue & existing : alternate_values_) {
        if (existing.graph_value == alternate.graph_value) {
            if (alternate_value_matches(existing, alternate)) {
                return true;
            }
            status.log("conflicting alternate value for graph value %d", alternate.graph_value.value);
            return false;
        }
    }
    alternate_values_.push_back(std::move(alternate));
    return true;
}

bool CommandPlanMetadata::append_moe_routing_bundle(CommandPlanMoeRoutingBundle bundle, Status & status) {
    for (const CommandPlanMoeRoutingBundle & existing : moe_routing_bundles_) {
        if (existing.route_ids == bundle.route_ids) {
            if (moe_routing_bundle_matches(existing, bundle)) {
                return true;
            }
            status.log("conflicting MoE routing bundle for route ids value %d", bundle.route_ids.value);
            return false;
        }
    }
    moe_routing_bundles_.push_back(std::move(bundle));
    return true;
}

const CommandPlanGeneratedResource * CommandPlanMetadata::find_generated_resource(ValueId               source_value,
                                                                                  GeneratedResourceRole role) const {
    for (const CommandPlanGeneratedResource & resource : generated_resources_) {
        if (resource.source_value == source_value && resource.role == role) {
            return &resource;
        }
    }
    return nullptr;
}

const CommandPlanAlternateValue * CommandPlanMetadata::find_alternate_value(ValueId graph_value) const {
    for (const CommandPlanAlternateValue & alternate : alternate_values_) {
        if (alternate.graph_value == graph_value) {
            return &alternate;
        }
    }
    return nullptr;
}

const CommandPlanAlternateValue * CommandPlanMetadata::find_alternate_value(ValueId   graph_value,
                                                                            ggml_type type,
                                                                            size_t    byte_count) const {
    const CommandPlanAlternateValue * alternate = find_alternate_value(graph_value);
    if (alternate == nullptr || alternate->type != type || alternate->byte_count != byte_count) {
        return nullptr;
    }
    return alternate;
}

const CommandPlanMoeRoutingBundle * CommandPlanMetadata::find_moe_routing_bundle(ValueId route_ids) const {
    for (const CommandPlanMoeRoutingBundle & bundle : moe_routing_bundles_) {
        if (bundle.route_ids == route_ids) {
            return &bundle;
        }
    }
    return nullptr;
}

}  // namespace ggml::hrx
