#pragma once

#include "dispatch.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace ggml::hrx {

enum class GeneratedResourceRole {
    MoeExpertTable,
    MoePartitionTable,
};

enum class CommandPlanResourceMetadataKind {
    None,
    MoeRoutingResource,
};

struct MoeRoutingResourceMetadata {
    int64_t token_count  = 0;
    int64_t route_count  = 0;
    int64_t route_stride = 0;
    int64_t expert_count = 0;
};

template <typename T> constexpr CommandPlanResourceMetadataKind command_plan_resource_metadata_kind() {
    static_assert(sizeof(T) == 0, "unsupported command plan resource metadata type");
    return CommandPlanResourceMetadataKind::None;
}

template <>
constexpr CommandPlanResourceMetadataKind command_plan_resource_metadata_kind<MoeRoutingResourceMetadata>() {
    return CommandPlanResourceMetadataKind::MoeRoutingResource;
}

struct CommandPlanResourceMetadata {
    static constexpr size_t kMaxBytes = 64;

    CommandPlanResourceMetadataKind kind                           = CommandPlanResourceMetadataKind::None;
    size_t                          size                           = 0;
    alignas(std::max_align_t) std::array<uint8_t, kMaxBytes> bytes = {};

    template <typename T> bool read(T & value) const {
        static_assert(std::is_trivially_copyable<T>::value, "metadata payload must be trivially copyable");
        if (kind != command_plan_resource_metadata_kind<T>() || size != sizeof(T)) {
            return false;
        }
        std::memcpy(&value, bytes.data(), sizeof(T));
        return true;
    }
};

template <typename T> CommandPlanResourceMetadata make_command_plan_resource_metadata(const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "metadata payload must be trivially copyable");
    static_assert(sizeof(T) <= CommandPlanResourceMetadata::kMaxBytes, "metadata payload is too large");

    CommandPlanResourceMetadata metadata;
    metadata.kind = command_plan_resource_metadata_kind<T>();
    metadata.size = sizeof(T);
    std::memcpy(metadata.bytes.data(), &value, sizeof(T));
    return metadata;
}

struct CommandPlanGeneratedResource {
    ValueId                     source_value;
    GeneratedResourceRole       role = GeneratedResourceRole::MoeExpertTable;
    ValueId                     generated_value;
    size_t                      byte_count = 0;
    CommandPlanResourceMetadata metadata;
};

struct CommandPlanAlternateValue {
    ValueId     graph_value;
    ValueId     alternate_value;
    ggml_type   type       = GGML_TYPE_COUNT;
    size_t      byte_count = 0;
    std::string name;
};

struct CommandPlanMoeRoutingBundle {
    ValueId route_ids;
    ValueId route_weights;
    ValueId expert_table;
    ValueId partition_table;
    size_t  expert_table_byte_count    = 0;
    size_t  partition_table_byte_count = 0;
    int64_t token_count                = 0;
    int64_t route_count                = 0;
    int64_t route_stride               = 0;
    int64_t expert_count               = 0;
};

class CommandPlanMetadata {
  public:
    void clear();

    bool append(CommandPlanMetadata && other, Status & status);

    bool append_generated_resource(CommandPlanGeneratedResource resource, Status & status);

    bool append_alternate_value(CommandPlanAlternateValue alternate, Status & status);

    bool append_moe_routing_bundle(CommandPlanMoeRoutingBundle bundle, Status & status);

    const CommandPlanGeneratedResource * find_generated_resource(ValueId               source_value,
                                                                 GeneratedResourceRole role) const;

    const CommandPlanAlternateValue * find_alternate_value(ValueId graph_value) const;

    const CommandPlanAlternateValue * find_alternate_value(ValueId   graph_value,
                                                           ggml_type type,
                                                           size_t    byte_count) const;

    const CommandPlanMoeRoutingBundle * find_moe_routing_bundle(ValueId route_ids) const;

    const std::vector<CommandPlanGeneratedResource> & generated_resources() const { return generated_resources_; }

    const std::vector<CommandPlanAlternateValue> & alternate_values() const { return alternate_values_; }

    const std::vector<CommandPlanMoeRoutingBundle> & moe_routing_bundles() const { return moe_routing_bundles_; }

  private:
    std::vector<CommandPlanGeneratedResource> generated_resources_;
    std::vector<CommandPlanAlternateValue>    alternate_values_;
    std::vector<CommandPlanMoeRoutingBundle>  moe_routing_bundles_;
};

}  // namespace ggml::hrx
