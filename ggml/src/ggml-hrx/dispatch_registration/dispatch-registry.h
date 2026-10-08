#pragma once

#include "dispatch/command-plan.h"
#include "ggml.h"
#include "graph/graph.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

class DispatchRegistry;

struct DispatchTarget {
    std::string architecture;
};

enum class DispatchMatchKind {
    Fused,
    SingleOp,
};

enum class DispatchSource {
    Common,
    Llm,
    Qwen,
};

enum DispatchActivationConsumerSource : uint32_t {
    DispatchActivationConsumerSourceCommon = 1u << 0,
    DispatchActivationConsumerSourceLlm    = 1u << 1,
    DispatchActivationConsumerSourceQwen   = 1u << 2,
    DispatchActivationConsumerSourceAll    = DispatchActivationConsumerSourceCommon |
                                             DispatchActivationConsumerSourceLlm |
                                             DispatchActivationConsumerSourceQwen,
};

struct DispatchMatchContext {
    const Graph &             graph;
    const GraphNode *         root_node  = nullptr;
    size_t                    root_index = 0;
    const std::vector<bool> & covered_nodes;
    const CommandPlan &       plan;
    ValueId                   next_plan_value;
    const DispatchRegistry *  registry = nullptr;
};

enum DispatchActivationInputFormat : uint32_t {
    DispatchActivationInputNone           = 0,
    DispatchActivationInputF16Row         = 1u << 0,
    DispatchActivationInputF16K16Major    = 1u << 1,
    DispatchActivationInputQ8_1X4         = 1u << 2,
    DispatchActivationInputSymmetricI4K32 = 1u << 3,
};

enum DispatchActivationProducerRequirement : uint32_t {
    DispatchActivationProducerRequirementNone             = 0,
    DispatchActivationProducerRequirementPackedK16Economy = 1u << 0,
    DispatchActivationProducerRequirementFactoredTiledF16 = 1u << 1,
};

enum DispatchActivationConsumerTrait : uint32_t {
    DispatchActivationConsumerTraitNone                   = 0,
    DispatchActivationConsumerTraitQ8PublicationPreferred = 1u << 0,
};

enum class DispatchActivationConsumerUse {
    BandwidthLimited,
    Tiled,
    Row,
    LowRow,
};

using DispatchActivationConsumerPredicate = bool (*)(const DispatchMatchContext & context,
                                                      const GraphNode &            consumer,
                                                      const Value &                input);

struct DispatchActivationConsumerRegistration {
    const char *                        name = "";
    ggml_op                             root_op = GGML_OP_NONE;
    uint32_t                            accepted_formats = DispatchActivationInputNone;
    DispatchActivationConsumerUse       use = DispatchActivationConsumerUse::Row;
    uint32_t                            producer_requirements = DispatchActivationProducerRequirementNone;
    DispatchSource                      source = DispatchSource::Common;
    DispatchActivationConsumerPredicate accepts = nullptr;
    uint32_t                            traits = DispatchActivationConsumerTraitNone;
};

struct DispatchValueAliasRequest {
    ValueId source_value;
    ValueId target_value;
};

struct DispatchMatch {
    std::vector<Dispatch>                            initialization_dispatches;
    std::vector<size_t>                              covered_nodes;
    std::vector<Dispatch>                            dispatches;
    std::vector<CommandPlanTransient>                transients;
    std::vector<CommandPlanConstantInitialization>   constant_initializations;
    std::vector<CommandPlanCompletionCounterRequest> completion_counter_requests;
    std::vector<DispatchValueAliasRequest>           value_aliases;
    CommandPlanMetadata                              metadata;
    Status                                           status;
};

using DispatchMatcher = bool (*)(const DispatchMatchContext & context, DispatchMatch & match);

struct DispatchRegistration {
    const char *      name     = "";
    ggml_op           root_op  = GGML_OP_NONE;
    DispatchMatchKind kind     = DispatchMatchKind::SingleOp;
    int               priority = 0;
    DispatchSource    source   = DispatchSource::Common;
    DispatchMatcher   matcher  = nullptr;
};

struct DispatchRegistrationAttempt {
    std::string              name;
    ggml_op                  root_op  = GGML_OP_NONE;
    DispatchMatchKind        kind     = DispatchMatchKind::SingleOp;
    int                      priority = 0;
    DispatchSource           source   = DispatchSource::Common;
    bool                     matched  = false;
    std::vector<size_t>      covered_nodes;
    std::vector<std::string> errors;
};

struct DispatchMatchDiagnostics {
    ggml_op                                  root_op = GGML_OP_NONE;
    std::vector<DispatchRegistrationAttempt> attempts;
};

class DispatchRegistry {
  public:
    bool match(const DispatchMatchContext & context, DispatchMatch & match) const;
    bool match(const DispatchMatchContext & context,
               DispatchMatch &              match,
               DispatchMatchDiagnostics *   diagnostics) const;

    const std::vector<DispatchRegistration> & registrations_for_root(ggml_op root_op) const;

    const std::vector<DispatchRegistration> & single_op_registrations() const { return single_op_registrations_; }

    const std::vector<DispatchActivationConsumerRegistration> &
    activation_consumers_for_root(ggml_op root_op) const;

  private:
    friend class DispatchRegistryBuilder;

    struct RegistrationGroup {
        std::vector<DispatchRegistration> fused;
        std::vector<DispatchRegistration> single_op;
        std::vector<DispatchRegistration> ordered;
    };

    std::vector<RegistrationGroup>    registrations_by_root_;
    std::vector<DispatchRegistration> single_op_registrations_;
    std::vector<std::vector<DispatchActivationConsumerRegistration>> activation_consumers_by_root_;
};

class DispatchRegistryBuilder {
  public:
    void             add(DispatchRegistration registration);
    void             add_activation_consumer(DispatchActivationConsumerRegistration registration);
    DispatchRegistry build();

  private:
    DispatchRegistry registry_;
};

const DispatchRegistry * find_dispatch_registry(const DispatchTarget & target);

}  // namespace ggml::hrx
