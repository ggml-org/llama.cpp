#pragma once

#include "llama-memory-plan.h"

enum class llama_memory_budget_component_kind {
    unspecified,
    compute_workspace,
    kv_pool,
    kv_attention_workspace,
    kv_writer_workspace,
    kv_graph_workspace,
    backend_scratch,
    executable,
    model_weights,
    output,
    host_kv,
    driver,
    other,
};

enum class llama_memory_budget_location {
    shared_parent,
    external_device,
    external_host,
    alias,
};

struct llama_memory_budget_stage_size {
    llama_memory_stage_id stage = 0;
    size_t                bytes = 0;
};

struct llama_memory_budget_component {
    uint64_t                                    id               = 0;
    llama_memory_budget_component_kind          kind             = llama_memory_budget_component_kind::unspecified;
    llama_memory_budget_location                location         = llama_memory_budget_location::external_device;
    llama_memory_allocation_class               allocation_class = LLAMA_MEMORY_ALLOCATION_HOST;
    size_t                                      alignment        = 1;
    // Inexact external bytes are a known lower bound. Shared and alias components must be exact.
    bool                                        exact            = true;
    uint64_t                                    alias_of         = 0;
    std::vector<llama_memory_budget_stage_size> stages;
};

struct llama_memory_budget_plan {
    llama_memory_allocation_class              shared_allocation = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
    std::vector<llama_memory_stage_id>         stages;
    std::vector<llama_memory_budget_component> components;
};

struct llama_memory_budget_stage_report {
    llama_memory_stage_id stage                              = 0;
    size_t                shared_payload_bytes               = 0;
    size_t                shared_required_bytes              = 0;
    size_t                shared_alignment_bytes             = 0;
    size_t                external_device_known_bytes        = 0;
    size_t                external_managed_allocation_bytes  = 0;
    size_t                external_host_known_bytes          = 0;
    size_t                unknown_external_device_components = 0;
    size_t                unknown_external_host_components   = 0;
};

struct llama_memory_budget_report {
    std::vector<llama_memory_budget_stage_report> stages;
    size_t                                        shared_parent_minimum     = 0;
    bool                                          device_residency_complete = true;
    bool                                          host_complete             = true;
};

enum class llama_memory_budget_status {
    success,
    invalid_stage,
    duplicate_stage,
    invalid_component,
    duplicate_component,
    invalid_alignment,
    invalid_allocation,
    inexact_shared_component,
    invalid_alias,
    invalid_measurement,
    overflow,
    allocation_failed,
};

struct llama_memory_budget_result {
    llama_memory_budget_status status    = llama_memory_budget_status::success;
    uint64_t                   component = 0;
    llama_memory_stage_id      stage     = 0;
};

struct llama_memory_budget_observation {
    llama_memory_stage_id stage                 = 0;
    size_t                observed_device_bytes = 0;
};

struct llama_memory_budget_reconciliation_stage {
    llama_memory_stage_id stage                     = 0;
    size_t                observed_device_bytes     = 0;
    size_t                declared_device_bytes     = 0;
    size_t                unclassified_device_bytes = 0;
};

struct llama_memory_budget_reconciliation {
    std::vector<llama_memory_budget_reconciliation_stage> stages;
    size_t                                                maximum_observed_device_bytes     = 0;
    size_t                                                maximum_unclassified_device_bytes = 0;
};

// Compare process-scoped observed device bytes with one allocated parent plus known external requests.
// Residual bytes stay explicit; failure leaves output unchanged.
llama_memory_budget_result llama_memory_budget_reconcile(
    const llama_memory_budget_report &                   report,
    size_t                                               shared_parent_capacity,
    const std::vector<llama_memory_budget_observation> & observations,
    llama_memory_budget_reconciliation &                 output);

struct llama_memory_budget_phase_bytes {
    size_t prefill    = 0;
    size_t decode     = 0;
    size_t transition = 0;
};

struct llama_text_memory_budget_input {
    llama_memory_stage_id           prefill_stage              = 0;
    llama_memory_stage_id           decode_stage               = 0;
    llama_memory_stage_id           transition_stage           = 0;
    size_t                          compute_prefill_bytes      = 0;
    size_t                          compute_decode_bytes       = 0;
    size_t                          compute_alignment          = 1;
    size_t                          kv_pool_bytes              = 0;
    size_t                          kv_writer_bytes            = 0;
    size_t                          kv_attention_prefill_bytes = 0;
    size_t                          kv_attention_decode_bytes  = 0;
    size_t                          kv_alignment               = 1;
    size_t                          host_kv_bytes              = 0;
    size_t                          host_alignment             = 1;
    size_t                          output_bytes               = 0;
    size_t                          model_weight_bytes         = 0;
    llama_memory_allocation_class   model_weight_allocation    = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
    llama_memory_budget_phase_bytes backend_scratch;
    llama_memory_budget_phase_bytes executable;
    llama_memory_budget_phase_bytes driver;
};

// Build the initial serial text inventory with shared components inactive during transition.
// The stable parent remains allocated and external transition bytes are explicit.
llama_memory_budget_result llama_text_memory_budget_plan_make(const llama_text_memory_budget_input & input,
                                                              llama_memory_budget_plan &             plan,
                                                              llama_memory_budget_report &           report);
// Account requested bytes only. Backend/driver physical rounding remains external unless an adapter reports it.
// Stage entries are explicit so a transition peak can differ from both steady phases.
// Failure leaves output unchanged.
llama_memory_budget_result llama_memory_budget_plan_make(const llama_memory_budget_plan & plan,
                                                         llama_memory_budget_report &     output);
