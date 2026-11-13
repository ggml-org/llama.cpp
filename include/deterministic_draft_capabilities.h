// deterministic_draft_capabilities.h -- Shared capability flags for deterministic draft plugins
//
// This header defines the capability bitmask values used by both the plugin
// contract (deterministic_draft_plugin.h) and the host/consumer APIs
// (llama.h, llama_deterministic_draft.h).  Keeping the values in one place
// prevents the three copies from drifting out of sync.

#ifndef DETERMINISTIC_DRAFT_CAPABILITIES_H
#define DETERMINISTIC_DRAFT_CAPABILITIES_H

// Capability flags (bitmask)
//
// The names describe the mechanism-level contract, not a use case. A plugin
// advertises any combination via deterministic_draft_get_capabilities() and
// the host degrades gracefully when a flag is absent. Unassigned bits are
// reserved for future capabilities; plugins must not define their own.
#define DETERMINISTIC_DRAFT_CAPABILITY_BITMASK      (1u << 1)
#define DETERMINISTIC_DRAFT_CAPABILITY_JUMP_FORWARD (1u << 2)

// Actions a plugin may request from deterministic_draft_filter_draft() through
// its rest_action out-parameter. The host pre-sets ENFORCE, so a plugin that
// never writes the parameter behaves exactly as it did before the parameter
// existed.
//
// ENFORCE - the plugin's verdict stands: the host keeps the returned prefix and
//           drops the rest of the draft.
// SKIP    - the plugin declines the round: its rule does not describe the
//           current content. The host ignores the returned prefix, keeps the
//           whole draft and verifies it with the target model, so the round
//           behaves like ordinary speculative decoding.
#define DETERMINISTIC_DRAFT_REST_ENFORCE 0
#define DETERMINISTIC_DRAFT_REST_SKIP    1

#endif // DETERMINISTIC_DRAFT_CAPABILITIES_H
