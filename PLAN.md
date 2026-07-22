

  # Shared-State Reasoning Sampling Implementation

  ## Summary

  Replace the two complete sampler chains with one logical chain whose eligible parameter values switch at reasoning boundaries. Preserve continuous RNG, history, adaptive,
  and Mirostat state.

  Terra drives implementation. Codex or Sol handles sampler-state and clone reviews. Luna or smaller models handle bounded searches, field inventories, documentation tables,
  and command execution. Terra remains the sole editor when work overlaps to avoid concurrent mutations.

  ## Implementation Changes

  - Replace chain_think with one top-level chain containing:
      - Section-aware wrappers for top-k, top-p, min-p, typical-p, top-n-sigma, temperature/dynatemp, penalties, and DRY.
      - One specialized section-aware XTC wrapper.
      - Exactly one terminal dist, adaptive-p, or Mirostat sampler.

  - Generic wrappers own base and reasoning children:
      - Apply only the active child.
      - Accept and reset both children so penalty and DRY histories cover the complete token timeline.
      - Clone both children and preserve the active section.

  - Before every chain application, including grammar resampling, derive the active section from the reasoning-budget sampler and update all section-aware wrappers.
  - Remove the reasoning seed, Mirostat mode/tau/eta, and adaptive target/decay fields from parameters, CLI options, server schema, tests, and documentation.
  - Continue supporting reasoning overrides for ordinary filters, temperature, penalties, DRY, XTC, and min_keep.
  - Do not change sampler topology at section boundaries. If an overridden sampler is absent from params.samplers, log that the override is inert.
  - When base Mirostat is enabled, allow only reasoning_temp; reject other reasoning overrides because their components are absent from that topology.
  - Keep completion's delayed marker configuration and current backend-sampling fallback unchanged. Missing-tag and per-turn generation-prompt lifecycle improvements remain
    separate work.

  ## Public API and Clone Behavior

  - Add:

    LLAMA_API void llama_sampler_xtc_set(
            struct llama_sampler * smpl,
            float probability,
            float threshold,
            size_t min_keep);

  - Make XTC coefficients mutable while preserving its seed, current seed, and RNG state. The setter must not reset or reseed the sampler.
  - The common-layer XTC wrapper owns one XTC sampler and switches coefficients before applying it. A disabled active profile skips application without consuming RNG.
  - After cloning the top-level chain, rediscover section wrappers through llama_sampler_chain_get() rather than copying source pointers.
  - Correct llama_sampler_penalties_clone() to copy both the token ring and token_count, enabling faithful speculative rollback with reasoning penalty overrides.
  - Leave the unrelated pre-existing Mirostat and adaptive-p clone defects outside this change.

  ## Tests and Verification

  - Update test-reasoning-overrides:
      - Remove seed, Mirostat, and adaptive reasoning-override cases.
      - Retain behavioral coverage for every supported override.
      - Add fixed-seed equal-value comparisons against no override.
      - Cover initial reasoning state, generated start/end markers, forced endings, multiple blocks, history continuity, reset, clone, and rollback.
      - Verify XTC uses one continuous RNG across boundaries.

  - Extend test-sampling with:
      - XTC coefficient changes that preserve RNG progress and clone state.
      - Penalties clone equivalence after accepted history.

  - Update argument-parser and server/chat schema tests for retained and removed fields.
  - Build llama-server, llama-completion, and the four targeted tests, then run:

    ctest --test-dir build --output-on-failure \
        -R '^(test-sampling|test-arg-parser|test-chat|test-reasoning-overrides)$'

  - Finish with static checks confirming no production references remain for chain_think, reasoning_seed, reasoning_mirostat, or reasoning_adaptive.

  ## Execution Controls and Assumptions

  - This is for a private fork, so Terra-led implementation is permitted.
  - No commit, push, PR creation, PR text, GitHub comment, or reviewer response will be produced.
  - Existing untracked prompt.md and gpt_sol_analysis.md remain untouched.
  - Terra pauses if the implementation requires a broader core sampler API or lifecycle redesign than the XTC setter and internal wrappers described above.
  - Execution starts only after explicit confirmation because this is a substantial custom-sampler refactor.
