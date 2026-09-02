#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// A provider profile owns model-specific compatibility and artifact details.
// The speculative driver only selects a profile and invokes its generic
// capabilities; it does not contain model dimensions or tensor identities.
enum common_spec_sidecar_kind {
    COMMON_SPEC_SIDECAR_KIND_MTP = 1,
    COMMON_SPEC_SIDECAR_KIND_DFLASH = 2,
};

struct common_spec_sidecar_profile;
using common_spec_sidecar_model_match_fn = bool (*)(
        const common_spec_sidecar_profile &, const llama_model *, std::string &);
using common_spec_sidecar_file_match_fn = bool (*)(
        const common_spec_sidecar_profile &, const std::string &, std::string &);

struct common_spec_sidecar_profile {
    const char * name = nullptr;
    common_spec_sidecar_kind kind = COMMON_SPEC_SIDECAR_KIND_MTP;

    // Strong target identity and shape contract. A negative nextn count means
    // that the profile does not constrain the target auxiliary-layer count.
    const char * target_architecture = nullptr;
    const char * target_name = nullptr;
    const char * target_size_label = nullptr;
    int32_t target_n_embd = 0;
    int32_t target_n_embd_out = 0;
    int32_t target_n_layer = 0;
    int32_t target_n_layer_nextn = -1;
    int32_t target_n_vocab = 0;

    // Provider capabilities consumed by the matching sidecar implementation.
    int32_t mtp_embedding_width = 0;
    int32_t mtp_head_rows = 0;
    int32_t dflash_encoded_width = 0;
    int32_t dflash_decoder_width = 0;
    int32_t dflash_block_size = 0;
    int32_t dflash_selector_top_k = 0;
    int32_t dflash_head_rows = 0;
    const int32_t * dflash_target_layer_ids = nullptr;
    uint32_t dflash_target_layer_ids_n = 0;

    // Configuration names are provider-owned so the core does not encode
    // artifact naming conventions. Existing neutral names are retained.
    const char * library_env = nullptr;
    const char * artifact_env = nullptr;
    const char * ids_env = nullptr;
    const char * full_head_env = nullptr;
    const char * default_library_name = nullptr;
    const char * default_artifact_dir_name = nullptr;
    // Experimental providers can require explicit paths so a colocated DLL
    // cannot silently replace a faster native path before qualification.
    bool explicit_paths_only = false;

    common_spec_sidecar_model_match_fn matches_model = nullptr;
    common_spec_sidecar_file_match_fn matches_target_file = nullptr;
};

struct common_spec_sidecar_paths {
    std::string library;
    std::string artifact_dir;
    std::string ids;
    bool dflash_full_head = false;
};

size_t common_spec_sidecar_profile_count();
const common_spec_sidecar_profile * common_spec_sidecar_profile_at(size_t index);
bool common_spec_sidecar_profile_name_matches(
        const common_spec_sidecar_profile & profile, const char * name);

const common_spec_sidecar_profile * common_spec_sidecar_profile_for_model(
        common_spec_sidecar_kind kind, const llama_model * model, std::string & error);
const common_spec_sidecar_profile * common_spec_sidecar_profile_for_target_file(
        common_spec_sidecar_kind kind, const std::string & path, std::string & error);
bool common_spec_sidecar_get_library(const common_spec_sidecar_profile & profile,
        std::string & library, std::string & error);
bool common_spec_sidecar_get_paths(const common_spec_sidecar_profile & profile,
        common_spec_sidecar_paths & paths, std::string & error);
bool common_spec_sidecar_validate_artifacts(const common_spec_sidecar_profile & profile,
        const common_spec_sidecar_paths & paths, std::string & error);
bool common_spec_sidecar_probe(const common_spec_sidecar_profile & profile,
        const common_spec_sidecar_paths & paths, uint32_t n_seq, std::string & error);
bool common_spec_sidecar_probe(const common_spec_sidecar_profile & profile,
        uint32_t n_seq, std::string & error);

// Probe only the sidecar binary/artifact contract. This intentionally performs
// no HIP initialization, model loading, or device allocation.
bool common_spec_sidecar_mtp_probe(const std::string & library_path,
        const std::string & weights_dir, const std::string & ids_path,
        int32_t embedding_width, int32_t head_rows, int32_t n_seq,
        std::string & error);
bool common_spec_sidecar_dflash_probe(const std::string & library_path,
        const std::string & artifact_dir, int32_t encoded_width,
        int32_t block_size, int32_t n_seq, std::string & error);

// Host-side loader for optional model-specific speculative sidecars.
// The sidecars are deliberately opt-in and model-specific. A loader object
// keeps the library resident for the lifetime of the process because the
// current release ABI has no shutdown operation. State/KV calls are serialized
// by the speculative driver and sequence-scoped in the release ABI.
class common_spec_sidecar_mtp {
public:
    common_spec_sidecar_mtp();
    ~common_spec_sidecar_mtp();

    common_spec_sidecar_mtp(const common_spec_sidecar_mtp &) = delete;
    common_spec_sidecar_mtp & operator=(const common_spec_sidecar_mtp &) = delete;

    bool load(const std::string & library_path,
              const std::string & weights_dir,
              const std::string & ids_path,
              int32_t embedding_width,
              int32_t head_rows,
              int32_t n_seq,
              int32_t max_context,
              std::string & error,
              int32_t device = -1);

    bool active() const;
    void disable();

    bool get_state(int32_t seq_id, std::vector<uint8_t> & data) const;
    bool set_state(int32_t seq_id, const std::vector<uint8_t> & data) const;
    bool reset_state(int32_t seq_id) const;
    bool truncate_state(int32_t seq_id, int32_t pos_max) const;
    bool commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) const;
    bool rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const;
    bool attach_target_stream(void * stream, int32_t device) const;

    int catchup(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
                const float * hidden_rows, int count) const;
    int catchup_device(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
                       const float * hidden_rows_device, int count) const;
    int draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
              const float * hidden, int max_draft, int32_t * output_ids) const;
    int draft_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
                     int max_draft, int32_t * output_ids) const;
    int draft_stochastic(int32_t seq_id, int32_t last_token, int32_t past_tokens,
                         const float * hidden, float temperature, float p_min,
                         uint64_t rng_key, int max_draft, int32_t * output_ids,
                         int32_t * dist_ids, float * dist_probs) const;
    int draft_stochastic_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
                                float temperature, float p_min, uint64_t rng_key,
                                int max_draft, int32_t * output_ids,
                                int32_t * dist_ids, float * dist_probs) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class common_spec_sidecar_dflash {
public:
    common_spec_sidecar_dflash();
    ~common_spec_sidecar_dflash();

    common_spec_sidecar_dflash(const common_spec_sidecar_dflash &) = delete;
    common_spec_sidecar_dflash & operator=(const common_spec_sidecar_dflash &) = delete;

    bool load(const std::string & library_path,
              const std::string & artifact_dir,
              int32_t encoded_width,
              int32_t block_size,
              int32_t n_seq,
              int32_t max_context,
              std::string & error);

    bool active() const;
    void disable();

    bool get_state(int32_t seq_id, std::vector<uint8_t> & data) const;
    bool set_state(int32_t seq_id, const std::vector<uint8_t> & data) const;
    bool reset_state(int32_t seq_id) const;
    bool truncate_state(int32_t seq_id, int32_t pos_max) const;
    bool commit_state(int32_t seq_id, int32_t pos_max) const;
    bool rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const;
    bool attach_target_stream(void * stream, int32_t device) const;

    int chunk(int32_t seq_id, const int32_t * positions,
              const float * target_features, int count) const;
    int chunk_device(int32_t seq_id, const int32_t * positions,
                     const void * const * target_layer_features_device,
                     int n_layers, int layer_width, int count) const;
    int draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
              int32_t * output_ids) const;
    int draft_stochastic(int32_t seq_id, int32_t last_token, int32_t past_tokens,
                         float temperature, float p_min, uint64_t rng_key,
                         int max_draft, int32_t * output_ids,
                         int32_t * dist_ids, float * dist_probs) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
