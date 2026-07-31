#pragma once

//
// tessera-higgs-cache.h
//
// Content-addressed cache for HIGGS alpha_l coefficients. Keyed by
// SHA-256 of the BF16 weight data (all layers concatenated). Cache
// files are JSON: {"hash": "...", "n_layers": N, "alpha": [...],
// "timestamp": "..."}. Default path: ~/.cache/tessera/higgs_alpha/.
//

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

struct ts_higgs_cache_key {
    uint8_t     hash[32];
    std::string hex;    // 64-char lowercase hex string
};

// Compute cache key from concatenated weight data (all layers).
// weights: array of n_layers pointers to weight matrices.
// out_dims, in_dims: per-layer dimensions.
ts_higgs_cache_key ts_higgs_cache_compute_key(const float ** weights,
                                              const int64_t * out_dims,
                                              const int64_t * in_dims,
                                              int64_t n_layers);

// Compute cache key from a flat byte buffer (e.g. raw BF16 data).
ts_higgs_cache_key ts_higgs_cache_key_from_bytes(const void * data, size_t len);

// Default cache directory: ~/.cache/tessera/higgs_alpha/
std::string ts_higgs_cache_default_dir();

// Store alpha vector to a JSON cache file under cache_dir.
// File is named <hash_hex>.json. Creates cache_dir if needed.
// Returns 0 on success.
int ts_higgs_cache_store(const ts_higgs_cache_key * key,
                         const float * alpha, int64_t n_layers,
                         const std::string * cache_dir);

// Load alpha vector from cache. Returns nullopt on miss (no file,
// hash mismatch, or parse error). On hit, returns the alpha vector.
std::optional<std::vector<float>> ts_higgs_cache_load(
    const ts_higgs_cache_key * key,
    const std::string * cache_dir);
