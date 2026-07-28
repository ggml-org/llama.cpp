#pragma once

#include "hf-cache.h"

#include <string>

// pure resolution of GGUF files from a repo listing, no network access:
// the primary model, its shards and the sidecar files are picked from the
// file paths alone, following the naming conventions of the known vendors

// the files to download for one model reference
struct common_download_hf_plan {
    hf_cache::hf_file primary;
    hf_cache::hf_files model_files;
    hf_cache::hf_file mmproj;
    hf_cache::hf_file mtp;
    hf_cache::hf_file eagle3;
    hf_cache::hf_file dflash;
    hf_cache::hf_file preset; // if set, only this file is downloaded
};

namespace model_resolution {

// the sidecar files requested along the primary model
struct opts {
    bool mmproj = false;
    bool mtp    = false;
    bool dflash = false;
    bool eagle3 = false;
};

// decomposition of a GGUF path into its shard prefix, quant tag and shard
// position, "m-Q8_0-00002-of-00003.gguf" gives {"m-Q8_0", "Q8_0", 2, 3}
struct gguf_split_info {
    std::string prefix; // tag included
    std::string tag;
    int index;
    int count;
};

gguf_split_info get_gguf_split_info(const std::string & path);

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
int extract_quant_bits(const std::string & filename);

// a GGUF file that is not an mmproj, imatrix or speculative sidecar
bool gguf_filename_is_model(const std::string & filepath);

// all the shards of `file`, or `file` alone when it is not sharded
hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                   const hf_cache::hf_file  & file);

// pick the best model for `tag`, or the default quant preference then the
// first model of the listing when the tag is empty
hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                  const std::string        & tag);

// pick the best sibling GGUF whose filename contains `keyword` (e.g. "mmproj" / "mtp-"),
// preferring deeper shared directory prefix with the model, then exact `tag` match,
// then closest quantization to the tag when given, or to the model otherwise
hf_cache::hf_file find_best_sibling(const hf_cache::hf_files & files,
                                    const std::string        & model,
                                    const std::string        & keyword,
                                    const std::string        & tag = "");

// build the download plan from a repo listing: a preset.ini short-circuits
// everything, an explicit `hf_file` picks that exact file, otherwise the best
// model for `tag` is picked with the requested sidecars, a requested sidecar
// resolves even without a full model at the tag, `repo` is only for the
// error messages
common_download_hf_plan resolve(const hf_cache::hf_files & files,
                                const std::string        & repo,
                                const std::string        & tag,
                                const std::string        & hf_file,
                                const opts               & o);

} // namespace model_resolution
