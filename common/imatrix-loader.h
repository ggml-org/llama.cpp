#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

struct common_imatrix_entry {
    std::vector<float>   sums;
    std::vector<int64_t> counts;
};

struct common_imatrix {
    std::map<std::string, common_imatrix_entry> entries;
    std::vector<std::string> datasets;
    int32_t chunk_count = 0;
    int32_t chunk_size  = 0;
    bool    is_legacy   = false;
};

bool common_imatrix_load(const std::string & fname, common_imatrix & imatrix);
