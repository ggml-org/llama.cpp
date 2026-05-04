#pragma once

#include "llama.h"

#include <cstdint>
#include <list>
#include <memory>
#include <vector>

enum class llama_sequence_group_status { PENDING, WAITING, RUNNING, SWAPPED, FINISHED };

using llama_block_ids = std::vector<uint32_t>;

struct llama_sequence_group {
    int32_t                     request_id = -1;
    llama_sequence_group_status status     = llama_sequence_group_status::PENDING;

    int64_t t_arrival_time   = 0;
    int64_t t_first_token_us = 0;

    uint32_t n_prompt  = 0;
    uint32_t n_decoded = 0;
    uint32_t n_past    = 0;

    std::vector<llama_token> logical_seq;
    llama_block_ids          block_table;
};

using llama_sequence_group_raw_list = std::vector<llama_sequence_group *>;
using llama_sequence_group_ptr      = std::unique_ptr<llama_sequence_group>;
using llama_sequence_group_list     = std::list<llama_sequence_group_ptr>;
