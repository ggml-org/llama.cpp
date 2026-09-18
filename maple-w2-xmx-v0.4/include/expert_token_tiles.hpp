// SPDX-License-Identifier: MIT
#pragma once
#include "expert_grouping_reference.hpp"
#include "moe_dispatch_policy.hpp"
namespace maple_w2 {
// First/length index the stable sorted_to_job array, not token IDs. Each tile
// belongs to a single expert. The extra invalid-ID bucket is retained.
struct ExpertTokenTileView {
    int32_t *first=nullptr,*length=nullptr,*expert=nullptr,*total=nullptr;
    size_t capacity=0;
    uint32_t tokens_per_tile=1;
};
inline size_t expert_token_tile_capacity(size_t jobs,uint32_t experts,uint32_t width) {
    validate_grouping_size(jobs,experts);check_tokens_per_tile(width);
    // sum ceil(c_e/width) <= floor((jobs+(E+1)*(width-1))/width).
    return std::min(jobs,(jobs+(size_t(experts)+1)*(width-1))/width);
}
struct ExpertTokenTilesHost {std::vector<int32_t> first,length,expert;};
inline ExpertTokenTilesHost token_tiles_reference(const ExpertGroupingHost& g,uint32_t width) {
    check_tokens_per_tile(width);
    if(g.counts.size()<2||g.counts.size()>257||g.offsets.size()!=g.counts.size()+1||g.offsets[0]!=0)
        throw std::invalid_argument("invalid grouping metadata");
    ExpertTokenTilesHost r;
    int64_t count=0;
    for(size_t e=0;e<g.counts.size();++e) {
        if(g.counts[e]<0||g.offsets[e]!=count)throw std::invalid_argument("invalid grouping offsets");
        count+=g.counts[e];
        if(count>INT32_MAX||g.offsets[e+1]!=count)throw std::invalid_argument("invalid grouping count");
        for(int64_t i=0;i<g.counts[e];i+=width) {
            r.first.push_back(int32_t(g.offsets[e]+i));r.length.push_back(int32_t(std::min(int64_t(width),int64_t(g.counts[e])-i)));r.expert.push_back(int32_t(e));
        }
    }
    if(count!=int64_t(g.sorted_to_job.size()))throw std::invalid_argument("invalid grouping total");
    return r;
}
} // namespace maple_w2
