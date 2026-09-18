// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace maple_w2 {
// Stable order: (expert_id, original token/selection index). Bad IDs go in a
// final bucket, NOT to a valid expert. Existing matvec status handling rejects them.
struct ExpertGroupingHost {
    std::vector<int32_t> counts, offsets, sorted_to_job;
};
inline uint32_t grouping_key(int32_t id,uint32_t experts) {
    return id>=0 && uint32_t(id)<experts ? uint32_t(id) : experts;
}
inline void validate_grouping_size(size_t jobs,uint32_t experts) {
    if(!jobs || jobs>size_t(std::numeric_limits<int32_t>::max()) || !experts || experts>256)
        throw std::invalid_argument("grouping needs 1..INT32_MAX jobs and 1..256 experts");
}
inline ExpertGroupingHost group_jobs_reference(const std::vector<int32_t>& ids,uint32_t experts) {
    validate_grouping_size(ids.size(),experts);
    ExpertGroupingHost r;
    r.counts.assign(experts+1,0);r.offsets.assign(experts+2,0);r.sorted_to_job.resize(ids.size());
    for(auto id:ids)++r.counts[grouping_key(id,experts)];
    for(uint32_t e=0;e<=experts;++e)r.offsets[e+1]=r.offsets[e]+r.counts[e];
    auto cursor=r.offsets;
    for(size_t j=0;j<ids.size();++j)r.sorted_to_job[size_t(cursor[grouping_key(ids[j],experts)]++)]=int32_t(j);
    return r;
}
// CPU emulation of the GPU's per-expert workgroup scans, deliberately separate
// from the reference counting-sort cursor implementation above.
inline ExpertGroupingHost group_jobs_chunk_emulation(const std::vector<int32_t>& ids,uint32_t experts,size_t lanes=128) {
    validate_grouping_size(ids.size(),experts);if(!lanes)throw std::invalid_argument("zero lanes");
    ExpertGroupingHost r;
    r.counts.assign(experts+1,0);r.offsets.assign(experts+2,0);r.sorted_to_job.assign(ids.size(),-1);
    for(uint32_t e=0;e<=experts;++e)for(size_t lane=0;lane<lanes;++lane)
        for(size_t j=lane;j<ids.size();j+=lanes)r.counts[e]+=grouping_key(ids[j],experts)==e;
    for(uint32_t e=0;e<=experts;++e)r.offsets[e+1]=r.offsets[e]+r.counts[e];
    for(uint32_t e=0;e<=experts;++e) {
        int32_t running=r.offsets[e];
        for(size_t first=0;first<ids.size();first+=lanes) {
            int32_t rank=0;
            for(size_t lane=0;lane<lanes;++lane) {
                const size_t j=first+lane;
                if(j<ids.size() && grouping_key(ids[j],experts)==e)r.sorted_to_job[size_t(running+rank++)]=int32_t(j);
            }
            running+=rank;
        }
    }
    return r;
}
inline bool same_grouping(const ExpertGroupingHost&a,const ExpertGroupingHost&b) {
    return a.counts==b.counts&&a.offsets==b.offsets&&a.sorted_to_job==b.sorted_to_job;
}
} // namespace maple_w2
