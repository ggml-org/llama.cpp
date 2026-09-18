// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace maple_w2 {
struct StageInterval {uint64_t begin=0,end=0;uint32_t parents=0;};
struct IntervalProfile {
    uint64_t begin=0,end=0,kernel_sum=0,union_busy=0,idle=0,overlap=0;
};
// Event durations may overlap. kernel_sum is additive work, not elapsed time.
// Validate DAG edges, not submission order. Only same-backend timestamps.
inline IntervalProfile profile_intervals(const std::vector<StageInterval>& v) {
    if(v.empty()||v.size()>31)throw std::invalid_argument("invalid stage count");
    IntervalProfile p;std::vector<StageInterval> sorted=v;
    p.begin=v[0].begin;p.end=v[0].end;
    for(size_t i=0;i<v.size();++i) {
        const auto& a=v[i];
        if(a.end<a.begin || (a.parents & ~((uint32_t(1)<<i)-1)))
            throw std::invalid_argument("invalid interval or forward DAG edge");
        for(size_t j=0;j<i;++j)if((a.parents>>j)&1u)
            if(a.begin<v[j].end)throw std::invalid_argument("event violates producer dependency");
        p.begin=std::min(p.begin,a.begin);p.end=std::max(p.end,a.end);p.kernel_sum+=a.end-a.begin;
    }
    std::sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){return a.begin<b.begin;});
    auto lo=sorted[0].begin,hi=sorted[0].end;
    for(size_t i=1;i<sorted.size();++i) {
        if(sorted[i].begin>hi){p.union_busy+=hi-lo;lo=sorted[i].begin;hi=sorted[i].end;}
        else hi=std::max(hi,sorted[i].end);
    }
    p.union_busy+=hi-lo;p.idle=(p.end-p.begin)-p.union_busy;p.overlap=p.kernel_sum-p.union_busy;
    return p;
}
} // namespace maple_w2
