// SPDX-License-Identifier: MIT
#include "expert_grouping.hpp"
namespace maple_w2 {
class ExpertGroupCount;class ExpertGroupPrefix;class ExpertGroupScatter;
ExpertGroupingRun enqueue_expert_grouping(sycl::queue&q,const int32_t* ids,size_t jobs,uint32_t experts,
                                          ExpertGroupingWorkspace w,const std::vector<sycl::event>& deps) {
    validate_grouping_size(jobs,experts);
    if(!ids||!w.counts||!w.offsets||!w.sorted_to_job||w.job_capacity<jobs||w.expert_capacity<experts)
        throw std::invalid_argument("missing/undersized persistent expert-grouping workspace");
    constexpr size_t lanes=128;
    const size_t buckets=size_t(experts)+1;
    ExpertGroupingRun r;
    r.count=q.submit([&](sycl::handler&h) {
        h.depends_on(deps);
        h.parallel_for<ExpertGroupCount>(sycl::nd_range<1>{sycl::range<1>(buckets*lanes),sycl::range<1>(lanes)},
            [=](sycl::nd_item<1>it) {
                const uint32_t e=uint32_t(it.get_group_linear_id());
                int32_t n=0;
                for(size_t j=it.get_local_linear_id();j<jobs;j+=lanes) {
                    const int32_t id=ids[j];const uint32_t key=id>=0&&uint32_t(id)<experts?uint32_t(id):experts;
                    n+=int32_t(key==e);
                }
                n=sycl::reduce_over_group(it.get_group(),n,sycl::plus<int32_t>());
                if(it.get_local_linear_id()==0)w.counts[e]=n;
            });
    });
    r.prefix=q.submit([&](sycl::handler&h) {
        h.depends_on(r.count);
        // E<=256. Tiny deterministic prefix: independent item for each offset.
        // No atomics or cross-workgroup spin waits.
        h.parallel_for<ExpertGroupPrefix>(sycl::range<1>(buckets+1),[=](sycl::id<1>ii) {
            const size_t e=ii[0];int32_t begin=0;
            for(size_t b=0;b<e;++b)begin+=w.counts[b];
            w.offsets[e]=begin;
        });
    });
    r.scatter=q.submit([&](sycl::handler&h) {
        h.depends_on(r.prefix);
        h.parallel_for<ExpertGroupScatter>(sycl::nd_range<1>{sycl::range<1>(buckets*lanes),sycl::range<1>(lanes)},
            [=](sycl::nd_item<1>it) {
                const uint32_t e=uint32_t(it.get_group_linear_id());
                const size_t lane=it.get_local_linear_id();
                // Uniform for the whole workgroup: no divergent collective use.
                if(w.counts[e]==0)return;
                int32_t running=w.offsets[e];
                for(size_t first=0;first<jobs;first+=lanes) {
                    const size_t j=first+lane;int32_t hit=0;
                    if(j<jobs) {
                        const int32_t id=ids[j];const uint32_t key=id>=0&&uint32_t(id)<experts?uint32_t(id):experts;
                        hit=int32_t(key==e);
                    }
                    const int32_t rank=sycl::exclusive_scan_over_group(it.get_group(),hit,sycl::plus<int32_t>());
                    const int32_t n=sycl::reduce_over_group(it.get_group(),hit,sycl::plus<int32_t>());
                    if(hit)w.sorted_to_job[size_t(running+rank)]=int32_t(j);
                    running+=n;
                }
            });
    });
    return r;
}
} // namespace maple_w2

namespace maple_w2 {
class ExpertTokenTiles;
sycl::event enqueue_expert_token_tiles(sycl::queue& q,size_t jobs,uint32_t experts,
    ExpertGroupingWorkspace g,ExpertTokenTileView w,const std::vector<sycl::event>& deps) {
    const size_t bound=expert_token_tile_capacity(jobs,experts,w.tokens_per_tile);
    if(!g.counts||!g.offsets||g.expert_capacity<experts||g.job_capacity<jobs||
       !w.first||!w.length||!w.expert||!w.total||w.capacity<bound)
        throw std::invalid_argument("missing/undersized expert token-tile workspace");
    const uint32_t width=w.tokens_per_tile;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for<ExpertTokenTiles>(sycl::range<1>(size_t(experts)+1),[=](sycl::id<1> i) {
            const uint32_t e=uint32_t(i[0]);int32_t first=0;
            for(uint32_t b=0;b<e;++b)first+=g.counts[b]/int32_t(width)+int32_t(g.counts[b]%int32_t(width)!=0);
            const int32_t count=g.counts[e],n=count/int32_t(width)+int32_t(count%int32_t(width)!=0);
            for(int32_t t=0;t<n;++t) {
                const size_t j=size_t(first+t);
                w.first[j]=g.offsets[e]+t*int32_t(width);
                w.length[j]=sycl::min(int32_t(width),count-t*int32_t(width));w.expert[j]=int32_t(e);
            }
            if(e==experts)*w.total=first+n;
        });
    });
}
} // namespace maple_w2
