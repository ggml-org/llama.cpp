// SPDX-License-Identifier: MIT
// Opt-in port of Vulkan's weight-tile reuse, NOT a copy of its dot instruction.
// s2tile8/G32 only. B is loaded once per K32 and reused for 2/4 token rows;
// every row still executes the proven DG2 s2*s8 RepeatCount=1 DPAS primitive.
#include "maple_w2a8.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
namespace maple_w2 {
namespace es=sycl::ext::intel::esimd;
namespace xmx=sycl::ext::intel::esimd::xmx;
template<int T,bool Pair>class W2TokenReuse;
template<int T,bool Pair>class W2TokenReuseReduce;
template<int T,bool Pair>
static A8Run launch_tiles(sycl::queue& q,DeviceProblem p,A8Options o,A8Workspace a,float* scratch,
                         const std::vector<sycl::event>& deps) {
    const size_t jobs=size_t(p.tokens)*p.topk,outn=jobs*p.shape.m;
    const size_t tiles=expert_token_tile_capacity(jobs,p.shape.experts,T);
    const uint32_t rtiles=p.shape.m/8,split=o.kernel.split_k;
    const size_t logical=tiles*rtiles*split;
    const size_t global=((logical+o.kernel.local_size-1)/o.kernel.local_size)*o.kernel.local_size;
    A8Run run;
    run.main=q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for<W2TokenReuse<T,Pair>>(
          sycl::nd_range<1>{sycl::range<1>(global),sycl::range<1>(o.kernel.local_size)},
          [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            size_t idx=it.get_global_linear_id();if(idx>=logical)return;
            const uint32_t sp=uint32_t(idx%split);idx/=split;
            const uint32_t rt=uint32_t(idx%rtiles);const size_t tile=idx/rtiles;
            const int32_t total=*o.token_tiles.total;
            // Descriptors are produced on GPU; no dynamic-count host readback.
            if(total<0||size_t(total)>tiles){if(idx==0&&sp==0)p.status[0]=2;return;}
            if(tile>=size_t(total))return;
            const int32_t start=o.token_tiles.first[tile],len=o.token_tiles.length[tile];
            const int32_t expert=o.token_tiles.expert[tile];
            if(start<0||len<1||len>T||size_t(start)+size_t(len)>jobs||expert<0||uint32_t(expert)>p.shape.experts) {
                // Unique first work item avoids flag races. Host must reject status.
                if(rt==0&&sp==0)p.status[tile]=2;return;
            }
            size_t job[T];bool valid[T];es::simd<float,8> acc0[T],acc1[T];
            #pragma unroll
            for(int t=0;t<T;++t) {
                job[t]=0;valid[t]=false;acc0[t]=0.f;acc1[t]=0.f;
                if(t<len) {
                    const int32_t j=o.job_order[size_t(start)+t];
                    if(j<0||size_t(j)>=jobs) {if(rt==0&&sp==0)p.status[tile]=2;return;}
                    job[t]=size_t(j);const int32_t id=p.ids[job[t]];
                    valid[t]=id>=0&&uint32_t(id)<p.shape.experts&&id==expert;
                    if(!valid[t]&&rt==0&&sp==0)p.status[job[t]]=1;
                }
            }
            if(uint32_t(expert)<p.shape.experts) {
                const uint32_t nk=p.shape.k/256,nsp=nk/split;
                const size_t tb=(size_t(expert)*rtiles+rt)*nk*528;
                es::simd<uint32_t,8> so(0,2);es::simd<uint32_t,16> bo(0,4);es::simd<uint32_t,32> ao(0,1);
                for(uint32_t kb=sp*nsp;kb<(sp+1)*nsp;++kb) {
                    es::simd<uint16_t,8> ds0=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w0+tb+size_t(kb)*528+512),so);
                    es::simd<sycl::half,8> hd0=ds0.template bit_cast_view<sycl::half>();
                    es::simd<float,8> dw0=es::convert<float>(hd0),dw1;
                    if constexpr(Pair) {
                        es::simd<uint16_t,8> ds1=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w1+tb+size_t(kb)*528+512),so);
                        es::simd<sycl::half,8> hd1=ds1.template bit_cast_view<sycl::half>();dw1=es::convert<float>(hd1);
                    }
                    for(uint32_t kg=0;kg<256;kg+=32) {
                        const size_t wb=tb+size_t(kb)*528+size_t(kg/32)*64;
                        es::simd<uint32_t,16> b0=es::gather<uint32_t,16>(reinterpret_cast<const uint32_t*>(p.w0+wb),bo),b1;
                        if constexpr(Pair)b1=es::gather<uint32_t,16>(reinterpret_cast<const uint32_t*>(p.w1+wb),bo);
                        #pragma unroll
                        for(int t=0;t<T;++t)if(t<len&&valid[t]) {
                            const size_t xr=p.per_selection?job[t]:job[t]/p.topk;
                            const size_t ax=xr*p.shape.k+kb*256+kg;
                            const float da=a.scales[ax/32];
                            es::simd<int8_t,32> av=es::gather<int8_t,32>(a.q+ax,ao);
                            es::simd<int32_t,8> ia0=0,ia1=0;
                            ia0=xmx::dpas<8,1,int32_t,int32_t,uint32_t,int8_t,
                                xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(ia0,b0,av);
                            acc0[t]+=es::convert<float>(ia0)*(dw0*da);
                            if constexpr(Pair) {
                                ia1=xmx::dpas<8,1,int32_t,int32_t,uint32_t,int8_t,
                                    xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(ia1,b1,av);
                                acc1[t]+=es::convert<float>(ia1)*(dw1*da);
                            }
                        }
                    }
                }
            }
            es::simd<uint32_t,8> oo(0,4);
            #pragma unroll
            for(int t=0;t<T;++t)if(t<len) {
                const size_t oi=job[t]*p.shape.m+rt*8;
                if(split==1) {
                    es::scatter<float,8>(p.y0+oi,oo,acc0[t]);
                    if constexpr(Pair)es::scatter<float,8>(p.y1+oi,oo,acc1[t]);
                } else {
                    es::scatter<float,8>(scratch+size_t(sp)*outn+oi,oo,acc0[t]);
                    if constexpr(Pair)es::scatter<float,8>(scratch+(size_t(split)+sp)*outn+oi,oo,acc1[t]);
                }
            }
        });
    });
    run.done=run.main;
    if(split>1) {
        const size_t n=outn*(Pair?2:1),global_reduce=((n+127)/128)*128;
        run.done=q.submit([&](sycl::handler& h) {
            h.depends_on(run.main);
            h.parallel_for<W2TokenReuseReduce<T,Pair>>(
              sycl::nd_range<1>{sycl::range<1>(global_reduce),sycl::range<1>(128)},[=](sycl::nd_item<1> it) {
                const size_t i=it.get_global_linear_id();if(i>=n)return;
                const size_t proj=i/outn,j=i%outn;float sum=0;
                for(uint32_t sp=0;sp<split;++sp)sum+=scratch[(proj*split+sp)*outn+j];
                (proj?p.y1:p.y0)[j]=sum;
            });
        });
        run.has_reduce=true;
    }
    return run;
}
A8Run enqueue_a8_token_tiles(sycl::queue& q,const DeviceProblem& p,A8Options o,A8Workspace a,
    float* scratch,const std::vector<sycl::event>& deps) {
    // Exported only for translation-unit dispatch; callers must use enqueue_a8.
    if(o.token_tiles.tokens_per_tile==2)
        return p.w1?launch_tiles<2,true>(q,p,o,a,scratch,deps):launch_tiles<2,false>(q,p,o,a,scratch,deps);
    if(o.token_tiles.tokens_per_tile==4)
        return p.w1?launch_tiles<4,true>(q,p,o,a,scratch,deps):launch_tiles<4,false>(q,p,o,a,scratch,deps);
    throw std::invalid_argument("token-tile dispatch requires T2/T4");
}
} // namespace maple_w2
