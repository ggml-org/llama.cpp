// SPDX-License-Identifier: MIT
#include "maple_w2a8.hpp"
#include "dg2_validation.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
#include <type_traits>

namespace maple_w2 {
namespace es=sycl::ext::intel::esimd;
namespace xmx=sycl::ext::intel::esimd::xmx;
template<int G> class QuantA8;
template<int G,Layout L,bool Pair,bool Fused> class Tq2Int2;
template<int G,Layout L,bool Pair,bool Fused> class Tq2Int2Grouped;
template<int G,Layout L,bool Pair> class Tq2A8Reduce;
class S2S8Probe;
static double event_us(const sycl::event& e) {
    auto b=e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto z=e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(z-b)/1000.;
}
double A8Run::quant_us() const {return has_quant?event_us(quant):0.;}
double A8Run::main_us() const {return event_us(main);}
double A8Run::reduce_us() const {return has_reduce?event_us(done):0.;}
static void check_device(const sycl::device& d) { validate_dg2_device(d); }
template<int G>
static sycl::event quantize(sycl::queue& q,DeviceProblem p,A8Workspace w,
                            const std::vector<sycl::event>& deps) {
    const size_t rows=size_t(p.tokens)*(p.per_selection?p.topk:1);
    const size_t groups=rows*(p.shape.k/G);
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for<QuantA8<G>>(sycl::nd_range<1>{sycl::range<1>(groups*32),sycl::range<1>(32)},
          [=](sycl::nd_item<1> it){
            const size_t g=it.get_group_linear_id(); const unsigned lane=it.get_local_linear_id();
            float values[G/32], mx=0; int bad=0;
            for(unsigned j=0;j<G/32;++j) {
                const size_t idx=g*G+j*32+lane;
                float v=p.activation_type==ActivationType::f16 ?
                    float(sycl::bit_cast<sycl::half>(static_cast<const uint16_t*>(p.x)[idx])) :
                    static_cast<const float*>(p.x)[idx];
                if(!sycl::isfinite(v)) {bad=1; v=0;}
                values[j]=v; mx=sycl::fmax(mx,sycl::fabs(v));
            }
            mx=sycl::reduce_over_group(it.get_group(),mx,sycl::maximum<float>());
            bad=sycl::reduce_over_group(it.get_group(),bad,sycl::maximum<int>());
            const float d=mx==0 ? 1.f : sycl::fmax(mx/127.f,1.1754943508222875e-38f);
            if(lane==0) {w.scales[g]=d; w.invalid[g]=bad;}
            for(unsigned j=0;j<G/32;++j) {
                float z=sycl::fmax(-127.f,sycl::fmin(127.f,values[j]/d));
                float mag=sycl::fabs(z); uint32_t k=uint32_t(mag); float frac=mag-float(k);
                k+=uint32_t(frac>0.5f || (frac==0.5f && (k&1u)));
                w.q[g*G+j*32+lane]=int8_t(z<0 ? -int(k):int(k));
            }
          });
    });
}
template<int N> SYCL_ESIMD_FUNCTION
float max_tree(es::simd<float,N> v) {
    if constexpr(N==1) return v[0];
    else {
        es::simd<float,N/2> a=v.template select<N/2,1>(0),b=v.template select<N/2,1>(N/2);
        a.merge(b,b>a); return max_tree<N/2>(a);
    }
}
template<int G> SYCL_ESIMD_FUNCTION
es::simd<int8_t,G> quant_registers(es::simd<float,G> v,float& d) {
    es::simd<float,G> mag=v; mag.merge(-v,v<0.f);
    float mx=max_tree<G>(mag);
    d=mx==0.f?1.f:mx/127.f; if(d<1.1754943508222875e-38f) d=1.1754943508222875e-38f;
    es::simd<float,G> z=v/d;
    z.merge(127.f,z>127.f); z.merge(-127.f,z< -127.f);
    es::simd<float,G> az=z; az.merge(-z,z<0.f);
    es::simd<uint32_t,G> qi=es::convert<uint32_t>(az);
    es::simd<float,G> frac=az-es::convert<float>(qi);
    es::simd<uint32_t,G> inc=0u;
    inc.merge(1u,(frac>0.5f)|((frac==0.5f)&((qi&1u)!=0u)));
    qi+=inc;
    es::simd<int32_t,G> si=es::convert<int32_t>(qi); si.merge(-si,z<0.f);
    return es::convert<int8_t>(si);
}
// Native TQ2 decode to s2 VNNI in registers; no model-wide expansion.
SYCL_ESIMD_FUNCTION es::simd<uint32_t,16>
pack_native_s2(es::simd<uint16_t,256> raw,uint32_t k32) {
    es::simd<uint32_t,16> b=0u;
    #pragma unroll
    for(int j=0;j<32;++j) {
        uint32_t k=k32+j;
        const uint32_t wi=(k/128)*16+(k%32)/2;
        const uint32_t sh=2*((k%128)/32)+8*(k%2);
        es::simd<uint16_t,8> words=raw.template select<8,32>(wi);
        es::simd<uint32_t,8> c=es::convert<uint32_t>((words>>sh)&uint16_t(3));
        // Each subtraction is in its own DWORD lane. No cross-code borrow.
        es::simd<uint32_t,8> signed_bits=(c-1u)&3u;
        es::simd<uint32_t,8> old=b.template select<8,1>((j/16)*8);
        b.template select<8,1>((j/16)*8)=old|(signed_bits<<(2*(j%16)));
    }
    return b;
}
template<int G,Layout L,bool Pair,bool Fused,bool Grouped=false>
static sycl::event launch(sycl::queue& q,DeviceProblem p,A8Options opt,A8Workspace w,float* scratch,
                           const std::vector<sycl::event>& deps) {
    const size_t jobs=size_t(p.tokens)*p.topk, outn=jobs*p.shape.m;
    const uint32_t rt_count=p.shape.m/8;
    const size_t logical=jobs*rt_count*opt.kernel.split_k;
    const size_t global=((logical+opt.kernel.local_size-1)/opt.kernel.local_size)*opt.kernel.local_size;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        using KernelName=std::conditional_t<Grouped,Tq2Int2Grouped<G,L,Pair,Fused>,Tq2Int2<G,L,Pair,Fused>>;
        h.parallel_for<KernelName>(
          sycl::nd_range<1>{sycl::range<1>(global),sycl::range<1>(opt.kernel.local_size)},
          [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            size_t idx=it.get_global_linear_id(); if(idx>=logical) return;
            uint32_t sp=uint32_t(idx%opt.kernel.split_k); idx/=opt.kernel.split_k;
            const uint32_t rt=uint32_t(idx%rt_count); const size_t scheduled_job=idx/rt_count;
            size_t job=scheduled_job;
            if constexpr(Grouped) {
                const int32_t mapped=opt.job_order[scheduled_job];
                if(mapped<0||size_t(mapped)>=jobs) {
                    if(rt==0&&sp==0)p.status[scheduled_job]=2;
                    return; // corrupted external mapping; caller must reject status
                }
                job=size_t(mapped);
            }
            const size_t oi=job*p.shape.m+rt*8;
            es::simd<float,8> acc0=0.f,acc1=0.f;
            const int32_t eid=p.ids[job];
            if(eid>=0 && uint32_t(eid)<p.shape.experts) {
                const size_t xr=p.per_selection?job:job/p.topk;
                const uint32_t nk=p.shape.k/256, nsp=nk/opt.kernel.split_k;
                const size_t row_stride=size_t(nk)*66;
                const size_t eb=size_t(eid)*p.shape.m*row_stride;
                const size_t tb=(size_t(eid)*rt_count+rt)*nk*528;
                const bool writer=rt==0 && (p.per_selection || job%p.topk==0);
                es::simd<uint32_t,32> off16(0,2),off32(0,4),off8(0,1);
                for(uint32_t kb=sp*nsp;kb<(sp+1)*nsp;++kb) {
                    es::simd<uint16_t,256> raw0,raw1;
                    es::simd<uint16_t,8> ds0,ds1;
                    if constexpr(L==Layout::native_tq2) {
                        #pragma unroll
                        for(int r=0;r<8;++r) {
                            const size_t wb=eb+size_t(rt*8+r)*row_stride+size_t(kb)*66;
                            raw0.template select<32,1>(r*32)=es::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(p.w0+wb),off16);
                            ds0[r]=*reinterpret_cast<const uint16_t*>(p.w0+wb+64);
                            if constexpr(Pair) {
                                raw1.template select<32,1>(r*32)=es::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(p.w1+wb),off16);
                                ds1[r]=*reinterpret_cast<const uint16_t*>(p.w1+wb+64);
                            }
                        }
                    } else {
                        es::simd<uint32_t,8> so(0,2);
                        ds0=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w0+tb+size_t(kb)*528+512),so);
                        if constexpr(Pair) ds1=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w1+tb+size_t(kb)*528+512),so);
                    }
                    es::simd<sycl::half,8> hd0=ds0.template bit_cast_view<sycl::half>();
                    es::simd<float,8> dw0=es::convert<float>(hd0),dw1;
                    if constexpr(Pair) {es::simd<sycl::half,8> hd1=ds1.template bit_cast_view<sycl::half>();dw1=es::convert<float>(hd1);}
                    for(uint32_t kg=0;kg<256;kg+=G) {
                        const size_t group_index=xr*(p.shape.k/G)+(kb*256+kg)/G;
                        const size_t ax=xr*p.shape.k+kb*256+kg;
                        es::simd<int8_t,G> aq; float da;
                        if constexpr(Fused) {
                            es::simd<float,G> av;
                            #pragma unroll
                            for(int j=0;j<G;j+=32) {
                                if(p.activation_type==ActivationType::f16) {
                                    es::simd<uint16_t,32> hv=es::gather<uint16_t,32>(static_cast<const uint16_t*>(p.x)+ax+j,off16);
                                    es::simd<sycl::half,32> hh=hv.template bit_cast_view<sycl::half>();
                                    av.template select<32,1>(j)=es::convert<float>(hh);
                                } else av.template select<32,1>(j)=es::gather<float,32>(static_cast<const float*>(p.x)+ax+j,off32);
                            }
                            const auto bad=(av!=av)|(av>3.402823466e38f)|(av< -3.402823466e38f);
                            if(writer) {es::simd<float,G> f=0.f;f.merge(1.f,bad); w.invalid[group_index]=int32_t(max_tree<G>(f));}
                            av.merge(0.f,bad); // invalid flag above requires caller rejection; no silent success
                            aq=quant_registers<G>(av,da);
                            if(writer && w.capture) {
                                w.scales[group_index]=da;
                                #pragma unroll
                                for(int j=0;j<G;j+=32) {
                                    es::simd<int8_t,32> part=aq.template select<32,1>(j);
                                    es::scatter<int8_t,32>(w.q+ax+j,off8,part);
                                }
                            }
                        } else {
                            da=w.scales[group_index];
                            #pragma unroll
                            for(int j=0;j<G;j+=32) aq.template select<32,1>(j)=es::gather<int8_t,32>(w.q+ax+j,off8);
                        }
                        es::simd<int32_t,8> ia0=0,ia1=0;
                        for(uint32_t sub=0;sub<G;sub+=32) {
                            es::simd<uint32_t,16> b0,b1;
                            if constexpr(L==Layout::native_tq2) {
                                b0=pack_native_s2(raw0,kg+sub);
                                if constexpr(Pair) b1=pack_native_s2(raw1,kg+sub);
                            } else {
                                es::simd<uint32_t,16> bo(0,4);
                                const size_t wb=tb+size_t(kb)*528+size_t((kg+sub)/32)*64;
                                b0=es::gather<uint32_t,16>(reinterpret_cast<const uint32_t*>(p.w0+wb),bo);
                                if constexpr(Pair) b1=es::gather<uint32_t,16>(reinterpret_cast<const uint32_t*>(p.w1+wb),bo);
                            }
                            es::simd<int8_t,32> a32=aq.template select<32,1>(sub);
                            // B=INT2, A=INT8; DG2 K32 x N8, RepeatCount1, INT32 accumulate.
                            ia0=xmx::dpas<8,1,int32_t,int32_t,uint32_t,int8_t,
                                xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(ia0,b0,a32);
                            if constexpr(Pair) ia1=xmx::dpas<8,1,int32_t,int32_t,uint32_t,int8_t,
                                xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(ia1,b1,a32);
                        }
                        acc0+=es::convert<float>(ia0)*(dw0*da);
                        if constexpr(Pair) acc1+=es::convert<float>(ia1)*(dw1*da);
                    }
                }
            } else if(rt==0 && sp==0) p.status[job]=1;
            es::simd<uint32_t,8> oo(0,4);
            if(opt.kernel.split_k==1) {
                es::scatter<float,8>(p.y0+oi,oo,acc0);
                if constexpr(Pair) es::scatter<float,8>(p.y1+oi,oo,acc1);
            } else {
                es::scatter<float,8>(scratch+size_t(sp)*outn+oi,oo,acc0);
                if constexpr(Pair) es::scatter<float,8>(scratch+(size_t(opt.kernel.split_k)+sp)*outn+oi,oo,acc1);
            }
          });
    });
}
template<int G,Layout L,bool Pair>
static A8Run dispatch_mode(sycl::queue& q,DeviceProblem p,A8Options o,A8Workspace w,float* scratch,const std::vector<sycl::event>& deps) {
    A8Run r;
    if(o.mode==A8Mode::staged) {
        r.quant=quantize<G>(q,p,w,deps); r.has_quant=true;
        if constexpr(G==32 && L==Layout::s2tile8) {
            r.main=o.job_order?launch<G,L,Pair,false,true>(q,p,o,w,scratch,{r.quant}):launch<G,L,Pair,false>(q,p,o,w,scratch,{r.quant});
        } else r.main=launch<G,L,Pair,false>(q,p,o,w,scratch,{r.quant});
    } else if(o.mode==A8Mode::prequantized) {
        if constexpr(G==32 && L==Layout::s2tile8) {
            r.main=o.job_order?launch<G,L,Pair,false,true>(q,p,o,w,scratch,deps):launch<G,L,Pair,false>(q,p,o,w,scratch,deps);
        } else r.main=launch<G,L,Pair,false>(q,p,o,w,scratch,deps);
    } else r.main=launch<G,L,Pair,true>(q,p,o,w,scratch,deps);
    r.done=r.main;
    if(o.kernel.split_k>1) {
        const size_t n=size_t(p.tokens)*p.topk*p.shape.m, logical=n*(Pair?2:1), global=((logical+127)/128)*128;
        r.done=q.submit([&](sycl::handler& h){
            h.depends_on(r.main);
            h.parallel_for<Tq2A8Reduce<G,L,Pair>>(sycl::nd_range<1>{sycl::range<1>(global),sycl::range<1>(128)},[=](sycl::nd_item<1> it){
                size_t i=it.get_global_linear_id(); if(i>=logical)return;
                size_t proj=i/n,j=i%n; float sum=0;
                for(uint32_t sp=0;sp<o.kernel.split_k;++sp)sum+=scratch[(proj*o.kernel.split_k+sp)*n+j];
                (proj?p.y1:p.y0)[j]=sum;
            });
        });
        r.has_reduce=true;
    }
    return r;
}
template<int G>
static A8Run dispatch_layout(sycl::queue& q,DeviceProblem p,A8Options o,A8Workspace w,float* scratch,const std::vector<sycl::event>& deps) {
    if(p.layout==Layout::native_tq2)
        return p.w1?dispatch_mode<G,Layout::native_tq2,true>(q,p,o,w,scratch,deps):dispatch_mode<G,Layout::native_tq2,false>(q,p,o,w,scratch,deps);
    return p.w1?dispatch_mode<G,Layout::s2tile8,true>(q,p,o,w,scratch,deps):dispatch_mode<G,Layout::s2tile8,false>(q,p,o,w,scratch,deps);
}
A8Run enqueue_a8(sycl::queue& q,const DeviceProblem& p,A8Options o,A8Workspace w,float* scratch,const std::vector<sycl::event>& deps) {
    check_device(q.get_device()); (void)nbytes(p.shape); check_a8_group(o.group);
    if(!p.tokens||p.tokens>2048||!p.topk||p.topk>256||p.topk>p.shape.experts||
       !p.w0||!p.x||!p.ids||!p.y0||!p.status||(p.w1&&!p.y1)||(!p.w1&&p.y1)||!w.invalid)
        throw std::invalid_argument("invalid A8 device view");
    if(p.layout!=Layout::native_tq2 && p.layout!=Layout::s2tile8)throw std::invalid_argument("A8 accepts native or s2tile8, NOT legacy A16 tile8");
    if(p.activation_type!=ActivationType::f16&&p.activation_type!=ActivationType::f32)throw std::invalid_argument("invalid activation type");
    if(o.mode!=A8Mode::staged&&o.mode!=A8Mode::fused&&o.mode!=A8Mode::prequantized)throw std::invalid_argument("invalid A8 mode");
    if((o.mode==A8Mode::staged||o.mode==A8Mode::prequantized||w.capture)&&(!w.q||!w.scales))throw std::invalid_argument("A8 staged/capture needs quantized scratch");
    if(o.job_order&&(o.group!=32||p.layout!=Layout::s2tile8||o.mode==A8Mode::fused))
        throw std::invalid_argument("v0.4 grouping supports G32/s2tile8 staged or prequantized only");
    const uint32_t sp=o.kernel.split_k;
    if((sp!=1&&sp!=2&&sp!=4&&sp!=8)||(p.shape.k/256)%sp||!o.kernel.local_size||o.kernel.local_size>32||
       (sp>1&&!scratch))throw std::invalid_argument("invalid split/local/scratch");
    check_tokens_per_tile(o.token_tiles.tokens_per_tile);
    if(o.token_tiles.tokens_per_tile>1) {
        const auto& t=o.token_tiles;
        if(o.group!=32||p.layout!=Layout::s2tile8||o.mode!=A8Mode::prequantized||!o.job_order||
           !t.first||!t.length||!t.expert||!t.total||
           t.capacity<expert_token_tile_capacity(size_t(p.tokens)*p.topk,p.shape.experts,t.tokens_per_tile))
            throw std::invalid_argument("grouped reuse needs fresh G32/s2tile8/prequantized tiles");
        return enqueue_a8_token_tiles(q,p,o,w,scratch,deps);
    }
    switch(o.group) {
        case 32:return dispatch_layout<32>(q,p,o,w,scratch,deps);
        case 128:return dispatch_layout<128>(q,p,o,w,scratch,deps);
        default:return dispatch_layout<256>(q,p,o,w,scratch,deps);
    }
}
sycl::event enqueue_a8_quant(sycl::queue& q,const void* x,ActivationType type,uint32_t rows,
    uint32_t k,uint32_t group,A8Workspace w,const std::vector<sycl::event>& deps) {
    check_device(q.get_device());check_a8_group(group);
    if(!x||!rows||!k||k%group||!w.q||!w.scales||!w.invalid||
       (type!=ActivationType::f32&&type!=ActivationType::f16))
        throw std::invalid_argument("invalid standalone A8 quantizer view");
    (void)checked_mul(rows,k);
    DeviceProblem p{};p.shape={k,8,1};p.tokens=rows;p.topk=1;p.x=x;p.activation_type=type;
    switch(group) {
        case 32:return quantize<32>(q,p,w,deps);
        case 128:return quantize<128>(q,p,w,deps);
        default:return quantize<256>(q,p,w,deps);
    }
}
void run_s2s8_probe(sycl::queue& q) {
    check_device(q.get_device());
    constexpr int cases=36;
    std::vector<uint32_t> wb(cases*16,0); std::vector<int8_t> ab(cases*32); std::vector<int32_t> ref(cases*8,0),got(cases*8);
    for(int c=0;c<cases;++c) for(int k=0;k<32;++k) {
        const int a=c==1?-128:(c>=2&&c<34?(k==c-2?1:0):((k*37+c*19)%256-128));
        ab[c*32+k]=int8_t(a);
        for(int r=0;r<8;++r) {
            const int b=c==1?-1:((k*13+r*7+c)%4-2); // also covers -2 to probe hardware sign extension
            wb[c*16+(k/16)*8+r]|=(uint32_t(b)&3u)<<(2*(k%16));
            ref[c*8+r]+=a*b;
        }
    }
    uint32_t* dw=sycl::malloc_device<uint32_t>(wb.size(),q);
    int8_t* da=sycl::malloc_device<int8_t>(ab.size(),q);
    int32_t* dy=sycl::malloc_device<int32_t>(got.size(),q);
    try {
        if(!dw||!da||!dy)throw std::bad_alloc();
        auto ew=q.memcpy(dw,wb.data(),wb.size()*4);auto ea=q.memcpy(da,ab.data(),ab.size());
        auto e=q.submit([&](sycl::handler& h){h.depends_on(std::vector<sycl::event>{ew,ea});
            h.parallel_for<S2S8Probe>(sycl::range<1>(cases),[=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                const size_t c=id[0]; es::simd<uint32_t,16> wo(0,4);es::simd<uint32_t,32> ao(0,1);es::simd<uint32_t,8> yo(0,4);
                es::simd<uint32_t,16>b=es::gather<uint32_t,16>(dw+c*16,wo);
                es::simd<int8_t,32>a=es::gather<int8_t,32>(da+c*32,ao);es::simd<int32_t,8> z=0;
                z=xmx::dpas<8,1,int32_t,int32_t,uint32_t,int8_t,xmx::dpas_argument_type::s2,xmx::dpas_argument_type::s8>(z,b,a);
                es::scatter<int32_t,8>(dy+c*8,yo,z);
            });});
        e.wait_and_throw();q.memcpy(got.data(),dy,got.size()*4).wait_and_throw();
        if(got!=ref)throw std::runtime_error("native s2/s8 DPAS probe FAILED: packing/sign/driver/compiler; no fallback claimed");
        std::cout<<"DPAS_S2_S8_PROBE PASS cases="<<cases<<" outputs="<<ref.size()<<" (ISA disassembly still required)\n";
    } catch(...) {q.wait();if(dw)sycl::free(dw,q);if(da)sycl::free(da,q);if(dy)sycl::free(dy,q);throw;}
    sycl::free(dw,q);sycl::free(da,q);sycl::free(dy,q);
}
} // namespace maple_w2
