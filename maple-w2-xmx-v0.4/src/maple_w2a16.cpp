// SPDX-License-Identifier: MIT
#include "maple_w2a16.hpp"
#include "dg2_validation.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace maple_w2 {
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
template<Layout L, bool Pair> class Tq2Dpas;
class Tq2Reduce;
class Tq2Fma;

static void check(sycl::queue &q,const DeviceProblem& p,Options o) {
    (void)nbytes(p.shape);
    if(p.shape.k>UINT32_MAX/4) throw std::invalid_argument("K byte offsets exceed 32-bit gather addressing");
    validate_dg2_device(q.get_device());
    if(p.activation_type!=ActivationType::f32 && p.activation_type!=ActivationType::f16)
        throw std::invalid_argument("unknown activation type");
    if(p.layout!=Layout::native_tq2 && p.layout!=Layout::tile8 && p.layout!=Layout::s2tile8)
        throw std::invalid_argument("unknown weight layout");
    if (!p.tokens || p.tokens>2048 || !p.topk || p.topk>256 || p.topk>p.shape.experts || !p.w0 || !p.x || !p.ids || !p.y0 || !p.status ||
        (p.w1 && !p.y1) || (!p.w1 && p.y1))
        throw std::invalid_argument("invalid device view / output / status");
    if (o.split_k!=1 && o.split_k!=2 && o.split_k!=4 && o.split_k!=8)
        throw std::invalid_argument("split_k must be 1,2,4,8");
    if ((p.shape.k/256)%o.split_k || !o.local_size || o.local_size>32)
        throw std::invalid_argument("split_k must divide K/256; local_size must be 1..32");
}
size_t scratch_floats(const DeviceProblem& p,Options o) {
    if(o.split_k==1) return 0;
    return checked_mul(checked_mul(checked_mul(checked_mul(p.tokens,p.topk),p.shape.m),o.split_k),p.w1?2:1);
}
static double duration_us(const sycl::event& e) {
    const auto a=e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto b=e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(b-a)/1000.;
}
double Run::main_us() const { return duration_us(main); }
double Run::reduce_us() const { return has_reduce ? duration_us(done) : 0.; }

template<Layout L, bool Pair>
static sycl::event launch(sycl::queue& q,DeviceProblem p,Options opt,float *scratch,
                          const std::vector<sycl::event>& deps) {
    const size_t jobs=size_t(p.tokens)*p.topk;
    const uint32_t row_tiles=p.shape.m/8;
    const size_t logical=jobs*row_tiles*opt.split_k;
    const size_t global=((logical+opt.local_size-1)/opt.local_size)*opt.local_size;
    const size_t out_count=jobs*p.shape.m;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for<Tq2Dpas<L,Pair>>(
          sycl::nd_range<1>{sycl::range<1>(global),sycl::range<1>(opt.local_size)},
          [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            size_t v=it.get_global_linear_id();
            if(v>=logical) return;
            const uint32_t sp=uint32_t(v%opt.split_k); v/=opt.split_k;
            const uint32_t rt=uint32_t(v%row_tiles);
            const size_t job=v/row_tiles;
            const int32_t eid=p.ids[job];
            const size_t oi=job*p.shape.m+rt*8;
            es::simd<float,8> acc0=0.f, acc1=0.f;
            if(eid>=0 && uint32_t(eid)<p.shape.experts) {
                const uint32_t nk=p.shape.k/256;
                const uint32_t count=nk/opt.split_k;
                const size_t xr=p.per_selection?job:job/p.topk;
                const uint8_t *xrow=static_cast<const uint8_t*>(p.x)+xr*p.shape.k*(p.activation_type==ActivationType::f16?2:4);
                const float *xp32=reinterpret_cast<const float*>(xrow);
                const uint16_t *xp16=reinterpret_cast<const uint16_t*>(xrow);
                const size_t row_stride=size_t(nk)*66;
                const size_t ebase=size_t(eid)*p.shape.m*row_stride;
                const size_t tilebase=(size_t(eid)*row_tiles+rt)*nk*528;
                es::simd<uint32_t,32> off32(0,2);
                es::simd<uint32_t,8> off8(0,4);
                for(uint32_t kb=sp*count;kb<(sp+1)*count;++kb) {
                    es::simd<uint16_t,256> native0, native1;
                    es::simd<uint16_t,8> ds0,ds1;
                    if constexpr(L==Layout::native_tq2) {
                        #pragma unroll
                        for(int r=0;r<8;++r) {
                            const size_t b=ebase+size_t(rt*8+r)*row_stride+size_t(kb)*66;
                            native0.template select<32,1>(r*32)=
                              es::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(p.w0+b),off32);
                            ds0[r]=*reinterpret_cast<const uint16_t*>(p.w0+b+64);
                            if constexpr(Pair) {
                                native1.template select<32,1>(r*32)=
                                  es::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(p.w1+b),off32);
                                ds1[r]=*reinterpret_cast<const uint16_t*>(p.w1+b+64);
                            }
                        }
                    } else {
                        es::simd<uint32_t,8> so(0,2);
                        const size_t b=tilebase+size_t(kb)*528+512;
                        ds0=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w0+b),so);
                        if constexpr(Pair) ds1=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(p.w1+b),so);
                    }
                    es::simd<uint16_t,16> s0,s1;
                    s0.template select<8,2>(0)=ds0;
                    s0.template select<8,2>(1)=ds0;
                    if constexpr(Pair) {
                        s1.template select<8,2>(0)=ds1;
                        s1.template select<8,2>(1)=ds1;
                    }
                    es::simd<uint16_t,128> scales0,scales1;
                    #pragma unroll
                    for(int kp=0;kp<8;++kp) {
                        scales0.template select<16,1>(kp*16)=s0;
                        if constexpr(Pair) scales1.template select<16,1>(kp*16)=s1;
                    }
                    // K=16 per FP16 DPAS. B VNNI index = (k/2)*16 + n*2 + k%2.
                    #pragma unroll
                    for(int step=0;step<16;++step) {
                        es::simd<uint16_t,128> c0,c1;
                        if constexpr(L==Layout::native_tq2) {
                            const int kk=step*16;
                            const int wi=(kk/128)*16+(kk%32)/2;
                            const int sh=2*((kk%128)/32);
                            #pragma unroll
                            for(int r=0;r<8;++r) {
                                es::simd<uint16_t,8> z0=native0.template select<8,1>(r*32+wi);
                                c0.template select<8,16>(r*2)=(z0>>sh)&uint16_t(3);
                                c0.template select<8,16>(r*2+1)=(z0>>(8+sh))&uint16_t(3);
                                if constexpr(Pair) {
                                    es::simd<uint16_t,8> z1=native1.template select<8,1>(r*32+wi);
                                    c1.template select<8,16>(r*2)=(z1>>sh)&uint16_t(3);
                                    c1.template select<8,16>(r*2+1)=(z1>>(8+sh))&uint16_t(3);
                                }
                            }
                        } else {
                            const size_t b=tilebase+size_t(kb)*528+step*32;
                            es::simd<uint32_t,8> z0=es::gather<uint32_t,8>(reinterpret_cast<const uint32_t*>(p.w0+b),off8);
                            es::simd<uint32_t,8> z1;
                            if constexpr(Pair) z1=es::gather<uint32_t,8>(reinterpret_cast<const uint32_t*>(p.w1+b),off8);
                            #pragma unroll
                            for(int j=0;j<16;++j) {
                                if constexpr(L==Layout::s2tile8) {
                                    c0.template select<8,2>((j/2)*16+j%2)=es::convert<uint16_t>((z0>>(2*j))&3u);
                                    if constexpr(Pair) c1.template select<8,2>((j/2)*16+j%2)=es::convert<uint16_t>((z1>>(2*j))&3u);
                                } else {
                                    c0.template select<8,16>(j)=es::convert<uint16_t>((z0>>(2*j))&3u);
                                    if constexpr(Pair) c1.template select<8,16>(j)=es::convert<uint16_t>((z1>>(2*j))&3u);
                                }
                            }
                        }
                        es::simd<sycl::half,16> a;
                        if(p.activation_type==ActivationType::f16) {
                            es::simd<uint32_t,16> axoff((kb*256+step*16)*2,2);
                            es::simd<uint16_t,16> ab=es::gather<uint16_t,16>(xp16,axoff);
                            a=ab.template bit_cast_view<sycl::half>();
                        } else {
                            es::simd<uint32_t,16> axoff((kb*256+step*16)*4,4);
                            es::simd<float,16> af=es::gather<float,16>(xp32,axoff);
                            a=es::convert<sycl::half>(af);
                        }
                        es::simd<uint16_t,128> bits0=scales0;
                        bits0.merge(scales0 ^ uint16_t(0x8000),c0==(L==Layout::s2tile8?3:0));
                        bits0.merge(uint16_t(0),c0==(L==Layout::s2tile8?0:1));
                        es::simd<sycl::half,128> b0=bits0.template bit_cast_view<sycl::half>();
                        acc0=xmx::dpas<8,1,float>(acc0,b0,a);
                        if constexpr(Pair) {
                            es::simd<uint16_t,128> bits1=scales1;
                            bits1.merge(scales1 ^ uint16_t(0x8000),c1==(L==Layout::s2tile8?3:0));
                            bits1.merge(uint16_t(0),c1==(L==Layout::s2tile8?0:1));
                            es::simd<sycl::half,128> b1=bits1.template bit_cast_view<sycl::half>();
                            acc1=xmx::dpas<8,1,float>(acc1,b1,a);
                        }
                    }
                }
            } else if(rt==0 && sp==0) {
                p.status[job]=1; // exactly one writer per invalid job; never index weights
            }
            es::simd<uint32_t,8> oo(0,sizeof(float));
            if(opt.split_k==1) {
                es::scatter<float,8>(p.y0+oi,oo,acc0);
                if constexpr(Pair) es::scatter<float,8>(p.y1+oi,oo,acc1);
            } else {
                es::scatter<float,8>(scratch+size_t(sp)*out_count+oi,oo,acc0);
                if constexpr(Pair) es::scatter<float,8>(scratch+(size_t(opt.split_k)+sp)*out_count+oi,oo,acc1);
            }
        });
    });
}
Run enqueue(sycl::queue& q,const DeviceProblem& p,Options o,float *scratch,
            const std::vector<sycl::event>& deps) {
    check(q,p,o);
    if(o.split_k>1 && !scratch) throw std::invalid_argument("missing persistent split scratch");
    sycl::event main;
    if(p.layout==Layout::native_tq2) {
        main=p.w1?launch<Layout::native_tq2,true>(q,p,o,scratch,deps):launch<Layout::native_tq2,false>(q,p,o,scratch,deps);
    } else if(p.layout==Layout::s2tile8) {
        main=p.w1?launch<Layout::s2tile8,true>(q,p,o,scratch,deps):launch<Layout::s2tile8,false>(q,p,o,scratch,deps);
    } else {
        main=p.w1?launch<Layout::tile8,true>(q,p,o,scratch,deps):launch<Layout::tile8,false>(q,p,o,scratch,deps);
    }
    if(o.split_k==1) return {main,main,false};
    const size_t n=size_t(p.tokens)*p.topk*p.shape.m;
    const size_t logical=n*(p.w1?2:1), global=((logical+127)/128)*128;
    auto done=q.submit([&](sycl::handler& h){
        h.depends_on(main);
        h.parallel_for<Tq2Reduce>(sycl::nd_range<1>{sycl::range<1>(global),sycl::range<1>(128)},[=](sycl::nd_item<1> it){
            size_t i=it.get_global_linear_id(); if(i>=logical) return;
            size_t projection=i/n, j=i%n;
            float sum=0;
            for(uint32_t sp=0;sp<o.split_k;++sp) sum+=scratch[(projection*o.split_k+sp)*n+j];
            (projection?p.y1:p.y0)[j]=sum;
        });
    });
    return {main,done,true};
}
Run enqueue_fma_reference(sycl::queue& q,const DeviceProblem& p,const std::vector<sycl::event>& deps) {
    check(q,p,Options{});
    if(p.layout!=Layout::native_tq2) throw std::invalid_argument("FMA comparator requires native TQ2");
    const size_t outputs=size_t(p.tokens)*p.topk*p.shape.m;
    auto ev=q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for<Tq2Fma>(sycl::nd_range<1>{sycl::range<1>(outputs*32),sycl::range<1>(32)},[=](sycl::nd_item<1> it){
            size_t oi=it.get_group_linear_id(); uint32_t lane=uint32_t(it.get_local_linear_id());
            size_t job=oi/p.shape.m; uint32_t row=uint32_t(oi%p.shape.m);
            int eid=p.ids[job]; float v0=0,v1=0;
            if(eid>=0 && uint32_t(eid)<p.shape.experts) {
                size_t xr=p.per_selection?job:job/p.topk;
                size_t rb=(size_t(eid)*p.shape.m+row)*(p.shape.k/256)*66;
                for(uint32_t k=lane;k<p.shape.k;k+=32) {
                    size_t b=rb+size_t(k/256)*66; uint32_t ki=k%256;
                    uint32_t qi=(ki/128)*32+ki%32, sh=2*((ki%128)/32);
                    uint16_t d0=uint16_t(p.w0[b+64])|uint16_t(p.w0[b+65])<<8;
                    float w0=float(sycl::bit_cast<sycl::half>(d0))*float(int((p.w0[b+qi]>>sh)&3)-1);
                    const size_t xi=xr*p.shape.k+k;
                    float av=p.activation_type==ActivationType::f16 ?
                        float(sycl::bit_cast<sycl::half>(static_cast<const uint16_t*>(p.x)[xi])) :
                        static_cast<const float*>(p.x)[xi];
                    v0=sycl::fma(w0,av,v0);
                    if(p.w1) {
                        uint16_t d1=uint16_t(p.w1[b+64])|uint16_t(p.w1[b+65])<<8;
                        float w1=float(sycl::bit_cast<sycl::half>(d1))*float(int((p.w1[b+qi]>>sh)&3)-1);
                        v1=sycl::fma(w1,av,v1);
                    }
                }
            } else if(lane==0 && row==0) p.status[job]=1;
            v0=sycl::reduce_over_group(it.get_group(),v0,sycl::plus<float>());
            v1=sycl::reduce_over_group(it.get_group(),v1,sycl::plus<float>());
            if(lane==0) { p.y0[oi]=v0; if(p.w1) p.y1[oi]=v1; }
        });
    });
    return {ev,ev,false};
}
} // namespace maple_w2
