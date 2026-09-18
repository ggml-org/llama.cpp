// SPDX-License-Identifier: MIT
#include "tq2_reference.hpp"
#include <iostream>
#include <random>

using namespace maple_w2;
static size_t checks=0;
static void require(bool ok,const char*what){++checks;if(!ok)throw std::runtime_error(what);}
template<class F>void rejects(F f,const char*what){bool threw=false;try{f();}catch(const std::exception&){threw=true;}require(threw,what);}
// Independent emulation of the register selection used in the native GPU kernel.
static std::array<uint16_t,128> native_slice(const std::vector<uint8_t>& w,Shape s,
                                           unsigned e,unsigned rt,unsigned kb,unsigned step) {
    std::array<uint16_t,128> result{};
    unsigned kk=step*16,wi=(kk/128)*16+(kk%32)/2,sh=2*((kk%128)/32);
    for(unsigned r=0;r<8;++r) {
        const auto *p=w.data()+native_offset(s,e,rt*8+r,kb);
        for(unsigned j=0;j<8;++j){
            uint16_t word=u16(p+2*(wi+j)),scale=u16(p+64);
            for(unsigned odd=0;odd<2;++odd){
                unsigned c=(word>>(sh+8*odd))&3;
                result[j*16+r*2+odd]=(c==1)?0:uint16_t(scale^(c==0?0x8000:0));
            }
        }
    }
    return result;
}
static std::array<uint16_t,128> tile_slice(const std::vector<uint8_t>& w,Shape s,
                                         unsigned e,unsigned rt,unsigned kb,unsigned step) {
    const auto *p=w.data()+tile_offset(s,e,rt,kb);
    std::array<uint16_t,128> result{};
    for(unsigned j=0;j<128;++j){
        unsigned c=(u32(p+step*32+(j/16)*4)>>(2*(j%16)))&3;
        uint16_t d=u16(p+512+2*((j%16)/2));
        result[j]=c==1?0:uint16_t(d^(c==0?0x8000:0));
    }
    return result;
}
static std::vector<float> emulate(const std::vector<uint8_t>& w,Shape s,
       const std::vector<float>& x,const std::vector<int32_t>& ids,unsigned tokens,
       unsigned topk,bool per,unsigned split,bool tiled) {
    (void)tokens;
    std::vector<float> out(ids.size()*s.m,0);
    for(size_t job=0;job<ids.size();++job) for(unsigned rt=0;rt<s.m/8;++rt)
    for(unsigned sp=0;sp<split;++sp) {
        std::array<float,8> sum{};unsigned n=s.k/256,chunk=n/split;
        for(unsigned kb=sp*chunk;kb<(sp+1)*chunk;++kb)for(unsigned step=0;step<16;++step){
            auto b=tiled?tile_slice(w,s,ids[job],rt,kb,step):native_slice(w,s,ids[job],rt,kb,step);
            for(unsigned k=0;k<16;++k)for(unsigned r=0;r<8;++r){
                float av=half_to_float(float_to_half(x[(per?job:job/topk)*s.k+kb*256+step*16+k]));
                // Matrix product independent of GPU region assignments above.
                sum[r]=std::fma(half_to_float(b[(k/2)*16+r*2+k%2]),av,sum[r]);
            }
        }
        for(unsigned r=0;r<8;++r)out[job*s.m+rt*8+r]+=sum[r];
    }
    return out;
}
int main()try{
    for(unsigned h=0;h<65536;++h){
        float f=half_to_float(uint16_t(h));
        if((h&0x7c00)!=0x7c00)require(float_to_half(f)==h,"binary16 roundtrip");
    }
    require(float_to_half(1.f+std::ldexp(1.f,-11))==0x3c00,"half ties-to-even low");
    require(float_to_half(1.f+3*std::ldexp(1.f,-11))==0x3c02,"half ties-to-even high");
    require(float_to_half(65520.f)==0x7c00,"half overflow tie");
    std::mt19937 gen(78213);
#ifdef __FLT16_MANT_DIG__
    for(int i=0;i<100000;++i){
        uint32_t b=gen();float f;std::memcpy(&f,&b,4);
        if(!std::isfinite(f))continue;
        _Float16 h=(_Float16)f;uint16_t bits;std::memcpy(&bits,&h,2);
        require(bits==float_to_half(f),"binary32->half vs compiler _Float16");
    }
#endif
    const std::vector<Shape> shapes={{256,8,4},{512,16,5},{2048,24,17},{512,8,256}};
    for(auto s:shapes){
        std::vector<uint8_t>w(nbytes(s),0);
        std::vector<float> logical(size_t(s.experts)*s.m*s.k);
        for(unsigned e=0;e<s.experts;++e)for(unsigned r=0;r<s.m;++r)for(unsigned kb=0;kb<s.k/256;++kb){
            auto*b=w.data()+native_offset(s,e,r,kb);
            float d=float(1+((e*7+r*3+kb)%17))/64.f;if((e+r+kb)%2)d=-d;
            uint16_t dh=float_to_half(d);put16(b+64,dh);
            for(unsigned k=0;k<256;++k){
                int t=int(gen()%3)-1;set_native_code(b,k,unsigned(t+1));
                logical[(size_t(e)*s.m+r)*s.k+kb*256+k]=float(t)*half_to_float(dh);
            }
        }
        validate_tq2(w,s);auto tile=repack_tile8(w,s);
        require(tile.size()==w.size(),"lossless repack must not grow weights");
        require(unpack_tile8(tile,s)==w,"byte-exact tile8 roundtrip");
        for(unsigned e=0;e<s.experts;++e)for(unsigned r=0;r<s.m;++r)for(unsigned k=0;k<s.k;++k)
            require(weight_at(w,s,e,r,k)==logical[(size_t(e)*s.m+r)*s.k+k],"native code ordering");
        unsigned topk=3,tokens=2;
        std::vector<int32_t> ids={int32_t(s.experts-1),1,0,1,int32_t(s.experts-1),2};
        for(bool per:{false,true}){
            std::vector<float>x(size_t(tokens)*(per?topk:1)*s.k);
            for(auto&v:x)v=float(int(gen()%20001)-10000)/1703.f;
            auto gold=reference(w,s,x,ids,tokens,topk,per,true);
            for(unsigned split:{1u,2u,4u,8u})if((s.k/256)%split==0)for(bool tiled:{false,true}){
                auto y=emulate(tiled?tile:w,s,x,ids,tokens,topk,per,split,tiled);
                auto er=errors(y,gold);
                require(er.finite&&er.nmse<1e-10&&er.max_abs_over_rms<1e-4,"VNNI/split/expert/x addressing");
            }
        }
        auto bad=w;bad[0]|=3;rejects([&]{validate_tq2(bad,s);},"reject fourth code");
        bad=w;put16(bad.data()+64,0x7c00);rejects([&]{validate_tq2(bad,s);},"reject infinity scale");
    }
    rejects([]{nbytes({255,8,1});},"reject bad K");
    rejects([]{nbytes({256,7,1});},"reject bad M");
    std::cout<<"PASS: "<<checks<<" CPU checks (half conversion, TQ2 order, tile8 roundtrip,\n"
      <<"native/tile8 DPAS-layout emulation, expert offsets, broadcast/per-selection, split-K).\n"
      <<"This does NOT compile or execute a SYCL/GPU kernel.\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
