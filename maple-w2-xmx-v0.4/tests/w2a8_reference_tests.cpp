// SPDX-License-Identifier: MIT
#include "w2a8_reference.hpp"
#include <iostream>
#include <random>
using namespace maple_w2;
static size_t checks=0;
static void require(bool v,const char* why){++checks;if(!v)throw std::runtime_error(why);}
template<class F> static void rejects(F f,const char* why){bool yes=false;try{f();}catch(const std::exception&){yes=true;}require(yes,why);}
static std::array<uint32_t,16> emulate_native_pack(const uint8_t* const rows[8],unsigned k32) {
    std::array<uint32_t,16> b{};
    for(unsigned j=0;j<32;++j) for(unsigned r=0;r<8;++r) {
        unsigned k=k32+j, wi=(k/128)*16+(k%32)/2, sh=2*((k%128)/32)+8*(k%2);
        uint32_t c=(u16(rows[r]+2*wi)>>sh)&3u;
        b[(j/16)*8+r]|=((c-1u)&3u)<<(2*(j%16));
    }
    return b;
}
int main()try {
    require(signed2_from_tq(0)==3 && signed2_from_tq(1)==0 && signed2_from_tq(2)==1,"signed ternary mapping");
    rejects([]{signed2_from_tq(3);},"reserved input code");
    for(int i=-127;i<=127;++i)require(quant_rne(float(i),1.f)==i,"exact integer quantization");
    require(quant_rne(0.5f,1)==0 && quant_rne(1.5f,1)==2 && quant_rne(2.5f,1)==2,"positive ties-even");
    require(quant_rne(-0.5f,1)==0 && quant_rne(-1.5f,1)==-2 && quant_rne(-2.5f,1)==-2,"negative ties-even");
    require(quant_rne(1000,1)==127 && quant_rne(-1000,1)==-127,"symmetric saturation");
    for(unsigned g:{32u,128u,256u}) {
        auto z=quantize_a8(std::vector<float>(512,0),1,512,g);
        for(auto v:z.q)require(v==0,"zero group codes");
        for(auto d:z.scale)require(d==1,"zero group scale");
        auto tiny=quantize_a8(std::vector<float>(256,1e-40f),1,256,g);
        for(auto d:tiny.scale)require(std::isnormal(d)&&d>0,"tiny group has normal positive scale");
    }
    rejects([]{quantize_a8({1,2},1,2,32);},"bad quant shape");
    rejects([]{quantize_a8(std::vector<float>(256,0),1,256,64);},"unsupported group explicit rejection");
    rejects([]{quantize_a8(std::vector<float>(256,std::numeric_limits<float>::quiet_NaN()),1,256,32);},"NaN reject");
    std::mt19937 gen(424242);
    for(Shape s:std::vector<Shape>{{256,8,4},{512,24,9},{2048,16,17},{512,8,256}}) {
        std::vector<uint8_t> native(nbytes(s),0);
        for(size_t b=0;b<native.size();b+=66) {
            for(unsigned k=0;k<256;++k)set_native_code(native.data()+b,k,gen()%3);
            float d=float(1+gen()%15)/128.f; if(gen()%2)d=-d;
            put16(native.data()+b+64,float_to_half(d));
        }
        auto packed=repack_s2tile8(native,s);
        require(packed.size()==native.size(),"s2 layout never expands weights");
        require(unpack_s2tile8(packed,s)==native,"s2 roundtrip byte exact including scale");
        for(unsigned e=0;e<s.experts;++e)for(unsigned rt=0;rt<s.m/8;++rt)for(unsigned kb=0;kb<s.k/256;++kb) {
            const uint8_t* rows[8];for(unsigned r=0;r<8;++r)rows[r]=native.data()+native_offset(s,e,rt*8+r,kb);
            const auto* tile=packed.data()+tile_offset(s,e,rt,kb);
            for(unsigned k32=0;k32<256;k32+=32) {
                auto b=emulate_native_pack(rows,k32);
                for(unsigned i=0;i<16;++i)require(b[i]==u32(tile+(k32/32)*64+4*i),"native register pack equals prepack");
                // Independently emulate eight depths of dot4 on the INT2 VNNI layout.
                std::array<int,32> a;for(auto& x:a)x=int(gen()%256)-128;
                for(unsigned r=0;r<8;++r) {
                    int32_t emu=0,direct=0;
                    for(unsigned depth=0;depth<8;++depth)for(unsigned j=0;j<4;++j) {
                        const unsigned k=depth*4+j;
                        unsigned bits=(b[(depth/4)*8+r]>>(2*((depth%4)*4+j)))&3u;
                        emu+=decode_s2(bits)*a[k];
                        direct+=(int(native_code(rows[r],k32+k))-1)*a[k];
                    }
                    require(emu==direct,"s2/s8 depth, sign, and output-column semantics");
                }
            }
            // A16 reads the SAME signed2 buffer; verify every FP16 VNNI position.
            for(unsigned step=0;step<16;++step)for(unsigned k=0;k<16;++k)for(unsigned r=0;r<8;++r) {
                unsigned c=(u32(tile+step*32+r*4)>>(2*k))&3u;
                uint16_t d=u16(tile+512+2*r);
                uint16_t actual=c==0?0:uint16_t(d^(c==3?0x8000u:0u));
                require(half_to_float(actual)==weight_at(native,s,e,rt*8+r,kb*256+step*16+k),"A16 signed2 tile restore");
            }
        }
        const uint32_t tokens=2,topk=3;
        std::vector<int32_t> ids={0,1,int32_t(s.experts-1),int32_t(s.experts-1),2,1};
        for(bool per:{false,true}) {
            uint32_t rows=tokens*(per?topk:1);std::vector<float>x(size_t(rows)*s.k);
            for(auto&v:x)v=float(int(gen()%8193)-4096)/713.f;
            for(unsigned g:{32u,128u,256u}) {
                auto qa=quantize_a8(x,rows,s.k,g);auto gold=reference_a8(native,s,qa,ids,tokens,topk,per);
                std::vector<float> deq(x.size());for(size_t i=0;i<x.size();++i)deq[i]=float(qa.q[i])*qa.scale[i/g];
                auto brute=reference(native,s,deq,ids,tokens,topk,per,false);
                require(errors(gold,brute).nmse<1e-12,"grouped integer ref vs independent float dequant ref");
                for(unsigned split:{1u,2u,4u,8u})if((s.k/256)%split==0) {
                    std::vector<float> out(gold.size(),0);
                    for(size_t job=0;job<ids.size();++job)for(unsigned rt=0;rt<s.m/8;++rt)for(unsigned sp=0;sp<split;++sp) {
                        std::array<float,8> acc{};const unsigned count=(s.k/256)/split;const size_t xr=per?job:job/topk;
                        for(unsigned kb=sp*count;kb<(sp+1)*count;++kb) {
                            const auto* tile=packed.data()+tile_offset(s,ids[job],rt,kb);
                            for(unsigned kg=0;kg<256;kg+=g)for(unsigned r=0;r<8;++r) {
                                int32_t dot=0;for(unsigned j=0;j<g;++j)dot+=s2tile8_at(tile,kg+j,r)*qa.q[xr*s.k+kb*256+kg+j];
                                const float dw=half_to_float(u16(tile+512+2*r)),da=qa.scale[xr*(s.k/g)+(kb*256+kg)/g];
                                acc[r]+=float(dot)*(dw*da);
                            }
                        }
                        for(unsigned r=0;r<8;++r)out[job*s.m+rt*8+r]+=acc[r];
                    }
                    auto er=errors(out,gold);require(er.finite&&er.nmse<1e-10&&er.max_abs_over_rms<1e-4,"split-K independent per-weight/per-activation scales");
                }
            }
        }
        auto bad=packed;bad[0]=uint8_t((bad[0]&~3u)|2u);
        rejects([&]{unpack_s2tile8(bad,s);},"reject -2 in strict ternary payload");
    }
    std::cout<<"PASS: "<<checks<<" W2A8 CPU checks: INT2 mapping/VNNI/depth semantics, shared A16 layout, A8 RNE/groups, expert offsets, split-K.\n"
             <<"No SYCL compilation or GPU execution is implied.\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
