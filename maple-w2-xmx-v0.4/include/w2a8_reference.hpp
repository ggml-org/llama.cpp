// SPDX-License-Identifier: MIT
#pragma once
#include "tq2_reference.hpp"

namespace maple_w2 {
// A8 group size is independent of the TQ2 weight scale group (always 256).
inline void check_a8_group(uint32_t g) {
    if(g!=32 && g!=128 && g!=256)
        throw std::invalid_argument("A8 group must be 32,128,256");
}
inline unsigned signed2_from_tq(unsigned c) {
    if(c>2) throw std::invalid_argument("reserved TQ2 code 3 cannot be ternary signed INT2");
    return (c+3u)&3u; // TQ 0,1,2 -> s2 bits 3,0,1 -> -1,0,+1
}
inline int decode_s2(unsigned c) { return c<2 ? int(c) : int(c)-4; }
// Signed INT2 VNNI: B K32xN8 uses exactly 16 DWORDs.
// DWORD[(k/16)*8+n] bit[2*(k%16)] stores B[k,n].
inline std::vector<uint8_t> repack_s2tile8(const std::vector<uint8_t>& w,Shape s) {
    validate_tq2(w,s);
    std::vector<uint8_t> out(w.size(),0);
    for(uint32_t e=0;e<s.experts;++e) for(uint32_t rt=0;rt<s.m/8;++rt)
    for(uint32_t kb=0;kb<s.k/256;++kb) {
        uint8_t* dst=out.data()+tile_offset(s,e,rt,kb);
        for(unsigned r=0;r<8;++r) {
            const uint8_t* src=w.data()+native_offset(s,e,rt*8+r,kb);
            put16(dst+512+2*r,u16(src+64));
            for(unsigned s16=0;s16<16;++s16) {
                uint32_t bits=0;
                for(unsigned j=0;j<16;++j)
                    bits|=signed2_from_tq(native_code(src,s16*16+j))<<(2*j);
                put32(dst+s16*32+r*4,bits);
            }
        }
    }
    return out;
}
inline int s2tile8_at(const uint8_t* t,unsigned k,unsigned r) {
    return decode_s2((u32(t+(k/16)*32+r*4)>>(2*(k%16)))&3u);
}
inline std::vector<uint8_t> unpack_s2tile8(const std::vector<uint8_t>& w,Shape s) {
    if(w.size()!=nbytes(s)) throw std::invalid_argument("s2tile8 length mismatch");
    std::vector<uint8_t> out(w.size(),0);
    for(uint32_t e=0;e<s.experts;++e) for(uint32_t rt=0;rt<s.m/8;++rt)
    for(uint32_t kb=0;kb<s.k/256;++kb) for(unsigned r=0;r<8;++r) {
        auto* dst=out.data()+native_offset(s,e,rt*8+r,kb);
        auto* src=w.data()+tile_offset(s,e,rt,kb);
        put16(dst+64,u16(src+512+2*r));
        for(unsigned k=0;k<256;++k) {
            int v=s2tile8_at(src,k,r);
            if(v == -2) throw std::invalid_argument("reserved s2 -2 in strict ternary weights");
            set_native_code(dst,k,unsigned(v+1));
        }
    }
    validate_tq2(out,s); return out;
}
// Explicit round-to-nearest-even, independent of process floating point mode.
inline int8_t quant_rne(float v,float d) {
    float z=v/d;
    z=std::max(-127.f,std::min(127.f,z));
    float a=std::abs(z); auto i=uint32_t(a); const float f=a-float(i);
    i+=uint32_t(f>0.5f || (f==0.5f && (i&1u)));
    return int8_t(z<0 ? -int(i):int(i));
}
struct QuantizedHost {
    uint32_t rows=0,k=0,group=0;
    std::vector<int8_t> q;
    std::vector<float> scale;
};
inline QuantizedHost quantize_a8(const std::vector<float>& x,uint32_t rows,uint32_t k,uint32_t group) {
    check_a8_group(group);
    if(!rows || !k || k%group || x.size()!=checked_mul(rows,k))
        throw std::invalid_argument("quantization shape mismatch");
    QuantizedHost a{rows,k,group,std::vector<int8_t>(x.size()),std::vector<float>(size_t(rows)*k/group)};
    for(size_t g=0;g<a.scale.size();++g) {
        float mx=0;
        for(uint32_t j=0;j<group;++j) {
            float v=x[g*group+j]; if(!std::isfinite(v)) throw std::invalid_argument("nonfinite activation");
            mx=std::max(mx,std::abs(v));
        }
        // Zero groups have d=1. Extremely tiny groups have a normal positive scale.
        float d=mx==0 ? 1.f : std::max(mx/127.f,std::numeric_limits<float>::min());
        a.scale[g]=d;
        for(uint32_t j=0;j<group;++j) a.q[g*group+j]=quant_rne(x[g*group+j],d);
    }
    return a;
}
inline std::vector<float> reference_a8(const std::vector<uint8_t>& w,Shape s,const QuantizedHost& a,
    const std::vector<int32_t>& ids,uint32_t tokens,uint32_t topk,bool per) {
    validate_tq2(w,s); check_a8_group(a.group);
    if(!tokens||!topk||ids.size()!=size_t(tokens)*topk || a.k!=s.k || a.rows!=tokens*(per?topk:1) ||
       a.q.size()!=size_t(a.rows)*a.k || a.scale.size()!=a.q.size()/a.group)
        throw std::invalid_argument("A8 reference dimensions mismatch");
    std::vector<float> out(ids.size()*s.m);
    for(size_t job=0;job<ids.size();++job) {
        if(ids[job]<0||uint32_t(ids[job])>=s.experts) throw std::invalid_argument("expert id out of range");
        size_t xr=per?job:job/topk;
        for(uint32_t r=0;r<s.m;++r) {
            double sum=0;
            for(uint32_t k=0;k<s.k;k+=a.group) {
                const auto* b=w.data()+native_offset(s,uint32_t(ids[job]),r,k/256);
                int32_t dot=0;
                for(uint32_t j=0;j<a.group;++j)
                    dot+=(int(native_code(b,(k+j)%256))-1)*int(a.q[xr*s.k+k+j]);
                sum+=double(dot)*half_to_float(u16(b+64))*a.scale[xr*(s.k/a.group)+k/a.group];
            }
            out[job*s.m+r]=float(sum);
        }
    }
    return out;
}
} // namespace maple_w2
