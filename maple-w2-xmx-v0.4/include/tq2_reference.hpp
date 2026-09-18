// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace maple_w2 {
constexpr uint32_t block_k = 256, block_bytes = 66, tile_rows = 8;
constexpr uint32_t tile_bytes = tile_rows * block_bytes; // 512 codes + 16 scale bytes
struct Shape { uint32_t k = 0, m = 0, experts = 0; };
inline size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("tensor byte-size overflow");
    return a * b;
}
inline size_t nbytes(Shape s) {
    if (!s.k || s.k % 256 || !s.m || s.m % 8 || !s.experts)
        throw std::invalid_argument("requires K % 256 == 0, M % 8 == 0 and positive dimensions");
    return checked_mul(checked_mul(checked_mul(s.experts, s.m), s.k/256), 66);
}
inline uint16_t u16(const uint8_t *p) { return uint16_t(p[0]) | uint16_t(p[1])<<8; }
inline uint32_t u32(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
inline void put16(uint8_t *p, uint16_t x) { p[0] = uint8_t(x); p[1] = uint8_t(x>>8); }
inline void put32(uint8_t *p, uint32_t x) {
    for (unsigned j=0; j<4; ++j) p[j] = uint8_t(x >> (j*8));
}
inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u)<<16;
    uint32_t exp = (h>>10)&31u, frac = h&1023u, v;
    if (!exp) {
        if (!frac) v=sign;
        else {
            int e=-14;
            while (!(frac&1024u)) { frac<<=1; --e; }
            v=sign | uint32_t(e+127)<<23 | (frac&1023u)<<13;
        }
    } else if (exp==31) v=sign | 0x7f800000u | frac<<13;
    else v=sign | (exp+112u)<<23 | frac<<13;
    float f; std::memcpy(&f, &v, 4); return f;
}
// IEEE binary32 -> binary16, round-to-nearest ties-to-even; no CPU ISA dependency.
inline uint16_t float_to_half(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint16_t sign=uint16_t((u>>16)&0x8000u);
    const uint32_t ex=(u>>23)&255u, fr=u&0x7fffffu;
    if (ex==255) return uint16_t(sign | (fr ? 0x7e00u : 0x7c00u));
    int e=int(ex)-127+15;
    if (e>=31) return uint16_t(sign|0x7c00u);
    if (e<=0) {
        if (e < -10) return sign;
        uint32_t v=fr|0x800000u;
        const unsigned shift=unsigned(14-e);
        uint32_t out=v>>shift, rem=v&((1u<<shift)-1);
        const uint32_t mid=1u<<(shift-1);
        out += (rem>mid || (rem==mid && (out&1)));
        return uint16_t(sign|out);
    }
    uint32_t out=(uint32_t(e)<<10)|(fr>>13), rem=fr&0x1fffu;
    out += (rem>0x1000u || (rem==0x1000u && (out&1)));
    return uint16_t(sign|out);
}
inline size_t native_offset(Shape s,uint32_t e,uint32_t r,uint32_t kb) {
    return ((size_t(e)*s.m+r)*(s.k/256)+kb)*66;
}
inline size_t tile_offset(Shape s,uint32_t e,uint32_t rt,uint32_t kb) {
    return ((size_t(e)*(s.m/8)+rt)*(s.k/256)+kb)*528;
}
inline unsigned native_code(const uint8_t *b, unsigned i) {
    return (b[(i/128)*32+i%32] >> (2*((i%128)/32)))&3u;
}
inline void set_native_code(uint8_t *b,unsigned i,unsigned c) {
    const unsigned ix=(i/128)*32+i%32, sh=2*((i%128)/32);
    b[ix]=uint8_t((b[ix]&~(3u<<sh)) | (c<<sh));
}
inline unsigned vnni_index(unsigned k16,unsigned r8) {
    return (k16/2)*16+r8*2+k16%2;
}
inline void validate_tq2(const std::vector<uint8_t>& w,Shape s) {
    if(w.size()!=nbytes(s)) throw std::invalid_argument("TQ2 payload length mismatch");
    for(size_t i=0;i<w.size();i+=66) {
        uint16_t d=u16(w.data()+i+64);
        if((d&0x7c00u)==0x7c00u) throw std::invalid_argument("non-finite weight scale");
        for(unsigned j=0;j<64;++j) {
            unsigned v=w[i+j];
            if ((v & (v>>1) & 0x55u)!=0)
                throw std::invalid_argument("TQ2 code 3 found; strict ternary kernel must fall back");
        }
    }
}
// Optional lossless repack. One tile still occupies exactly 8*66 bytes.
// [expert][8-row tile][256-K block][16 DPAS slices of 32 bytes][8 half scales].
inline std::vector<uint8_t> repack_tile8(const std::vector<uint8_t>& w,Shape s) {
    validate_tq2(w,s);
    std::vector<uint8_t> out(w.size(),0);
    for(uint32_t e=0;e<s.experts;++e) for(uint32_t rt=0;rt<s.m/8;++rt)
    for(uint32_t kb=0;kb<s.k/256;++kb) {
        uint8_t *dst=out.data()+tile_offset(s,e,rt,kb);
        for(unsigned r=0;r<8;++r) {
            const uint8_t *src=w.data()+native_offset(s,e,rt*8+r,kb);
            std::memcpy(dst+512+r*2,src+64,2);
            for(unsigned k=0;k<256;++k) {
                const unsigned linear=vnni_index(k%16,r);
                const unsigned bit=2*linear;
                const unsigned off=(k/16)*32+bit/8;
                dst[off] |= uint8_t(native_code(src,k) << (bit%8));
            }
        }
    }
    return out;
}
inline unsigned tile_code(const uint8_t *tile,unsigned k,unsigned r) {
    unsigned bit=vnni_index(k%16,r)*2;
    return (tile[(k/16)*32+bit/8]>>(bit%8))&3;
}
inline std::vector<uint8_t> unpack_tile8(const std::vector<uint8_t>& w,Shape s) {
    if(w.size()!=nbytes(s)) throw std::invalid_argument("tile8 length mismatch");
    std::vector<uint8_t> out(w.size(),0);
    for(uint32_t e=0;e<s.experts;++e) for(uint32_t rt=0;rt<s.m/8;++rt)
    for(uint32_t kb=0;kb<s.k/256;++kb) {
        const uint8_t *src=w.data()+tile_offset(s,e,rt,kb);
        for(unsigned r=0;r<8;++r) {
            uint8_t *dst=out.data()+native_offset(s,e,rt*8+r,kb);
            std::memcpy(dst+64,src+512+r*2,2);
            for(unsigned k=0;k<256;++k) set_native_code(dst,k,tile_code(src,k,r));
        }
    }
    return out;
}
inline float weight_at(const std::vector<uint8_t>& w,Shape s,uint32_t e,uint32_t r,uint32_t k) {
    auto *b=w.data()+native_offset(s,e,r,k/256);
    return float(int(native_code(b,k%256))-1)*half_to_float(u16(b+64));
}
inline std::vector<float> reference(const std::vector<uint8_t>& w,Shape s,
        const std::vector<float>& x,const std::vector<int32_t>& ids,
        uint32_t tokens,uint32_t topk,bool per_selection,bool round_x_half) {
    validate_tq2(w,s);
    if (!tokens || !topk || ids.size()!=size_t(tokens)*topk ||
        x.size()!=size_t(tokens)*(per_selection?topk:1)*s.k)
        throw std::invalid_argument("reference dimensions mismatch");
    std::vector<float> out(ids.size()*s.m);
    for(size_t j=0;j<ids.size();++j) {
        if(ids[j]<0 || uint32_t(ids[j])>=s.experts) throw std::invalid_argument("expert id out of range");
        size_t xr=per_selection?j:j/topk;
        for(uint32_t r=0;r<s.m;++r) {
            double sum=0;
            for(uint32_t k=0;k<s.k;++k) {
                float a=x[xr*s.k+k];
                if(round_x_half) a=half_to_float(float_to_half(a));
                sum+=double(weight_at(w,s,uint32_t(ids[j]),r,k))*a;
            }
            out[j*s.m+r]=float(sum);
        }
    }
    return out;
}
struct Error { double nmse=0, max_abs_over_rms=0, max_abs=0; bool finite=true; };
inline Error errors(const std::vector<float>& y,const std::vector<float>& ref) {
    if(y.empty() || y.size()!=ref.size()) throw std::invalid_argument("error metric sizes mismatch");
    double se=0,rr=0,ma=0; bool finite=true;
    for(size_t i=0;i<y.size();++i) {
        if(!std::isfinite(y[i])||!std::isfinite(ref[i])) finite=false;
        double d=double(y[i])-ref[i]; se+=d*d; rr+=double(ref[i])*ref[i]; ma=std::max(ma,std::abs(d));
    }
    double rms=std::sqrt(rr/y.size());
    return {se/std::max(rr,1e-30),ma/std::max(rms,1e-15),ma,finite};
}
struct Capsule { Shape shape; std::vector<uint8_t> bytes; };
inline Capsule load_capsule(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open capsule: "+path);
    std::array<uint8_t,32> h{}; f.read(reinterpret_cast<char*>(h.data()),h.size());
    const char magic[8]={'M','W','2','T','Q','2',0,0};
    if(!f || std::memcmp(h.data(),magic,8) || u32(h.data()+8)!=1)
        throw std::runtime_error("invalid MW2TQ2 capsule header");
    Shape s{u32(h.data()+12),u32(h.data()+16),u32(h.data()+20)};
    size_t sz=nbytes(s);
    uint64_t stored=uint64_t(u32(h.data()+24)) | uint64_t(u32(h.data()+28))<<32;
    if(stored!=sz) throw std::runtime_error("capsule byte length mismatch");
    f.seekg(0,std::ios::end);
    if(f.tellg()!=std::streamoff(32+sz)) throw std::runtime_error("capsule truncated or trailing data");
    f.seekg(32); Capsule c{s,std::vector<uint8_t>(sz)};
    f.read(reinterpret_cast<char*>(c.bytes.data()),sz);
    if(!f) throw std::runtime_error("capsule read failed");
    validate_tq2(c.bytes,s); return c;
}
} // namespace maple_w2
