#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace maple_tq2 {
constexpr std::size_t block_values=256, block_bytes=66;
inline int decode_code(const std::uint8_t * block,std::size_t index) {
    if(index>=block_values) throw std::out_of_range("TQ2 element index");
    const auto byte=32*(index/128)+(index%32);
    return int((block[byte]>>(2*((index%128)/32)))&3)-1;
}
inline std::uint32_t unpack4(const std::uint8_t * block,std::size_t packed_index) {
    if(packed_index>=64) throw std::out_of_range("TQ2 packed index");
    const auto i=packed_index*4;
    const auto byte=32*(i/128)+(i%32);
    std::uint32_t raw=std::uint32_t(block[byte])|(std::uint32_t(block[byte+1])<<8)
        |(std::uint32_t(block[byte+2])<<16)|(std::uint32_t(block[byte+3])<<24);
    const auto q=(raw>>(2*((i%128)/32)))&0x03030303u;
    return ((q|0x80808080u)-0x01010101u)^0x80808080u;
}
inline int signed_lane(std::uint32_t packed,int lane) {
    const int x=int((packed>>(lane*8))&255u);
    return x>=128?x-256:x;
}
inline std::int32_t dot4(std::uint32_t a,std::uint32_t b) {
    std::int32_t s=0;for(int i=0;i<4;++i)s+=signed_lane(a,i)*signed_lane(b,i);return s;
}
inline std::size_t checked_bytes(std::size_t experts,std::size_t rows,std::size_t k) {
    if(!experts||!rows||!k||k%256) throw std::invalid_argument("positive shape and K%256 == 0 required");
    constexpr auto limit=static_cast<std::size_t>(-1);
    std::size_t v=k/256;
    for(const auto f: {block_bytes,rows,experts}) {
        if(v>limit/f) throw std::overflow_error("TQ2 tensor size overflow");
        v*=f;
    }
    return v;
}
inline std::size_t job_capacity(std::size_t assignments,std::size_t experts,std::size_t rows) {
    // sum ceil(c_e/4) <= min(A, floor((A+3E)/4)). Empty experts cost zero.
    if(!assignments||!experts||!rows) throw std::invalid_argument("positive job dimensions required");
    const auto active=assignments<experts?assignments:experts;
    constexpr auto limit=static_cast<std::size_t>(-1);
    if(active>(limit-assignments)/3 || rows>limit-31) throw std::overflow_error("job count overflow");
    const auto token_tiles=(assignments+3*active)/4;
    const auto row_tiles=(rows+31)/32;
    if(token_tiles>limit/row_tiles) throw std::overflow_error("job capacity overflow");
    return token_tiles*row_tiles;
}
} // namespace maple_tq2
