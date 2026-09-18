// SPDX-License-Identifier: MIT
#pragma once
#include <sycl/ext/intel/esimd.hpp>
namespace sycl::ext::intel::esimd::xmx {
enum class dpas_argument_type{s2,s8};
template<int SD,int RC,class RT,class CT,class BT,class AT,dpas_argument_type BP,dpas_argument_type AP>
simd<RT,8>dpas(const simd<CT,8>&c,const simd<BT,16>&b,const simd<AT,32>&a){
    static_assert(SD==8&&RC==1&&BP==dpas_argument_type::s2&&AP==dpas_argument_type::s8);
    simd<RT,8>r;
    for(int n=0;n<8;++n){int32_t sum=c[n];for(int k=0;k<32;++k){
        int bits=int((b[(k/16)*8+n]>>(2*(k%16)))&3u);int v=bits<2?bits:bits-4;
        sum+=v*int(a[k]);}r[n]=RT(sum);}
    return r;
}
} // namespace
