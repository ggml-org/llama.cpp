// SPDX-License-Identifier: MIT
#pragma once
#include "w2a8_reference.hpp"
namespace maple_w2 {
// Maple clipped SwiGLU: clip gate BEFORE SiLU, upper side only.
// This is not GPT-OSS's alpha=1.702 or its (up+1) variant.
inline float maple_swiglu(float gate,float up,float limit=7.f) {
    if(!std::isfinite(gate)||!std::isfinite(up)||!std::isfinite(limit)||limit<=0)
        throw std::invalid_argument("nonfinite SwiGLU input or invalid clamp");
    const float g=std::min(gate,limit),u=std::max(-limit,std::min(up,limit));
    const float z=std::exp(-std::abs(g));
    const float si=g>=0 ? g/(1.f+z) : (g*z)/(1.f+z);
    return si*u;
}
inline std::vector<float> swiglu_reference(const std::vector<float>& gate,const std::vector<float>& up,float limit=7.f) {
    if(gate.size()!=up.size())throw std::invalid_argument("gate/up length mismatch");
    std::vector<float> y(gate.size());
    for(size_t i=0;i<y.size();++i)y[i]=maple_swiglu(gate[i],up[i],limit);
    return y;
}
inline void validate_routes(const std::vector<float>& routes,uint32_t tokens,uint32_t topk) {
    if(!tokens||!topk||routes.size()!=size_t(tokens)*topk)throw std::invalid_argument("route shape mismatch");
    for(uint32_t t=0;t<tokens;++t){double sum=0;for(uint32_t j=0;j<topk;++j){float x=routes[size_t(t)*topk+j];
        if(!std::isfinite(x)||x<0)throw std::invalid_argument("route weight must be finite and nonnegative");sum+=x;}
        if(std::abs(sum-1.0)>1e-4)throw std::invalid_argument("pass already normalized top-k routing weights; no implicit renormalization");}
}
inline std::vector<float> weighted_sum_reference(const std::vector<float>& down,const std::vector<float>& routes,
                                                uint32_t tokens,uint32_t topk,uint32_t width) {
    validate_routes(routes,tokens,topk);
    if(!width||down.size()!=checked_mul(checked_mul(tokens,topk),width))throw std::invalid_argument("down length mismatch");
    std::vector<float> y(size_t(tokens)*width);
    for(uint32_t t=0;t<tokens;++t)for(uint32_t k=0;k<width;++k){double sum=0;
        for(uint32_t e=0;e<topk;++e){float v=down[(size_t(t)*topk+e)*width+k];
            if(!std::isfinite(v))throw std::invalid_argument("nonfinite down output");sum+=double(v)*routes[size_t(t)*topk+e];}
        y[size_t(t)*width+k]=float(sum);}
    return y;
}
} // namespace maple_w2
