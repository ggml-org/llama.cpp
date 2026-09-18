// SPDX-License-Identifier: MIT
#pragma once
#include "w2a8_reference.hpp"
#include <cstring>
namespace maple_w2 {
// XMX v0.4 uses RNE (ties-to-even), NOT the Vulkan lab's half-away rule.
// Keep the production quantizer unchanged; classify a boundary mismatch rather
// than calling the whole projection wrong or admitting arbitrary +/-1 codes.
struct A8AuditPolicy {double max_scale_relative=5e-6,max_normalized_error=2e-4;};
struct A8Audit {
    bool pass=true;
    size_t codes=0,groups=0,cpu_code_differences=0,cpu_scale_differences=0;
    size_t actual_scale_code_differences=0,boundary_codes=0,bad_codes=0,bad_scales=0,nonfinite_inputs=0;
    uint32_t max_scale_ulp=0;
    double max_scale_relative=0;
};
inline uint32_t f32_bits(float x) {uint32_t u;std::memcpy(&u,&x,4);return u;}
inline int rne_normalized(double z) {
    z=std::max(-127.,std::min(127.,z));double a=std::abs(z);auto k=uint32_t(a);double f=a-k;
    k+=uint32_t(f>.5||(f==.5&&(k&1u)));return z<0?-int(k):int(k);
}
inline A8Audit audit_a8(const QuantizedHost& got,const std::vector<float>& x,A8AuditPolicy policy={}) {
    check_a8_group(got.group);
    if(!got.rows||!got.k||got.k%got.group||x.size()!=checked_mul(got.rows,got.k)||
       got.q.size()!=x.size()||got.scale.size()!=x.size()/got.group||
       !std::isfinite(policy.max_scale_relative)||!std::isfinite(policy.max_normalized_error)||
       policy.max_scale_relative<0||policy.max_normalized_error<0||policy.max_normalized_error>=.5)
        throw std::invalid_argument("invalid A8 audit contract/shape");
    A8Audit a;a.codes=x.size();a.groups=got.scale.size();
    for(size_t g=0;g<got.scale.size();++g) {
        float mx=0;
        for(uint32_t j=0;j<got.group;++j) {
            float v=x[g*got.group+j];
            if(!std::isfinite(v)){++a.nonfinite_inputs;continue;}
            mx=std::max(mx,std::abs(v));
        }
        const float ref=mx==0?1.f:std::max(mx/127.f,std::numeric_limits<float>::min());
        const float d=got.scale[g];
        if(f32_bits(d)!=f32_bits(ref))++a.cpu_scale_differences;
        if(!std::isfinite(d)||d<=0) {++a.bad_scales;continue;}
        const auto db=f32_bits(d),rb=f32_bits(ref);a.max_scale_ulp=std::max(a.max_scale_ulp,db>rb?db-rb:rb-db);
        const double relative=std::abs(double(d)-ref)/ref;a.max_scale_relative=std::max(a.max_scale_relative,relative);
        if((mx==0&&d!=1.f)||relative>policy.max_scale_relative)++a.bad_scales;
        for(uint32_t j=0;j<got.group;++j) {
            size_t i=g*got.group+j;float v=x[i];if(!std::isfinite(v))continue;
            const int q=got.q[i];a.cpu_code_differences+=q!=int(quant_rne(v,ref));
            const int strict_actual=int(quant_rne(v,d));
            if(q!=strict_actual)++a.actual_scale_code_differences;
            if(q==-128){++a.bad_codes;continue;}
            // A code must be an RNE result within an explicit normalized error
            // interval around the ACTUAL scale, not simply 'within one int'.
            const double z=double(v)/d;
            const int lo=rne_normalized(z-policy.max_normalized_error),hi=rne_normalized(z+policy.max_normalized_error);
            if(q<lo||q>hi)++a.bad_codes;
            else if(q!=strict_actual)++a.boundary_codes;
        }
    }
    a.pass=!a.bad_codes&&!a.bad_scales&&!a.nonfinite_inputs;return a;
}
} // namespace maple_w2
