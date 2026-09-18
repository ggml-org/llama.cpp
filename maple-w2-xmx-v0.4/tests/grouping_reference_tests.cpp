// SPDX-License-Identifier: MIT
#include "expert_grouping_reference.hpp"
#include "moe_reference.hpp"
#include <iostream>
#include <random>
#include <set>
#include <string>
using namespace maple_w2;
static size_t checks=0;
static void need(bool x,const char*msg){++checks;if(!x)throw std::runtime_error(msg);}
static void validate_case(const std::vector<int32_t>&ids,uint32_t e) {
    auto a=group_jobs_reference(ids,e),b=group_jobs_chunk_emulation(ids,e);
    need(same_grouping(a,b),"chunk-scan != cursor-sort reference");
    std::vector<int32_t>gold(ids.size());std::iota(gold.begin(),gold.end(),0);
    std::stable_sort(gold.begin(),gold.end(),[&](int32_t i,int32_t j){return grouping_key(ids[size_t(i)],e)<grouping_key(ids[size_t(j)],e);});
    need(a.sorted_to_job==gold,"stable_sort mismatch");need(a.offsets.back()==int32_t(ids.size()),"bad total");
    std::vector<int>seen(ids.size(),0);
    for(uint32_t bucket=0;bucket<=e;++bucket){need(a.offsets[bucket+1]-a.offsets[bucket]==a.counts[bucket],"bad count");
        int32_t prev=-1;
        for(int32_t p=a.offsets[bucket];p<a.offsets[bucket+1];++p){int32_t j=a.sorted_to_job[size_t(p)];
            need(j>=0&&size_t(j)<ids.size(),"bad job index");need(j>prev,"unstable bucket");prev=j;
            need(grouping_key(ids[size_t(j)],e)==bucket,"wrong bucket");++seen[size_t(j)];}}
    for(auto n:seen)need(n==1,"not a permutation");
}
static void address_checks() {
    std::mt19937 rng(770);
    for(uint32_t q:{1u,4u,13u,64u})for(uint32_t top:{1u,8u}) {
        const size_t jobs=size_t(q)*top;std::vector<int32_t>ids(jobs);
        for(auto&v:ids)v=int32_t(rng()%17);auto plan=group_jobs_reference(ids,17);
        for(uint32_t width:{256u,512u,2048u})for(uint32_t split:{1u,2u,4u}) {
            const size_t logical=jobs*(width/8)*split;
            std::vector<uint8_t>seen(jobs*width*split,0);
            for(size_t idx=0;idx<logical;++idx){size_t z=idx;uint32_t sp=uint32_t(z%split);z/=split;
                uint32_t rt=uint32_t(z%(width/8));size_t original=size_t(plan.sorted_to_job[z/(width/8)]);
                for(unsigned lane=0;lane<8;++lane)++seen[size_t(sp)*jobs*width+original*width+rt*8+lane];}
            for(auto x:seen)need(x==1,"grouped matvec writes wrong output/scratch lane");
        }
    }
}
static void quant_and_route_invariance() {
    // Existing ternary scalar arithmetic executed in the grouped job order,
    // written back to original slots. Quantizers and weighted sum do not change.
    Shape s{256,256,3};std::vector<uint8_t>w(nbytes(s),0);
    for(uint32_t e=0;e<3;++e)for(uint32_t r=0;r<256;++r){auto*b=w.data()+native_offset(s,e,r,0);
        put16(b+64,float_to_half(float(1+(e+r)%7)/512));for(unsigned k=0;k<256;++k)set_native_code(b,k,(k+r+e)%3);}
    const uint32_t q=4,top=2;std::vector<int32_t>ids={2,0,1,2,0,1,2,1};std::vector<float>x(q*256);
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%37)-18)/19;auto a=quantize_a8(x,q,256,32);
    auto baseline=reference_a8(w,s,a,ids,q,top,false);auto plan=group_jobs_reference(ids,3);std::vector<float>grouped(baseline.size(),NAN);
    for(auto jj:plan.sorted_to_job){size_t j=size_t(jj),xr=j/top;
        for(uint32_t r=0;r<256;++r){double sum=0;
            for(uint32_t k=0;k<256;k+=32){int32_t dot=0;const uint8_t*b=w.data()+native_offset(s,uint32_t(ids[j]),r,0);
                for(uint32_t z=0;z<32;++z)dot+=(int(native_code(b,k+z))-1)*int(a.q[xr*256+k+z]);
                sum+=double(dot)*half_to_float(u16(b+64))*a.scale[xr*8+k/32];}
            grouped[j*256+r]=float(sum);}}
    need(baseline==grouped,"group scheduling changes arithmetic");
    auto h0=swiglu_reference(baseline,baseline),h1=swiglu_reference(grouped,grouped);
    need(h0==h1,"GLU changed");auto z0=quantize_a8(h0,q*top,256,32),z1=quantize_a8(h1,q*top,256,32);
    need(z0.q==z1.q&&z0.scale==z1.scale,"hidden quant changed");
    auto d0=reference_a8(w,s,z0,ids,q,top,true),d1=reference_a8(w,s,z1,ids,q,top,true);
    std::vector<float>routes={.25,.75,.5,.5,.75,.25,.1,.9};
    need(weighted_sum_reference(d0,routes,q,top,256)==weighted_sum_reference(d1,routes,q,top,256),"route sum changed");
}
int main()try {
    std::mt19937 rng(256);
    for(uint32_t e:{1u,3u,8u,256u})for(size_t n:{size_t(1),size_t(8),size_t(104),size_t(127),size_t(128),size_t(129),size_t(257),size_t(2048),size_t(16384)}) {
        std::vector<int32_t>v(n);
        for(auto&x:v)x=int32_t(rng()%e);validate_case(v,e);
        std::fill(v.begin(),v.end(),int32_t(e-1));validate_case(v,e);
        for(size_t j=0;j<n;++j)v[j]=j%3==0?-1:(j%3==1?int32_t(e):int32_t(j%e));validate_case(v,e);
    }
    bool bad=false;try{group_jobs_reference({},3);}catch(const std::invalid_argument&){bad=true;}need(bad,"empty jobs accepted");
    for(uint32_t e:{0u,257u}){bad=false;try{group_jobs_reference({0},e);}catch(const std::invalid_argument&){bad=true;}need(bad,"invalid expert count accepted");}
    address_checks();quant_and_route_invariance();
    std::cout<<"PASS "<<checks<<" CPU grouping checks: stable permutation, invalid bucket, tails, original job/scatter, split-K, quant/route invariance. NOT GPU execution.\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
