// SPDX-License-Identifier: MIT
// Executes src/maple_w2a8_grouped.cpp with scalar gather/DPAS test doubles.
// Tests real indexing/control code; NOT real SYCL/DPAS compiler or GPU execution.
#include "maple_w2a8.hpp"
#include <iostream>
#include <random>
using namespace maple_w2;
static size_t checks=0;
static void test(bool ok){++checks;if(!ok)throw std::runtime_error("reuse emulation assertion "+std::to_string(checks));}
static std::vector<uint8_t> weights(Shape s,unsigned seed) {
    std::mt19937 gen(seed);std::vector<uint8_t>w(nbytes(s),0);
    for(size_t i=0;i<w.size();i+=66){for(unsigned j=0;j<256;++j)set_native_code(w.data()+i,j,gen()%3);
        const float scale=(int(gen()%17)-8)/512.f;put16(w.data()+i+64,float_to_half(scale));}return w;
}
int main()try {
    sycl::mock_execute=true;sycl::queue q;std::mt19937 gen(7);
    for(Shape s:{Shape{512,24,7},Shape{2048,24,7},Shape{512,2048,3}}) {
    const uint32_t Q=s.m>24?2:13,topk=s.m>24?2:3;const size_t jobs=Q*topk;
    const auto w0=weights(s,1),w1=weights(s,2),pw0=repack_s2tile8(w0,s),pw1=repack_s2tile8(w1,s);
    std::vector<int32_t>ids(jobs);for(size_t i=0;i<jobs;++i)ids[i]=int32_t((i*i+3)%s.experts);
    auto g=group_jobs_reference(ids,s.experts);
    for(bool per:{false,true})for(bool pair:{false,true})for(uint32_t width:{2u,4u})for(uint32_t split:{1u,2u,4u,8u}) {
        if((s.k/256)%split)continue;
        std::vector<float>x((per?jobs:Q)*s.k);for(auto&v:x)v=float(int(gen()%255)-127)*.013f;
        auto a=quantize_a8(x,per?uint32_t(jobs):Q,s.k,32);auto t=token_tiles_reference(g,width);int32_t total=int32_t(t.first.size());
        std::vector<int32_t>status(jobs,0),invalid(a.scale.size(),0);
        constexpr float poison=123456.f;std::vector<float>y0(jobs*s.m+16,poison),y1(jobs*s.m+16,poison),scratch(jobs*s.m*split*(pair?2:1),poison);
        DeviceProblem p{s,Q,topk,per,Layout::s2tile8,pw0.data(),pair?pw1.data():nullptr,x.data(),ids.data(),y0.data()+8,pair?y1.data()+8:nullptr,status.data()};
        A8Options o{32,A8Mode::prequantized,{split,4},g.sorted_to_job.data(),
            {t.first.data(),t.length.data(),t.expert.data(),&total,expert_token_tile_capacity(jobs,s.experts,width),width}};
        enqueue_a8_token_tiles(q,p,o,{a.q.data(),a.scale.data(),invalid.data(),false},scratch.data(),{});
        auto r0=reference_a8(w0,s,a,ids,Q,topk,per),r1=reference_a8(w1,s,a,ids,Q,topk,per);
        auto check=[&](const auto&y,const auto&ref){for(size_t i=0;i<ref.size();++i)test(std::abs(y[i+8]-ref[i])<=2e-5f);
            for(int i=0;i<8;++i)test(y[i]==poison&&y[y.size()-1-i]==poison);};
        check(y0,r0);if(pair)check(y1,r1);for(auto z:status)test(z==0);
    }
    // Invalid-ID final bucket: valid rows remain produced, bad row zero/status1.
    ids[3]=-1;g=group_jobs_reference(ids,s.experts);auto t=token_tiles_reference(g,4);int32_t total=int32_t(t.first.size());
    std::vector<float>x(Q*s.k,.5f);auto a=quantize_a8(x,Q,s.k,32);std::vector<float>y(jobs*s.m,123.f);
    std::vector<int32_t>st(jobs,0),iv(a.scale.size(),0);
    DeviceProblem p{s,Q,topk,false,Layout::s2tile8,pw0.data(),nullptr,x.data(),ids.data(),y.data(),nullptr,st.data()};
    A8Options o{32,A8Mode::prequantized,{1,4},g.sorted_to_job.data(),{t.first.data(),t.length.data(),t.expert.data(),&total,expert_token_tile_capacity(jobs,s.experts,4),4}};
    enqueue_a8_token_tiles(q,p,o,{a.q.data(),a.scale.data(),iv.data(),false},nullptr,{});
    test(st[3]==1);for(uint32_t r=0;r<s.m;++r)test(y[3*s.m+r]==0.f);
    // Descriptor corruption is rejected via device status instead of OOB reads.
    std::fill(st.begin(),st.end(),0);total=int32_t(o.token_tiles.capacity+1);
    enqueue_a8_token_tiles(q,p,o,{a.q.data(),a.scale.data(),iv.data(),false},nullptr,{});test(st[0]==2);
    }
    std::cout<<"ACTUAL new kernel body, CPU scalar DPAS emulation PASS checks="<<checks<<"; NOT Intel/GPU validation\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
