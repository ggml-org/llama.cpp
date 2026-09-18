// SPDX-License-Identifier: MIT
#include "expert_token_tiles.hpp"
#include "a8_contract_audit.hpp"
#include "moe_profiling.hpp"
#include <iostream>
#include <random>
#include <numeric>
#include <functional>
using namespace maple_w2;
static size_t checks=0;
static void test(bool yes){++checks;if(!yes)throw std::runtime_error("architecture reference assertion "+std::to_string(checks));}
static void throws(const std::function<void()>& fn){bool ok=false;try{fn();}catch(const std::exception&){ok=true;}test(ok);}
int main()try {
    using S=MoeSchedule;
    test(!choose_moe_schedule(S::auto_select,false,2048,8,256,true).grouped);
    test(!choose_moe_schedule(S::auto_select,false,1,8,256,true,{1,0}).grouped);
    test(choose_moe_schedule(S::grouped,false,1,8,256,true).grouped);
    test(!choose_moe_schedule(S::direct,true,2048,8,256,true).grouped);
    test(choose_moe_schedule(S::inherit_v04,true,184,8,256,true).grouped);
    test(!choose_moe_schedule(S::auto_select,false,64,8,256,false,{32,1}).grouped);
    test(!choose_moe_schedule(S::auto_select,false,32,8,256,true,{64,1}).grouped);
    test(!choose_moe_schedule(S::auto_select,false,32,8,256,true,{32,2}).grouped);
    test(choose_moe_schedule(S::auto_select,false,64,8,256,true,{32,2}).grouped);
    throws([]{choose_moe_schedule(S::grouped,false,32,8,256,false);});
    throws([]{choose_moe_schedule(S::direct,false,0,8,256,true);});
    throws([]{check_tokens_per_tile(8);});
    std::mt19937 rng(8123);
    for(uint32_t e:{1u,3u,8u,256u})for(size_t jobs:{1u,8u,13u,127u,128u,129u,1472u,16384u})for(uint32_t width:{1u,2u,4u}) {
        std::vector<int32_t> ids(jobs);
        for(size_t i=0;i<jobs;++i)ids[i]=i%19==0?-1:(i%23==0?int32_t(e):int32_t(rng()%e));
        for(int mode=0;mode<2;++mode) {
            if(mode)std::fill(ids.begin(),ids.end(),int32_t(e-1));
            auto g=group_jobs_reference(ids,e);auto t=token_tiles_reference(g,width);
            test(t.first.size()<=expert_token_tile_capacity(jobs,e,width));std::vector<int>visits(jobs,0);
            for(size_t i=0;i<t.first.size();++i) {
                test(t.length[i]>0&&t.length[i]<=int32_t(width));test(t.expert[i]>=0&&uint32_t(t.expert[i])<=e);
                for(int j=0;j<t.length[i];++j) {
                    size_t orig=size_t(g.sorted_to_job[size_t(t.first[i])+j]);++visits[orig];
                    test(grouping_key(ids[orig],e)==uint32_t(t.expert[i]));
                }
            }
            for(auto v:visits)test(v==1);
        }
    }
    // Numeric reference vs a tile loop: preserve original job addressing, K32
    // scale placement, split reduction order, broadcast and per-selection rows.
    for(bool per:{false,true})for(uint32_t width:{2u,4u})for(uint32_t split:{1u,2u,4u}) {
        const uint32_t Q=13,K=1024,topk=3,E=7,M=16;const size_t J=Q*topk;
        std::vector<int32_t>ids(J);for(auto&v:ids)v=int32_t(rng()%E);
        auto g=group_jobs_reference(ids,E);auto tiles=token_tiles_reference(g,width);
        std::vector<int8_t> weights(E*M*K),aq((per?J:Q)*K);
        std::vector<float>dw(E*M*(K/256)),da(aq.size()/32);
        for(auto&v:weights)v=int8_t(int(rng()%3)-1);for(auto&v:aq)v=int8_t(int(rng()%255)-127);
        for(auto&v:dw)v=(1+rng()%31)/512.f;for(auto&v:da)v=(1+rng()%13)/127.f;
        auto calc=[&](bool tiled) {
            std::vector<float> out(J*M,0.f);
            auto one=[&](size_t job,uint32_t row) {
                float sum=0;
                for(uint32_t sp=0;sp<split;++sp){float part=0;
                    for(uint32_t k=sp*(K/split);k<(sp+1)*(K/split);k+=32){int dot=0;size_t xr=per?job:job/topk;
                        for(uint32_t j=0;j<32;++j)dot+=int(weights[(size_t(ids[job])*M+row)*K+k+j])*int(aq[xr*K+k+j]);
                        part+=float(dot)*(dw[(size_t(ids[job])*M+row)*(K/256)+k/256]*da[xr*(K/32)+k/32]);}
                    sum+=part;}out[job*M+row]=sum;
            };
            if(tiled)for(size_t t=0;t<tiles.first.size();++t)for(uint32_t row=0;row<M;row+=8)
                for(int ti=0;ti<tiles.length[t];++ti)for(uint32_t r=0;r<8;++r)one(size_t(g.sorted_to_job[size_t(tiles.first[t])+ti]),row+r);
            else for(size_t j=0;j<J;++j)for(uint32_t r=0;r<M;++r)one(j,r);
            return out;
        };
        auto a=calc(false),b=calc(true);for(size_t i=0;i<a.size();++i)test(a[i]==b[i]);
    }
    std::vector<float>x(128,0);x[0]=127;x[1]=110.5f;x[2]=111.5f;x[3]=-110.5f;x[4]=-111.5f;
    x[32]=1e-40f;x[64]=1.f;x[65]=0.345f;
    auto q=quantize_a8(x,1,128,32);test(audit_a8(q,x).pass);
    test(q.q[1]==110&&q.q[2]==112&&q.q[3]==-110&&q.q[4]==-112);
    auto bad=q;bad.q[65]++;test(!audit_a8(bad,x).pass); // +/-1 away from a tie is a real bug
    bad=q;bad.q[1]=111;auto a=audit_a8(bad,x);test(a.pass&&a.boundary_codes==1);
    bad=q;bad.q[1]=112;test(!audit_a8(bad,x).pass);
    bad=q;bad.q[0]=-128;test(!audit_a8(bad,x).pass);
    bad=q;bad.scale[3]=2.f;test(!audit_a8(bad,x).pass); // zero group d=1 invariant
    bad=q;bad.scale[2]*=1.01f;test(!audit_a8(bad,x).pass);
    bad=q;bad.scale[0]=std::numeric_limits<float>::quiet_NaN();test(!audit_a8(bad,x).pass);
    auto nanx=x;nanx[0]=std::numeric_limits<float>::infinity();test(!audit_a8(q,nanx).pass);
    // Vulkan legacy half-away diagnostic vector: do NOT copy its expected 111
    // into XMX's ties-even reference (110).
    test(quant_rne(0.7057453393936157f,0.006386835593730211f)==110);
    auto p=profile_intervals({{0,10,0},{5,15,0},{17,27,3}});
    test(p.kernel_sum==30&&p.union_busy==25&&p.overlap==5&&p.idle==2&&p.end-p.begin==27);
    // Submit-order != device start order for independent branches.
    p=profile_intervals({{10,20,0},{0,15,0},{22,30,3}});
    test(p.kernel_sum==33&&p.union_busy==28&&p.overlap==5&&p.idle==2);
    throws([]{profile_intervals({{0,10,0},{9,20,1}});});
    throws([]{profile_intervals({{10,9,0}});});
    throws([]{profile_intervals({{0,1,1}});});
    test(profile_intervals({{0,1,0},{1,2,1}}).idle==0);
    std::cout<<"architecture CPU reference PASS checks="<<checks<<" (not GPU validation)\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
