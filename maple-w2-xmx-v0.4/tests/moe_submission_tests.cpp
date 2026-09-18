// SPDX-License-Identifier: MIT
// Tests actual src/maple_moe.cpp and src/expert_grouping.cpp HOST enqueue logic
// against a non-executing SYCL test double. Not device compilation/execution.
#include "maple_moe.hpp"
#include <iostream>
#include <set>
#include <map>
using namespace maple_w2;
namespace maple_w2 {
static sycl::event stub(sycl::queue&q,const std::vector<sycl::event>&d) {return q.submit([&](sycl::handler&h){h.depends_on(d);});}
Run enqueue(sycl::queue&q,const DeviceProblem&,Options o,float*,const std::vector<sycl::event>&d){
    Run r;r.main=stub(q,d);r.done=r.main;if(o.split_k>1){r.has_reduce=true;r.done=stub(q,{r.main});}return r;}
A8Run enqueue_a8(sycl::queue&q,const DeviceProblem&,A8Options o,A8Workspace,float*,const std::vector<sycl::event>&d){
    A8Run r;auto deps=d;if(o.mode==A8Mode::staged){r.has_quant=true;r.quant=stub(q,d);deps={r.quant};}
    if(o.token_tiles.tokens_per_tile>1&&(!o.job_order||o.mode!=A8Mode::prequantized))throw std::runtime_error("bad reuse wiring");
    r.main=stub(q,deps);r.done=r.main;if(o.kernel.split_k>1){r.has_reduce=true;r.done=stub(q,{r.main});}return r;}
sycl::event enqueue_a8_quant(sycl::queue&q,const void*,ActivationType,uint32_t,uint32_t,uint32_t,A8Workspace,const std::vector<sycl::event>&d){return stub(q,d);}
} // namespace maple_w2
static int checks=0;
static void test(bool yes){++checks;if(!yes)throw std::runtime_error("enqueue test "+std::to_string(checks));}
static bool before(const sycl::queue&q,int src,int dst) {
    if(src==dst)return true;if(dst==0)return false;
    for(auto e:q.graph.at(size_t(dst-1)))if(before(q,src,e.number))return true;return false;
}
int main()try {
    uint8_t weights[8]{};float data[32]{};int32_t ints[32]{};int8_t codes[8]{};
    MoeProblem p{{2048,512,256},184,8,Layout::s2tile8,weights,weights,weights,data,data,ints,data};
    MoeWorkspace w{data,data,data,data,data,data,{codes,data,ints,false},{codes,data,ints,false},ints,ints,ints,ints,
        {ints,ints+1,ints+2,1472,256}, {ints+3,ints+4,ints+5,ints+6,1472,2},{ints+7,ints+8,ints+9,ints+10,1472,4}};
    for(bool ordered:{false,true})for(bool overlap:{false,true})for(bool pre:{false,true})for(uint32_t gt:{1u,2u,4u})for(uint32_t dt:{1u,2u,4u}) {
        sycl::queue q(ordered);MoeOptions o;o.schedule=MoeSchedule::grouped;o.overlap_grouping_quant=overlap;o.input_prequantized=pre;
        o.gate_tokens_per_tile=gt;o.down_tokens_per_tile=dt;o.gate_kernel.split_k=4;o.down_kernel.split_k=2;
        auto root=q.submit([](sycl::handler&){});sycl::mock_waits=0;
        auto r=enqueue_moe(q,p,o,w,{root});test(sycl::mock_waits==0);test(r.dispatch.grouped);test(r.queue_in_order==ordered);
        std::map<std::string,int> ev;
        for(size_t i=0;i<r.count;++i){ev[r.stages[i].label]=r.stages[i].event.number;
            test(before(q,root.number,r.stages[i].event.number));test(before(q,r.stages[i].event.number,r.done.number));
            for(size_t j=0;j<i;++j)if((r.stages[i].parents>>j)&1u)test(before(q,r.stages[j].event.number,r.stages[i].event.number));}
        test(before(q,ev["group_scatter"],ev["gate_up"]));
        if(!pre) {
            test(before(q,ev["input_quant"],ev["gate_up"]));
            test(before(q,ev["group_scatter"],ev["input_quant"])==(!overlap||ordered));
        }else test(ev.count("input_quant")==0);
        if(gt>1)test(before(q,ev["group_gate_tiles"],ev["gate_up"]));
        if(dt>1)test(before(q,ev[gt==dt?"group_gate_tiles":"group_down_tiles"],ev["down"]));
    }
    {sycl::queue q;auto bad=w;bad.gate_tiles.first=nullptr;MoeOptions o;o.schedule=MoeSchedule::grouped;o.gate_tokens_per_tile=2;
        bool threw=false;try{enqueue_moe(q,p,o,bad);}catch(const std::exception&){threw=true;}test(threw&&q.graph.empty());}
    {sycl::queue q;auto bad=w;bad.down_tiles=bad.gate_tiles;MoeOptions o;o.schedule=MoeSchedule::grouped;o.gate_tokens_per_tile=2;o.down_tokens_per_tile=4;
        bool threw=false;try{enqueue_moe(q,p,o,bad);}catch(const std::exception&){threw=true;}test(threw&&q.graph.empty());}
    {sycl::queue q;MoeOptions o;o.schedule=MoeSchedule::auto_select;o.gate_tokens_per_tile=4;o.down_tokens_per_tile=4;
        auto r=enqueue_moe(q,p,o,w);test(!r.dispatch.grouped&&r.gate_tile==1&&r.down_tile==1);}
    {sycl::queue q;MoeOptions o;o.schedule=MoeSchedule::auto_select;o.auto_policy={32,2};
        auto r=enqueue_moe(q,p,o,w);test(r.dispatch.grouped);}
    {sycl::queue q;MoeOptions o;o.schedule=MoeSchedule::auto_select;o.auto_policy={1,0};auto one=p;one.tokens=1;
        auto r=enqueue_moe(q,one,o,w);test(!r.dispatch.grouped);}
    {sycl::queue q;MoeOptions o;o.diagnostic_host_waits=true;sycl::mock_waits=0;enqueue_moe(q,p,o,w);test(sycl::mock_waits==3);}
    std::cout<<"MOCK HOST submission/DAG tests PASS checks="<<checks<<"; kernel lambdas NOT executed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
