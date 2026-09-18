// SPDX-License-Identifier: MIT
#include "maple_moe.hpp"
#include "dg2_validation.hpp"
#include <cmath>
#include <utility>
namespace maple_w2 {
class MoeSwiGlu; template<int G> class MoeSwiGluQuant;class MoeWeightedSum;
static void check(const MoeProblem&p,const MoeOptions&o,const MoeWorkspace&w){
    (void)nbytes(p.gate_shape);const Shape d{p.gate_shape.m,p.gate_shape.k,p.gate_shape.experts};(void)nbytes(d);
    if(!p.tokens||p.tokens>2048||!p.topk||p.topk>p.gate_shape.experts||p.topk>256||!p.gate||!p.up||!p.down||!p.x||!p.ids||!p.routes||!p.y)
        throw std::invalid_argument("invalid MoE problem");
    if(p.gate_shape.k>UINT32_MAX/4||p.gate_shape.m>UINT32_MAX/4)
        throw std::invalid_argument("MoE dimensions exceed 32-bit gather byte offsets");
    if(p.layout!=Layout::native_tq2&&p.layout!=Layout::s2tile8)throw std::invalid_argument("MoE supports native or s2tile8");
    if(!w.gate||!w.up||!w.hidden||!w.down||!w.gate_status||!w.down_status||!w.hidden_invalid||!w.output_invalid)
        throw std::invalid_argument("missing persistent MoE workspace");
    if(o.expert_grouping) {
        validate_grouping_size(size_t(p.tokens)*p.topk,p.gate_shape.experts);
        if(o.path!=MoePath::a8_gluquant||o.input_group!=32||o.hidden_group!=32||p.layout!=Layout::s2tile8)
            throw std::invalid_argument("v0.4 grouping requires A8 G32/H32 gluquant s2tile8");
        if(!w.grouping.counts||!w.grouping.offsets||!w.grouping.sorted_to_job||
           w.grouping.job_capacity<size_t(p.tokens)*p.topk||w.grouping.expert_capacity<p.gate_shape.experts)
            throw std::invalid_argument("missing/undersized expert-grouping workspace");
    }
    if(o.input_prequantized&&o.path==MoePath::a16)
        throw std::invalid_argument("prequantized input is an A8 option");
    for(auto item:std::array<std::pair<uint32_t,ExpertTokenTileView>,2>{{
            {o.gate_tokens_per_tile,w.gate_tiles},
            {o.down_tokens_per_tile,o.gate_tokens_per_tile==o.down_tokens_per_tile?w.gate_tiles:w.down_tiles}}}) {
        check_tokens_per_tile(item.first);
        if(item.first>1) {
            const auto& t=item.second;
            if(!o.expert_grouping||!t.first||!t.length||!t.expert||!t.total||
               t.capacity<expert_token_tile_capacity(size_t(p.tokens)*p.topk,p.gate_shape.experts,item.first))
                throw std::invalid_argument("missing/undersized grouped-reuse descriptors");
        }
    }
    if(o.gate_tokens_per_tile>1&&o.down_tokens_per_tile>1&&o.gate_tokens_per_tile!=o.down_tokens_per_tile) {
        if(w.gate_tiles.first==w.down_tiles.first||w.gate_tiles.length==w.down_tiles.length||
           w.gate_tiles.expert==w.down_tiles.expert||w.gate_tiles.total==w.down_tiles.total)
            throw std::invalid_argument("different token tiles require nonaliasing descriptor buffers");
    }
    if(!std::isfinite(o.clamp)||o.clamp<=0)throw std::invalid_argument("invalid SwiGLU clamp");
    if(o.path!=MoePath::a16&&o.path!=MoePath::a8_separate&&o.path!=MoePath::a8_gluquant)throw std::invalid_argument("invalid MoE path");
    // Validate BOTH projections before submitting gate/up. A bad down configuration
    // must not be discovered after its producer is already in flight.
    for (const auto &item : std::array<std::pair<uint32_t,Options>,2>{{
            {p.gate_shape.k,o.gate_kernel},{p.gate_shape.m,o.down_kernel}}}) {
        const uint32_t sp=item.second.split_k, local=item.second.local_size;
        if ((sp!=1&&sp!=2&&sp!=4&&sp!=8) || (item.first/256)%sp || !local || local>32)
            throw std::invalid_argument("invalid MoE split/local configuration");
    }
    if ((o.gate_kernel.split_k>1&&!w.gate_split)||(o.down_kernel.split_k>1&&!w.down_split))
        throw std::invalid_argument("missing MoE split scratch");
    if(o.path!=MoePath::a16){check_a8_group(o.input_group);check_a8_group(o.hidden_group);
        if(!w.input_a8.q||!w.input_a8.scales||!w.input_a8.invalid||!w.hidden_a8.q||!w.hidden_a8.scales||!w.hidden_a8.invalid)
            throw std::invalid_argument("A8 path needs persistent quantization planes");}
}
static sycl::event epilogue(sycl::queue&q,const MoeProblem&p,const MoeOptions&o,const MoeWorkspace&w,const sycl::event&dep){
    const size_t n=size_t(p.tokens)*p.topk*p.gate_shape.m;
    return q.submit([&](sycl::handler&h){h.depends_on(dep);h.parallel_for<MoeSwiGlu>(sycl::range<1>(n),[=](sycl::id<1>ii){
        size_t i=ii[0];float g=w.gate[i],u=w.up[i];int bad=!sycl::isfinite(g)||!sycl::isfinite(u);
        if(bad){g=0;u=0;}g=sycl::fmin(g,o.clamp);u=sycl::fmax(-o.clamp,sycl::fmin(u,o.clamp));
        float z=sycl::exp(-sycl::fabs(g));float si=g>=0?g/(1.f+z):(g*z)/(1.f+z);
        w.hidden[i]=si*u;w.hidden_invalid[i]=bad;});});
}
template<int G>
static sycl::event epilogue_quant(sycl::queue&q,const MoeProblem&p,const MoeOptions&o,const MoeWorkspace&w,const sycl::event&dep){
    const size_t ng=size_t(p.tokens)*p.topk*p.gate_shape.m/G;
    return q.submit([&](sycl::handler&h){h.depends_on(dep);
        h.parallel_for<MoeSwiGluQuant<G>>(sycl::nd_range<1>{sycl::range<1>(ng*32),sycl::range<1>(32)},[=](sycl::nd_item<1>it){
            const size_t group=it.get_group_linear_id();const unsigned lane=it.get_local_linear_id();float v[G/32],mx=0;int bad=0;
            for(unsigned j=0;j<G/32;++j){const size_t i=group*G+j*32+lane;float g=w.gate[i],u=w.up[i];
                const int b=!sycl::isfinite(g)||!sycl::isfinite(u);bad|=b;w.hidden_invalid[i]=b;if(b){g=0;u=0;}
                g=sycl::fmin(g,o.clamp);u=sycl::fmax(-o.clamp,sycl::fmin(u,o.clamp));float z=sycl::exp(-sycl::fabs(g));
                float si=g>=0?g/(1.f+z):(g*z)/(1.f+z);v[j]=si*u;w.hidden[i]=v[j];mx=sycl::fmax(mx,sycl::fabs(v[j]));}
            mx=sycl::reduce_over_group(it.get_group(),mx,sycl::maximum<float>());
            bad=sycl::reduce_over_group(it.get_group(),bad,sycl::maximum<int>());
            const float d=mx==0?1.f:sycl::fmax(mx/127.f,1.1754943508222875e-38f);
            if(lane==0){w.hidden_a8.scales[group]=d;w.hidden_a8.invalid[group]=bad;}
            for(unsigned j=0;j<G/32;++j){float z=sycl::fmax(-127.f,sycl::fmin(127.f,v[j]/d));float az=sycl::fabs(z);
                uint32_t k=uint32_t(az);float f=az-float(k);k+=uint32_t(f>.5f||(f==.5f&&(k&1u)));
                w.hidden_a8.q[group*G+j*32+lane]=int8_t(z<0?-int(k):int(k));}
        });});
}
MoeRun enqueue_moe(sycl::queue&q,const MoeProblem&p,const MoeOptions&requested,const MoeWorkspace&w,const std::vector<sycl::event>&deps){
    validate_dg2_device(q.get_device());
    const bool compatible=requested.path==MoePath::a8_gluquant && requested.input_group==32 &&
        requested.hidden_group==32 && p.layout==Layout::s2tile8;
    const auto decision=choose_moe_schedule(requested.schedule,requested.expert_grouping,
        p.tokens,p.topk,p.gate_shape.experts,compatible,requested.auto_policy);
    MoeOptions o=requested;o.expert_grouping=decision.grouped;
    check_tokens_per_tile(o.gate_tokens_per_tile);check_tokens_per_tile(o.down_tokens_per_tile);
    if(!o.expert_grouping)o.gate_tokens_per_tile=o.down_tokens_per_tile=1;
    check(p,o,w); // all views validated before first submission
    MoeRun run;run.dispatch=decision;run.fork_join_requested=o.expert_grouping&&o.overlap_grouping_quant&&!o.input_prequantized;
    run.queue_in_order=q.has_property<sycl::property::queue::in_order>();
    run.gate_tile=o.gate_tokens_per_tile;run.down_tile=o.down_tokens_per_tile;
    auto add=[&](const char*label,const sycl::event&e,uint32_t parents=0){
        const uint32_t bit=uint32_t(1)<<run.count;run.add(label,e,parents);return bit;};
    auto events=[&](uint32_t mask){std::vector<sycl::event> e;
        if(!mask)return deps;
        for(size_t i=0;i<run.count;++i)if((mask>>i)&1u)e.push_back(run.stages[i].event);
        return e;};
    DeviceProblem gate{p.gate_shape,p.tokens,p.topk,false,p.layout,p.gate,p.up,p.x,p.ids,w.gate,w.up,w.gate_status};
    DeviceProblem down{{p.gate_shape.m,p.gate_shape.k,p.gate_shape.experts},p.tokens,p.topk,true,p.layout,p.down,nullptr,w.hidden,p.ids,w.down,nullptr,w.down_status};
    uint32_t group_ready=0,gate_tiles_ready=0,down_tiles_ready=0;
    const int32_t* job_order=nullptr;ExpertTokenTileView gate_tiles{},down_tiles{};
    if(o.expert_grouping) {
        auto gr=enqueue_expert_grouping(q,p.ids,size_t(p.tokens)*p.topk,p.gate_shape.experts,w.grouping,deps);
        auto c=add("group_count",gr.count),pr=add("group_prefix",gr.prefix,c);
        group_ready=add("group_scatter",gr.scatter,pr);job_order=w.grouping.sorted_to_job;
        if(o.gate_tokens_per_tile>1) {
            gate_tiles=w.gate_tiles;gate_tiles.tokens_per_tile=o.gate_tokens_per_tile;
            // Serial after scatter unless fork/join was explicitly requested.
            auto e=enqueue_expert_token_tiles(q,size_t(p.tokens)*p.topk,p.gate_shape.experts,w.grouping,gate_tiles,{gr.scatter});
            gate_tiles_ready=add("group_gate_tiles",e,group_ready);
        }
        if(o.down_tokens_per_tile>1) {
            if(o.gate_tokens_per_tile==o.down_tokens_per_tile) {
                down_tiles=gate_tiles;down_tiles_ready=gate_tiles_ready;
            } else {
                down_tiles=w.down_tiles;down_tiles.tokens_per_tile=o.down_tokens_per_tile;
                const auto wait=group_ready|gate_tiles_ready;
                auto e=enqueue_expert_token_tiles(q,size_t(p.tokens)*p.topk,p.gate_shape.experts,w.grouping,down_tiles,events(wait));
                down_tiles_ready=add("group_down_tiles",e,wait);
            }
        }
    }
    const uint32_t all_group=group_ready|gate_tiles_ready|down_tiles_ready;
    uint32_t gdmask=0;
    if(o.path==MoePath::a16) {
        auto r=enqueue(q,gate,o.gate_kernel,w.gate_split,deps);gdmask=add("gate_up",r.main);
        if(r.has_reduce)gdmask=add("gate_reduce",r.done,gdmask);
    } else {
        uint32_t quant_ready=0;
        if(!o.input_prequantized) {
            const auto wait=o.overlap_grouping_quant?0u:all_group;
            auto e=enqueue_a8_quant(q,p.x,ActivationType::f32,p.tokens,p.gate_shape.k,o.input_group,w.input_a8,events(wait));
            quant_ready=add("input_quant",e,wait);
        }
        const uint32_t wait=group_ready|gate_tiles_ready|quant_ready;
        // A prequantized input belongs to caller deps. They are already inherited
        // through grouping when present, and passed directly for an empty mask.
        A8Options a{o.input_group,A8Mode::prequantized,o.gate_kernel,job_order,gate_tiles};
        auto r=enqueue_a8(q,gate,a,w.input_a8,w.gate_split,events(wait));gdmask=add("gate_up",r.main,wait);
        if(r.has_reduce)gdmask=add("gate_reduce",r.done,gdmask);
    }
    auto gd=run.done;if(o.diagnostic_host_waits)gd.wait_and_throw();
    sycl::event hd;uint32_t hdmask;
    if(o.path==MoePath::a8_gluquant) {
        switch(o.hidden_group){case 32:hd=epilogue_quant<32>(q,p,o,w,gd);break;
            case 128:hd=epilogue_quant<128>(q,p,o,w,gd);break;default:hd=epilogue_quant<256>(q,p,o,w,gd);}
        hdmask=add("swiglu_quant",hd,gdmask);
    } else {hd=epilogue(q,p,o,w,gd);hdmask=add("swiglu",hd,gdmask);}
    if(o.diagnostic_host_waits)hd.wait_and_throw();
    uint32_t ddmask;
    if(o.path==MoePath::a16) {
        auto r=enqueue(q,down,o.down_kernel,w.down_split,{hd});ddmask=add("down",r.main,hdmask);
        if(r.has_reduce)ddmask=add("down_reduce",r.done,ddmask);
    } else {
        auto mode=o.path==MoePath::a8_gluquant?A8Mode::prequantized:A8Mode::staged;
        const uint32_t wait=hdmask|down_tiles_ready;
        auto r=enqueue_a8(q,down,{o.hidden_group,mode,o.down_kernel,job_order,down_tiles},w.hidden_a8,w.down_split,events(wait));
        uint32_t main_wait=wait;if(r.has_quant)main_wait=add("hidden_quant",r.quant,wait);
        ddmask=add("down",r.main,main_wait);if(r.has_reduce)ddmask=add("down_reduce",r.done,ddmask);
    }
    auto dd=run.done;if(o.diagnostic_host_waits)dd.wait_and_throw();
    const size_t n=size_t(p.tokens)*p.gate_shape.k;const uint32_t width=p.gate_shape.k;
    auto last=q.submit([&](sycl::handler&h){h.depends_on(dd);h.parallel_for<MoeWeightedSum>(sycl::range<1>(n),[=](sycl::id<1>ii){
        const size_t i=ii[0],t=i/width,k=i%width;float sum=0;int bad=0;
        for(uint32_t e=0;e<p.topk;++e){size_t j=t*p.topk+e;float a=w.down[j*width+k],r=p.routes[j];
            if(!sycl::isfinite(a)||!sycl::isfinite(r)||r<0)bad=1;sum+=a*r;}
        p.y[i]=sum;w.output_invalid[i]=bad||!sycl::isfinite(sum);
    });});
    add("weighted_sum",last,ddmask);return run;
}
} // namespace maple_w2
