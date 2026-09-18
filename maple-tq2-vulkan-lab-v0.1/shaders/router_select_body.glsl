// Included by router_select.comp and router_fused.comp after writing logits_shared.
// All invocations execute all barriers. No global atomic floating-point reduction.
shared float prob[256];
shared float red[256];
shared uint winners[256];
shared uint chosen[8];
shared float denom;
void select_topk(uint token,uint lid) {
    float v=logits_shared[lid];
    if(lid<p.experts && (isnan(v)||isinf(v))) {atomicOr(ctl.errors,1u);v=0.0;}
    red[lid]=lid<p.experts?v:-3.402823466e38;
    barrier();
    for(uint s=128u;s>0u;s>>=1u){if(lid<s)red[lid]=max(red[lid],red[lid+s]);barrier();}
    float pv=lid<p.experts?exp(v-red[0]):0.0;
    barrier();red[lid]=pv;barrier();
    for(uint s=128u;s>0u;s>>=1u){if(lid<s)red[lid]+=red[lid+s];barrier();}
    prob[lid]=pv/red[0];
    barrier();
    // Choose on the rounded FP32 probabilities, not on raw logits.
    float candidate=lid<p.experts?prob[lid]:-1.0;
    for(uint pick=0u;pick<p.topk;++pick){
        red[lid]=candidate;winners[lid]=lid;barrier();
        for(uint s=128u;s>0u;s>>=1u){
            if(lid<s){
                float a=red[lid],b=red[lid+s];uint ia=winners[lid],ib=winners[lid+s];
                bool tie=p.tie_high!=0u?ib>ia:ib<ia;
                if(b>a||(b==a&&tie)){red[lid]=b;winners[lid]=ib;}
            }
            barrier();
        }
        uint winner=winners[0];
        if(lid==0u)chosen[p.ascending!=0u?p.topk-1u-pick:pick]=winner;
        if(lid==winner)candidate=-1.0;
        barrier();
    }
    if(lid==0u){precise float s=0.0;for(uint i=0u;i<p.topk;++i)s+=prob[chosen[i]];denom=s+1e-20;}
    barrier();
    if(lid<p.topk){indices[token*p.topk+lid]=chosen[lid];scores[token*p.topk+lid]=prob[chosen[lid]]/denom;}
}
