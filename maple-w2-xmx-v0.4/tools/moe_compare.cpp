// SPDX-License-Identifier: MIT
#include "maple_moe.hpp"
#include "moe_reference.hpp"
#include <filesystem>
#include <sstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>

using Clock=std::chrono::steady_clock;
using namespace maple_w2;
static double micros(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::micro>(b-a).count();
}
static double percentile(std::vector<double> v,double p) {
    std::sort(v.begin(),v.end()); return v[std::min(v.size()-1,size_t(p*(v.size()-1)))];
}
template<class T> struct DeviceMem {
    sycl::queue *q; T *p=nullptr; size_t count;
    DeviceMem(sycl::queue& queue,size_t n):q(&queue),count(n) {
        if(n) p=sycl::malloc_device<T>(n,queue);
        if(n && !p) throw std::bad_alloc();
    }
    ~DeviceMem(){if(p){try {q->wait(); sycl::free(p,*q);} catch(...) { /* process-level diagnostic harness teardown */ }}}
    DeviceMem(const DeviceMem&)=delete;
};
template<class T> static std::vector<T> read_raw(const std::string& path,size_t n) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(n*sizeof(T))) throw std::runtime_error("raw file length mismatch: "+path);
    std::vector<T> r(n);f.seekg(0);f.read(reinterpret_cast<char*>(r.data()),r.size()*sizeof(T));
    if(!f) throw std::runtime_error("raw file read failed: "+path); return r;
}
static std::vector<uint8_t> synth(Shape s,uint32_t seed) {
    std::mt19937 gen(seed); std::uniform_int_distribution<int> code(0,2);
    std::vector<uint8_t> w(nbytes(s),0);
    for(size_t i=0;i<w.size();i+=66) {
        for(unsigned j=0;j<64;++j) {
            unsigned v=0; for(int sh=0;sh<8;sh+=2) v|=unsigned(code(gen))<<sh;
            w[i+j]=uint8_t(v);
        }
        // Per-block scales deliberately differ: no row-scale assumption.
        float d=(1+int(gen()%31))/512.f;
        put16(w.data()+i+64,float_to_half(d));
    }
    return w;
}
static sycl::device choose_device(const std::string& match) {
    for(const auto& plat:sycl::platform::get_platforms()) for(const auto& d:plat.get_devices(sycl::info::device_type::gpu)) {
        if(d.get_backend()==sycl::backend::ext_oneapi_level_zero &&
           d.get_info<sycl::info::device::vendor_id>()==0x8086 &&
           d.get_info<sycl::info::device::name>().find(match)!=std::string::npos) return d;
    }
    throw std::runtime_error("no Intel Level Zero GPU matching '"+match+"'; run sycl-ls and check driver");
}
static void print_error(const char* label,const Error& e) {
    std::cout<<label<<" finite="<<e.finite<<" nmse="<<e.nmse
      <<" max_abs_over_rms="<<e.max_abs_over_rms<<" max_abs="<<e.max_abs<<"\n";
}
struct ChainSample {double wall=0,enqueue_return=0,tail_wait=0,gpu_span=0,kernel_sum=0,gpu_gap=0;};
struct ChainVariant {std::string name;MoeOptions opt;std::vector<ChainSample> samples;bool pass=true;double total_nmse=0;};
static bool good(const Error&e){return e.finite&&e.nmse<=1e-6&&e.max_abs_over_rms<=.005;}
static double value(const ChainVariant&v,double ChainSample::*field,double p=.5){std::vector<double>a;for(auto&s:v.samples)a.push_back(s.*field);return percentile(a,p);}
template<class T>static std::vector<T> copy_back(sycl::queue&q,const T*p,size_t n){std::vector<T>v(n);q.memcpy(v.data(),p,n*sizeof(T)).wait_and_throw();return v;}
static QuantizedHost capture_quant(sycl::queue&q,const A8Workspace&w,uint32_t rows,uint32_t k,uint32_t group){
    return {rows,k,group,copy_back(q,w.q,size_t(rows)*k),copy_back(q,w.scales,size_t(rows)*k/group)};}
static bool quant_ok(const QuantizedHost&got,const std::vector<float>&x){
    const auto ref=quantize_a8(x,got.rows,got.k,got.group);bool ok=true;
    for(size_t g=0;g<got.scale.size();++g){float d=got.scale[g],r=ref.scale[g];
        if(!std::isfinite(d)||d<=0||std::abs(d-r)>std::max(1e-38f,5e-6f*std::abs(r)))ok=false;}
    for(size_t i=0;i<got.q.size();++i){int delta=int(got.q[i])-int(ref.q[i]);if(got.q[i]==-128)ok=false;
        if(delta){float a=std::abs(x[i]/ref.scale[i/ref.group]),f=a-std::floor(a);
            if(std::abs(delta)!=1||std::abs(f-.5f)>2e-4f)ok=false;}}
    return ok;
}
static bool all_zero(const std::vector<int32_t>&v){return std::all_of(v.begin(),v.end(),[](int32_t x){return x==0;});}
int main(int argc,char**argv)try{
    std::map<std::string,std::string>a;
    for(int i=1;i<argc;++i){std::string k=argv[i];if(k=="--help"){
        std::cout<<"Maple MoE-chain comparator v0.3 (not a server/backend)\n"
          <<"--gate gate.mw2 --up up.mw2 --down down.mw2 OR --k 2048 --hidden 512 --experts 256\n"
          <<"--tokens 1..8 --topk 8 --layout native|s2tile8 --groups 32,128,256 [--down-group 128]\n"
          <<"--host-waits async|waited|both --gate-split 1|2|4|8 --down-split 1|2 --local 1..32\n"
          <<"--x-file x.f32 --ids-file ids.i32 --routes-file routes.f32 (already normalized top-k weights)\n"
          <<"--pattern normal|zeros|outliers --repeats 28 --warmup 3 --device A750 --out dir\n"
          <<"--skip-probe 0|1 --in-order 0|1 --clamp 7 --id-mode rotate|fixed --scrub-mib 0..128\n"
          <<"A8-separate: input_quant -> gate/up -> GLU -> hidden_quant -> down -> weighted_sum\n"
          <<"A8-gluquant: input_quant -> gate/up -> GLU+hidden_quant -> down -> weighted_sum\n"
          <<"waited is a SYCL-only host-boundary diagnostic, NOT Vulkan interop timing.\n";return 0;}
        if(k.rfind("--",0)!=0||i+1==argc||a.count(k))throw std::invalid_argument("expected unique --key value");a[k]=argv[++i];}
    const std::set<std::string>allowed={"--gate","--up","--down","--k","--hidden","--experts","--tokens","--topk","--layout","--groups","--down-group","--host-waits","--gate-split","--down-split","--local","--x-file","--ids-file","--routes-file","--pattern","--repeats","--warmup","--device","--out","--skip-probe","--in-order","--clamp","--id-mode","--scrub-mib"};
    for(auto&kv:a)if(!allowed.count(kv.first))throw std::invalid_argument("unknown argument "+kv.first);
    auto get=[&](const char*k,const char*d){return a.count(k)?a.at(k):std::string(d);};
    auto parse=[](const std::string&s){size_t e=0;auto v=std::stoull(s,&e);if(s.empty()||s[0]=='-'||e!=s.size()||v>UINT32_MAX)throw std::invalid_argument("bad unsigned integer");return uint32_t(v);};
    auto num=[&](const char*k,const char*d){return parse(get(k,d));};
    auto bit=[&](const char*k,const char*d){auto v=num(k,d);if(v>1)throw std::invalid_argument("boolean must be 0 or 1");return v!=0;};
    uint32_t tokens=num("--tokens","1"),topk=num("--topk","8"),repeats=num("--repeats","28"),warmup=num("--warmup","3");
    const uint32_t scrub_mib=num("--scrub-mib","0");
    if(!tokens||tokens>8||!topk||topk>256||!repeats||repeats>10000||warmup>1000||scrub_mib>128)throw std::invalid_argument("invalid benchmark limits");
    const auto layout_name=get("--layout","s2tile8"),waits=get("--host-waits","both"),pattern=get("--pattern","normal"),idmode=get("--id-mode","rotate");
    if(layout_name!="native"&&layout_name!="s2tile8")throw std::invalid_argument("invalid layout");
    if(waits!="async"&&waits!="waited"&&waits!="both")throw std::invalid_argument("invalid host-waits");
    if(pattern!="normal"&&pattern!="zeros"&&pattern!="outliers")throw std::invalid_argument("invalid pattern");
    if(idmode!="rotate"&&idmode!="fixed")throw std::invalid_argument("invalid id-mode");
    const bool real=a.count("--gate");if(real!=bool(a.count("--up"))||real!=bool(a.count("--down")))throw std::invalid_argument("gate,up,down must be provided together");
    Shape s{num("--k","2048"),num("--hidden","512"),num("--experts","256")};std::vector<uint8_t>wg,wu,wd;
    if(real){auto c=load_capsule(a.at("--gate"));s=c.shape;wg=std::move(c.bytes);
        auto u=load_capsule(a.at("--up")),d=load_capsule(a.at("--down"));
        if(u.shape.k!=s.k||u.shape.m!=s.m||u.shape.experts!=s.experts||d.shape.k!=s.m||d.shape.m!=s.k||d.shape.experts!=s.experts)
            throw std::invalid_argument("expected matching gate/up and transposed down dimensions");wu=std::move(u.bytes);wd=std::move(d.bytes);}
    else{wg=synth(s,1234);wu=synth(s,1235);wd=synth({s.m,s.k,s.experts},1236);}
    Shape ds{s.m,s.k,s.experts};(void)nbytes(ds);if(topk>s.experts)throw std::invalid_argument("topk exceeds experts");
    const size_t xn=size_t(tokens)*s.k,jobs=size_t(tokens)*topk,hn=jobs*s.m,dn=jobs*s.k;
    Options go{num("--gate-split","1"),num("--local","4")},dopt{num("--down-split","1"),go.local_size};
    for(auto entry:std::vector<std::pair<uint32_t,Options>>{{s.k,go},{s.m,dopt}}){auto sp=entry.second.split_k;
        if((sp!=1&&sp!=2&&sp!=4&&sp!=8)||(entry.first/256)%sp||!go.local_size||go.local_size>32)throw std::invalid_argument("invalid split/local");}
    const auto clamp_text=get("--clamp","7");size_t clamp_end=0;
    float clamp=std::stof(clamp_text,&clamp_end);
    if(clamp_end!=clamp_text.size()||!std::isfinite(clamp)||clamp<=0)throw std::invalid_argument("bad clamp");
    std::vector<uint32_t>groups;std::istringstream gf(get("--groups","32,128,256"));std::string g;
    while(std::getline(gf,g,',')){uint32_t n=parse(g);check_a8_group(n);if(std::find(groups.begin(),groups.end(),n)!=groups.end())throw std::invalid_argument("duplicate group");groups.push_back(n);}
    if(groups.empty())throw std::invalid_argument("no groups");
    uint32_t down_override=a.count("--down-group")?num("--down-group","32"):0;if(down_override)check_a8_group(down_override);
    std::vector<ChainVariant>vars;std::vector<bool>wm=waits=="both"?std::vector<bool>{false,true}:std::vector<bool>{waits=="waited"};
    for(bool hw:wm){MoeOptions o;o.path=MoePath::a16;o.gate_kernel=go;o.down_kernel=dopt;o.clamp=clamp;o.diagnostic_host_waits=hw;
        std::string suffix=hw?"-waited":"-async";vars.push_back({"w2a16"+suffix,o});
        for(uint32_t group:groups)for(MoePath path:{MoePath::a8_separate,MoePath::a8_gluquant}){o.path=path;o.input_group=group;o.hidden_group=down_override?down_override:group;
            vars.push_back({"w2a8-g"+std::to_string(group)+"-h"+std::to_string(o.hidden_group)+(path==MoePath::a8_separate?"-separate":"-gluquant")+suffix,o});}}
    std::mt19937 rng(98765);std::normal_distribution<float>normal(0,.5f);std::vector<float>x(xn),routes(jobs);
    if(a.count("--x-file"))x=read_raw<float>(a.at("--x-file"),xn);else for(size_t i=0;i<xn;++i)x[i]=pattern=="zeros"?0.f:normal(rng);
    if(!a.count("--x-file")&&pattern=="outliers")for(size_t i=0;i<xn;i+=257)x[i]=16.f;
    for(float v:x)if(!std::isfinite(v)||std::abs(v)>65504.f)throw std::invalid_argument("input must be finite within FP16 range");
    if(a.count("--routes-file"))routes=read_raw<float>(a.at("--routes-file"),jobs);
    else for(uint32_t t=0;t<tokens;++t){float sum=0;for(uint32_t j=0;j<topk;++j){float v=float(j+1);routes[size_t(t)*topk+j]=v;sum+=v;}
        for(uint32_t j=0;j<topk;++j)routes[size_t(t)*topk+j]/=sum;}
    validate_routes(routes,tokens,topk);
    uint32_t rounds=1+warmup+repeats;std::vector<int32_t>ids(size_t(rounds)*jobs),fixed;
    if(a.count("--ids-file"))fixed=read_raw<int32_t>(a.at("--ids-file"),jobs);
    for(uint32_t r=0;r<rounds;++r){if(!fixed.empty())std::copy(fixed.begin(),fixed.end(),ids.begin()+size_t(r)*jobs);
        else if(idmode=="fixed"&&r)std::copy(ids.begin(),ids.begin()+jobs,ids.begin()+size_t(r)*jobs);
        else for(uint32_t t=0;t<tokens;++t){std::vector<int32_t>pool(s.experts);std::iota(pool.begin(),pool.end(),0);
            if(r==0)std::rotate(pool.begin(),pool.end()-1,pool.end());else std::shuffle(pool.begin(),pool.end(),rng);
            std::copy(pool.begin(),pool.begin()+topk,ids.begin()+size_t(r)*jobs+t*topk);}}
    for(auto id:ids)if(id<0||uint32_t(id)>=s.experts)throw std::invalid_argument("invalid expert id");
    auto d=choose_device(get("--device","A750"));bool in_order=bit("--in-order","1");
    sycl::property_list props=in_order?sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}:sycl::property_list{sycl::property::queue::enable_profiling{}};
    sycl::queue q(d,[](sycl::exception_list es){for(auto e:es)std::rethrow_exception(e);},props);
    std::cout<<std::setprecision(10)<<"device="<<d.get_info<sycl::info::device::name>()<<"\ndriver="<<d.get_info<sycl::info::device::driver_version>()<<'\n';
    if(!bit("--skip-probe","0"))run_s2s8_probe(q);
    const std::filesystem::path out=get("--out","build/results/moe-manual");if(std::filesystem::exists(out/"manifest.txt"))throw std::runtime_error("evidence exists; use a new output path");std::filesystem::create_directories(out);
    std::ofstream manifest(out/"manifest.txt"),numerical(out/"numerical.csv"),samples(out/"samples.csv"),stages(out/"stages.csv");
    if(!manifest||!numerical||!samples||!stages)throw std::runtime_error("cannot create output files");
    manifest<<"version=0.3\ndevice="<<d.get_info<sycl::info::device::name>()<<"\ndriver="<<d.get_info<sycl::info::device::driver_version>()<<"\n";
    for(auto&kv:a)manifest<<kv.first<<'='<<kv.second<<'\n';
    manifest<<"weight_source="<<(real?"real_capsules":"synthetic")<<"\nactivation_source="<<(a.count("--x-file")?"captured":"synthetic")
      <<"\nroute_source="<<(a.count("--routes-file")?"captured":"synthetic_normalized")<<"\nids_source="<<(a.count("--ids-file")?"captured_fixed":idmode)<<"\n";
    manifest<<"scope=one_MoE_layer_no_attention_router_norm_residual_sampling_or_Vulkan_interop\nwaited=SYCL_only_three_host_waits_NOT_Vulkan_measurement\n";
    manifest<<"k="<<s.k<<"\nhidden="<<s.m<<"\nexperts="<<s.experts<<"\nQ="<<tokens<<"\nlayout="<<layout_name<<"\nin_order="<<in_order<<"\n";
    Layout ly=layout_name=="native"?Layout::native_tq2:Layout::s2tile8;
    std::vector<uint8_t>sg,su,sd;if(ly==Layout::s2tile8){sg=repack_s2tile8(wg,s);su=repack_s2tile8(wu,s);sd=repack_s2tile8(wd,ds);}
    DeviceMem<uint8_t>dg(q,wg.size()),du(q,wu.size()),dd(q,wd.size()),scrub(q,size_t(scrub_mib)*1024*1024);
    DeviceMem<float>dx(q,xn),dr(q,jobs),dy(q,xn),yg(q,hn),yu(q,hn),hidden(q,hn),yd(q,dn),gs(q,go.split_k>1?size_t(go.split_k)*hn*2:0),dscratch(q,dopt.split_k>1?size_t(dopt.split_k)*dn:0);
    DeviceMem<int32_t>di(q,ids.size()),gst(q,jobs),dst(q,jobs),hi(q,hn),oi(q,xn),ai0(q,xn/32),ai1(q,hn/32);
    DeviceMem<int8_t>aq0(q,xn),aq1(q,hn);DeviceMem<float>as0(q,xn/32),as1(q,hn/32);
    MoeProblem p{s,tokens,topk,ly,dg.p,du.p,dd.p,dx.p,dr.p,di.p,dy.p};
    MoeWorkspace w{yg.p,yu.p,hidden.p,yd.p,gs.p,dscratch.p,{aq0.p,as0.p,ai0.p,false},{aq1.p,as1.p,ai1.p,false},gst.p,dst.p,hi.p,oi.p};
    q.memcpy(dg.p,ly==Layout::s2tile8?sg.data():wg.data(),wg.size());q.memcpy(du.p,ly==Layout::s2tile8?su.data():wu.data(),wu.size());q.memcpy(dd.p,ly==Layout::s2tile8?sd.data():wd.data(),wd.size());
    q.memcpy(dx.p,x.data(),xn*4);q.memcpy(dr.p,routes.data(),jobs*4);q.memcpy(di.p,ids.data(),ids.size()*4);
    q.memset(gst.p,0,jobs*4);q.memset(dst.p,0,jobs*4);q.wait_and_throw();
    manifest<<"weight_resident_bytes="<<wg.size()+wu.size()+wd.size()<<"\nexplicit_workspace_bytes="<<(xn*2+jobs+hn*3+dn+gs.count+dscratch.count+as0.count+as1.count)*4+(ids.size()+jobs*2+hn+xn+ai0.count+ai1.count)*4+xn+hn+scrub.count<<"\n";
    std::cout<<"MoE chain Q="<<tokens<<" K="<<s.k<<" H="<<s.m<<" variants="<<vars.size()<<"; one shared queue, no Vulkan handoff\n";
    // Stage-local correctness uses the actual GPU quantizer outputs and captured intermediate
    // activations. End-to-end F32 error is separate and is not a model-quality certificate.
    numerical<<std::setprecision(12)<<"snapshot,variant,gate_nmse,up_nmse,swiglu_nmse,down_nmse,weighted_sum_nmse,chain_total_nmse_vs_f32,quant_pass,status_pass,kernel_pass\n";
    for(uint32_t snapshot:{0u,rounds-1}){
        std::vector<int32_t>id(ids.begin()+size_t(snapshot)*jobs,ids.begin()+size_t(snapshot+1)*jobs);
        auto rg32=reference(wg,s,x,id,tokens,topk,false,false),ru32=reference(wu,s,x,id,tokens,topk,false,false);
        auto rh32=swiglu_reference(rg32,ru32,clamp);auto rd32=reference(wd,ds,rh32,id,tokens,topk,true,false);
        auto ry32=weighted_sum_reference(rd32,routes,tokens,topk,s.k);
        auto rg16=reference(wg,s,x,id,tokens,topk,false,true),ru16=reference(wu,s,x,id,tokens,topk,false,true);
        for(auto&v:vars){p.ids=di.p+size_t(snapshot)*jobs;auto run=enqueue_moe(q,p,v.opt,w);run.done.wait_and_throw();
            auto cg=copy_back(q,yg.p,hn),cu=copy_back(q,yu.p,hn),ch=copy_back(q,hidden.p,hn),cd=copy_back(q,yd.p,dn),cy=copy_back(q,dy.p,xn);
            bool status=all_zero(copy_back(q,gst.p,jobs))&&all_zero(copy_back(q,dst.p,jobs))&&all_zero(copy_back(q,hi.p,hn))&&all_zero(copy_back(q,oi.p,xn));
            bool qp=true;std::vector<float>goldg,goldu,goldd;
            if(v.opt.path==MoePath::a16){goldg=rg16;goldu=ru16;goldd=reference(wd,ds,ch,id,tokens,topk,true,true);}
            else{auto qi=capture_quant(q,w.input_a8,tokens,s.k,v.opt.input_group),qh=capture_quant(q,w.hidden_a8,uint32_t(jobs),s.m,v.opt.hidden_group);
                qp=quant_ok(qi,x)&&quant_ok(qh,ch);status=status&&all_zero(copy_back(q,ai0.p,xn/v.opt.input_group))&&all_zero(copy_back(q,ai1.p,hn/v.opt.hidden_group));
                goldg=reference_a8(wg,s,qi,id,tokens,topk,false);goldu=reference_a8(wu,s,qi,id,tokens,topk,false);goldd=reference_a8(wd,ds,qh,id,tokens,topk,true);}
            auto goldh=swiglu_reference(cg,cu,clamp),goldy=weighted_sum_reference(cd,routes,tokens,topk,s.k);
            auto eg=errors(cg,goldg),eu=errors(cu,goldu),eh=errors(ch,goldh),ed=errors(cd,goldd),ey=errors(cy,goldy),et=errors(cy,ry32);
            bool ok=good(eg)&&good(eu)&&good(eh)&&good(ed)&&good(ey)&&qp&&status&&et.finite;
            v.pass&=ok;v.total_nmse=std::max(v.total_nmse,et.nmse);
            numerical<<snapshot<<','<<v.name<<','<<eg.nmse<<','<<eu.nmse<<','<<eh.nmse<<','<<ed.nmse<<','<<ey.nmse<<','<<et.nmse<<','<<qp<<','<<status<<','<<ok<<'\n';
            if(!ok){numerical.flush();throw std::runtime_error("chain correctness failed: "+v.name+"; timing aborted, inspect numerical.csv");}
        }
    }
    samples<<std::setprecision(12)<<"round,phase,order,variant,enqueue_return_us,tail_wait_us,wall_us,gpu_span_us,kernel_sum_us,gpu_gap_us\n";
    stages<<std::setprecision(12)<<"round,phase,variant,stage,start_ns,end_ns,duration_us\n";
    for(uint32_t r=0;r<rounds;++r){const char*phase=r==0?"first_timed_post_validation":r<=warmup?"warmup":"repeat";
        for(size_t j=0;j<vars.size();++j){size_t vi=(j+r)%vars.size();if(r%2)vi=vars.size()-1-vi;auto&v=vars[vi];p.ids=di.p+size_t(r)*jobs;
            if(scrub.p){const size_t n=scrub.count/4;auto*b=reinterpret_cast<uint32_t*>(scrub.p);q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>i){b[i[0]]=uint32_t(i[0])+r;}).wait_and_throw();}
            auto t0=Clock::now();auto run=enqueue_moe(q,p,v.opt,w);auto ts=Clock::now();run.done.wait_and_throw();auto t1=Clock::now();
            ChainSample sm;sm.enqueue_return=micros(t0,ts);sm.tail_wait=micros(ts,t1);sm.wall=micros(t0,t1);
            uint64_t begin=0,end=0;for(size_t k=0;k<run.count;++k){auto&e=run.stages[k];auto a0=e.event.get_profiling_info<sycl::info::event_profiling::command_start>();auto a1=e.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                if(k==0)begin=a0;end=a1;double dur=double(a1-a0)/1000.;sm.kernel_sum+=dur;stages<<r<<','<<phase<<','<<v.name<<','<<e.label<<','<<a0<<','<<a1<<','<<dur<<'\n';}
            sm.gpu_span=double(end-begin)/1000.;sm.gpu_gap=sm.gpu_span-sm.kernel_sum;
            samples<<r<<','<<phase<<','<<j<<','<<v.name<<','<<sm.enqueue_return<<','<<sm.tail_wait<<','<<sm.wall<<','<<sm.gpu_span<<','<<sm.kernel_sum<<','<<sm.gpu_gap<<'\n';
            if(r>warmup)v.samples.push_back(sm);
        }
    }
    std::ofstream summary(out/"summary.csv");if(!summary)throw std::runtime_error("cannot create summary");
    summary<<std::setprecision(12)<<"variant,layout,input_group,hidden_group,host_waits,gate_split,down_split,local_size,kernel_sum_median_us,gpu_span_median_us,gpu_gap_median_us,enqueue_return_median_us,tail_wait_median_us,wall_median_us,wall_p95_us,speedup_vs_a16_same_waits,kernel_pass,chain_total_nmse_vs_f32\n";
    for(auto&v:vars){auto base=std::find_if(vars.begin(),vars.end(),[&](const ChainVariant&b){return b.opt.path==MoePath::a16&&b.opt.diagnostic_host_waits==v.opt.diagnostic_host_waits;});
        double speed=value(*base,&ChainSample::wall)/value(v,&ChainSample::wall);bool a8=v.opt.path!=MoePath::a16;
        summary<<v.name<<','<<layout_name<<','<<(a8?v.opt.input_group:0)<<','<<(a8?v.opt.hidden_group:0)<<','<<v.opt.diagnostic_host_waits<<','<<go.split_k<<','<<dopt.split_k<<','<<go.local_size<<','
          <<value(v,&ChainSample::kernel_sum)<<','<<value(v,&ChainSample::gpu_span)<<','<<value(v,&ChainSample::gpu_gap)<<','<<value(v,&ChainSample::enqueue_return)<<','<<value(v,&ChainSample::tail_wait)<<','<<value(v,&ChainSample::wall)<<','<<value(v,&ChainSample::wall,.95)<<','<<speed<<','<<v.pass<<','<<v.total_nmse<<'\n';
        std::cout<<v.name<<" wall_median_us="<<value(v,&ChainSample::wall)<<" p95="<<value(v,&ChainSample::wall,.95)<<" kernel_sum_us="<<value(v,&ChainSample::kernel_sum)<<" total_nmse="<<v.total_nmse<<" PASS\n";
    }
    std::cout<<"RESULT=PASS output="<<out.string()<<"\nNot server TG or a model-quality acceptance. One layer; no Vulkan/oneDNN boundary.\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
