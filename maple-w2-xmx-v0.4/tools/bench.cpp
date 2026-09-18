// SPDX-License-Identifier: MIT
#include "maple_w2a16.hpp"
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
int main(int argc,char**argv) try {
    std::map<std::string,std::string> a;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {
            std::cout<<"maple-w2-bench --weights TENSOR.mw2 [--weights2 UP.mw2]\n"
             "  --input f32|f16 --mode xmx|fma --layout native|tile8 --split 1|2|4|8 --local 1..32\n"
             "  --tokens 1..8 --topk 8 --per-selection 0|1 --repeats 30 --warmup 3\n"
             "  --device A750 --id-mode fixed|rotate --csv run.csv\n"
             "  --x-file captured.f32 --ids-file captured.i32 (little-endian contiguous)\n"
             "Without --weights: synthetic --k 2048 --m 512 --experts 256 --seed 1234.\n"
             "FMA comparator is this package's SYCL kernel, NOT the llama Vulkan baseline.\n";return 0;
        }
        if(key.rfind("--",0)!=0 || i+1>=argc) throw std::invalid_argument("expected --key value");
        a[key]=argv[++i];
    }
    const std::vector<std::string> known={"--weights","--weights2","--synthetic-pair","--input","--mode","--layout","--split","--local","--tokens","--topk","--per-selection","--repeats","--warmup","--device","--id-mode","--csv","--x-file","--ids-file","--k","--m","--experts","--seed"};
    for(const auto& kv:a) if(std::find(known.begin(),known.end(),kv.first)==known.end()) throw std::invalid_argument("unknown argument "+kv.first);
    auto get=[&](const char*k,const char*def){auto it=a.find(k);return it==a.end()?std::string(def):it->second;};
    auto num=[&](const char*k,const char*def){auto s=get(k,def);size_t end=0;unsigned long long v=std::stoull(s,&end);if(end!=s.size()||v>UINT32_MAX||s[0]=='-')throw std::invalid_argument("invalid integer "+std::string(k));return uint32_t(v);};
    Shape s{num("--k","2048"),num("--m","512"),num("--experts","256")};
    const auto input=get("--input","f32");
    if(input!="f32"&&input!="f16")throw std::invalid_argument("bad --input");
    const auto mode=get("--mode","xmx"), layout=get("--layout","native"),idmode=get("--id-mode","rotate");
    if((mode!="xmx"&&mode!="fma")||(layout!="native"&&layout!="tile8")||(idmode!="fixed"&&idmode!="rotate"))throw std::invalid_argument("bad mode/layout/id-mode");
    if(mode=="fma"&&layout!="native")throw std::invalid_argument("FMA comparator only supports native TQ2");
    uint32_t tokens=num("--tokens","1"),topk=num("--topk","8"),repeats=num("--repeats","30"),warmup=num("--warmup","3"),seed=num("--seed","1234");
    const uint32_t ps=num("--per-selection","0");
    if(ps>1 || !tokens || tokens>8 || !topk || topk>256 || !repeats || repeats>10000 || warmup>1000)throw std::invalid_argument("pilot limits: tokens 1..8, topk 1..256, repeats 1..10000, per-selection 0|1");
    const bool per=ps!=0;
    std::vector<uint8_t> native0,native1;
    if(a.count("--weights")){auto c=load_capsule(a.at("--weights"));s=c.shape;native0=std::move(c.bytes);}
    else native0=synth(s,seed);
    if(topk>s.experts) throw std::invalid_argument("topk exceeds experts");
    if(a.count("--weights2")) {auto c=load_capsule(a.at("--weights2"));if(c.shape.k!=s.k||c.shape.m!=s.m||c.shape.experts!=s.experts)throw std::invalid_argument("paired tensor shape mismatch");native1=std::move(c.bytes);}
    const auto synpair=num("--synthetic-pair","0");
    if(synpair>1 || (synpair && (a.count("--weights")||a.count("--weights2"))))throw std::invalid_argument("--synthetic-pair is 0|1 and only valid without capsule weights");
    if(synpair)native1=synth(s,seed+1);
    validate_tq2(native0,s); if(!native1.empty())validate_tq2(native1,s);
    std::mt19937 rng(seed+2);std::normal_distribution<float> normal(0.f,0.5f);
    const size_t jobs=size_t(tokens)*topk,outn=jobs*s.m;
    std::vector<float> x(size_t(tokens)*(per?topk:1)*s.k);
    if(a.count("--x-file"))x=read_raw<float>(a.at("--x-file"),x.size());
    else for(auto&v:x)v=normal(rng);
    for(float v:x)if(!std::isfinite(v)||std::abs(v)>65504.f)throw std::invalid_argument("activation not finite/FP16-representable; fallback required, no clipping is performed");
    const uint32_t iterations=1+warmup+repeats;
    std::vector<int32_t> ids(size_t(iterations)*jobs),fixed;
    if(a.count("--ids-file"))fixed=read_raw<int32_t>(a.at("--ids-file"),jobs);
    for(uint32_t it=0;it<iterations;++it) {
        if(!fixed.empty())std::copy(fixed.begin(),fixed.end(),ids.begin()+size_t(it)*jobs);
        else if(idmode=="fixed" && it>0)std::copy(ids.begin(),ids.begin()+jobs,ids.begin()+size_t(it)*jobs);
        else for(uint32_t t=0;t<tokens;++t) {
            std::vector<int32_t> pool(s.experts);std::iota(pool.begin(),pool.end(),0);std::shuffle(pool.begin(),pool.end(),rng);
            std::copy(pool.begin(),pool.begin()+topk,ids.begin()+size_t(it)*jobs+t*topk);
        }
    }
    for(auto id:ids)if(id<0||uint32_t(id)>=s.experts)throw std::invalid_argument("expert id out of range");
    auto packstart=Clock::now();
    std::vector<uint8_t> packed0,packed1;
    if(layout=="tile8") {packed0=repack_tile8(native0,s);if(!native1.empty())packed1=repack_tile8(native1,s);}
    const auto& w0=layout=="tile8"?packed0:native0;
    const auto& w1=layout=="tile8"?packed1:native1;
    auto packend=Clock::now();
    auto d=choose_device(get("--device","A750"));
    sycl::queue q(d,[](sycl::exception_list es){for(auto ep:es)std::rethrow_exception(ep);},
                  {sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
    DeviceMem<uint8_t> dw0(q,w0.size()),dw1(q,w1.size());
    DeviceMem<uint8_t> dx(q,x.size()*(input=="f16"?2:4));
    DeviceMem<float> dy0(q,outn),dy1(q,w1.empty()?0:outn);
    DeviceMem<int32_t> dids(q,ids.size()),status(q,jobs);
    DeviceProblem p{s,tokens,topk,per,layout=="tile8"?Layout::tile8:Layout::native_tq2,dw0.p,dw1.p,dx.p,dids.p,dy0.p,dy1.p,status.p};
    p.activation_type=input=="f16"?ActivationType::f16:ActivationType::f32;
    Options opt{num("--split","1"),num("--local","4")};
    if(mode=="fma"&&opt.split_k!=1)throw std::invalid_argument("FMA comparator split must be 1");
    DeviceMem<float> scratch(q,scratch_floats(p,opt));
    q.memcpy(dw0.p,w0.data(),w0.size());if(dw1.p)q.memcpy(dw1.p,w1.data(),w1.size());
    std::vector<uint16_t> xhalf(x.size());for(size_t i=0;i<x.size();++i)xhalf[i]=float_to_half(x[i]);
    q.memcpy(dx.p,input=="f16"?static_cast<const void*>(xhalf.data()):static_cast<const void*>(x.data()),dx.count);q.memcpy(dids.p,ids.data(),ids.size()*4);q.memset(status.p,0,jobs*4);q.wait_and_throw();
    std::cout<<std::setprecision(9)<<"device="<<d.get_info<sycl::info::device::name>()<<"\n"
      <<"source="<<(a.count("--weights")?a.at("--weights"):"SYNTHETIC")<<"\n"
      <<"mode="<<mode<<" input="<<input<<" layout="<<layout<<" K="<<s.k<<" M="<<s.m<<" experts="<<s.experts<<" tokens="<<tokens<<" topk="<<topk<<" pair="<<bool(p.w1)<<" per_selection="<<per<<" split_k="<<opt.split_k<<" local_size="<<opt.local_size<<"\n"
      <<"requested_instruction="<<(mode=="xmx"?"DPAS FP16xFP16 FP32 accumulate ExecSize8 RepeatCount1 (inspect ISA to confirm)":"SYCL F32 FMA comparator")<<"\n"
      <<"weight_device_bytes="<<w0.size()+w1.size()<<" scratch_bytes="<<scratch.count*4<<" repack_host_us="<<micros(packstart,packend)<<"\n"
      <<"timing_scope=standalone_SYCL_submit_to_completion; excludes_initial_upload,repack,readback,Vulkan_handoff,FA,router,sampling\n"
      <<"id_mode="<<(a.count("--ids-file")?"captured_fixed":idmode)<<"\n";
    std::ofstream csv;
    if(a.count("--csv")){csv.open(a.at("--csv"));if(!csv)throw std::runtime_error("cannot open CSV");csv<<"iteration,phase,main_us,reduce_us,event_span_us,submit_wait_us\n";}
    std::vector<double> wall,gpu,span;
    Run run;
    for(uint32_t it=0;it<iterations;++it) {
        p.ids=dids.p+size_t(it)*jobs;
        auto t0=Clock::now();
        run=mode=="xmx"?enqueue(q,p,opt,scratch.p):enqueue_fma_reference(q,p);
        run.done.wait_and_throw(); auto t1=Clock::now();
        double m=run.main_us(),r=run.reduce_us(),w=micros(t0,t1);
        const auto begin=run.main.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=run.done.get_profiling_info<sycl::info::event_profiling::command_end>();
        double sp=double(end-begin)/1000.;
        const char*phase=it==0?"first":it<=warmup?"warmup":"repeat";
        if(csv)csv<<it<<','<<phase<<','<<m<<','<<r<<','<<sp<<','<<w<<'\n';
        if(it==0)std::cout<<"first_main_us="<<m<<" first_reduce_us="<<r<<" first_submit_wait_us="<<w<<"\n";
        if(it>warmup){wall.push_back(w);gpu.push_back(m+r);span.push_back(sp);}
    }
    std::vector<float> y0(outn),y1(p.w1?outn:0);std::vector<int32_t> st(jobs);
    q.memcpy(y0.data(),dy0.p,outn*4);if(p.w1)q.memcpy(y1.data(),dy1.p,outn*4);q.memcpy(st.data(),status.p,jobs*4);q.wait_and_throw();
    for(auto v:st)if(v)throw std::runtime_error("device reported invalid expert ID");
    std::vector<int32_t> lastids(ids.end()-jobs,ids.end());
    auto f16ref0=reference(native0,s,x,lastids,tokens,topk,per,true);
    auto f32ref0=reference(native0,s,x,lastids,tokens,topk,per,false);
    auto same0=errors(y0,(mode=="xmx"||input=="f16")?f16ref0:f32ref0);
    print_error("y0_vs_matching_precision_gold",same0);print_error("y0_vs_original_F32_activation_gold",errors(y0,f32ref0));
    print_error("F16_activation_change_only",errors(f16ref0,f32ref0));
    bool passed=same0.finite&&same0.nmse<=1e-6&&same0.max_abs_over_rms<=0.005;
    if(p.w1){auto r=reference(native1,s,x,lastids,tokens,topk,per,mode=="xmx"||input=="f16");auto er=errors(y1,r);print_error("y1_vs_matching_precision_gold",er);passed=passed&&er.finite&&er.nmse<=1e-6&&er.max_abs_over_rms<=0.005;}
    std::cout<<"repeat_kernel_sum_median_us="<<percentile(gpu,.5)<<" repeat_event_span_median_us="<<percentile(span,.5)
      <<" repeat_submit_wait_median_us="<<percentile(wall,.5)<<" repeat_submit_wait_p95_us="<<percentile(wall,.95)
      <<" numerical_check="<<(passed?"PASS":"FAIL")<<"\n";
    return passed?0:2;
} catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
