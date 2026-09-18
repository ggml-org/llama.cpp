// SPDX-License-Identifier: MIT
#include "maple_w2a8.hpp"
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
struct Sample { double quant=0,main=0,reduce=0,span=0,wall=0,enqueue_return=0,tail_wait=0; };
struct Variant {
    std::string name; Layout layout; bool a8=false,fma=false; uint32_t group=0; A8Mode mode=A8Mode::staged;
    Sample first; std::vector<Sample> repeat;
    bool numerical=true,quant_ok=true; Error arithmetic{},total{},precision{};
    size_t quant_mismatch=0,near_tie=0;
};
static void worst(Error& a,const Error& b) {
    a.nmse=std::max(a.nmse,b.nmse);a.max_abs_over_rms=std::max(a.max_abs_over_rms,b.max_abs_over_rms);
    a.max_abs=std::max(a.max_abs,b.max_abs);a.finite=a.finite&&b.finite;
}
static bool arithmetic_pass(const Error& a) {return a.finite&&a.nmse<=1e-6&&a.max_abs_over_rms<=0.005;}
static std::vector<std::string> list(const std::string& s) {
    std::istringstream f(s);std::vector<std::string>v;std::string t;
    while(std::getline(f,t,',')) {if(t.empty())throw std::invalid_argument("empty list item");v.push_back(t);}
    if(v.empty())throw std::invalid_argument("empty list");
    return v;
}
static uint64_t hash_bytes(const void* p,size_t n) {
    const auto*b=static_cast<const uint8_t*>(p);uint64_t h=14695981039346656037ull;
    for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}return h;
}
static double stat(const Variant& v,double Sample::*member,double percentile_value=.5) {
    std::vector<double>x;for(auto&r:v.repeat)x.push_back(r.*member);return percentile(x,percentile_value);
}
int main(int argc,char**argv)try {
    std::map<std::string,std::string> a;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i];
        if(k=="--help") {
            std::cout<<"Maple W2A16 / native W2A8 shared-queue comparator v0.3\n"
              <<"--weights gate.mw2 [--weights2 up.mw2] OR --k 2048 --m 512 --experts 256 --synthetic-pair 0|1\n"
              <<"--input f32|f16 --tokens 1..8 --topk 8 --per-selection 0|1\n"
              <<"--layouts native,s2tile8 --groups 32,128,256 --a8-modes staged,fused --include-fma 0|1\n"
              <<"--split 1|2|4|8 --local 1..32 --repeats 28 --warmup 3 --seed 1234 --device A750\n"
              <<"--x-file x.f32 --ids-file ids.i32 --id-mode rotate|fixed\n"
              <<"--x-pattern normal|zeros|ties|outliers|alternating --scrub-mib 0..128\n"
              <<"--out result-directory --probe-only 0|1 --skip-probe 0|1\n"
              <<"Scope: kernel experiment, NOT full-model SYCL support or Vulkan comparison.\n";
            return 0;
        }
        if(k.rfind("--",0)!=0||i+1>=argc||a.count(k))throw std::invalid_argument("expected unique --key value");
        a[k]=argv[++i];
    }
    const std::set<std::string> known={"--weights","--weights2","--k","--m","--experts","--synthetic-pair","--input","--tokens","--topk","--per-selection","--layouts","--groups","--a8-modes","--include-fma","--split","--local","--repeats","--warmup","--seed","--device","--x-file","--ids-file","--id-mode","--x-pattern","--scrub-mib","--out","--probe-only","--skip-probe"};
    for(auto&kv:a)if(!known.count(kv.first))throw std::invalid_argument("unknown argument "+kv.first);
    auto get=[&](const char*k,const char*d){return a.count(k)?a.at(k):std::string(d);};
    auto number=[&](const std::string& s){size_t e=0;auto v=std::stoull(s,&e);if(s.empty()||s[0]=='-'||e!=s.size()||v>UINT32_MAX)throw std::invalid_argument("invalid unsigned integer");return uint32_t(v);};
    auto num=[&](const char*k,const char*d){return number(get(k,d));};
    auto boolean=[&](const char*k,const char*d){auto v=num(k,d);if(v>1)throw std::invalid_argument(std::string(k)+" requires 0|1");return v!=0;};
    auto d=choose_device(get("--device","A750"));
    sycl::queue q(d,[](sycl::exception_list es){for(auto e:es)std::rethrow_exception(e);},
      {sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
    std::cout<<std::setprecision(10)<<"device="<<d.get_info<sycl::info::device::name>()<<"\ndriver="<<d.get_info<sycl::info::device::driver_version>()<<'\n';
    if(!boolean("--skip-probe","0")||boolean("--probe-only","0"))run_s2s8_probe(q);
    if(boolean("--probe-only","0"))return 0;
    Shape s{num("--k","2048"),num("--m","512"),num("--experts","256")};
    uint32_t seed=num("--seed","1234"),tokens=num("--tokens","1"),topk=num("--topk","8");
    uint32_t repeats=num("--repeats","28"),warmup=num("--warmup","3"),scrub_mib=num("--scrub-mib","0");
    bool per=boolean("--per-selection","0"),synpair=boolean("--synthetic-pair","0"),fma=boolean("--include-fma","0");
    if(!tokens||tokens>8||!topk||topk>256||!repeats||repeats>10000||warmup>1000||scrub_mib>128)throw std::invalid_argument("invalid benchmark limits");
    const auto input=get("--input","f32"),idmode=get("--id-mode","rotate"),pattern=get("--x-pattern","normal");
    if((input!="f16"&&input!="f32")||(idmode!="rotate"&&idmode!="fixed"))throw std::invalid_argument("bad input/id-mode");
    if(pattern!="normal"&&pattern!="zeros"&&pattern!="ties"&&pattern!="outliers"&&pattern!="alternating")throw std::invalid_argument("bad x-pattern");
    if(a.count("--weights2")&&!a.count("--weights"))throw std::invalid_argument("weights2 requires weights");
    if(synpair&&(a.count("--weights")||a.count("--weights2")))throw std::invalid_argument("synthetic-pair cannot be mixed with real weights");
    std::vector<uint8_t>w0,w1;
    if(a.count("--weights")){auto c=load_capsule(a.at("--weights"));s=c.shape;w0=std::move(c.bytes);}else w0=synth(s,seed);
    if(a.count("--weights2")){auto c=load_capsule(a.at("--weights2"));if(c.shape.k!=s.k||c.shape.m!=s.m||c.shape.experts!=s.experts)throw std::invalid_argument("pair shape mismatch");w1=std::move(c.bytes);}
    if(synpair)w1=synth(s,seed+1);
    if(topk>s.experts)throw std::invalid_argument("topk exceeds experts");
    validate_tq2(w0,s);if(!w1.empty())validate_tq2(w1,s);
    Options opt{num("--split","1"),num("--local","4")};
    if((opt.split_k!=1&&opt.split_k!=2&&opt.split_k!=4&&opt.split_k!=8)||(s.k/256)%opt.split_k||!opt.local_size||opt.local_size>32)throw std::invalid_argument("invalid split/local for this K");
    std::vector<Variant> variants;
    const auto layouts=list(get("--layouts","native,s2tile8")),gs=list(get("--groups","32,128,256")),modes=list(get("--a8-modes","staged,fused"));
    for(auto l:layouts) {
        if(l!="native"&&l!="s2tile8")throw std::invalid_argument("layouts must be native,s2tile8; legacy tile8 is only in old bench");
        Layout ly=l=="native"?Layout::native_tq2:Layout::s2tile8;
        variants.push_back(Variant{l+"-w2a16",ly});
        for(auto g:gs)for(auto m:modes) {
            uint32_t group=number(g);check_a8_group(group);
            if(m!="staged"&&m!="fused")throw std::invalid_argument("bad A8 mode");
            Variant v{l+"-w2a8-g"+g+"-"+m,ly};v.a8=true;v.group=group;v.mode=m=="staged"?A8Mode::staged:A8Mode::fused;variants.push_back(v);
        }
    }
    if(fma){Variant v{"native-fma-SYCL",Layout::native_tq2};v.fma=true;variants.push_back(v);}
    {std::set<std::string> n;for(auto&v:variants)if(!n.insert(v.name).second)throw std::invalid_argument("duplicate variant");}
    const size_t rows=size_t(tokens)*(per?topk:1),xn=rows*s.k,jobs=size_t(tokens)*topk,outn=jobs*s.m;
    std::mt19937 rng(seed+2);std::normal_distribution<float>normal(0,.5f);
    std::vector<float>x(xn);
    if(a.count("--x-file"))x=read_raw<float>(a.at("--x-file"),xn);
    else for(size_t i=0;i<xn;++i) {
        if(pattern=="zeros")x[i]=0;
        else if(pattern=="ties") {const float v[]={127.f,-127.f,.5f,-.5f,1.5f,-1.5f,2.5f,-2.5f};x[i]=v[i%8];}
        else if(pattern=="alternating")x[i]=(i%2?-1.f:1.f)*(1+(i%13))*.125f;
        else {x[i]=normal(rng);if(pattern=="outliers"&&i%128==0)x[i]*=100.f;}
    }
    std::vector<uint16_t>xh(xn);std::vector<float>effective=x;
    for(size_t i=0;i<xn;++i){if(!std::isfinite(x[i])||std::abs(x[i])>65504.f)throw std::invalid_argument("input must be finite and within the finite FP16 range for shared A16/A8 comparison");xh[i]=float_to_half(x[i]);if(input=="f16")effective[i]=half_to_float(xh[i]);}
    const uint32_t rounds=1+warmup+repeats;
    std::vector<int32_t> ids(size_t(rounds)*jobs),fixed;
    if(a.count("--ids-file"))fixed=read_raw<int32_t>(a.at("--ids-file"),jobs);
    for(uint32_t it=0;it<rounds;++it) {
        if(!fixed.empty())std::copy(fixed.begin(),fixed.end(),ids.begin()+size_t(it)*jobs);
        else if(idmode=="fixed"&&it>0)std::copy(ids.begin(),ids.begin()+jobs,ids.begin()+size_t(it)*jobs);
        else for(uint32_t t=0;t<tokens;++t){
            std::vector<int32_t>pool(s.experts);std::iota(pool.begin(),pool.end(),0);
            if(it==0)std::rotate(pool.begin(),pool.end()-1,pool.end()); // last expert, 0, 1... guaranteed in snapshot 0
            else std::shuffle(pool.begin(),pool.end(),rng);
            std::copy(pool.begin(),pool.begin()+topk,ids.begin()+size_t(it)*jobs+t*topk);
        }
    }
    for(auto v:ids)if(v<0||uint32_t(v)>=s.experts)throw std::invalid_argument("invalid expert ID");
    bool need_native=fma,need_s2=false;for(auto&v:variants){need_native|=v.layout==Layout::native_tq2;need_s2|=v.layout==Layout::s2tile8;}
    auto tpack0=Clock::now();std::vector<uint8_t> sw0,sw1;
    if(need_s2){sw0=repack_s2tile8(w0,s);if(!w1.empty())sw1=repack_s2tile8(w1,s);}
    const double repack_us=micros(tpack0,Clock::now());
    DeviceMem<uint8_t>dn0(q,need_native?w0.size():0),dn1(q,need_native?w1.size():0),ds0(q,sw0.size()),ds1(q,sw1.size());
    DeviceMem<uint8_t>dx(q,xn*(input=="f16"?2:4)),scrub(q,size_t(scrub_mib)*1024*1024);
    DeviceMem<int32_t> di(q,ids.size()),status(q,jobs),invalid(q,xn/32);
    DeviceMem<int8_t> aq(q,xn);DeviceMem<float>asc(q,xn/32),y0(q,outn),y1(q,w1.empty()?0:outn);
    DeviceProblem p{s,tokens,topk,per,Layout::native_tq2,dn0.p,dn1.p,dx.p,di.p,y0.p,y1.p,status.p};
    p.activation_type=input=="f16"?ActivationType::f16:ActivationType::f32;
    // Scratch count uses whether a paired output exists, independent of selected layout.
    p.w1=w1.empty()?nullptr:(dn1.p?dn1.p:ds1.p);
    DeviceMem<float>scratch(q,scratch_floats(p,opt));
    if(dn0.p)q.memcpy(dn0.p,w0.data(),w0.size());if(dn1.p)q.memcpy(dn1.p,w1.data(),w1.size());
    if(ds0.p)q.memcpy(ds0.p,sw0.data(),sw0.size());if(ds1.p)q.memcpy(ds1.p,sw1.data(),sw1.size());
    q.memcpy(dx.p,input=="f16"?static_cast<const void*>(xh.data()):static_cast<const void*>(x.data()),dx.count);
    q.memcpy(di.p,ids.data(),ids.size()*4);q.memset(status.p,0,jobs*4);q.memset(invalid.p,0,invalid.count*4);q.wait_and_throw();
    const std::filesystem::path out=get("--out","build/results/manual");
    for(const char* existing:{"manifest.txt","samples.csv","summary.csv","numerical.csv"})
        if(std::filesystem::exists(out/existing))throw std::runtime_error("output already contains evidence; choose a new --out directory");
    std::filesystem::create_directories(out);
    std::ofstream manifest(out/"manifest.txt"),samples(out/"samples.csv");
    if(!manifest||!samples)throw std::runtime_error("cannot open output files");
    manifest<<"version=0.3\ndevice="<<d.get_info<sycl::info::device::name>()<<"\ndriver="<<d.get_info<sycl::info::device::driver_version>()<<'\n';
    for(auto&kv:a)manifest<<kv.first<<'='<<kv.second<<'\n';
    manifest<<"weight0_fnv1a64="<<hash_bytes(w0.data(),w0.size())<<"\nweight1_fnv1a64="<<hash_bytes(w1.data(),w1.size())<<"\nactivation_fnv1a64="<<hash_bytes(x.data(),x.size()*4)<<"\nids_fnv1a64="<<hash_bytes(ids.data(),ids.size()*4)<<'\n';
    manifest<<"weight_source="<<(a.count("--weights")?"capsule":"synthetic")<<"\nactivation_source="<<(a.count("--x-file")?"captured":"synthetic")<<"\nids_source="<<(a.count("--ids-file")?"captured_fixed":idmode)<<'\n';
    manifest<<"weight_bytes_per_layout="<<w0.size()+w1.size()<<"\nweight_bytes_resident_in_comparator="<<dn0.count+dn1.count+ds0.count+ds1.count<<"\nsplit_scratch_bytes="<<scratch.count*4<<"\na8_q_bytes="<<aq.count<<"\na8_scale_bytes_max="<<asc.count*4<<"\nrepack_host_us="<<repack_us<<'\n';
    manifest<<"queue=one_shared_in_order_Level_Zero_SYCL_queue\nscope=standalone_submit_to_completion; excludes upload,repack,readback,Vulkan_handoff,FA,router,GLU,sampling\n";
    manifest<<"schedule=rotating_and_reversing_variant_order; identical_IDs_per_round\ncache_note=not_a_full_model_cache_trace; scrub_is_optional_and_not_proof_of_cold_DRAM\n";
    samples<<std::setprecision(12)<<"round,phase,order,variant,ids_fnv1a64,quant_us,main_us,reduce_us,kernel_sum_us,event_span_us,submit_wait_us,enqueue_return_us,tail_wait_us\n";
    std::cout<<"K="<<s.k<<" M="<<s.m<<" experts="<<s.experts<<" Q="<<tokens<<" topk="<<topk<<" pair="<<!w1.empty()<<" per_selection="<<per<<" variants="<<variants.size()<<"\n"
             <<"requested_native_op=DPAS.s2.s8 depth8 repeat1 ExecSize8 K32; NOT DP4A\n"
             <<"requested_a16_op=DPAS.hf.hf FP32 accumulation; disassembly not observed here\n"
             <<"A8 includes every-call activation quantization. Fused q/scales capture is validation-only.\n";
    auto launch_one=[&](Variant& v,uint32_t round,bool capture)->Sample {
        p.layout=v.layout;p.w0=v.layout==Layout::native_tq2?dn0.p:ds0.p;p.w1=v.layout==Layout::native_tq2?dn1.p:ds1.p;
        p.ids=di.p+size_t(round)*jobs;
        if(scrub.p) { // Untimed cache-control experiment; not claimed to evict every cache.
            const size_t n=scrub.count/4;uint32_t* b=reinterpret_cast<uint32_t*>(scrub.p);
            q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){b[i[0]]=uint32_t(i[0])*1664525u+round;}).wait_and_throw();
        }
        auto begin_cpu=Clock::now();sycl::event first,last;Sample sample;
        if(v.a8) {
            A8Workspace w{aq.p,asc.p,invalid.p,capture};A8Options ao{v.group,v.mode,opt};
            auto r=enqueue_a8(q,p,ao,w,scratch.p);auto submitted=Clock::now();r.done.wait_and_throw();auto end_cpu=Clock::now();sample.enqueue_return=micros(begin_cpu,submitted);sample.tail_wait=micros(submitted,end_cpu);
            first=r.first_event();last=r.done;sample.quant=r.quant_us();sample.main=r.main_us();sample.reduce=r.reduce_us();sample.wall=micros(begin_cpu,end_cpu);
        } else {
            auto r=v.fma?enqueue_fma_reference(q,p):enqueue(q,p,opt,scratch.p);auto submitted=Clock::now();r.done.wait_and_throw();auto end_cpu=Clock::now();sample.enqueue_return=micros(begin_cpu,submitted);sample.tail_wait=micros(submitted,end_cpu);
            first=r.main;last=r.done;sample.main=r.main_us();sample.reduce=r.reduce_us();sample.wall=micros(begin_cpu,end_cpu);
        }
        auto b=first.get_profiling_info<sycl::info::event_profiling::command_start>();auto e=last.get_profiling_info<sycl::info::event_profiling::command_end>();sample.span=double(e-b)/1000.;
        return sample;
    };
    for(uint32_t r=0;r<rounds;++r) {
        const char* phase=r==0?"first":r<=warmup?"warmup":"repeat";
        for(size_t j=0;j<variants.size();++j) {
            size_t i=(j+r)%variants.size();if(r%2)i=variants.size()-1-i;
            auto&v=variants[i];auto s0=launch_one(v,r,false);
            if(r==0)v.first=s0;if(r>warmup)v.repeat.push_back(s0);
            samples<<r<<','<<phase<<','<<j<<','<<v.name<<','<<hash_bytes(ids.data()+size_t(r)*jobs,jobs*4)<<','<<s0.quant<<','<<s0.main<<','<<s0.reduce<<','<<s0.quant+s0.main+s0.reduce<<','<<s0.span<<','<<s0.wall<<','<<s0.enqueue_return<<','<<s0.tail_wait<<'\n';
        }
    }
    samples.close();
    // Validate two routing snapshots AFTER timing. Capture writes are outside measured samples.
    std::ofstream numerical(out/"numerical.csv");numerical<<std::setprecision(12)<<"snapshot,variant,projection,arithmetic_nmse,total_nmse,precision_only_nmse,arithmetic_max_abs_over_rms,kernel_pass,quant_pass,q_mismatch,q_near_tie\n";
    if(!numerical)throw std::runtime_error("cannot create numerical.csv");
    std::vector<float>gy0(outn),gy1(w1.empty()?0:outn);std::vector<int32_t>st(jobs);
    for(uint32_t snapshot:std::vector<uint32_t>{0,rounds-1}) {
        std::vector<int32_t>id(ids.begin()+size_t(snapshot)*jobs,ids.begin()+size_t(snapshot+1)*jobs);
        auto ref32=reference(w0,s,x,id,tokens,topk,per,false),ref16=reference(w0,s,x,id,tokens,topk,per,true);
        std::vector<float>ref321,ref161;if(!w1.empty()){ref321=reference(w1,s,x,id,tokens,topk,per,false);ref161=reference(w1,s,x,id,tokens,topk,per,true);}
        std::map<uint32_t,QuantizedHost>hostq;for(auto g:gs){auto gi=number(g);hostq.emplace(gi,quantize_a8(effective,uint32_t(rows),s.k,gi));}
        for(auto&v:variants) {
            launch_one(v,snapshot,true);
            q.memcpy(gy0.data(),y0.p,outn*4);if(y1.p)q.memcpy(gy1.data(),y1.p,outn*4);q.memcpy(st.data(),status.p,jobs*4);q.wait_and_throw();
            for(auto code:st)if(code)throw std::runtime_error("invalid expert ID flagged by device");
            std::vector<float>gold0,gold1;bool quant_ok=true;size_t mismatch=0,near_tie=0;
            if(v.a8) {
                auto&hq=hostq.at(v.group);
                QuantizedHost cap{uint32_t(rows),s.k,v.group,std::vector<int8_t>(xn),std::vector<float>(xn/v.group)};
                std::vector<int32_t>bad(xn/v.group);
                q.memcpy(cap.q.data(),aq.p,xn);q.memcpy(cap.scale.data(),asc.p,cap.scale.size()*4);q.memcpy(bad.data(),invalid.p,bad.size()*4);q.wait_and_throw();
                for(size_t g=0;g<cap.scale.size();++g) {
                    const float d0=cap.scale[g],dh=hq.scale[g];
                    if(bad[g]||!std::isfinite(d0)||d0<=0||std::abs(d0-dh)>std::max(1e-38f,std::abs(dh)*5e-6f))quant_ok=false;
                }
                for(size_t k=0;k<xn;++k) {
                    int delta=int(cap.q[k])-int(hq.q[k]);
                    if(cap.q[k]==-128)quant_ok=false;
                    if(delta) {
                        ++mismatch;float z=std::abs(effective[k]/hq.scale[k/v.group]);float frac=z-std::floor(z);
                        if(std::abs(delta)==1&&std::abs(frac-.5f)<=2e-4f)++near_tie;else quant_ok=false;
                    }
                }
                gold0=reference_a8(w0,s,cap,id,tokens,topk,per);if(!w1.empty())gold1=reference_a8(w1,s,cap,id,tokens,topk,per);
            } else {gold0=(v.fma&&input=="f32")?ref32:ref16;if(!w1.empty())gold1=(v.fma&&input=="f32")?ref321:ref161;}
            const auto ar=errors(gy0,gold0),tot=errors(gy0,ref32),prec=errors(gold0,ref32);
            v.numerical=v.numerical&&arithmetic_pass(ar);v.quant_ok=v.quant_ok&&quant_ok;worst(v.arithmetic,ar);worst(v.total,tot);worst(v.precision,prec);v.quant_mismatch+=mismatch;v.near_tie+=near_tie;
            numerical<<snapshot<<','<<v.name<<",0,"<<ar.nmse<<','<<tot.nmse<<','<<prec.nmse<<','<<ar.max_abs_over_rms<<','<<arithmetic_pass(ar)<<','<<quant_ok<<','<<mismatch<<','<<near_tie<<'\n';
            if(!w1.empty()) {
                auto b=errors(gy1,gold1),t=errors(gy1,ref321),pc=errors(gold1,ref321);v.numerical=v.numerical&&arithmetic_pass(b);worst(v.arithmetic,b);worst(v.total,t);worst(v.precision,pc);
                numerical<<snapshot<<','<<v.name<<",1,"<<b.nmse<<','<<t.nmse<<','<<pc.nmse<<','<<b.max_abs_over_rms<<','<<arithmetic_pass(b)<<','<<quant_ok<<','<<mismatch<<','<<near_tie<<'\n';
            }
        }
    }
    std::ofstream summary(out/"summary.csv");if(!summary)throw std::runtime_error("cannot create summary");
    summary<<std::setprecision(12)<<"variant,layout,group,a8_mode,split_k,local_size,quant_median_us,main_median_us,reduce_median_us,event_span_median_us,submit_wait_median_us,submit_wait_p95_us,first_submit_wait_us,speedup_vs_same_layout_a16_wall,kernel_pass,quant_pass,arithmetic_nmse,total_nmse,precision_only_nmse,arithmetic_max_abs_over_rms,q_mismatch,q_near_tie,enqueue_return_median_us,tail_wait_median_us\n";
    bool pass=true;
    for(auto&v:variants) {
        auto baseline=std::find_if(variants.begin(),variants.end(),[&](const Variant&b){return !b.a8&&!b.fma&&b.layout==v.layout;});
        double speed=baseline==variants.end()?0:stat(*baseline,&Sample::wall)/stat(v,&Sample::wall);
        summary<<v.name<<','<<(v.layout==Layout::native_tq2?"native":"s2tile8")<<','<<v.group<<','<<(v.a8?(v.mode==A8Mode::staged?"staged":"fused"):"none")<<','<<(v.fma?1:opt.split_k)<<','<<opt.local_size<<','
               <<stat(v,&Sample::quant)<<','<<stat(v,&Sample::main)<<','<<stat(v,&Sample::reduce)<<','<<stat(v,&Sample::span)<<','<<stat(v,&Sample::wall)<<','<<stat(v,&Sample::wall,.95)<<','<<v.first.wall<<','<<speed<<','<<v.numerical<<','<<v.quant_ok<<','<<v.arithmetic.nmse<<','<<v.total.nmse<<','<<v.precision.nmse<<','<<v.arithmetic.max_abs_over_rms<<','<<v.quant_mismatch<<','<<v.near_tie<<','<<stat(v,&Sample::enqueue_return)<<','<<stat(v,&Sample::tail_wait)<<'\n';
        std::cout<<v.name<<" wall_median_us="<<stat(v,&Sample::wall)<<" p95="<<stat(v,&Sample::wall,.95)<<" main_us="<<stat(v,&Sample::main)<<" gpu_span_us="<<stat(v,&Sample::span)<<" enqueue_return_us="<<stat(v,&Sample::enqueue_return)<<" total_nmse="<<v.total.nmse<<" arithmetic="<<(v.numerical?"PASS":"FAIL")<<" quant="<<(v.quant_ok?"PASS":"FAIL")<<'\n';
        pass=pass&&v.numerical&&v.quant_ok;
    }
    std::cout<<"RESULT="<<(pass?"PASS":"FAIL")<<" output="<<out.string()<<"\n"
             <<"PASS means kernel and quantizer implementation checks; activation precision/model quality is NOT certified.\n";
    return pass?0:2;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
