// SPDX-License-Identifier: MIT
#include "maple_moe.hpp"
#include "moe_reference.hpp"
#include "a8_contract_audit.hpp"
#include "moe_profiling.hpp"
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
#include <cstring>
#include <array>

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
struct ChainSample {double wall=0,enqueue_return=0,tail_wait=0,gpu_span=0,kernel_sum=0,gpu_gap=0,group_us=0,core_span=0,gate_us=0,down_us=0,union_busy=0,overlap=0;};
struct ChainVariant {std::string name;MoeOptions opt;std::vector<ChainSample> samples;bool pass=true;double total_nmse=0;};
static bool good(const Error&e){return e.finite&&e.nmse<=1e-6&&e.max_abs_over_rms<=.005;}
static double value(const ChainVariant&v,double ChainSample::*field,double p=.5){std::vector<double>a;for(auto&s:v.samples)a.push_back(s.*field);return percentile(a,p);}
template<class T>static std::vector<T> copy_back(sycl::queue&q,const T*p,size_t n){std::vector<T>v(n);q.memcpy(v.data(),p,n*sizeof(T)).wait_and_throw();return v;}
static QuantizedHost capture_quant(sycl::queue&q,const A8Workspace&w,uint32_t rows,uint32_t k,uint32_t group){
    return {rows,k,group,copy_back(q,w.q,size_t(rows)*k),copy_back(q,w.scales,size_t(rows)*k/group)};}
static bool quant_ok(const QuantizedHost&got,const std::vector<float>&x){return audit_a8(got,x).pass;}
template<class T>static void dump_raw(const std::filesystem::path&p,const std::vector<T>&v) {
    std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot create dump "+p.string());
    f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*sizeof(T)));
    if(!f)throw std::runtime_error("short dump write "+p.string());
}
static bool all_zero(const std::vector<int32_t>&v){return std::all_of(v.begin(),v.end(),[](int32_t x){return x==0;});}
// A scheduling-only experiment. The two arms use identical tensor layout,
// quantizers, DPAS repeat=1, split, epilogue, output indexing and route sum order.
static void require(bool yes,const std::string& message) {if(!yes)throw std::runtime_error(message);}
template<class T>static bool bits_equal(const std::vector<T>&a,const std::vector<T>&b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;
}
template<class T>static std::vector<T> pick_tokens(const std::vector<T>&a,const std::vector<uint32_t>&ts,size_t stride) {
    std::vector<T> out;out.reserve(ts.size()*stride);
    for(auto t:ts){require((size_t(t)+1)*stride<=a.size(),"sample outside captured tensor");
        out.insert(out.end(),a.begin()+size_t(t)*stride,a.begin()+(size_t(t)+1)*stride);}
    return out;
}
static QuantizedHost pick_quant(const QuantizedHost&a,const std::vector<uint32_t>&ts,uint32_t rows_per_token) {
    return {uint32_t(ts.size())*rows_per_token,a.k,a.group,
        pick_tokens(a.q,ts,size_t(rows_per_token)*a.k),pick_tokens(a.scale,ts,size_t(rows_per_token)*a.k/a.group)};
}
static ExpertGroupingHost capture_mapping(sycl::queue&q,const ExpertGroupingWorkspace&w,size_t jobs,uint32_t e) {
    return {copy_back(q,w.counts,e+1),copy_back(q,w.offsets,e+2),copy_back(q,w.sorted_to_job,jobs)};
}
static void grouping_probe(sycl::queue&q) {
    // Test on an out-of-order queue as well: every stage must depend on its producer.
    sycl::queue ooo(q.get_context(),q.get_device(),[](sycl::exception_list es){for(auto e:es)std::rethrow_exception(e);},
                    sycl::property_list{sycl::property::queue::enable_profiling{}});
    const std::array<size_t,8> sizes={1,8,127,128,129,257,104,16384};
    size_t cases=0,checked=0;
    for(uint32_t e:{1u,3u,256u})for(size_t n:sizes) {
        std::vector<int32_t>ids(n);
        for(size_t j=0;j<n;++j)ids[j]=j%11==0?-1:(j%17==0?int32_t(e):int32_t((j*37+5)%e));
        DeviceMem<int32_t>di(ooo,n),c(ooo,e+2),off(ooo,e+3),order(ooo,n+1);
        ExpertGroupingWorkspace w{c.p,off.p,order.p,n,e};
        // Poison trailing canaries to detect off-by-one writes. Enqueue depends on all writes.
        std::vector<sycl::event>deps={ooo.memcpy(di.p,ids.data(),n*4),ooo.fill(c.p,int32_t(-999),e+2),
            ooo.fill(off.p,int32_t(-999),e+3),ooo.fill(order.p,int32_t(-999),n+1)};
        auto run=enqueue_expert_grouping(ooo,di.p,n,e,w,deps);run.scatter.wait_and_throw();
        require(same_grouping(capture_mapping(ooo,w,n,e),group_jobs_reference(ids,e)),"GPU grouping probe: mapping mismatch");
        require(copy_back(ooo,c.p+e+1,1)[0]==-999&&copy_back(ooo,off.p+e+2,1)[0]==-999&&copy_back(ooo,order.p+n,1)[0]==-999,"GPU grouping probe: canary overwritten");
        for(uint32_t width:{2u,4u}) {
            const size_t cap=expert_token_tile_capacity(n,e,width);
            DeviceMem<int32_t> first(ooo,cap+1),len(ooo,cap+1),expert(ooo,cap+1),total(ooo,1);
            auto z0=ooo.fill(first.p,int32_t(-999),cap+1),z1=ooo.fill(len.p,int32_t(-999),cap+1),z2=ooo.fill(expert.p,int32_t(-999),cap+1);
            ExpertTokenTileView tv{first.p,len.p,expert.p,total.p,cap,width};
            auto event=enqueue_expert_token_tiles(ooo,n,e,w,tv,{run.scatter,z0,z1,z2});event.wait_and_throw();
            auto expected=token_tiles_reference(group_jobs_reference(ids,e),width);auto size=expected.first.size();
            require(copy_back(ooo,total.p,1)[0]==int32_t(size)&&copy_back(ooo,first.p,size)==expected.first&&
                copy_back(ooo,len.p,size)==expected.length&&copy_back(ooo,expert.p,size)==expected.expert,"GPU tile probe mismatch");
            require(copy_back(ooo,first.p+cap,1)[0]==-999&&copy_back(ooo,len.p+cap,1)[0]==-999&&
                copy_back(ooo,expert.p+cap,1)[0]==-999,"GPU tile descriptor canary overwritten");
        }
        // Reuse SAME workspace with entirely different IDs: stale routing cannot pass.
        std::fill(ids.begin(),ids.end(),int32_t(e-1));
        auto upload=ooo.memcpy(di.p,ids.data(),n*4);
        auto again=enqueue_expert_grouping(ooo,di.p,n,e,w,{upload});again.scatter.wait_and_throw();
        require(same_grouping(capture_mapping(ooo,w,n,e),group_jobs_reference(ids,e)),"GPU grouping probe: stale mapping");
        checked+=2*n;cases+=2;
    }
    std::cout<<"GROUPING_PROBE PASS cases="<<cases<<" job_entries="<<checked<<" invalid bucket + canaries + changed IDs + OOO\n";
}
static void quantizer_probe(sycl::queue& q) {
    std::vector<float> x(4*256,0.f);
    x[0]=127.f;x[1]=110.5f;x[2]=111.5f;x[3]=-110.5f;x[4]=-111.5f;
    x[32]=.8111281f;x[33]=.7057453393936157f;
    x[64]=1e-40f;x[65]=-1e-40f;
    x[96]=1e30f;x[97]=-1e29f;
    for(size_t i=128;i<x.size();++i)x[i]=float(int(i%113)-56)*.0111f;
    DeviceMem<float>dx(q,x.size()),ds(q,x.size()/32);DeviceMem<int8_t>dq(q,x.size());DeviceMem<int32_t>invalid(q,x.size()/32);
    A8Workspace w{dq.p,ds.p,invalid.p,false};auto upload=q.memcpy(dx.p,x.data(),x.size()*4);
    auto e=enqueue_a8_quant(q,dx.p,ActivationType::f32,4,256,32,w,{upload});e.wait_and_throw();
    auto got=capture_quant(q,w,4,256,32);auto audit=audit_a8(got,x);
    require(audit.pass&&all_zero(copy_back(q,invalid.p,x.size()/32)),"standalone A8 boundary probe failed");
    require(got.q[1]==110&&got.q[2]==112&&got.q[3]==-110&&got.q[4]==-112,"A8 RNE tie contract changed");
    x[128]=std::numeric_limits<float>::quiet_NaN();upload=q.memcpy(dx.p,x.data(),x.size()*4);
    e=enqueue_a8_quant(q,dx.p,ActivationType::f32,4,256,32,w,{upload});e.wait_and_throw();
    auto bad=copy_back(q,invalid.p,x.size()/32);require(bad[4]!=0,"nonfinite input was silently accepted");
    std::cout<<"A8_CONTRACT_PROBE PASS RNE/tiny/large/nonfinite; cpu_code_diff="<<audit.cpu_code_differences<<" max_scale_ulp="<<audit.max_scale_ulp<<'\n';
}
struct Snapshot {
    std::vector<float> gate,up,hidden,down,y;
    QuantizedHost input_q,hidden_q;
};
static Snapshot capture_snapshot(sycl::queue&q,const MoeProblem&p,const MoeWorkspace&w) {
    size_t jobs=size_t(p.tokens)*p.topk,hn=jobs*p.gate_shape.m,dn=jobs*p.gate_shape.k,xn=size_t(p.tokens)*p.gate_shape.k;
    Snapshot s{copy_back(q,w.gate,hn),copy_back(q,w.up,hn),copy_back(q,w.hidden,hn),copy_back(q,w.down,dn),copy_back(q,p.y,xn),
        capture_quant(q,w.input_a8,p.tokens,p.gate_shape.k,32),capture_quant(q,w.hidden_a8,uint32_t(jobs),p.gate_shape.m,32)};
    require(all_zero(copy_back(q,w.gate_status,jobs))&&all_zero(copy_back(q,w.down_status,jobs))&&
        all_zero(copy_back(q,w.hidden_invalid,hn))&&all_zero(copy_back(q,w.output_invalid,xn))&&
        all_zero(copy_back(q,w.input_a8.invalid,xn/32))&&all_zero(copy_back(q,w.hidden_a8.invalid,hn/32)),"nonzero device status");
    return s;
}
static bool near_scheduling_equal(const Error&e) {return e.finite&&e.nmse<=1e-10&&e.max_abs_over_rms<=1e-4;}
int main(int argc,char**argv)try {
    std::map<std::string,std::string>a;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i];
        if(k=="--help") {std::cout<<"Maple W2 XMX v0.4 Vulkan architecture port: policy, fork/join, grouped reuse\n"
            <<"--gate gate.mw2 --up up.mw2 --down down.mw2 OR --k 2048 --hidden 512 --experts 256\n"
            <<"--tokens 1..2048 --topk 8 --device A750 --out NEW_DIR --repeats 28 --warmup 3\n"
            <<"--gate-split 1 --down-split 1 --local 4 --in-order 1 --scrub-mib 0\n"
            <<"--x-file x.f32 --ids-file ids.i32 --routes-file routes.f32\n"
            <<"--id-mode rotate|fixed --pattern normal|zeros|outliers --probe-only 1 --skip-probe 1\n"
            <<"--candidate grouped|auto|direct --overlap 0|1 --gate-tile 1|2|4 --down-tile 1|2|4\n"
            <<"--auto-min-tokens 0 (uncalibrated) --auto-min-mean 0 --dump-contract 0|1 --freeze-input 0|1\n"
            <<"T2/T4 reuse B but keep RepeatCount=1; not multi-row DPAS. Baseline/candidate AB/BA.\n";return 0;}
        if(k.rfind("--",0)!=0||i+1==argc||a.count(k))throw std::invalid_argument("expected unique --key value");a[k]=argv[++i];
    }
    const std::set<std::string>allowed={"--gate","--up","--down","--k","--hidden","--experts","--tokens","--topk","--gate-split","--down-split","--local","--x-file","--ids-file","--routes-file","--pattern","--repeats","--warmup","--device","--out","--skip-probe","--probe-only","--in-order","--id-mode","--scrub-mib","--candidate","--overlap","--gate-tile","--down-tile","--auto-min-tokens","--auto-min-mean","--dump-contract","--freeze-input"};
    for(auto&kv:a)require(allowed.count(kv.first)!=0,"unknown argument "+kv.first);
    auto get=[&](const char*k,const char*d){return a.count(k)?a.at(k):std::string(d);};
    auto num=[&](const char*k,const char*d){auto s=get(k,d);size_t end=0;auto n=std::stoull(s,&end);
        if(s.empty()||s[0]=='-'||end!=s.size()||n>UINT32_MAX)throw std::invalid_argument("bad unsigned value for "+std::string(k));return uint32_t(n);};
    auto bit=[&](const char*k,const char*d){auto n=num(k,d);require(n<=1,"boolean must be 0/1");return n!=0;};
    uint32_t tokens=num("--tokens","1"),topk=num("--topk","8"),repeats=num("--repeats","28"),warmup=num("--warmup","3"),scrub_mib=num("--scrub-mib","0");
    require(tokens>=1&&tokens<=2048&&topk>=1&&topk<=256&&repeats>=2&&repeats<=10000&&warmup<=1000&&scrub_mib<=128,"benchmark limit exceeded");

    const uint32_t gate_tile=num("--gate-tile","1"),down_tile=num("--down-tile","1");
    check_tokens_per_tile(gate_tile);check_tokens_per_tile(down_tile);
    const std::string candidate=get("--candidate","grouped");
    require(candidate=="grouped"||candidate=="auto"||candidate=="direct","invalid candidate schedule");
    const bool overlap=bit("--overlap","0"),dump=bit("--dump-contract","0"),freeze=bit("--freeze-input","0");
    MoeAutoPolicy policy{num("--auto-min-tokens","0"),num("--auto-min-mean","0")};
    auto device=choose_device(get("--device","A750"));bool in_order=bit("--in-order","0");
    sycl::property_list props=in_order?sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}:sycl::property_list{sycl::property::queue::enable_profiling{}};
    sycl::queue q(device,[](sycl::exception_list es){for(auto e:es)std::rethrow_exception(e);},props);
    std::cout<<std::setprecision(12)<<"device="<<device.get_info<sycl::info::device::name>()<<"\ndriver="<<device.get_info<sycl::info::device::driver_version>()<<'\n';
    if(bit("--probe-only","0")){run_s2s8_probe(q);grouping_probe(q);quantizer_probe(q);return 0;}
    if(!bit("--skip-probe","0")){run_s2s8_probe(q);grouping_probe(q);quantizer_probe(q);}
    bool real=a.count("--gate")!=0;require(real==bool(a.count("--up"))&&real==bool(a.count("--down")),"gate/up/down must be supplied together");
    Shape s{num("--k","2048"),num("--hidden","512"),num("--experts","256")};std::vector<uint8_t>wg,wu,wd;
    if(real){auto c=load_capsule(a.at("--gate"));s=c.shape;wg=std::move(c.bytes);auto u=load_capsule(a.at("--up")),d=load_capsule(a.at("--down"));
        require(u.shape.k==s.k&&u.shape.m==s.m&&u.shape.experts==s.experts&&d.shape.k==s.m&&d.shape.m==s.k&&d.shape.experts==s.experts,"capsule dimensions mismatch");wu=std::move(u.bytes);wd=std::move(d.bytes);}
    else{wg=synth(s,1234);wu=synth(s,1235);wd=synth({s.m,s.k,s.experts},1236);}
    require(topk<=s.experts&&s.experts<=256,"invalid expert count");Shape ds{s.m,s.k,s.experts};
    size_t jobs=checked_mul(tokens,topk),xn=checked_mul(tokens,s.k),hn=checked_mul(jobs,s.m),dn=checked_mul(jobs,s.k);
    validate_grouping_size(jobs,s.experts);
    Options go{num("--gate-split","1"),num("--local","4")},dopt{num("--down-split","1"),go.local_size};
    for(auto pair:std::array<std::pair<uint32_t,Options>,2>{{{s.k,go},{s.m,dopt}}})
        require((pair.second.split_k==1||pair.second.split_k==2||pair.second.split_k==4||pair.second.split_k==8)&&
            (pair.first/256)%pair.second.split_k==0&&go.local_size>=1&&go.local_size<=32,"invalid split/local");
    const std::string pattern=get("--pattern","normal"),idmode=get("--id-mode","rotate");
    require(pattern=="normal"||pattern=="zeros"||pattern=="outliers","bad pattern");require(idmode=="rotate"||idmode=="fixed","bad id mode");
    std::mt19937 rng(98765);std::normal_distribution<float> normal(0,.5f);std::vector<float>x(xn),routes(jobs);
    if(a.count("--x-file"))x=read_raw<float>(a.at("--x-file"),xn);else for(auto&v:x)v=pattern=="zeros"?0.f:normal(rng);
    if(!a.count("--x-file")&&pattern=="outliers")for(size_t i=0;i<xn;i+=257)x[i]=16.f;
    for(auto v:x)require(std::isfinite(v)&&std::abs(v)<=65504.f,"nonfinite/out-of-range activation");
    if(a.count("--routes-file"))routes=read_raw<float>(a.at("--routes-file"),jobs);
    else for(uint32_t t=0;t<tokens;++t){float sum=0;for(uint32_t j=0;j<topk;++j){float v=float(j+1);routes[size_t(t)*topk+j]=v;sum+=v;}
        for(uint32_t j=0;j<topk;++j)routes[size_t(t)*topk+j]/=sum;}
    validate_routes(routes,tokens,topk);
    uint32_t rounds=1+warmup+repeats;std::vector<int32_t>ids(checked_mul(rounds,jobs)),fixed;
    if(a.count("--ids-file"))fixed=read_raw<int32_t>(a.at("--ids-file"),jobs);
    for(uint32_t r=0;r<rounds;++r){if(!fixed.empty())std::copy(fixed.begin(),fixed.end(),ids.begin()+size_t(r)*jobs);
        else if(idmode=="fixed"&&r)std::copy(ids.begin(),ids.begin()+jobs,ids.begin()+size_t(r)*jobs);
        else for(uint32_t t=0;t<tokens;++t){std::vector<int32_t>pool(s.experts);std::iota(pool.begin(),pool.end(),0);
            if(r==0)std::rotate(pool.begin(),pool.end()-1,pool.end());else std::shuffle(pool.begin(),pool.end(),rng);
            std::copy(pool.begin(),pool.begin()+topk,ids.begin()+size_t(r)*jobs+size_t(t)*topk);}}
    for(auto id:ids)require(id>=0&&uint32_t(id)<s.experts,"invalid expert ID in benchmark input");
    const std::filesystem::path out=get("--out","build/results/architecture-manual");
    require(!std::filesystem::exists(out/"manifest.txt"),"evidence exists; use a new output directory");std::filesystem::create_directories(out);
    std::ofstream manifest(out/"manifest.txt"),numerical(out/"numerical.csv"),mapping(out/"mapping_validation.csv"),pairout(out/"pair_correctness.csv"),hist(out/"routing_histogram.csv"),samples(out/"samples.csv"),stages(out/"stages.csv");
    require(manifest&&numerical&&mapping&&pairout&&hist&&samples&&stages,"cannot create evidence files");
    manifest<<"version=0.4-vulkan-port-v1\nDPAS_repeat=1\nbase_contract=RNE_G32_original_v04\n";
    manifest<<"candidate="<<candidate<<"\noverlap_requested="<<overlap<<"\ngate_tile_requested="<<gate_tile
        <<"\ndown_tile_requested="<<down_tile<<"\nauto_min_tokens="<<policy.min_tokens
        <<"\nauto_min_mean="<<policy.min_mean_assignments<<"\nfreeze_input="<<freeze
        <<"\nfreeze_warning=diagnostic_only_not_comparable_production_timing\n";
    manifest<<"device="<<device.get_info<sycl::info::device::name>()<<"\ndriver="<<device.get_info<sycl::info::device::driver_version>()<<'\n';
    for(auto&kv:a)manifest<<kv.first<<'='<<kv.second<<'\n';
    manifest<<"weight_source="<<(real?"real_capsules":"synthetic")<<"\nactivation_source="<<(a.count("--x-file")?"captured":"synthetic")
        <<"\nids_source="<<(fixed.empty()?idmode:"captured_fixed")<<"\nroute_source="<<(a.count("--routes-file")?"captured":"synthetic_normalized")<<'\n';
    manifest<<"Q="<<tokens<<"\nk="<<s.k<<"\nhidden="<<s.m<<"\nexperts="<<s.experts<<"\ntopk="<<topk
        <<"\nlayout=s2tile8\ninput_group=32\nhidden_group=32\ngluquant=1\nasync=1\nin_order="<<in_order
        <<"\ngate_split="<<go.split_k<<"\ndown_split="<<dopt.split_k<<"\nlocal="<<go.local_size
        <<"\ngroup_workgroup=128\nseed=98765\nrepeats="<<repeats<<"\nwarmup="<<warmup<<"\n";
    manifest<<"scope=one_MoE_layer_no_FA_router_norm_residual_sampling_Vulkan_handoff\n"
        <<"order=alternate_AB_BA_with_identical_ID_frame_for_both_arms\n"
        <<"group_plan_rebuilt_every_call=1\nCPU_reference=up_to_4_tokens_including_token12_per_snapshot_all_topk_outputs\n"
        <<"full_GPU_AB_comparison=all_tokens_all_intermediates_all_quant_planes_two_snapshots\n"
        <<"first_timed=post_numerical_validation_NOT_cold_JIT\n";
    hist<<"round,phase,expert,selected_jobs\n";
    for(uint32_t r=0;r<rounds;++r){std::vector<int32_t>frame(ids.begin()+size_t(r)*jobs,ids.begin()+size_t(r+1)*jobs);
        auto plan=group_jobs_reference(frame,s.experts);for(uint32_t e=0;e<=s.experts;++e)
            hist<<r<<','<<(r==0?"first":r<=warmup?"warmup":"repeat")<<','<<e<<','<<plan.counts[e]<<'\n';}
    hist.flush();
    auto sg=repack_s2tile8(wg,s),su=repack_s2tile8(wu,s),sd=repack_s2tile8(wd,ds);
    DeviceMem<uint8_t>dg(q,wg.size()),du(q,wu.size()),dd(q,wd.size()),scrub(q,size_t(scrub_mib)*1024*1024);
    DeviceMem<float>dx(q,xn),dr(q,jobs),dy(q,xn),yg(q,hn),yu(q,hn),hidden(q,hn),yd(q,dn),
        gs(q,go.split_k>1?size_t(go.split_k)*hn*2:0),dscratch(q,dopt.split_k>1?size_t(dopt.split_k)*dn:0);
    DeviceMem<int32_t>di(q,ids.size()),gst(q,jobs),dst(q,jobs),hi(q,hn),oi(q,xn),ai0(q,xn/32),ai1(q,hn/32),
        gc(q,s.experts+1),gof(q,s.experts+2),order(q,jobs);
    DeviceMem<int8_t>aq0(q,xn),aq1(q,hn);DeviceMem<float>as0(q,xn/32),as1(q,hn/32);
    const size_t gtcap=expert_token_tile_capacity(jobs,s.experts,gate_tile),dtcap=expert_token_tile_capacity(jobs,s.experts,down_tile);
    DeviceMem<int32_t>gtfirst(q,gtcap),gtlen(q,gtcap),gtexpert(q,gtcap),gttotal(q,1),
        dtfirst(q,dtcap),dtlen(q,dtcap),dtexpert(q,dtcap),dttotal(q,1);
    MoeProblem p{s,tokens,topk,Layout::s2tile8,dg.p,du.p,dd.p,dx.p,dr.p,di.p,dy.p};
    MoeWorkspace w{yg.p,yu.p,hidden.p,yd.p,gs.p,dscratch.p,{aq0.p,as0.p,ai0.p,false},{aq1.p,as1.p,ai1.p,false},gst.p,dst.p,hi.p,oi.p,
        {gc.p,gof.p,order.p,jobs,s.experts},
        {gtfirst.p,gtlen.p,gtexpert.p,gttotal.p,gtcap,gate_tile},
        {dtfirst.p,dtlen.p,dtexpert.p,dttotal.p,dtcap,down_tile}};
    q.memcpy(dg.p,sg.data(),wg.size());q.memcpy(du.p,su.data(),wu.size());q.memcpy(dd.p,sd.data(),wd.size());
    q.memcpy(dx.p,x.data(),xn*4);q.memcpy(dr.p,routes.data(),jobs*4);q.memcpy(di.p,ids.data(),ids.size()*4);
    q.memset(gst.p,0,jobs*4);q.memset(dst.p,0,jobs*4);q.wait_and_throw();
    manifest<<"weight_resident_bytes="<<wg.size()+wu.size()+wd.size()<<"\ngrouping_extra_bytes="<<(jobs+2*s.experts+3)*4
        <<"\nextra_tile_descriptor_bytes="<<(gtcap+dtcap)*12+8<<"\nexplicit_resident_bytes="<<wg.size()+wu.size()+wd.size()+scrub.count+
        (xn*2+jobs+hn*3+dn+gs.count+dscratch.count+as0.count+as1.count)*4+
        (ids.size()+jobs*2+hn+xn+ai0.count+ai1.count+gc.count+gof.count+order.count)*4+xn+hn+(gtcap+dtcap)*12+8<<'\n';
    manifest.flush();
    std::vector<ChainVariant>vars;
    MoeOptions base;base.path=MoePath::a8_gluquant;base.gate_kernel=go;base.down_kernel=dopt;
    base.schedule=MoeSchedule::direct;base.input_prequantized=freeze;
    vars.push_back({"baseline",base});
    MoeOptions opt=base;opt.schedule=candidate=="grouped"?MoeSchedule::grouped:candidate=="direct"?MoeSchedule::direct:MoeSchedule::auto_select;
    opt.auto_policy=policy;opt.overlap_grouping_quant=overlap;opt.gate_tokens_per_tile=gate_tile;opt.down_tokens_per_tile=down_tile;
    vars.push_back({"candidate",opt});
    const auto selected=choose_moe_schedule(opt.schedule,false,tokens,topk,s.experts,true,policy);
    manifest<<"actual_grouped="<<selected.grouped<<"\ndispatch_reason="<<selected.reason<<"\n";
    manifest.flush();
    if(overlap&&in_order)std::cout<<"NOTE: in-order queue serializes fork/join; overlap is not promised.\n";
    if(freeze){const auto fq=quantize_a8(x,tokens,s.k,32);
        q.memcpy(aq0.p,fq.q.data(),fq.q.size());q.memcpy(as0.p,fq.scale.data(),fq.scale.size()*4);
        q.memset(ai0.p,0,ai0.count*4);q.wait_and_throw();}
    std::vector<uint32_t>ts={0,std::min(12u,tokens-1),tokens/2,tokens-1};std::sort(ts.begin(),ts.end());ts.erase(std::unique(ts.begin(),ts.end()),ts.end());
    numerical<<std::setprecision(12)<<"snapshot,variant,cpu_sample_tokens,gate_nmse,up_nmse,swiglu_nmse,down_nmse,weighted_sum_nmse,chain_sample_nmse_vs_f32,quant_pass,kernel_pass\n";
    mapping<<"snapshot,counts_equal,offsets_equal,permutation_equal,invalid_jobs,pass\n";
    pairout<<std::setprecision(12)<<"snapshot,stage,nmse,max_abs_over_rms,bitwise_equal,pass\n";
    for(uint32_t snap:{0u,rounds-1}) {
        std::cout<<"Numerical snapshot "<<snap<<" Q="<<tokens<<" (full GPU A/B, CPU tokens="<<ts.size()<<")\n";
        p.ids=di.p+size_t(snap)*jobs;
        std::vector<int32_t>frame(ids.begin()+size_t(snap)*jobs,ids.begin()+size_t(snap+1)*jobs);
        auto sid=pick_tokens(frame,ts,topk);auto sx=pick_tokens(x,ts,s.k),sr=pick_tokens(routes,ts,topk);
        const uint32_t nq=uint32_t(ts.size());
        auto fg=reference(wg,s,sx,sid,nq,topk,false,false),fu=reference(wu,s,sx,sid,nq,topk,false,false);
        auto fh=swiglu_reference(fg,fu),fd=reference(wd,ds,fh,sid,nq,topk,true,false),fy=weighted_sum_reference(fd,sr,nq,topk,s.k);
        Snapshot baseline;
        for(size_t vi=0;vi<vars.size();++vi){auto&v=vars[vi];
            // Poison outputs outside timing: missing/duplicate remaps must not reuse old values.
            q.memset(yg.p,255,hn*4);q.memset(yu.p,255,hn*4);q.memset(hidden.p,255,hn*4);q.memset(yd.p,255,dn*4);q.memset(dy.p,255,xn*4);q.wait_and_throw();
            auto run=enqueue_moe(q,p,v.opt,w);run.done.wait_and_throw();auto got=capture_snapshot(q,p,w);
            auto qi_audit=audit_a8(got.input_q,x),qh_audit=audit_a8(got.hidden_q,got.hidden);
            std::cout<<"QUANT_CONTRACT "<<v.name<<" input_code_diff="<<qi_audit.cpu_code_differences
                <<" scale_diff="<<qi_audit.cpu_scale_differences<<" scale_max_ulp="<<qi_audit.max_scale_ulp
                <<" boundary="<<qi_audit.boundary_codes<<" bad_codes="<<qi_audit.bad_codes
                <<" hidden_bad_codes="<<qh_audit.bad_codes<<"\n";
            {
                const auto dest=out/("snapshot-"+std::to_string(snap)+"-"+v.name);
                std::filesystem::create_directories(dest);
                std::ofstream audit(dest/"quant_audit.csv");
                audit<<"plane,codes,groups,cpu_code_diff,cpu_scale_diff,max_scale_ulp,max_scale_relative,boundary_codes,bad_codes,bad_scales,pass\n";
                for(auto item:std::array<std::pair<const char*,A8Audit>,2>{{{"input",qi_audit},{"hidden",qh_audit}}}) {
                    const auto& a=item.second;audit<<item.first<<','<<a.codes<<','<<a.groups<<','<<a.cpu_code_differences<<','<<a.cpu_scale_differences
                        <<','<<a.max_scale_ulp<<','<<a.max_scale_relative<<','<<a.boundary_codes<<','<<a.bad_codes<<','<<a.bad_scales<<','<<a.pass<<'\n';}
                if(dump) {
                    dump_raw(dest/"input.f32",x);dump_raw(dest/"hidden.f32",got.hidden);
                    dump_raw(dest/"input.q8",got.input_q.q);dump_raw(dest/"input.scales.f32",got.input_q.scale);
                    dump_raw(dest/"hidden.q8",got.hidden_q.q);dump_raw(dest/"hidden.scales.f32",got.hidden_q.scale);
                    dump_raw(dest/"gate.f32",got.gate);dump_raw(dest/"up.f32",got.up);dump_raw(dest/"down.f32",got.down);dump_raw(dest/"output.f32",got.y);
                    dump_raw(dest/"ids.i32",frame);dump_raw(dest/"routes.f32",routes);
                    const auto cq=quantize_a8(x,tokens,s.k,32);
                    dump_raw(dest/"input.cpu.q8",cq.q);dump_raw(dest/"input.cpu.scales.f32",cq.scale);
                    std::ofstream meta(dest/"shape.txt");meta<<"tokens="<<tokens<<"\nk="<<s.k<<"\nhidden="<<s.m<<"\ntopk="<<topk<<"\ngroup=32\nrounding=RNE\n";
                }
            }
            if(run.dispatch.grouped){auto actual=capture_mapping(q,w.grouping,jobs,s.experts),ref=group_jobs_reference(frame,s.experts);
                bool ok=same_grouping(actual,ref);mapping<<snap<<','<<(actual.counts==ref.counts)<<','<<(actual.offsets==ref.offsets)<<','<<(actual.sorted_to_job==ref.sorted_to_job)<<','<<actual.counts.back()<<','<<ok<<'\n';mapping.flush();require(ok,"group mapping mismatch");}
            if(run.dispatch.grouped) {
                for(auto tv:std::array<ExpertTokenTileView,2>{{w.gate_tiles,gate_tile==down_tile?w.gate_tiles:w.down_tiles}}) {
                    if(tv.tokens_per_tile==1)continue;
                    auto tr=token_tiles_reference(group_jobs_reference(frame,s.experts),tv.tokens_per_tile);
                    const auto total=copy_back(q,tv.total,1)[0];require(total==int32_t(tr.first.size()),"tile count mismatch");
                    require(copy_back(q,tv.first,tr.first.size())==tr.first&&copy_back(q,tv.length,tr.length.size())==tr.length&&
                        copy_back(q,tv.expert,tr.expert.size())==tr.expert,"tile descriptor mismatch");
                }
            }
            auto g=pick_tokens(got.gate,ts,size_t(topk)*s.m),u=pick_tokens(got.up,ts,size_t(topk)*s.m),h=pick_tokens(got.hidden,ts,size_t(topk)*s.m),
                d=pick_tokens(got.down,ts,size_t(topk)*s.k),y=pick_tokens(got.y,ts,s.k);
            auto qi=pick_quant(got.input_q,ts,1),qh=pick_quant(got.hidden_q,ts,topk);
            bool qp=quant_ok(got.input_q,x)&&quant_ok(got.hidden_q,got.hidden);
            auto gg=reference_a8(wg,s,qi,sid,nq,topk,false),gu=reference_a8(wu,s,qi,sid,nq,topk,false),
                gh=swiglu_reference(g,u),gd=reference_a8(wd,ds,qh,sid,nq,topk,true),gy=weighted_sum_reference(d,sr,nq,topk,s.k);
            auto eg=errors(g,gg),eu=errors(u,gu),eh=errors(h,gh),ed=errors(d,gd),ey=errors(y,gy),et=errors(y,fy);
            bool ok=qp&&good(eg)&&good(eu)&&good(eh)&&good(ed)&&good(ey)&&et.finite;
            v.pass&=ok;v.total_nmse=std::max(v.total_nmse,et.nmse);
            numerical<<snap<<','<<v.name<<','<<nq<<','<<eg.nmse<<','<<eu.nmse<<','<<eh.nmse<<','<<ed.nmse<<','<<ey.nmse<<','<<et.nmse<<','<<qp<<','<<ok<<'\n';numerical.flush();require(ok,"CPU arithmetic/quantizer check failed");
            if(vi==0)baseline=std::move(got);
            else {
                for(auto item:std::array<std::pair<const char*,std::pair<const std::vector<float>*,const std::vector<float>*>>,5>{{
                    {"gate",{&got.gate,&baseline.gate}},{"up",{&got.up,&baseline.up}},{"hidden",{&got.hidden,&baseline.hidden}},
                    {"down",{&got.down,&baseline.down}},{"weighted_sum",{&got.y,&baseline.y}}}}) {
                    auto e=errors(*item.second.first,*item.second.second);bool pass=near_scheduling_equal(e);
                    pairout<<snap<<','<<item.first<<','<<e.nmse<<','<<e.max_abs_over_rms<<','<<bits_equal(*item.second.first,*item.second.second)<<','<<pass<<'\n';pairout.flush();require(pass,"full GPU A/B output mismatch");}
                bool exact=bits_equal(got.input_q.q,baseline.input_q.q)&&bits_equal(got.input_q.scale,baseline.input_q.scale)&&
                    bits_equal(got.hidden_q.q,baseline.hidden_q.q)&&bits_equal(got.hidden_q.scale,baseline.hidden_q.scale);
                pairout<<snap<<",all_quant_planes,0,0,"<<exact<<','<<exact<<'\n';pairout.flush();require(exact,"grouping altered quantized q/scale planes");
            }
        }
    }
    samples<<std::setprecision(12)<<"round,phase,order,variant,enqueue_return_us,tail_wait_us,wall_us,gpu_span_us,kernel_sum_us,gpu_gap_us,group_kernel_us,core_gpu_span_us,gate_up_us,down_us,union_busy_us,overlap_us,actual_grouped,dispatch_reason\n";
    stages<<std::setprecision(12)<<"round,phase,variant,stage,start_ns,end_ns,duration_us,parent_mask\n";
    // Ring contains different ID frames; identical frame per paired round. Grouping is not amortized.
    for(uint32_t r=0;r<rounds;++r){const char*phase=r==0?"first_timed_post_validation":r<=warmup?"warmup":"repeat";
        p.ids=di.p+size_t(r)*jobs;
        for(size_t j=0;j<2;++j){auto&v=vars[(r+j)%2];
            if(scrub.p){const size_t n=scrub.count/4;auto*b=reinterpret_cast<uint32_t*>(scrub.p);q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>i){b[i[0]]=uint32_t(i[0])+r;}).wait_and_throw();}
            auto t0=Clock::now();auto run=enqueue_moe(q,p,v.opt,w);auto tr=Clock::now();run.done.wait_and_throw();auto t1=Clock::now();
            ChainSample sm;sm.enqueue_return=micros(t0,tr);sm.tail_wait=micros(tr,t1);sm.wall=micros(t0,t1);
            uint64_t core_begin=UINT64_MAX;std::vector<StageInterval> intervals;
            for(size_t si=0;si<run.count;++si){auto&st=run.stages[si];auto b=st.event.get_profiling_info<sycl::info::event_profiling::command_start>(),e=st.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                intervals.push_back({b,e,st.parents});double us=double(e-b)/1000.;
                const std::string label=st.label;if(label.rfind("group_",0)==0)sm.group_us+=us;else core_begin=std::min(core_begin,b);
                if(label=="gate_up")sm.gate_us=us;if(label=="down")sm.down_us=us;
                stages<<r<<','<<phase<<','<<v.name<<','<<label<<','<<b<<','<<e<<','<<us<<','<<st.parents<<'\n';}
            const auto profile=profile_intervals(intervals);
            sm.gpu_span=double(profile.end-profile.begin)/1000.;sm.kernel_sum=double(profile.kernel_sum)/1000.;
            sm.gpu_gap=double(profile.idle)/1000.;sm.core_span=double(profile.end-core_begin)/1000.;
            sm.union_busy=double(profile.union_busy)/1000.;sm.overlap=double(profile.overlap)/1000.;
            samples<<r<<','<<phase<<','<<j<<','<<v.name<<','<<sm.enqueue_return<<','<<sm.tail_wait<<','<<sm.wall<<','<<sm.gpu_span<<','<<sm.kernel_sum<<','<<sm.gpu_gap<<','<<sm.group_us<<','<<sm.core_span<<','<<sm.gate_us<<','<<sm.down_us<<','<<sm.union_busy<<','<<sm.overlap<<','<<run.dispatch.grouped<<','<<run.dispatch.reason<<'\n';
            if(r>warmup)v.samples.push_back(sm);
        }
    }
    // Final routing was rebuilt inside the LAST grouped call, not in an untimed prepass.
    std::vector<int32_t>last(ids.end()-jobs,ids.end());if(selected.grouped)require(same_grouping(capture_mapping(q,w.grouping,jobs,s.experts),group_jobs_reference(last,s.experts)),"final timed mapping mismatch");
    require(all_zero(copy_back(q,gst.p,jobs))&&all_zero(copy_back(q,dst.p,jobs))&&all_zero(copy_back(q,oi.p,xn)),"timed run device status failure");
    std::ofstream summary(out/"summary.csv");require(bool(summary),"cannot create summary");
    summary<<std::setprecision(12)<<"variant,layout,input_group,hidden_group,host_waits,gate_split,down_split,local_size,expert_grouping,group_kernel_median_us,core_gpu_span_median_us,gate_up_median_us,down_median_us,kernel_sum_median_us,gpu_span_median_us,gpu_gap_median_us,enqueue_return_median_us,tail_wait_median_us,wall_median_us,wall_p95_us,speedup_vs_ungrouped_same_build,kernel_pass,pair_pass,chain_sample_nmse_vs_f32,tokens,k,hidden,experts,topk,source,union_busy_median_us,overlap_median_us,candidate,gate_tile,down_tile,fork_join,freeze_input,in_order\n";
    for(auto&v:vars){const double wall=value(v,&ChainSample::wall),speed=value(vars[0],&ChainSample::wall)/wall;
        summary<<v.name<<",s2tile8,32,32,0,"<<go.split_k<<','<<dopt.split_k<<','<<go.local_size<<','<<(v.name=="candidate"&&selected.grouped)<<','
            <<value(v,&ChainSample::group_us)<<','<<value(v,&ChainSample::core_span)<<','<<value(v,&ChainSample::gate_us)<<','<<value(v,&ChainSample::down_us)<<','
            <<value(v,&ChainSample::kernel_sum)<<','<<value(v,&ChainSample::gpu_span)<<','<<value(v,&ChainSample::gpu_gap)<<','<<value(v,&ChainSample::enqueue_return)<<','
            <<value(v,&ChainSample::tail_wait)<<','<<wall<<','<<value(v,&ChainSample::wall,.95)<<','<<speed<<','<<v.pass<<",1,"<<v.total_nmse<<','<<tokens<<','<<s.k<<','<<s.m<<','<<s.experts<<','<<topk<<','<<(real?"real_weights":"synthetic")<<','<<value(v,&ChainSample::union_busy)<<','<<value(v,&ChainSample::overlap)<<','<<candidate<<','<<(v.name=="candidate"?gate_tile:1)<<','<<(v.name=="candidate"?down_tile:1)<<','<<v.opt.overlap_grouping_quant<<','<<freeze<<','<<in_order<<'\n';
        std::cout<<v.name<<" wall_median_us="<<wall<<" p95="<<value(v,&ChainSample::wall,.95)<<" group_kernel_us="<<value(v,&ChainSample::group_us)<<" speedup="<<speed<<" PASS\n";}
    std::cout<<"RESULT=PASS output="<<out.string()<<"\nGrouping INCLUDED each enabled call. Reuse keeps DPAS RepeatCount=1. No Vulkan handoff or server TG. Freeze-input timings are diagnostic only.\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
