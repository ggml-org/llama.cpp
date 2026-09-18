// SPDX-License-Identifier: MIT
// HOST-ONLY TEST DOUBLE. Never on a production include path.
// Default: record submissions without executing lambdas. mock_execute enables
// serial scalar execution ONLY for the new token-reuse kernel tests. Collective
// placeholders do NOT emulate real subgroup/workgroup execution. No GPU claim.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "tq2_reference.hpp"
#define SYCL_ESIMD_KERNEL
#define SYCL_ESIMD_FUNCTION
namespace sycl {
namespace info {namespace device {struct name{};struct vendor_id{};struct driver_version{};}enum class device_type{gpu};namespace event_profiling{struct command_start{};struct command_end{};}}
namespace property {namespace queue {struct in_order{};struct enable_profiling{};}}
enum class backend{ext_oneapi_level_zero};
struct context{};
using exception_list=std::vector<std::exception_ptr>;
struct property_list {bool ordered=false;template<class... P>property_list(P...){ordered=(std::is_same_v<P,property::queue::in_order>||...);}};
enum class aspect {fp16};
struct device {
    template<class T> auto get_info()const {
        if constexpr(std::is_same_v<T,info::device::vendor_id>)return uint32_t(0x8086);else return std::string("MOCK A750");
    }
    backend get_backend()const{return backend::ext_oneapi_level_zero;}
    bool is_gpu()const{return true;}bool has(aspect)const{return true;}
    bool operator==(const device&)const{return true;}
};
struct platform {static std::vector<platform>get_platforms(){return {platform{}};}std::vector<device>get_devices(info::device_type)const{return {device{}};}};
inline int mock_waits=0;
inline bool mock_execute=false;
struct half {uint16_t bits=0;operator float()const{return maple_w2::half_to_float(bits);}};
struct event {
    int number=0;
    void wait_and_throw()const{++mock_waits;}
    template<class T>uint64_t get_profiling_info()const{return 0;}
};
template<int D>struct range {size_t n;explicit range(size_t x):n(x){}};
template<int D>struct id {size_t n=0;size_t operator[](size_t)const{return n;}};
template<int D>struct nd_range {range<D>global,local;nd_range(range<D>g,range<D>l):global(g),local(l){}};
struct group{};
template<int D>struct nd_item {
    size_t index=0,local=1;
    size_t get_group_linear_id()const{return index/local;}size_t get_local_linear_id()const{return index%local;}
    size_t get_global_linear_id()const{return index;}group get_group()const{return {};}
};
template<class T>using plus=std::plus<T>;
template<class T>struct maximum {T operator()(T a,T b)const{return std::max(a,b);}};
template<class T,class F>T reduce_over_group(group,T x,F){return x;}
template<class T,class F>T exclusive_scan_over_group(group,T,F){return T{};}
using std::fmin;using std::fmax;using std::fabs;using std::isfinite;using std::exp;using std::min;
struct handler {
    std::vector<event>deps;
    void depends_on(event e){deps.push_back(e);}
    void depends_on(const std::vector<event>&e){deps.insert(deps.end(),e.begin(),e.end());}
    template<class K,int D,class F>void parallel_for(range<D> r,F fn){if(mock_execute)for(size_t i=0;i<r.n;++i)fn(id<D>{i});}
    template<class K,int D,class F>void parallel_for(nd_range<D> r,F fn){if(mock_execute)for(size_t i=0;i<r.global.n;++i)fn(nd_item<D>{i,r.local.n});}
};
class queue {
public:
    bool ordered=false;std::vector<std::vector<event>> graph;
    explicit queue(bool b=false):ordered(b){}
    template<class F>queue(device,F,property_list p):ordered(p.ordered){}
    template<class F>queue(context,device,F,property_list p):ordered(p.ordered){}
    context get_context()const{return {};}
    void wait()const{++mock_waits;}void wait_and_throw()const{++mock_waits;}
    event memcpy(void*,const void*,size_t){return submit([](handler&){});}
    event memset(void*,int,size_t){return submit([](handler&){});}
    template<class T>event fill(T*,T,size_t){return submit([](handler&){});}
    template<class R,class F>event parallel_for(R r,F f){return submit([&](handler&h){h.parallel_for<void>(r,f);});}
    device get_device()const{return {};}
    template<class P>bool has_property()const{return ordered;}
    template<class F>event submit(F f){handler h;f(h);if(ordered&&!graph.empty())h.deps.push_back({int(graph.size())});graph.push_back(h.deps);return {int(graph.size())};}
};
template<class T>T*malloc_device(size_t n,queue&){return new T[n];}
inline void free(void*,queue&){} // parsing aid, NEVER a real allocator/free API
} // namespace sycl
