// SPDX-License-Identifier: MIT
#include "moe_reference.hpp"
#include <cfenv>
#include <iostream>
#include <random>
using namespace maple_w2;
static size_t checks=0;
static void expect(bool b,const char*msg){++checks;if(!b)throw std::runtime_error(msg);}
template<class F>static void rejects(F f,const char*msg){bool caught=false;try{f();}catch(const std::exception&){caught=true;}expect(caught,msg);}
static std::vector<uint8_t> weight(Shape s,unsigned seed){std::mt19937 r(seed);std::vector<uint8_t>w(nbytes(s),0);
 for(size_t b=0;b<w.size();b+=66){for(unsigned k=0;k<256;++k)set_native_code(w.data()+b,k,r()%3);put16(w.data()+b+64,float_to_half(float(1+r()%13)/512));}return w;}
int main()try{
 std::fesetround(FE_TONEAREST);
 for(float g:{-100.f,-9.f,-7.f,-1.f,0.f,1.f,7.f,9.f,100.f})for(float u:{-12.f,-7.f,-.1f,0.f,1.f,7.f,12.f}){
  double a=std::min(double(g),7.0),b=std::max(-7.,std::min(double(u),7.));double gold=(a/(1.+std::exp(-a)))*b;
  double got=maple_swiglu(g,u);expect(std::abs(got-gold)<1e-5*std::max(1.,std::abs(gold)),"clipped SwiGLU mismatch");}
 expect(maple_swiglu(-9,1)!=maple_swiglu(-7,1),"gate must not be lower-clipped");
 expect(maple_swiglu(9,1)==maple_swiglu(7,1),"gate upper clipping missing");
 rejects([]{maple_swiglu(std::numeric_limits<float>::quiet_NaN(),1);},"NaN hidden by clamp");
 rejects([]{maple_swiglu(1,1,0);},"invalid limit accepted");
 rejects([]{validate_routes({.3f,.3f},1,2);},"not-normalized route accepted");
 rejects([]{validate_routes({-1.f,2.f},1,2);},"negative route accepted");
 std::mt19937 r(3387);std::normal_distribution<float> n(0,8);
 for(unsigned rows:{1u,8u,32u})for(unsigned width:{256u,512u}){
  std::vector<float>g(size_t(rows)*width),u(g.size());for(size_t i=0;i<g.size();++i){g[i]=n(r);u[i]=n(r);}
  auto h=swiglu_reference(g,u);
  for(unsigned group:{32u,128u,256u}){auto q=quantize_a8(h,rows,width,group);
   // Independent serial GLU+quant reference with nearbyint for RNE.
   for(size_t ix=0;ix<q.scale.size();++ix){float mx=0;std::vector<float>v(group);
    for(unsigned j=0;j<group;++j){auto i=ix*group+j;float a=std::min(g[i],7.f),b=std::max(-7.f,std::min(u[i],7.f));
     float z=std::exp(-std::abs(a));v[j]=(a>=0?a/(1+z):(a*z)/(1+z))*b;mx=std::max(mx,std::abs(v[j]));}
    float d=mx==0?1:std::max(mx/127.f,std::numeric_limits<float>::min());expect(d==q.scale[ix],"fused scale mismatch");
    for(unsigned j=0;j<group;++j){int8_t a=int8_t(std::nearbyint(std::max(-127.f,std::min(127.f,v[j]/d))));expect(a==q.q[ix*group+j],"fused code mismatch");}
   }
  }
 }
 const Shape s{256,256,3};auto g=weight(s,1),u=weight(s,2),d=weight(s,3);
 std::vector<float>x(2*256);for(auto&v:x)v=n(r)*.03f;
 std::vector<int32_t>ids={2,0,1,2};std::vector<float>routes={.25f,.75f,.9f,.1f};
 for(unsigned group:{32u,128u,256u}){auto qi=quantize_a8(x,2,256,group);
  auto yg=reference_a8(g,s,qi,ids,2,2,false),yu=reference_a8(u,s,qi,ids,2,2,false);auto h=swiglu_reference(yg,yu);
  auto qh=quantize_a8(h,4,256,group);auto yd=reference_a8(d,s,qh,ids,2,2,true);auto y=weighted_sum_reference(yd,routes,2,2,256);
  expect(y.size()==512,"chain output shape");for(size_t i=0;i<y.size();++i){size_t t=i/256,k=i%256;
   double z=double(yd[(t*2)*256+k])*routes[t*2]+double(yd[(t*2+1)*256+k])*routes[t*2+1];expect(y[i]==float(z),"expert reduction mismatch");}
  auto repacked=repack_s2tile8(d,s);expect(unpack_s2tile8(repacked,s)==d,"chain layout mismatch");
 }
 std::cout<<"PASS: "<<checks<<" CPU checks for clipped SwiGLU, fused quant math, route validation, and MoE composition.\nNo SYCL/GPU execution in this test.\n";
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
