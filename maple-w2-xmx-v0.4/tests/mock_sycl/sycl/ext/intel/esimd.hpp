// SPDX-License-Identifier: MIT
// CPU scalar model for the NEW T2/T4 kernel only. Not Intel ESIMD compilation.
#pragma once
#include <array>
#include <cstring>
#include <sycl/sycl.hpp>
namespace sycl::ext::intel::esimd {
template<class T,int N>struct simd {
    std::array<T,N> v{};
    simd()=default;simd(T x){v.fill(x);}simd(T x,T step){for(int i=0;i<N;++i)v[i]=T(x+i*step);}
    T& operator[](int i){return v[i];}const T& operator[](int i)const{return v[i];}
    simd& operator=(T x){v.fill(x);return *this;}
    simd& operator+=(const simd& x){for(int i=0;i<N;++i)v[i]+=x.v[i];return *this;}
    template<class U>simd<U,N*sizeof(T)/sizeof(U)> bit_cast_view()const {
        simd<U,N*sizeof(T)/sizeof(U)> x;std::memcpy(x.v.data(),v.data(),N*sizeof(T));return x;
    }
};
template<class T,int N>simd<T,N>operator*(const simd<T,N>&a,T b){simd<T,N>r;for(int i=0;i<N;++i)r[i]=a[i]*b;return r;}
template<class T,int N>simd<T,N>operator*(const simd<T,N>&a,const simd<T,N>&b){simd<T,N>r;for(int i=0;i<N;++i)r[i]=a[i]*b[i];return r;}
template<class To,class T,int N>simd<To,N>convert(const simd<T,N>&a){simd<To,N>r;for(int i=0;i<N;++i)r[i]=To(a[i]);return r;}
template<class T,int N>simd<T,N>gather(const T*p,const simd<uint32_t,N>&o){
    simd<T,N>r;for(int i=0;i<N;++i)std::memcpy(&r[i],reinterpret_cast<const uint8_t*>(p)+o[i],sizeof(T));return r;}
template<class T,int N>void scatter(T*p,const simd<uint32_t,N>&o,const simd<T,N>&a){
    for(int i=0;i<N;++i)std::memcpy(reinterpret_cast<uint8_t*>(p)+o[i],&a[i],sizeof(T));}
} // namespace
