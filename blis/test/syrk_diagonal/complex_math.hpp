/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Southern Methodist University

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <cmath>
#include <algorithm>
#include <type_traits>

#include "blis.h"

template <typename T>
struct is_complex : std::false_type {};

template <>
struct is_complex<scomplex> : std::true_type {};

template <>
struct is_complex<dcomplex> : std::true_type {};

template <typename T>
struct is_real : std::integral_constant<bool,!is_complex<T>::value> {};

template <typename T> struct make_complex;

template <> struct make_complex<float   > { using type = scomplex; };
template <> struct make_complex<double  > { using type = dcomplex; };
template <> struct make_complex<scomplex> { using type = scomplex; };
template <> struct make_complex<dcomplex> { using type = dcomplex; };

template <typename T>
using make_complex_t = typename make_complex<T>::type;

template <typename T> struct make_real;

template <> struct make_real<float   > { using type = float; };
template <> struct make_real<double  > { using type = double; };
template <> struct make_real<scomplex> { using type = float; };
template <> struct make_real<dcomplex> { using type = double; };

template <typename T>
using make_real_t = typename make_real<T>::type;

template <typename T, bool Cond>
struct make_complex_if : std::conditional<Cond,make_complex_t<T>,make_real_t<T>> {};

template <typename T, bool Cond>
using make_complex_if_t = typename make_complex_if<T,Cond>::type;

template <typename T>
struct real_imag_part
{
    real_imag_part& operator=(T) { return *this; }

    operator T() const { return T(); }
};

template <typename T>
std::enable_if_t<std::is_arithmetic<typename std::remove_cv<T>::type>::value,T&> real(T& x) { return x; }

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value,real_imag_part<T>> imag(T x) { return {}; }

inline float& real(scomplex& x) { return x.real; }

inline float& imag(scomplex& x) { return x.imag; }

inline double& real(dcomplex& x) { return x.real; }

inline double& imag(dcomplex& x) { return x.imag; }

inline const float& real(const scomplex& x) { return x.real; }

inline const float& imag(const scomplex& x) { return x.imag; }

inline const double& real(const dcomplex& x) { return x.real; }

inline const double& imag(const dcomplex& x) { return x.imag; }

template <typename T>
std::enable_if_t<is_real<T>::value,T> conj(T x) { return x; }

template <typename T>
std::enable_if_t<is_complex<T>::value,T> conj(const T& x) { return {x.real, -x.imag}; }

template <typename T, typename U, typename=void>
struct convert_impl;

template <typename T, typename U>
struct convert_impl<T, U, std::enable_if_t<is_real<T>::value && is_real<U>::value>>
{
    void operator()(T x, U& y) const { y = x; }
};

template <typename T, typename U>
struct convert_impl<T, U, std::enable_if_t<is_real<T>::value && is_complex<U>::value>>
{
    void operator()(T x, U& y) const { y.real = x; y.imag = 0; }
};

template <typename T, typename U>
struct convert_impl<T, U, std::enable_if_t<is_complex<T>::value && is_real<U>::value>>
{
    void operator()(T x, U& y) const { y = x.real; }
};

template <typename T, typename U>
struct convert_impl<T, U, std::enable_if_t<is_complex<T>::value && is_complex<U>::value>>
{
    void operator()(T x, U& y) const { y.real = x.real; y.imag = x.imag; }
};

template <typename U, typename T>
U convert(T x)
{
    U y;
    convert_impl<T,U>{}(x,y);
    return y;
}

template <typename U, typename T>
auto convert_prec(T x) -> make_complex_if_t<U,is_complex<T>::value>
{
    return convert<make_complex_if_t<U,is_complex<T>::value>>(x);
}

#define COMPLEX_MATH_OPS(rtype, ctype) \
\
inline bool operator==(rtype x, ctype y) \
{ \
    return x == y.real && y.imag == 0; \
} \
\
inline bool operator==(ctype x, rtype y) \
{ \
    return y == x.real && x.imag == 0; \
} \
\
inline bool operator==(ctype x, ctype y) \
{ \
    return x.real == y.real && \
           x.imag == y.imag; \
 } \
 \
inline ctype operator-(ctype x) \
{ \
    return {-x.real, -x.imag}; \
} \
\
inline ctype operator+(rtype x, ctype y) \
{ \
    return {x+y.real, y.imag}; \
} \
\
inline ctype operator+(ctype x, rtype y) \
{ \
    return {y+x.real, x.imag}; \
} \
\
inline ctype operator+(ctype x, ctype y) \
{ \
    return {x.real+y.real, x.imag+y.imag}; \
} \
\
inline ctype operator-(rtype x, ctype y) \
{ \
    return {x-y.real, -y.imag}; \
} \
\
inline ctype operator-(ctype x, rtype y) \
{ \
    return {x.real-y, x.imag}; \
} \
\
inline ctype operator-(ctype x, ctype y) \
{ \
    return {x.real-y.real, x.imag-y.imag}; \
} \
\
inline ctype operator*(rtype x, ctype y) \
{ \
    return {x*y.real, x*y.imag}; \
} \
\
inline ctype operator*(ctype x, rtype y) \
{ \
    return {y*x.real, y*x.imag}; \
} \
\
inline ctype operator*(ctype x, ctype y) \
{ \
    return {x.real*y.real - x.imag*y.imag, \
            x.real*y.imag + x.imag*y.real}; \
} \
\
inline ctype operator/(rtype x, ctype y) \
{ \
    auto scale = std::max(std::abs(y.real), std::abs(y.imag)); \
    auto n = std::ilogb(scale); \
    auto yrs = std::scalbn(y.real, -n); \
    auto yis = std::scalbn(y.imag, -n); \
    auto denom = y.real*yrs + y.imag*yis; \
    return {x*yrs/denom, -x*yis/denom}; \
} \
\
inline ctype operator/(ctype x, rtype y) \
{ \
    return {x.real/y, x.imag/y}; \
} \
\
inline ctype operator/(ctype x, ctype y) \
{ \
    auto scale = std::max(std::abs(y.real), std::abs(y.imag)); \
    auto n = std::ilogb(scale); \
    auto yrs = std::scalbn(y.real, -n); \
    auto yis = std::scalbn(y.imag, -n); \
    auto denom = y.real*yrs + y.imag*yis; \
    return {(x.real*yrs + x.imag*yis)/denom, \
            (x.imag*yrs - x.real*yis)/denom}; \
} \
\
inline ctype& operator+=(ctype& x, rtype y) \
{ \
    x.real += y; \
    return x; \
} \
\
inline ctype& operator+=(ctype& x, ctype y) \
{ \
    x.real += y.real; x.imag += y.imag; \
    return x; \
} \
\
inline ctype& operator-=(ctype& x, rtype y) \
{ \
    x.real -= y; \
    return x; \
} \
\
inline ctype& operator-=(ctype& x, ctype y) \
{ \
    x.real -= y.real; x.imag -= y.imag; \
    return x; \
} \
\
inline ctype& operator*=(ctype& x, rtype y) \
{ \
    x.real *= y; x.imag *= y; \
    return x; \
} \
\
inline ctype& operator*=(ctype& x, ctype y) \
{ \
    x = x * y; \
    return x; \
} \
\
inline ctype& operator/=(ctype& x, rtype y) \
{ \
    x.real /= y; x.imag /= y; \
    return x; \
} \
\
inline ctype& operator/=(ctype& x, ctype y) \
{ \
    x = x / y; \
    return x; \
}

COMPLEX_MATH_OPS(float,  scomplex);
COMPLEX_MATH_OPS(double, dcomplex);

