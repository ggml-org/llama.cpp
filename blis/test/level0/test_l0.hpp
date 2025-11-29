/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Southern Methodist University

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

#ifndef BLIS_UNIT_TESTING_H
#define BLIS_UNIT_TESTING_H

#include <exception>
#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>
#include <cassert>
#include <signal.h>
#include <type_traits>
#include <functional>

#include "blis.h"

using unit_test_t = std::function<void()>;

struct variable_printer_base
{
    virtual ~variable_printer_base() {}

    virtual void print() const = 0;
};

struct unit_test_failure : std::exception {};

struct unit_test_registrar
{
    std::vector<unit_test_t> tests;
    std::vector<const variable_printer_base*> vars;

    static const char* red()
    {
        #ifdef BLIS_OS_WINDOWS
        static std::string s = _isatty(_fileno(stdout)) ? "\e[0;31m" : "";
        #else
        static std::string s = isatty(fileno(stdout)) ? "\e[0;31m" : "";
        #endif
        return s.c_str();
    }

    static const char* green()
    {
        #ifdef BLIS_OS_WINDOWS
        static std::string s = _isatty(_fileno(stdout)) ? "\e[0;32m" : "";
        #else
        static std::string s = isatty(fileno(stdout)) ? "\e[0;32m" : "";
        #endif
        return s.c_str();
    }

    static const char* normal()
    {
        #ifdef BLIS_OS_WINDOWS
        static std::string s = _isatty(_fileno(stdout)) ? "\e[0m" : "";
        #else
        static std::string s = isatty(fileno(stdout)) ? "\e[0m" : "";
        #endif
        return s.c_str();
    }

    size_t register_test(unit_test_t test)
    {
        tests.push_back(test);
        return tests.size()-1;
    }

    bool run_tests()
    {
        auto failed = 0;
        auto total = 0;

        for (auto& test : tests)
        {
            try
            {
                test();
            }
            catch (const unit_test_failure&)
            {
                failed++;
            }

            total++;
        }

        printf("\n");
        printf("Total tests: %d\n", total);
        printf("%sPassed: %d (%.1f%%)%s\n", green(), total-failed, 100.0*(total-failed)/total, normal());
        if (failed)
            printf("%sFailed: %d (%.1f%%)%s\n\n", red(), failed, 100.0*failed/total, normal());

        return failed;
    }

    void push_var(const variable_printer_base* var)
    {
        vars.push_back(var);
    }

    void pop_var(const variable_printer_base* var)
    {
        assert(vars.back() == var);
        vars.pop_back();
    }

    [[noreturn]]
    void fail(const char* cond)
    {
        printf("%sFAILURE%s\n\n", red(), normal());

        for (auto& var : vars)
            var->print();

        printf("\nAssertion failed: %s\n\n", cond);

        signal(SIGTRAP, [](int) {});
        raise(SIGTRAP);

        throw unit_test_failure();
    }
};

unit_test_registrar& get_unit_test_registrar();

inline size_t register_unit_test(unit_test_t test)
{
    return get_unit_test_registrar().register_test(test);
}

template <typename T>
struct variable_printer : variable_printer_base
{
    const char* message{};
    T var{};

    variable_printer()
    {
        get_unit_test_registrar().push_var(this);
    }

    virtual ~variable_printer() override
    {
        get_unit_test_registrar().pop_var(this);
    }

    variable_printer& operator<<(const char* m)
    {
        message = m;
        return *this;
    }

    variable_printer& operator<<(const T& v)
    {
        var = v;
        return *this;
    }

    virtual void print() const final override
    {
        std::cout << message << var << std::endl;
    }
};

template <>
struct variable_printer<void> : variable_printer_base
{
    const char* message;

    variable_printer()
    {
        get_unit_test_registrar().push_var(this);
    }

    virtual ~variable_printer() override
    {
        get_unit_test_registrar().pop_var(this);
    }

    variable_printer& operator<<(const char* m)
    {
        message = m;
        return *this;
    }

    virtual void print() const final override
    {
        std::cout << message << std::endl;
    }
};

template <typename T>
struct variable_printer_helper
{
    using type = variable_printer<T>;

    template <typename U>
    variable_printer_helper<U> operator<<(U) const;

    variable_printer_helper<void> operator<<(const char*) const;
};

#define VARIABLE_PRINTER(...) typename decltype(variable_printer_helper<void>{} << __VA_ARGS__)::type

#define VAR_NAME_(line) variable_printer_##line
#define VAR_NAME(line) VAR_NAME_(line)

#define INFO_(id, ...) \
VARIABLE_PRINTER(__VA_ARGS__) VAR_NAME(id); \
VAR_NAME(id) << __VA_ARGS__;

#ifdef ENABLE_INFO
#define INFO(...) INFO_(__COUNTER__, __VA_ARGS__)
#else
#define INFO(...)
#endif

#define TEST_NAME_(line,name) unit_test_##name##_##line
#define TEST_NAME(line,name) TEST_NAME_(line,name)

#define TEST_ID_(line,name) unit_test_id_##name##_##line
#define TEST_ID(line,name) TEST_ID_(line,name)

#define TEST_CASE_(id,name) \
extern "C" void TEST_NAME(id,name)(); \
auto TEST_ID(id,name) = register_unit_test(TEST_NAME(id,name)); \
void TEST_NAME(id,name)()
#define TEST_CASE(name) TEST_CASE_(__LINE__,name)

#define REQUIRE(cond) \
do { \
    if ( !__builtin_expect( !!(cond), 1 ) ) \
    { \
        get_unit_test_registrar().fail( #cond ); \
    } \
} while (0)

#define FAIL(...) \
do { \
    INFO(__VA_ARGS__); \
    REQUIRE(false); \
} while (0)

class Approx
{
    protected:
        double target_;
        double margin_ = 0;

    public:
        Approx(double target) : target_(target) {}

        Approx& margin(double value)
        {
            margin_ = value;
            return *this;
        }

        bool operator==(double other) const
        {
            return std::abs(other - target_) <= margin_;
        }

        friend bool operator==(double lhs, const Approx& rhs)
        {
            return rhs == lhs;
        }
};

#define UNIT_TEST1( ch1, opname ) \
TEST_CASE(ch1##opname) \
{ \
    INFO("Type character 1: " << #ch1); \
    printf("Testing: %s...", STRINGIFY_INT(ch1##opname));

#define UNIT_TEST2( ch1, ch2, opname ) \
TEST_CASE(ch1##ch2##opname) \
{ \
    INFO("Type character 1: " << #ch1); \
    INFO("Type character 2: " << #ch2); \
    printf("Testing: %s...", STRINGIFY_INT(ch1##ch2##opname));

#define UNIT_TEST3( ch1, ch2, ch3, opname ) \
TEST_CASE(ch1##ch2##ch3##opname) \
{ \
    INFO("Type character 1: " << #ch1); \
    INFO("Type character 2: " << #ch2); \
    INFO("Type character 3: " << #ch3); \
    printf("Testing: %s...", STRINGIFY_INT(ch1##ch2##ch3##opname));

#define UNIT_TEST4( ch1, ch2, ch3, ch4, opname ) \
TEST_CASE(ch1##ch2##ch3##ch4##opname) \
{ \
    INFO("Type character 1: " << #ch1); \
    INFO("Type character 2: " << #ch2); \
    INFO("Type character 3: " << #ch3); \
    INFO("Type character 4: " << #ch4); \
    printf("Testing: %s...", STRINGIFY_INT(ch1##ch2##ch3##ch4##opname));

#define UNIT_TEST5( ch1, ch2, ch3, ch4, ch5, opname ) \
TEST_CASE(ch1##ch2##ch3##ch4##ch5##opname) \
{ \
    INFO("Type character 1: " << #ch1); \
    INFO("Type character 2: " << #ch2); \
    INFO("Type character 3: " << #ch3); \
    INFO("Type character 4: " << #ch4); \
    INFO("Type character 5: " << #ch5); \
    printf("Testing: %s...", STRINGIFY_INT(ch1##ch2##ch3##ch4##ch5##opname));

#define UNIT_TEST_BODY( ... ) \
    __VA_ARGS__; \
    printf("%sPASS%s\n", unit_test_registrar::green(), unit_test_registrar::normal()); \
}

#define UNIT_TEST_SELECTOR_( ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ... ) ARG7

#define UNIT_TEST_SELECTOR( ... ) \
UNIT_TEST_SELECTOR_( __VA_ARGS__, \
                     UNIT_TEST5, \
                     UNIT_TEST4, \
                     UNIT_TEST3, \
                     UNIT_TEST2, \
                     UNIT_TEST1)

#define UNIT_TEST( ... ) UNIT_TEST_SELECTOR(__VA_ARGS__)(__VA_ARGS__) UNIT_TEST_BODY

enum
{
    BLIS_TEST_ZERO      = 0x01,
    BLIS_TEST_NEGATIVE  = 0x02,
    BLIS_TEST_INFINITY  = 0x04,
    BLIS_TEST_NAN       = 0x08,
    BLIS_TEST_DEFAULT   = ~BLIS_TEST_INFINITY
};

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
std::enable_if_t<std::is_arithmetic<T>::value,real_imag_part<T>> imag(T) { return {}; }

inline float& real(scomplex& x) { return x.real; }

inline float& imag(scomplex& x) { return x.imag; }

inline double& real(dcomplex& x) { return x.real; }

inline double& imag(dcomplex& x) { return x.imag; }

inline const float& real(const scomplex& x) { return x.real; }

inline const float& imag(const scomplex& x) { return x.imag; }

inline const double& real(const dcomplex& x) { return x.real; }

inline const double& imag(const dcomplex& x) { return x.imag; }

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value,T> norm(T x) { return x*x; }

inline float norm(const scomplex& x) { return x.real*x.real + x.imag*x.imag; }

inline double norm(const dcomplex& x) { return x.real*x.real + x.imag*x.imag; }

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value,T> absolute(T x) { return std::abs(x); }

inline float absolute(const scomplex& x) { return std::hypot(x.real, x.imag); }

inline double absolute(const dcomplex& x) { return std::hypot(x.real, x.imag); }

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value,T> square_root(T x) { return std::sqrt(x); }

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
}

COMPLEX_MATH_OPS(float,  scomplex);
COMPLEX_MATH_OPS(double, dcomplex);

template <typename T>
std::enable_if_t<is_real<T>::value,T> conj(T x) { return x; }

template <typename T>
std::enable_if_t<is_complex<T>::value,T> conj(const T& x) { return {x.real, -x.imag}; }

template <typename T>
std::enable_if_t<is_complex<T>::value,T> swapri(const T& x) { return {x.imag, x.real}; }

inline bool bli_isnan( float x ) { return bli_sisnan( x ); }

inline bool bli_isnan( double x ) { return bli_disnan( x ); }

inline bool bli_isinf( float x ) { return bli_sisinf( x ); }

inline bool bli_isinf( double x ) { return bli_disinf( x ); }

template <typename C, typename T>
std::enable_if_t<is_real<T>::value> check(T x, T y)
{
    auto tol = 8*std::numeric_limits<make_real_t<C>>::epsilon();
    INFO("x: " << x);
    INFO("y: " << y);
    INFO("|x-y|: " << std::abs(x-y));
    INFO("eps: " << tol);
    if ( bli_isnan( x ) || bli_isnan( y ) )
        REQUIRE( bli_isnan( x ) == bli_isnan( y ) );
    else if ( bli_isinf( x ) || bli_isinf( y ) )
        REQUIRE( x == y );
    else
        REQUIRE( x == Approx(y).margin(tol) );
}

template <typename C, typename T>
std::enable_if_t<is_complex<T>::value> check(const T& x, const T& y)
{
    {
        INFO("Real part:");
        check<C>( x.real, y.real );
    }
    {
        INFO("Imag part:");
        check<C>( x.imag, y.imag );
    }
}

template <typename T>
std::enable_if_t<is_real<T>::value,std::vector<T>> test_values(int mask = BLIS_TEST_DEFAULT)
{
    std::vector<T> vals{0.439};

    if (mask & BLIS_TEST_NEGATIVE)
        vals.push_back(-0.563);

    if (mask & BLIS_TEST_ZERO)
        vals.push_back(0);

    if (mask & BLIS_TEST_INFINITY)
    {
        vals.push_back(INFINITY);
        if (mask & BLIS_TEST_NEGATIVE)
            vals.push_back(-INFINITY);
    }

    if (mask & BLIS_TEST_NAN)
        vals.push_back(NAN);

    return vals;
}

template <typename T>
std::enable_if_t<is_complex<T>::value,std::vector<T>> test_values(int mask = BLIS_TEST_DEFAULT)
{
    auto real_vals = test_values<make_real_t<T>>(mask);
    std::vector<T> vals;
    for (auto& r : real_vals)
    for (auto& i : real_vals)
        vals.push_back({r, i});
    return vals;
}

template <typename T>
std::enable_if_t<is_complex<T>::value,std::ostream&> operator<<(std::ostream& os, const T& val)
{
    return os << '(' << val.real << ", " << val.imag << ')';
}

template <size_t M, size_t N, typename T>
std::array<std::array<T,N>,M> tile(const T& val = T())
{
    std::array<std::array<T,N>,M> ret;
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
        ret[i][j] = val;
    return ret;
}

template <size_t M, size_t N, typename T>
std::array<std::array<T,N>,M> conj(const std::array<std::array<T,N>,M>& x)
{
    std::array<std::array<T,N>,M> ret;
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
        ret[i][j] = conj(x[i][j]);
    return ret;
}

template <size_t M, size_t N, typename T>
std::array<std::array<make_real_t<T>,N>,M> real(const std::array<std::array<T,N>,M>& x)
{
    std::array<std::array<make_real_t<T>,N>,M> ret;
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
        ret[i][j] = real(x[i][j]);
    return ret;
}

template <size_t M, size_t N, typename T>
std::array<std::array<make_real_t<T>,N>,M> imag(const std::array<std::array<T,N>,M>& x)
{
    std::array<std::array<make_real_t<T>,N>,M> ret;
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
        ret[i][j] = imag(x[i][j]);
    return ret;
}

template <size_t D, size_t M, size_t N, typename T>
std::enable_if_t<!is_complex<T>::value,std::array<std::array<T,D*N>,M>>
bcast(const std::array<std::array<T,N>,M>& x)
{
    std::array<std::array<T,D*N>,M> ret;
    for (size_t d = 0;d < D;d++)
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
        ret[i][d + j*D] = x[i][j];
    return ret;
}

template <size_t D, size_t M, size_t N, typename T>
std::enable_if_t<is_complex<T>::value,std::array<std::array<T,D*N>,M>>
bcast(const std::array<std::array<T,N>,M>& x)
{
    std::array<std::array<make_real_t<T>,2*D*N>,M> ret_r;
    std::array<std::array<T,D*N>,M> ret;
    for (size_t d = 0;d < D;d++)
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
    {
        ret_r[i][d + 0*D + j*2*D] = real(x[i][j]);
        ret_r[i][d + 1*D + j*2*D] = imag(x[i][j]);
    }
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < D*N;j++)
    {
        real(ret[i][j]) = ret_r[i][j*2+0];
        imag(ret[i][j]) = ret_r[i][j*2+1];
    }
    return ret;
}

struct dense_cond
{
    bool operator()(dim_t, dim_t) const { return true; }
};

constexpr dense_cond dense;

struct is_below
{
    doff_t diagoff;

    is_below(doff_t d) : diagoff(d) {}

    bool operator()(dim_t i, dim_t j) const { return j-i <= diagoff; }
};

struct is_above
{
    doff_t diagoff;

    is_above(doff_t d) : diagoff(d) {}

    bool operator()(dim_t i, dim_t j) const { return j-i >= diagoff; }
};

template <typename C, typename T, size_t M, size_t N>
void check(const std::array<std::array<T,N>,M>& x,
           const std::array<std::array<T,N>,M>& y)
{
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
    {
        INFO("i = " << i);
        INFO("j = " << j);
        check<C>(x[i][j], y[i][j]);
    }
}

template <typename C, int Transpose, typename A, typename X, typename B, typename Y, size_t M, size_t N>
void axpbys_mxn(const A& a, const std::array<std::array<X,N>,M>& x,
                const B& b,       std::array<std::array<Y,N>,M>& y, const std::function<bool(size_t,size_t)>& cond)
{
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
    if (Transpose == BLIS_NO_TRANSPOSE ? cond(i, j) : cond(j, i))
    {
        if (real(b) == 0 && imag(b) == 0)
            y[i][j] = convert<Y>(convert_prec<C>(a) *
                                 convert_prec<C>(x[i][j]));
        else
            y[i][j] = convert<Y>((convert_prec<C>(a) *
                                  convert_prec<C>(x[i][j])) +
                                 (convert_prec<C>(b) *
                                  convert_prec<C>(y[i][j])));
    }
}

namespace std
{

template <typename T, size_t M, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<std::array<T,N>,M>& x)
{
    for (size_t i = 0;i < M;i++)
    for (size_t j = 0;j < N;j++)
        os << '[' << i << "][" << j << "]: " << x[i][j] << std::endl;
    return os;
}

} // namespace std

#define BLIS_FOR_ALL_TYPES0(macro, ...) macro(__VA_ARGS__);

#define BLIS_FOR_TYPES_1R(...) \
BLIS_FOR_ALL_TYPES0(__VA_ARGS__, float, s) \
BLIS_FOR_ALL_TYPES0(__VA_ARGS__, double, d)

#define BLIS_FOR_TYPES_1C(...) \
BLIS_FOR_ALL_TYPES0(__VA_ARGS__, scomplex, c) \
BLIS_FOR_ALL_TYPES0(__VA_ARGS__, dcomplex, z)

#define BLIS_FOR_TYPES_1RC(...) \
BLIS_FOR_TYPES_1R(__VA_ARGS__) \
BLIS_FOR_TYPES_1C(__VA_ARGS__)

#define BLIS_FOR_ALL_TYPES1(type, ...) PASTECH(BLIS_FOR_TYPES_1, type)(__VA_ARGS__)

#define BLIS_FOR_TYPES_2R(...) \
BLIS_FOR_ALL_TYPES1(__VA_ARGS__, float, s) \
BLIS_FOR_ALL_TYPES1(__VA_ARGS__, double, d)

#define BLIS_FOR_TYPES_2C(...) \
BLIS_FOR_ALL_TYPES1(__VA_ARGS__, scomplex, c) \
BLIS_FOR_ALL_TYPES1(__VA_ARGS__, dcomplex, z)

#define BLIS_FOR_TYPES_2RC(...) \
BLIS_FOR_TYPES_2R(__VA_ARGS__) \
BLIS_FOR_TYPES_2C(__VA_ARGS__)

#define BLIS_FOR_ALL_TYPES2(type, ...) PASTECH(BLIS_FOR_TYPES_2, type)(__VA_ARGS__)

#define BLIS_FOR_TYPES_3R(...) \
BLIS_FOR_ALL_TYPES2(__VA_ARGS__, float, s) \
BLIS_FOR_ALL_TYPES2(__VA_ARGS__, double, d)

#define BLIS_FOR_TYPES_3C(...) \
BLIS_FOR_ALL_TYPES2(__VA_ARGS__, scomplex, c) \
BLIS_FOR_ALL_TYPES2(__VA_ARGS__, dcomplex, z)

#define BLIS_FOR_TYPES_3RC(...) \
BLIS_FOR_TYPES_3R(__VA_ARGS__) \
BLIS_FOR_TYPES_3C(__VA_ARGS__)

#define BLIS_FOR_ALL_TYPES3(type, ...) PASTECH(BLIS_FOR_TYPES_3, type)(__VA_ARGS__)

#define BLIS_FOR_TYPES_4R(...) \
BLIS_FOR_ALL_TYPES3(__VA_ARGS__, float, s) \
BLIS_FOR_ALL_TYPES3(__VA_ARGS__, double, d)

#define BLIS_FOR_TYPES_4C(...) \
BLIS_FOR_ALL_TYPES3(__VA_ARGS__, scomplex, c) \
BLIS_FOR_ALL_TYPES3(__VA_ARGS__, dcomplex, z)

#define BLIS_FOR_TYPES_4RC(...) \
BLIS_FOR_TYPES_4R(__VA_ARGS__) \
BLIS_FOR_TYPES_4C(__VA_ARGS__)

#define BLIS_FOR_ALL_TYPES4(type, ...) PASTECH(BLIS_FOR_TYPES_4, type)(__VA_ARGS__)

#define BLIS_FOR_TYPES_5R(...) \
BLIS_FOR_ALL_TYPES4(__VA_ARGS__, float, s) \
BLIS_FOR_ALL_TYPES4(__VA_ARGS__, double, d)

#define BLIS_FOR_TYPES_5C(...) \
BLIS_FOR_ALL_TYPES4(__VA_ARGS__, scomplex, c) \
BLIS_FOR_ALL_TYPES4(__VA_ARGS__, dcomplex, z)

#define BLIS_FOR_TYPES_5RC(...) \
BLIS_FOR_TYPES_5R(__VA_ARGS__) \
BLIS_FOR_TYPES_5C(__VA_ARGS__)

#define BLIS_FOR_ALL_TYPES5(type, ...) PASTECH(BLIS_FOR_TYPES_5, type)(__VA_ARGS__)

#define INSERT_GENTFUNC_MIX1(t1, opname) \
BLIS_FOR_ALL_TYPES1(t1, GENTFUNC, opname)

#define INSERT_GENTFUNC_MIX2(t1, t2, opname) \
BLIS_FOR_ALL_TYPES2(t1, t2, GENTFUNC, opname)

#define INSERT_GENTFUNC_MIX3(t1, t2, t3, opname) \
BLIS_FOR_ALL_TYPES3(t1, t2, t3, GENTFUNC, opname)

#define INSERT_GENTFUNC_MIX4(t1, t2, t3, t4, opname) \
BLIS_FOR_ALL_TYPES4(t1, t2, t3, t4, GENTFUNC, opname)

#define INSERT_GENTFUNC_MIX5(t1, t2, t3, t4, t5, opname) \
BLIS_FOR_ALL_TYPES5(t1, t2, t3, t4, t5, GENTFUNC, opname)

#endif

