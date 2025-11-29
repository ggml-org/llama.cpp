/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2023, Southern Methodist University

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

#ifndef BLIS_COMPLEX_MACRO_DEFS_H
#define BLIS_COMPLEX_MACRO_DEFS_H


// -- Real and imaginary accessor macros --


#define bli_sreal( x )  ( x )
#define bli_simag( x )  ( 0.0F )
#define bli_dreal( x )  ( x )
#define bli_dimag( x )  ( 0.0 )
#define bli_ireal( x )  ( x )
#define bli_iimag( x )  ( 0 )


#if defined(__cplusplus) && defined(BLIS_ENABLE_STD_COMPLEX)

} // extern "C"

// Create functions bli_[cz]{real,imag} for std::complex<T> which mimic those
// for the simple struct version. Since normally x.real/x.imag are
// lvalues, we have to create a wrapper since x.real()/x.imag() in std::complex
// are rvalues. These will only be used if the user has typedef'd scomplex as
// std::complex<float> and dcomplex as std::complex<double> themselves.

#include <complex>

template <typename T, bool Imag>
struct bli_complex_wrapper
{
	std::complex<T>& ref;

	bli_complex_wrapper(std::complex<T>& ref) : ref(ref) {}

	operator T() const { return Imag ? ref.imag() : ref.real(); }

	bli_complex_wrapper& operator=(const bli_complex_wrapper& other)
	{
		return *this = static_cast<T>( other );
	}

	bli_complex_wrapper& operator=(T other)
	{
		if (Imag)
			ref.imag(other);
		else
			ref.real(other);
		return *this;
	}
};

inline bli_complex_wrapper<float,false> bli_creal( std::complex<float>& x )
{
	return x;
}

inline float bli_creal( const std::complex<float>& x )
{
	return x.real();
}

inline bli_complex_wrapper<float,true> bli_cimag( std::complex<float>& x )
{
	return x;
}

inline float bli_cimag( const std::complex<float>& x )
{
	return x.imag();
}

inline bli_complex_wrapper<double,false> bli_zreal( std::complex<double>& x )
{
	return x;
}

inline double bli_zreal( const std::complex<double>& x )
{
	return x.real();
}

inline bli_complex_wrapper<double,true> bli_zimag( std::complex<double>& x )
{
	return x;
}

inline double bli_zimag( const std::complex<double>& x )
{
	return x.imag();
}

#define __typeof__(x) auto

extern "C"
{

#elif !defined(BLIS_ENABLE_C99_COMPLEX)


#define bli_creal( x )  ( (x).real )
#define bli_cimag( x )  ( (x).imag )
#define bli_zreal( x )  ( (x).real )
#define bli_zimag( x )  ( (x).imag )


#else // ifdef BLIS_ENABLE_C99_COMPLEX

// Note that these definitions probably don't work because of constructs
// like `bli_zreal( x ) = yr`.

#define bli_creal( x )  ( crealf(x) )
#define bli_cimag( x )  ( cimagf(x) )
#define bli_zreal( x )  ( creal(x) )
#define bli_zimag( x )  ( cimag(x) )


#endif // BLIS_ENABLE_C99_COMPLEX


#endif

