#include <stdio.h>
#include <string.h>

#define ARRAY_LEN  4096

double x[ ARRAY_LEN ];
double y[ ARRAY_LEN ];

int main( int argc, char **argv )
{
	const double alpha = 2.1;

	for ( int i = 0; i < ARRAY_LEN; ++i )
	{
		y[ i ] = 0.0;
		x[ i ] = 1.0;
	}

	#pragma omp simd
	for ( int i = 0; i < ARRAY_LEN; ++i )
	{
		y[ i ] += alpha * x[ i ];
	}

#if 0
	_Pragma( "omp simd" )
	for ( int i = 0; i < ARRAY_LEN; ++i )
	{
		x[ i ] += alpha * y[ i ];
	}
#endif

    return 0;
}

