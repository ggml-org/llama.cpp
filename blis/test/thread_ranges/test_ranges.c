/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include <unistd.h>
#include "blis.h"


//#define PRINT

int main( int argc, char** argv )
{
	//bli_init();

#if 0
	obj_t a, b, c;
	obj_t aa, bb, cc;
	dim_t m, n, k;
	num_t dt;
	uplo_t uploa, uplob, uploc;

	{
		dt = BLIS_DOUBLE;

		m = 6;
		k = 6;
		n = 6;

		bli_obj_create( dt, m, k, 0, 0, &a );
		bli_obj_create( dt, k, n, 0, 0, &b );
		bli_obj_create( dt, m, n, 0, 0, &c );

		uploa = BLIS_UPPER;
		uploa = BLIS_LOWER;
		bli_obj_set_struc( BLIS_TRIANGULAR, &a );
		bli_obj_set_uplo( uploa, &a );
		bli_obj_set_diag_offset( -2, &a );

		uplob = BLIS_UPPER;
		uplob = BLIS_LOWER;
		bli_obj_set_struc( BLIS_TRIANGULAR, &b );
		bli_obj_set_uplo( uplob, &b );
		bli_obj_set_diag_offset( -2, &b );

		uploc = BLIS_UPPER;
		//uploc = BLIS_LOWER;
		//uploc = BLIS_ZEROS;
		//uploc = BLIS_DENSE;
		bli_obj_set_struc( BLIS_HERMITIAN, &c );
		//bli_obj_set_struc( BLIS_TRIANGULAR, &c );
		bli_obj_set_uplo( uploc, &c );
		bli_obj_set_diag_offset(  1, &c );

		bli_obj_alias_to( &a, &aa ); (void)aa;
		bli_obj_alias_to( &b, &bb ); (void)bb;
		bli_obj_alias_to( &c, &cc ); (void)cc;

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );
		//bli_mkherm( &a );
		//bli_mktrim( &a );

		bli_prune_unref_mparts( &cc, BLIS_M,
		                        &aa, BLIS_N );

		bli_printm( "c orig", &c, "%4.1f", "" );
		bli_printm( "c alias", &cc, "%4.1f", "" );
		bli_printm( "a orig", &a, "%4.1f", "" );
		bli_printm( "a alias", &aa, "%4.1f", "" );
		//bli_obj_print( "a struct", &a );
	}
#endif

	dim_t  p_begin, p_max, p_inc;
	gint_t m_input, n_input;
	char   uploa_ch;
	doff_t diagoffa;
	dim_t  bf;
	dim_t  n_way;
	char   part_dim_ch;
	dim_t  go_fwd;
	char   out_ch;

	obj_t   a;

	thrinfo_t thrinfo;
	dim_t  m, n;
	uplo_t uploa;
	dim_t  part_m_dim, part_n_dim;
	dim_t  go_bwd;
	dim_t  p;
	num_t  dt;
	dim_t  start, end;

	dim_t  width;
	siz_t  area;

	gint_t t_begin, t_stop, t_inc;
	dim_t  t;

	if ( argc == 13 )
	{
		sscanf( argv[1], "%u", &p_begin );
		sscanf( argv[2], "%u", &p_max );
		sscanf( argv[3], "%u", &p_inc );
		sscanf( argv[4], "%d", &m_input );
		sscanf( argv[5], "%d", &n_input );
		sscanf( argv[6], "%c",  &uploa_ch );
		sscanf( argv[7], "%d", &diagoffa );
		sscanf( argv[8], "%u", &bf );
		sscanf( argv[9], "%u", &n_way );
		sscanf( argv[10], "%c", &part_dim_ch );
		sscanf( argv[11], "%u", &go_fwd );
		sscanf( argv[12], "%c", &out_ch );
	}
	else
	{
		printf( "\n" );
		printf( " %s\n", argv[0] );
		printf( "\n" );
		printf( "  Simulate the dimension ranges assigned to threads when\n" );
		printf( "  partitioning a matrix for parallelism in BLIS.\n" );
		printf( "\n" );
		printf( " Usage:\n" );
		printf( "\n" );
		printf( "  %s p_beg p_max p_inc m n uplo doff bf n_way part_dim go_fwd out\n", argv[0] );
		printf( "\n" );
		printf( "  p_beg:    the first problem size p to test.\n" );
		printf( "  p_max:    the maximum problem size p to test.\n" );
		printf( "  p_inc:    the increase in problem size p between tests.\n" );
		printf( "  m:        the m dimension:\n" );
		printf( "  n:        the n dimension:\n" );
		printf( "            if m,n = -1: bind m,n to problem size p.\n" );
		printf( "            if m,n =  0: bind m,n to p_max.\n" );
		printf( "            if m,n >  0: hold m,n = c constant for all p.\n" );
		printf( "  uplo:     the uplo field of the matrix being partitioned:\n" );
		printf( "            'l': lower-stored (BLIS_LOWER)\n" );
		printf( "            'u': upper-stored (BLIS_UPPER)\n" );
		printf( "            'd': densely-stored (BLIS_DENSE)\n" );
		printf( "  doff:     the diagonal offset of the matrix being partitioned.\n" );
		printf( "  bf:       the simulated blocking factor. all thread ranges must\n" );
		printf( "            be a multiple of bf, except for the range that contains\n" );
		printf( "            the edge case (if one exists). the blocking factor\n" );
		printf( "            would typically correspond to a register blocksize.\n" );
		printf( "  n_way:    the number of ways of parallelism for which we are\n" );
		printf( "            partitioning (i.e.: the number of threads, or thread\n" );
		printf( "            groups).\n" );
		printf( "  part_dim: the dimension to partition:\n" );
		printf( "            'm': partition the m dimension.\n" );
		printf( "            'n': partition the n dimension.\n" );
		printf( "  go_fwd:   the direction to partition:\n" );
		printf( "            '1': forward, e.g. left-to-right (part_dim = 'm') or\n" );
		printf( "                 top-to-bottom (part_dim = 'n')\n" );
		printf( "            '0': backward, e.g. right-to-left (part_dim = 'm') or\n" );
		printf( "                 bottom-to-top (part_dim = 'n')\n" );
		printf( "            NOTE: reversing the direction does not change the\n" );
		printf( "            subpartitions' widths, but it does change which end of\n" );
		printf( "            the index range receives the edge case, if it exists.\n" );
		printf( "  out:      the type of output per thread-column:\n" );
		printf( "            'w': the width (and area) of the thread's subpartition\n" );
		printf( "            'r': the actual ranges of the thread's subpartition\n" );
		printf( "                 where the start and end points of each range are\n" );
		printf( "                 inclusive and exclusive, respectively.\n" );
		printf( "\n" );

		exit(1);
	}

	if ( m_input == 0 ) m_input = p_max;
	if ( n_input == 0 ) n_input = p_max;

	if ( part_dim_ch == 'm' ) { part_m_dim = TRUE;  part_n_dim = FALSE; }
	else                      { part_m_dim = FALSE; part_n_dim = TRUE;  }

	go_bwd = !go_fwd;

	if      ( uploa_ch == 'l' ) uploa = BLIS_LOWER;
	else if ( uploa_ch == 'u' ) uploa = BLIS_UPPER;
	else                        uploa = BLIS_DENSE;

	if ( part_n_dim )
	{
		if ( bli_is_upper( uploa ) ) { t_begin = n_way-1; t_stop = -1;    t_inc = -1; }
		else /* if lower or dense */ { t_begin = 0;       t_stop = n_way; t_inc =  1; }
	}
	else // if ( part_m_dim )
	{
		if ( bli_is_lower( uploa ) ) { t_begin = n_way-1; t_stop = -1;    t_inc = -1; }
		else /* if upper or dense */ { t_begin = 0;       t_stop = n_way; t_inc =  1; }
	}

	printf( "\n" );
	printf( "  part: %3s   doff: %3d   bf: %3d   output: %s\n",
	        ( part_n_dim ? ( go_fwd ? "l2r" : "r2l" )
	                     : ( go_fwd ? "t2b" : "b2t" ) ),
	        ( int )diagoffa, ( int )bf,
            ( out_ch == 'w' ? "width(area)" : "ranges" ) );
	printf( "              uplo: %3c   nt: %3u\n", uploa_ch, ( unsigned )n_way );
	printf( "\n" );

	printf( "             " );
	for ( t = t_begin; t != t_stop; t += t_inc )
	{
		if ( part_n_dim )
		{
			if      ( t == t_begin      ) printf( "left...      " );
			else if ( t == t_stop-t_inc ) printf( "     ...right" );
			else                          printf( "             " );
		}
		else // if ( part_m_dim )
		{
			if      ( t == t_begin      ) printf( "top...       " );
			else if ( t == t_stop-t_inc ) printf( "    ...bottom" );
			else                          printf( "             " );
		}
	}
	printf( "\n" );


	printf( "%4c x %4c  ", 'm', 'n' );
	for ( t = t_begin; t != t_stop; t += t_inc )
	{
		printf( "%9s %u  ", "thread", ( unsigned )t );
	}
	printf( "\n" );
	printf( "-------------" );
	for ( t = t_begin; t != t_stop; t += t_inc )
	{
		printf( "-------------" );
	}
	printf( "\n" );


	for ( p = p_begin; p <= p_max; p += p_inc )
	{
		if ( m_input < 0 ) m = ( dim_t )p;
		else               m = ( dim_t )m_input;
		if ( n_input < 0 ) n = ( dim_t )p;
		else               n = ( dim_t )n_input;

		dt = BLIS_DOUBLE;

		bli_obj_create( dt, m, n, 0, 0, &a );

		bli_obj_set_struc( BLIS_TRIANGULAR, &a );
		bli_obj_set_uplo( uploa, &a );
		bli_obj_set_diag_offset( diagoffa, &a );

		bli_randm( &a );

		printf( "%4u x %4u  ", ( unsigned )m, ( unsigned )n );

		for ( t = t_begin; t != t_stop; t += t_inc )
		{
			thrinfo.n_way   = n_way;
			thrinfo.work_id = t;

			if      ( part_n_dim && go_fwd )
				area = bli_thread_range( &thrinfo, &a, bf, BLIS_N, BLIS_FWD, TRUE, &start, &end );
			else if ( part_n_dim && go_bwd )
				area = bli_thread_range( &thrinfo, &a, bf, BLIS_N, BLIS_BWD, TRUE, &start, &end );
			else if ( part_m_dim && go_fwd )
				area = bli_thread_range( &thrinfo, &a, bf, BLIS_M, BLIS_FWD, TRUE, &start, &end );
			else // ( part_m_dim && go_bwd )
				area = bli_thread_range( &thrinfo, &a, bf, BLIS_M, BLIS_BWD, TRUE, &start, &end );

			width = end - start;

			if ( out_ch == 'w' ) printf( "%4u(%6u) ", ( unsigned )width,
			                                            ( unsigned )area );
			else                 printf( "[%4u,%4u)  ", ( unsigned )start,
			                                              ( unsigned )end );
		}

		printf( "\n" );

		bli_obj_free( &a );
	}

	//bli_finalize();

	return 0;
}

