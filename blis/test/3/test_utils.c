/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#include "blis.h"
#include "test_utils.h"

// Global string constants.
const char* GLOB_DEF_DT_STR    = "d";
const char* GLOB_DEF_SC_STR    = "ccc";
const char* GLOB_DEF_IM_STR    = "auto";

const char* GLOB_DEF_PS_STR    = "50 1000 50";
const char* GLOB_DEF_M_STR     = "-1";
const char* GLOB_DEF_N_STR     = "-1";
const char* GLOB_DEF_K_STR     = "-1";

const char* GLOB_DEF_NR_STR    = "3";

const char* GLOB_DEF_ALPHA_STR = "1.0";
const char* GLOB_DEF_BETA_STR  = "1.0";


void parse_cl_params( int argc, char** argv, init_fp fp, params_t* params )
{
	bool     gave_option_c = FALSE;
	bool     gave_option_d = FALSE;
	bool     gave_option_s = FALSE;

	bool     gave_option_i = FALSE;

	bool     gave_option_p = FALSE;
	bool     gave_option_m = FALSE;
	bool     gave_option_n = FALSE;
	bool     gave_option_k = FALSE;

	bool     gave_option_r = FALSE;

	bool     gave_option_a = FALSE;
	bool     gave_option_b = FALSE;

	int      opt;
	char     opt_ch;

	getopt_t state;

	// Initialize the params_t struct with the caller-supplied function.
	fp( params );

	// Copy the binary name pointer so we can use it later.
	params->bin = argv[0];

	// Alias the binary name for conciseness.
	const char* bin = params->bin;

	// Initialize the state for running bli_getopt(). Here, 0 is the
	// initial value for opterr, which suppresses error messages.
	bli_getopt_init_state( 0, &state );

	// Process all option arguments until we get a -1, which means we're done.
	while( (opt = bli_getopt( argc, ( const char* const * )argv, "c:d:s:i:p:m:n:k:r:a:b:qvh", &state )) != -1 )
	{
		// Explicitly typecast opt, which is an int, to a char. (Failing to
		// typecast resulted in at least one user-reported problem whereby
		// opt was being filled with garbage.)
		opt_ch = ( char )opt;

		switch( opt_ch )
		{
			case 'c':
			params->pc_str = state.optarg;
			gave_option_c = TRUE;
			break;

			case 'd':
			params->dt_str = state.optarg;
			gave_option_d = TRUE;
			break;

			case 's':
			params->sc_str = state.optarg;
			gave_option_s = TRUE;
			break;


			case 'i':
			params->im_str = state.optarg;
			gave_option_i = TRUE;
			break;


			case 'p':
			params->ps_str = state.optarg;
			gave_option_p = TRUE;
			break;

			case 'm':
			params->m_str = state.optarg;
			gave_option_m = TRUE;
			break;

			case 'n':
			params->n_str = state.optarg;
			gave_option_n = TRUE;
			break;

			case 'k':
			params->k_str = state.optarg;
			gave_option_k = TRUE;
			break;


			case 'r':
			params->nr_str = state.optarg;
			gave_option_r = TRUE;
			break;


			case 'a':
			params->alpha_str = state.optarg;
			gave_option_a = TRUE;
			break;

			case 'b':
			params->beta_str = state.optarg;
			gave_option_b = TRUE;
			break;


			case 'q':
			params->verbose = FALSE;
			break;

			case 'v':
			params->verbose = TRUE;
			break;

			case 'h':
			{
				bool has_trans = FALSE;
				bool has_side  = FALSE;
				bool has_uplo  = FALSE;
				bool has_unit  = FALSE;

				if ( is_gemm( params ) ||
					 is_herk( params ) ||
					 is_trmm( params ) ||
					 is_trsm( params ) ) has_trans = TRUE;

				if ( is_hemm( params ) ||
					 is_trmm( params ) ||
					 is_trsm( params ) ) has_side = TRUE;

				if ( is_hemm( params ) ||
					 is_herk( params ) ||
					 is_trmm( params ) ||
					 is_trsm( params ) ) has_uplo = TRUE;

				if ( is_trmm( params ) ||
					 is_trsm( params ) ) has_unit = TRUE;

				printf( "\n" );
				printf( "  %s performance driver\n", params->opname );
				printf( "  -----------------------\n" );
				printf( "  (part of the BLIS framework)\n" );
				printf( "\n" );
				printf( "  Measure performance of the '%s' implementation of the '%s' operation:\n", params->impl, params->opname );
				printf( "\n" );
				if ( is_gemm( params ) )
				{
				printf( "      C := beta * C + alpha * trans(A) * trans(B)\n" );
				printf( "\n" );
				printf( "  where C is an m x n matrix, trans(A) is an m x k matrix, and\n" );
				printf( "  trans(B) is a k x n matrix.\n" );
				}
				else if ( is_hemm( params ) )
				{
				printf( "      C := beta * C + alpha * uplo(A) * B     (side = left)\n" );
				printf( "      C := beta * C + alpha * B * uplo(A)     (side = right)\n" );
				printf( "\n" );
				printf( "  where C and B are m x n matrices and A is a Hermitian matrix stored\n" );
				printf( "  in the lower or upper triangle, as specified by uplo(A). When side =\n" );
				printf( "  left, A is m x m, and when side = right, A is n x n.\n" );
				}
				else if ( is_herk( params ) )
				{
				printf( "      uplo(C) := beta * uplo(C) + alpha * trans(A) * trans(A)^H\n" );
				printf( "\n" );
				printf( "  where C is an m x m Hermitian matrix stored in the lower or upper\n" );
				printf( "  triangle, as specified by uplo(C), and trans(A) is an m x k matrix.\n" );
				}
				else if ( is_trmm( params ) )
				{
				printf( "      B := alpha * trans(uplo(A)) * B      (side = left)\n" );
				printf( "      B := alpha * B * trans(uplo(A))      (side = right)\n" );
				printf( "\n" );
				printf( "  where B is an m x n matrix and A is a triangular matrix stored in\n" );
				printf( "  the lower or upper triangle, as specified by uplo(A), with unit/non-unit\n" );
				printf( "  diagonal specified by diag(A). When side = left, A is m x m, and when\n" );
				printf( "  side = right, A is n x n.\n" );
				}
				else if ( is_trsm( params ) )
				{
				printf( "      B := alpha * trans(uplo(A))^{-1} * B     (side = left)\n" );
				printf( "      B := alpha * B * trans(uplo(A))^{-1}     (side = right)\n" );
				printf( "\n" );
				printf( "  where B is an m x n matrix and A is a triangular matrix stored in\n" );
				printf( "  the lower or upper triangle, as specified by uplo(A), with unit/non-unit\n" );
				printf( "  diagonal specified by diag(A). When side = left, A is m x m, and when\n" );
				printf( "  side = right, A is n x n. Note that while ^{-1} indicates inversion,\n" );
				printf( "  trsm does not explicitly invert A, but rather solves for an m x n\n" );
				printf( "  solution matrix X, which then overwrites the original contents of B.\n" );
				}
				printf( "\n" );
				printf( "  Performance measurements are taken for a range of problem sizes with a fixed\n" );
				printf( "  set of parameters, and results are printed to stdout in a matlab/octave-\n" );
				printf( "  friendly format.\n" );
				printf( "\n" );
				printf( "  Usage:\n" );
				printf( "\n" );
				printf( "    %s [options]\n", bin );
				printf( "\n" );
				printf( "  The following computational options are supported:\n" );
				printf( "\n" );
				printf( "    -c pc\n" );
				printf( "            Use the operation-specific parameter combination specified by\n" );
				printf( "            the 'pc' string. The following tables list expected parameters\n" );
				printf( "            for the '%s' operation and the valid values for each parameter.\n", params->opname );
				printf( "\n" );
				printf( "               Operation   List (order) of parameters          Example\n" );
				printf( "               -------------------------------------------------------\n" );
				if ( is_gemm( params ) )
				{
				printf( "               gemm        trans(A) trans(A)                   -c tn\n" );
				}
				else if ( is_hemm( params ) )
				{
				printf( "               hemm/symm   side(A) uplo(A)                     -c rl\n" );
				}
				else if ( is_herk( params ) )
				{
				printf( "               herk/syrk   uplo(C) trans(A)                    -c ln\n" );
				}
				else if ( is_trmm( params ) )
				{
				printf( "               trmm        side(A) uplo(A) trans(A) unit(A)    -c lutn\n" );
				}
				else if ( is_trsm( params ) )
				{
				printf( "               trsm        side(A) uplo(A) trans(A) unit(A)    -c rlnn\n" );
				}
				printf( "\n" );
				printf( "                           Valid\n" );
				printf( "               Param       chars    Interpretation\n" );
				printf( "               ---------------------------------------\n" );
				if ( has_trans )
				{
				printf( "               trans       n        No transpose\n" );
				printf( "                           t        Transpose only\n" );
				printf( "                           c        Conjugate only*\n" );
				printf( "                           h        Hermitian transpose\n" );
				printf( "\n" );
				}
				if ( has_side )
				{
				printf( "               side        l        Left\n" );
				printf( "                           r        Right\n" );
				printf( "\n" );
				}
				if ( has_uplo )
				{
				printf( "               uplo        l        Lower-stored\n" );
				printf( "                           u        Upper-stored\n" );
				printf( "\n" );
				}
				if ( has_unit )
				{
				printf( "               unit        u        Unit diagonal\n" );
				printf( "                           n        Non-unit diagonal\n" );
				printf( "\n" );
				}
				if ( has_trans )
				{
				printf( "               *This option is supported by BLIS but not by classic BLAS.\n" );
				}
				printf( "\n" );
				printf( "    -d dt\n" );
				printf( "            Allocate matrix elements using the datatype character specified\n" );
				printf( "            by dt, and also perform computation in that same datatype. Valid\n" );
				printf( "            char values for dt are:\n" );
				printf( "\n" );
				printf( "               Valid\n" );
				printf( "               chars     Interpretation\n" );
				printf( "               -----------------------------------------\n" );
				printf( "                s        single-precision real domain\n" );
				printf( "                d        double-precision real domain\n" );
				printf( "                c        single-precision complex domain\n" );
				printf( "                z        double-precision complex domain\n" );
				printf( "\n" );
				printf( "    -s sc\n" );
				printf( "            Use the characters in sc to determine the storage formats\n" );
				printf( "            of each operand matrix used in the performance measurements.\n" );
				printf( "            Valid chars are 'r' (row storage) and 'c' (column storage).\n" );
				printf( "            The characters encode the storage format for each operand\n" );
				printf( "            used by %s, with the mapping of chars to operand interpreted\n", params->opname );
				printf( "            in the following order:\n" );
				printf( "\n" );
				printf( "                            Order of\n" );
				printf( "                            operand      \n" );
				printf( "               Operation    mapping      Example     Interpretation\n" );
				printf( "               ----------------------------------------------------------\n" );
				if ( is_gemm( params ) )
				{
				printf( "               gemm         C A B        -s crr      C is col-stored;\n" );
				printf( "                                                     A and B are row-stored.\n" );
				}
				else if ( is_hemm( params ) )
				{
				printf( "               hemm/symm    C A B        -s rcc      C is row-stored;\n" );
				printf( "                                                     A and B are col-stored.\n" );
				}
				else if ( is_herk( params ) )
				{
				printf( "               herk/syrk    C A          -s rc       C is row-stored;\n" );
				printf( "                                                     A is col-stored.\n" );
				}
				else if ( is_trmm( params ) )
				{
				printf( "               trmm         B A          -s cr       B is col-stored;\n" );
				printf( "                                                     A is row-stored.\n" );
				}
				else if ( is_trsm( params ) )
				{
				printf( "               trsm         B A          -s cc       B and A are col-stored.\n" );
				}
				printf( "\n" );
				printf( "    -i im\n" );
				printf( "            Use native execution if im is 'native' (or 'nat'). If im is '1m',\n" );
				printf( "            use the 1m method to induce complex computation using the\n" );
				printf( "            equivalent real-domain microkernels. If im is 'auto', do not\n" );
				printf( "            explicitly set the induced method and instead use the default.\n" );
				printf( "\n" );
				printf( "    -p 'lo hi in'\n" );
				printf( "            Perform a sweep of measurements of problem sizes ranging from \n" );
				printf( "            'lo' to 'hi' in increments of 'in'. Note that measurements will\n" );
				printf( "            be taken in descending order, starting from 'hi', and so 'lo'\n" );
				printf( "            will act as a floor and may not be measured (see 2nd example).\n" );
				printf( "\n" );
				printf( "               Example             Interpretation\n" );
				printf( "               -------------------------------------------------------\n" );
				printf( "               -p '40 400 40'      Measure performance from 40 to 400\n" );
				printf( "                                   (inclusive) in increments of 40.\n" );
				printf( "               -p '40 400 80'      Measure performance for problem sizes\n" );
				printf( "                                   {80,160,240,320,400}.\n" );
				printf( "\n" );
				printf( "            Note that unlike the other option arguments, quotes are required\n" );
				printf( "            around the 'lo hi in' string in order to facilitate parsing.\n" );
				printf( "\n" );
				printf( "    -m M\n" );
				if ( is_gemm( params ) || is_hemm( params ) || is_trmm( params ) || is_trsm( params ) )
				printf( "    -n N\n" );
				if ( is_gemm( params ) || is_herk( params ) )
				printf( "    -k K\n" );
				if ( is_gemm( params ) )
				{
				printf( "            Bind the m, n, or k dimensions to M, N, or K, respectively.\n" );
				printf( "            Binding of matrix dimensions takes place as follows:\n" );
				}
				else if ( is_herk( params ) )
				{
				printf( "            Bind the m or k dimensions to M or K, respectively. Binding\n" );
				printf( "            of matrix dimensions takes place as follows:\n" );
				}
				else if ( is_hemm( params ) || is_trmm( params ) || is_trsm( params ) )
				{
				printf( "            Bind the m or n dimensions to M or N, respectively. Binding\n" );
				printf( "            of matrix dimensions takes place as follows:\n" );
				}
				printf( "\n" );
				printf( "               if 0 <  X: Bind the x dimension to X and hold it constant.\n" );
				printf( "               if X = -1: Bind the x dimension to p.\n" );
				printf( "               if X < -1: Bind the x dimension to p / abs(x).\n" );
				printf( "\n" );
				printf( "            where p is the current problem size. Note: X = 0 is undefined.\n" );
				printf( "\n" );
				printf( "               Examples             Interpretation\n" );
				printf( "               ---------------------------------------------------------\n" );
				if ( is_gemm( params ) )
				{
				printf( "               -m -1 -n -1 -k -1    Bind m, n, and k to the problem size\n" );
				printf( "                                    to keep all matrices square.\n" );
				printf( "               -m -1 -n -1 -k 100   Bind m and n to the problem size, but\n" );
				printf( "                                    hold k = 100 constant.\n" );
				}
				else if ( is_hemm( params ) )
				{
				printf( "               -m -1 -n -1          Bind m and n to the problem size to\n" );
				printf( "                                    keep all matrices square.\n" );
				printf( "               -m -1 -n 500         Bind m to the problem size, but hold\n" );
				printf( "                                    n = 500 constant.\n" );
				}
				else if ( is_herk( params ) )
				{
				printf( "               -m -1 -k -1          Bind m and k to the problem size to\n" );
				printf( "                                    keep both matrices square.\n" );
				printf( "               -m -1 -k 200         Bind m to the problem size, but hold\n" );
				printf( "                                    k = 200 constant.\n" );
				}
				else if ( is_trmm( params ) || is_trsm( params ) )
				{
				printf( "               -m -1 -n -1          Bind m and n to the problem size to\n" );
				printf( "                                    keep both matrices square.\n" );
				printf( "               -m -1 -n 300         Bind m to the problem size, but hold\n" );
				printf( "                                    n = 300 constant.\n" );
				}
				printf( "\n" );
				printf( "    -r num\n" );
				printf( "            When measuring performance for a given problem size, perform num\n" );
				printf( "            repetitions and report performance using the best timing.\n" );
				printf( "\n" );
				if ( is_gemm( params ) || is_hemm( params ) || is_herk( params ) )
				{
				printf( "    -a alpha\n" );
				printf( "    -b beta\n" );
				printf( "            Specify the value to use for the alpha and/or beta scalars.\n" );
				}
				else // if ( is_trmm( params ) || is_trsm( params ) )
				{
				printf( "    -a alpha\n" );
				printf( "            Specify the value to use for the alpha scalar.\n" );
				}
				printf( "\n" );
				printf( "  If any of the computational options is not specified, its default value will\n" );
				printf( "  be used. (Please use the -v option to see how the driver is interpreting each\n" );
				printf( "  option.)\n" );
				printf( "\n" );
				printf( "  The following IO options are also supported:\n" );
				printf( "\n" );
				printf( "    -q\n" );
				printf( "    -v\n" );
				printf( "            Enable quiet or verbose output. (By default, output is quiet.)\n" );
				printf( "            The verbose option is useful if you are unsure whether your options\n" );
				printf( "            are being interpreted as you intended.\n" );
				printf( "\n" );
				printf( "    -h\n" );
				printf( "            Display this help and exit.\n" );
				printf( "\n" );
				printf( "\n" );

				exit(0);

				break;
			}


			case '?':
			printf( "%s: unexpected option '%c' given or missing option argument\n", bin, state.optopt );
			exit(1);
			break;

			default:
			printf( "%s: unexpected option chararcter returned from getopt: %c\n", bin, opt_ch );
			exit(1);
		}
	}

	// Process the command line options from strings to integers/enums/doubles,
	// as appropriate.
	proc_params( params );

	// Inform the user of the values that were chosen (or defaulted to).
	if ( params->verbose )
	{
		const char* def_str = " (default)";
		const char* nul_str = " ";

		printf( "%%\n" );
		printf( "%% operation:              %s\n", params->opname );
		printf( "%% parameter combination:  %s%s\n", params->pc_str, ( gave_option_c ? nul_str : def_str ) );
		printf( "%% datatype:               %s%s\n", params->dt_str, ( gave_option_d ? nul_str : def_str ) );
		printf( "%% storage combination:    %s%s\n", params->sc_str, ( gave_option_s ? nul_str : def_str ) );
		printf( "%% induced method:         %s%s\n", params->im_str, ( gave_option_i ? nul_str : def_str ) );
		printf( "%% problem size range:     %s%s\n", params->ps_str, ( gave_option_p ? nul_str : def_str ) );
		printf( "%% m dim specifier:        %s%s\n", params->m_str, ( gave_option_m ? nul_str : def_str ) );
		if ( is_gemm( params ) || is_hemm( params ) || is_trmm( params ) || is_trsm( params )  )
		printf( "%% n dim specifier:        %s%s\n", params->n_str, ( gave_option_n ? nul_str : def_str ) );
		if ( is_gemm( params ) || is_herk( params ) )
		printf( "%% k dim specifier:        %s%s\n", params->k_str, ( gave_option_k ? nul_str : def_str ) );
		printf( "%% number of repeats:      %s%s\n", params->nr_str, ( gave_option_r ? nul_str : def_str ) );
		printf( "%% alpha scalar:           %s%s\n", params->alpha_str, ( gave_option_a ? nul_str : def_str ) );
		if ( is_gemm( params ) || is_hemm( params ) || is_herk( params ) )
		printf( "%% beta scalar:            %s%s\n", params->beta_str, ( gave_option_b ? nul_str : def_str ) );
		printf( "%% ---\n" );
		printf( "%% implementation:         %s\n", params->impl );
		if ( params->nt == -1 )
		printf( "%% number of threads:      %s\n", "unset (defaults to 1)" );
		else
		printf( "%% number of threads:      %ld\n", params->nt );
		printf( "%% thread affinity:        %s\n", ( params->af_str == NULL ? "unset" : params->af_str ) );
		printf( "%%\n" );
	}


	// If there are still arguments remaining after getopt() processing is
	// complete, print an error.
	if ( state.optind < argc )
	{
		printf( "%s: encountered unexpected non-option argument: %s\n", bin, argv[ state.optind ] );
		exit(1);
	}
}

// -----------------------------------------------------------------------------

void proc_params( params_t* params )
{
	dim_t nt;

	// Binary name doesn't need any conversion.

	// Operation name doesn't need any conversion.

	// Implementation name doesn't need any conversion.

	// Query the multithreading strings and convert them to integers.
	if ( strncmp( params->impl, "blis", MAX_STRING_SIZE ) == 0 )
	{
		nt = bli_thread_get_num_threads();
	}
	else if ( strncmp( params->impl, "mkl", MAX_STRING_SIZE ) == 0 )
	{
		nt = bli_env_get_var( "OMP_NUM_THREADS", -1 );

		if ( nt == -1 ) nt = bli_env_get_var( "MKL_NUM_THREADS", -1 );
	}
	else if ( strncmp( params->impl, "openblas", MAX_STRING_SIZE ) == 0 )
	{
		nt = bli_env_get_var( "OMP_NUM_THREADS", -1 );

		if ( nt == -1 ) nt = bli_env_get_var( "OPENBLAS_NUM_THREADS", -1 );
	}
	else
	{
		nt = bli_env_get_var( "OMP_NUM_THREADS", -1 );
	}

	// Store nt to the params_t struct.
	params->nt = ( long int )nt;

	// Store the affinity string pointer to the params_t struct.
	params->af_str = bli_env_get_str( "GOMP_CPU_AFFINITY" );

#if 0
	dim_t nt    = bli_thread_get_num_threads();
	dim_t jc_nt = bli_thread_get_jc_nt();
	dim_t pc_nt = bli_thread_get_pc_nt();
	dim_t ic_nt = bli_thread_get_ic_nt();
	dim_t jr_nt = bli_thread_get_jr_nt();
	dim_t ir_nt = bli_thread_get_ir_nt();

	if (    nt == -1 ) nt    = 1;
	if ( jc_nt == -1 ) jc_nt = 1;
	if ( pc_nt == -1 ) pc_nt = 1;
	if ( ic_nt == -1 ) ic_nt = 1;
	if ( jr_nt == -1 ) jr_nt = 1;
	if ( ir_nt == -1 ) ir_nt = 1;

	params->nt    = ( long int )nt;
	params->jc_nt = ( long int )jc_nt;
	params->pc_nt = ( long int )pc_nt;
	params->ic_nt = ( long int )ic_nt;
	params->jr_nt = ( long int )jr_nt;
	params->ir_nt = ( long int )ir_nt;
#endif

	// Parameter combinations, datatype, and operand storage combination,
	// need no conversion.

	// Convert the datatype to a num_t.
	bli_param_map_char_to_blis_dt( params->dt_str[0], &params->dt );

	// Parse the induced method to the corresponding ind_t.
	if    ( strncmp( params->im_str, "native", 6 ) == 0 ||
	        strncmp( params->im_str, "nat",    3 ) == 0 )
	{
		params->im         = BLIS_NAT;
		params->im_is_auto = FALSE;
	}
	else if ( strncmp( params->im_str, "1m",     2 ) == 0 )
	{
		params->im         = BLIS_1M;
		params->im_is_auto = FALSE;
	}
	else if ( strncmp( params->im_str, "auto",   4 ) == 0 )
	{
		params->im         = BLIS_1M;
		params->im_is_auto = TRUE;
	}
	else
	{
		printf( "%s: invalid induced method '%s'.\n", params->bin, params->im_str );
		exit(1);
	}

	// Convert the problem size range and dimension specifier strings to
	// integers.
	sscanf( params->ps_str, "%ld %ld %ld", &(params->sta),
	                                       &(params->end),
	                                       &(params->inc) );
	sscanf( params->m_str, "%ld", &(params->m) );
	sscanf( params->n_str, "%ld", &(params->n) );
	sscanf( params->k_str, "%ld", &(params->k) );

	// Convert the number of repeats to an integer.
	sscanf( params->nr_str, "%ld", &(params->nr) );

	// Convert the alpha and beta strings to doubles.
	//params->alpha = ( double )atof( params->alpha_str );
	//params->beta  = ( double )atof( params->beta_str );
	//sscanf( params->alpha_str, "%lf", &(params->alpha) );
	//sscanf( params->beta_str,  "%lf", &(params->beta) );
	params->alpha = strtod( params->alpha_str, NULL );
	params->beta  = strtod( params->beta_str,  NULL );
}

// -----------------------------------------------------------------------------

bool is_match( const char* str1, const char* str2 )
{
	if ( strncmp( str1, str2, MAX_STRING_SIZE ) == 0 ) return TRUE;
	return FALSE;
}

bool is_gemm( params_t* params )
{
	if ( is_match( params->opname, "gemm" ) ) return TRUE;
	return FALSE;
}

bool is_hemm( params_t* params )
{
	if ( is_match( params->opname, "hemm" ) ) return TRUE;
	return FALSE;
}

bool is_herk( params_t* params )
{
	if ( is_match( params->opname, "herk" ) ) return TRUE;
	return FALSE;
}

bool is_trmm( params_t* params )
{
	if ( is_match( params->opname, "trmm" ) ) return TRUE;
	return FALSE;
}

bool is_trsm( params_t* params )
{
	if ( is_match( params->opname, "trsm" ) ) return TRUE;
	return FALSE;
}

