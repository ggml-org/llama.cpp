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

static const char OPT_MARKER = '-';

//bool bli_char_is_in_str( char ch, const char* str );

void bli_getopt_init_state( int opterr, getopt_t* state )
{
	state->nextchar = NULL;
	state->optarg   = NULL;
	state->optind   = 1;
	state->opterr   = opterr;
	state->optopt   = 0;
}

int bli_getopt( int argc, const char* const * argv, const char* optstring, getopt_t* state )
{
	const char* elem_str;
	const char* optstr_char;

	// If argv contains no more arguments to process, return.
	if ( state->optind == argc ) return -1;

	// Get a pointer to the current argv element string to process. If
	// nextchar is non-NULL, then it means the previous call processed
	// an element of argv with more than one option character, in which
	// case we need to pick up where we left off (which is the address
	// contained in nextchar).
	if ( state->nextchar == NULL )
	{
		elem_str = argv[ state->optind ];

		// elem_str[0] should be an OPT_MARKER if it is an option. In the
		// event that it is not an option, argv should be permuted so that
		// the non-option argument moves back toward the end of the list.
		// This functionality is not supported/implemented here. Therefore,
		// we require all of the program's option arguments to precede all of
		// its non-option arguments.
		if ( elem_str[0] != OPT_MARKER )
		{
			state->optarg = NULL;
			//state->optind += 1;
			return -1;
		}

		// Skip over the OPT_MARKER.
		elem_str++;
	}
	else
	{
		// Note we don't need to skip the OPT_MARKER here since we are
		// continuing processing of a string with more than one option
		// character.

		// Use the nextchar pointer as our element string.
		elem_str = state->nextchar;

		// Reset nextchar to NULL.
		state->nextchar = NULL;
	}

	// Find the first occurrence of elem_str[0] in optstring.
	optstr_char = strchr( optstring, elem_str[0] );

	// If the option character in elem_str[0] is absent from the option
	// string, store it and return '?'.
	if ( optstr_char == NULL )
	{
		if ( state->opterr == 1 ) fprintf( stderr, "bli_getopt(): **error**: option character '%c' missing from option string \"%s\"\n", elem_str[0], optstring );

		// We can't dereference optstr_char since it is NULL, so we use
		// elem_str[0] instead.
		state->optopt = elem_str[0];
		state->optind += 1;
		return '?';
	}

	// We can now safely assume that an option characer was found in the
	// option string. Now we need to check if the option takes an argument.
	if ( optstr_char[1] == ':' )
	{
		// If the current element string ends after the option character,
		// then the companion argument must be stored in the next element
		// of argv. Otherwise, the argument begins immediately after the
		// option character.
		if ( elem_str[1] == '\0' )
		{
			// If there are no more elements in argv, the argument was
			// omitted. Store the corresponding option character and
			// return '?'.
			if ( state->optind + 1 >= argc )
			{
				if ( state->opterr == 1 ) fprintf( stderr, "bli_getopt(): **error**: option character '%c' is missing an argument (end of argv)\n", elem_str[0] );

				state->optopt = *optstr_char;
				state->optind += 1;
				return '?';
			}
			// If there are still more elements in argv yet to process AND the
			// next one is an option marker, then the argument was omitted
			// (unless the option marker is actually part of the argument,
			// such as with negative numbers, e.g. -1, which is very likely
			// if the char *after* the option marker is missing from optstring).
			else if ( argv[ state->optind + 1 ][0] == OPT_MARKER )
			{
				// If the char after the option marker is present in optstring,
				// then the first option argument is missing.
				if ( strchr( optstring, argv[ state->optind + 1 ][1] ) != NULL )
				{
					if ( state->opterr == 1 ) fprintf( stderr, "bli_getopt(): **error**: option character '%c' is missing an argument (next element of argv is option '%c')\n", elem_str[0], argv[ state->optind + 1 ][1] );

					state->optopt = *optstr_char;
					state->optind += 1;
					return '?';
				}
			}
			// If no error was deteced above, we can safely assign optarg
			// to be the next element in argv and increment optind by two.
			state->optarg = argv[ state->optind + 1 ];
			state->optind += 2;
		}
		else
		{
			// We don't need to check for missing arguments since we know
			// that because the char after the option character is not NULL,
			// the character(s) after it must constitute the argument.

			state->optarg = &elem_str[1];
			state->optind += 1;
		}

		return *optstr_char;
	}

	// The current option character does NOT take an argument. However, we
	// still need to check if the next char is an option argument (such as
	// occurs when the user runs "program -rv" instead of "program -r -v").
	if ( elem_str[1] != '\0' )
	{
		if ( strchr( optstring, elem_str[1] ) != NULL )
		{
			state->nextchar = &elem_str[1];
			return *optstr_char;
		}
	}

	state->optarg = NULL;
	state->optind += 1;
	return *optstr_char;
}

#if 0
bool bli_char_is_in_str( char ch, const char* str )
{
	int chi = ( int )ch;

	if ( strchr( str, chi ) == NULL ) return FALSE;

	return TRUE;
}
#endif
