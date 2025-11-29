/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#ifndef BLIS_GENARRAY_MACRO_DEFS_H
#define BLIS_GENARRAY_MACRO_DEFS_H


// -- Macros to generate function arrays ---------------------------------------

// -- "Smart" one-operand macro --

#define GENARRAY_FPA(tname,opname) \
\
static tname PASTECH(opname,_fpa)[BLIS_NUM_FP_TYPES] = \
{ \
	( tname )PASTEMAC(s,opname), \
	( tname )PASTEMAC(c,opname), \
	( tname )PASTEMAC(d,opname), \
	( tname )PASTEMAC(z,opname)  \
}

#define GENARRAYRO_FPA(tname,opname) \
\
static tname PASTECH(opname,_fpa)[BLIS_NUM_FP_TYPES] = \
{ \
	( tname )PASTEMAC(s,opname), \
	( tname )PASTEMAC(d,opname), \
	NULL, \
	NULL \
}

// -- "Smart" one-operand macro (with integer support) --

#define GENARRAY_FPA_I(tname,opname) \
\
static tname PASTECH(opname,_fpa)[BLIS_NUM_FP_TYPES+1] = \
{ \
	( tname )PASTEMAC(s,opname), \
	( tname )PASTEMAC(c,opname), \
	( tname )PASTEMAC(d,opname), \
	( tname )PASTEMAC(z,opname), \
	( tname )PASTEMAC(i,opname)  \
}

// -- "Smart" two-operand macro --

#define GENARRAY_FPA2(tname,op) \
\
static tname PASTECH(op,_fpa2)[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ ( tname )PASTEMAC(s,s,op), ( tname )PASTEMAC(s,c,op), ( tname )PASTEMAC(s,d,op), ( tname )PASTEMAC(s,z,op) }, \
	{ ( tname )PASTEMAC(c,s,op), ( tname )PASTEMAC(c,c,op), ( tname )PASTEMAC(c,d,op), ( tname )PASTEMAC(c,z,op) }, \
	{ ( tname )PASTEMAC(d,s,op), ( tname )PASTEMAC(d,c,op), ( tname )PASTEMAC(d,d,op), ( tname )PASTEMAC(d,z,op) }, \
	{ ( tname )PASTEMAC(z,s,op), ( tname )PASTEMAC(z,c,op), ( tname )PASTEMAC(z,d,op), ( tname )PASTEMAC(z,z,op) }  \
}

// -- "Smart" two-operand macro --

/*
#define GENARRAY2_VFP(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ PASTEMAC(s,s,op), PASTEMAC(s,c,op), PASTEMAC(s,d,op), PASTEMAC(s,z,op) }, \
	{ PASTEMAC(c,s,op), PASTEMAC(c,c,op), PASTEMAC(c,d,op), PASTEMAC(c,z,op) }, \
	{ PASTEMAC(d,s,op), PASTEMAC(d,c,op), PASTEMAC(d,d,op), PASTEMAC(d,z,op) }, \
	{ PASTEMAC(z,s,op), PASTEMAC(z,c,op), PASTEMAC(z,d,op), PASTEMAC(z,z,op) }  \
}
*/



// -- One-operand macro --

#define GENARRAY(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES] = \
{ \
	PASTEMAC(s,op), \
	PASTEMAC(c,op), \
	PASTEMAC(d,op), \
	PASTEMAC(z,op)  \
}

#define GENARRAY_I(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES+1] = \
{ \
	PASTEMAC(s,op), \
	PASTEMAC(c,op), \
	PASTEMAC(d,op), \
	PASTEMAC(z,op), \
	PASTEMAC(i,op)  \
}

/*
#define GENARRAYR(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ PASTEMAC(s,s,op), NULL,              PASTEMAC(s,d,op), NULL,             }, \
	{ PASTEMAC(c,s,op), NULL,              PASTEMAC(c,d,op), NULL,             }, \
	{ PASTEMAC(d,s,op), NULL,              PASTEMAC(d,d,op), NULL,             }, \
	{ PASTEMAC(z,s,op), NULL,              PASTEMAC(z,d,op), NULL,             }  \
}
*/



// -- One-operand macro (with custom prefix) --

#define GENARRAY_PREF(arrayname,prefix,op) \
\
arrayname[BLIS_NUM_FP_TYPES] = \
{ \
	PASTECH(prefix,s,op), \
	PASTECH(prefix,c,op), \
	PASTECH(prefix,d,op), \
	PASTECH(prefix,z,op)  \
}



// -- Two-operand macros --


#define GENARRAY2_ALL(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ PASTEMAC(s,s,op), PASTEMAC(s,c,op), PASTEMAC(s,d,op), PASTEMAC(s,z,op) }, \
	{ PASTEMAC(c,s,op), PASTEMAC(c,c,op), PASTEMAC(c,d,op), PASTEMAC(c,z,op) }, \
	{ PASTEMAC(d,s,op), PASTEMAC(d,c,op), PASTEMAC(d,d,op), PASTEMAC(d,z,op) }, \
	{ PASTEMAC(z,s,op), PASTEMAC(z,c,op), PASTEMAC(z,d,op), PASTEMAC(z,z,op) }  \
}


#define GENARRAY2_MIXP(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ PASTEMAC(s,s,op), NULL,              PASTEMAC(s,d,op), NULL,             }, \
	{ NULL,              PASTEMAC(c,c,op), NULL,              PASTEMAC(c,z,op) }, \
	{ PASTEMAC(d,s,op), NULL,              PASTEMAC(d,d,op), NULL,             }, \
	{ NULL,              PASTEMAC(z,c,op), NULL,              PASTEMAC(z,z,op) }  \
}


#define GENARRAY2_EXT(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ PASTEMAC(s,s,op), PASTEMAC(s,c,op), NULL,              NULL,             }, \
	{ PASTEMAC(c,s,op), PASTEMAC(c,c,op), NULL,              NULL,             }, \
	{ NULL,              NULL,              PASTEMAC(d,d,op), PASTEMAC(d,z,op) }, \
	{ NULL,              NULL,              PASTEMAC(z,d,op), PASTEMAC(z,z,op) }  \
}


#define GENARRAY2_MIN(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ PASTEMAC(s,s,op), NULL,              NULL,              NULL,             }, \
	{ NULL,              PASTEMAC(c,c,op), NULL,              NULL,             }, \
	{ NULL,              NULL,              PASTEMAC(d,d,op), NULL,             }, \
	{ NULL,              NULL,              NULL,              PASTEMAC(z,z,op) }  \
}


// -- Three-operand macros --


#define GENARRAY3_ALL(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ \
	{ PASTEMAC(s,s,s,op), PASTEMAC(s,s,c,op), PASTEMAC(s,s,d,op), PASTEMAC(s,s,z,op) }, \
	{ PASTEMAC(s,c,s,op), PASTEMAC(s,c,c,op), PASTEMAC(s,c,d,op), PASTEMAC(s,c,z,op) }, \
	{ PASTEMAC(s,d,s,op), PASTEMAC(s,d,c,op), PASTEMAC(s,d,d,op), PASTEMAC(s,d,z,op) }, \
	{ PASTEMAC(s,z,s,op), PASTEMAC(s,z,c,op), PASTEMAC(s,z,d,op), PASTEMAC(s,z,z,op) }  \
	}, \
	{ \
	{ PASTEMAC(c,s,s,op), PASTEMAC(c,s,c,op), PASTEMAC(c,s,d,op), PASTEMAC(c,s,z,op) }, \
	{ PASTEMAC(c,c,s,op), PASTEMAC(c,c,c,op), PASTEMAC(c,c,d,op), PASTEMAC(c,c,z,op) }, \
	{ PASTEMAC(c,d,s,op), PASTEMAC(c,d,c,op), PASTEMAC(c,d,d,op), PASTEMAC(c,d,z,op) }, \
	{ PASTEMAC(c,z,s,op), PASTEMAC(c,z,c,op), PASTEMAC(c,z,d,op), PASTEMAC(c,z,z,op) }  \
	}, \
	{ \
	{ PASTEMAC(d,s,s,op), PASTEMAC(d,s,c,op), PASTEMAC(d,s,d,op), PASTEMAC(d,s,z,op) }, \
	{ PASTEMAC(d,c,s,op), PASTEMAC(d,c,c,op), PASTEMAC(d,c,d,op), PASTEMAC(d,c,z,op) }, \
	{ PASTEMAC(d,d,s,op), PASTEMAC(d,d,c,op), PASTEMAC(d,d,d,op), PASTEMAC(d,d,z,op) }, \
	{ PASTEMAC(d,z,s,op), PASTEMAC(d,z,c,op), PASTEMAC(d,z,d,op), PASTEMAC(d,z,z,op) }  \
	}, \
	{ \
	{ PASTEMAC(z,s,s,op), PASTEMAC(z,s,c,op), PASTEMAC(z,s,d,op), PASTEMAC(z,s,z,op) }, \
	{ PASTEMAC(z,c,s,op), PASTEMAC(z,c,c,op), PASTEMAC(z,c,d,op), PASTEMAC(z,c,z,op) }, \
	{ PASTEMAC(z,d,s,op), PASTEMAC(z,d,c,op), PASTEMAC(z,d,d,op), PASTEMAC(z,d,z,op) }, \
	{ PASTEMAC(z,z,s,op), PASTEMAC(z,z,c,op), PASTEMAC(z,z,d,op), PASTEMAC(z,z,z,op) }  \
	} \
}


#define GENARRAY3_EXT(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ \
	{ PASTEMAC(s,s,s,op), PASTEMAC(s,s,c,op), NULL,                NULL,               }, \
	{ PASTEMAC(s,c,s,op), PASTEMAC(s,c,c,op), NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }  \
	}, \
	{ \
	{ PASTEMAC(c,s,s,op), PASTEMAC(c,s,c,op), NULL,                NULL,               }, \
	{ PASTEMAC(c,c,s,op), PASTEMAC(c,c,c,op), NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }  \
	}, \
	{ \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                PASTEMAC(d,d,d,op), PASTEMAC(d,d,z,op) }, \
	{ NULL,                NULL,                PASTEMAC(d,z,d,op), PASTEMAC(d,z,z,op) }  \
	}, \
	{ \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                PASTEMAC(z,d,d,op), PASTEMAC(z,d,z,op) }, \
	{ NULL,                NULL,                PASTEMAC(z,z,d,op), PASTEMAC(z,z,z,op) }  \
	} \
}


#define GENARRAY3_MIN(arrayname,op) \
\
arrayname[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] = \
{ \
	{ \
	{ PASTEMAC(s,s,s,op), NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }  \
	}, \
	{ \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                PASTEMAC(c,c,c,op), NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }  \
	}, \
	{ \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                PASTEMAC(d,d,d,op), NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }  \
	}, \
	{ \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                NULL,               }, \
	{ NULL,                NULL,                NULL,                PASTEMAC(z,z,z,op) }  \
	} \
}


#endif
