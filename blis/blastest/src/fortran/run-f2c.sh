#!/bin/bash

# This script converts netlib [sdcz]blat[123].f files from Fortran to C.
# 

# Start by converting to C with f2c. 
# Options used:
#  -A  Produce ANSI C (instead of old-style C).
#  -R  Do  not promote REAL functions and operations to DOUBLE PRECISION.
#  -a  Make local variables automatic rather than static (unless they
#      appear in a DATA, EQUIVALENCE, NAMELIST, or SAVE statement).
f2c -A -R -a *.f

# Add 'const' qualifier to certain function delcarations so they match
# the prototypes taken from libf2c.
recursive-sed.sh -c "s/s_cmp(char \*, char/s_cmp(const char \*, const char/g" -p "*.c"
recursive-sed.sh -c "s/s_copy(char \*, char/s_copy(char \*, const char/g" -p "*.c"
recursive-sed.sh -c "s/d_cnjg(doublecomplex \*, doublecomplex/d_cnjg(doublecomplex *, const doublecomplex/g" -p "*.c"
recursive-sed.sh -c "s/d_imag(doublecomplex/d_imag(const doublecomplex/g" -p "*.c"
recursive-sed.sh -c "s/c_abs(complex/c_abs(const complex/g" -p "*.c"
recursive-sed.sh -c "s/z_abs(doublecomplex/c_abs(const doublecomplex/g" -p "*.c"

# Use main() and 'void' instead of MAIN__ and VOID.
recursive-sed.sh -c "s/MAIN__/main/g" -p "*.c"
recursive-sed.sh -c "s/VOID/void/g" -p "*.c"

# Add prefix to calls to epsilon_() based on the file in which the
# function is called. [sd]_epsilon_() are not libf2c functions, but
# they are present in the local subset of libf2c used to link the
# BLAS testsuite drivers. 
recursive-sed.sh -c "s/epsilon_/s_epsilon_/g" -p "[sc]*.c"
recursive-sed.sh -c "s/epsilon_/d_epsilon_/g" -p "[dz]*.c"

# The dsdot_() check needs s_epsilon_(), not d_epsilon_().
recursive-sed.sh -c "s/real d_epsilon_()/real s_epsilon_()/g" -p "d*1.c"
recursive-sed.sh -c "s/d_epsilon_(\&c_b81)/s_epsilon_(\&c_b81)/g" -p "d*1.c"

# Fix type inconsistencies in the original Fortran file vis-a-vis
# epsilon() and abs().
recursive-sed.sh -c "s/real d_epsilon_(doublereal/double d_epsilon_(doublereal/g" -p "[dz]*.c"
recursive-sed.sh -c "s/c_abs/z_abs/g" -p "z*.c"

# Fix missing braces around struct initializers.
recursive-sed.sh -c "s/equiv_3 = {/equiv_3 = {{/g" -p "[sd]*1.c"
recursive-sed.sh -c "s/equiv_7 = {/equiv_7 = {{/g" -p "[sd]*1.c"
recursive-sed.sh -c "s/0., 0., 0. }/0., 0., 0. }}/g" -p "d*1.c"
recursive-sed.sh -c "s/2.9, .2, -4. }/2.9, .2, -4. }}/g" -p "d*1.c"
recursive-sed.sh -c "s/0.f, 0.f, 0.f }/0.f, 0.f, 0.f }}/g" -p "s*1.c"
recursive-sed.sh -c "s/-4.f };/-4.f }};/g" -p "s*1.c"

# Convert from brain-dead f2c complex calling conventions to normal
# return-based conventions.
subst1='\n#ifdef BLIS_ENABLE_COMPLEX_RETURN_INTEL\n&\n#else\n'
subst2='\n#endif\n'
recursive-sed.sh -c "s/ void cdotc_(complex \*,/${subst1}complex cdotc_(${subst2}/g" -p "c*1.c"
recursive-sed.sh -c "s/ void cdotu_(complex \*,/${subst1}complex cdotu_(${subst2}/g" -p "c*1.c"
recursive-sed.sh -c "s/\(.*\)cdotc_(&q__1,/${subst1}\1q__1 = cdotc_(${subst2}\1/g" -p "c*1.c"
recursive-sed.sh -c "s/\(.*\)cdotu_(&q__1,/${subst1}\1q__1 = cdotu_(${subst2}\1/g" -p "c*1.c"

recursive-sed.sh -c "s/ void zdotc_(doublecomplex \*,/${subst1}doublecomplex zdotc_(${subst2}/g" -p "z*1.c"
recursive-sed.sh -c "s/ void zdotu_(doublecomplex \*,/${subst1}doublecomplex zdotu_(${subst2}/g" -p "z*1.c"
recursive-sed.sh -c "s/\(.*\)zdotc_(\&z__1,/${subst1}\1z__1 = zdotc_(${subst2}\1/g" -p "z*1.c"
recursive-sed.sh -c "s/\(.*\)zdotu_(\&z__1,/${subst1}\1z__1 = zdotu_(${subst2}\1/g" -p "z*1.c"

