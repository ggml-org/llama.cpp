#!/bin/sh
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2015, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# auto-detect.sh
#
# Zhang Xianyi
#


main()
{
	if [ clang -v > /dev/null 2>&1 ]; then
	    CC=clang
	else
	    CC=gcc
    fi
	CPUID_SRC=cpuid_x86.c
	CPUID_BIN=blis_cpu_detect
	ARCH=generic

	# The name of the script, stripped of any preceeding path.
	script_name=${0##*/}

	# The path to the script. We need this to find the top-level directory
	# of the source distribution in the event that the user has chosen to
	# build elsewhere.
	dist_path=${0%/${script_name}}

	# The path to the directory in which we are building. We do this to
	# make explicit that we distinguish between the top-level directory
	# of the distribution and the directory in which we are building.
	cur_dirpath="."

	#
	# Detect architecture by predefined macros
	#

	out1=`$CC -E ${dist_path}/arch_detect.c`

	ARCH=`echo $out1 | grep -o "ARCH_[a-zA-Z0-9_]*" | head -n1`

	if [ $ARCH = "ARCH_X86_64" ]; then
		CPUID_SRC=cpuid_x86.c
	elif [ $ARCH = "ARCH_X86" ]; then
		CPUID_SRC=cpuid_x86.c
	elif [ $ARCH = "ARCH_ARM" ]; then
		CPUID_SRC=cpuid_arm.c
	elif [ $ARCH = "ARCH_AARCH64" ]; then
		# Only support armv8 now
		echo "armv8a"
		return 0
	else
		echo "generic"
		return 0
	fi

	#
	# Detect CPU cores
	#

	$CC -o ${cur_dirpath}/$CPUID_BIN ${dist_path}/$CPUID_SRC
	${cur_dirpath}/$CPUID_BIN
	rm -rf ${cur_dirpath}/$CPUID_BIN

	# Exit peacefully.
	return 0
}


# The script's main entry point, passing all parameters given.
main "$@"
