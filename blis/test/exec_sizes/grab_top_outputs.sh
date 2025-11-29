#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
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

output_file="output_mem_sizes.txt"

exec_prefix="test"
exec_blis="blis1 blis2 blis3 blis4 blis5 blis6"
exec_oblas="oblas1 oblas2 oblas3 oblas4 oblas5 oblas6"
exec_ablas="ablas1 ablas2 ablas3 ablas4 ablas5 ablas6"
exec_mblas="mblas1 mblas2 mblas3 mblas4 mblas5 mblas6"

execs="${exec_blis} ${exec_oblas} ${exec_ablas} ${exec_mblas}"

# Send column labels to the output file.
top -n 1 -b | grep COMMAND >> ${output_file}

for e in ${execs}; do

	exec_name="${exec_prefix}_${e}.x"

	echo "Capturing ${exec_name}..."

	./${exec_name} &

	sleep 1

	top -n 1 -b | grep "${exec_prefix}" >> ${output_file}

	pkill "${exec_name}" > /dev/null
done

