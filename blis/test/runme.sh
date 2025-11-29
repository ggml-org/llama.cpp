#!/bin/bash

exec_root="test"
out_root="output"
#out_root="output_square"

# Operations to test.
l2_ops="gemv ger hemv her her2 trmv trsv"
l3_ops="gemm hemm herk her2k trmm trsm"
test_ops="${l2_ops} ${l3_ops}"

# Implementations to test.
#test_impls="openblas mkl blis"
test_impls="blis"

for im in ${test_impls}; do

	for op in ${test_ops}; do

		# Construct the name of the test executable.
		exec_name="${exec_root}_${op}_${im}.x"

		# Construct the name of the output file.
		out_file="${out_root}_${op}_${im}.m"

		echo "Running ${exec_name} > ${out_file}"

		# Run executable.
		./${exec_name} > ${out_file}

		sleep 1

	done
done
