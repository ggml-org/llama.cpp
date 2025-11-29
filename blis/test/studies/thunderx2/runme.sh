#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

out_rootdir=$(date +%Y%m%d)
#out_rootdir=20180830
mkdir -p $out_rootdir

sys="thunderx2"

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55"
unset GOMP_CPU_AFFINITY

# Threading scheme to use when multithreading
if [ ${sys} = "blis" ]; then

	jc_nt=1 # 5th loop
	ic_nt=4 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "thunderx2" ]; then

	jc_1_nt=2 # 5th loop
	ic_1_nt=14 # 3rd loop
	jr_1_nt=1 # 2nd loop
	ir_1_nt=1 # 1st loop
	nt_1=28
	jc_2_nt=4 # 5th loop
	ic_2_nt=14 # 3rd loop
	jr_2_nt=1 # 2nd loop
	ir_2_nt=1 # 1st loop
	nt_2=56
fi

# Threadedness to test.
#threads="mt1 mt2"
#threads_r="mt"
#threads="st"
#threads_r="st"

# Datatypes to test.
dts="c z"
dts_r="s d"

# Operations to test.
#l3_ops="gemm syrk hemm trmm"
l3_ops="gemm"
test_ops="${l3_ops}"
test_ops_r="${l3_ops}"

# Complex domain implementations to test.
if [ ${sys} = "blis" ]; then

	#test_impls="openblas mkl 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis"
	test_impls="openblas 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"

elif [ ${sys} = "thunderx2" ]; then

	#test_impls="openblas"
	#test_impls="armpl"
	#test_impls="1m_blis armpl"
	test_impls="openblas armpl 1m_blis"
fi

# Real domain implementations to test.
test_impls_r="openblas armpl asm_blis"
#test_impls_r="openblas"
#test_impls_r="asm_blis"
#test_impls_r="armpl"

cores_r="1 28 56"
cores="1 28 56"


# First perform real test cases.
for nc in ${cores_r}; do
	for dt in ${dts_r}; do

		for im in ${test_impls_r}; do

			for op in ${test_ops_r}; do
				# Set the number of threads according to th.
				if [ ${nc} -gt 1 ]; then
					# Unset GOMP_CPU_AFFINITY for MKL when using mkl_intel_thread.
					if [ ${im} = "openblas" ]; then
						unset GOMP_CPU_AFFINITY
					elif [ ${im} = "armpl" ]; then
						unset GOMP_CPU_AFFINITY
					else
						export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55"

					fi
					if [ ${nc} -eq 28 ]; then

						export BLIS_JC_NT=${jc_1_nt}
						export BLIS_IC_NT=${ic_1_nt}
						export BLIS_JR_NT=${jr_1_nt}
						export BLIS_IR_NT=${ir_1_nt}
						export OMP_NUM_THREADS=${nt_1}
						out_dir="${out_rootdir}/1socket"
						mkdir -p $out_rootdir/1socket
					elif [ ${nc} -eq 56 ]; then
                                        	export BLIS_JC_NT=${jc_2_nt}
                                        	export BLIS_IC_NT=${ic_2_nt}
                                        	export BLIS_JR_NT=${jr_2_nt}
                                        	export BLIS_IR_NT=${ir_2_nt}
                                        	export OMP_NUM_THREADS=${nt_2}
						out_dir="${out_rootdir}/2sockets"
						mkdir -p $out_rootdir/2sockets
					fi
					th="mt"
				else

					export BLIS_JC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
					export OMP_NUM_THREADS=1
					out_dir="${out_rootdir}/st"
					mkdir -p $out_rootdir/st
					th="st"
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${dt}${op}_${im}_${th}.x"

				# Construct the name of the output file.
				out_file="${out_dir}/${out_root}_${th}_${dt}${op}_${im}.m"

				echo "Running (nt = ${OMP_NUM_THREADS}) ./${exec_name} > ${out_file}"

				# Run executable.
				./${exec_name} > ${out_file}

				sleep 1

			done
		done
	done
done

# Now perform complex test cases.
for nc in ${cores}; do

	for dt in ${dts}; do

		for im in ${test_impls}; do

			for op in ${test_ops}; do

				# Set the number of threads according to th.
				if [ ${nc} -gt 1 ]; then
					# Unset GOMP_CPU_AFFINITY for MKL when using mkl_intel_thread.
					if [ ${im} = "openblas" ]; then
						unset GOMP_CPU_AFFINITY
					elif [ ${im} = "armpl" ]; then
						unset GOMP_CPU_AFFINITY
					else
						export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55"

					fi
					if [ ${nc} -eq 28 ]; then

						export BLIS_JC_NT=${jc_1_nt}
						export BLIS_IC_NT=${ic_1_nt}
						export BLIS_JR_NT=${jr_1_nt}
						export BLIS_IR_NT=${ir_1_nt}
						export OMP_NUM_THREADS=${nt_1}
						out_dir="${out_rootdir}/1socket"
					elif [ ${nc} -eq 56 ]; then
                                        	export BLIS_JC_NT=${jc_2_nt}
                                        	export BLIS_IC_NT=${ic_2_nt}
                                        	export BLIS_JR_NT=${jr_2_nt}
                                        	export BLIS_IR_NT=${ir_2_nt}
                                        	export OMP_NUM_THREADS=${nt_2}
						out_dir="${out_rootdir}/2sockets"
					fi
					th="mt"
				else

                                        export BLIS_JC_NT=1
                                        export BLIS_IC_NT=1
                                        export BLIS_JR_NT=1
                                        export BLIS_IR_NT=1
					export OMP_NUM_THREADS=1
					out_dir="${out_rootdir}/st"
					th="st"
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${dt}${op}_${im}_${th}.x"

				# Construct the name of the output file.
				out_file="${out_dir}/${out_root}_${th}_${dt}${op}_${im}.m"

				echo "Running (nt = ${OMP_NUM_THREADS}) ./${exec_name} > ${out_file}"
				# Run executable.
				./${exec_name} > ${out_file}

				sleep 1

			done
		done
	done
done
