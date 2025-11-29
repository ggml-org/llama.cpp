#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

out_rootdir=$(date +%Y%m%d)
mkdir -p $out_rootdir

#sys="blis"
#sys="stampede"
#sys="stampede2"
#sys="lonestar"
#sys="wahlberg"
#sys="arm-softiron"
sys="skx"

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 2 4 6 1 3 5 7"
#export GOMP_CPU_AFFINITY="0 4 1 5 2 6 3 7"
#export GOMP_CPU_AFFINITY="0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 32 33 36 37 40 41 44 45"
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 16 18 20 22 1 3 5 7 9 11 13 15 17 19 21 23"
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"

# Modify LD_LIBRARY_PATH.
if [ ${sys} = "blis" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

elif [ ${sys} = "stampede" ]; then

	# A hack to use libiomp5 with gcc.
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/13/composer_xe_2013.2.146/compiler/lib/intel64"

elif [ ${sys} = "stampede2" ]; then

	:

elif [ ${sys} = "lonestar" ]; then

	# A hack to use libiomp5 with gcc.
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64"

elif [ ${sys} = "wahlberg" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/flame/lib/acml/5.3.1/gfortran64_int64/lib"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/flame/lib/acml/5.3.1/gfortran64_mp_int64/lib"
fi

# Threading scheme to use when multithreading
if [ ${sys} = "blis" ]; then

	jc_nt=1 # 5th loop
	ic_nt=4 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "stampede" ]; then

	jc_nt=2 # 5th loop
	ic_nt=8 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=16

elif [ ${sys} = "lonestar" ]; then

	jc_nt=2 # 5th loop
	ic_nt=12 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=24

elif [ ${sys} = "wahlberg" ]; then

	jc_nt=1 # 5th loop
	ic_nt=2 # 3rd loop
	jr_nt=2 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "arm-softiron" ]; then

	#jc_nt=1 # 5th loop
	#ic_nt=2 # 3rd loop
	#jr_nt=2 # 2nd loop
	#ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "stampede2" ]; then

	jc_nt=2 # 5th loop
	ic_nt=1 # 3rd loop
	jr_nt=10 # 2nd loop
	ir_nt=1 # 1st loop
	nt=20

elif [ ${sys} = "skx" ]; then

        jc_1_nt=1 # 5th loop
        ic_1_nt=20 # 3rd loop
        jr_1_nt=1 # 2nd loop
        ir_1_nt=1 # 1st loop
        nt_1=20
        jc_2_nt=2 # 5th loop
        ic_2_nt=20 # 3rd loop
        jr_2_nt=1 # 2nd loop
        ir_2_nt=1 # 1st loop
        nt_2=40

fi

# Threadedness to test.
#threads="mt"
#threads_r="mt"
threads="st mt"
threads_r="st mt"

# Datatypes to test.
dts="c z "
#dts="c z"
#dts_r="s d"
dts_r="s d"

# Operations to test.
l3_ops="gemm syrk hemm trmm"
#l3_ops="gemm"
test_ops="${l3_ops}"
test_ops_r="${l3_ops}"

# Complex domain implementations to test.
if [ ${sys} = "blis" ]; then

	#test_impls="openblas mkl 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis"
	test_impls="openblas 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"

elif [ ${sys} = "stampede" ]; then

	test_impls="openblas mkl asm_blis 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"
	#test_impls="openblas mkl asm_blis"

elif [ ${sys} = "stampede2" ]; then

	test_impls="openblas mkl 1m_blis"
	#test_impls="1m_blis"

elif [ ${sys} = "lonestar" ]; then

	test_impls="asm_blis 4mhw_blis 4m1a_blis 1m_blis 3m1_blis"
	#test_impls="1m_blis 3m1_blis"
	#test_impls="4m1a_blis"
	#test_impls="mkl"
	#test_impls="openblas mkl asm_blis"

elif [ ${sys} = "wahlberg" ]; then

	test_impls="openblas acml asm_blis 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"
	test_impls="openblas acml asm_blis"

elif [ ${sys} = "arm-softiron" ]; then

	test_impls="openblas 1m_blis mkl"

elif [ ${sys} = "skx" ]; then

	test_impls="openblas 1m_blis mkl"
fi

# Real domain implementations to test.
test_impls_r="openblas mkl asm_blis"

cores_r="20 40"
cores="20 40"

# First perform real test cases.
for nc in ${cores_r}; do

	for dt in ${dts_r}; do

		for im in ${test_impls_r}; do

			for op in ${test_ops_r}; do

				if [ ${nc} -gt 1 ]; then
                                        # Unset GOMP_CPU_AFFINITY for MKL when using mkl_intel_thread.
                                        if [ ${im} = "openblas" ]; then
                                                unset GOMP_CPU_AFFINITY
                                        else
                                                export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"

                                        fi
                                        if [ ${nc} -eq 20 ]; then

                                                export BLIS_JC_NT=${jc_1_nt}
                                                export BLIS_IC_NT=${ic_1_nt}
                                                export BLIS_JR_NT=${jr_1_nt}
                                                export BLIS_IR_NT=${ir_1_nt}
                                                export OMP_NUM_THREADS=${nt_1}
                                                out_dir="${out_rootdir}/1socket"
                                                mkdir -p $out_rootdir/1socket
                                        elif [ ${nc} -eq 40 ]; then
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

                                        export BLIS_NUM_THREADS=1
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
for nc in ${cores_r}; do

	for dt in ${dts}; do

		for im in ${test_impls}; do

			for op in ${test_ops}; do
				
				if [ ${nc} -gt 1 ]; then
                                        # Unset GOMP_CPU_AFFINITY for MKL when using mkl_intel_thread.
                                        if [ ${im} = "openblas" ]; then
                                                unset GOMP_CPU_AFFINITY
                                        else
                                                export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"

                                        fi
                                        if [ ${nc} -eq 20 ]; then

                                                export BLIS_JC_NT=${jc_1_nt}
                                                export BLIS_IC_NT=${ic_1_nt}
                                                export BLIS_JR_NT=${jr_1_nt}
                                                export BLIS_IR_NT=${ir_1_nt}
                                                export OMP_NUM_THREADS=${nt_1}
                                                out_dir="${out_rootdir}/1socket"
                                        elif [ ${nc} -eq 40 ]; then
                                                export BLIS_JC_NT=${jc_2_nt}
                                                export BLIS_IC_NT=${ic_2_nt}
                                                export BLIS_JR_NT=${jr_2_nt}
                                                export BLIS_IR_NT=${ir_2_nt}
                                                export OMP_NUM_THREADS=${nt_2}
                                                out_dir="${out_rootdir}/2sockets"
                                        fi
                                        th="mt"
                                else

                                        export BLIS_NUM_THREADS=1
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
