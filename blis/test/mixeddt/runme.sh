#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

sys="blis"
#sys="stampede2"
#sys="lonestar5"
#sys="ul252"
sys="tx2"

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 2 4 6 1 3 5 7"
#export GOMP_CPU_AFFINITY="0 4 1 5 2 6 3 7"
#export GOMP_CPU_AFFINITY="0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 32 33 36 37 40 41 44 45"
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 16 18 20 22 1 3 5 7 9 11 13 15 17 19 21 23"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
export GOMP_CPU_AFFINITY="0 1 2 3"

# Modify LD_LIBRARY_PATH.
if [ ${sys} = "blis" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

	export GOMP_CPU_AFFINITY="0 1 2 3"

	jc_nt=1 # 5th loop
	ic_nt=4 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "stampede2" ]; then

	echo "Need to set GOMP_CPU_AFFINITY."
	exit 1

	jc_nt=4 # 5th loop
	ic_nt=12 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=48

elif [ ${sys} = "lonestar5" ]; then

	echo "Need to set GOMP_CPU_AFFINITY."
	exit 1

	# A hack to use libiomp5 with gcc.
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64"

	jc_nt=2 # 5th loop
	ic_nt=12 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=24

elif [ ${sys} = "ul252" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103"
	export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51"

	#jc_nt=4 # 5th loop
	jc_nt=2 # 5th loop
	ic_nt=13 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	#nt=52
	nt=26
elif [ ${sys} = "tx2" ]; then

	export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55"

        jc_nt=8 # 5th loop
        ic_nt=7 # 3rd loop
        jr_nt=1 # 2nd loop
        ir_nt=1 # 1st loop
        nt=56

fi

# Save a copy of GOMP_CPU_AFFINITY so that if we have to unset it, we can
# restore the value.
GOMP_CPU_AFFINITYsave=${GOMP_CPU_AFFINITY}

# Datatypes to test.
#dts="s d c z"

# Threadedness to test.
threads="mt"
#threads="st"

# Implementations to test.
test_impls="ad_hoc intern"

# Operations to test.
l3_ops="gemm"
test_ops="${l3_ops}"

# Define the list of datatype chars and precision chars.
dt_chars="s d c z"
pr_chars="s d"

# Construct the datatype combination strings.
dt_combos=""
for dtc in ${dt_chars}; do
	for dta in ${dt_chars}; do
		for dtb in ${dt_chars}; do
			for pre in ${pr_chars}; do
				dt_combos="${dt_combos} ${dtc}${dta}${dtb}${pre}"
			done
		done
	done
done

# Overrides, in case something goes wrong for a subset of tests.
#test_impls="ad_hoc"
#dt_combos="ssss sssd ssds sdss dsss ddds dddd"


# Now perform complex test cases.
for th in ${threads}; do

	for dt in ${dt_combos}; do

		for im in ${test_impls}; do

			for op in ${test_ops}; do

				# Set the number of threads according to th.
				if [ ${th} = "mt" ]; then

					export BLIS_JC_NT=${jc_nt}
					export BLIS_IC_NT=${ic_nt}
					export BLIS_JR_NT=${jr_nt}
					export BLIS_IR_NT=${ir_nt}
					export OMP_NUM_THREADS=${nt}
					export OPENBLAS_NUM_THREADS=${nt}

					# Unset GOMP_CPU_AFFINITY for OpenBLAS.
					if [ ${im} = "openblas" ]; then
						unset GOMP_CPU_AFFINITY
					else
						export GOMP_CPU_AFFINITY=${GOMP_CPU_AFFINITYsave}
					fi
				else

					export BLIS_JC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
					export OMP_NUM_THREADS=1
					export OPENBLAS_NUM_THREADS=1
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${dt}${op}_${im}_${th}.x"

				# Construct the name of the output file.
				out_file="${out_root}_${th}_${dt}${op}_${im}.m"

				echo "Running (nt = ${OMP_NUM_THREADS}) ./${exec_name} > ${out_file}"

				# Run executable.
				./${exec_name} > ${out_file}

				#sleep 1

			done
		done
	done
done
