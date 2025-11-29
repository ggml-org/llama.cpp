#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"
delay=0.1

#sys="blis"
#sys="stampede2"
#sys="lonestar5"
#sys="ul252"
sys="ul264"

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 16 18 20 22 1 3 5 7 9 11 13 15 17 19 21 23"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103"

# Most systems don't run the executables through anything else, but ul264
# uses numactl.
runcmd=""

if [ ${sys} = "blis" ]; then

	export GOMP_CPU_AFFINITY="0-3"

	threads="jc1ic1jr1_2400
	         jc2ic3jr2_6000
	         jc4ic3jr2_8000"

elif [ ${sys} = "stampede2" ]; then

	echo "Need to set GOMP_CPU_AFFINITY."
	exit 1

	threads="jc1ic1jr1_2400
	         jc4ic6jr1_6000
	         jc4ic12jr1_8000"

elif [ ${sys} = "lonestar5" ]; then

	export GOMP_CPU_AFFINITY="0-23"

	# A hack to use libiomp5 with gcc.
	#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64"

	#threads="jc1ic1jr1_2400
	#         jc2ic3jr2_4800
	#         jc4ic3jr2_9600"
	threads="jc1ic1jr1_2400
	         jc4ic3jr2_7200"

elif [ ${sys} = "ul252" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-51"

	threads="jc1ic1jr1_2400
	         jc2ic13jr1_6000
	         jc4ic13jr1_8000"

elif [ ${sys} = "ul264" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-63"

	#threads="jc1ic1jr1_2400"
	threads="jc1ic1jr1_2400
	         jc1ic8jr4_4800
	         jc2ic8jr4_7200"

	#runcmd="numactl -i all"
fi

# Datatypes to test.
test_dts="s d c z"

# Operations to test.
#test_ops="gemm hemm herk trmm trsm"
test_ops="gemm"

# Implementations to test.
#test_impls="openblas vendor asm_blis 1m_blis"
#test_impls="asm_blis 1m_blis"
#test_impls="asm_blis"
test_impls="asm_blis 1m_blis"

# Save a copy of GOMP_CPU_AFFINITY so that if we have to unset it, we can
# restore the value.
GOMP_CPU_AFFINITYsave=${GOMP_CPU_AFFINITY}


# First perform real test cases.
for th in ${threads}; do

	# Start with one way of parallelism in each loop. We will now begin
	# parsing the 'th' variable to update one or more of these threading
	# parameters.
	jc_nt=1; pc_nt=1; ic_nt=1; jr_nt=1; ir_nt=1

	# Strip everything before and after the underscore so that what remains
	# is the problem size and threading parameter string, respectively.
	psize=${th##*_}; thinfo=${th%%_*}

	# Identify each threading parameter and insert a space before it.
	thsep=$(echo -e ${thinfo} | sed -e "s/\([jip][cr]\)/ \1/g" )

	nt=1

	for loopnum in ${thsep}; do

		# Given the current string, which identifies a loop and the
		# number of ways of parallelism for that loop, strip out
		# the ways and loop separately to identify each.
		loop=$(echo -e ${loopnum} | sed -e "s/[0-9]//g" )
		num=$(echo -e ${loopnum} | sed -e "s/[a-z]//g" )

		# Construct a string that we can evaluate to set the number
		# of ways of parallelism for the current loop.
		loop_nt_eq_num="${loop}_nt=${num}"

		# Update the total number of threads.
		nt=$(expr ${nt} \* ${num})

		# Evaluate the string to assign the ways to the variable.
		eval ${loop_nt_eq_num}

	done

	echo "Switching to: jc${jc_nt} pc${pc_nt} ic${ic_nt} jr${jr_nt} ir${ir_nt} (nt = ${nt}) p_max${psize}"


	for dt in ${test_dts}; do

		for im in ${test_impls}; do

			if [ "${dt}" = "s"       -o "${dt}" = "d"         ] && \
			   [ "${im}" = "1m_blis" ]; then
				continue
			fi

			for op in ${test_ops}; do

				# Eigen does not support multithreading for hemm, herk, trmm,
				# or trsm. So if we're getting ready to execute an Eigen driver
				# for one of these operations and nt > 1, we skip this test.
				if [ "${im}"  = "eigen" ] && \
				   [ "${op}" != "gemm"  ] && \
				   [ "${nt}" != "1"     ]; then
					continue;
				fi

				# Find the threading suffix by probing the executable.
				binname=$(ls ${exec_root}_${dt}${op}_${psize}_${im}_*.x)
				suf_ext=${binname##*_}
				suf=${suf_ext%%.*}

				#echo "found file: ${binname} with suffix ${suf}"

				# Set the number of threads according to th.
				if [ "${suf}" = "1s" ] || [ "${suf}" = "2s" ]; then

					# Set the threading parameters based on the implementation
					# that we are preparing to run.
					if   [ "${im}" = "asm_blis"  ] || \
					     [ "${im}" = "1m_blis" ]; then
						unset  OMP_NUM_THREADS
						export BLIS_JC_NT=${jc_nt}
						export BLIS_PC_NT=${pc_nt}
						export BLIS_IC_NT=${ic_nt}
						export BLIS_JR_NT=${jr_nt}
						export BLIS_IR_NT=${ir_nt}
					elif [ "${im}" = "openblas" ]; then
						unset  OMP_NUM_THREADS
						export OPENBLAS_NUM_THREADS=${nt}
					elif [ "${im}" = "eigen" ]; then
						export OMP_NUM_THREADS=${nt}
					elif [ "${im}" = "vendor" ]; then
						unset  OMP_NUM_THREADS
						export MKL_NUM_THREADS=${nt}
					fi
					export nt_use=${nt}

					# Multithreaded OpenBLAS seems to have a problem running
					# properly if GOMP_CPU_AFFINITY is set. So we temporarily
					# unset it here if we are about to execute OpenBLAS, but
					# otherwise restore it.
					if [ ${im} = "openblas" ]; then
						unset GOMP_CPU_AFFINITY
					else
						export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITYsave}"
					fi
				else

					export BLIS_JC_NT=1
					export BLIS_PC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
					export OMP_NUM_THREADS=1
					export OPENBLAS_NUM_THREADS=1
					export MKL_NUM_THREADS=1
					export nt_use=1
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${dt}${op}_${psize}_${im}_${suf}.x"

				# Construct the name of the output file.
				out_file="${out_root}_${suf}_${dt}${op}_${im}.m"

				#echo "Running (nt = ${nt_use}) ./${exec_name} > ${out_file}"
				echo "Running: ${runcmd} ./${exec_name} > ${out_file}"

				# Run executable.
				#./${exec_name} > ${out_file}
				#numactl -i all ./${exec_name} > ${out_file}
				eval "${runcmd} ./${exec_name} > ${out_file}"

				sleep ${delay}

			done
		done
	done
done

