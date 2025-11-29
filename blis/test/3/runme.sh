#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"
delay=0.1

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 16 18 20 22 1 3 5 7 9 11 13 15 17 19 21 23"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103"

# ------------------

# Problem size range for single- and multithreaded execution. Set psr_st and
# psr_mt on a per-system basis below to override these default values.
psr_st="100 1000 100"
psr_mt="200 2000 200"

sys="blis"
#sys="stampede2"
#sys="lonestar5"
#sys="ul252"
#sys="ul264"
#sys="ul2128"

if [ ${sys} = "blis" ]; then

	export GOMP_CPU_AFFINITY="0-3"

	numactl=""
	threads="jc1ic1jr1_st
	         jc2ic2jr1_mt"
	#psr_st="40 1000 40"
	#psr_mt="40 4000 40"

elif [ ${sys} = "stampede2" ]; then

	echo "Need to set GOMP_CPU_AFFINITY."
	exit 1

	numactl=""
	threads="jc1ic1jr1_st
	         jc4ic12jr1_mt"
	#psr_st="40 1000 40"
	#psr_mt="40 4000 40"

elif [ ${sys} = "lonestar5" ]; then

	export GOMP_CPU_AFFINITY="0-23"

	# A hack to use libiomp5 with gcc.
	#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64"

	numactl=""
	threads="jc1ic1jr1_st
	         jc4ic3jr2_mt"
	#psr_st="40 1000 40"
	#psr_mt="40 4000 40"

elif [ ${sys} = "ul252" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-51"

	numactl=""
	threads="jc1ic1jr1_st
	         jc4ic13jr1_mt"
	#psr_st="40 1000 40"
	#psr_mt="40 4000 40"

elif [ ${sys} = "ul264" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-63"

	numactl="numactl --interleave=all"
	threads="jc1ic1jr1_st
	         jc2ic8jr4_mt"
	#psr_st="40 1000 40"
	#psr_mt="40 4000 40"

elif [ ${sys} = "ul2128" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-127"

	numactl="numactl --interleave=all"
	threads="jc1ic1jr1_st
	         jc8ic4jr4_mt"

	#psr_st="40 1000 40"
	#psr_mt="40 4000 40"
fi

# Datatypes to test.
test_dts="s d c z"
#test_dts="d"

# Operations to test.
test_ops="gemm_nn hemm_ll herk_ln trmm_llnn trsm_runn"
#test_ops="gemm_nn"

# Implementations to test.
test_impls="blis"
#test_impls="openblas"
#test_impls="vendor"
#test_impls="eigen"
#test_impls="all"

if [ "${impls}" = "all" ]; then
	test_impls="openblas blis vendor eigen"
fi

# Number of repeats per problem size.
nrepeats=3

# The induced method to use ('auto', 'native', or '1m') for executing
# complex-domain level-3 operations.
ind="auto"

# Quiet mode?
#quiet="yes"

# For testing purposes.
#dryrun="yes"

# Save a copy of GOMP_CPU_AFFINITY so that if we have to unset it, we can
# restore the value.
GOMP_CPU_AFFINITYsave=${GOMP_CPU_AFFINITY}


# Iterate over the threading configs.
for th in ${threads}; do

	#threads="jc1ic1jr1_st
	#         jc8ic4jr4_mt"

	# Start with one way of parallelism in each loop. We will now begin
	# parsing the 'th' variable to update one or more of these threading
	# parameters.
	jc_nt=1; pc_nt=1; ic_nt=1; jr_nt=1; ir_nt=1

	# Strip everything before the understore so that what remains is the
	# threading suffix.
	tsuf=${th##*_};

	# Strip everything after the understore so that what remains is the
	# parallelism (threading) info.
	thinfo=${th%%_*}

	# Identify each threading parameter and insert a space before it.
	thinfo_sep=$(echo -e ${thinfo} | sed -e "s/\([jip][cr]\)/ \1/g" )

	nt=1

	for loopnum in ${thinfo_sep}; do

		# Given the current string, which identifies a loop and the number of
		# ways of parallelism to be obtained from that loop, strip out the ways
		# and loop separately to identify each.
		loop=$(echo -e ${loopnum} | sed -e "s/[0-9]//g" )
		nways=$(echo -e ${loopnum} | sed -e "s/[a-z]//g" )

		# Construct a string that we can evaluate to set the number of ways of
		# parallelism for the current loop (e.g. jc_nt, ic_nt, jr_nt).
		loop_nt_eq_num="${loop}_nt=${nways}"

		# Update the total number of threads.
		nt=$(expr ${nt} \* ${nways})

		# Evaluate the string to assign the ways to the variable.
		eval ${loop_nt_eq_num}

	done

	# Find a binary using the test driver prefix and the threading suffix.
	# Then strip everything before and after the max problem size that's
	# encoded into the name of the binary.
	binname=$(ls -1 ${exec_root}_*_${tsuf}.x | head -n1)

	# Sanity check: If 'ls' couldn't find any binaries, then the user
	# probably didn't build them. Inform the user and proceed to the next
	# threading config.
	if [ "${binname}" = "" ]; then

		echo "Could not find binaries corresponding to '${tsuf}' threading config. Skipping."
		continue
	fi

	# Let the user know what threading config we are working on.
	echo "Switching to: jc${jc_nt} pc${pc_nt} ic${ic_nt} jr${jr_nt} ir${ir_nt} (nt = ${nt})"

	# Iterate over the datatypes.
	for dt in ${test_dts}; do

		# Iterate over the implementations.
		for im in ${test_impls}; do

			# Iterate over the operations.
			for op in ${test_ops}; do

				# Strip everything before the understore so that what remains is
				# the operation parameter string.
				oppars=${op##*_};

				# Strip everything after the understore so that what remains is
				# the operation name (sans parameter encoding).
				opname=${op%%_*}

				# Eigen does not support multithreading for hemm, herk, trmm,
				# or trsm. So if we're getting ready to execute an Eigen driver
				# for one of these operations and nt > 1, we skip this test.
				if [ "${im}"      = "eigen" ] && \
				   [ "${opname}" != "gemm"  ] && \
				   [ "${nt}"     != "1"     ]; then
					continue;
				fi

				# Set the number of threads according to th.
				if [ "${tsuf}" = "mt" ]; then

					# Set the threading parameters based on the implementation
					# that we are preparing to run.
					if   [ "${im}" = "blis" ]; then
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

					# Choose the mt problem size range.
					psr="${psr_mt}"

				else

					# Set all environment variables to 1 to ensure single-
					# threaded execution.
					export BLIS_JC_NT=1
					export BLIS_PC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
					export OMP_NUM_THREADS=1
					export OPENBLAS_NUM_THREADS=1
					export MKL_NUM_THREADS=1
					export nt_use=1

					# Choose the st problem size range.
					psr="${psr_st}"
				fi

				if [ "${quiet}" = "yes" ]; then
					qv="-q" # quiet
				else
					qv="-v" # verbose (the default)
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${opname}_${im}_${tsuf}.x"

				# Construct the name of the output file.
				out_file="${out_root}_${tsuf}_${dt}${opname}_${oppars}_${im}.m"

				# Use printf for its formatting capabilities.
				printf 'Running %s %-21s %s %-7s %s %s %s %s > %s\n' \
				       "${numactl}" "./${exec_name}" "-d ${dt}" \
				                                     "-c ${oppars}" \
				                                     "-i ${ind}" \
				                                     "-p \"${psr}\"" \
				                                     "-r ${nrepeats}" \
				                                     "${qv}" \
				                                     "${out_file}"

				# Run executable with or without numactl, depending on how
				# the numactl variable was set.
				if [ "${dryrun}" != "yes" ]; then
					${numactl} ./${exec_name} -d ${dt} -c ${oppars} -i ${ind} -p "${psr}" -r ${nrepeats} ${qv} > ${out_file}
				fi

				# Bedtime!
				sleep ${delay}

			done
		done
	done
done

