#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

sys="blis"
#sys="lonestar5"
#sys="ul252"
#sys="ul264"

if [ ${sys} = "blis" ]; then

	export GOMP_CPU_AFFINITY="0-3"
	nt=4

elif [ ${sys} = "lonestar5" ]; then

	export GOMP_CPU_AFFINITY="0-23"
	nt=12

elif [ ${sys} = "ul252" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-51"
	nt=26

elif [ ${sys} = "ul264" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-63"
	nt=32

fi

# Delay between test cases.
delay=0.02

# Threadedness to test.
#threads="st mt"
threads="st mt"

# Datatypes to test.
#dts="d s"
dts="d"

# Operations to test.
ops="gemm"

# Transpose combintions to test.
trans="nn nt tn tt"

# Storage combinations to test.
#stors="rrr rrc rcr rcc crr crc ccr ccc"
stors="rrr ccc"

# Problem shapes to test.
shapes="sll lsl lls lss sls ssl lll"

# FGVZ: figure out how to probe what's in the directory and
# execute everything that's there?
sms="6"
sns="8"
sks="10"

# Implementations to test.
impls="vendor blissup blislpab openblas eigen"
#impls="vendor"
#impls="blissup"
#impls="blislpab"
#impls="openblas"
#impls="eigen"

# Save a copy of GOMP_CPU_AFFINITY so that if we have to unset it, we can
# restore the value.
GOMP_CPU_AFFINITYsave=${GOMP_CPU_AFFINITY}

# Example: test_dgemm_nn_rrc_m6npkp_blissup_st.x

for th in ${threads}; do

	for dt in ${dts}; do

		for op in ${ops}; do

			for tr in ${trans}; do

				for st in ${stors}; do

					for sh in ${shapes}; do

						for sm in ${sms}; do

							for sn in ${sns}; do

								for sk in ${sks}; do

									for im in ${impls}; do

										if [ "${th}" = "mt" ]; then

											# Specify the multithreading depending on which
											# implementation is about to be tested.
											if   [ "${im:0:4}" = "blis" ]; then
												unset  OMP_NUM_THREADS
												export BLIS_NUM_THREADS=${nt}
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

										else # if [ "${th}" = "st" ];

											# Use single-threaded execution.
											export OMP_NUM_THREADS=1
											export BLIS_NUM_THREADS=1
											export OPENBLAS_NUM_THREADS=1
											export MKL_NUM_THREADS=1
											export nt_use=1
										fi

										# Multithreaded OpenBLAS seems to have a problem
										# running properly if GOMP_CPU_AFFINITY is set.
										# So we temporarily unset it here if we are about
										# to execute OpenBLAS, but otherwise restore it.
										if [ ${im} = "openblas" ]; then
											unset GOMP_CPU_AFFINITY
										else
											export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITYsave}"
										fi

										# Limit execution of non-BLIS implementations to
										# rrr/ccc storage cases.
										if [ "${im:0:4}" != "blis" ] && \
										   [ "${st}" != "rrr" ] && \
										   [ "${st}" != "ccc" ]; then
											continue;
										fi

										# Further limit execution of libxsmm to
										# ccc storage cases.
										if [ "${im:0:7}" = "libxsmm" ] && \
										   [ "${st}" != "ccc" ]; then
											continue;
										fi

										# Extract the shape chars for m, n, k.
										chm=${sh:0:1}
										chn=${sh:1:1}
										chk=${sh:2:1}

										# Construct the shape substring (e.g. m6npkp)
										shstr=""

										if [ ${chm} = "s" ]; then
											shstr="${shstr}m${sm}"
										else
											shstr="${shstr}mp"
										fi

										if [ ${chn} = "s" ]; then
											shstr="${shstr}n${sn}"
										else
											shstr="${shstr}np"
										fi

										if [ ${chk} = "s" ]; then
											shstr="${shstr}k${sk}"
										else
											shstr="${shstr}kp"
										fi

										# Ex: test_dgemm_nn_rrc_m6npkp_blissup_st.x

										# Construct the name of the test executable.
										exec_name="${exec_root}_${dt}${op}_${tr}_${st}_${shstr}_${im}_${th}.x"

										# Construct the name of the output file.
										out_file="${out_root}_${th}_${dt}${op}_${tr}_${st}_${shstr}_${im}.m"

										echo "Running (nt = ${nt_use}) ./${exec_name} > ${out_file}"

										# Run executable.
										./${exec_name} > ${out_file}

										sleep ${delay}

									done
								done
							done
						done
					done
				done
			done
		done
	done
done

