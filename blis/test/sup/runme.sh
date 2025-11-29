#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

#sys="blis"
#sys="lonestar5"
#sys="ul252"
#sys="ul264"
sys="ul2128"

if [ ${sys} = "blis" ]; then

	export GOMP_CPU_AFFINITY="0-3"

	numactl=""
	nt=4

elif [ ${sys} = "lonestar5" ]; then

	export GOMP_CPU_AFFINITY="0-23"

	numactl=""
	nt=12

elif [ ${sys} = "ul252" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-51"

	numactl="numactl --interleave=all"
	nt=26

elif [ ${sys} = "ul264" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-63"

	numactl="numactl --interleave=all"
	nt=32

elif [ ${sys} = "ul2128" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/field/intel/mkl/lib/intel64"
	export GOMP_CPU_AFFINITY="0-127"

	numactl="numactl --interleave=all"
	nt=32

fi

# Delay between test cases.
delay=0.02

# Threadedness to test.
#threads="st mt"
threads="st"

# Datatypes to test.
dts="s d"

# Operations to test.
ops="gemm"

# Transpose combintions to test.
trans="nn nt tn tt"

# Storage combinations to test.
# NOTE: mixed storage cases are not yet implemented in test_gemm.c.
#stors="rrr rrc rcr rcc crr crc ccr ccc"
stors="rrr ccc"

# Problem shapes to test.
shapes="sll lsl lls lss sls ssl lll"

# Small problem dimensions to use.
# FGVZ: figure out how to probe what's in the directory and
# execute everything that's there?
# st, single real
sms_st_s="6"
sns_st_s="16"
sks_st_s="4"
# st, double real
sms_st_d="6"
sns_st_d="8"
sks_st_d="4"
# mt, single real
sms_mt_s="6"
sns_mt_s="16"
sks_mt_s="10"
# mt, double real
sms_mt_d="6"
sns_mt_d="8"
sks_mt_d="10"

# Leading dimensions to use (small or large).
# When a leading dimension is large, it is constant and set to the largest
# problem size that will be run.
#ldims="s l"
ldims="s"

# Packing combinations for blissup. The first char encodes the packing status
# of matrix A and the second char encodes the packing status of matrix B.
# NOTE: This string must always contain 'uu' if other implementations are also
# being tested at the same time.
#pcombos="uu up pu pp"
pcombos="uu"

# Implementations to test.
impls="vendor blissup blisconv openblas eigen blasfeo libxsmm"
#impls="vendor blissup blisconv openblas eigen"
#impls="vendor"
#impls="blissup"
#impls="blisconv"
#impls="openblas"
#impls="eigen"
#impls="blasfeo"

# Save a copy of GOMP_CPU_AFFINITY so that if we have to unset it, we can
# restore the value.
GOMP_CPU_AFFINITYsave=${GOMP_CPU_AFFINITY}

# Example: test_dgemm_nn_rrc_m6npkp_blissup_st.x

for th in ${threads}; do

	for dt in ${dts}; do

		# Choose the small m, n, and k values based on the threadedness and
		# datatype currently being executed.
		if   [ ${th} = "st" ]; then
			if   [ ${dt} = "s" ]; then
				sms=${sms_st_s}
				sns=${sns_st_s}
				sks=${sks_st_s}
			elif [ ${dt} = "d" ]; then
				sms=${sms_st_d}
				sns=${sns_st_d}
				sks=${sks_st_d}
			else
				exit 1
			fi
		elif [ ${th} = "mt" ]; then
			if   [ ${dt} = "s" ]; then
				sms=${sms_mt_s}
				sns=${sns_mt_s}
				sks=${sks_mt_s}
			elif [ ${dt} = "d" ]; then
				sms=${sms_mt_d}
				sns=${sns_mt_d}
				sks=${sks_mt_d}
			else
				exit 1
			fi
		fi

		for op in ${ops}; do

			for tr in ${trans}; do

				for st in ${stors}; do

					for sh in ${shapes}; do

						for sm in ${sms}; do

							for sn in ${sns}; do

								for sk in ${sks}; do

									for ld in ${ldims}; do

										for im in ${impls}; do

											for pc in ${pcombos}; do

												if [ "${th}" = "mt" ]; then

													# Prohibit attempts to run blasfeo or libxsmm as
													# multithreaded.
													if [ "${im}" = "blasfeo" ] || \
													   [ "${im}" = "libxsmm" ]; then
														continue;
													fi

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

												# Isolate the individual chars in the current pcombo
												# string.
												packa=${pc:0:1}
												packb=${pc:1:1}

												# For blissup implementations, set the BLIS_PACK_A and
												# BLIS_PACK_B environment variables according to the
												# chars in the current pcombo string.
												if [ "${im:0:7}" = "blissup" ]; then

													# Set BLIS_PACK_A if the pcombo char is 'p'; otherwise
													# unset the variable altogether.
													if [ ${packa} = "p" ]; then
														export BLIS_PACK_A=1
													else
														unset BLIS_PACK_A
													fi

													# Set BLIS_PACK_B if the pcombo char is 'p'; otherwise
													# unset the variable altogether.
													if [ ${packb} = "p" ]; then
														export BLIS_PACK_B=1
													else
														unset BLIS_PACK_B
													fi
												else

													# Unset the variables for non-blissup implementations,
													# just to be paranoid-safe.
													unset BLIS_PACK_A
													unset BLIS_PACK_B
												fi

												# Limit execution of non-blissup implementations to the
												# 'uu' packing combination. (Those implementations don't
												# use the pcombos string, but since we iterate over its
												# words for all implementations, we have to designate one
												# of them as a placeholder to allow those implementations
												# to execute. The 'uu' string was chosen over the 'pp'
												# string because it's more likely that this script will be
												# used to run blissup on unpacked matrices, and so the
												# sorting for the output files is nicer if the non-blissup
												# implementations use the 'uu' string, even if it's more
												# likely that those implementations use packing. Think of
												# 'uu' as encoding the idea that explicit packing was not
												# requested.)
												if [ "${im:0:7}" != "blissup" ] && \
												   [ "${pc}" != "uu" ]; then
													continue;
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

												# Construct the ldim substring (e.g. lds or ldl)
												ldstr="ld${ld}"

												# Construct the pack substring (e.g. uaub, uapb, paub, or papb)
												packstr="${packa}a${packb}b"

												# Ex: test_dgemm_nn_rrc_m6npkp_blissup_st.x
												# Ex: test_dgemm_nt_rrr_m6npkp_ldl_blissup_st.x

												# Construct the name of the test executable.
												exec_name="${exec_root}_${dt}${op}_${tr}_${st}_${shstr}_${ldstr}_${im}_${th}.x"

												# Construct the name of the output file.
												out_file="${out_root}_${th}_${dt}${op}_${tr}_${st}_${shstr}_${ldstr}_${packstr}_${im}.m"

												echo "Running (nt = ${nt_use}) ${numactl} ./${exec_name} > ${out_file}"

												# Run executable.
												${numactl} ./${exec_name} > ${out_file}

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
	done
done

