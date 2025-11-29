#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

# Placeholder until we add multithreading.
nt=1

# Delay between test cases.
delay=0.02

# Threadedness to test.
threads="st"

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
sks="4"

# Implementations to test.
impls="vendor blissup blislpab openblas eigen libxsmm blasfeo"
#impls="vendor"
#impls="blissup"
#impls="blislpab"
#impls="openblas"
#impls="eigen"
#impls="libxsmm"
#impls="blasfeo"

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

										echo "Running (nt = ${nt}) ./${exec_name} > ${out_file}"

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

