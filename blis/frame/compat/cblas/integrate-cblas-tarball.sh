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
# bump-version.sh
#
# Field G. Van Zee
#


print_usage()
{
	#local script_name
	
	# Get the script name
	#script_name=${0##*/}
	
	# Echo usage info
	echo " "
	echo " "$script_name
	echo " "
	echo " Field G. Van Zee"
	echo " "
	echo " Unpacks a CBLAS tarball and performs whatever preprocessing is"
	echo " necessary and appropriate in order to integrate the CBLAS source"
	echo " code into BLIS."
	echo " "
	echo " IMPORTANT: This script is designed to be run from the following"
	echo " directory:"
	echo " "
	echo "   frame/compat/cblas"
	echo " "
	echo " Usage:"
	echo "   ${script_name} tarball"
	echo " "
	echo " Arguments:"
	echo " "
	echo "   tarball      The name of the CBLAS package that will be unpacked."
	echo "                If tarball is not in the current directory, the full"
	echo "                directory path should be given."
	echo " "
	
	# Exit with non-zero exit status
	exit 1
}


main()
{
	# -- BEGIN GLOBAL VARIABLE DECLARATIONS --

	# The name of the script, stripped of any preceeding path.
	script_name=${0##*/}

	# The name and path of the CBLAS tarball.
	tarball_path=

	# The name of the CBLAS directory after it is unpacked.
	cblas_dir=CBLAS

	# The name of the sub-directory that we will create and into which
	# we will copy the source code for CBLAS wrappers.
	src_dir=src

	# -- END GLOBAL VARIABLE DECLARATIONS --


	# Process our command line options.
	while getopts ":h" opt; do
		case $opt in
			h  ) print_usage ;;
			\? ) print_usage
		esac
	done
	shift $(($OPTIND - 1))


	# Check the number of arguments after command line option processing.
	if [ $# = "1" ]; then

		tarball_path=$1
		echo "${script_name}: preparing to extract from '${tarball_path}'."

	else
		print_usage
	fi

	# Check that src_dir does not already exist. If it does, abort.
	if [ -d ${src_dir} ] ; then

		echo "${script_name}: found '${src_dir}' directory; please remove before proceeding."
		return 0
	fi

	# Un-tar and un-gzip the tarball.
	echo "${script_name}: extracting '${tarball_path}'."
	echo "${script_name}: expecting unpacked directory to be named '${cblas_dir}'."
	tar xzf ${tarball_path}

	# Create the directory into which we will copy the source code for the
	# CBLAS wrappers.
	echo "${script_name}: creating local '${src_dir}' directory."
	mkdir -p ${src_dir}

	# Copy the cblas.h header file.
	echo "${script_name}: copying cblas.h from '${cblas_dir}/include' to '${src_dir}'."
	cp ${cblas_dir}/include/cblas.h ${src_dir}/cblas.h

	# Copy the cblas_f77.h header file, removing all prototypes.
	echo "${script_name}: copying cblas_f77.h from '${cblas_dir}/include' to '${src_dir}'"
	cp ${cblas_dir}/include/cblas_f77.h ${src_dir}/cblas_f77.h

	# Create some temporary files to facilitate #including BLIS-specific
	# cpp macros.
	echo "${script_name}: creating temporary files."
	echo "#include \"bli_config.h\""    > include_bli_config.h
	echo "#include \"bli_system.h\""    > include_bli_system.h
	echo "#include \"bli_type_defs.h\"" > include_bli_type_defs.h
	echo "#include \"bli_cblas.h\""     > include_bli_cblas.h
	echo "#ifdef BLIS_ENABLE_CBLAS"     > ifdef_cblas.h
	echo "#endif"                       > endif_cblas.h

	# Process each CBLAS source file.
	echo "${script_name}: copying source from '${cblas_dir}/src' to '${src_dir}' with"
	echo "${script_name}: '#ifdef BLIS_ENABLE_CBLAS' guard:"
	for cbl_src_filepath in ${cblas_dir}/src/cblas_*.c; do

		# Strip the path to obtain just the filename.
		cbl_src_file=${cbl_src_filepath##*/}

		# Append the ifdef and prepend the endif macro statements to the
		# current file and output to its new location in ${src_dir}.
		echo "${script_name}: ...copying/BLIS-ifying ${cbl_src_file}"
		cat include_bli_config.h \
		    include_bli_system.h \
		    include_bli_type_defs.h \
		    include_bli_cblas.h \
		    ifdef_cblas.h \
		    ${cbl_src_filepath} \
		    endif_cblas.h > ${src_dir}/${cbl_src_file}
	done

	# Remove the temporary files.
	echo "${script_name}: cleaning up temporary files."
	rm -f include_bli_config.h
	rm -f include_bli_system.h
	rm -f include_bli_type_defs.h
	rm -f include_bli_cblas.h
	rm -f ifdef_cblas.h
	rm -f endif_cblas.h

	# Process some bugfixes to syntax errors present in the CBLAS source.

	echo "${script_name}: fixing syntax errors in CBLAS source:"

	fix_file ${src_dir}/cblas_chpmv.c "s/ F77_K=K,//g"
	fix_file ${src_dir}/cblas_chpmv.c "s/ F77_lda=lda,//g"

	fix_file ${src_dir}/cblas_zhpmv.c "s/ F77_K=K,//g"
	fix_file ${src_dir}/cblas_zhpmv.c "s/ F77_lda=lda,//g"

	fix_file ${src_dir}/cblas_ssyr2.c "s/F77__lda/F77_lda/g"
	fix_file ${src_dir}/cblas_dsyr2.c "s/F77__lda/F77_lda/g"

	fix_file ${src_dir}/cblas_strsm.c "s/F77_N=M/F77_M=M/g"

	# Now process some optional fixes that eliminate compiler warnings.

	echo "${script_name}: fixing compiler warnings in CBLAS source:"

	incx_string="s/, incx=incX//g"
	incy_string="s/, incy=incY//g"

	fix_file ${src_dir}/cblas_cgbmv.c "${incx_string}"
	fix_file ${src_dir}/cblas_cgemv.c "${incx_string}"
	fix_file ${src_dir}/cblas_cgerc.c "${incy_string}"
	fix_file ${src_dir}/cblas_chbmv.c "${incx_string}"
	fix_file ${src_dir}/cblas_chemv.c "${incx_string}"
	fix_file ${src_dir}/cblas_cher.c  "${incx_string}"
	fix_file ${src_dir}/cblas_cher2.c "${incx_string}"
	fix_file ${src_dir}/cblas_cher2.c "${incy_string}"
	fix_file ${src_dir}/cblas_chpmv.c "${incx_string}"
	fix_file ${src_dir}/cblas_chpr.c  "${incx_string}"
	fix_file ${src_dir}/cblas_chpr2.c "${incx_string}"
	fix_file ${src_dir}/cblas_chpr2.c "${incy_string}"

	fix_file ${src_dir}/cblas_zgbmv.c "${incx_string}"
	fix_file ${src_dir}/cblas_zgemv.c "${incx_string}"
	fix_file ${src_dir}/cblas_zgerc.c "${incy_string}"
	fix_file ${src_dir}/cblas_zhbmv.c "${incx_string}"
	fix_file ${src_dir}/cblas_zhemv.c "${incx_string}"
	fix_file ${src_dir}/cblas_zher.c  "${incx_string}"
	fix_file ${src_dir}/cblas_zher2.c "${incx_string}"
	fix_file ${src_dir}/cblas_zher2.c "${incy_string}"
	fix_file ${src_dir}/cblas_zhpmv.c "${incx_string}"
	fix_file ${src_dir}/cblas_zhpr.c  "${incx_string}"
	fix_file ${src_dir}/cblas_zhpr2.c "${incx_string}"
	fix_file ${src_dir}/cblas_zhpr2.c "${incy_string}"

	# Now that we're done with everything, we can remove the CBLAS directory.
	echo "${script_name}: removing '${cblas_dir}' directory."
	rm -rf ${cblas_dir}


	# Exit peacefully.
	return 0
}


fix_file()
{
	# Get the first function argument: the filename and path to fix.
	local filepath="$1"

	# Get the second function argument: the sed command to apply.
	local sedstring="$2"

	filename=${filepath##*/}

	echo "${script_name}: ...fixing ${filename} with 'sed -e ${sedstring}'"

	cat ${filepath} | sed -e "${sedstring}" > ${filepath}.new
	mv ${filepath}.new ${filepath}
}


# The script's main entry point, passing all parameters given.
main "$@"
