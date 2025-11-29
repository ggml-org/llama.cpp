#!/bin/sh
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
	echo " Performs a series of actions needed when incrementing (bumping) the"
	echo " BLIS version number:"
	echo "   1. Overwrite the version file with the version string passed"
	echo "      into this script (new_vers)."
	echo "   2. Commit the updated version file."
	echo "   3. Create a new tag (named the same as new_vers) which refers to"
	echo "      the commit created in (2)."
	echo "   4. Update the CHANGELOG file."
	echo "   5. Commit the updated CHANGELOG file."
	echo " "
	echo " Usage:"
	echo "   ${script_name} [options] new_vers"
	echo " "
	echo " Arguments:"
	echo " "
	echo "   new_vers     The new version string."
	echo " "
	echo " Options:"
	echo " "
	echo "   -d           dry-run"
	echo "                  Go through all the motions, but don't actually make any"
	echo "                  changes to files or perform any git commits. Note that"
	echo "                  this will result in the commits for (2) and (5) above"
	echo "                  being equal to the initial commit in the script output."
	echo "   -f VERSFILE  version file name"
	echo "                  Update VERSFILE with new version string instead of default"
	echo "                  'version' file."
	
	# Exit with non-zero exit status
	exit 1
}


main()
{
	# -- BEGIN GLOBAL VARIABLE DECLARATIONS --

	# The name of the script, stripped of any preceeding path.
	script_name=${0##*/}

	# The name of the config.mk file.
	configmk_file='config.mk'

	# The name of the CHANGELOG file.
	changelog_file='CHANGELOG'

	# The name and location of the default version file.
	version_file_def='build/version'

	# The name and location of the specified version file.
	version_file=''

	# Strings used during version query.
	git_commit_str=''
	new_version_str=''

	# The script name to use instead of the $0 when outputting messages.
	output_name=''

	# The git directory.
	gitdir='.git'
	
	# Whether we are performing a dry run or not.
	dry_run_flag=""	

	# -- END GLOBAL VARIABLE DECLARATIONS --


	# Process our command line options.
	while getopts ":dhf:" opt; do
		case $opt in
			d  ) dry_run_flag="1" ;;
			f  ) version_file=$OPTARG ;;
			h  ) print_usage ;;
			\? ) print_usage
		esac
	done
	shift $(($OPTIND - 1))


	# If a version file name was not given, set version_file to the default
	# value.
	if [ -n "${version_file}" ]; then

		echo "${script_name}: version file specified: '${version_file}'."
	else

		echo "${script_name}: no version file specified; defaulting to '${version_file_def}'."
		version_file="${version_file_def}"
	fi


	# Check the number of arguments after command line option processing.
	if [ $# = "1" ]; then

		new_version_str=$1
		echo "${script_name}: preparing to bump to version '${new_version_str}'."

	else
		print_usage
	fi


	# Check if the .git dir exists; if it does not, we do nothing.
	if [ -d "${gitdir}" ]; then

		echo "${script_name}: found '${gitdir}' directory; assuming git clone."

		git_commit_str=$(git describe --always)
		echo "${script_name}: initial commit: ${git_commit_str}."

		echo "${script_name}: updating version file '${version_file}'."
		if [ -z "$dry_run_flag" ]; then
			echo "${new_version_str}" > ${version_file}
		fi

		echo "${script_name}: executing: git commit -m \"Version file update (${new_version_str})\" ${version_file}."
		if [ -z "$dry_run_flag" ]; then
			git commit -m "Version file update (${new_version_str})" ${version_file}
		fi

		git_commit_str=$(git describe --always)
		echo "${script_name}: commit to be tagged: ${git_commit_str}."

		echo "${script_name}: executing: git tag ${new_version_str} ${git_commit_str}."
		if [ -z "$dry_run_flag" ]; then
			git tag ${new_version_str} ${git_commit_str}
		fi

		echo "${script_name}: updating ${changelog_file}."
		if [ -z "$dry_run_flag" ]; then

			# If 'make distclean' was run recently, we need to re-run
			# configure in order for 'make changelog' to work properly.
			if [ ! -f "${configmk_file}" ]; then
				./configure auto
			fi
			make changelog
		fi

		echo "${script_name}: executing: git commit -m \"CHANGELOG update (${new_version_str})\" ${changelog_file}."
		if [ -z "$dry_run_flag" ]; then
			git commit -m "CHANGELOG update (${new_version_str})" ${changelog_file}
		fi

		git_commit_str=$(git describe --always)
		echo "${script_name}: latest commit: ${git_commit_str}."

	else

		echo "${script_name}: could not find '${gitdir}' directory; bailing out."

	fi


	# Exit peacefully.
	return 0
}


# The script's main entry point, passing all parameters given.
main "$@"
