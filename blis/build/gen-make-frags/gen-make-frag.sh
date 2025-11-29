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
# gen-make-frag.sh
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
	echo " Automatically generates makefile fragments for a specified directory"
	echo " tree. "
	echo " "
	echo " Usage:"
	echo "   ${script_name} [options] root_dir frag_dir templ.mk suff_list ign_list"
	echo " "
	echo " Arguments (mandatory):"
	echo " "
	echo "   root_dir    The root directory to scan when generating makefile"
	echo "               fragments."
	echo " "
	echo "   frag_dir    The root directory in which makefile fragments will be"
	echo "               generated."
	echo " "
	echo "   templ.mk    The template makefile fragment used to generate the actual"
	echo "               fragments."
	echo " "
	echo "   suff_list   File containing a newline-separated list of file suffixes"
	echo "               of source files to that the top-level makefile expects to"
	echo "               access."
	echo " "
	echo "   ign_list    File containing a newline-separated list of directory names"
	echo "               to ignore when descending recursively into "
	echo " "
	echo " The following options are accepted:"
	echo " "
	echo "   -d          dry-run"
	echo "                 Go through all the motions, but don't actually generate any"
	echo "                 makefile fragments."
	echo "   -r          recursive"
	echo "                 Also generate makefile fragments for subdirectories of"
	echo "                 root_dir."
	echo "   -h          hide"
	echo "                 Hide the makefile fragments by prepending filenames with '.'."
	echo "   -i LIST     ignore"
	echo "                 Augment the list of directory names contained in ign_list"
	echo "                 with the directory names in LIST."
	echo "   -p PREFIX   prefix name"
	echo "                 Use PREFIX instead of uppercased root_dir in the makefile"
	echo "                 variable name. If the root_dir were 'stuff' and -p was not"
	echo "                 used, then source would be accumulated into a makefile"
	echo "                 variable named 'MK_STUFF', but if -p JUNK were given, then"
	echo "                 the variable name would instead be MK_JUNK."
	echo "   -o SCRIPT   output script name"
	echo "                 Use SCRIPT when outputting messages instead of the script's"
	echo "                 actual name."
	echo "   -v [0|1|2]  verboseness level"
	echo "                 level 0: silent  (no output)"
	echo "                 level 1: default (one line per directory)"
	echo "                 level 2: verbose (several lines per directory)."
	echo " "

	# Exit with non-zero exit status
	exit 1
}







#
# gen_mkfile()
#
# Creates a single makefile fragment in a user-specified directory and adds
# any local source files found to a top-level Makefile variable.
#
gen_mkfile()
{
	# Local variable declarations
	local mkfile_frag_var_name
	local this_dir
	local this_frag_dir
	local mkfile_frag_tmpl_name
	local mkfile_name
	local mkfile_frag_path
	local cur_frag_dir
	local cur_frag_path
	local local_src_files
	local sub_items
	local item_path
	local item_suffix
	local cur_frag_sub_dirs


	# Extract our arguments to local variables
	mkfile_frag_var_name=$1
	this_dir=$2
	this_frag_dir=$3


	# Make sure the target directory exists
	mkdir -p $this_frag_dir


	# Strip the leading path from the template makefile path to get its
	# simple filename. Hide the output makefile fragment filename, if
	# requested.
	mkfile_frag_tmpl_name=${mkfile_frag_tmpl_path##*/}
	if [ -n "$hide_flag" ]; then
		mkfile_frag_path=$this_frag_dir/.$mkfile_frag_tmpl_name
	else
		mkfile_frag_path=$this_frag_dir/$mkfile_frag_tmpl_name
	fi


	# Determine the directory in which the fragment will reside.
	cur_frag_path=$this_dir
	cur_frag_dir=${this_dir##*/}


	# Initialize the local source list to empty
	local_src_files=""

	# Get a listing of the items in $this_dir
	sub_items=$(ls $this_dir)

	# Generate a list of the source files we've chosen
	for item in $sub_items; do

		# Prepend the directory to the item to get a relative path
		item_path=$this_dir/$item

		# Acquire the item's suffix, if it has one
		item_suffix=${item_path##*.}

		# If the suffix matches, then add it to our list
		if is_in_list $item_suffix "$src_file_suffixes"
		then
			local_src_files="$local_src_files $item"
		fi
	done

	# Delete the leading " " space character in the local source files list.
	local_src_files=${local_src_files##" "}


	# Initialize the fragment subdirectory list to empty
	cur_frag_sub_dirs=""

	# Capture the relative path listing of items in $this_dir.
	sub_items=$(ls $this_dir)

	# Determine the fragment's subdirectory names, if any exist
	for item in $sub_items; do

		# Prepend the directory to the item to get a relative path
		item_path=$this_dir/$item

		# If item is a directory, and it's not in the ignore list, descend into it.
		#if [ -d $item_path ] && ! should_ignore $item; then
		if [ "$recursive_flag" = "1" ] && [ -d $item_path ] && ! is_in_list $item "$ignore_dirs" ; then
			cur_frag_sub_dirs=$cur_frag_sub_dirs" "$item
		fi
	done

	# Delete the leading " " space character in fragment's subdirectory list.
	cur_frag_sub_dirs=${cur_frag_sub_dirs##" "}


	# Be verbose, if level 2 was requested.
	if [ "$verbose_flag" = "2" ]; then
		echo "mkf frag tmpl path: $mkfile_frag_tmpl_path"
		echo "mkf frag path:      $mkfile_frag_path"
		echo "cur frag path:      $cur_frag_path"
		echo "cur frag dir:       $cur_frag_dir"
		echo "cur frag sub dirs:  $cur_frag_sub_dirs"
		echo "local src files:    $local_src_files"
		echo "src file suffixes:  $src_file_suffixes"
		echo "mkf frag var name:  $mkfile_frag_var_name"
		echo "--------------------------------------------------"
	fi


	# Copy the template makefile to the directory given, using the new
	# makefile name we just created above.
	if [ -z "$dry_run_flag" ]; then
		cat $mkfile_frag_tmpl_path | sed -e s/"$mkfile_fragment_cur_dir_name_anchor"/"$cur_frag_dir"/g \
		                           | sed -e s/"$mkfile_fragment_sub_dir_names_anchor"/"$cur_frag_sub_dirs"/g \
		                           | sed -e s/"$mkfile_fragment_local_src_files_anchor"/"$local_src_files"/g \
		                           | sed -e s/"$mkfile_fragment_src_var_name_anchor"/"$mkfile_frag_var_name"/g \
		                           > $mkfile_frag_path
	fi


	# Return peacefully.
	return 0
}


#
# gen_mkfiles
#
# Recursively generates makefile fragments for a directory and all
# subdirectories. All of the actual work happens in gen_mkfile().
#
gen_mkfiles()
{
	# Local variable declarations
	local item sub_items cur_dir this_frag_dir this_dir


	# Extract our argument
	cur_dir=$1
	this_frag_dir=$2


	# Append a relevant suffix to the makefile variable name, if necesary
	# NOTE: This step is disabled because special directories are presently
	# ignored when generating makefile variable names.
	#all_add_src_var_name "$cur_dir"


	# Be verbose if level 2 was requested
	if   [ "$verbose_flag" = "2" ]; then
		echo ">>>" $script_name ${src_var_name}_$SRC $cur_dir $this_frag_dir
	elif [ "$verbose_flag" = "1" ]; then
		echo "$script_name: creating makefile fragment in $this_frag_dir from $cur_dir"
	fi


	# Call our function to generate a makefile in the directory given.
	gen_mkfile "${src_var_name}_$SRC" $cur_dir $this_frag_dir


	# Get a listing of the directories in $directory
	sub_items=$(ls $cur_dir)

	# Descend into the contents of root_dir to generate the subdirectories'
	# makefile fragments.
	for item in $sub_items; do

		# If item is a directory, and it's not in the ignore list, descend into it.
		#if [ -d "$cur_dir/$item" ] && ! should_ignore $item; then
		if [ -d "$cur_dir/$item" ] && ! is_in_list $item "$ignore_dirs" ; then
			gen_mkfiles $cur_dir/$item $this_frag_dir/$item
		fi
	done


	# Remove a relevant suffix from the makefile variable name, if necesary
	# NOTE: This step is disabled because special directories are presently
	# ignored when generating makefile variable names.
	#all_del_src_var_name "$cur_dir"


	# Return peacefully
	return 0
}



#update_src_var_name_special()
#{
#	local dir act i name var_suffix
#
#	# Extract arguments.
#	act="$1"
#	dir="$2"
#
#	# Strip / from end of directory path, if there is one, and then strip
#	# path from directory name.
#	dir=${dir%/}
#	dir=${dir##*/}
#
#	# Run through our list.
#	# NOTE: CURRENTLY, SPECIAL DIRECTORY NAMES ARE IGNORED. In order to
#	#       re-enable them, remove the quotes from "${special_dirs}".
#	for specdir in "${special_dirs}"; do
#
#		# If the current item matches sdir, then we'll have
#		# to make a modification of some form.
#		if [ "$dir" = "$specdir" ]; then
#
#			# Convert the directory name to uppercase.
#			var_suffix=$(echo "$dir" | tr '[:lower:]' '[:upper:]')
#
#			# Either add or remove the suffix, and also update the
#			# source file suffix variable.
#			if [ "$act" == "+" ]; then
#				src_var_name=${src_var_name}_$var_suffix
#			else
#				src_var_name=${src_var_name%_$var_suffix}
#			fi
#
#			# No need to continue iterating.
#			break;
#		fi
#	done
#}

#init_src_var_name()
#{
#	local dir="$1"
#
#	# Strip off the leading / if there is one
#	dir=${dir%%/}
#
#	# Convert the / directory separators into spaces to make a list of
#	# directories.
#	list=${dir//\// }
#
#	# Inspect each item in $list
#	for item in $list; do
#
#		# Try to initialize the source variable name
#		all_add_src_var_name $item
#	done
#}

#all_add_src_var_name()
#{
#	local dir="$1"
#
#	update_src_var_name_special "+" "$dir"
#
#}

#all_del_src_var_name()
#{
#	local dir="$1"
#
#	update_src_var_name_special "-" "$dir"
#}

read_mkfile_config()
{
	# Read the file describing file suffixes.
	src_file_suffixes=$(cat "${suffix_file}")

	# Read the file listing the directories to ignore.
	ignore_dirs=$(cat "${ignore_file}")

	# Change newlines into spaces. This is optional, but helps when
	# printing these values out (so they appear on one line).
	src_file_suffixes=$(echo ${src_file_suffixes} | sed "s/\n/ /g")
	ignore_dirs=$(echo ${ignore_dirs} | sed "s/\n/ /g")

}

main()
{
	# -- BEGIN GLOBAL VARIABLE DECLARATIONS --

	# Define these makefile template "anchors" used in gen_mkfile().
	mkfile_fragment_cur_dir_name_anchor="_mkfile_fragment_cur_dir_name_"
	mkfile_fragment_sub_dir_names_anchor="_mkfile_fragment_sub_dir_names_"
	mkfile_fragment_local_src_files_anchor="_mkfile_fragment_local_src_files_"
	mkfile_fragment_src_var_name_anchor="_mkfile_fragment_src_var_name_"

	# The name of the script, stripped of any preceeding path.
	script_name=${0##*/}

	# The prefix for all makefile variables.
	src_var_name_prefix='MK'

	# The variable that always holds the string that will be passed to
	# gen_mkfile() as the source variable to insert into the fragment.mk.
	src_var_name=''

	# The suffix appended to all makefile fragment source variables.
	SRC='SRC'

	# The list of source file suffixes to add to the makefile variables.
	src_file_suffixes=''

	# The lists of directories to ignore.
	ignore_dirs=''

	# The arguments to this function. They'll get assigned meaningful
	# values after getopts.
	root_dir=""
	frag_dir=""
	mkfile_frag_tmpl_path=""
	suffix_file=""
	ignore_file=""

	# Flags set by getopts.
	dry_run_flag=""
	hide_flag=""
	ignore_list=""
	recursive_flag=""
	output_name=""
	prefix_flag=""
	verbose_flag=""

	# -- END GLOBAL VARIABLE DECLARATIONS --


	# Local variable declarations.
	local item sub_items this_dir


	# Process our command line options.
	while getopts ":dhi:o:p:rv:" opt; do
		case $opt in
			d  ) dry_run_flag="1" ;;
			h  ) hide_flag="1" ;;
			r  ) recursive_flag="1" ;;
			i  ) ignore_list="${ignore_list} $OPTARG" ;;
			o  ) output_name=$OPTARG ;;
			p  ) prefix_flag=$OPTARG ;;
			v  ) verbose_flag=$OPTARG ;;
			\? ) print_usage
		esac
	done
	shift $(($OPTIND - 1))


	# Make sure that verboseness level is valid.
	if [ "$verbose_flag" != "0" ] &&
	   [ "$verbose_flag" != "1" ] &&
	   [ "$verbose_flag" != "2" ]; then
		verbose_flag="1"
	fi

	# Check the number of arguments after command line option processing.
	if [ $# != "5" ]; then
		print_usage
	fi

	# If an output script name was given, overwrite script_name with it.
	if [ -n "${output_name}" ]; then
		script_name="${output_name}"
	fi


	# Extract our arguments.
	root_dir=$1
	frag_dir=$2
	mkfile_frag_tmpl_path=$3
	suffix_file=$4
	ignore_file=$5


	# Read the makefile config files to be used in the makefile fragment
	# generation.
	read_mkfile_config

	# Append the command line ignore_list to the ignore_dirs variable read from
	# ignore_file by read_mkfile_config().
	ignore_dirs="${ignore_dirs} ${ignore_list}"


	# Strip / from end of directory path, if there is one.
	root_dir=${root_dir%/}
	frag_dir=${frag_dir%/}


	# Initialize the name of the makefile source variable.
	if [ -n "$prefix_flag" ]; then

		# If prefix_flag is not null, then we construct src_var_name using
		# it instead of root_dir. So if the prefix is 'junk', we will get
		# makefile variables that begin with 'MK_JUNK'.
		root_dir_upper=$(echo "$prefix_flag" | tr '[:lower:]' '[:upper:]')
		src_var_name="${src_var_name_prefix}_${root_dir_upper}"

	else

		# Otherwise, we use root_dir. If the root directory is 'foo' then
		# makefile variables will begin with 'MK_FOO'.
		# We are also careful to convert forward slashes into underscore so
		# root directories such as foo/bar result in makefile variables
		# that begin with 'MK_FOO_BAR'.
		root_dir_upper=$(echo "$root_dir" | tr '[:lower:]' '[:upper:]')
		root_dir_upper=$(echo "$root_dir_upper" | tr '/' '_')
		src_var_name="${src_var_name_prefix}_${root_dir_upper}"
	fi


	# Be verbose if level 2 was requested.
	if   [ "$verbose_flag" = "2" ]; then
		echo ">>>" $script_name ${src_var_name}_$SRC $root_dir $frag_dir
	elif [ "$verbose_flag" = "1" ]; then
		echo "$script_name: creating makefile fragment in $frag_dir from $root_dir"
	fi


	# Call our function to generate a makefile in the root directory given.
	gen_mkfile "${src_var_name}_$SRC" $root_dir $frag_dir


	# If we were asked to act recursively, then continue processing
	# root_dir's contents.
	if [ -n "$recursive_flag" ]; then

		# Get a listing of the directories in $directory.
		sub_items=$(ls $root_dir)

		# Descend into the contents of root_dir to generate the makefile
		# fragments.
		for item in $sub_items; do

			# If item is a directory, and it's not in the ignore list, descend into it.
			#if [ -d "$root_dir/$item" ] && ! should_ignore $item ; then
			if [ -d "$root_dir/$item" ] && ! is_in_list $item "$ignore_dirs" ; then

				gen_mkfiles $root_dir/$item $frag_dir/$item
			fi
		done
	fi


	# Exit peacefully.
	return 0
}

is_in_list()
{
	local cur_item the_item item_list

	# Extract argument.
	the_item="$1"
	item_list="$2"

	# Check each item in the list against the item of interest.
	for cur_item in ${item_list}; do

		# If the current item in the list matches the one of interest.
		if [ "${cur_item}" = "${the_item}" ]; then

			# Return success (ie: item was found).
			return 0
		fi
	done

	# If we made it this far, return failure (ie: item not found).
	return 1
}

# The script's main entry point, passing all parameters given.
main "$@"
