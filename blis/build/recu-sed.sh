#!/bin/bash

#
# recursive-sed.sh
#
# Field G. Van Zee
#

print_usage()
{
	# Echo usage info
	echo " "
	echo " "$script_name
	echo " "
	echo " Field G. Van Zee"
	echo " "
	echo " Recusively descend a directory tree and perform sed commands, either on"
	echo " the filename or the file contents, or both." 
	echo " "
	echo " Usage:"
	echo "   ${script_name} [options]"
	echo " "
	echo " The following options are accepted:"
	echo " "
	echo "   -d "
	echo "                 Dry run. Go through all the motions, but don't actually"
	echo "                 apply any of the sed expressions to file names or contents."
	echo "   -N "
	echo "                 Do not proceed recursively into subdirectories; consider"
	echo "                 only the files within the current directory. Default"
	echo "                 behavior is to act recursively."
	echo "   -h "
	echo "                 Consider hidden files and directories. Default behavior is"
	echo "                 to ignore them."
	echo "   -n "
	echo "                 Use svn mv instead of mv when renaming the file."
	echo "                 Notice that this only applies if the filename changes."
	echo "   -p pattern "
	echo "                 Specifies the filename pattern, as would be given to the"
	echo "                 ls utility, to limit which files are affected. Default is"
	echo "                 the to consider all files present."
	echo "   -r dir"
	echo "                 The root directory for the recursive action to be performed."
	echo "                 Default is to use the current working directory."
	echo "   -v [0|1|2]"
	echo "                 verboseness level"
	echo "                 level 0: silent  (no output)"
	echo "                 level 1: default (one line per directory; supress ls stderr)"
	echo "                 level 2: verbose (one line per directory; show ls stderr)"
	echo " "
	echo " At least one of the following option-argument pairs is required:"
	echo " "
	echo "   -f sed_expr "
	echo "                 Specifies the sed expression that will be applied to the"
	echo "                 filenames of the files touched by the script. This expression"
    echo "                 must be a search-and-replace pattern."
	echo "   -c sed_expr "
	echo "                 Specifies the sed expression that will be applied to the"
	echo "                 contents of the files touched by the script. This expression"
    echo "                 should be a search-and-replace pattern."
	echo "   -s sed_script"
	echo "                 Specifies an arbitrary sed script that will be applied to the"
	echo "                 file contents of the files touched by the script."
	echo " "
	echo " Note: -c and -s options are mutually exclusive."
	echo " "
	
	# Exit with non-zero exit status
	exit 1
}




perform_sed()
{
	# Variables set by getopts.
	local exist_dir="$1"
	
	#echo "exist_dir: $exist_dir"

	# The suffix used to create temporary files
	local temp_file_suffix="sed_temp"

	# Check that exist_dir actually exists and is a directory
	if [ ! -d "${exist_dir}" ]; then
		echo "${script_name}: ${exist_dir} does not seem to be a valid directory."
		exit 1
	fi
	
	# Check that the filename sed expression, if given, begins with an 's'.
	if [ -n "$filename_sed_expr" ]; then
		
		# If it's a valid search-and-replace expression, this should return an 's'.
		filename_sed_char=${filename_sed_expr%%/*}
		
		if [ "$filename_sed_char" != "s" ]; then
			echo "${script_name}: sed expression given with -f must be search-and-replace."
			exit 1
		fi
	fi
	
	# Check that the sed script, if given, exists.
	if [ -n "$contents_sed_script" ]; then
		
		if [ ! -f ${contents_sed_script} ]; then
			echo "${script_name}: ${contents_sed_script} is not a regular file or does not exist."
			exit 1
		fi
	fi
	
	# Assume that the sed expression is a search-and-replace. Extract the patterns
	# to match on. (Arbitrary sed expressions should be applied through a sed script.)
	if [ "$filename_sed_expr" != "" ]; then
		filename_sed_match=${filename_sed_expr#s/}
		filename_sed_match=${filename_sed_match%%/*}
	fi
	

	# Get the list of source files in the directory given. Supress stderr if
	# level 0 or 1 verbosity was requested.
	#if [ "$verbose_level" != "2" ]; then
	#	old_filepaths=$(ls -d -b ${exist_dir}/${filename_pattern} 2> /dev/null)
	#else
	#	old_filepaths="$(ls -d -b ${exist_dir}/${filename_pattern})"
	#fi
	
	#echo $old_filepaths
	#echo "$exist_dir/$filename_pattern"
	
	#for old_filepath in $old_filepaths; do
	#echo "exist_dir:    $exist_dir"

	# Find all files that match the pattern in the current directory.
	find "${exist_dir}" -maxdepth 1 -name "${filename_pattern}" -print | while read old_filepath
	do
		#echo "old_filepath: $old_filepath"

		# Skip the current directory.
		if [ "${old_filepath}" == "${exist_dir}" ]; then
			continue
		fi
		
		# Skip any non-regular files.
		if [ ! -f "$old_filepath" ]; then
			
			# And say we are doing so if verboseness was requested.
			if [ "$verbose_level" = "2" ]; then
				echo "${script_name}: Ignoring $old_filepath"
			fi
			continue
		fi
		
		# Strip exist_dir from filename.
		old_filename=${old_filepath##*/}

		# Strip the filename from old_filepath to leave the directory path.
		old_dirpath=${old_filepath%/*}
		
		# Create a new filename from the old one. If a filename sed expression was given,
		# it will be applied now.
		if [ "$filename_sed_expr" != "" ]; then
			new_filename=$(echo "${old_filename}" | sed "${filename_sed_expr}")
		else
			new_filename="${old_filename}"
		fi

		#echo "new_filename: $new_filename"
			
		# Create the filepath to the new file location.
		new_filepath="${old_dirpath}/${new_filename}"
		#echo "new_filepath: $new_filepath"
			
		# Grep for the filename pattern within the filename of the current file.
		if [ "$filename_sed_expr" != "" ]; then
			grep_filename=$(echo "${old_filename}" | grep "${filename_sed_match}")
		fi
		

		# If we are not performing a dry run, proceed.
		if [ -z "$dry_run_flag" ]; then
			
			# Save the old file permissions so we can re-apply them to the
			# new file if its contents change (ie: if it's not just a 'mv',
			# which inherently preserves file permissions).
			old_perms=$(stat -c %a "${old_filepath}")

			# If the old and new filepaths are different, then we start off by
			# renaming the file. (Otherwise, if the old and new filepaths are
			# identical, then we don't need to do anything to the file.) If
			# the user requested that we use svn mv, then do that, otherwise we
			# use regular mv.
			if [ "${old_filepath}" != "${new_filepath}" ]; then
	
				if [ -n "$use_svn_mv_flag" ]; then
	
					svn mv "${old_filepath}" "${new_filepath}"
				else
	
					mv -f "${old_filepath}" "${new_filepath}"
				fi
			fi
		#else

			# A dry run still needs the act upon the "new" file, so if the
			# filepaths are different, simply set the new filepath to the
			# old one. (We won't need the previous value of new_filepath 
			# anymore.)
			#if [ "${old_filepath}" != "${new_filepath}" ]; then
			#	new_filepath="${old_filepath}"
			#fi
		fi

		# Handle the cases that might change the contents of the file.
		if [ "$contents_sed_expr" != "" ] ||
		   [ "$contents_sed_script" != "" ]; then
			
			# Execute the sed command based on whether the sed action was given
			# as a command line expression or a script residing in a file.
			if   [ "$contents_sed_script" != "" ]; then
				
				# Perform the action, saving the result to a temporary file.
				cat "${new_filepath}" | sed -f ${contents_sed_script} \
				                      > ${new_filepath}.${temp_file_suffix}
			
			elif [ "$contents_sed_expr" != "" ]; then
				
				# Perform the action, saving the result to a temporary file.
				cat "${new_filepath}" | sed -e "${contents_sed_expr}" \
				                      > ${new_filepath}.${temp_file_suffix}
			fi
			
			# Check the difference.
			file_diff=$(diff "${new_filepath}" "${new_filepath}.${temp_file_suffix}")
			
			
			# If we are not performing a dry run, proceed.
			if [ -z "$dry_run_flag" ]; then
			
				# If the file contents change.
				if [ -n "$file_diff" ]; then
				
					# Apply the old file permissions to the new file (before we
					# potentially overwrite the old file with the new one).
					chmod ${old_perms} "${new_filepath}.${temp_file_suffix}"

					# Apply the file contents changes to the new filepath (which may
					# or may not be the same as the old filepath).
					mv -f "${new_filepath}.${temp_file_suffix}" "${new_filepath}"

				else
					# Otherwise remove the new temporary file since it is identical
					# to the original.
					rm -f "${new_filepath}.${temp_file_suffix}"
				fi
			else
				# Simply remove the file since we are only performing a dry run.
				rm -f "${new_filepath}.${temp_file_suffix}"
			fi

		fi
		
		# Check for dos2unix. If it's not here, we'll just substitute cat.
		#type_dos2unix=$(type -path dos2unix)
		#if [ -n "$type_dos2unix" ]; then
		#	dos2unix -q ${new_filepath}
		#fi

		# Create a string that indicates what we are changing. We'll use this in
		# the verbose progress echo to indicate how the file is or would be changed.
		if   [ -n "$grep_filename" ] && [ -n "$file_diff" ]; then
			which_matches="filename/contents"
			file_touched="yes"
		elif [ -n "$grep_filename" ] && [ -z "$file_diff" ]; then
			which_matches="filename         "
			file_touched="yes"
		elif [ -z "$grep_filename" ] && [ -n "$file_diff" ]; then
			which_matches="         contents"
			file_touched="yes"
		else
			which_matches=""
			file_touched="no"
		fi
		
		# Be verbose, if requested, about which file we're looking at.
		if [ "$verbose_level" != "0" ]; then
			
			# But we only need to output a line if the file was touched.
			if [ "$file_touched" != "no" ]; then
				
				# Construct a relative filepath by stripping the initial root
				# directory so that the output does not span as many columns on
				# the terminal.
				rel_old_filepath=${old_filepath#${initial_root_dir}/}

				# Add a "dry run" condition to the output if we're doing a dry-run
				# so that the user knows we didn't really change anything.
				if [ -z "$dry_run_flag" ]; then
					echo "$script_name: Changing [${which_matches}] of ${rel_old_filepath}"
				else
					echo "$script_name: Changing (dry run) [${which_matches}] of ${rel_old_filepath}"
				fi
			fi
		fi
		
	done
	
	# Exit peacefully.
	return 0
}




recursive_sed()
{
	# Local variable declarations
	local item sub_items curr_dir this_dir
	
	
	# Extract our argument
	curr_dir="$1"
	

	# Call our function to perform the sed operations on the files in the
	# directory given.
	perform_sed "${curr_dir}"
	

	# If we were asked to act recursively, then continue processing
	# curr_dir's contents.
	if [ "$recursive_flag" = "1" ]; then
		
		# Get a listing of items in the directory according to the hidden
		# files/directories flag.
		if [ -n "$hidden_files_dirs_flag" ]; then

			# Get a listing of the directories in curr_dir (including hidden
			# files and directories).
			sub_items=$(ls -a "$curr_dir")

		else

			# Get a listing of the directories in curr_dir.
			sub_items=$(ls "$curr_dir")
		fi

		#echo "sub_items: $sub_items"
	
		# Descend into the contents of curr_dir, calling recursive_sed on
		# any items that are directories.
		find "${curr_dir}" -maxdepth 1 -name "*" -print | while read item
		do

			#echo "conisdering item: $item"

			# Skip the current directory.
			if [ "${item}" == "${curr_dir}" ]; then
				continue
			fi

			# If item is a directory, descend into it.
			if [ -d "$item" ]; then
			
				#echo "item is dir: $item"

				recursive_sed "$item"
			fi
		done

	fi
	
	
	# Return peacefully
	return 0
}




main()
{
	# Variables set by getopts.
	dry_run_flag=""
	hidden_files_dirs_flag=""
	use_svn_mv_flag=""
	filename_pattern=""
	root_dir=""
	initial_root_dir=""
	verbose_level=""
	filename_sed_expr=""
	contents_sed_expr=""
	contents_sed_script=""

	recursive_flag="1"	

	
	# Get the script name
	script_name=${0##*/}
	
	
	# Local variable declarations.
	local item sub_items this_dir
	
	
	# Process our command line options.
	while getopts ":c:df:hp:r:s:nNv:" opt; do
		case $opt in
			d  ) dry_run_flag="1" ;;
			h  ) hidden_files_dirs_flag="1" ;;
			n  ) use_svn_mv_flag="1" ;;
			N  ) recursive_flag="0" ;;
			v  ) verbose_level="$OPTARG" ;;
			p  ) filename_pattern="$OPTARG" ;;
			r  ) root_dir="$OPTARG" ;;
			f  ) filename_sed_expr="$OPTARG" ;;
			c  ) contents_sed_expr="$OPTARG" ;;
			s  ) contents_sed_script="$OPTARG" ;;
			\? ) print_usage
		esac
	done
	shift $(($OPTIND - 1))
	
	
	# Make sure we've parsed all command line arguments by now.
	if [ $# != "0" ]; then
		echo "${script_name}: Unparsed command line arguments! Try running with no arguments for help."
		exit 1
	fi
	
	
	# Make sure we received at least one of the required options.
	if [ -z "$filename_sed_expr" ] &&
	   [ -z "$contents_sed_expr" ] &&
	   [ -z "$contents_sed_script" ]; then
		print_usage
	fi
	
	
	# Make sure that both a file contents sed expression and sed script were
	# not given.
	if [ "$contents_sed_expr"   != "" ] &&
	   [ "$contents_sed_script" != "" ] ; then
		echo "${script_name}: The -c and -s options may not be used at the same time."
		exit 1
	fi


	# Make sure that verboseness level is valid.
	if [ "$verbose_level" != "0" ] && 
	   [ "$verbose_level" != "1" ] &&
	   [ "$verbose_level" != "2" ]; then
		verbose_level="1"
	fi
	
	# Prepare the filename pattern arguments to perform_sed().
	if [ "$filename_pattern" = "" ] ; then
		filename_pattern='*'
	fi

	# Prepare the directory arguments to perform_sed().
	if [ "$root_dir" != "" ] ; then
		
		# Strip / from end of directory paths, if there is one.
		root_dir=${root_dir%/}
	else
		root_dir=$PWD
	fi
	initial_root_dir=${root_dir}
	

	#echo "root_dir: $root_dir"


	# Begin recursing on the root directory.
	recursive_sed "$root_dir"


	# Exit peacefully
	return 0
}




# The script's main entry point, passing all parameters given.
main "$@"

