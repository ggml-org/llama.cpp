#!/usr/bin/env python
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

# Import modules
import os
import sys
import getopt
import re

def print_usage():

	my_print( " " )
	my_print( " %s" % script_name )
	my_print( " " )
	my_print( " Field G. Van Zee" )
	my_print( " " )
	my_print( " Generate a monolithic header by recursively replacing all #include" )
	my_print( " directives in a selected file with the contents of the header files" )
	my_print( " they reference." )
	my_print( " " )
	my_print( " Usage:" )
	my_print( " " )
	my_print( "   %s header header_out temp_dir dir_list" % script_name )
	my_print( " " )
	my_print( " Arguments:" )
	my_print( " " )
	my_print( "   header        The filepath to the top-level header, which is the file" )
	my_print( "                 that will #include all other header files." )
	my_print( " " )
	my_print( "   header_out    The filepath of the file into which the script will output" )
	my_print( "                 the monolithic header." )
	my_print( " " )
	my_print( "   temp_dir      A directory in which temporary files may be created." )
	my_print( "                 NOTE: No temporary files are created in the current" )
	my_print( "                 implementation, but this argument must still be specified." )
	my_print( " " )
	my_print( "   dir_list      The list of directory paths in which to search for the" )
	my_print( "                 headers that are #included by 'header'. By default, these" )
	my_print( "                 directories are scanned for .h files, but sub-directories" )
	my_print( "                 within the various directories are not inspected. If the" )
	my_print( "                 -r option is given, these directories are recursively" )
	my_print( "                 scanned. In either case, the subset of directories scanned" )
	my_print( "                 that actually contains .h files is then searched whenever" )
	my_print( "                 a #include directive is encountered in 'header' (or any" )
	my_print( "                 file subsequently #included). If a referenced header file" )
	my_print( "                 is not found, the #include directive is left untouched and" )
	my_print( "                 translated directly into 'header_out'." )
	my_print( " " )
	my_print( " The following options are accepted:" )
	my_print( " " )
	my_print( "   -r          recursive" )
	my_print( "                 Scan the directories listed in 'dir_list' recursively when" )
	my_print( "                 searching for .h header files. By default, the directories" )
	my_print( "                 are not searched recursively." )
	my_print( " " )
	my_print( "   -c          strip C-style comments" )
	my_print( "                 Strip comments enclosed in /* */ delimiters from the" )
	my_print( "                 output, including multi-line comments. By default, C-style" )
	my_print( "                 comments are not stripped." )
	my_print( " " )
	my_print( "   -o SCRIPT   output script name" )
	my_print( "                 Use SCRIPT as a prefix when outputting messages instead" )
	my_print( "                 the script's actual name. Useful when the current script" )
	my_print( "                 is going to be called from within another, higher-level" )
	my_print( "                 driver script and seeing the current script's name might" )
	my_print( "                 unnecessarily confuse the user." )
	my_print( " " )
	my_print( "   -v [0|1|2]  verboseness level" )
	my_print( "                 level 0: silent  (no output)" )
	my_print( "                 level 1: default (single character '.' per header)" )
	my_print( "                 level 2: verbose (several lines per header)." )
	my_print( " " )
	my_print( "   -h          help" )
	my_print( "                 Output this information and exit." )
	my_print( " " )


# ------------------------------------------------------------------------------

def canonicalize_ws( s ):

	return re.sub( r'\s+', ' ', s ).strip()

# ---

def my_print( s ):

	sys.stdout.write( "%s\n" % s )

# ---

#def echov1( s ):
#
#	if verbose_flag == "1":
#		print "%s: %s" % ( output_name, s )

def echov1_n( s ):

	if verbose_flag == "1":
		sys.stdout.write( s )
		sys.stdout.flush()

def echov1_n2( s ):

	if verbose_flag == "1":
		sys.stdout.write( "%s\n" % s )
		sys.stdout.flush()

# ---

def echov2( s ):

	if verbose_flag == "2":
		sys.stdout.write( "%s: %s\n" % ( output_name, s ) )
		sys.stdout.flush()

def echov2_n( s ):

	if verbose_flag == "2":
		sys.stdout.write( output_name )
		sys.stdout.write( ": " )
		sys.stdout.write( s )
		sys.stdout.flush()

def echov2_n2( s ):

	if verbose_flag == "2":
		sys.stdout.write( "%s\n" % s )
		sys.stdout.flush()

# ------------------------------------------------------------------------------

def list_contains_header( items ):

	rval = False
	for item in items:

		is_h = re.search( r"\.h", item )

		if is_h:
			rval = True
			break

	return rval

# ------------------------------------------------------------------------------

def get_header_path( filename, header_dirpaths ):

	filepath = None

	# Search each directory path for the filename given.
	for dirpath in header_dirpaths:

		# Construct a possible path to the sought-after file.
		cur_filepath = "%s/%s" % ( dirpath, filename )

		# Check whether the file exists.
		found = os.path.exists( cur_filepath )
		if found:
			filepath = cur_filepath
			break

	return filepath

# ------------------------------------------------------------------------------

def strip_cstyle_comments( string ):

	return re.sub( r"/\*.*?\*/", "", string, flags=re.S )

# ------------------------------------------------------------------------------

def flatten_header( inputfile, header_dirpaths, cursp ):

	# This string is inserted after #include directives after having
	# determined that they are not present in the directory tree.
	skipstr  = "// skipped"
	beginstr = "// begin "
	endstr   = "// end "

	ostring  = ""

	# Open the input file to process.
	ifile = open( inputfile, "r" )

	# A counter to track the line number being parsed within the current file.
	# This counter, when selectively encoded into the flattened header via #line
	# directives, facilitates easier debugging. (When the compiler finds an
	# issue, it will be able to refer to the line number within the constituent
	# header file rather than the flattened one.)
	lineno = 0

	# Iterate over the lines in the file.
	while True:

		# Increment the line number.
		lineno += 1

		# Read a line in the file.
		line = ifile.readline()

		# Check for EOF.
		if line == '': break

		# Check for the #include directive and isolate the header name within
		# a group (parentheses).
		#result = re.search( '^[\s]*#include (["<])([\w\.\-/]*)([">])', line )
		result = regex.search( line )

		# If the line contained a #include directive, we must try to replace
		# it with the contents of the header referenced by the directive.
		if result:

			# Extract the header file referenced in the #include directive,
			# saved as the second group in the regular expression
			# above.
			header = result.group(2)

			echov2( "%sfound reference to '%s'." % ( cursp, header ) )

			# Search for the path to the header referenced in the #include
			# directive.
			header_path = get_header_path( header, header_dirpaths )

			# First, check if the header is our root header (and if so, ignore it).
			# Otherwise, if the header was found, we recurse. Otherwise, we output
			# the #include directive with a comment indicating that it as skipped
			if header == root_inputfile:

				markl = result.group(1)
				markr = result.group(3)

				echov2( "%sthis is the root header '%s'; commenting out / skipping." \
				        % ( cursp, header ) )

				# If the header found is our root header, then we cannot
				# recurse into it lest we enter an infinite loop. Output the
				# line but make sure it's commented out entirely.
				ostring += "%s #include %c%s%c %c" \
				           % ( skipstr, markl, header, markr, '\n' )

			elif header_path:

				echov2( "%slocated file '%s'; recursing." \
				        % ( cursp, header_path ) )

				# Mark the beginning of the header being inserted.
				ostring += "%s%s%c" % ( beginstr, header, '\n' )
				if line_numbers:
					ostring += "#line %d \"%s\"%c\n" % ( 1, header_path, '\n' )

				# Recurse on the header, accumulating the string.
				ostring += flatten_header( header_path, header_dirpaths, cursp + "  " )

				# Mark the end of the header being inserted.
				ostring += "%s%s%c" % ( endstr, header, '\n' )
				if line_numbers:
					ostring += "#line %d \"%s\"%c\n" % ( lineno+1, inputfile, '\n' )

				echov2( "%sheader file '%s' fully processed." \
				        % ( cursp, header_path ) )

			else:

				markl = result.group(1)
				markr = result.group(3)

				echov2( "%scould not locate file '%s'; marking as skipped." \
				        % ( cursp, header ) )

				# If the header was not found, output the line with a
				# comment that the header was skipped.
				ostring += "#include %c%s%c %s%c" \
				           % ( markl, header, markr, skipstr, '\n' )
			# endif

		else:
			# If the line did not contain a #include directive, simply output
			# the line verbatim.
			ostring += "%s" % line

		# endif

	# endwhile

	# Close the input file.
	ifile.close()

	echov1_n( "." )

	return ostring

# ------------------------------------------------------------------------------

def find_header_dirs( dirpath ):

	header_dirpaths = []
	for root, dirs, files in os.walk( dirpath, topdown=True ):

		echov2_n( "scanning contents of %s" % root )

		if list_contains_header( files ):

			echov2_n2( "...found headers" )
			header_dirpaths.append( root )

		else:
			echov2_n2( "" )

		#endif

	#endfor

	return header_dirpaths

# ------------------------------------------------------------------------------

# Global variables.
script_name    = None
output_name    = None
strip_comments = None
recursive_flag = None
line_numbers   = None
verbose_flag   = None
regex          = None
root_inputfile = None

def main():

	global script_name
	global output_name
	global strip_comments
	global recursive_flag
	global line_numbers
	global verbose_flag
	global regex
	global root_inputfile

	# Obtain the script name.
	path, script_name = os.path.split(sys.argv[0])

	output_name    = script_name

	strip_comments = False
	recursive_flag = False
	line_numbers   = False
	verbose_flag   = "1"

	nestsp         = "  "

	# Process our command line options.
	try:
		opts, args = getopt.getopt( sys.argv[1:], "o:rclhv:" )

	except getopt.GetoptError as err:
		# print help information and exit:
		my_print( str(err) ) # will print something like "option -a not recognized"
		print_usage()
		sys.exit(2)

	for opt, optarg in opts:
		if   opt == "-o":
			output_name = optarg
		elif opt == "-r":
			recursive_flag = True
		elif opt == "-l":
			line_numbers = True
		elif opt == "-c":
			strip_comments = True
		elif opt == "-v":
			verbose_flag = optarg
		elif opt == "-h":
			print_usage()
			sys.exit()
		else:
			print_usage()
			sys.exit()

	if line_numbers and strip_comments:
		my_print( "WARNING: stripping comments will result in inaccurate line numbers" )

	# Make sure that the verboseness level is valid.
	if ( verbose_flag != "0" and
	     verbose_flag != "1" and
	     verbose_flag != "2" ):
		my_print( "%s Invalid verboseness argument: %s" \
		                  % output_name, verbose_flag )
		sys.exit()

	# Print usage if we don't have exactly four arguments.
	if len( args ) != 4:
		print_usage()
		sys.exit()

	# Acquire the four required arguments:
	# - the input header file,
	# - the output header file,
	# - the temporary directory in which we can write intermediate files,
	# - the list of directories in which to search for the headers.
	inputfile  = args[0]
	outputfile = args[1]
	temp_dir   = args[2]
	dir_list   = args[3]

	# Save the filename (basename) part of the input file (or root file) into a
	# global variable that we can access later from within flatten_header().
	root_inputfile = os.path.basename( inputfile )

	# Separate the directories into distinct strings.
	dir_list = dir_list.split()

	# First, confirm that the directories in dir_list are valid.
	dir_list_checked = []
	for item in dir_list:

		#absitem = os.path.abspath( item )

		echov2_n( "checking " + item )

		if os.path.exists( item ):
			dir_list_checked.append( item )
			echov2_n2( "...directory exists." )
		else:
			echov2_n2( "...invalid directory; omitting." )

	# endfor

	# Overwrite the original dir_list with the updated copy that omits
	# invalid directories.
	dir_list = dir_list_checked

	echov2( "check summary:" )
	echov2( "  accessible directories:" )
	echov2( "  %s" % ' '.join( dir_list ) )

	# Generate a list of directories (header_dirpaths) which will be searched
	# whenever a #include directive is encountered. The method by which
	# header_dirpaths is compiled will depend on whether the recursive flag
	# was given.
	if recursive_flag:

		header_dirpaths = []
		for d in dir_list:

			# For each directory in dir_list, recursively walk that directory
			# and return a list of directories that contain headers.
			d_dirpaths = find_header_dirs( d )

			# Add the list resulting from the current search to the running
			# list of directory paths that contain headers.
			header_dirpaths += d_dirpaths

		# endfor

	else:

		# If the recursive flag was not given, we can just use dir_list
		# as-is, though we opt to filter out the directories that don't
		# contain .h files.

		header_dirpaths = []
		for d in dir_list:

			echov2_n( "scanning %s" % d )

			# Acquire a list of the directory's contents.
			sub_items = os.listdir( d )

			# If there is at least one header present, add the current
			# directory to the list of header directories.
			if list_contains_header( sub_items ):
				header_dirpaths.append( d )
				echov2_n2( "...found headers." )
			else:
				echov2_n2( "...no headers found." )
			# endif

		# endfor

	# endfor

	echov2( "scan summary:" )
	echov2( "  headers found in:" )
	echov2( "  %s" % ' '.join( header_dirpaths ) )

	echov2( "preparing to monolithify '%s'" % inputfile )

	echov2( "new header will be saved to '%s'" % outputfile )

	echov1_n( "." )

	# Open the output file.
	ofile = open( outputfile, "w" )

	# Precompile the main regular expression used to isolate #include
	# directives and the headers they reference. This regex object will
	# get reused over and over again in flatten_header().
	regex = re.compile( r'^[\s]*#include (["<])([\w\.\-/]*)([">])' )

	# Recursively substitute headers for occurrences of #include directives.
	final_string = flatten_header( inputfile, header_dirpaths, nestsp )

	# Strip C-style comments from the final output, if requested.
	if strip_comments:
		final_string = strip_cstyle_comments( final_string )

	# Write the lines to the file.
	ofile.write( final_string )

	# Close the output file.
	ofile.close()

	echov2( "substitution complete." )
	echov2( "monolithic header saved as '%s'" % outputfile )

	echov1_n2( "." )

	return 0




if __name__ == "__main__":
	main()
