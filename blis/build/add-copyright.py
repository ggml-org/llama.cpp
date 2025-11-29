#!/usr/bin/env python3
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2018, The University of Texas at Austin
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
import subprocess
import datetime


def print_usage():

	my_print( " " )
	my_print( " %s" % script_name )
	my_print( " " )
	my_print( " Field G. Van Zee" )
	my_print( " " )
	my_print( " Update copyright lines of all created or modified source files currently" )
	my_print( " staged in the git index, and also insert new copyright lines where they" )
	my_print( " currently are missing. This script targets copyright lines for one" )
	my_print( " organization at a time." )
	my_print( " " )
	my_print( " Usage:" )
	my_print( " " )
	my_print( "   %s [options]" % script_name )
	my_print( " " )
	my_print( " Arguments:" )
	my_print( " " )
	my_print( " " )
	my_print( " The following options are accepted:" )
	my_print( " " )
	my_print( "   -o org      organization name" )
	my_print( "                 Update and add copyrights for an organization named <org>." )
	my_print( "                 By default, <org> is 'Advanced Micro Devices, Inc.'" )
	my_print( " " )
	my_print( "   -u          update only" )
	my_print( "                 Update existing copyrights to reflect the current year," )
	my_print( "                 but do not add any additional copyright lines. With this" )
	my_print( "                 option, the script still only updates copyright lines for" )
	my_print( "                 the specified (or default) organization. The default is" )
	my_print( "                 to update but also add copyright lines where missing." )
	my_print( " " )
	my_print( "   -d          dry run" )
	my_print( "                 Go through all of the motions, but don't actually modify" )
	my_print( "                 any files. The default behavior is to not enable dry run." )
	my_print( " " )
	my_print( "   -q          quiet" )
	my_print( "                 Do not output feedback while processing each file. The" )
	my_print( "                 default behavior is to output one line of text to stdout" )
	my_print( "                 per file updated." )
	my_print( " " )
	my_print( "   -h          help" )
	my_print( "                 Output this information and exit." )
	my_print( " " )


# ------------------------------------------------------------------------------

def my_print( s ):

	sys.stdout.write( "%s\n" % s )
	#sys.stdout.flush()

def my_echo( s ):

	if not quiet:
		sys.stdout.write( "%s: %s\n" % ( output_name, s ) )
	#sys.stdout.flush()

# ------------------------------------------------------------------------------

def main():

	global script_name
	global output_name
	global quiet

	# Obtain the script name.
	path, script_name = os.path.split(sys.argv[0])

	output_name = script_name

	# Default values for optional arguments.
	the_org     = 'Advanced Micro Devices, Inc.'
	update_only = False
	dry_run     = False
	quiet       = False

	# Process our command line options.
	try:
		opts, args = getopt.getopt( sys.argv[1:], "do:uhq" )

	except getopt.GetoptError as err:
		# print help information and exit:
		my_print( str(err) ) # will print something like "option -a not recognized"
		print_usage()
		sys.exit(2)

	for opt, optarg in opts:
		if opt == "-o":
			the_org = optarg
		elif opt == "-u":
			update_only = True
		elif opt == "-d":
			dry_run = True
		elif opt == "-q":
			quiet = True
		elif opt == "-h":
			print_usage()
			sys.exit()
		else:
			print_usage()
			sys.exit()

	# Print usage if we don't have exactly zero arguments.
	if len( args ) != 0:
		print_usage()
		sys.exit()

	# Acquire our only mandatory argument.
	#driverfile = args[0]

	# Query the current year.
	the_time = datetime.datetime.now()
	cur_year = str(the_time.year)

	# We run 'git status' with --porcelain to make the output easily parseable.
	gitstatus = 'git status --porcelain'

	# Run the 'git status' command and capture the output.
	p = subprocess.run( gitstatus, stdout=subprocess.PIPE, shell=True )
	git_lines = p.stdout.decode().splitlines()
	git_num_lines = int( len( git_lines ) )

	# Consider each line of output from 'git status'
	for i in range( git_num_lines ):

		# Parse the current line to find the performance value.
		git_line  = git_lines[i]
		git_words = git_line.split()
		mod_char  = git_line[0]

		# Check the first character of the git output. We want to only update
		# files that are new ('A'), modified ('M'), or renamed ('R').
		if mod_char != 'A' and \
		   mod_char != 'M' and \
		   mod_char != 'R': continue

		# Identify the filename for the current line of 'git status' output.
		if mod_char == 'R':
			# For renamed files, we need to reference them by their new names,
			# which appear after the "->" char sequence in git_words[2].
			filename = git_words[3]
		else:
			filename = git_words[1]

		#my_echo( "-debug---- %s" % filename )

		# Start by opening the file. (We can assume it exists since it
		# was found by 'git status', so no need to check for existence.)
		# Read all lines in the file and then close it.
		f = open( filename, "r" )
		file_lines = f.readlines()
		f.close()

		# Concatenate all lines in the file into one string.
		file_string = "".join( file_lines )

		# Search for an existing copyright line.
		has_cr = re.search( r'Copyright \(C\)', file_string )

		# If the file does not have any copyright notice in it already, we
		# assume we don't need to update it.
		if not has_cr:
			my_echo( "[nocrline] %s" % filename )
			continue

		# Check whether the file already has a copyright for the_org. We may
		# need to use this information later.
		has_org_cr = re.search( r'Copyright \(C\) ([0-9][0-9][0-9][0-9]), %s' % the_org, file_string )

		# Initialize the list of processed (potentially modified) file lines.
		mod_file_lines = []

		# At this point we know that the file has at least one copyright, and
		# has_org_cr encodes whether it already has a copyright for the_org.

		# We process the files that we know already have copyrights for the_org
		# differently from the files that do not yet have them.
		if has_org_cr:

			# Iterate through the lines in the current file.
			for line in file_lines:

				result = re.search( r'Copyright \(C\) ([0-9][0-9][0-9][0-9]), %s' % the_org, line )

				# If the current line matches a copyright line for the_org...
				if result:

					# Extract the year saved as the first/only group in the
					# regular expression.
					old_year = result.group(1)

					# Don't need to update the year if it's already up-to-date.
					if old_year != cur_year:

						# Substitute the old year for the current year.
						find_line = ' %s, ' % old_year
						repl_line = ' %s, ' % cur_year
						line_ny = re.sub( find_line, repl_line, line )

						my_echo( "[updated ] %s" % filename )

						# Add the updated line to the running list.
						mod_file_lines += line_ny

					else:

						my_echo( "[up2date ] %s" % filename )

						# Add the unchanged line to the running list.
						mod_file_lines += line

				else:
					# Add the unchanged line to the running list.
					mod_file_lines += line

				# endif result

			# endfor

		else:

			# Don't go any further if we're only updating existing copyright
			# lines.
			if update_only:
				my_echo( "[nocrline] %s" % filename )
				continue

			num_file_lines = len( file_lines )

			# Iterate through the lines in the current file.
			for i in range( int(num_file_lines) ):

				line = file_lines[i]

				# Only look at the next line if we are not at the last line.
				if i < int(num_file_lines) - 1:
					line_next = file_lines[i+1]
				else:
					line_next = file_lines[i]

				# Try to match both the current line and the next line.
				result  = re.search( r'Copyright \(C\) ([0-9][0-9][0-9][0-9]), (.*)', line )
				resnext = re.search( r'Copyright \(C\) ([0-9][0-9][0-9][0-9]), (.*)', line_next )

				# Parse the results.
				if result:

					if resnext:

						# The current line matches but so does the next. Add the
						# current line unchanged to the running list.
						mod_file_lines += line

					else:

						# The current line matches but the next does not. Thus,
						# this branch only executes for the *last* copyright line
						# in the file.

						# Extract the year and organization from the matched
						# string.
						old_year = result.group(1)
						old_org  = result.group(2)

						# Set up search/replace strings to convert the current
						# line into one that serves as copyright for the_org.
						find_line = '%s, %s' % (old_year, old_org)
						repl_line = '%s, %s' % (cur_year, the_org)
						line_nyno = re.sub( find_line, repl_line, line )

						# Add the current line and then also insert our new
						# copyright line for the_org into the running list.
						mod_file_lines += line
						mod_file_lines += line_nyno

						my_echo( "[added   ] %s" % filename )

					# endif resnext

				else:

					# The current line does not match. Pass it through unchanged.
					mod_file_lines += line

				# endif result

			# endfor

		# endif has_org_cr

		if not dry_run:

			# Open the file for writing.
			f = open( filename, "w" )

			# Join the modified file lines into a single string.
			final_string = "".join( mod_file_lines )

			# Write the lines to the file.
			f.write( final_string )

			# Close the file.
			f.close()

		# endif not dry_run

	# Return from main().
	return 0




if __name__ == "__main__":
	main()
