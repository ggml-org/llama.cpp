#!/usr/bin/env python3
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2018, The University of Texas at Austin
#  Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
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
import time
import statistics


def print_usage():

	my_print( " " )
	my_print( " %s" % script_name )
	my_print( " " )
	my_print( " Field G. Van Zee" )
	my_print( " " )
	my_print( " Repeatedly run a test driver and accumulate statistics for the" )
	my_print( " output." )
	my_print( " " )
	my_print( " Usage:" )
	my_print( " " )
	my_print( "   %s [options] drivername" % script_name )
	my_print( " " )
	my_print( " Arguments:" )
	my_print( " " )
	my_print( "   drivername    The filename/path of the test driver to run. The" )
	my_print( "                 test driver must output its performance data to" )
	my_print( "                 standard output." )
	my_print( " " )
	my_print( " The following options are accepted:" )
	my_print( " " )
	my_print( "   -c num      performance column index" )
	my_print( "                 Find the performance result in column index <num> of" )
	my_print( "                 the test driver's output. Here, a column is defined" )
	my_print( "                 as a contiguous sequence of non-whitespace characters," )
	my_print( "                 with the column indices beginning at 0. By default," )
	my_print( "                 the second-to-last column index in the output is used." )
	my_print( " " )
	my_print( "   -d delay    sleep() delay" )
	my_print( "                 Wait <delay> seconds after each execution of the" )
	my_print( "                 test driver. The default delay is 0." )
	my_print( " " )
	my_print( "   -n niter    number of iterations" )
	my_print( "                 Execute the test driver <niter> times. The default" )
	my_print( "                 value is 10." )
	my_print( " " )
	my_print( "   -q          quiet; summary only" )
	my_print( "                 Do not output statistics after every new execution of" )
	my_print( "                 the test driver; instead, only output the final values" )
	my_print( "                 after all iterations are complete. The default is to" )
	my_print( "                 output updated statistics after each iteration." )
	my_print( " " )
	my_print( "   -h          help" )
	my_print( "                 Output this information and exit." )
	my_print( " " )


# ------------------------------------------------------------------------------

def my_print( s ):

	sys.stdout.write( "%s\n" % s )
	#sys.stdout.flush()

# ------------------------------------------------------------------------------

# Global variables.
script_name    = None
output_name    = None

def main():

	global script_name
	global output_name

	# Obtain the script name.
	path, script_name = os.path.split(sys.argv[0])

	output_name = script_name

	# Default values for optional arguments.
	#perf_col = 9
	perf_col = -1
	delay    = 0
	niter    = 10
	quiet    = False

	# Process our command line options.
	try:
		opts, args = getopt.getopt( sys.argv[1:], "c:d:n:hq" )

	except getopt.GetoptError as err:
		# print help information and exit:
		my_print( str(err) ) # will print something like "option -a not recognized"
		print_usage()
		sys.exit(2)

	for opt, optarg in opts:
		if   opt == "-c":
			perf_col = optarg
		elif opt == "-d":
			delay = optarg
		elif opt == "-n":
			niter = optarg
		elif opt == "-q":
			quiet = True
		elif opt == "-h":
			print_usage()
			sys.exit()
		else:
			print_usage()
			sys.exit()

	# Print usage if we don't have exactly one argument.
	if len( args ) != 1:
		print_usage()
		sys.exit()

	# Acquire our only mandatory argument: the name of the test driver.
	driverfile = args[0]

	#my_print( "test driver: %s" % driverfile )
	#my_print( "column num:  %s" % perf_col )
	#my_print( "delay:       %s" % delay )
	#my_print( "num iter:    %s" % niter )

	# Build a list of iterations.
	iters = range( int(niter) )

	# Run the test driver once to detect the number of lines of output.
	p = subprocess.run( driverfile, stdout=subprocess.PIPE )
	lines0 = p.stdout.decode().splitlines()
	num_lines0 = int(len(lines0))

	# Initialize the list of lists (one list per performance result).
	aperf = []
	for i in range( num_lines0 ):
		aperf.append( [] )

	for it in iters:

		# Run the test driver.
		p = subprocess.run( driverfile, stdout=subprocess.PIPE )

		# Acquire the lines of output.
		lines = p.stdout.decode().splitlines()

		# Accumulate the test driver's latest results into aperf.
		for i in range( num_lines0 ):

			# Parse the current line to find the performance value.
			line  = lines[i]
			words = line.split()
			if perf_col == -1:
				perf  = words[ len(words)-2 ]
			else:
				perf  = words[ int(perf_col) ]

			# As unlikely as it is, guard against Inf and NaN.
			if float(perf) ==  float('Inf') or \
			   float(perf) == -float('Inf') or \
			   float(perf) ==  float('NaN'): perf = 0.0

			# Add the performance value to the list at the ith entry of aperf.
			aperf[i].append( float(perf) )

			# Compute stats for the current line.
			avgp = statistics.mean( aperf[i] )
			maxp =             max( aperf[i] )
			minp =             min( aperf[i] )

			# Only compute stdev() when we have two or more data points.
			if len( aperf[i] ) > 1: stdp = statistics.stdev( aperf[i] )
			else:                   stdp = 0.0

			# Construct a string to match the performance value and then
			# use that string to search-and-replace with four format specs
			# for the min, avg, max, and stdev values computed above.
			search = '%8s' % perf
			newline = re.sub( str(search), ' %7.2f %7.2f %7.2f %6.2f', line )

			# Search for the column index range that would be present if this were
			# matlab-compatible output. The index range will typically be 1:n,
			# where n is the number of columns of data.
			found_index = False
			for word in words:
				if re.match( '1:', word ):
					index_str = word
					found_index = True
					break

			# If we find the column index range, we need to update it to reflect
			# the replacement of one column of data with four, for a net increase
			# of columns. We do so via another instance of re.sub() in which we
			# search for the old index string and replace it with the new one.
			if found_index:
				last_col = int(index_str[2]) + 3
				new_index_str = '1:%1s' % last_col
				newline = re.sub( index_str, new_index_str, newline )

			# If the quiet flag was not give, output the intermediate results.
			if not quiet:
				print( newline % ( float(minp), float(avgp), float(maxp), float(stdp) ) )

		# Flush stdout after each set of output prior to sleeping.
		sys.stdout.flush()

		# Sleep for a bit until the next iteration.
		time.sleep( int(delay) )

	# If the quiet flag was given, output the final results.
	if quiet:

		for i in range( num_lines0 ):

			# Parse the current line to find the performance value (only
			# needed for call to re.sub() below).
			line  = lines0[i]
			words = line.split()
			if perf_col == -1:
				perf  = words[ len(words)-2 ]
			else:
				perf  = words[ int(perf_col) ]

			# Compute stats for the current line.
			avgp = statistics.mean( aperf[i] )
			maxp =             max( aperf[i] )
			minp =             min( aperf[i] )

			# Only compute stdev() when we have two or more data points.
			if len( aperf[i] ) > 1: stdp = statistics.stdev( aperf[i] )
			else:                   stdp = 0.0

			# Construct a string to match the performance value and then
			# use that string to search-and-replace with four format specs
			# for the min, avg, max, and stdev values computed above.
			search = '%8s' % perf
			newline = re.sub( str(search), ' %7.2f %7.2f %7.2f %6.2f', line )

			# Search for the column index range that would be present if this were
			# matlab-compatible output. The index range will typically be 1:n,
			# where n is the number of columns of data.
			found_index = False
			for word in words:
				if re.match( '1:', word ):
					index_str = word
					found_index = True
					break

			# If we find the column index range, we need to update it to reflect
			# the replacement of one column of data with four, for a net increase
			# of columns. We do so via another instance of re.sub() in which we
			# search for the old index string and replace it with the new one.
			if found_index:
				last_col = int(index_str[2]) + 3
				new_index_str = '1:%1s' % last_col
				newline = re.sub( index_str, new_index_str, newline )

			# Output the results for the current line.
			print( newline % ( float(minp), float(avgp), float(maxp), float(stdp) ) )

		# Flush stdout afterwards.
		sys.stdout.flush()


	# Return from main().
	return 0




if __name__ == "__main__":
	main()
