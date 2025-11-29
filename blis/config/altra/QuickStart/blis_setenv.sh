#!/bin/bash
#######################################################################
# Brought to you by Oracle Labs
#######################################################################
# Tested in bash and zsh
#######################################################################
# Sets up all the environment variables needed for running blis.
# For this reason, the script MUST be sourced, NOT executed!
# Needs to be run from BLIS directory to have a portable definition of
# BLIS_HOME.  If this setup doesn't work for you, you may hard code
# the path to BLIS_HOME, but then be careful if you copy or move it!
#######################################################################
# This is the top level blis directory - it is recommended to set to an absolulte path
# Can be overridden by user to be called anywhere, but then less portable
# export BLIS_HOME=.
# PORTABLE - Set BLIS_HOME to the blis directory containing this script
# We need to get the full path to the file in case this is called from another directory

if [ "$1" = "quiet" ]; then
    quiet_setenv="quiet"
else
    quiet_setenv=""
fi

if [[ -n "$BASH_VERSION" ]] ; then
    file_path_and_name="$( dirname "${BASH_SOURCE[0]}" )/blis_set_home_dir.sh"
else
    file_path_and_name="$( dirname "$0" )/blis_set_home_dir.sh"
fi

if [ -f "$file_path_and_name" ] ; then
	. $file_path_and_name quiet
else
  echo "ERROR - this file is not being executed from a blis home directory."
  echo "If you cannot use this script in a home directory, you can hardcode"
  echo "the absolute location of BLIS_HOME in blis_setenv,bash, but this"
  echo "is then less portable and more error prone with multiple blis"
  echo "directories."
  return
fi

#######################################################################
# Platform Specific:
#######################################################################
# Important!  Set the firmware number to 107 for firmware version 1.07 or earlier,
# and 108 for 1.08 or later.  We were unable to test 1.08 at this time.
#
firmware=108

qualifier="or later"
if (( firmware == 107 )); then
  qualifier="or earlier"
fi

# Use altra for both single and double socket - this might change
export BLIS_ARCH="altra"
export BLIS_LIB=$BLIS_HOME/lib/$BLIS_ARCH
export BLIS_INC=$BLIS_HOME/include/$BLIS_ARCH

# Verify:
if [ "$quiet_setenv" = "" ]; then
  echo "#################################################################"
  echo "CoreID affinity assumes firmware version on this machine is $firmware $qualifier"
  echo "BLIS_HOME set to $BLIS_HOME"
  echo "BLIS_INC set to $BLIS_INC"
  echo "================================================================="
  ls -l $BLIS_INC
  echo "-----------------------------------------------------------------"
  echo "BLIS_LIB set to $BLIS_LIB"
  echo "-----------------------------------------------------------------"
  ls -l $BLIS_LIB
  echo "#################################################################"
fi

# Affinity Macros, etc
export BLIS_NUMA="numactl --localalloc"

# Use with firmware versions 1.07 and earlier.

export BLIS_AFFINITY_2S_1_07="0 40 20 60 4 44 24 64 8 48 28 68 12 52 32 72 2 42 22 62 6 46 26 66 10 50 30 70 14 54 34 74 1 41 21 61 5 45 25 65 9 49 29 69 13 53 33 73 3 43 23 63 7 47 27 67 11 51 31 71 15 55 35 75 16 56 36 76 18 58 38 78 17 57 37 77 19 59 39 79 80 120 100 140 84 124 104 144 88 128 108 148 92 132 112 152 82 122 102 142 86 126 106 146 90 130 110 150 94 134 114 154 81 121 101 141 85 125 105 145 89 129 109 149 93 133 113 153 83 123 103 143 87 127 107 147 91 131 111 151 95 135 115 155 96 136 116 156 98 138 118 158 97 137 117 157 99 139 119 159"

export BLIS_AFFINITY_1S_1_07="0 40 20 60 4 44 24 64 8 48 28 68 12 52 32 72 2 42 22 62 6 46 26 66 10 50 30 70 14 54 34 74 1 41 21 61 5 45 25 65 9 49 29 69 13 53 33 73 3 43 23 63 7 47 27 67 11 51 31 71 15 55 35 75 16 56 36 76 18 58 38 78 17 57 37 77 19 59 39 79"

# Use with firmware versions 1.08+
# Warning - this has not been tested.
#
export BLIS_AFFINITY_2S_1_08="28, 29, 38, 39, 2, 3, 12, 13, 6, 7, 16, 17, 0, 1, 10, 11, 68, 69, 78, 79, 42, 43, 52, 53, 46, 47, 56, 57, 40, 41, 50, 51, 24, 25, 34, 35, 20, 21, 30, 31, 26, 27, 36, 37, 22, 23, 32, 33, 64, 65, 74, 75, 60, 61, 70, 71, 66, 67, 76, 77, 62, 63, 72, 73, 8, 9, 18, 19, 4, 5, 14, 15, 48, 49, 58, 59, 44, 45, 54, 55, 108, 109, 118, 119, 82, 83, 92, 93, 86, 87, 96, 97, 80, 81, 90, 91, 148, 149, 158, 159, 122, 123, 132, 133, 126, 127, 136, 137, 120, 121, 130, 131, 104, 105, 114, 115, 100, 101, 110, 111, 106, 107, 116, 117, 102, 103, 112, 113, 144, 145, 154, 155, 140, 141, 150, 151, 146, 147, 156, 157, 142, 143, 152, 153, 88, 89, 98, 99, 84, 85, 94, 95, 128, 129, 138, 139, 124, 125, 134, 135"

export BLIS_AFFINITY_1S_1_08="28, 29, 38, 39, 2, 3, 12, 13, 6, 7, 16, 17, 0, 1, 10, 11, 68, 69, 78, 79, 42, 43, 52, 53, 46, 47, 56, 57, 40, 41, 50, 51, 24, 25, 34, 35, 20, 21, 30, 31, 26, 27, 36, 37, 22, 23, 32, 33, 64, 65, 74, 75, 60, 61, 70, 71, 66, 67, 76, 77, 62, 63, 72, 73, 8, 9, 18, 19, 4, 5, 14, 15, 48, 49, 58, 59, 44, 45, 54, 55"

# Parallelism on the Altra is very flat:

# Set JC to number of sockets:
export BLIS_JC_NT=2

# Set JR to groups of 8:
export BLIS_HR_NT=8

# Set IC to the number of cores per socket / 8:
export BLIS_IC_NT=10

# Experimental:  Allow you to set threading and
# Core affinity on single or dual sockets for
# N threads.  Currently, we only support N as
# a multple of 8

# Max Altra cores per socket
CPS=80

# Use Bash Arrays:

# Choose which CoreID mapping to go with based on the firmware ID
if (($firmware == 107)); then
arrayCoreIDs=(0 40 20 60 4 44 24 64 8 48 28 68 12 52 32 72 2 42 22 62 6 46 26 66 10 50 30 70 14 54 34 74 1 41 21 61 5 45 25 65 9 49 29 69 13 53 33 73 3 43 23 63 7 47 27 67 11 51 31 71 15 55 35 75 16 56 36 76 18 58 38 78 17 57 37 77 19 59 39 79 80 120 100 140 84 124 104 144 88 128 108 148 92 132 112 152 82 122 102 142 86 126 106 146 90 130 110 150 94 134 114 154 81 121 101 141 85 125 105 145 89 129 109 149 93 133 113 153 83 123 103 143 87 127 107 147 91 131 111 151 95 135 115 155 96 136 116 156 98 138 118 158 97 137 117 157 99 139 119 159)
elif (($firmware == 108)); then
arrayCoreIDs=(28 29 38 39 2 3 12 13 6 7 16 17 0 1 10 11 68 69 78 79 42 43 52 53 46 47 56 57 40 41 50 51 24 25 34 35 20 21 30 31 26 27 36 37 22 23 32 33 64 65 74 75 60 61 70 71 66 67 76 77 62 63 72 73 8 9 18 19 4 5 14 15 48 49 58 59 44 45 54 55 108 109 118 119 82 83 92 93 86 87 96 97 80 81 90 91 148 149 158 159 122 123 132 133 126 127 136 137 120 121 130 131 104 105 114 115 100 101 110 111 106 107 116 117 102 103 112 113 144 145 154 155 140 141 150 151 146 147 156 157 142 143 152 153 88 89 98 99 84 85 94 95 128 129 138 139 124 125 134 135)
else
  echo "ERROR - UNSUPPORTED FIRMWARE $firmware"
  exit -1
fi

# Brief check: @ = list all numbers, loop for i in ${}; do ... done
# for Array Size, do ${#arr[@]}
# echo "CoreID array has ${#arrayCoreIDs[@]} elements"
# echo "CoreID array set to: ${arrayCoreIDs[@]}"

# Give the TOTAL core count:
# Single socket runs
blis_set_cores_and_sockets() {
  cores=$1
  sockets=$2
  # echo "Cores = $cores, sockets=$sockets"
  
	# Round up to nearest 8 cores per socket:
	cores_per_group=8
	if (( $sockets == 2 )); then
	  cores_per_group=16;
  fi
  core_round_inc=$(($cores_per_group-1))
	
	cores_per_socket=$(($cores))
	cores=$(($cores + $core_round_inc))
	groups_per_socket=$(($cores / $cores_per_group))
	rounded_cores=$(( $groups_per_socket * $cores_per_group ))

	# echo "Rounded Cores = $rounded_cores"
	# echo "Groups Per Socket = $groups_per_socket"
	
	# set the parallelism for one socket with N cores:
  # Set JC to number of sockets:
  export BLIS_JC_NT=$sockets

  # Set JR to groups of 8:
  export BLIS_JR_NT=8

  # Set IC to the number of cores per socket / 8:
  export BLIS_IC_NT=$groups_per_socket

  # Using an old version of zsh syntax that's compatible with bash
  
  if (( $sockets == 1 )); then
  
    # Simple single socket case
    # quotes
    # export GOMP_CPU_AFFINITY="\"${arrayCoreIDs[@]:0:$rounded_cores}\""
    # No quotes...
    export GOMP_CPU_AFFINITY="${arrayCoreIDs[@]:0:$rounded_cores}"
    
	else
	
    # Dual socket case
	  half_cores=$(( $rounded_cores / 2 ))
    # echo "Half cores are $half_cores"
    # quotes
    # export GOMP_CPU_AFFINITY="\"${arrayCoreIDs[@]:0:$half_cores} ${arrayCoreIDs[@]:$CPS:$half_cores}\""
    # No quotes
    export GOMP_CPU_AFFINITY="${arrayCoreIDs[@]:0:$half_cores} ${arrayCoreIDs[@]:$CPS:$half_cores}"
	fi

  echo "Activating $rounded_cores cores across $sockets sockets..."
  echo "GOMP_CPU_AFFINITY set to $GOMP_CPU_AFFINITY"
  echo "JC/IC/JR = $BLIS_JC_NT/$BLIS_IC_NT/$BLIS_JR_NT"
	}
	
# Convenience functions:
blis_set_cores_1S() { blis_set_cores_and_sockets $1 1 ; }
blis_set_cores_2S() { blis_set_cores_and_sockets $1 2 ; }

# For safety:
. ./blis_unset_par.sh





