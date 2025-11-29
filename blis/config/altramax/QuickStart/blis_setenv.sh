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
# Important!  Set the firmware flag to 204 for 2.04 or earlier,
# and 205 for 2.05 or later.
firmware=205
# Use altramax for both single and double socket - this might change
export BLIS_ARCH="altramax"
export BLIS_LIB=$BLIS_HOME/lib/$BLIS_ARCH
export BLIS_INC=$BLIS_HOME/include/$BLIS_ARCH


# Verify:
if [ "$quiet_setenv" = "" ]; then
  echo "BLIS_HOME set to $BLIS_HOME"
  echo "BLIS_INC set to $BLIS_INC"
  echo "-----------------------------------------------------------------"
  ls -l $BLIS_INC
  echo "-----------------------------------------------------------------"
  echo "BLIS_LIB set to $BLIS_LIB"
  echo "-----------------------------------------------------------------"
  ls -l $BLIS_LIB
  echo "-----------------------------------------------------------------"
fi

# Affinity Macros, etc
export BLIS_NUMA="numactl --localalloc"

# Use with firmware versions 2.04 and earlier.
# You can check the firmware version using dmidecode

export BLIS_AFFINITY_2S_2_04="0 64 32 96 4 68 36 100 1 65 33 97 5 69 37 101 2 66 34 98 6 70 38 102 3 67 35 99 7 71 39 103 8 72 40 104 12 76 44 108 9 73 41 105 13 77 45 109 10 74 42 106 14 78 46 110 11 75 43 107 15 79 47 111 16 80 48 112 20 84 52 116 17 81 49 113 21 85 53 117 18 82 50 114 22 86 54 118 19 83 51 115 23 87 55 119 24 88 56 120 26 90 58 122 25 89 57 121 27 91 59 123 28 92 60 124 30 94 62 126 29 93 61 125 31 95 63 127 128 192 160 224 132 196 164 228 129 193 161 225 133 197 165 229 130 194 162 226 134 198 166 230 131 195 163 227 135 199 167 231 136 200 168 232 140 204 172 236 137 201 169 233 141 205 173 237 138 202 170 234 142 206 174 238 139 203 171 235 143 207 175 239 144 208 176 240 148 212 180 244 145 209 177 241 149 213 181 245 146 210 178 242 150 214 182 246 147 211 179 243 151 215 183 247 152 216 184 248 154 218 186 250 153 217 185 249 155 219 187 251 156 220 188 252 158 222 190 254 157 221 189 253 159 223 191 255"

export BLIS_AFFINITY_1S_2_04="0 64 32 96 4 68 36 100 1 65 33 97 5 69 37 101 2 66 34 98 6 70 38 102 3 67 35 99 7 71 39 103 8 72 40 104 12 76 44 108 9 73 41 105 13 77 45 109 10 74 42 106 14 78 46 110 11 75 43 107 15 79 47 111 16 80 48 112 20 84 52 116 17 81 49 113 21 85 53 117 18 82 50 114 22 86 54 118 19 83 51 115 23 87 55 119 24 88 56 120 26 90 58 122 25 89 57 121 27 91 59 123 28 92 60 124 30 94 62 126 29 93 61 125 31 95 63 127"

# Use with firmware versions 2.05 and later
# You can check the firmware version using dmidecode

export BLIS_AFFINITY_2S_2_05="0 1 64 65 8 9 72 73 2 3 66 67 10 11 74 75 4 5 68 69 12 13 76 77 6 7 70 71 14 15 78 79 16 17 80 81 24 25 88 89 18 19 82 83 26 27 90 91 20 21 84 85 28 29 92 93 22 23 86 87 30 31 94 95 32 33 96 97 40 41 104 105 34 35 98 99 42 43 106 107 36 37 100 101 44 45 108 109 38 39 102 103 46 47 110 111 48 49 112 113 52 53 116 117 50 51 114 115 54 55 118 119 56 57 120 121 60 61 124 125 58 59 122 123 62 63 126 127 128 129 192 193 136 137 200 201 130 131 194 195 138 139 202 203 132 133 196 197 140 141 204 205 134 135 198 199 142 143 206 207 144 145 208 209 152 153 216 217 146 147 210 211 154 155 218 219 148 149 212 213 156 157 220 221 150 151 214 215 158 159 222 223 160 161 224 225 168 169 232 233 162 163 226 227 170 171 234 235 164 165 228 229 172 173 236 237 166 167 230 231 174 175 238 239 176 177 240 241 180 181 244 245 178 179 242 243 182 183 246 247 184 185 248 249 188 189 252 253 186 187 250 251 190 191 254 255"

export BLIS_AFFINITY_1S_2_05="0 1 64 65 8 9 72 73 2 3 66 67 10 11 74 75 4 5 68 69 12 13 76 77 6 7 70 71 14 15 78 79 16 17 80 81 24 25 88 89 18 19 82 83 26 27 90 91 20 21 84 85 28 29 92 93 22 23 86 87 30 31 94 95 32 33 96 97 40 41 104 105 34 35 98 99 42 43 106 107 36 37 100 101 44 45 108 109 38 39 102 103 46 47 110 111 48 49 112 113 52 53 116 117 50 51 114 115 54 55 118 119 56 57 120 121 60 61 124 125 58 59 122 123 62 63 126 127"

# Parallelism on the Altramax is very flat:

# Set JC to number of sockets:
export BLIS_JC_NT=2

# Set JR to groups of 8:
export BLIS_HR_NT=8

# Set IC to the number of cores per socket / 8:
export BLIS_IC_NT=16

# Experimental:  Allow you to set threading and
# Core affinity on single or dual sockets for
# N threads.  Currently, we only support N as
# a multple of 8

# Maximum Altramax cores per socket
CPS=128

# Use Bash Arrays:


if (($firmware == 204)); then
    arrayCoreIDs=(0 64 32 96 4 68 36 100 1 65 33 97 5 69 37 101 2 66 34 98 6 70 38 102 3 67 35 99 7 71 39 103 8 72 40 104 12 76 44 108 9 73 41 105 13 77 45 109 10 74 42 106 14 78 46 110 11 75 43 107 15 79 47 111 16 80 48 112 20 84 52 116 17 81 49 113 21 85 53 117 18 82 50 114 22 86 54 118 19 83 51 115 23 87 55 119 24 88 56 120 26 90 58 122 25 89 57 121 27 91 59 123 28 92 60 124 30 94 62 126 29 93 61 125 31 95 63 127 128 192 160 224 132 196 164 228 129 193 161 225 133 197 165 229 130 194 162 226 134 198 166 230 131 195 163 227 135 199 167 231 136 200 168 232 140 204 172 236 137 201 169 233 141 205 173 237 138 202 170 234 142 206 174 238 139 203 171 235 143 207 175 239 144 208 176 240 148 212 180 244 145 209 177 241 149 213 181 245 146 210 178 242 150 214 182 246 147 211 179 243 151 215 183 247 152 216 184 248 154 218 186 250 153 217 185 249 155 219 187 251 156 220 188 252 158 222 190 254 157 221 189 253 159 223 191 255)
elif (($firmware == 205)); then
    arrayCoreIDs=(0 1 64 65 8 9 72 73 2 3 66 67 10 11 74 75 4 5 68 69 12 13 76 77 6 7 70 71 14 15 78 79 16 17 80 81 24 25 88 89 18 19 82 83 26 27 90 91 20 21 84 85 28 29 92 93 22 23 86 87 30 31 94 95 32 33 96 97 40 41 104 105 34 35 98 99 42 43 106 107 36 37 100 101 44 45 108 109 38 39 102 103 46 47 110 111 48 49 112 113 52 53 116 117 50 51 114 115 54 55 118 119 56 57 120 121 60 61 124 125 58 59 122 123 62 63 126 127 128 129 192 193 136 137 200 201 130 131 194 195 138 139 202 203 132 133 196 197 140 141 204 205 134 135 198 199 142 143 206 207 144 145 208 209 152 153 216 217 146 147 210 211 154 155 218 219 148 149 212 213 156 157 220 221 150 151 214 215 158 159 222 223 160 161 224 225 168 169 232 233 162 163 226 227 170 171 234 235 164 165 228 229 172 173 236 237 166 167 230 231 174 175 238 239 176 177 240 241 180 181 244 245 178 179 242 243 182 183 246 247 184 185 248 249 188 189 252 253 186 187 250 251 190 191 254 255)
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

