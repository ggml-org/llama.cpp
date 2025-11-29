#!/usr/bin/env bash

get_config_var()
{
	# Parse the compiler assigned to the CC variable within the config.mk file.
	echo "$(grep "^ *$1 *:=" config.mk | sed 's/'$1' *:= *//')"
}

main()
{
	if [ ! -e config.mk ]; then
		echo "No config.mk file detected; have you configured BLIS?"
		exit 1
	fi

	CC=$(get_config_var CC)
	CONFIG_NAME=$(get_config_var CONFIG_NAME)
	BLIS_H_FLAT="include/${CONFIG_NAME}/blis.h"

	if [ ! -e ${BLIS_H_FLAT} ]; then
		echo "No monolithic blis.h file detected at ${BLIS_H_FLAT}; have you run 'make'?"
		exit 1
	fi

	#
	# Header line
	#
	echo "EXPORTS"

	#
	# Breakdown of commands:
	# $(CC) ...		# Pre-process blis.h, making sure to include all BLAS and CBLAS symbols
	#	| tr ...	# Make sure to split lines at ';' so that each declaration is on its own line
	#	| grep ...	# Find exported symbols
	#	| sed -E
	#	    -e ...	# 1. Remove all __attribute__ clauses
	#	    -e ...	# 2. Select only the portion before an opening '(' (if any)
	#	    -e ...	# 3. Pull out the last word, which is the function name.
	#	| grep ...	# Remove constants
	#	| grep ...	# Remove blank lines
	#	| sed  ...  # Remove trailing spaces
	#	| sort
	#	| uniq
	#
	${CC} -DBLIS_ENABLE_CBLAS=1 -DBLIS_ENABLE_BLAS=1 -E ${BLIS_H_FLAT} \
		| tr ';' '\n' \
		| grep visibility \
		| sed -E \
		    -e 's/__attribute__ *\( *\([^\)]+(\([^\)]+\) *)\) *\)//g' \
		    -e 's/(.*) *\(.*/\1/' \
		    -e 's/.* ([^ ].*)/\1/' \
		| grep -v BLIS \
		| grep -E '[^ ]' \
		| sed -e 's/[[:space:]]*$//g' \
		| sort \
		| uniq
}

main "$@"

