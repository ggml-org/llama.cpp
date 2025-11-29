#!/bin/bash
# This utility will remove all the configuration
# Specific QuickStart files from the blis directory.
# This is very useful when switching configurations!
#
if [[ -n "$BLIS_HOME" ]]; then
  echo "REMOVING ALL ALTRAMAX QUICKSTART FILES FROM $BLIS_HOME"
    
  rm $BLIS_HOME/blis_build_altramax_pthreads.sh
	rm $BLIS_HOME/blis_build_altramax.sh
	rm $BLIS_HOME/blis_build_both_libraries.sh
	
	rm $BLIS_HOME/blis_configure_altramax_pthreads.sh
	rm $BLIS_HOME/blis_configure_altramax.sh
	
	rm $BLIS_HOME/blis_quick_start_altramax.txt
	rm $BLIS_HOME/blis_setenv.sh
	
  rm $BLIS_HOME/blis_unset_par.sh
  rm $BLIS_HOME/blis_test.sh

	rm $BLIS_HOME/TimeDGEMM.c
	rm $BLIS_HOME/time_gemm.x
	
	rm $BLIS_HOME/blis_quick_start_uninstall_altramax.sh

else
  echo "ONLY USE THIS SCRIPT FROM THE BLIS HOME DIRECTORY!"
  echo "BLIS_HOME is not set!"
fi
