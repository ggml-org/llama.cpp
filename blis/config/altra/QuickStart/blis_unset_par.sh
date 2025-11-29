#!/bin/blis

if [ "$1" = "quiet" ]; then
    quiet_unsetpar="quiet"
else
    quiet_unsetpar=""
fi

if [ "$quiet_unsetpar" = "" ]; then
    echo "#########################################################"
    echo " UNSETTING BLIS ENVIRONMENT VARIABLES THAT SET THREADING"
    echo " AND AFFINITY."
    echo "#########################################################"
fi

unset BLIS_JC_NT
unset BLIS_JR_NT
unset BLIS_IC_NT
unset BLIS_NUM_THREADS
unset OMP_NUM_THREADS
unset GOMP_CPU_AFFINITY

