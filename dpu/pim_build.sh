#!/bin/bash
dpu-upmem-dpurte-clang -Wall -Wextra -O2 -DNR_TASKLETS=8 -DBL=11 -o gemv_dpu dpu_main.c
