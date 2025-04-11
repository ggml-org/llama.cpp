#!/bin/bash
dpu-upmem-dpurte-clang -Wall -Wextra -O3 -DNR_TASKLETS=16 -DBL=11 -o gemv_dpu dpu_main.c
