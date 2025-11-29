#!/bin/bash

# execute in the general partition
#SBATCH --partition=general

# execute with 40 processes/tasks
#SBATCH --ntasks=1

# maximum time is 30 minutes
#SBATCH --time=00:30:00

# job name is my_job
#SBATCH --job-name=blis

# send email for status updates
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --mail-user=ntukanov

# change default output file name
#SBATCH --output=cfig.out

# load environment
module load gcc/8.2

# application execution
srun cfig.sh
