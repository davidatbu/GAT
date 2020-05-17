#!/bin/bash -l

#$ -P llamagrp

#$ -l h_rt=24:00:00   # Specify the hard time limit for the job
#$ -N hparam_gat           # Give job a name
#$ -j y               # Merge the error and output streams into a single file

# Out log file
#$ -o out1.qlog

# Num of cores
#$ -pe omp 16

# Send an email when the job finishes or if it is aborted (by default no email is sent).

#$ -m ea

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

source /usr3/graduate/davidat/.bashrc

source /project/llamagrp/davidat/venvs/torch15/bin/activate

python train.py
