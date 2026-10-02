#!/bin/bash

# Name of this job
JOBNAME="ecat-${MYID}"

# Number of GPUs you want to use
GPUNUM=1
# CPUNUM=8

# The path where this job starts
JOBDIR="/path/to/EleCatML"

# The path that needs to be added to the environment variable $PATH
# such as the path of Python
MYENV=()
# MYENV+=("/path/to/conda/envs/pyg/bin")  # path 1
# MYENV+=("/path/to/vasp/bin")    # path 2
# You can add more path in this way

# The commands you want to run in this job
MYCOMMAND=(0)
MYCOMMAND+=("nvidia-smi")       # Command 1
MYCOMMAND+=("source /path/to/miniconda3/bin/activate fair-chem")        # Command 2
MYCOMMAND+=("bash scripts/run.sh ${MYID}")   # Command 3
# MYCOMMAND+=("sleep 5m")      # Command 4
# You can add more commands in this way
