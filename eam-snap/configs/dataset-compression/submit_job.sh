#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "get_index_environments"
#SBATCH -A iap
#SBATCH -p pbatch

# source compress.sh
python get_idx_env.py -r eam_ruby_10000/
