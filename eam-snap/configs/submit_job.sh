#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "extract_candidate_environments"
#SBATCH -A iap
#SBATCH -p pbatch

python extract_candidate_environments.py -t candidates_2000 -i quests_compressed_envs/eam-index-compressed.npz -d quests_compressed_envs/eam-quests-compressed.npz
srun -n $SLURM_CPUS_ON_NODE lmp -i extract_candidate_environments.in
python generate_xyz.py -t candidates_2000
