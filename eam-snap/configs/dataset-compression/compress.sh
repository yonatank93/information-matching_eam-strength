#!/bin/bash
#
# 36 jobs for Quartz/Borax
NUM_JOBS=56

# source $WORKSPACE/envs/quests/bin/activate
nsamples=10000
SAVE_DIR="eam_ruby_$nsamples_new"
mkdir -p $SAVE_DIR

# splits the input files eam.npz, which contains the descriptors
echo "First splits"
python scripts/split_dump.py ../quests_compressed_envs/eam-quests.npz -o $SAVE_DIR/splits-0

# FPS to sample each round
for i in $(seq 0 6)
do
    echo "Round $i"
    j=$(echo "$i + 1" | bc -l)

    # samples the splits in parallel
    echo "Sample the splits"
    time parallel -j $NUM_JOBS "python scripts/fps.py {1} -o $SAVE_DIR/samples-${i}/{1/.} -n $nsamples" ::: $SAVE_DIR/splits-${i}/*

    # recreates the splits from the sampled configurations
    echo "Recreate the splits from the sampled configurations"
    time python scripts/consolidate.py $SAVE_DIR/samples-${i} -o $SAVE_DIR/splits-${j}
done

# Your results should be in $SAVE_DIR/samples-6
