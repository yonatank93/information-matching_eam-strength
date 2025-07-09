The scripts and data here are to generate the candidate configurations.



## How to use

Running the following commands will extract the local atom environments from
the supercell and writes .dump files for each atom environment.

```
python extract_candidate_environments.py -t candidates_2000 -i quests_compressed_envs/eam-index-compressed.npz
mpirun -n 64 lmp -i extract_candidate_environments.in
python generate_xyz.py -t candidates_2000
python generate_candidates_Ta_from_W.py -s candidates_2000 -t candidates_2000_Ta_from_W
```



## Main content

* *final.W0.dump* - the same file as /p/vast1/iap/thrust3/FIM-matching/EAM/Dumps/final.W0.dump
* *final.W0.mod.in* - lammps script to add absolute xyz coordinates, which are used in
  the local environment extraction.

* *atom_shift.py* - a script to get information about shifted atom envionment positions.
  Note that to avoid dealing with PBC, all central atoms are shifted to the middle of
  simulation box.

* *extract_candidate_environments.py* - this script does NOT do the extraction of local
  atomic environments centered at specified central atoms. However, this script will
  prepare all the information and necessary LAMMPS files to run the extraction. The
  extracted environments are exported as .dump files.

* *generate_xyz.py* - this script converts the .dump files of the extracted local atomic
  environments to .xyz (extended xyz format) files that can be used with the
  FIMTrainingSetScore module in the Orchestrator.

* *quests_compressed_envs* - same as /p/vast1/iap/thrust3/FIM-matching/CompressedEnvs/EAM

* *dataset-compression* - Daniel's code to extract compressed environments from the
  supercell. I modified it so that I can extract how many ever environments, and I can
  also find the index of these compressed environments.

* *generate_candidates_Ta_from_W.py* - this script takes the generated candidate
  configurations for W and rescale the lattice spacing to represent Ta candodate
  configurations.
