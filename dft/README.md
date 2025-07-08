# FIM-Matching for EAM potential v2

In short, the changes made in this version are so that we can utilize
functionalities in the Orchestrator as much as possible. With these changes, we
can expand the list of target properties to those supported by the Orchestrator,
including KIMRun.

**Disclaimer:**
This version depends on several on-going developments in the Orchestrator,
including the refactored Potential module and new FIM modules that depend on
those developments. Because of it, then there might be quite a lot of changes
that need to be done in the future.
Additionally, I haven't done an extensive testing for this version, so there
might be some parts that don't work properly. Hopefully this won't happen, but
fingers crossed!



## Preliminary notes

* The potential that we use is the EAM potential for W by Zhou et al.
* We only vary parameter 1, 6, 7, 8, 9, 19, and 20.
* The derivative is taken with respect to the log of the parameters to resolve
  the difference in the parameter scale.
* The derivative of the target property is done by Vasily.
* The candidate data consists of 2,000 environments that Daniel extracted using
  QUEST from /p/vast1/iap/thrust3/FIM-matching/EAM/Dumps/final.W0.dump
* The candidate FIMs are computed with the atomic force values.
* I run the FIM-matching calculation in Intel-based machine (Ruby, Dane, or
  Borax).



## Additional requirements

1. Orchestrator branch 84-fim-using-refactored-potential
2. Yonatan's information-matching package (https://github.com/yonatank93/information-matching)
3. ASE >= 3.23.0

Note: It is much easier to install requirement 2 and its dependencies in x86
      machine.


## Content

These are additional folders and script that are used indirectly.
* configs - contains data and scripts related to candidate configurations (or
  environments), including scripts to generate new sets of candidate
  environments.
* data - contains additional required data, including original potential,
  ground truth, KIM query predictions ensemble, etc.
* properties - is designed as a module that contains scripts atomic forces
  (soon to be deprecated). But most importantly, this folder includes JSON
  files that defines inputs for KIMRun.
* transform.py - this is a new module for transforming the 7 eam parameters
  that we want to tune to tabulated values that can be (indirectly) used to
  update the potential parameters in ASE KIM calculator.



## How to use

Here is an example how I use these scripts for FIM-matching calculation.

1. Compute FIMs of the target properties. An example command looks like:

   ```
   python fim_target.py \
       -p data/original_parameters_W0.txt \
       -t results/ \
       -a properties/bcc_W_cohesive-potential-energy-cubic-crystal.json \
          properties/bcc_W_elastic-constants-isothermal-cubic-crystal-npt.json
   ```

   In this command:
   * (flag `-p`) I want to evaluate the FIM evaluated at parameters stored in
     `data/original_parameters_W0.txt`. Note that this .txt file contains all
     20 parameter values as used in the paper by Zhou et. al. Internal function
     will index these parameters and extract only 7 values that we care.
   * (flag `-t`) I want to export the FIM calculation results to
     `results/`. Actually, the results will be exported in `results/FIM_target`,
     but the inner-most directory is automatically named.
   * (flag `-a`) I want to evaluate the FIMs of properties defined in
     cohesive potential energy cubic crystal and elastic constants isothermal
     cubic crystal KIM tests, which are run using KIMRun. See the input JSON
     file in `properties` folder for more detail.
   * (optional flag `-d`) If a flag `-d` is provided, the argument value should
     point to the strength derivative data. If it is not specified, then the
     strength property is excluded from the target properties.

   Note: Currently we need to hard-code the covariance within `fim_target.py`
   script.

2. Compute the FIMs of the candidate environments. An example command looks
   like:

   ```
   python fim_candidates.py \
       -p data/original_parameters_W0.txt \
       -c configs/candidates_2000/configs/ \
       -t results/
   ```

   In this command:
   * (flag `-p`) - same as `fim_target.py`.
   * (flag `-c`) This path should point to the folders containing a bunch of
     `config_*.xyz` and `config_*_mask.txt` files. The first set of files are
     the candidate configuration files, and the second set of files contain
     masking arrays to all atoms except the central atom for the correspoding
     configuration.
   * (flag `-t`) - same as `fim_target.py`, but the inner-most folder is names
     `FIM_candidates`

   Notes:
   * This calculation seems to run the longest. Currently, I requested to use
     30 parallel processes, but it seems that there are effectively only about
     10 can run simultaneously, perhaps due to I/O overhead. With this
     parallelization, it took me about 1 hour to complete the calculation in
     Borax.
   * This calculation writes (2*nparams + 1) * ncandidates intermediate files,
     which will be deleted at the end of the calculation. These intermediate
     files are the byproduct of the `EAMTransform` module inside `transform.py`.

3. Run FIM-matching calculation using information-matching package and
   Orchestrator's wrapper for the package. An example command to run this
   calculation looks like:

   ```
   python fim_matching.py -t results/
   ```

   assuming all FIM results are stored in `results/iteration_1`.

   The main results are stored in `results/optimal_weights_with_zeros.txt`
   for the weights of all environments (including those with zero weights) and
   `results/optimal_weights_without_zeros.txt` for only the nonzero weights.
   For the latter, the columns show the environment identifier, optimal
   weights, and the corresponding target DFT force accuracy, in this order.

   For the environment identifier, it is just a zero-based index that
   correspond to the `config_*.xyz` files inside the configurations directory
   (see the `-c` flag for `fim_candidates.py` script).


4. After obtaining the optimal weights through FIM-matching, we can proceed by
   training the EAM potential to fit the forces data of the optimal
   environments with the optimal weights. The script `training_new.py` can be
   used to run one training. An example command for using this script looks
   like:
   
   ```
   python training_new.py \
       -p0 data/original_parameters_W0.txt \
       -r data/ground_truth/SNAP_W/forces_snap.npy \
       -f results/ \
       -c configs/candidates_2000 \
       -t results/training \
       -s 1
   ```

   Note: `training_new.py` utilizes several functionalities for potential in
   Orchestrator, such as the `.set_params` and `.evaluate` methods.
   
   In this command:
   * (flag `-p0`) The training starts from `data/original_parameters_W0.txt` as
     the starting initial guess. Note that we again need to input all 20 EAM
     parameter values.
   * (flag `-r`) The reference ground truth data are stored in
     `data/ground_truth/SNAP_W/forces_snap.npy`, which contains a proxy ground
     truth forces for W calculated using SNAP potential.
   * (flag `-f`) The FIM-matching results (especially the
     `optimal_weights_without_zeros.txt` file) are stored in `results/` folder.
   * (flag `-c`) Similar to the flag for `fim_candidates.py`, but it should
     point to the parent directory, which also contain the information about
     the box center, central atom positions, etc. (Although, I'm not sure if
     these additional information are needed for the new version.)
   * (flag `-t`) Same as the other scripts, which point to the folder in which
     all result files are exported/written to.
   * (flag `-s`) Defines the scale of the regularization term. In general,
     larger value means weaker regularization, with an exception of 0, which
     implies no regularization.
   
   Alternatively, we can submit a bunch of training jobs with varying
   regularization scale and initial parameter guess. The script to do so is
   `training_submit_jobs.py`, and an example command to run the calculations
   looks like:
   
   ```
   python training_submit_jobs.py \
       -p0 data/original_parameters_W0.txt \
       -f results/ \
       -c configs/candidates_2000 \
       -r data/ground_truth/SNAP_Ta/forces_snap.npy
       -t results/training \
       -p Dane -n all -s
   ```
   
   Additional options for this script are option `-p` to specify x86-based
   partition---which currently only support Dane, Ruby, or Borax, option `-n`
   is for specifying how many concurrent training calculations to run per job,
   and option `-s` is an option to submit the training jobs.
   
5. Then, we can use `training_postprocess.py` to look at the training results.
   This script assumes that we did training using multiple settings, e.g.,
   after running training_submit_jobs.py. This scripts finds the best training
   results over different initial guesses for each regularization scale. An
   example command for using this script looks like:

   ```
   python training_postprocess.py \
       -p0 data/original_parameters_W0_7values.txt \
       -r results/training \
       -f results/
   ```

   where option `-p0` specifies the EAM parameters (7-value format) that will
   be shown on the plots, `-r` is for specifying the directory where the
   training results are stored, and `-f` specifies the path to folder where the
   FIM-matching results are stored. There is an extra option `-t` for copying
   the best results to different directory, for example to shared directory.