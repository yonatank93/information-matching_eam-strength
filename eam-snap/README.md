# FIM-Matching for EAM potential fitted to EAM and SNAP proxy ground truth
   
**Disclaimer:**
This version depends on the still-in-development Orchestrator Python package, and thus
there might be some changes that need to be done in the future.



## Preliminary notes

* The potential that we use is the EAM potential by Zhou et al.
* We only vary parameter 1, 6, 7, 8, 9, 19, and 20.
* The derivative is taken with respect to the log of the parameters to resolve the
  difference in the parameter scale.
* The derivative of the target property is done by Vasily Bulatov.
* The candidate data consists Ta configurations from Amit Samanta with DFT energy and 
  forces labels. The labels for energies have been subtracted by isolated atom energy.
* I run the FIM-matching calculation in Intel-based machine (Ruby, Dane, or Borax).



## Additional requirements

1. Orchestrator branch 84-fim-using-refactored-potential
2. Yonatan's information-matching package (https://github.com/yonatank93/information-matching)

Note: It is much easier to install requirement 2 and its dependencies in x86 machine.



## How to use

The main script is `automatic_fim_matching_run.py`, which takes a yaml file containing
the calculation settings as an argument. The specific settings used for our calculations
are given in `settings` folder. Example to run:

``` bash
python automatic_fim_matching_run.py -s settings/DFT_nocorr_energy_max-frobenius.yaml
```



## Content

Other scripts:
* *fim_candidates.py* - For computing the FIMs of candidate configurations.
* *fim_target.py* - For computing the FIM of the (intermediate) target QoIs.
* *fim_matching.py* - For matching the FIMs and compute the optimal weights.
* *training.py* - For running a single instance of training procedure, given the required
  data to construct the loss function and the initial parameter guess.
* *trainng_submit_jobs.py* - For iterating over several hyperparameters and initial
  guesses and submit multiple training instances.
* *training_postprocess.py* - For collecting all results from different training instances
  and obtain the best training results over different initial guesses.
* *transform.py* - Containing a parameter transformation class that converts the EAM
  parameterization we use (except the log-transform part) to the original EAM
  parameterization.
* *evaluate_kimrun.py* - For running KIMRun to evaluate the (intermediate) target
  properties.
* *optimal_dataset_analysis.py* - For visualizing the optimal data that
  information-matching select using PCA plot.
* *fim_matching_final_strength.py* - For the final information-matching analysis between
  the intermediate QoIs with plastic strength.
* *qois_ensemble.py* - Generate a prediction ensemble of the intermediate QoIs and
  strength from the potentials used to estimate the initial strength FIM.
  
Other folders:
* *settings* - Containing the yaml files for settings used in the study.
* *configs* - Containing the candidate configurations related scripts and data.
* *data* - Containing target covariance, original EAM parameters, ground truth data, and
  other additional data needed for the calculation.
  
**Note:** Large files in configs and data folders need to be downloaded separately.
