This folder contains additional data that are used in FIM-matching calculation, as well
as the ground truth data.



## Content

* *vasily_strength_deriv_iteration_1.json* - Derivative data for strength property that
  Vasily sent.
* * *.eam.alloy* - some default options of .eam.alloy parameter files
* *original_parameters_*.txt* - Original EAM parameter values in 20-value or 7-value
  format.
* *translate.* * - Fortran script and its compiled executable by Zhou to convert the
  parameters of EAM that they use into .eam.alloy file.
* *ground_truth* - this folder contains scripts to generate the proxy ground truth data.
* *covariance* - Containing script and results for the target covariance of the target
  QoIs.
* *lc_resource_info.json* - just a JSON file with information about resoureces in LC,
  like number of cpu and memory that I have access to.
* *ground_truth* - containing script and data to compute ground truth data using SNAP
  proxy, and DFT reference data that Amit provided.
* *potential* - this folder contains the EAM parameter files to generate the QoI ensemble.
* *stress_simulation* - Containing files and scripts to visualize the stress simulation
  results.
