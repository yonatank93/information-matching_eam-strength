This folder contains additional data that are used in FIM-matching calculation.

**Notes:** Some folders (covariance and stress_simulation) need to be downloaded
separately due to their sizes.



## Content

* *vasily_strength_deriv_iteration_1.json* - Derivative data for strength
  property that Vasily sent.
* * *.eam.alloy* - some default options of .eam.alloy parameter files
* *original_parameters_*.txt* - Original EAM parameter values in 20-value or 7-value
  format.
* *translate.* * - Fortran script and its compiled executable by Zhou to convert the
  parameters of EAM that they use into .eam.alloy file.
* *lc_resource_info.json* - A JSON file with information about resoureces in LC, like
  number of cpu and memory that I have access to.
* *covariance* - Containing script and results for the target covariance of the target
  QoIs.
* *stress_simulation* - Containing files and scripts to visualize the stress simulation
  results.
		
