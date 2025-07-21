"""This script is to automatically run the FIM-matching calculation over
multiple iterations. This script is basically responsible for submitting jobs
or calculations and checking if the calculations are done.
"""

from pathlib import Path
import argparse
import yaml
import pickle
import subprocess
import re
import pprint

import numpy as np
import matplotlib.pyplot as plt
from utils import (extract_strength_simulation_data, block_until_completed,
                   run_python_command, sync_calc)

# Command line arguments
parser = argparse.ArgumentParser(
    "Automatically run FIM-matching active learning loop")
parser.add_argument("-s",
                    "--settings-file",
                    type=str,
                    required=True,
                    help="Settings file",
                    dest="settings_file")
parser.add_argument("--predict",
                    action="store_true",
                    help="An option to run KIMRun target property predictions "
                    "at the end of the active learning loop",
                    dest="predict")
parser.add_argument("--plot",
                    action="store_true",
                    help="An option for showing relevant plots "
                    "at the end of the active learning loop",
                    dest="plot")
args = parser.parse_args()

###############################################################################
# CALCULATION SETTINGS
# ====================

# Read calculation settings from a yaml file
with open(args.settings_file) as f:
    settings = yaml.safe_load(f)

# General settings
# ----------------
general_settings = settings["general"]
# Initial parameters
INIT_PARAMS = Path(general_settings["init_params"])
# This is where all the results are written to
RESULTS_DIR = Path(general_settings["results_dir"])
# Candidate configurations directory
CANDIDATES_DIR = Path(general_settings["candidates_dir"])
# The suffix of the folder for each iteration:
# iteration_<iter>_<suffix>
SUFFIX = general_settings["suffix"]
# Number of iterations
NITERS = general_settings["niters"]
NBURNINS = general_settings[
    "nburnins"]  # How many iterations used as burn-in period
# Regularization strength that we pick from the training results for the next
# iterations (float, int, or 'inf')
LAMBDA_REG = general_settings["lambda_reg"]

# FIM candidates settings
# -----------------------
fim_candidates_settings = settings["fim_candidates"]
# Split forces FIMs byy atoms
FIM_PER_ATOM = fim_candidates_settings["per_atom"]

# FIM target settings
# -------------------
fim_target_settings = settings["fim_target"]
# Strength derivative file
STRENGTH_DERIV = fim_target_settings["strength_deriv"]
# Additional KIMRun properties input files
ADDITIONAL_PROPS = fim_target_settings["additional_props"]
# Target covariance of the target properties
COVARIANCE = fim_target_settings["target_cov"]

# FIM-matching settings
# ---------------------
# Other flags are covered by general settings, w0 flag is handled by NITERS
# and NBURNINS settings.

# Submit training jobs settings
# -----------------------------
submit_training_settings = settings["submit_training"]
# Parameter file for regularization
PARAMS_REG = submit_training_settings["params_reg"]
# Training directory relative to target directory in each iteration
REL_TRAINING_DIR = submit_training_settings["rel_training_dir"]
# Number of calculations per job
NCALCS = submit_training_settings["ncalcs"]

# Postprocess training settings
# -----------------------------
# None - All flags are covered by the other settings

# Postprocess the settings
# Create directories
DIRECTORIES = [RESULTS_DIR]
for DIR in DIRECTORIES:
    DIR.mkdir(parents=True, exist_ok=True)

# Get the machine name
result = subprocess.run("printenv HOSTNAME | awk -F[[:digit:]] '{print $1}'",
                        shell=True,
                        capture_output=True,
                        text=True)
MACHINE = result.stdout.strip().capitalize()

# Regularization string
if isinstance(LAMBDA_REG, int):
    reg_str = f"reg{LAMBDA_REG}sig"
elif isinstance(LAMBDA_REG, float):
    reg_str = f"reg{LAMBDA_REG:0.1e}sig"
elif isinstance(LAMBDA_REG, str):
    assert LAMBDA_REG.lower() == "inf"
    reg_str = "noreg"

###############################################################################
# CALCULATION SEQUENCE
# ====================
"""These are the calculation sequence for each iteration:
* FIM candidates
* FIM target
* FIM-matching (and check for convergence)
* Training -- submit jobs
* Training -- postprocess to find the best result over multiple starting points
"""

for ii in range(NITERS):
    ITER = ii + 1  # Iteration index is base 1
    TARGET_DIR = RESULTS_DIR / f"iteration_{ITER}_{SUFFIX}"
    print("FIM-matching active learning loop for iteration", ITER)

    # FIM CANDIDATES
    # --------------
    print("Compute FIMs of candidate configurations")
    # Parameter to evaluate the FIM
    if ii == 0:
        # First iteration
        params_eval = INIT_PARAMS
    else:
        # Optimal parameters from the previous training results
        PREV_TARGET_DIR = RESULTS_DIR / f"iteration_{ii}_{SUFFIX}"
        PREV_TRAINING_DIR = PREV_TARGET_DIR / REL_TRAINING_DIR
        # Load the training results from the previous iteration
        with open(PREV_TRAINING_DIR / "optimal_training_results.pkl",
                  "rb") as f:
            prev_training_results = pickle.load(f)
        # From this training results, we extract the index of starting point
        # that gives the best training result.
        starting_point_idx = prev_training_results[LAMBDA_REG][
            "starting_point_idx"]
        # Finally, we can get the parameter to evaluate for this iteration
        params_eval = (PREV_TRAINING_DIR / reg_str /
                       f"{starting_point_idx:03d}" / "Wopt_20values.txt")

    # This calculation involves submitting job, and there is no point of
    # queueing for the job if the calculation is already completed
    check_file = TARGET_DIR / "FIM_candidates" / "fim_candidates_list.txt"
    if not check_file.exists():
        # Prepare the job script
        python_command = "".join([
            "fim_candidates.py ",
            f"-p {params_eval} ",
            f"-c {CANDIDATES_DIR} ",
            f"-t {TARGET_DIR}",
        ])
        if FIM_PER_ATOM:
            # Add a flag to split the candidate forces FIM by atom
            python_command += " --per-atom"

        # # Submit job
        # job_script = "\n".join([
        #     "#!/bin/bash",
        #     "#SBATCH --nodes=1",
        #     "#SBATCH --exclusive",
        #     "#SBATCH --mem=0",
        #     "#SBATCH --time=06:00:00",
        #     f"#SBATCH --job-name=fim_candidates_{ITER}",
        #     "#SBATCH --account=iap",
        #     f"#SBATCH --output={str(TARGET_DIR)}/%j-fim_candidates.out",
        #     f"python {python_command}",
        # ])
        # process = subprocess.run(['sbatch'],
        #                          input=job_script,
        #                          text=True,
        #                          capture_output=True)
        # # Block until completed
        # try:
        #     # Extract job ID
        #     job_id_match = re.search(r"Submitted batch job (\d+)",
        #                              process.stdout)
        #     if job_id_match:
        #         job_id = int(job_id_match.group(1))
        #         print(f"Submitted batch job {job_id}")
        #     else:
        #         print("Failed to extract job ID")
        #     # Block calculation until the job is completed
        #     block_until_completed([job_id])
        # except NameError as e:
        #     print(process.stderr)
        #     raise e

        # Run FIM calculation on login node
        run_python_command(python_command)
        # Synchronization - make sure that the files are ready for the next step
        sync_calc(check_file)

    # FIM TARGET
    # ----------
    print("Compute FIM of target properties")
    # fim_target.py calculation doesn't need to be submitted as a job. But, we
    # will still skip running the calculation if the calculation has been done
    # previously, to speed up the process.
    check_file = TARGET_DIR / "FIM_target" / "fim_target.npy"
    if not check_file.exists():
        python_command = " ".join([
            "fim_target.py",
            f"-p {params_eval}",
            f"-C {COVARIANCE}",
            f"-t {TARGET_DIR}",
        ])
        # Add strength derivative
        if STRENGTH_DERIV is not None:
            python_command += f" -d {STRENGTH_DERIV}"
        # Add other KIMRun target properties
        if ADDITIONAL_PROPS is not None:
            additional_props_str = " ".join(
                [str(path) for path in ADDITIONAL_PROPS])
            python_command += f" -a {additional_props_str}"
        # Run the FIM calculation
        run_python_command(python_command)
        # Synchronization - make sure that the files are ready for the next step
        sync_calc(check_file)

    # FIM-MATCHING
    # ------------
    print("FIM-matching calculation")
    # fim_matching.py calculation doesn't need to be submitted as a job,. But,
    # we will still skip the calculation if it is already completed.
    check_file = TARGET_DIR / "optimal_weights_without_zeros.txt"
    if not check_file.exists():
        python_command = " ".join(["fim_matching.py", f"-t {TARGET_DIR}"])
        if ii >= NBURNINS and ii > 0:
            # Pass the burn-in period, we combine the optimal weights with the
            # ones from the previous iteration.
            prev_weights = PREV_TARGET_DIR / "optimal_weights_with_zeros.txt"
            python_command += f" -w0 {prev_weights}"

        # # Submit job
        # job_script = "\n".join([
        #     "#!/bin/bash",
        #     "#SBATCH --nodes=1",
        #     "#SBATCH --exclusive",
        #     "#SBATCH --mem=0",
        #     "#SBATCH --time=12:00:00",
        #     f"#SBATCH --job-name=fim_matching_{ITER}",
        #     "#SBATCH --account=iap",
        #     f"#SBATCH --output={str(TARGET_DIR)}/%j-fim_matching.out",
        #     f"python {python_command}",
        # ])
        # process = subprocess.run(['sbatch'],
        #                          input=job_script,
        #                          text=True,
        #                          capture_output=True)
        # # Block until completed
        # try:
        #     # Extract job ID
        #     job_id_match = re.search(r"Submitted batch job (\d+)",
        #                              process.stdout)
        #     if job_id_match:
        #         job_id = int(job_id_match.group(1))
        #         print(f"Submitted batch job {job_id}")
        #     else:
        #         print("Failed to extract job ID")
        #     # Block calculation until the job is completed
        #     block_until_completed([job_id])
        # except NameError as e:
        #     print(process.stderr)
        #     raise e

        # Run FIM-matching calculation on login node
        run_python_command(python_command)
        # Synchronization - make sure that the files are ready for the next step
        sync_calc(check_file)

    # Check for convergence
    if ii > 0:
        # The better way to check for convergence is to compare the arrays
        # of optimal weights WITH zeros.
        opt_weights_file = "optimal_weights_with_zeros.txt"
        # Load weights
        cur_opt_weights = np.loadtxt(TARGET_DIR / opt_weights_file)[:, 1]
        PREV_TARGET_DIR = RESULTS_DIR / f"iteration_{ii}_{SUFFIX}"
        prev_opt_weights = np.loadtxt(PREV_TARGET_DIR / opt_weights_file)[:, 1]
        converged = np.allclose(cur_opt_weights,
                                prev_opt_weights,
                                rtol=1e-4,
                                atol=1e-4)
        if converged:
            print("#" * 70)
            print("# FIM-matching active learning loop has "
                  "successfully converged!")
            print("#" * 70)
            print()
            NITERS = ITER - 1

            # Print the optimal configurations
            # Load candidate identifiers
            fim_path_list = np.loadtxt(TARGET_DIR / "FIM_candidates" /
                                       "fim_candidates_list.txt",
                                       dtype=str)
            # Load the optimal weights with index and sigma
            opt_weights = np.loadtxt(check_file)
            # Print report
            print("Number of environments we need:", len(opt_weights))
            print("Optimal environment identifiers and their weights:")
            print("Identifier \t Weight \t\t Sigma")
            idx_sort = np.argsort(opt_weights[:, 0])
            for jj in idx_sort:
                idx = int(opt_weights[jj, 0])
                print(f"{idx} \t {opt_weights[jj, 1]} "
                      f"\t {opt_weights[jj, 2]}")

            print()
            print("Descriptions of these optimal environments:")
            for jj in idx_sort:
                idx = int(opt_weights[jj, 0])
                print(f"{idx} \t {fim_path_list[idx]}")

            break

    # SUBMIT TRAINING JOBS
    # --------------------
    print("Submit training jobs")
    # For this step, it involves submitting Slurm jobs, and it may take a while
    # to complete. So, if this step is already completed, we can just skip it.
    TRAINING_DIR = TARGET_DIR / REL_TRAINING_DIR
    check_file = TRAINING_DIR / "optimal_training_results.pkl"
    if not check_file.exists():
        python_command = " ".join([
            "training_submit_jobs.py",
            f"-p0 {INIT_PARAMS}",
            f"-pr {PARAMS_REG}",
            f"-f {TARGET_DIR}",
            f"-c {CANDIDATES_DIR}",
            f"-t {TRAINING_DIR}",
            f"-p {MACHINE}",
            f"-n {NCALCS} -s",
        ])
        # Submit jobs
        run_python_command(python_command)
        # Block calculation until all training jobs are completed
        jobid_file = TRAINING_DIR / "job_ids.txt"
        sync_calc(jobid_file)
        job_ids = np.loadtxt(jobid_file, dtype=int)
        block_until_completed(job_ids)

    # POSTPROCESS TRAINING RESULTS
    # ----------------------------
    print("Postprocess training results")
    # This step is also cheap and doesn't need job submission, but we will
    # still skip it if it is already completed.
    check_file = TRAINING_DIR / "optimal_training_results.pkl"
    if not check_file.exists():
        python_command = " ".join([
            "training_postprocess.py",
            f"-p0 {INIT_PARAMS}",
            f"-r {TRAINING_DIR}",
            f"-f {TARGET_DIR}",
        ])
        run_python_command(python_command)
        # Print some training results
        sync_calc(check_file)
        with open(check_file, "rb") as f:
            opt = pickle.load(f)
        pprint.pprint(opt[LAMBDA_REG])
    print()

###############################################################################
# (OPTIONAL) COMPUTE KIMRUN TARGET PROPERTIES
# ===========================================

if args.predict:
    # Compute KIMRun
    print("Compute KIMRun target properties using the last results")
    # Check if the predictions are already done. Only run the calculations if
    # there is any missing property predictions.
    property_prediction_files = [
        TARGET_DIR / Path(prop).with_suffix(".npy").name
        for prop in ADDITIONAL_PROPS
    ]
    if not all([ff.exists() for ff in property_prediction_files]):
        # Get the parameters to evaluate -- Since we don't run training if the
        # loop is converged, we need to retrieve it from the previous iteration.
        PREV_TARGET_DIR = RESULTS_DIR / f"iteration_{NITERS}_{SUFFIX}"
        PREV_TRAINING_DIR = PREV_TARGET_DIR / REL_TRAINING_DIR
        with open(PREV_TRAINING_DIR / "optimal_training_results.pkl",
                  "rb") as f:
            opt_training_results = pickle.load(f)[LAMBDA_REG]
        params_file = (PREV_TRAINING_DIR / reg_str /
                       f"{opt_training_results['starting_point_idx']:03d}" /
                       "Wopt.txt")
        # Run
        python_command = " ".join([
            "evaluate_kimrun.py",
            f"-p0 {INIT_PARAMS}",
            f"-p {params_file}",  # Parameter file to evaluate
            f"-f {' '.join(ADDITIONAL_PROPS)}",  # List of KIMRun input files
            f"-t {TARGET_DIR}"
        ])
        run_python_command(python_command)

    # Prepare for propagating the perdictions uncertainties
    # Parameter uncertaitny
    fim_opt = np.load(TARGET_DIR / "fim_environments_optimal.npy")
    cov_params = np.linalg.pinv(fim_opt)
    # Prediction Jacobian
    with open(TARGET_DIR / "FIM_target" / "jacobian_other_properties.pkl",
              "rb") as f:
        jac_qois = pickle.load(f)

    # Print the property predictions
    for ff in property_prediction_files:
        key = ff.with_suffix("").name
        print("Predictions for", key)
        # Mean
        prop_values = np.load(ff)
        print("Mean:", prop_values)
        # Uncertainty
        jac = jac_qois[key]
        cov_qois = jac @ cov_params @ jac.T
        print("Stdev:", np.sqrt(np.diag(cov_qois)))
        print()

    # Plastic strength prediction, if available
    deform_file = TARGET_DIR / "out.deform"
    if deform_file.exists():
        deform_data = extract_strength_simulation_data(deform_file, "all")
        # Stress data is on column index 5
        strength_data = deform_data[:, 5]
        # Mean and simulation stdev
        strength_mean_bar = np.mean(strength_data[-1500:])
        strength_std_bar = np.std(strength_data[-1500:])
        # Convert unit to GPa
        strength_mean_gpa = strength_mean_bar / 1e4
        strength_std_gpa = strength_std_bar / 1e4
        # Print
        print("Predictions for plastic strength")
        print("Simulation mean:", strength_mean_gpa)
        print("Simulation stdev:", strength_std_gpa)
        # Propagated uncertainty
        jac_strength_file = TARGET_DIR / "FIM_target" / "jacobian_strength.npy"
        if jac_strength_file.exists():
            jac_strength = np.load(jac_strength_file)
            cov_strength = jac_strength @ cov_params @ jac_strength.T
            print("Propagated stdev:", np.sqrt(cov_strength[0, 0]))
        print()

###############################################################################
# (OPTIONAL) PLOT RESULTS
# =======================

if args.plot:
    # Plot the eigenvalues of the target FIM
    print("Plot target FIM eigenvalues evolution")
    for ii in range(NITERS):
        ITER = ii + 1
        TARGET_DIR = RESULTS_DIR / f"iteration_{ITER}_{SUFFIX}"
        fim = np.load(TARGET_DIR / "FIM_target" / "fim_target.npy")
        lamb = np.linalg.eigvalsh(fim)
        if ii == 0:
            eigvalsh = lamb
        else:
            eigvalsh = np.vstack((eigvalsh, lamb))

    plt.figure()
    plt.title("Eigenvalues of the target FIM")
    # Plot the eigenvalues for each iteration, relative to the first iteration
    for pidx, lamb in enumerate(eigvalsh.T):
        if np.mean(np.abs(eigvalsh[:, pidx]) < 1e-8):
            # Include the sloppy eigenvalues, but make them fading.
            alpha = 0.25
        else:
            alpha = 1
        plt.plot(np.abs(lamb / eigvalsh[0, pidx]), alpha=alpha, label=pidx)
    plt.xticks(range(NITERS), np.arange(NITERS) + 1)
    plt.xlabel("Iteration")
    plt.ylabel("Scaled eigenvalue")
    plt.legend(title="Param idx\n(0 $\\rightarrow$ sloppy)",
               bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Plot the eigenvalues of the optimal candidates FIM
    print("Plot optimal data FIM eigenvalues evolution")
    for ii in range(NITERS):
        ITER = ii + 1
        TARGET_DIR = RESULTS_DIR / f"iteration_{ITER}_{SUFFIX}"
        fim = np.load(TARGET_DIR / "fim_environments_optimal.npy")
        lamb = np.linalg.eigvalsh(fim)
        if ii == 0:
            eigvalsh = lamb
        else:
            eigvalsh = np.vstack((eigvalsh, lamb))

    plt.figure()
    plt.title("Eigenvalues of the optimal environments FIM")
    # Plot the eigenvalues for each iteration, relative to the first iteration
    for pidx, lamb in enumerate(eigvalsh.T):
        if np.mean(np.abs(eigvalsh[:, pidx]) < 1e-8):
            # Include the sloppy eigenvalues, but make them fading.
            alpha = 0.25
        else:
            alpha = 1
        plt.plot(np.abs(lamb / lamb[0]), alpha=alpha, label=pidx)
    plt.xticks(range(NITERS), np.arange(NITERS) + 1)
    plt.xlabel("Iteration")
    plt.ylabel("Scaled eigenvalue")
    plt.legend(title="Eigenvalue idx\n(0 $\\rightarrow$ sloppy)",
               bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Plot optimal parameters
    print("Plot the optimal parameters")
    for ii in range(NITERS):
        ITER = ii + 1
        TARGET_DIR = RESULTS_DIR / f"iteration_{ITER}_{SUFFIX}"
        TRAINING_DIR = TARGET_DIR / REL_TRAINING_DIR
        # Load
        with open(TRAINING_DIR / "optimal_training_results.pkl", "rb") as f:
            opt_results = pickle.load(f)
        log_opt_params = opt_results[LAMBDA_REG]["x"]
        abs_opt_params = np.exp(log_opt_params)
        if ii == 0:
            opt_params_list = abs_opt_params
        else:
            opt_params_list = np.vstack((opt_params_list, abs_opt_params))

    plt.figure()
    plt.title("Optimal parameter values")
    for pidx, params in enumerate(opt_params_list.T):
        plt.plot(np.abs(params / params[0]), label=pidx)
    plt.xticks(range(NITERS), np.arange(NITERS) + 1)
    plt.xlabel("Iteration")
    plt.ylabel("Scaled parameter value")
    plt.legend(title="Param idx", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.show()
