"""Use this script to train the EAM potential after finding the optimal
environments and the corresponding weights.

Steps:
1. Read the optimal environments and weights from file. We use this information
   to return only elements that we need.

2. Read the ground truth forces data. Only store the elements that we need.

3. Define a function to compute the forces predictions:
   a. Given 7 tunable parameter values in log-scale, convert to linear scale.
      Export to tmp folder.
   b. Read the parameter file, write EAM potential file, put it to tmp folder.
      Use generate_eam_parameter_file.py.
   c. Use this EAM potential file, run LAMMPS forces calculation, and put the
      results into tmp folder. Use compute_forces.py.
   d. Read the exported file and return the forces value.

4. Define a residual function using this forces prediction function, ground
   truth data, and the optimal weights.

5. Optimize.

6. Export the optimal parameters, in 7-value, 20-value, and eam.alloy format.
"""

from pathlib import Path
import argparse
import subprocess
import pickle
import json

import numpy as np
from ase.io import read
from scipy.optimize import least_squares

from orchestrator.potential import potential_builder
from information_matching.transform import LogTransform, ComposedTransform
from transform import EAMTransform

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"
# Index location of the 7 parameters in the 20 elements list of parameters
idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]

###############################################################################
# READ COMMAND LINE ARGUMENTS
# ---------------------------
# Command line arguments
parser = argparse.ArgumentParser(
    "Train EAM potential using optimal environments")
parser.add_argument(
    "-p0",
    "--params_init",
    type=str,
    help="Path to the initial parameter file",
    default=DATA_DIR / "original_parameters_W0.txt",
    dest="params_init_file",
)
parser.add_argument(
    "-pr",
    "--params_regularization",
    type=str,
    help="Path to the parameter file for the center of regularization, "
    "7-value format",
    default=DATA_DIR / "original_parameters_W0_7values.txt",
    dest="params_reg_file",
)
parser.add_argument(
    "-r",
    "--reference_data",
    type=str,
    help="Path to the reference data numpy file",
    dest="reference_data_path",
)
parser.add_argument(
    "-c",
    "--config_dir",
    type=str,
    help="Directory where the configurations are stored",
    dest="config_dir",
)
parser.add_argument(
    "-f",
    "--fimmatch-dir",
    type=str,
    help="Location of fim-matching results, e.g., ./results/iteration_1",
    dest="fimmatch_dir",
)
parser.add_argument(
    "-t",
    "--target_dir",
    type=str,
    help="Path to the target directory to store the results",
    dest="target_dir",
)
parser.add_argument(
    "-s",
    "--sigma_scale",
    type=float,
    help="Scale of sigma in the regularization term - "
    "scale=0 is reserved to mean no regularization",
    dest="sigma_scale",
)
parser.add_argument("--validation",
                    action="store_true",
                    help="An option for evaluating the predictions "
                    "using all configuration for validation",
                    dest="validation")

args = parser.parse_args()
# Read the original parameter values as the baseline
params_init_file = Path(args.params_init_file)
params_init = np.loadtxt(params_init_file)[idx_7vals]
# We want to use log-scale
params_sign = np.sign(params_init)  # To preserve the sign later
log_params_init = np.log(np.abs(params_init))
nparams = len(log_params_init)  # Number of tunable parameters
# This is where the FIM-matching data needed are stored
FIMMATCH_DIR = Path(args.fimmatch_dir)
# This is where the results will be exported
target_dir = Path(args.target_dir)
target_dir.mkdir(exist_ok=True)

###############################################################################
# DATA
# ----
print("Reading data...")

# Optimal environments and weights
print("* Optimal environments and weights")
optimal_envs = np.loadtxt(FIMMATCH_DIR / "optimal_weights_without_zeros.txt",
                          skiprows=1)
identifier = [int(val) for val in optimal_envs[:, 0]]
if args.validation:
    idx_configs = range(2000)
else:
    idx_configs = identifier
weights = optimal_envs[:, 1]
# Training atomic configurations
config_dir = Path(args.config_dir)
configs = [
    read(config_dir / "configs" / f"config_{idx}.xyz") for idx in idx_configs
]
masks = [
    np.loadtxt(config_dir / "configs" / f"config_{idx}_mask.txt", dtype=int)
    for idx in idx_configs
]
nconfigs = len(configs)
nprocs = min([nconfigs, 20])  # Limit the number of parallel processes

# Ground truth data
print("* Ground truth data")
ground_truth_xyz = np.load(args.reference_data_path)[idx_configs]
ndata = len(ground_truth_xyz.flatten())  # Number of training data points

###############################################################################
# DEFINE FORCES PREDICTION FUNCTION
# ---------------------------------
print("Define forces prediction function...")
TMP_DIR = target_dir / "tmp"  # Place to store temporary files
TMP_DIR.mkdir(exist_ok=True)

# Prepare the potential
# This parameter transformation converts the 7-value format into tabulated
# values .eam.alloy format
eam_transform = EAMTransform(str(params_init_file), str(TMP_DIR),
                             str(DATA_DIR / "translate.x"))
log_transform = LogTransform(params_sign)
transform = ComposedTransform([eam_transform, log_transform])
print(json.dumps(transform.jsonable_kwargs, indent=4))

# Potential information dictionary
potential_dict = {
    "potential_type": "KIM",
    "potential_args": {
        "kim_id":
        "EAM_Dynamo_ZhouJohnsonWadley_2004_W__MO_524392058194_005",
        "kim_api":
        "/usr/gapps/iap/kim-storage/kim-api/quartz/bin/kim-api-collections-management",
        "species": [eam_transform.symb],
        "model_driver":
        "EAM_Dynamo__MD_120291908751_005",
        "param_files":
        [str(eam_transform.TMP_DIR / eam_transform.eam_alloy_filename)]
    }
}

# Build initial potential
potential = potential_builder.build(
    potential_type=potential_dict["potential_type"],
    potential_args=potential_dict["potential_args"])
potential.build_potential()


def forces_predictions(params):
    """Compute the forces predictions using EAM potential. The input parameters
    are in log-scale.
    """
    # To update the potential parameters, we need to convert the 7-value format
    # into a dictionary first
    params_eam = transform.inverse_transform(params)
    params_eam_dict = {
        "cutoff": [[0], [params_eam[0]]],
        "deltaR": [[0], [params_eam[1]]],
        "deltaRho": [[0], [params_eam[2]]],
        "embeddingData": [np.arange(2000), params_eam[3:2003]],
        "rPhiData": [np.arange(2000), params_eam[2003:4003]],
        "densityData": [np.arange(2000), params_eam[4003:]],
    }
    # Update the potential parameters
    potential.set_params(**params_eam_dict)

    # Iterate over the atoms and compute the forces predictions
    preds = []
    for ii, atom in enumerate(configs):
        _, forces_all, _ = potential.evaluate(atom)
        # Apply masking array and extract only the forces of the central atom
        idx_central = np.where(masks[ii])[0]
        preds.append(forces_all.flatten()[idx_central])
    return np.array(preds)


if not args.validation:
    ###########################################################################
    # DEFINE RESIDUAL FUNCTION
    # ------------------------
    print("Define residual function...")

    scale = args.sigma_scale
    # Scale down the weights uniformly --- this might help with convergence
    # during training. In the we use a regularization, this scaling step can be
    # used to balance the contribution from the likelihood and the prior.
    # Let's use the natural temperature as the scaling factor
    # Original parameters, transformed into log scale
    p0 = np.log(np.abs(np.loadtxt(args.params_reg_file)))
    preds0 = forces_predictions(p0)
    error0 = np.sum(weights.reshape((-1, 1)) * (ground_truth_xyz - preds0)**2)
    T = error0 / nparams
    print("T =", T)
    scaled_weights = weights / T

    # L2-norm regularization around the original parameters
    # The regularization is 0.5 gamma 1/sigma^2 ||p - p0||_2^2
    # Gamma --- I use this to add a scaling factor, for example since I will
    # divide the likelihood term with the number of data to help the optimizer,
    # I will need to do the same with the prior term.
    gamma = 1 / ndata
    # Sigma --- We want the standard deviation to be large enough so that it
    # won't realy affect the other stiff parameters.
    # First, estimate sigma from the FIM of the optimal environments
    opt_fim = np.load(FIMMATCH_DIR / "fim_environments_optimal.npy")
    var_inferred = np.diag(np.linalg.inv(opt_fim))
    # We can take the largest sigma. To make sure it is wide enough to not
    # affect the stiff parameters, I will use 2*sigma
    # Combine all regularization magnitude, ignore the factor 0.5
    if scale == 0:
        print("Turn off regularization")
        reg_lambda = 0
    else:
        reg_lambda = gamma / (scale**2 * max(var_inferred))

    def residual(params, reg=0):
        """Residual function that will be used in the least-squares
        optimization. This residual function is similar to RMSE residual, but
        with weighted sum.

        The additional argument `reg` set the strength of the L2-norm
        regularization term. The loss function with the regularization term is:
        L(p) = sum(||y-preds||_2^2) + reg sum(||p-p0||_2^2)
        """
        # Compute predictions
        predictions = forces_predictions(params)
        # Difference --- Still in N x 3 format
        difference = ground_truth_xyz - predictions
        # Multiply by weights
        scaled_difference = difference * np.sqrt(scaled_weights).reshape(
            (-1, 1))
        # Normalize by the number of data to mimic RMSE
        scaled_difference *= np.sqrt(gamma)
        if reg == 0:
            return scaled_difference.flatten()
        else:
            # Regularization
            params_diff = params - p0
            # Return the regularized residual
            return np.append(scaled_difference.flatten(),
                             np.sqrt(reg) * params_diff)

    ###########################################################################
    # MODEL TRAINING
    # --------------
    print("Train EAM...")

    opt_file = target_dir / "training_optimal_raw.pkl"
    if opt_file.exists():
        # Training has been done, load the training results
        with open(opt_file, "rb") as f:
            opt = pickle.load(f)
    else:
        opt = least_squares(residual,
                            log_params_init,
                            method="lm",
                            kwargs={"reg": reg_lambda})
        # Export the raw optimal results
        with open(opt_file, "wb") as f:
            pickle.dump(opt, f, protocol=4)

    # Get the optimal parameters
    opt_x = opt.x
    # Convert to linear scale
    opt_x_lin = params_sign * np.exp(opt_x)
    print("Optimal parameter values:", opt_x_lin)
    # Export as a txt file
    param_file = target_dir / "Wopt.txt"
    np.savetxt(param_file, opt_x_lin)
    # Export 20-values parameter format
    eam_transform = EAMTransform(str(params_init_file), str(target_dir),
                                 str(DATA_DIR / "translate.x"))
    params20 = eam_transform._convert_from_7values_to_20values(opt_x_lin)[:20]
    np.savetxt(target_dir / "Wopt_20values.txt", params20)
    # Generate EAM parameter file
    subprocess.run(
        "python generate_eam_parameter_file.py "
        f"-p {param_file} "
        f"-d {target_dir} "
        f"-b {DATA_DIR}",
        shell=True,
        check=True,
    )
