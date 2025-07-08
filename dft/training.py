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

from information_matching.transform import LogTransform, ComposedTransform
from transform import EAMTransform
from utils import ENERGY_KEY, FORCES_KEY
from properties.compute_eam import compute

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
# Training atomic configurations
config_dir = Path(args.config_dir)

###############################################################################
# DATA
# ----
print("Reading data: configurations, ground truth, weights")
# We need this list of configuration paths to instantiate the configurations
configs_path_list = np.loadtxt(config_dir / "config_list.txt", dtype=str)
# Optimal environments and weights
optimal_weights = np.loadtxt(FIMMATCH_DIR /
                             "optimal_weights_without_zeros.txt",
                             skiprows=1)
idx_identifier = [int(val) for val in optimal_weights[:, 0]]

# The identifier stored in the optimal weight file only gives the index. We
# need to refer to the list of FIM candidates path to get the actual
# identifier.
fim_cand_path_list = np.loadtxt(FIMMATCH_DIR / "FIM_candidates" /
                                "fim_candidates_list.txt",
                                dtype=str)
# This list actually contain the relative path to the candidate FIMs. But, if
# we remove the extension, that can be use as the actual identifier.
opt_fim_path_list = fim_cand_path_list[idx_identifier]

# We'll create 2 dictionaries (1 for energy and 1 for forces). In each
# dictionary, we'll store all the necessary information for training, including
# the ASE atoms object to compute the properties, the ground truth values, and
# the weights. For forces, we also need to store the index because we might not
# need to use predictions of all atoms within.
configs_energy = {}
configs_forces = {}
for ii, ff in enumerate(opt_fim_path_list):  # Iterate over optimal configs
    # Remove .npy extension to get the identifier
    ident = Path(ff).with_suffix("").name
    weight = optimal_weights[ii, 1]  # Optimal weight for this configuration

    if ENERGY_KEY in ff:
        # Instantiate Atoms object for this configuration
        atoms_path = [path for path in configs_path_list if ident in path][0]
        atoms = read(config_dir / atoms_path, format="extxyz")
        # Extract the ground truth - Energy per atom
        gt = atoms.info[ENERGY_KEY] / atoms.get_global_number_of_atoms()
        # Update the energy configurations dictionary
        configs_energy.update(
            {ident: {
                "atoms": atoms,
                "ground_truth": gt,
                "weight": weight
            }})
    elif FORCES_KEY in ff:
        # For this case, we need to have an extra step. Sometimes, one
        # configuration is treated as 1 force configuration, but sometimes the
        # configuration is splitted by atoms. For the latter case, the
        # identifier has an extra atom index at the end.

        # Atom idx is integer before the extension
        config_name_split = ident.split("_")
        try:
            # If the configuration is only for 1 atom, then the last element of
            # the identifier is an int of atom idx.
            idx = int(config_name_split[-1])
            config_name = "_".join(config_name_split[:-1])  # Remove atom idx
        except ValueError:
            # If the last elemennt of atom idx is not an int, then the
            # configuration is treated as a single configuration.
            idx = None
            config_name = ident

        # Instantiate Atoms object for this configuration
        atoms_path = [
            path for path in configs_path_list if config_name in path
        ][0]
        atoms = read(config_dir / atoms_path, format="extxyz")
        natoms = atoms.get_global_number_of_atoms()
        # Ground truth - For either case, we'll store the forces of the all
        # atoms. We will just index it later.
        gt = atoms.arrays[FORCES_KEY]

        # This is the case we consider a single force configuration as a whole.
        # For this case, we still need the idx information, but we should set
        # the idx to just include all atoms.
        if idx is None:
            idx = np.arange(natoms)

        # The following if-else should work for both cases.
        # If the identifier (config_name) is not in the forces dict yet, we
        # should add it. Otherwise, we can just update the idx and weight. The
        # latter should only apply when each atom is treated separately.
        if config_name not in configs_forces:
            idx = list(np.atleast_1d(idx))  # Idx is a 1d list
            # The weights should be defined for all atoms. Those not selected
            # by FIM-matching will have weight of 0.0
            weights = np.zeros(natoms)
            weights[idx] = weight  # Assign non-zero weights

            # Update
            configs_forces.update({
                config_name: {
                    "atoms": atoms,
                    "ground_truth": gt,
                    "idx": idx,
                    "weight": weights
                }
            })
        else:
            configs_forces[config_name]["idx"].append(idx)
            configs_forces[config_name]["weight"][idx] = weight

# Print some information
print("Energy configurations:")
for ident in configs_energy:
    print("*", ident)
print("Forces configurations:")
for ident in configs_forces:
    print("*", ident)
print()

###############################################################################
# DEFINE RESIDUAL FUNCTION
# ------------------------
print("Define predictions and residual functions...")
TMP_DIR = target_dir / "tmp"  # Place to store temporary files
TMP_DIR.mkdir(exist_ok=True)

# Prepare the potential
# This parameter transformation converts the 7-value format into tabulated
# values .eam.alloy format
eam_transform = EAMTransform(str(params_init_file), str(TMP_DIR),
                             str(DATA_DIR / "translate.x"), "Ta")
eam_param_file = str(eam_transform.TMP_DIR / eam_transform.eam_alloy_filename)
log_transform = LogTransform(params_sign)
transform = ComposedTransform([eam_transform, log_transform])
print("Parameter transformation:")
print(json.dumps(transform.jsonable_kwargs, indent=4))

# Parameter regularization center (transformed into log scale)
reg_center = np.log(np.abs(np.loadtxt(args.params_reg_file)))


# Energy residual --- For all configurations combined
def energy_residual(params):
    if isinstance(params, str):
        # Parameter file is given
        potential = params
    else:
        # Inverse parameter transformation. But in the end, we just need the
        # .eam.alloy file that the transformation class writes
        _ = transform.inverse_transform(params)
        potential = eam_param_file

    # Compute residual vector
    residual = []
    for vals in configs_energy.values():
        # Predictions
        atoms = vals["atoms"]
        preds = compute(atoms,
                        potential,
                        compute_energy=True,
                        compute_forces=False)["energy"]
        # Error: preds - gt
        gt = vals["ground_truth"]
        error = preds - gt
        # Residual: sqrt(weight) * error
        weight = vals["weight"]
        res = np.sqrt(weight) * error
        residual = np.append(residual, res)

    return residual


# Forces residual --- For all configurations combined
def forces_residual(params):
    if isinstance(params, str):
        # Parameter file is given
        potential = params
    else:
        # Inverse parameter transformation. But in the end, we just need the
        # .eam.alloy file that the transformation class writes
        _ = transform.inverse_transform(params)
        potential = eam_param_file

    # Compute residual vector
    residual = []
    for vals in configs_forces.values():
        # Note that we have the atom idx information to extract only the atoms
        # that we need
        idx = vals["idx"]
        # Predictions
        atoms = vals["atoms"]
        preds = compute(atoms,
                        potential,
                        compute_energy=False,
                        compute_forces=True)["forces"][idx]
        # Error: preds - gt
        gt = vals["ground_truth"][idx]
        error = preds - gt
        # Residual: sqrt(weight) * error
        weight = vals["weight"][idx].reshape((-1, 1))
        res = np.sqrt(weight) * error
        residual = np.append(residual, res.flatten())

    return residual


# Regularization term
def regularization_residual(params, reg_scale=0.0):
    if reg_scale == 0.0:
        return np.empty(0)
    else:
        return np.sqrt(reg_scale) * (params - reg_center)


# Combined residual
def residuals(params, T=1.0, reg_scale=0.0):
    # Inverse parameter transformation. But in the end, we just need the
    # .eam.alloy file that the transformation class writes
    _ = transform.inverse_transform(params)

    sqrtT = np.sqrt(T)
    res_energy = energy_residual(eam_param_file) / sqrtT
    res_forces = forces_residual(eam_param_file) / sqrtT
    res_reg = regularization_residual(params, reg_scale)

    return np.concatenate((res_energy, res_forces, res_reg))


# With this residual function, there are some other informations we need to
# compute. In particular, all the scaling factors: T and reg_scale.
print("Compute scaling factors in the cost function...")
# Temperature - 2C0 / N
T = 1.0  # np.linalg.norm(residuals(log_params_init))**2 / nparams
print("T:", T)
# Regularization scale - 1 / (lambda * sigma) **2
opt_fim = np.load(FIMMATCH_DIR / "fim_environments_optimal.npy")
var_inferred = np.diag(np.linalg.pinv(opt_fim))
sigma2 = max(var_inferred)

lamb = args.sigma_scale
if lamb == 0:
    reg_scale = 0
else:
    reg_scale = 1 / (lamb**2 * sigma2)
print("Regularization scale (lambda, sigma):", lamb, np.sqrt(sigma2))

###########################################################################
# MODEL TRAINING
# --------------
print("Train EAM...")

opt_file = target_dir / "training_optimal_raw.pkl"
if opt_file.exists():
    # Training has been done, load the training results
    with open(opt_file, "rb") as f:
        opt_dict = pickle.load(f)
else:
    opt = least_squares(residuals,
                        log_params_init,
                        method="lm",
                        kwargs={
                            "T": T,
                            "reg_scale": reg_scale
                        })
    # Add regularization parameter to the optimization result
    opt_dict = dict(opt)
    opt_dict.update({
        "temperature": T,
        "regularization": {
            "lambda": lamb,
            "sigma": np.sqrt(sigma2),
            "fim_path": FIMMATCH_DIR / "fim_environments_optimal.npy"
        }
    })
    # Export the raw optimal results
    with open(opt_file, "wb") as f:
        pickle.dump(opt_dict, f, protocol=4)

# Get the optimal parameters
opt_x = opt_dict["x"]
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
