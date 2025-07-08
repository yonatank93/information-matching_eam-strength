"""Using FIMTrainScore to compute the FIM of candidate environmnents using EAM
potential.
"""

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from orchestrator.computer.score.fim import FIMPropertyScore
from information_matching.transform import LogTransform, ComposedTransform
from transform import EAMTransform

# Base directories
FILE_DIR = Path(__file__).parent.resolve()
DATA_DIR = FILE_DIR / "data"
PROP_DIR = FILE_DIR / "properties"
# Index location of the 7 parameters in the 20 elements list of parameters
idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]

# Command line arguments
parser = argparse.ArgumentParser("Compute the FIMs of the target properties")
# What arguments should we request: initial parameters, target folder where to
# place the results, atomic configurations path
parser.add_argument(
    "-p",
    "--params",
    type=str,
    help="Path to the parameter file to evaluate",
    default=DATA_DIR / "original_parameters_W0_7values.txt",
    dest="params_file",
)
parser.add_argument(
    "-d",
    "--strength-derivative",
    type=str,
    help="Path to JSON file for Vasily's strength derivative result",
    default=None,
    dest="strength_deriv_file")
parser.add_argument(
    "-a",
    "--additional-properties",
    nargs="+",
    help="Path to JSON files that define the additional target properties",
    dest="additional_properties")
parser.add_argument(
    "-C",
    "--target-covariance",
    help=
    "Path to .npy file for the target covariance of the target predictions",
    dest="target_covariance")
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to write the resulst",
    dest="target_dir",
)
parser.add_argument("-s",
                    "--show",
                    action="store_true",
                    help="An option for showing relevant plots")
args = parser.parse_args()

TARGET_DIR = Path(args.target_dir)
TARGET_DIR.mkdir(parents=True, exist_ok=True)
FIM_DIR = TARGET_DIR / "FIM_target"
FIM_DIR.mkdir(exist_ok=True)

###############################################################################
# Parameter transformation
# ------------------------
print("Parameter transformation")
# Load the initial parameters
p0_file = Path(args.params_file)
p0 = np.loadtxt(p0_file)
p0_7vals = p0[idx_7vals]
nparams = len(p0_7vals)

# The complete transformation is composition of EAMTransform first, the
# LogTransform
eam_transform = EAMTransform(p0, str(FIM_DIR), str(DATA_DIR / "translate.x"))
log_transform = LogTransform(np.sign(p0_7vals))
transform = ComposedTransform([eam_transform, log_transform])
print(json.dumps(transform.jsonable_kwargs, indent=4))

###############################################################################
# Potential
# ---------
print("Potential")
# Potential information dictionary - Even if we use Ta, but the target
# properties don't depend on the atomic mass. So, we can still use the potential
# for W, we just need to make sure to use the right parameters.
potential_dict = {
    "potential": {
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
    },
    "parameters_optimize": {
        "cutoff": [["default"]],
        "deltaR": [["default"]],
        "deltaRho": [["default"]],
        "embeddingData": [["default"]] * 2000,
        "rPhiData": [["default"]] * 2000,
        "densityData": [["default"]] * 2000,
    }
}
print(json.dumps(potential_dict["potential"], indent=4))

###############################################################################
# Plastic strength
# ----------------
strength_deriv_file = args.strength_deriv_file
if strength_deriv_file is None:
    # Even though we don't include strength property, but we should still
    # initialize jacobian matrix for strength to make it more uniform with the
    # case where we include strength property.
    jac_strength = np.empty((0, nparams))
else:
    print("Load the Jacobian for the strength from Vasily's data")
    # Load Vasily's results as a dictionary, which is stored as a JSON file
    # Format: The key index the parameters (base 1) and the array elements are:
    # theta1, theta2, dQ1, dQ2, alpha
    # theta are the parameters theta_i=log(p_i/p0_i) (theta1 < theta2),
    # dQi = Q0 -(+) Q(theta_i) (I don't exactly know the sign, but it shouldn't
    # matter), alpha is the estimated derivative wrt theta, i.e., log of
    # potential parameters
    with open(strength_deriv_file, "r") as f:
        deriv_data = json.load(f)
    # Construct the Jacobian
    jac_strength = np.empty(nparams)
    for ii, val in enumerate(deriv_data.values()):
        jac_strength[int(ii)] = val[-1]
    # Reshape to be a proper jacobian matrix
    jac_strength = jac_strength.reshape((1, -1))
    np.save(FIM_DIR / "jacobian_strength.npy", jac_strength)
# jac_strength = np.vstack((jac_strength, jac_strength))

###############################################################################
# Additional target properties
# ----------------------------
# Note that in this FIM property calculation, we basically just compute the
# Jacobian. The reason is so that later we can encode covariance with the
# strength property, which is calculated separately.
additional_properties_files = args.additional_properties
if additional_properties_files is None:
    list_of_target_property = []
    jac_additional = [np.empty((0, nparams))]
else:
    print("Adding more target properties")
    print(json.dumps(additional_properties_files, indent=4))

    jacobian_file = FIM_DIR / "jacobian_other_properties.pkl"
    if not jacobian_file.exists():
        jacobian_dict = {}
    else:
        with open(jacobian_file, "rb") as f:
            jacobian_dict = pickle.load(f)

    # Load the dictionaries containing target properties information
    list_of_target_property = []
    list_of_target_property_to_compute = []
    for path in additional_properties_files:
        with open(path, "r") as f:
            target_property = json.load(f)
        list_of_target_property.append(target_property)

        key = Path(path).with_suffix("").name
        if key not in jacobian_dict:
            list_of_target_property_to_compute.append(target_property)

    if len(list_of_target_property_to_compute) > 0:
        # Load dummy covariance matrix
        with open(DATA_DIR / "covariance" /
                  "target_property_covariance_dummy.json") as f:
            cov_dummy = json.load(f)
        cov_list = []
        for path in additional_properties_files:
            for cov_key, cov in cov_dummy.items():
                if cov_key in path:
                    break
            if not Path(path).with_suffix("").name in jacobian_dict:
                cov_list.append(np.array(cov))

        # Prepare the FIM property calculation
        fim_property_inst = FIMPropertyScore()
        compute_batch_args = {
            "score_quantity": "SENSITIVITY",
            "list_of_target_property": list_of_target_property_to_compute,
            "cov": cov_list,
            "return_jacobian": True,
            "transform": transform,
            **potential_dict
        }
        # Compute the FIM of the target properties
        fim_property, list_of_jac = fim_property_inst.compute_batch(
            **compute_batch_args)
        # Save the list of Jacobian matrices as a dictionary
        keys = []
        for path in additional_properties_files:
            key = Path(path).with_suffix("").name
            if key not in jacobian_dict:
                keys.append(key)

        for key, jac in zip(keys, list_of_jac):
            if key not in jacobian_dict:
                jacobian_dict.update({key: jac})
        with open(FIM_DIR / "jacobian_other_properties.pkl", "wb") as f:
            pickle.dump(jacobian_dict, f)

    # Convert the Jacobian from a dictionary to a list of array
    jac_additional = [val for val in jacobian_dict.values()]

# Combine the Jacobian matrices
jac = np.vstack((jac_strength, *jac_additional))
if len(jac) == 0:
    raise ValueError("No target property is requested")
# Save this combined Jacobbian matrix
np.save(FIM_DIR / "jacobian_target.npy", jac)

###############################################################################
# Construct the FIM
# -----------------
print("Construct the FIM")
# Load covariance matrix
cov_file = args.target_covariance
cov = np.load(cov_file)
np.save(FIM_DIR / "cov_target.npy", cov)

fim = jac.T @ np.linalg.pinv(cov) @ jac
# Save FIM of the target properties
np.save(FIM_DIR / "fim_target.npy", fim)

if args.show:
    # Plot FIM
    plt.figure()
    plt.imshow(fim)
    plt.colorbar()
    plt.xticks(range(nparams), deriv_data)
    plt.yticks(range(nparams), deriv_data)

# FIM eigenspectrum analysis
print("Eigenvalue analysis")
eigval, eigvec = np.linalg.eigh(fim)
# The FIM is only rank 1, so there is only 1 nonzero eigenvalue, and that's
# what we will display
print("Eigenvalue:", eigval)

if args.show:
    # Plot the eigenvalues
    plt.figure()
    plt.plot(eigval)
    plt.yscale("log")
    plt.ylabel("Eigenvalue")

    # Plot the eigenvectors
    plt.figure()
    for ii, v in enumerate(eigvec.T):
        plt.plot(v**2, label=ii)
    plt.xticks(range(nparams), deriv_data)
    plt.xlabel("Parameter index (base 1)")
    plt.ylabel("Eigenvector participation factor")
    plt.ylim(0, 1)
    plt.legend(title="Eigenvector index\n(0 -> sloppy)", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Plot eigenvectors as a matrix
    plt.figure()
    plt.imshow(eigvec, vmin=-1, vmax=1, cmap="bwr")
    plt.colorbar(label="Eigenvector component value")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Parameter index")
    plt.xticks(range(nparams), [f"{val:0.2e}" for val in eigval], rotation=90)
    plt.yticks(range(nparams), deriv_data)
    plt.tight_layout()

    plt.show()
