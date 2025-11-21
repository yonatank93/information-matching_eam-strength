"""Final FIM-matching analysis after the FIM-matching calculation targeting the
ingermediate QoIs are completed.

Requirements:
* Completed FIM-matching run
* Strength Jacobian (from Vasily) evaluated at the optimal parameters

List of analysis:
* FIM-matching between the intermediate QoIs and plastic strength
* Overlap between the eigenspaces

Justifications:
* By performing FIM-matching between the intermediate QoIs and strength, we
  can assess if the intermediate QoIs are sufficient to represent the
  information constrain for strength predictions.
  If the problem is infeasible, then there are some information relevant for
  strength predictions that are not captured by the intermediate QoIs.
* If the FIM-matching calculation is infeasible, we can look at the overlap
  between the eigenspaces, which will tell us about the information missing
  from the intermediate QoIs that are required for the strength prediction.
  Furthermore, the overlap also shows how much information can be captured by
  the intermediate QoIs.
"""


from pathlib import Path
import argparse
import yaml
from glob import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from orchestrator.computer.score.fim import FIMMatchingScore

FILE_DIR = Path(__file__).resolve().parent

# Command line arguments
parser = argparse.ArgumentParser("Final FIM-matching analysis")
parser.add_argument(
    "-s",
    "--settings-file",
    type=str,
    required=True,
    help="Settings file",
    dest="settings_file",
)
args = parser.parse_args()

# Read calculation settings from a yaml file
with open(args.settings_file) as f:
    settings = yaml.safe_load(f)

# General settings
# ----------------
general_settings = settings["general"]
# This is where all the results are written to
RESULTS_DIR = Path(general_settings["results_dir"])
# FIM target settings
# -------------------
fim_target_settings = settings["fim_target"]
# Strength derivative file
STRENGTH_DERIV = fim_target_settings["strength_deriv"]
if STRENGTH_DERIV is not None:
    print(
        "FIM-matching targets strength directly. "
        "The analysis here excludes this target property"
    )
# Additional KIMRun properties input files
ADDITIONAL_PROPS = fim_target_settings["additional_props"]
# Target covariance of the target properties
COVARIANCE = fim_target_settings["target_cov"]
cov = np.load(COVARIANCE)
var = np.diag(cov)  # The variance is just the diagonal elements of covariance

# Obtain the folder results for the last iteration
niters = len(glob(str(RESULTS_DIR / "iteration_*")))
LAST_RESULTS_DIR = Path(glob(str(RESULTS_DIR / f"iteration_{niters}*"))[0])
FIM_TARGET_DIR = LAST_RESULTS_DIR / "FIM_target"

###############################################################################
# FIM-Matching between intermediate QoIs and strength

print("FIM-matching between intermetdiate QoIs and plastic strength")

# Load the FIMs for FIM-matching
idx = [0, 1, 2, 3, 4, 5, 6]  # Indices for filtering parameters to include
# Candidates --- Intermediate properties
with open(FIM_TARGET_DIR / "jacobian_other_properties.pkl", "rb") as f:
    jac_others_dict = pickle.load(f)

for ii, key_raw in enumerate(ADDITIONAL_PROPS):
    key = Path(key_raw).with_suffix("").name
    if ii == 0:
        jac_others = jac_others_dict[key][:, idx]
    else:
        jac_others = np.vstack((jac_others, jac_others_dict[key][:, idx]))

npreds, nparams = jac_others.shape
nparams = len(idx)
fim_candidates = np.empty((npreds, nparams, nparams))
for ii, jac in enumerate(jac_others):
    jac = jac.reshape((1, -1))
    fim_candidates[ii] = jac.T @ jac

# Target --- plastic strength
try:
    jac_strength = np.load(FIM_TARGET_DIR / "jacobian_strength.npy")[:, idx]
except FileNotFoundError as e:
    raise FileNotFoundError("Jacobian of strength hasn't been calculated yet")
fim_target = jac_strength.T @ jac_strength

# FIM-matching
fim_matching = FIMMatchingScore()
tol = 1e-12
try:
    opt_weights = fim_matching.fim_match(
        fim_target,
        fim_candidates,
        solver_kwargs={
            "solver": "SDPA",
            # "verbose": True,
            "epsilonStar": tol,
            "lambdaStar": 1e4,
            "numThreads": 4,
        },
        weight_tolerance={"zero_tol": np.sqrt(tol), "zero_tol_dual": np.sqrt(tol)},
    )
except AttributeError:
    print("FIM-matching between intermediate QoIs and strength failed.")
    print("The problem is infeasible")
    print()

###############################################################################
# Compare the information from the data and the target strength

print("Check if the data actually contain all necessary information.")

# Load the FIM of the optimal data
fim_data = np.load(LAST_RESULTS_DIR / "fim_environments_optimal.npy")
# Check the eigenvalues - If the data contain all necessary information, then
# the eigenvalue of the difference matrix should all be non-negative.
l = np.linalg.eigvalsh(fim_data - fim_target)
print("Data contains all necessary information for strength prediction:")
print(np.all(l > -1e-15))
print()

###############################################################################
# Eigenspaces overlap

print("Overlap between the eigenspaces")

# Retrieve the eigenvectors - Only compare the eigenvectors corresponding to
# non-zero eigenvalues
# Intermediate QoIs
# fim_cand_comb = np.sum(fim_candidates, axis=0)
fim_cand_comb = np.load(LAST_RESULTS_DIR / "FIM_target" / "fim_target.npy")
lcand, vcand = np.linalg.eigh(fim_cand_comb)
idx = np.where(lcand > 1e-8)[0]
lcand = lcand[idx]
vcand = vcand[:, idx]
# Target strength
# eigenvalue
ltar, vtar = np.linalg.eigh(fim_target)
ltar = ltar[-1]
vtar = vtar[:, -1].reshape((-1, 1))

# We measure the overlap by taking the dot product between the strength
# eigenvector and the intermediate QoIs eigenvectors. Then, we square the
# values and sum them. The number gives a ratio of the information that the
# intermediate QoIs can capture.
info_ratio = np.sum((vtar.T @ vcand) ** 2)
print(
    f"Intermediate QoIs are able to capture {(info_ratio*100):0.2f}% "
    "of the required information"
)
print()

###############################################################################
# Plot the difference of the FIMs

print("Plot of the difference between intermediate QoIs FIM and strength FIM.")

fim_diff = fim_cand_comb - fim_target
l_diff, v_diff = np.linalg.eigh(fim_diff)
nparams = len(l_diff)

# Plot the eigenvectors
plt.figure()
plt.imshow(v_diff[:, ::-1], vmin=-1, vmax=1, cmap="bwr")
plt.colorbar()
# List the eigenvalues as xticks
param_labels = [r"$r_e$", r"$\beta$", r"$A$", r"$B$", r"$\kappa$", r"$\eta$", r"$F_e$"]
plt.xticks(range(nparams), [f"{l:0.2e}" for l in l_diff[::-1]], rotation=90)
plt.yticks(range(nparams), param_labels)
plt.tight_layout()
plt.show()
