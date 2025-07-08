"""After we complete the FIM-matching targetting the intermediate properties
and Vasily computed the Jacobian of the strength evaluated at the final optimal
parameters, we validate the results by performing FIM-matching between the
intermediate properties and strength. This is to see if the intermediate
properties are adequate to predict strength. Additionally, we can analyze if we
need all intermediate properties or if there are any that we can throw away.
"""

from pathlib import Path
import argparse
import yaml
from glob import glob
import pickle

import numpy as np
from orchestrator.computer.score.fim import FIMMatchingScore

FILE_DIR = Path(__file__).resolve().parent

# Command line arguments
parser = argparse.ArgumentParser(
    "Final FIM-matching between intermediate properties and strength")
parser.add_argument("-s",
                    "--settings-file",
                    type=str,
                    required=True,
                    help="Settings file",
                    dest="settings_file")
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
    print("FIM-matching targets strength directly. "
          "The analysis here excludes this target property")
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
opt_weights = fim_matching.fim_match(fim_target,
                                     fim_candidates,
                                     solver_kwargs={
                                         "solver": "SDPA",
                                         "verbose": True,
                                         "epsilonStar": tol,
                                         "lambdaStar": 1e4,
                                         "numThreads": 4
                                     },
                                     weight_tolerance={
                                         "zero_tol": np.sqrt(tol),
                                         "zero_tol_dual": np.sqrt(tol)
                                     })
