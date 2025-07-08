"""Using FIMTrainingSetScore to compute the FIM of candidate environmnents
using EAM potential.
"""

from pathlib import Path
import argparse
import json
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import shutil

import numpy as np
from ase.io import read

from orchestrator.computer.score.fim import FIMTrainingSetScore
from information_matching.transform import LogTransform, ComposedTransform
from transform import EAMTransform

# Base directories
FILE_DIR = Path(__file__).parent.resolve()
DATA_DIR = FILE_DIR / "data"
# Index location of the 7 parameters in the 20 elements list of parameters
idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]
# Number of parallel processes
nprocs = 30

# Command line arguments
parser = argparse.ArgumentParser(
    "Compute the FIMs of the candidate environments")
# What arguments should we request: initial parameters, target folder where to
# place the results, atomic configurations path
parser.add_argument(
    "-p",
    "--params",
    type=str,
    help="Path to the parameter file to evaluate (20-value format)",
    dest="params_file",
)
parser.add_argument(
    "-c",
    "--config-dir",
    type=str,
    help="Folder where .xyz files of the candidate configurations and "
    "the masking arrays are stored",
    dest="config_dir",
)
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to write the resulst",
    dest="target_dir",
)
args = parser.parse_args()

CONFIGS_DIR = Path(args.config_dir) / "configs"
TARGET_DIR = Path(args.target_dir)
TARGET_DIR.mkdir(exist_ok=True)
FIM_DIR = TARGET_DIR / "FIM_candidates"
FIM_DIR.mkdir(exist_ok=True)

###############################################################################
# Parameter transformation
# ------------------------
print("Parameter transformation")
# Load the initial parameters
p0_file = Path(args.params_file)
p0 = np.loadtxt(p0_file)
p0_7vals = p0[idx_7vals]

# The complete transformation is composition of EAMTransform first, the
# LogTransform
eam_transform = EAMTransform(
    p0,
    str(FIM_DIR),
    str(DATA_DIR / "translate.x"),
    subdir="random",
    # Handle the cleaning after all calculations are done
    cleanup=False)
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
# Read candidate environments
# ---------------------------
print("Load candidate environments")
# Find xyz files
nconfigs = len(glob(str(CONFIGS_DIR / "*.xyz")))
xyz_files = [CONFIGS_DIR / f"config_{ii}.xyz" for ii in range(nconfigs)]
# Read configuration files
list_of_atoms = [read(ff) for ff in tqdm(xyz_files)]
# For bookkeeping, add the atoms file information into atoms.info
for atoms, ff in zip(list_of_atoms, xyz_files):
    atoms.info.update({"filepath": str(ff)})
# Additionally, read masking arrays
list_of_mask = [
    str(CONFIGS_DIR / f"config_{ii}_mask.txt") for ii in tqdm(range(nconfigs))
]

###############################################################################
# FIM calculation
# ---------------
final_results_file = FIM_DIR / "score_results.xyz"
if final_results_file.exists():
    print("FIM candidate calculation has been completed.")
else:
    print("Compute FIM of the candidates")
    # Run FIM calculations
    fim = FIMTrainingSetScore()
    compute_batch_args = {
        "list_of_atoms": list_of_atoms,
        "score_quantity": "SENSITIVITY",
        "transform": transform,
        "list_of_mask": list_of_mask,
        "nprocs": nprocs,
        **potential_dict
    }
    list_of_fims = fim.compute_batch(**compute_batch_args)

    # Clean temporary files first
    print("Cleaning up and saving the results")
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        list(
            tqdm(executor.map(shutil.rmtree, FIM_DIR.iterdir()),
                 total=len(list(FIM_DIR.iterdir()))))

    # Save as numpy file
    for ii, fim_mat in tqdm(enumerate(list_of_fims)):
        filename = FIM_DIR / f"fim_{ii}.npy"
        np.save(filename, fim_mat)

    # Save results as xyz file -- This uses internal save_results method
    fim.save_results(compute_results=list_of_fims,
                     save_dir=str(FIM_DIR),
                     list_of_configs=list_of_atoms)

    # # Setup -- Instantiate class and prepare arguments
    # fim = FIMTrainingSetScore()
    # compute_args_base = {
    #     "score_quantity": "SENSITIVITY",
    #     "transform": transform,
    #     **potential_dict
    # }

    # # Computation -- Manually iterate over atoms and compute the FIM
    # # one-by-one, while also saving the FIMs on the way
    # list_of_fims = []  # To collect the results
    # for ii in tqdm(range(nconfigs)):
    #     filename = FIM_DIR / f"fim_{ii}.npy"
    #     if filename.exists():
    #         fim_mat = np.load(filename)
    #     else:
    #         # Prepare the compute arguments
    #         compute_args = compute_args_base.copy()
    #         compute_args.update({
    #             "atoms": list_of_atoms[ii],
    #             "mask": list_of_mask[ii],
    #         })
    #         # Compute
    #         fim_mat = fim.compute(**compute_args)
    #         # Save as numpy file
    #         np.save(filename, fim_mat)
    #     list_of_fims.append(fim_mat)

    # # Save results as xyz file -- This uses internal save_results method
    # fim.save_results(compute_results=list_of_fims,
    #                  save_dir=str(FIM_DIR),
    #                  list_of_configs=list_of_atoms)
