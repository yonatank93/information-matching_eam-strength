"""Using FIMTrainingSetScore to compute the FIM of candidate environmnents
using EAM potential.
"""

from pathlib import Path
import argparse
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import shutil

import numpy as np
from ase.io import read
import numdifftools as nd

from information_matching.transform import (LogTransform, ComposedTransform,
                                            transform_builder)
from transform import EAMTransform

from properties.compute_eam import compute
from utils import ENERGY_KEY, FORCES_KEY

# Base directories
FILE_DIR = Path(__file__).parent.resolve()
DATA_DIR = FILE_DIR / "data"
# Index location of the 7 parameters in the 20 elements list of parameters
idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]
# Number of parallel processes
nprocs = 30

# Command line arguments
parser = argparse.ArgumentParser("Compute the FIMs of candidate environments")
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
    help="Folder where the candidate configurations are stored. This folder"
    "should contain config_list.txt file containing a list of relative path "
    "to the .xyz files.",
    dest="config_dir",
)
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to write the resulst",
    dest="target_dir",
)
parser.add_argument("-a",
                    "--per-atom",
                    action="store_true",
                    help="An option to split the forces FIM by atoms "
                    "(not affecting energy configurations)",
                    dest="per_atom")
args = parser.parse_args()

CONFIGS_DIR = Path(args.config_dir)
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
    symb="Ta",
    subdir="random",
    # Handle the cleaning after all calculations are done
    cleanup=False)
log_transform = LogTransform(np.sign(p0_7vals))
transform = ComposedTransform([eam_transform, log_transform])
print(json.dumps(transform.jsonable_kwargs, indent=4))

###############################################################################
# Read candidate environments
# ---------------------------
print("Load candidate environments")
# Find xyz files
config_path_list = np.loadtxt(CONFIGS_DIR / "config_list.txt", dtype=str)
nconfigs = len(config_path_list)
xyz_files = [CONFIGS_DIR / ff for ff in config_path_list]
# Read configuration files
list_of_atoms = [read(ff, format="extxyz") for ff in tqdm(xyz_files)]
# For bookkeeping, add the atoms file information into atoms.info
for atoms, ff in zip(list_of_atoms, xyz_files):
    atoms.info.update({"filepath": str(ff)})
# We will compute the FIM using energy quantity for some configurations and
# forces for the others. So, we need to create a dictionary to tell the FIM
# module which quantity to compute for each configuration.
evaluate_kwargs = {"compute_energy": [], "compute_forces": []}
for atoms in list_of_atoms:
    fpath = atoms.info["filepath"]
    evaluate_kwargs["compute_energy"].append(ENERGY_KEY in fpath)
    evaluate_kwargs["compute_forces"].append(FORCES_KEY in fpath)

###############################################################################
# FIM calculation
# ---------------

final_results_file = FIM_DIR / "fim_candidates_list.txt"
if final_results_file.exists():
    print("FIM candidate calculation has been completed.")
    fim_path_list = np.loadtxt(final_results_file, dtype=str)
    list_of_fims = [np.load(FIM_DIR / ff) for ff in tqdm(fim_path_list)]
else:
    print("Compute FIM of the candidates")

    def evaluate_predictions(params, atoms, compute_energy, compute_forces,
                             compute_stress):
        # print(params)
        # Inverse parameter transformation
        tmp_transform = transform_builder("ComposedTransform",
                                          transform.jsonable_kwargs)
        # We only need the path to the .eam.alloy file
        _ = tmp_transform.inverse_transform(params)

        # Potential
        tmp_eam_transform = tmp_transform.transform_list[0]
        eam_file = (tmp_eam_transform._target_dir /
                    tmp_eam_transform.eam_alloy_filename)

        # Compute predictions
        preds_dict = compute(atoms, str(eam_file), compute_energy,
                             compute_forces, compute_stress)
        preds_array = []
        for val in preds_dict.values():
            if isinstance(val, float):
                preds_array = np.append(preds_array, val)
            else:
                preds_array = np.append(preds_array, val.flatten())
        if len(preds_array) == 1:
            preds_array = preds_array[0]
        return preds_array

    def compute_jacobian(params, atoms, compute_energy, compute_forces,
                         compute_stress):
        jac_fn = nd.Jacobian(evaluate_predictions, step=0.1 * np.abs(params))
        jac = jac_fn(params, atoms, compute_energy, compute_forces,
                     compute_stress)
        return jac

    def parallel_fn_wrapper(args):
        return compute_jacobian(*args)

    # Create an iterable for parallelization
    iterables = []
    for ii, atoms in enumerate(list_of_atoms):
        item = [
            transform(p0_7vals), atoms, evaluate_kwargs["compute_energy"][ii],
            evaluate_kwargs["compute_forces"][ii], False
        ]
        iterables.append(item)

    # Run Jacobian calculations in parallel
    with Pool(nprocs) as p:
        list_of_jac = list(
            tqdm(p.imap(parallel_fn_wrapper, iterables), total=len(iterables)))

    # Compute and save the FIMs
    print("Saving FIM....")
    fim_path_list = []  # A list containing relative path of FIM files
    list_of_fims = []
    for atoms, jac in tqdm(zip(list_of_atoms, list_of_jac), total=nconfigs):
        fpath = atoms.info["filepath"]
        fname = Path(fpath).with_suffix("").name
        if ENERGY_KEY in fpath:
            fim = jac.T @ jac
            fname = f"{fname}.npy"
            np.save(FIM_DIR / fname, fim)
            fim_path_list.append(fname)
            list_of_fims.append(fim)
        elif FORCES_KEY in fpath:
            if args.per_atom:
                # FIMs for each atom force
                natoms = atoms.get_global_number_of_atoms()
                # In some configurations, there are atoms that have forces of 0
                # because the local environment of those atoms are very
                # symmetric. I found that there are almost 2e3 of such atoms.
                # We can exclude those atoms to reduce the number of candidates
                # further.
                forces = atoms.arrays["atomic_forces_forces"]
                idx_exclude = np.where(
                    np.linalg.norm(forces, axis=1) < 1e-12)[0]

                for ii in range(natoms):
                    if ii not in idx_exclude:
                        jac_per_atom = jac[ii * 3:(ii + 1) * 3]
                        fim_per_atom = jac_per_atom.T @ jac_per_atom
                        # Save
                        fname_per_atom = fname + f"_{ii}.npy"
                        np.save(FIM_DIR / fname_per_atom, fim_per_atom)
                        fim_path_list.append(fname_per_atom)
                        list_of_fims.append(fim_per_atom)
            else:
                fim = jac.T @ jac
                fname = f"{fname}.npy"
                np.save(FIM_DIR / fname, fim)
                fim_path_list.append(fname)
                list_of_fims.append(fim)
    # Export FIM path list file --- This can be used as a check file if the
    # calculation has been completed.
    np.savetxt(final_results_file, fim_path_list, fmt="%s")

    # Clean temporary files first
    print("Cleaning up and saving the results")
    folders = [ff for ff in list(FIM_DIR.iterdir()) if ff.is_dir()]
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        list(tqdm(executor.map(shutil.rmtree, folders), total=len(folders)))
