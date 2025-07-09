"""Postprocess the configurations, e.g., converting the dump file into extxyz file and
find the binary masking array to only extract the forces of the central atom.
"""

from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
from ase.io import read, write

# SETUP
FILE_DIR = Path(__file__).parent.resolve()
ORIG_LAMMPS_DUMP = FILE_DIR / "final.W0.dump"
nprocs = 10  # Just use 10 processes due to memory limitation

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to save the results",
    dest="target_dir",
)
args = parser.parse_args()

SAVE_DIR = Path(args.target_dir).resolve()
CONF_DIR = SAVE_DIR / "configs"

#########################################################################################
# Convert the .dump files into .extxyz files
# ------------------------------------------
print("Converting .dump files into .extxyz files")
# Find the total number of configurations are in the directory
nconfigs = len(glob(str(CONF_DIR / "*.dump")))
# Conversion
for ii in tqdm(range(nconfigs)):
    # Read .dump file
    atoms = read(CONF_DIR / f"config_{ii}.dump")
    # ASE read the elements from the .dump file as H, but we want W
    atoms.set_chemical_symbols(["W"] * atoms.get_global_number_of_atoms())
    # Write .extxyz file
    write(CONF_DIR / f"config_{ii}.xyz", atoms, format="extxyz")

#########################################################################################
# Find central atom mask
# ----------------------
print("Find the central atom binary mask")
# The central atom should be located at the center of the box
box_center = np.loadtxt(SAVE_DIR / "box_center.txt")
# Iterate
for ii in tqdm(range(nconfigs)):
    # Get the positions
    atoms = read(CONF_DIR / f"config_{ii}.xyz")
    positions = atoms.positions
    # Find which atom is the closes to the center of the box
    dist_from_center = np.linalg.norm(positions - box_center, axis=1)
    if np.min(dist_from_center) < 1e-8:
        idx = np.argmin(dist_from_center)
    else:
        # Just in case -- If this line is executed, there might be something wrong with
        # shifting the atom or other steps.
        print("Central atom not found")
    # Create a binary masking array
    mask_atom = np.zeros(atoms.get_global_number_of_atoms(), dtype=int)
    mask_atom[idx] = 1
    # Masking array is per prediction, not just per atom
    mask = np.repeat(mask_atom, 3)
    np.savetxt(CONF_DIR / f"config_{ii}_mask.txt", mask, fmt="%d")
