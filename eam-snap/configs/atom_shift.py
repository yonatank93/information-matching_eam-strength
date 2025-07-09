"""If we just take the coordinate of the central atom, draw a spherical region centered
at the central atom with large enough radius, and export atoms within that sphere, the
issue is that lammps' region command doesn't wrap over the periodic boundary, so that if
we have central atom near the boundary we will be missing some neighboring atoms and itq
will affect the force value.

To resolve this, what I will do is to shift the central atom to the center of the box.
With lammps' displace_atom move` command, lammps will automatically remap the atoms to
wrap over the periodic boundary. Then, if we use `region` command to get the local
environment, it will be guaranteed to include all neighboring atoms.

This script will compute the shifting factor to shift the central atoms to the center of
the box.

Note: The companion `atom_shift.ipynb` file has some more detail, e.g., about the
transformation matrix.
"""

from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np

# SETUP
parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to save the results",
    dest="target_dir",
)
parser.add_argument(
    "-i",
    "--index-compressed",
    type=str,
    help=
    "Path to the numpy file containing the index of compressed environments",
    dest="index_compressed",
)
args = parser.parse_args()

SAVE_DIR = Path(args.target_dir).resolve()
index_compressed = np.load(args.index_compressed).astype(
    int)  # Make sure it's int

#########################################################################################
# Get the cell vector
# -------------------
print("Computing the cell vectors")

# Triclinic box --- hard coded from the configuration files
# Parameters of the restricted triclinic box
xlo, xhi, xy = 9.4529510586485480e2, 1.7371266170496060e3, 6.2188278662727674e0
ylo, yhi, xz = -4.0063532727924689e2, 4.1956442404085044e2, -2.4918818131154445e0
zlo, zhi, yz = -3.1011905422045612e2, 4.9472260784626945e2, 3.0074771675674690e0

# Parameters of the generalized triclinic boc
O = np.array([xlo, ylo, zlo])
A = np.array([xhi - xlo, 0, 0])
B = np.array([xy, yhi - ylo, 0])
C = np.array([xz, yz, zhi - zlo])
# Transformation matrix
T = np.array([A, B, C]).T
# This is the center of the box in the real space
box_center = T @ (0.5 * np.ones(3)) + O
# Save box center
np.savetxt(SAVE_DIR / "box_center.txt", box_center)

# # Export the cell vector
# with open(SAVE_DIR / "cell_vectors.json", "w") as f:
#     json.dump(
#         {
#             "origin": O.tolist(),
#             "center": box_center.tolist(),
#             "v1": A.tolist(),
#             "v2": B.tolist(),
#             "v3": C.tolist(),
#         },
#         f,
#         indent=4,
#     )

#########################################################################################
# Shift the central atoms
# -----------------------
print("Shift central atoms")
# Get the coordinates of the central atoms in the real space

central_atoms = np.loadtxt(SAVE_DIR / "candidates_central_atom_positions.txt")
nconfigs = len(central_atoms)

# Shift
shift = np.empty((nconfigs, 3))
for ii, pos in tqdm(enumerate(central_atoms), total=nconfigs):
    shift[ii] = box_center - pos

# Export the shift
np.savetxt(SAVE_DIR / "central_atom_shift.txt", shift)
