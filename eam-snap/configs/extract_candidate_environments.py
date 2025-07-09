"""I use this script to extract the 2,000 candidate environments that Daniel suggested
from the entire supercell.
"""

from pathlib import Path
import argparse
import shutil
import subprocess
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np

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
parser.add_argument(
    "-i",
    "--index-compressed",
    type=str,
    help="Path to the np file containing the index of compressed environments",
    dest="index_compressed",
)
parser.add_argument(
    "-d",
    "--descriptor-compressed",
    type=str,
    help=
    "Path to the npz file containing the descriptor of compressed environments",
    dest="desc_compressed",
)
args = parser.parse_args()

SAVE_DIR = Path(args.target_dir).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
index_compressed_file = args.index_compressed
desc_compressed_file = args.desc_compressed

# Copy the descriptor of the compressed environment into the target directory
desc_target_file = SAVE_DIR / "candidates_quests_descriptor.npz"
shutil.copy(desc_compressed_file, desc_target_file)

#########################################################################################
# Candidates with respect to original dump file
# ---------------------------------------------
print("Reading the original lammps dump file")
orig_pos_scaled_cand_file = SAVE_DIR / "candidates_central_atom_positions_scaled.txt"
if orig_pos_scaled_cand_file.exists():
    orig_positions_scaled_candidates = np.loadtxt(orig_pos_scaled_cand_file)
else:
    # Read the original dump file as a numpy array
    orig_dump = np.loadtxt(ORIG_LAMMPS_DUMP, skiprows=9)
    orig_positions_scaled = orig_dump[:, 2:]  # Just get the scaled coordinates

    # The candidates are index by the row of the original dump file in
    # eam-index-compressed.npz
    orig_idx_candidates = np.load(index_compressed_file).astype(
        int)  # Make sure int

    # Finally, these are the scaled positions of the candidate environments
    orig_positions_scaled_candidates = orig_positions_scaled[
        orig_idx_candidates]
    np.savetxt(orig_pos_scaled_cand_file, orig_positions_scaled_candidates)
ncandidates = len(orig_positions_scaled_candidates)

#########################################################################################
# Find the absolute coordinates of the candidate central atoms
# ------------------------------------------------------------
print("Finding the absolute positions of the candidate environments")
positions_absolute_candidates_file = SAVE_DIR / "candidates_central_atom_positions.txt"
if positions_absolute_candidates_file.exists():
    positions_absolute_candidates = np.loadtxt(
        positions_absolute_candidates_file)
else:
    # Take the original lammps dump file, read it and add atoms absolute positions
    MOD_LAMMPS_DUMP = FILE_DIR / "final.W0.mod.dump"
    if not MOD_LAMMPS_DUMP.exists():
        print("Get the atoms absolute positions")
        subprocess.run("srun -n 16 lmp -i final.W0.mod.in", shell=True)

    # We'll read the modified dump file
    mod_dump = np.loadtxt(MOD_LAMMPS_DUMP, skiprows=9)
    mod_positions_scaled = mod_dump[:, 2:5]  # The scaled coordinates
    mod_positions_absolute = mod_dump[:, 5:]  # The absolute coordinates

    # Finally, these are the absolute positions of the candidate environments


    def find_candidate_absolute_position(ii):
        # Find the atoms by checking the scaled distance.
        ref_pos = orig_positions_scaled_candidates[
            ii]  # Position of the atom we want
        dist = np.linalg.norm(mod_positions_scaled - ref_pos, axis=1)
        idx = np.argmin(dist)
        return mod_positions_absolute[idx]

    with Pool(nprocs) as p:
        positions_absolute_candidates = list(
            tqdm(
                p.imap(find_candidate_absolute_position, range(ncandidates)),
                total=ncandidates,
            ))
    # Convert to np.array
    positions_absolute_candidates = np.array(positions_absolute_candidates)
    np.savetxt(positions_absolute_candidates_file,
               positions_absolute_candidates)

#########################################################################################
# Shift the central atoms to the center of the box
# ------------------------------------------------
print("Shift central atoms to the center of the box")

# Get the information to shift the atoms
# Run the script to generate this information
subprocess.run(
    f"python atom_shift.py -t {SAVE_DIR} -i {index_compressed_file}",
    shell=True)
shift_info = np.loadtxt(SAVE_DIR / "central_atom_shift.txt")
# After shifting, the central atom should always be at the center of the box.
box_center = np.loadtxt(SAVE_DIR / "box_center.txt")

#########################################################################################
# Generate the helper files for lammps to carve out central atom environment
# --------------------------------------------------------------------------
print("Generate lammps include files")
helper_dir = FILE_DIR / "lammps_helper"
helper_dir.mkdir(exist_ok=True)

for ii, shift in tqdm(enumerate(shift_info), total=ncandidates):
    # The following file containing lammps command will be imported in the lammps file
    # inside the loop
    # These atoms need to be shifted
    content = "\n".join([
        "# Shifting distance to make central atom far away from the boundary",
        "print      'Shifting the atoms so that the atoms we want are far from boundary'",
        f"displace_atoms all move {shift[0]:0.16f} {shift[1]:0.16f} {shift[2]:0.16f} units box",
        "",
        "# Coordinate of the central atom in the shifted lattice",
        "print	'Extrating central atom environment for atom ${n}'",
        "# Define region around central atom",
        f"region	reg1 sphere {box_center[0]:0.16f} {box_center[1]:0.16f} {box_center[2]:0.16f} 10 units box",
        "group	local region reg1",
        "",
        "# Output atoms with computed forces into a new .dump file",
        "dump 	output local custom 1 %s/config_${n}.dump id type xs ys zs x y z"
        % (str(SAVE_DIR / "configs")),
        "# Sort by index to help with atom indexing later",
        "dump_modify output sort id format line '%d %d %.16g %.16g %.16g %.16g %.16g %.16g'",
        "# Run a single step to compute forces and output results",
        "run   	0",
        "",
        "# Clean up",
        "undump	output",
        "region     reg1 delete",
        "group      local delete",
        "",
        "# Unshift the atoms so we don't need to reload the dump file for the next iteration",
        f"displace_atoms all move {-shift[0]:0.16f} {-shift[1]:0.16f} {-shift[2]:0.16f} units box",
    ])
    with open(helper_dir / f"lammps_include_{ii}.txt", "w") as f:
        f.write(content)

# Prepare the directory that will be used to store the dump files of the compressed
# environments
(SAVE_DIR / "configs").mkdir(parents=True, exist_ok=True)

# Finally, we can run the lammps script to generate the stripped out candidates
print("Please run lmp -i extract_candidate_environments.in")
print("Then, run python generate_xyz.py")
