"""I use this script to generate the candidate configurations for Ta from
candidate configurations for W. The only thing we need to take care of is
scaling the lattice parameter. W and Ta have different lattice parameters, so
if we don't fix it and use the candidate configurations for W for problem for
Ta, the candidates are suboptimal.
"""

from pathlib import Path
import pickle
import argparse
import subprocess
from glob import glob
from tqdm import tqdm

import numpy as np
from ase.io import read

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
DATA_DIR = ROOT_DIR / "data"

# Get the lattice parameter scaling factor --- This will be calculated from the
# ratio of the predictions
print("Computing lattice parameter scaling factor")
# Import the predictions
with open(DATA_DIR / "covariance" / "kim_query_Ta.pkl", "rb") as f:
    kim_query_Ta = pickle.load(f)
with open(DATA_DIR / "covariance" / "kim_query_W.pkl", "rb") as f:
    kim_query_W = pickle.load(f)
# Extract the lattice parameters
latparams_ens_Ta = kim_query_Ta["cohesive-potential-energy-cubic-crystal"][:,
                                                                           0]
latparams_ens_W = kim_query_W["cohesive-potential-energy-cubic-crystal"][:, 0]
# Ratio of lattice parameter
latparams_ratio_Ta_W = np.mean(latparams_ens_Ta) / np.mean(latparams_ens_W)
print("Scaling factor (Ta/W):", latparams_ratio_Ta_W)

# Scale the configurations
parser = argparse.ArgumentParser()
parser.add_argument("--source",
                    "-s",
                    type=str,
                    required=True,
                    dest="source",
                    help="Source candidate configurations to scale")
parser.add_argument("--target",
                    "-t",
                    type=str,
                    required=True,
                    dest="target",
                    help="Where to store the scaled configurations")
args = parser.parse_args()
configs_source_dir = Path(args.source) / "configs"
target_dir = Path(args.target)
configs_target_dir = Path(args.target) / "configs"

# Let's just first copy all files in configs_source_dir to configs_target_dir
target_dir.mkdir(parents=True, exist_ok=True)  # Make sure target dir exists
if not configs_target_dir.exists():
    subprocess.run(["cp", "-rv", configs_source_dir, target_dir])
    # We also probably don't want to keep the dump files, which are output from
    # extracting these configurations
    subprocess.run(f"rm -v {configs_target_dir / '*.dump'}", shell=True)
# Load the configuration
print("Loading atoms to scale....")
configs_files = glob(str(configs_target_dir / "*.xyz"))
list_of_atoms = [read(ff) for ff in tqdm(configs_files)]
# Rescale the atoms
print("Rescaling lattice parameters....")
for ii, atoms in enumerate(tqdm(list_of_atoms)):
    atoms.set_cell(atoms.get_cell() * latparams_ratio_Ta_W, scale_atoms=True)
    atoms.write(configs_files[ii], format="extxyz")
