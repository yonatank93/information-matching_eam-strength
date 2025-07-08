"""In this script, I want to try visualize the optimal configurations picked by
information-matching method. The way I will do this is to take the QUESTS
descriptor for all candidate environments and compute the PCA. Then, I will plot
the projection onto the first 2 principle components and plot the representation
as black dots.
For the optimal environments, I will plot as red dots, and the size of the dots
will be related to the weights.
"""

from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.io import read

from utils import natural_key

FILE_DIR = Path(__file__).resolve().parent

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_dir",
    type=str,
    help="Directory where the configurations are stored, to retrieve "
    "the descriptor of the candidate environments",
    dest="config_dir",
)
parser.add_argument(
    "-f",
    "--fimmatch-dir",
    type=str,
    help="Location of fim-matching results, to retrieve identifiers of "
    "optimal environments",
    dest="fimmatch_dir",
)
parser.add_argument(
    "-i",
    "--iterations",
    type=int,
    nargs="+",
    help="Result from which iteration loop to plot",
    dest="iterations",
)
args = parser.parse_args()

# FIM-matching results directories
fimmatch_dir = Path(args.fimmatch_dir)
iter_dirs = sorted(glob(str(fimmatch_dir / "iteration_*")), key=natural_key)

# Load the QUESTS descriptor of the candidate environments --- The descriptor
# should be precomputed and stored in the xyz file
print("Load the descriptors")
config_dir = Path(args.config_dir) / "configs"
nfiles = len(glob(str(config_dir / "*.xyz")))

cand_desc = np.empty((nfiles, 63))  # QUESTS desc has 63 elements
for ii in tqdm(range(nfiles)):
    # Read the atoms
    atoms = read(config_dir / f"config_{ii}.xyz", format="extxyz")
    # Get the descriptors of all atoms in the configuration
    desc_all = atoms.arrays["desc"]
    # But, we only want to take the descriptor of the central atom, pointed
    # by the masking array
    mask_forces = np.loadtxt(config_dir / f"config_{ii}_mask.txt").reshape(
        (-1, 3))
    idx = np.product(mask_forces, axis=1).astype(int)
    cand_desc[ii] = desc_all[np.where(idx)[0]]

################################################################################
# PCA
# ---
# We can also use PCA to do similar analysis
# Normalize
cand_desc_mean = np.mean(cand_desc, axis=0)
cand_desc_std = np.std(cand_desc, axis=0)
# Some elements have std of 0.0, and we need to fix this
idx_fixed = np.where(cand_desc_std == 0.0)[0]
cand_desc_std[idx_fixed] = 1.0
cand_desc_norm = (cand_desc - cand_desc_mean) / cand_desc_std

# SVD
u, s, vh = np.linalg.svd(cand_desc_norm)
# Project the descriptors into the embedding space
pca_cand = (vh @ cand_desc_norm.T).T

# PCA plot
for ii in args.iterations:
    print("Load the optimal weights for iteration", ii)
    # Load the optimal weights
    fimmatch_res = np.loadtxt(Path(iter_dirs[ii - 1]) /
                              "optimal_weights_without_zeros.txt",
                              skiprows=1)
    opt_identifier = fimmatch_res[:, 0].astype(int)
    opt_weights = fimmatch_res[:, 1]
    # Sort the weights from large to small --- Configs with small weights are
    # plotted with higher zorder
    idx_sort = np.argsort(opt_weights)[::-1]
    opt_identifier = opt_identifier[idx_sort]
    opt_weights = opt_weights[idx_sort]

    # Print some reports
    print(f"FIM-matching result in {fimmatch_dir}")
    print(f"identifies {len(opt_weights)} optimal environments")

    # PCA of the optimal environments
    opt_desc = cand_desc_norm[opt_identifier]
    pca_opt = (vh @ opt_desc.T).T

    npc = 2
    plt.figure()
    plt.scatter(*(pca_cand[:, :npc].T), c="k", s=5, lw=0, alpha=0.5)
    plt.scatter(*(pca_opt[:, :npc].T),
                c="r",
                marker="o",
                alpha=1,
                lw=0.5,
                ec="w",
                s=500 * np.sqrt(opt_weights / max(opt_weights)))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)

    # # Legends handles
    leg_cand = mpl.lines.Line2D([], [],
                                color="black",
                                marker="o",
                                markersize=10,
                                linestyle="None",
                                label="Candidate environments")
    leg_opt = mpl.lines.Line2D([], [],
                               color="red",
                               marker="o",
                               markersize=10,
                               linestyle="None",
                               label="Optimal environmennts")
    plt.legend(handles=[leg_cand, leg_opt], fontsize=14)
    plt.savefig(Path(iter_dirs[ii - 1]) / "optimal_environments_pca.png",
                bbox_inches="tight")
plt.show()
