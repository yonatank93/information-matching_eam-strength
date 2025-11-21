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
import time
import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.io import read

from utils import natural_key, ENERGY_KEY, FORCES_KEY

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
config_dir = Path(args.config_dir)
config_file = np.loadtxt(config_dir / "config_list.txt", dtype=str)
config_paths = [config_dir / ff for ff in config_file]
nfiles = len(config_paths)

cand_desc_dict = {}
energy_keys = []
forces_keys = []
for ff in tqdm(config_paths):
    # Dictionary key
    key = Path(ff).with_suffix("").name  # Remove suffix
    # Remove quantity key --- Forces configs are subset of energy configs
    if FORCES_KEY in key:
        key = key.replace(FORCES_KEY, "")
        forces_keys.append(key[:-1])
    elif ENERGY_KEY in key:
        key = key.replace(ENERGY_KEY, "")
        energy_keys.append(key[:-1])
    key = key[:-1]  # Remove trailing "_"

    if key not in cand_desc_dict:
        # Read the atoms
        atoms = read(ff, format="extxyz")
        # Get the descriptors of all atoms in the configuration
        desc = atoms.arrays["desc"]
        cand_desc_dict.update({key: {"atoms": atoms, "desc": desc}})

################################################################################
# PCA
# ---
print("Training PCA")
# Put the descriptor into a single array so that we can compute mean and std
# over the entire descriptors
cand_desc = np.empty((0, 63))  # QUESTS descriptor has 63 elements
for val in cand_desc_dict.values():
    cand_desc = np.vstack((cand_desc, val["desc"]))

# Normalize
cand_desc_mean = np.mean(cand_desc, axis=0)
cand_desc_std = np.std(cand_desc, axis=0)
# Some QUESTS descriptor elements have std of 0.0, and we need to fix this
idx_fixed = np.where(cand_desc_std == 0.0)[0]
cand_desc_std[idx_fixed] = 1.0
for val in cand_desc_dict.values():
    desc = val["desc"]
    val.update({"desc_norm": (desc - cand_desc_mean) / cand_desc_std})
# Put the normalized descriptor into a single array for PCA
cand_desc_norm = np.empty((0, 63))
for val in cand_desc_dict.values():
    cand_desc_norm = np.vstack((cand_desc_norm, val["desc_norm"]))

# SVD
svd_file = config_dir / "quests_pca.npz"
if svd_file.exists():
    print("Loading SVD matrices")
    start_time = time.perf_counter()
    SVD = np.load(svd_file)
    u = SVD["U"]
    s = SVD["S"]
    vh = SVD["V"].T
    end_time = time.perf_counter()
    print("SVD completed in", datetime.timedelta(seconds=end_time - start_time))
else:
    start_time = time.perf_counter()
    u, s, vh = np.linalg.svd(cand_desc_norm)
    np.savez(svd_file, U=u, S=s, V=vh.T)
    end_time = time.perf_counter()
    print("SVD completed in", datetime.timedelta(seconds=end_time - start_time))

# Project the descriptors into the embedding space
for val in cand_desc_dict.values():
    desc_proj = (vh @ val["desc_norm"].T).T
    val.update({"desc_pca": desc_proj})
# Put the projected descriptor into an array for easy plotting
pca_cand = np.empty((0, 63))  # QUESTS descriptor has 63 elements
for val in cand_desc_dict.values():
    pca_cand = np.vstack((pca_cand, val["desc_pca"]))
# Separate the embedding into energy and forces configs
pca_cand_energy = np.empty((0, 63))
pca_cand_forces = np.empty((0, 63))
for key, val in cand_desc_dict.items():
    if key in forces_keys:
        # For the forces, we want to exclude symmetric environments
        forces = val["atoms"].arrays[FORCES_KEY]
        idx_include = np.where(np.linalg.norm(forces, axis=1) > 0.0)[0]
        idx_exclude = [ii for ii in np.arange(len(forces)) if ii not in idx_include]
        pca_cand_forces = np.vstack((pca_cand_forces, val["desc_pca"][idx_include]))
        # But even if the configuration contains asymmetric environments, some of them
        # might still be symmetric
        pca_cand_energy = np.vstack((pca_cand_energy, val["desc_pca"][idx_exclude]))
    elif key in energy_keys:
        pca_cand_energy = np.vstack((pca_cand_energy, val["desc_pca"]))


# PCA plot
for ii in args.iterations:
    print("Plot the optimal environments for iteration", ii)

    npc = 2
    plt.figure(dpi=300)
    # Plot all environments
    plt.scatter(*(pca_cand_energy[:, :npc].T), c="k", s=5, lw=0, alpha=0.5)
    plt.scatter(
        *(pca_cand_forces[:, :npc].T), c="orange", zorder=-5, s=5, lw=0, alpha=0.5
    )
    # # Legends candidates
    # all_cands = mpl.lines.Line2D(
    #     [],
    #     [],
    #     color="black",
    #     marker="o",
    #     markersize=10,
    #     linestyle="None",
    #     label="Centrosymmetric\nenvironments",
    # )
    # forces_cands = mpl.lines.Line2D(
    #     [],
    #     [],
    #     color="orange",
    #     marker="o",
    #     markersize=10,
    #     linestyle="None",
    #     label="Non-centrosymmetric\nenvironmennts",
    # )

    # Load the optimal weights
    fimmatch_res_file = Path(iter_dirs[ii - 1]) / "optimal_weights_without_zeros.csv"
    if not fimmatch_res_file.exists():
        continue
    fimmatch_res = np.loadtxt(fimmatch_res_file, skiprows=1, delimiter=",", dtype=object)
    opt_identifier = fimmatch_res[:, 1].astype(str)
    opt_weights = fimmatch_res[:, 2].astype(float)
    # Sort the weights from large to small --- Configs with small weights are
    # plotted with higher zorder
    idx_sort = np.argsort(opt_weights)[::-1]
    opt_identifier = opt_identifier[idx_sort]
    opt_weights = opt_weights[idx_sort]

    # Print some reports
    print(f"FIM-matching result in {fimmatch_dir}")
    print(f"identifies {len(opt_weights)} optimal data")

    # PCA of the optimal environments
    pca_opt = np.empty((0, 63))
    plot_weights = []
    for ident, wm in zip(opt_identifier, opt_weights):
        if ENERGY_KEY in ident:
            key = ident.replace(ENERGY_KEY, "")
            key = key[:-1]
            # Add the descriptor
            pca_opt = np.vstack((pca_opt, cand_desc_dict[key]["desc_pca"]))
            # Add the weights - need to duplicate by the number of atoms
            atoms = cand_desc_dict[key]["atoms"]
            natoms = atoms.get_global_number_of_atoms()
            plot_weights = np.append(plot_weights, np.ones(natoms) * wm)
            # print("energy", key)
        elif FORCES_KEY in ident:
            key = ident.replace(FORCES_KEY, "")
            try:
                atom_idx = int(key.split("_")[-1])
                key = "_".join(key.split("_")[:-1])
                key = key[:-1]
                # Add the descriptor - only for one atom
                pca_opt = np.vstack((pca_opt, cand_desc_dict[key]["desc_pca"][atom_idx]))
                # Add weights - only for one atom
                plot_weights = np.append(plot_weights, wm)
                # print("forces per-atom", key)
            except ValueError:
                key = key[:-1]
                # Add the descriptor
                pca_opt = np.vstack((pca_opt, cand_desc_dict[key]["desc_pca"]))
                # Add the weights - need to duplicate by the number of atoms
                atoms = cand_desc_dict[key]["atoms"]
                natoms = atoms.get_global_number_of_atoms()
                plot_weights = np.append(plot_weights, np.ones(natoms) * wm * natoms**2)
                # print("forces per-configuration", key)

    # Plot the optimal environments
    plt.scatter(
        *(pca_opt[:, :npc].T),
        c="r",
        marker="o",
        alpha=1,
        lw=0.25,
        ec="w",
        s=250 * np.sqrt(plot_weights / max(plot_weights)),
        label="Optimal environments",
    )

    # Labels, ticks, etc
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)
    # plt.legend(handles=[all_cands, forces_cands], fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(
        Path(iter_dirs[ii - 1]) / "optimal_environments_pca.png", bbox_inches="tight"
    )
plt.show()
